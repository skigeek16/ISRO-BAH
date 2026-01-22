#!/usr/bin/env python3
"""
Distributed Training for AMD MI325X GPUs
Uses PyTorch DDP with RCCL backend for multi-GPU training
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import json
from datetime import datetime

from diffusion_model import DiffusionModel, EMA
from dataset import FrameSequenceDataset


def setup_distributed(rank, world_size):
    """Initialize distributed training with proper AMD/ROCm settings"""
    # Set environment for RCCL (AMD's NCCL)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RCCL_SOCKET_IFNAME'] = 'lo'  # Use loopback
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    os.environ['NCCL_DEBUG'] = 'WARN'
    
    # Set GPU for this process
    torch.cuda.set_device(rank)
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',  # RCCL on AMD
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    if rank == 0:
        print(f"✓ Distributed setup complete: {world_size} GPUs")


def cleanup():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def train_one_epoch(model, loader, optimizer, scaler, device, rank, use_amp=True):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(loader, desc='Training', disable=(rank != 0))
    
    for context, target in pbar:
        context = context.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        batch_size = context.shape[0]
        t = torch.randint(0, 1000, (batch_size,), device=device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            noise = torch.randn_like(target)
            noisy = model.module.forward_diffusion(target, t, noise)
            pred = model.module.predict_noise(noisy, context, t)
            loss = nn.functional.mse_loss(pred, noise)
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if rank == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / max(num_batches, 1)


def train_worker(rank, world_size, config):
    """Training worker for each GPU"""
    try:
        setup_distributed(rank, world_size)
        device = torch.device(f'cuda:{rank}')
        
        if rank == 0:
            print("\n" + "="*60)
            print("🚀 MULTI-GPU TRAINING (AMD MI325X)")
            print("="*60)
            print(f"   GPUs: {world_size}")
            print(f"   Batch per GPU: {config['batch_size']}")
            print(f"   Effective batch: {config['batch_size'] * world_size}")
            print(f"   Epochs: {config['num_epochs']}")
            print("="*60 + "\n")
        
        # Create model
        model = DiffusionModel().to(device)
        model = DDP(model, device_ids=[rank])
        
        if rank == 0:
            params = sum(p.numel() for p in model.parameters())
            print(f"Model parameters: {params:,}")
        
        # Dataset with distributed sampler
        dataset = FrameSequenceDataset(config['data_dir'], sequence_length=6)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        
        loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            sampler=sampler,
            num_workers=config['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        
        if rank == 0:
            print(f"Dataset: {len(dataset)} samples")
            print(f"Batches per epoch: {len(loader)}\n")
        
        # Optimizer and scaler
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.05)
        scaler = torch.amp.GradScaler('cuda') if config['use_amp'] else None
        
        # EMA (rank 0 only)
        ema = EMA(model.module, decay=0.999) if rank == 0 else None
        
        # Training history
        history = {'train_loss': [], 'epoch': []}
        best_loss = float('inf')
        
        if rank == 0:
            os.makedirs(config['save_dir'], exist_ok=True)
        
        # Training loop
        for epoch in range(1, config['num_epochs'] + 1):
            sampler.set_epoch(epoch)  # Important for shuffling
            
            avg_loss = train_one_epoch(
                model, loader, optimizer, scaler, device, rank, config['use_amp']
            )
            
            # Update EMA
            if ema:
                ema.update(model.module)
            
            # Sync loss across GPUs
            loss_tensor = torch.tensor([avg_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
            
            if rank == 0:
                print(f"\nEpoch {epoch}/{config['num_epochs']} | Loss: {avg_loss:.6f}")
                
                history['train_loss'].append(avg_loss)
                history['epoch'].append(epoch)
                
                # Save best
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save({
                        'epoch': epoch,
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss': avg_loss,
                        'ema': ema.shadow.state_dict() if ema else None,
                    }, os.path.join(config['save_dir'], 'best_model.pt'))
                    print(f"  ✓ Saved best model")
                
                # Checkpoint every 10 epochs
                if epoch % 10 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss': avg_loss,
                    }, os.path.join(config['save_dir'], f'checkpoint_e{epoch}.pt'))
                    
                    with open(os.path.join(config['save_dir'], 'history.json'), 'w') as f:
                        json.dump(history, f, indent=2)
        
        cleanup()
        
        if rank == 0:
            print("\n✅ Training complete!")
            
    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        cleanup()
        raise


def main():
    # Check GPUs
    if not torch.cuda.is_available():
        print("❌ No GPU available!")
        sys.exit(1)
    
    world_size = torch.cuda.device_count()
    print(f"\n🔍 Found {world_size} GPU(s)")
    
    for i in range(world_size):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1e9
        print(f"   GPU {i}: {props.name} ({mem_gb:.0f} GB)")
    
    if world_size < 2:
        print("\n⚠️  Only 1 GPU found. Use train.py instead.")
        print("   For multi-GPU, ensure both GPUs are visible:")
        print("   export HIP_VISIBLE_DEVICES=0,1")
        sys.exit(1)
    
    config = {
        'data_dir': 'Data/',
        'batch_size': 8,       # Per GPU (effective = 16 with 2 GPUs)
        'num_epochs': 200,
        'lr': 2e-4,
        'num_workers': 4,
        'use_amp': True,
        'save_dir': 'checkpoints'
    }
    
    # Spawn workers
    print("\n🚀 Starting distributed training...")
    mp.spawn(
        train_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()
