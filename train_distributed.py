#!/usr/bin/env python3
"""
Distributed Training Script for Multi-GPU AMD Systems
Uses DistributedDataParallel for proper multi-GPU utilization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import os
import json
from datetime import datetime

from diffusion_model import DiffusionModel, EMA
from dataset import FrameSequenceDataset


def setup(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()


def train_epoch(model, train_loader, optimizer, scaler, device, rank, use_amp=True):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Only show progress bar on rank 0
    pbar = tqdm(train_loader, desc='Training', disable=(rank != 0))
    
    for context, target in pbar:
        context = context.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # Sample timesteps
        batch_size = context.shape[0]
        t = torch.randint(0, model.module.timesteps, (batch_size,), device=device).long()
        
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda', enabled=use_amp):
            noise = torch.randn_like(target)
            noisy_target = model.module.forward_diffusion(target, t, noise)
            predicted_noise = model.module.predict_noise(noisy_target, context, t)
            loss = nn.functional.mse_loss(predicted_noise, noise)
        
        # Backward pass
        optimizer.zero_grad()
        if scaler is not None:
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
            pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches if num_batches > 0 else 0


def main():
    # Get rank and world size from environment
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Configuration
    config = {
        'data_dir': 'Data/',
        'batch_size': 8,  # Per-GPU batch size
        'num_epochs': 200,
        'learning_rate': 2e-4,
        'save_dir': 'checkpoints',
        'num_workers': 4,
        'use_amp': True,
    }
    
    # Setup distributed training
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        print("\n" + "="*60)
        print("🚀 DISTRIBUTED TRAINING")
        print("="*60)
        print(f"World size: {world_size} GPUs")
        print(f"Per-GPU batch size: {config['batch_size']}")
        print(f"Effective batch size: {config['batch_size'] * world_size}")
        print("="*60)
    
    # Create model and move to GPU
    model = DiffusionModel().to(device)
    model = DDP(model, device_ids=[rank])
    
    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {param_count:,}")
    
    # Create dataset with distributed sampler
    train_dataset = FrameSequenceDataset(
        config['data_dir'],
        sequence_length=6
    )
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    # Optimizer and scaler
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.05)
    scaler = torch.amp.GradScaler('cuda') if config['use_amp'] else None
    
    # EMA (only on rank 0)
    ema = EMA(model.module, decay=0.999) if rank == 0 else None
    
    # Training history
    history = {'train_loss': [], 'epoch': []}
    
    # Create save directory
    if rank == 0:
        os.makedirs(config['save_dir'], exist_ok=True)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(1, config['num_epochs'] + 1):
        train_sampler.set_epoch(epoch)  # Important for shuffling
        
        avg_loss = train_epoch(
            model, train_loader, optimizer, scaler, device, rank, config['use_amp']
        )
        
        # Update EMA
        if ema is not None:
            ema.update(model.module)
        
        # Gather loss from all GPUs
        loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()
        
        if rank == 0:
            print(f"\nEpoch {epoch}/{config['num_epochs']}")
            print(f"Train Loss: {avg_loss:.6f}")
            
            history['train_loss'].append(avg_loss)
            history['epoch'].append(epoch)
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'ema_state_dict': ema.shadow.state_dict() if ema else None,
                }, os.path.join(config['save_dir'], 'best_loss.pt'))
                print(f"✓ Saved best model (loss: {avg_loss:.6f})")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch}.pt'))
                
                # Save history
                with open(os.path.join(config['save_dir'], 'training_history.json'), 'w') as f:
                    json.dump(history, f, indent=2)
    
    cleanup()
    
    if rank == 0:
        print("\n✅ Training complete!")


if __name__ == '__main__':
    main()
