#!/usr/bin/env python3
"""
Enhanced Distributed Training for AMD MI325X GPUs
- Full validation with PSNR/SSIM metrics
- Larger model (base_channels=128 for ~4x parameters)
- Uses PyTorch DDP with RCCL backend
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
from metrics import calculate_metrics_for_frames


def setup_distributed(rank, world_size):
    """Initialize distributed training with AMD/ROCm settings"""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RCCL_SOCKET_IFNAME'] = 'lo'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    os.environ['NCCL_DEBUG'] = 'WARN'
    
    torch.cuda.set_device(rank)
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )


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


@torch.no_grad()
def validate(model, ema, val_loader, device, rank, use_amp=True, max_batches=5):
    """Run validation with PSNR/SSIM metrics (rank 0 only)"""
    if rank != 0 or val_loader is None:
        return None, None, None
    
    # Use EMA model for validation if available
    eval_model = ema.get_model() if ema else model.module
    eval_model.eval()
    
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    num_batches = 0
    
    pbar = tqdm(val_loader, desc='Validating', total=min(max_batches, len(val_loader)))
    
    for context, target in pbar:
        if num_batches >= max_batches:
            break
        
        try:
            context = context.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # Generate predictions using DDIM sampling
            with torch.amp.autocast('cuda', enabled=use_amp):
                predicted = eval_model.sample(context, device, use_ddim=True, ddim_steps=100)
                loss = nn.functional.mse_loss(predicted, target)
            
            # Calculate metrics
            metrics = calculate_metrics_for_frames(predicted, target)
            
            total_loss += loss.item()
            total_psnr += metrics['psnr_avg']
            total_ssim += metrics['ssim_avg']
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'psnr': f'{metrics["psnr_avg"]:.2f}',
                'ssim': f'{metrics["ssim_avg"]:.4f}'
            })
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n⚠️ OOM during validation, skipping batch...")
                torch.cuda.empty_cache()
                continue
            raise
    
    if num_batches == 0:
        return None, None, None
    
    return (
        total_loss / num_batches,
        total_psnr / num_batches,
        total_ssim / num_batches
    )


def train_worker(rank, world_size, config):
    """Training worker for each GPU"""
    try:
        setup_distributed(rank, world_size)
        device = torch.device(f'cuda:{rank}')
        
        if rank == 0:
            print("\n" + "="*60)
            print("🚀 ENHANCED MULTI-GPU TRAINING")
            print("="*60)
            print(f"   GPUs: {world_size}")
            print(f"   Base Channels: {config['base_channels']} (larger model)")
            print(f"   Batch per GPU: {config['batch_size']}")
            print(f"   Effective batch: {config['batch_size'] * world_size}")
            print(f"   Epochs: {config['num_epochs']}")
            print(f"   Validation: Enabled with PSNR/SSIM")
            print("="*60 + "\n")
        
        # Create LARGER model with more parameters
        model = DiffusionModel(base_channels=config['base_channels']).to(device)
        model = DDP(model, device_ids=[rank])
        
        if rank == 0:
            params = sum(p.numel() for p in model.parameters())
            print(f"Model parameters: {params:,} ({params/1e6:.1f}M)")
        
        # Training dataset with distributed sampler
        train_dataset = FrameSequenceDataset(config['data_dir'], sequence_length=6)
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            sampler=train_sampler,
            num_workers=config['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        
        # Validation dataset (rank 0 only, no distributed sampler needed)
        val_loader = None
        if rank == 0:
            val_dataset = FrameSequenceDataset(config['data_dir'], sequence_length=6)
            val_loader = DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
        
        if rank == 0:
            print(f"Training samples: {len(train_dataset)}")
            print(f"Batches per epoch: {len(train_loader)}")
            print(f"Validation batches per epoch: {config['val_batches']}\n")
        
        # Optimizer and scaler
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.05)
        scaler = torch.amp.GradScaler('cuda') if config['use_amp'] else None
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['num_epochs'], eta_min=1e-6
        )
        
        # EMA (rank 0 only)
        ema = EMA(model.module, decay=0.999) if rank == 0 else None
        
        # Training history
        history = {
            'train_loss': [], 
            'val_loss': [],
            'psnr': [],
            'ssim': [],
            'lr': [],
            'epoch': []
        }
        best_psnr = 0
        best_loss = float('inf')
        start_epoch = 1
        
        if rank == 0:
            os.makedirs(config['save_dir'], exist_ok=True)
        
        # Load checkpoint if resuming
        if config.get('resume_from') and os.path.exists(config['resume_from']):
            if rank == 0:
                print(f"\n📂 Loading checkpoint: {config['resume_from']}")
            
            checkpoint = torch.load(config['resume_from'], map_location=device, weights_only=False)
            model.module.load_state_dict(checkpoint['model'])
            
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_psnr = checkpoint.get('best_psnr', 0)
            best_loss = checkpoint.get('train_loss', float('inf'))
            
            # Load history if exists
            history_path = os.path.join(config['save_dir'], 'training_history.json')
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history = json.load(f)
            
            if rank == 0:
                print(f"   ✓ Resuming from epoch {start_epoch}")
                print(f"   ✓ Best PSNR so far: {best_psnr:.2f} dB")
        
        # Training loop
        for epoch in range(start_epoch, config['num_epochs'] + 1):
            train_sampler.set_epoch(epoch)
            
            # Training
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scaler, device, rank, config['use_amp']
            )
            
            # Update EMA
            if ema:
                ema.update(model.module)
            
            # Sync training loss across GPUs
            loss_tensor = torch.tensor([train_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            train_loss = loss_tensor.item()
            
            # Validation (rank 0 only)
            val_loss, psnr, ssim = None, None, None
            if rank == 0 and epoch % config['val_every'] == 0:
                val_loss, psnr, ssim = validate(
                    model, ema, val_loader, device, rank, 
                    config['use_amp'], config['val_batches']
                )
            
            # Step scheduler
            scheduler.step()
            
            if rank == 0:
                current_lr = scheduler.get_last_lr()[0]
                
                # Print epoch summary
                print(f"\n{'='*50}")
                print(f"Epoch {epoch}/{config['num_epochs']}")
                print(f"  Train Loss: {train_loss:.6f}")
                if val_loss is not None:
                    print(f"  Val Loss:   {val_loss:.6f}")
                    print(f"  PSNR:       {psnr:.2f} dB")
                    print(f"  SSIM:       {ssim:.4f}")
                print(f"  LR:         {current_lr:.2e}")
                print(f"{'='*50}")
                
                # Update history
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['psnr'].append(psnr)
                history['ssim'].append(ssim)
                history['lr'].append(current_lr)
                history['epoch'].append(epoch)
                
                # Save best model by PSNR
                if psnr is not None and psnr > best_psnr:
                    best_psnr = psnr
                    torch.save({
                        'epoch': epoch,
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'psnr': psnr,
                        'ssim': ssim,
                        'ema': ema.shadow.state_dict() if ema else None,
                        'config': config,
                    }, os.path.join(config['save_dir'], 'best_psnr.pt'))
                    print(f"  ✓ Saved best PSNR model ({psnr:.2f} dB)")
                
                # Save best model by loss
                if train_loss < best_loss:
                    best_loss = train_loss
                    torch.save({
                        'epoch': epoch,
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'ema': ema.shadow.state_dict() if ema else None,
                        'config': config,
                    }, os.path.join(config['save_dir'], 'best_loss.pt'))
                    print(f"  ✓ Saved best loss model ({train_loss:.6f})")
                
                # Checkpoint every 10 epochs
                if epoch % 10 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'train_loss': train_loss,
                        'best_psnr': best_psnr,
                    }, os.path.join(config['save_dir'], f'checkpoint_e{epoch}.pt'))
                    
                    with open(os.path.join(config['save_dir'], 'training_history.json'), 'w') as f:
                        json.dump(history, f, indent=2)
                    print(f"  ✓ Saved checkpoint epoch {epoch}")
        
        cleanup()
        
        if rank == 0:
            # Final save
            with open(os.path.join(config['save_dir'], 'training_history.json'), 'w') as f:
                json.dump(history, f, indent=2)
            
            print("\n" + "="*60)
            print("✅ TRAINING COMPLETE!")
            print("="*60)
            print(f"   Best PSNR: {best_psnr:.2f} dB")
            print(f"   Best Loss: {best_loss:.6f}")
            print(f"   Checkpoints saved to: {config['save_dir']}/")
            print("="*60)
            
    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        import traceback
        traceback.print_exc()
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
    
    # Configuration - LARGER MODEL for more VRAM usage
    config = {
        'data_dir': 'Data/',
        'batch_size': 12,        # Per GPU (effective = 24 with 2 GPUs)
        'base_channels': 128,    # 2x larger (was 64) = ~4x more params
        'num_epochs': 400,
        'lr': 2e-4,
        'num_workers': 4,
        'use_amp': True,
        'save_dir': 'checkpoints',
        'val_every': 5,          # Validate every N epochs
        'val_batches': 5,        # Number of validation batches
        'resume_from': 'checkpoints/checkpoint_e100.pt',  # Resume from last checkpoint
    }
    
    # Spawn workers
    print("\n🚀 Starting enhanced distributed training...")
    print(f"   Model: base_channels={config['base_channels']} (~86M params)")
    print(f"   Expected VRAM: ~50-60% per GPU")
    
    mp.spawn(
        train_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()
