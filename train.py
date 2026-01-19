import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
from datetime import datetime

from diffusion_model import DiffusionModel
from dataset import create_dataloader
from metrics import calculate_metrics_for_frames


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        lr=1e-4,
        device='cuda',
        save_dir='checkpoints',
        gradient_accumulation_steps=1,
        use_amp=True,
        resume_from=None,
        use_multi_gpu=True  # Enable multi-GPU by default
    ):
        self.device = device
        self.save_dir = save_dir
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_amp = use_amp and device == 'cuda'
        self.use_multi_gpu = use_multi_gpu and torch.cuda.device_count() > 1
        
        # Multi-GPU support with DataParallel
        if self.use_multi_gpu:
            print(f"✓ Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = nn.DataParallel(model).to(device)
        else:
            self.model = model.to(device)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Mixed precision training
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.05)
        
        # Learning rate scheduler - warmup + cosine decay
        self.base_lr = lr
        self.warmup_steps = min(500, len(train_loader) * 2)
        self.total_steps = len(train_loader) * 100  # Total training steps
        self.current_step = 0
        
        # Loss function - MSE for noise prediction (standard diffusion training)
        self.criterion = nn.MSELoss()
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_psnr': [],
            'val_ssim': [],
            'learning_rate': []
        }
        
        self.start_epoch = 1
        self.best_psnr = -float('inf')
        self.best_ssim = -float('inf')
        
        # Resume from checkpoint if provided
        if resume_from and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)
    
    def get_lr(self):
        """Get learning rate with warmup and cosine decay"""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return self.base_lr * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches_processed = 0
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (context, target) in enumerate(pbar):
            try:
                context = context.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                # Sample random timesteps
                batch_size = context.shape[0]
                t = torch.randint(0, self.model.timesteps, (batch_size,), device=self.device).long()
                
                # Mixed precision training
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    # Add noise to target frames
                    noise = torch.randn_like(target)
                    noisy_target = self.model.forward_diffusion(target, t, noise)
                    
                    # Predict noise
                    predicted_noise = self.model.predict_noise(noisy_target, context, t)
                    
                    # Calculate loss
                    loss = self.criterion(predicted_noise, noise)
                    loss = loss / self.gradient_accumulation_steps
                
                # Backpropagation with gradient scaling
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights every gradient_accumulation_steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                    
                    # Update learning rate
                    lr = self.get_lr()
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    
                    self.optimizer.zero_grad()
                    self.current_step += 1
                
                total_loss += loss.item() * self.gradient_accumulation_steps
                num_batches_processed += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item() * self.gradient_accumulation_steps,
                    'lr': self.optimizer.param_groups[0]['lr']
                })
                
            except RuntimeError as e:
                print(f"\n\nRuntimeError at batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                if "out of memory" in str(e):
                    print("Attempting to clear cache and continue...")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            except Exception as e:
                print(f"\n\nERROR in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        if num_batches_processed == 0:
            print(f"\n\nWARNING: No batches were processed! All {len(self.train_loader)} batches failed.")
            return 0.0
        
        avg_loss = total_loss / num_batches_processed
        print(f"\nProcessed {num_batches_processed}/{len(self.train_loader)} batches successfully")
        self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        return avg_loss
    
    @torch.no_grad()
    def validate(self, max_batches=5):
        """Run validation on a subset of data for faster feedback"""
        if self.val_loader is None:
            return None, None, None
        
        self.model.eval()
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        num_batches = 0
        
        for context, target in tqdm(self.val_loader, desc='Validating', total=min(max_batches, len(self.val_loader))):
            if num_batches >= max_batches:
                break
                
            try:
                context = context.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                # Generate predictions (use DDIM with more steps for better quality)
                predicted = self.model.sample(context, self.device, use_ddim=True, ddim_steps=100)
                
                # Calculate loss
                loss = self.criterion(predicted, target)
                total_loss += loss.item()
                
                # Calculate metrics
                metrics = calculate_metrics_for_frames(predicted, target)
                total_psnr += metrics['psnr_avg']
                total_ssim += metrics['ssim_avg']
                num_batches += 1
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nOOM during validation, skipping batch...")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        if num_batches == 0:
            return None, None, None
            
        avg_loss = total_loss / num_batches
        avg_psnr = total_psnr / num_batches
        avg_ssim = total_ssim / num_batches
        
        return avg_loss, avg_psnr, avg_ssim
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and resume training"""
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.current_step = checkpoint.get('current_step', 0)
        self.history = checkpoint.get('history', self.history)
        self.best_psnr = checkpoint.get('best_psnr', -float('inf'))
        self.best_ssim = checkpoint.get('best_ssim', -float('inf'))
        
        print(f"Resumed from epoch {self.start_epoch}, step {self.current_step}")
        print(f"Best PSNR: {self.best_psnr:.2f}, Best SSIM: {self.best_ssim:.4f}")
    
    def save_checkpoint(self, epoch, metrics):
        checkpoint = {
            'epoch': epoch,
            'current_step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history,
            'best_psnr': self.best_psnr,
            'best_ssim': self.best_ssim
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.save_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save epoch checkpoint
        if epoch % 10 == 0:
            epoch_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, epoch_path)
        
        # Save best checkpoint based on PSNR
        if metrics.get('psnr') is not None:
            if metrics['psnr'] > self.best_psnr:
                self.best_psnr = metrics['psnr']
                best_path = os.path.join(self.save_dir, 'best_psnr_checkpoint.pt')
                torch.save(checkpoint, best_path)
                print(f"✓ Saved best PSNR checkpoint: {self.best_psnr:.2f} dB")
        
        # Save best checkpoint based on SSIM
        if metrics.get('ssim') is not None:
            if metrics['ssim'] > self.best_ssim:
                self.best_ssim = metrics['ssim']
                best_path = os.path.join(self.save_dir, 'best_ssim_checkpoint.pt')
                torch.save(checkpoint, best_path)
                print(f"✓ Saved best SSIM checkpoint: {self.best_ssim:.4f}")
    
    def train(self, num_epochs, validate_every=5, save_every=5):
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
        print(f"Effective Batch Size: {self.train_loader.batch_size * self.gradient_accumulation_steps}")
        print(f"Starting from epoch: {self.start_epoch}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.start_epoch, self.start_epoch + num_epochs):
            # Training
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.start_epoch + num_epochs - 1}")
            print(f"{'='*60}")
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Validation
            if epoch % validate_every == 0:
                val_loss, val_psnr, val_ssim = self.validate()
                
                if val_loss is not None:
                    self.history['val_loss'].append(val_loss)
                    self.history['val_psnr'].append(val_psnr)
                    self.history['val_ssim'].append(val_ssim)
                    
                    print(f"Val Loss:  {val_loss:.6f}")
                    print(f"Val PSNR:  {val_psnr:.2f} dB")
                    print(f"Val SSIM:  {val_ssim:.4f}")
                    
                    # Save checkpoint
                    metrics = {
                        'loss': val_loss,
                        'psnr': val_psnr,
                        'ssim': val_ssim
                    }
                    self.save_checkpoint(epoch, metrics)
                else:
                    self.save_checkpoint(epoch, {'loss': train_loss})
            
            # Periodic save
            elif epoch % save_every == 0:
                self.save_checkpoint(epoch, {'loss': train_loss})
            
            print(f"{'='*60}\n")
            
            # Save training history
            history_path = os.path.join(self.save_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                # Convert numpy values to Python floats for JSON serialization
                json_history = {k: [float(v) if hasattr(v, 'item') else v for v in vals] 
                              for k, vals in self.history.items()}
                json.dump(json_history, f, indent=4)
        
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"Best PSNR: {self.best_psnr:.2f} dB")
        print(f"Best SSIM: {self.best_ssim:.4f}")
        print(f"{'='*60}\n")


def main():
    # Configuration optimized for AMD MI325X GPUs with 256GB VRAM
    config = {
        'data_dir': 'data/APR25',
        'batch_size': 16,  # Increased for MI325X with 256GB VRAM
        'gradient_accumulation_steps': 1,  # No accumulation needed with large batch
        'num_epochs': 200,  # Increased for better convergence
        'learning_rate': 2e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'checkpoints',
        'num_workers': 16,  # Increased for 128-core CPU
        'validate_every': 5,
        'save_every': 10,
        'use_amp': True,  # Mixed precision works on MI325X
        'resume_from': 'checkpoints/best_psnr_checkpoint.pt',  # Resume from best checkpoint
        'pin_memory': True,
        'persistent_workers': True
    }
    
    print("\n" + "="*60)
    print("DIFFUSION MODEL TRAINING - Configuration")
    print("="*60)
    for key, value in config.items():
        print(f"{key:30s}: {value}")
    print("="*60 + "\n")
    
    # Create data loaders
    print("Loading data...")
    try:
        train_loader = create_dataloader(
            config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            shuffle=True,
            pin_memory=config['pin_memory'],
            persistent_workers=config['persistent_workers'],
            augment=True  # Enable data augmentation for training
        )
        print(f"✓ Training data loaded: {len(train_loader.dataset)} sequences\n")
    except Exception as e:
        print(f"✗ Error loading training data: {e}")
        raise
    
    # Create validation loader (using subset of training data for monitoring)
    val_loader = create_dataloader(
        config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False,  # No shuffle for validation
        pin_memory=config['pin_memory'],
        persistent_workers=False,  # Don't need persistent for val
        augment=False  # No augmentation for validation
    )
    
    # Create model
    print("Initializing model...")
    model = DiffusionModel(timesteps=1000)
    
    # Enable optimizations based on GPU type
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU available: {gpu_name}")
        print(f"✓ GPU count: {torch.cuda.device_count()}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Enable TF32 for NVIDIA Ampere+ GPUs only
        if 'NVIDIA' in gpu_name or 'A100' in gpu_name or 'H100' in gpu_name:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"✓ TF32 enabled for NVIDIA GPU")
        elif 'AMD' in gpu_name or 'MI' in gpu_name or 'Instinct' in gpu_name:
            # AMD GPU optimizations
            print(f"✓ AMD GPU detected - using ROCm backend")
        
        # Set memory fraction to avoid OOM
        # torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of VRAM
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=config['learning_rate'],
        device=config['device'],
        save_dir=config['save_dir'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        use_amp=config['use_amp'],
        resume_from=config['resume_from']
    )
    
    # Train
    try:
        trainer.train(
            num_epochs=config['num_epochs'],
            validate_every=config['validate_every'],
            save_every=config['save_every']
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Saving checkpoint...")
        trainer.save_checkpoint(trainer.start_epoch, {'loss': 0.0})
        print("Checkpoint saved. Exiting.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        print("\nSaving emergency checkpoint...")
        trainer.save_checkpoint(trainer.start_epoch, {'loss': 0.0})
        raise


if __name__ == '__main__':
    main()

