#!/usr/bin/env python3
"""
Quick inference script to check training progress.
Run alongside training to visualize current model output.

Usage: python3 quick_inference.py
"""

import torch
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from diffusion_model import DiffusionModel


def find_latest_checkpoint(checkpoint_dir='checkpoints'):
    """Find the most recent checkpoint"""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
    if not checkpoints:
        return None
    
    # Prefer best_psnr, then best_loss, then latest epoch
    for name in ['best_psnr.pt', 'best_loss.pt']:
        path = os.path.join(checkpoint_dir, name)
        if os.path.exists(path):
            return path
    
    # Get most recent by modification time
    return max(checkpoints, key=os.path.getmtime)


def load_sample_context(data_dir='Data/', start_idx=0):
    """Load 4 consecutive frames as context"""
    files = sorted(glob.glob(os.path.join(data_dir, '*.pt')))
    if len(files) < 6:
        raise ValueError(f"Need at least 6 frames, found {len(files)}")
    
    frames = []
    for i in range(4):
        f = files[start_idx + i]
        data = torch.load(f, weights_only=False)
        if isinstance(data, dict):
            data = data.get('frame_data', data)
        frames.append(data)
    
    # Also load ground truth (frames 5 and 6)
    gt_frames = []
    for i in range(4, 6):
        f = files[start_idx + i]
        data = torch.load(f, weights_only=False)
        if isinstance(data, dict):
            data = data.get('frame_data', data)
        gt_frames.append(data)
    
    context = torch.cat(frames, dim=0).unsqueeze(0)  # (1, 20, 720, 720)
    ground_truth = torch.cat(gt_frames, dim=0).unsqueeze(0)  # (1, 10, 720, 720)
    
    return context, ground_truth


def visualize_channels(frame, title, ax):
    """Visualize first 3 channels as RGB"""
    # frame shape: (5, H, W) or (10, H, W)
    if frame.shape[0] >= 3:
        rgb = frame[:3].transpose(1, 2, 0)  # (H, W, 3)
        rgb = (rgb + 1) / 2  # [-1,1] -> [0,1]
        rgb = np.clip(rgb, 0, 1)
        ax.imshow(rgb)
    else:
        ax.imshow(frame[0], cmap='gray')
    ax.set_title(title)
    ax.axis('off')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Find checkpoint
    checkpoint_path = find_latest_checkpoint()
    if not checkpoint_path:
        print("❌ No checkpoint found in checkpoints/")
        print("   Training may not have saved a checkpoint yet.")
        return
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    # Load model
    model = DiffusionModel(timesteps=1000, base_channels=128)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint.get('epoch', '?')
        psnr = checkpoint.get('best_psnr', checkpoint.get('psnr', '?'))
        print(f"   Epoch: {epoch}, Best PSNR: {psnr}")
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Load sample data
    print("📊 Loading sample data...")
    try:
        context, ground_truth = load_sample_context()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    context = context.to(device)
    
    # Generate prediction
    print("🎨 Generating prediction with DDIM (100 steps)...")
    with torch.no_grad():
        predicted = model.sample(context, device, use_ddim=True, ddim_steps=100)
    
    # Move to CPU for visualization
    predicted = predicted.cpu().numpy()[0]
    ground_truth = ground_truth.cpu().numpy()[0]
    context_np = context.cpu().numpy()[0]
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: Context frames
    for i in range(4):
        frame = context_np[i*5:(i+1)*5]
        visualize_channels(frame, f'Context {i+1}', axes[0, i])
    
    # Row 2: Predicted vs Ground Truth
    visualize_channels(predicted[:5], 'Predicted Frame 1', axes[1, 0])
    visualize_channels(ground_truth[:5], 'Ground Truth 1', axes[1, 1])
    visualize_channels(predicted[5:], 'Predicted Frame 2', axes[1, 2])
    visualize_channels(ground_truth[5:], 'Ground Truth 2', axes[1, 3])
    
    # Calculate quick metrics
    mse = np.mean((predicted - ground_truth) ** 2)
    psnr = 10 * np.log10(4 / mse) if mse > 0 else float('inf')  # range [-1,1] so max^2 = 4
    
    plt.suptitle(f'Checkpoint: {os.path.basename(checkpoint_path)} | Quick PSNR: {psnr:.2f} dB', fontsize=14)
    plt.tight_layout()
    
    # Save
    output_path = 'inference_check.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved visualization to: {output_path}")
    print(f"   Quick PSNR: {psnr:.2f} dB")
    
    # Also try to display if possible
    try:
        plt.show()
    except:
        pass
    
    plt.close()


if __name__ == '__main__':
    main()
