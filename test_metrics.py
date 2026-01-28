#!/usr/bin/env python3
"""
Test metrics calculation with/without AMP to check for precision issues.
"""

import torch
import glob
import os
from diffusion_model import DiffusionModel
from metrics import calculate_metrics_for_frames


def load_sample_context(data_dir='Data/', start_idx=0):
    """Load 4 context frames + 2 ground truth frames"""
    files = sorted(glob.glob(os.path.join(data_dir, '*.pt')))
    if len(files) < 6:
        raise ValueError(f"Need at least 6 frames, found {len(files)}")
    
    frames = []
    for i in range(6):
        f = files[start_idx + i]
        data = torch.load(f, weights_only=False)
        if isinstance(data, dict):
            data = data.get('frame_data', data)
        frames.append(data)
    
    context = torch.cat(frames[:4], dim=0).unsqueeze(0)  # (1, 20, 720, 720)
    ground_truth = torch.cat(frames[4:6], dim=0).unsqueeze(0)  # (1, 10, 720, 720)
    
    return context, ground_truth


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Find checkpoint
    checkpoint_path = 'checkpoints/best_psnr.pt'
    if not os.path.exists(checkpoint_path):
        checkpoint_path = 'checkpoints/best_loss.pt'
    if not os.path.exists(checkpoint_path):
        print("❌ No checkpoint found")
        return
    
    print(f"Loading: {checkpoint_path}")
    
    # Load model
    model = DiffusionModel(timesteps=1000, base_channels=128)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print(f"Epoch: {checkpoint.get('epoch', '?')}")
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # Load data
    context, ground_truth = load_sample_context()
    context = context.to(device)
    ground_truth = ground_truth.to(device)
    
    print("\n" + "="*60)
    print("TEST 1: DDIM with AMP (current validation setting)")
    print("="*60)
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True):
            predicted_amp = model.sample(context, device, use_ddim=True, ddim_steps=200)
    
    metrics_amp = calculate_metrics_for_frames(predicted_amp, ground_truth)
    print(f"PSNR: {metrics_amp['psnr_avg']:.2f} dB")
    print(f"SSIM: {metrics_amp['ssim_avg']:.4f}")
    
    print("\n" + "="*60)
    print("TEST 2: DDIM without AMP (full precision)")
    print("="*60)
    with torch.no_grad():
        # Explicit float32
        model_fp32 = model.float()
        context_fp32 = context.float()
        predicted_fp32 = model_fp32.sample(context_fp32, device, use_ddim=True, ddim_steps=200)
    
    metrics_fp32 = calculate_metrics_for_frames(predicted_fp32, ground_truth)
    print(f"PSNR: {metrics_fp32['psnr_avg']:.2f} dB")
    print(f"SSIM: {metrics_fp32['ssim_avg']:.4f}")
    
    print("\n" + "="*60)
    print("TEST 3: DDIM with more steps (500 steps, no AMP)")
    print("="*60)
    with torch.no_grad():
        predicted_500 = model_fp32.sample(context_fp32, device, use_ddim=True, ddim_steps=500)
    
    metrics_500 = calculate_metrics_for_frames(predicted_500, ground_truth)
    print(f"PSNR: {metrics_500['psnr_avg']:.2f} dB")
    print(f"SSIM: {metrics_500['ssim_avg']:.4f}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Setting':<30} {'PSNR (dB)':<12} {'SSIM':<10}")
    print("-"*60)
    print(f"{'AMP + 200 steps':<30} {metrics_amp['psnr_avg']:<12.2f} {metrics_amp['ssim_avg']:<10.4f}")
    print(f"{'FP32 + 200 steps':<30} {metrics_fp32['psnr_avg']:<12.2f} {metrics_fp32['ssim_avg']:<10.4f}")
    print(f"{'FP32 + 500 steps':<30} {metrics_500['psnr_avg']:<12.2f} {metrics_500['ssim_avg']:<10.4f}")
    
    # Check improvement
    psnr_improvement = metrics_fp32['psnr_avg'] - metrics_amp['psnr_avg']
    if psnr_improvement > 0.5:
        print(f"\n⚠️ AMP is reducing PSNR by {psnr_improvement:.2f} dB!")
        print("   Recommendation: Disable AMP during validation")
    else:
        print(f"\n✅ AMP impact is minimal ({psnr_improvement:.2f} dB)")


if __name__ == '__main__':
    main()
