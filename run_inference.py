"""
Inference script for diffusion model - generates predictions on APR25 data
Run on Lightning AI after training is complete
"""

import torch
import os
import glob
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from diffusion_model import DiffusionModel
from metrics import calculate_metrics_for_frames


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")
    
    model = DiffusionModel(timesteps=1000)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'metrics' in checkpoint:
        print(f"  Checkpoint metrics: {checkpoint['metrics']}")
    
    return model


def load_frame(file_path):
    """Load and preprocess a single frame"""
    data = torch.load(file_path, weights_only=False)
    
    if isinstance(data, dict):
        frame = data.get('frame_data', data.get('frame', data.get('data', None)))
    else:
        frame = data
    
    if not isinstance(frame, torch.Tensor):
        frame = torch.tensor(frame, dtype=torch.float32)
    
    if frame.dtype != torch.float32:
        frame = frame.float()
    
    return frame


def save_visualization(save_path, context, target, prediction, metrics, idx):
    """Save comparison visualization"""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'Sample {idx} - PSNR: {metrics["psnr_avg"]:.2f} dB, SSIM: {metrics["ssim_avg"]:.4f}', fontsize=14)
    
    # Row 0: Context frames (show first channel of each)
    for i in range(4):
        ax = axes[0, i]
        channel_idx = i * 5
        ax.imshow(context[channel_idx].cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
        ax.set_title(f'Context Frame {i+1}')
        ax.axis('off')
    
    # Row 1: Target vs Prediction
    axes[1, 0].imshow(target[0].cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
    axes[1, 0].set_title('Target Frame 1 (GT)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(prediction[0].cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
    axes[1, 1].set_title('Predicted Frame 1')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(target[5].cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
    axes[1, 2].set_title('Target Frame 2 (GT)')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(prediction[5].cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
    axes[1, 3].set_title('Predicted Frame 2')
    axes[1, 3].axis('off')
    
    # Row 2: Difference maps
    diff1 = np.abs(target[0].cpu().numpy() - prediction[0].cpu().numpy())
    axes[2, 0].imshow(diff1, cmap='hot', vmin=0, vmax=0.5)
    axes[2, 0].set_title(f'Error Frame 1 (MAE: {diff1.mean():.4f})')
    axes[2, 0].axis('off')
    
    diff2 = np.abs(target[5].cpu().numpy() - prediction[5].cpu().numpy())
    axes[2, 1].imshow(diff2, cmap='hot', vmin=0, vmax=0.5)
    axes[2, 1].set_title(f'Error Frame 2 (MAE: {diff2.mean():.4f})')
    axes[2, 1].axis('off')
    
    # Per-channel PSNR
    axes[2, 2].axis('off')
    channel_names = ['VIS', 'WV', 'SWIR', 'TIR1', 'TIR2']
    text = "Per-Channel PSNR (Frame 1):\n"
    for c, name in enumerate(channel_names):
        psnr = metrics.get(f'psnr_frame1_ch{c}', metrics['psnr_avg'])
        text += f"  {name}: {psnr:.2f} dB\n"
    axes[2, 2].text(0.1, 0.5, text, transform=axes[2, 2].transAxes, 
                   fontsize=10, verticalalignment='center', fontfamily='monospace')
    
    # Statistics
    axes[2, 3].axis('off')
    stats = f"Target range: [{target.min():.3f}, {target.max():.3f}]\n"
    stats += f"Pred range: [{prediction.min():.3f}, {prediction.max():.3f}]\n"
    stats += f"PSNR: {metrics['psnr_avg']:.2f} dB\n"
    stats += f"SSIM: {metrics['ssim_avg']:.4f}"
    axes[2, 3].text(0.1, 0.5, stats, transform=axes[2, 3].transAxes,
                   fontsize=11, verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_inference(data_dir, checkpoint_path, output_dir, num_samples=10, device='cuda'):
    """Run inference on data and save results"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = load_model(checkpoint_path, device)
    
    # Get data files
    data_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    print(f"Found {len(data_files)} data files in {data_dir}")
    
    if len(data_files) < 6:
        raise ValueError(f"Need at least 6 frames, found {len(data_files)}")
    
    # Limit samples
    num_samples = min(num_samples, len(data_files) - 5)
    
    # Collect metrics
    all_psnr = []
    all_ssim = []
    
    print(f"\nRunning inference on {num_samples} samples...")
    
    for idx in tqdm(range(num_samples)):
        # Load 6 consecutive frames
        frames = []
        for i in range(6):
            frame = load_frame(data_files[idx + i])
            frames.append(frame)
        
        # Prepare context (4 frames) and target (2 frames)
        context = torch.cat(frames[:4], dim=0).unsqueeze(0).to(device)  # (1, 20, H, W)
        target = torch.cat(frames[4:6], dim=0).unsqueeze(0).to(device)  # (1, 10, H, W)
        
        # Generate prediction
        with torch.no_grad():
            prediction = model.sample(context, device, use_ddim=True, ddim_steps=50)
        
        # Calculate metrics
        metrics = calculate_metrics_for_frames(prediction, target)
        all_psnr.append(metrics['psnr_avg'])
        all_ssim.append(metrics['ssim_avg'])
        
        # Save visualization for all samples
        save_path = os.path.join(output_dir, f'sample_{idx:03d}.png')
        save_visualization(save_path, context[0], target[0], prediction[0], metrics, idx)
        
        # Save prediction tensors
        pred_path = os.path.join(output_dir, f'prediction_{idx:03d}.pt')
        torch.save({
            'prediction': prediction.cpu(),
            'target': target.cpu(),
            'context': context.cpu(),
            'metrics': metrics
        }, pred_path)
    
    # Summary statistics
    avg_psnr = np.mean(all_psnr)
    avg_ssim = np.mean(all_ssim)
    std_psnr = np.std(all_psnr)
    std_ssim = np.std(all_ssim)
    
    print(f"\n{'='*60}")
    print("INFERENCE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Samples processed: {num_samples}")
    print(f"Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
    print(f"Best PSNR: {max(all_psnr):.2f} dB")
    print(f"Best SSIM: {max(all_ssim):.4f}")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_dir}")
    
    # Save summary
    summary = {
        'num_samples': num_samples,
        'avg_psnr': float(avg_psnr),
        'std_psnr': float(std_psnr),
        'avg_ssim': float(avg_ssim),
        'std_ssim': float(std_ssim),
        'best_psnr': float(max(all_psnr)),
        'best_ssim': float(max(all_ssim)),
        'all_psnr': [float(p) for p in all_psnr],
        'all_ssim': [float(s) for s in all_ssim],
        'checkpoint': checkpoint_path,
        'data_dir': data_dir,
        'timestamp': datetime.now().isoformat()
    }
    
    import json
    with open(os.path.join(output_dir, 'inference_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    return summary


def main():
    # Configuration
    config = {
        'data_dir': 'data/APR25',  # APR25 data folder
        'checkpoint_path': 'checkpoints/best_ssim_checkpoint.pt',  # Best SSIM checkpoint
        'output_dir': 'inference_results',
        'num_samples': 3,  # Only 3 samples as requested
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("="*60)
    print("DIFFUSION MODEL INFERENCE")
    print("="*60)
    for key, value in config.items():
        print(f"{key:20s}: {value}")
    print("="*60 + "\n")
    
    # Check if checkpoint exists
    if not os.path.exists(config['checkpoint_path']):
        # Try alternative checkpoint paths
        alternatives = [
            'checkpoints/best_psnr_checkpoint.pt',
            'checkpoints/latest_checkpoint.pt',
            'checkpoints/checkpoint_epoch_100.pt'
        ]
        for alt in alternatives:
            if os.path.exists(alt):
                print(f"Using alternative checkpoint: {alt}")
                config['checkpoint_path'] = alt
                break
        else:
            print("ERROR: No checkpoint found!")
            print("Searched for:")
            print(f"  - {config['checkpoint_path']}")
            for alt in alternatives:
                print(f"  - {alt}")
            return
    
    # Run inference
    run_inference(
        data_dir=config['data_dir'],
        checkpoint_path=config['checkpoint_path'],
        output_dir=config['output_dir'],
        num_samples=config['num_samples'],
        device=config['device']
    )


if __name__ == '__main__':
    main()
