import torch
import torchvision.transforms.functional as TF
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

from diffusion_model import DiffusionModel
from metrics import calculate_metrics_for_frames


class FramePredictor:
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Initialize the frame predictor
        
        Args:
            checkpoint_path: Path to the trained model checkpoint
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = DiffusionModel(timesteps=1000)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {checkpoint_path}")
        print(f"Using device: {self.device}")
    
    def load_frames(self, frame_paths):
        """
        Load 4 frames from paths
        
        Args:
            frame_paths: List of 4 paths to .pt files
        
        Returns:
            context tensor of shape (1, 20, 720, 720)
        """
        if len(frame_paths) != 4:
            raise ValueError("Exactly 4 frames required for context")
        
        frames = []
        for path in frame_paths:
            frame = torch.load(path, weights_only=False)
            
            # Handle different formats
            if isinstance(frame, dict):
                frame = frame.get('frame_data', frame.get('frame', frame.get('data', None)))
            
            if not isinstance(frame, torch.Tensor):
                frame = torch.tensor(frame)
            
            # Ensure correct shape and format (5, 720, 720)
            if len(frame.shape) == 2:
                raise ValueError(f"Expected 3D tensor (C, H, W), got 2D: {frame.shape}")
            elif len(frame.shape) == 3:
                if frame.shape[0] > frame.shape[2]:
                    frame = frame.permute(2, 0, 1)
                if frame.shape[0] != 5:
                    raise ValueError(f"Expected 5 channels, got {frame.shape[0]}")
            
            # Keep at 720x720 (native resolution)
            target_size = 720
            if frame.shape[1:] != (target_size, target_size):
                frame = torch.nn.functional.interpolate(
                    frame.unsqueeze(0),
                    size=(target_size, target_size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            # Data is already normalized to [-1, 1]
            
            frames.append(frame)
        
        # Concatenate frames
        context = torch.cat(frames, dim=0).unsqueeze(0)  # (1, 20, 720, 720)
        return context
    
    @torch.no_grad()
    def predict(self, context, use_ddim=False, ddim_steps=50):
        """
        Predict 2 future frames given 4 context frames
        
        Args:
            context: Tensor of shape (1, 20, 720, 720) or (20, 720, 720)
            use_ddim: Use DDIM for faster sampling (50 steps vs 1000)
            ddim_steps: Number of steps for DDIM sampling
        
        Returns:
            Predicted frames of shape (1, 10, 720, 720)
        """
        if len(context.shape) == 3:
            context = context.unsqueeze(0)
        
        context = context.to(self.device)
        
        # Generate frames
        predicted = self.model.sample(context, self.device, use_ddim=use_ddim, ddim_steps=ddim_steps)
        
        return predicted
    
    def predict_from_paths(self, frame_paths, use_ddim=True):
        """
        Predict frames directly from file paths
        
        Args:
            frame_paths: List of 4 paths to .pt files
            use_ddim: Use fast DDIM sampling (recommended for inference)
        
        Returns:
            Two predicted frames as numpy arrays
        """
        context = self.load_frames(frame_paths)
        predicted = self.predict(context, use_ddim=use_ddim)
        
        # Split into individual frames
        frame1 = predicted[0, :5].cpu().numpy()  # (5, 720, 720)
        frame2 = predicted[0, 5:].cpu().numpy()  # (5, 720, 720)
        
        # Clip to [-1, 1]
        frame1 = np.clip(frame1, -1, 1)
        frame2 = np.clip(frame2, -1, 1)
        
        return frame1, frame2
    
    def visualize_prediction(self, context_paths, output_path=None, ground_truth_paths=None):
        """
        Visualize the prediction along with context frames
        
        Args:
            context_paths: List of 4 paths to context frames
            output_path: Path to save the visualization
            ground_truth_paths: Optional list of 2 paths to ground truth frames
        """
        # Load context
        context = self.load_frames(context_paths)
        
        # Predict
        predicted = self.predict(context)
        
        # Prepare visualization
        num_rows = 2 if ground_truth_paths else 1
        fig, axes = plt.subplots(num_rows, 6, figsize=(18, 6 * num_rows))
        
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Show context frames (use first 3 channels for visualization: VIS, WV, SWIR)
        for i in range(4):
            frame = context[0, i*5:i*5+3].cpu().numpy().transpose(1, 2, 0)
            # Normalize from [-1,1] to [0,1] for display
            frame = (frame + 1) / 2
            axes[0, i].imshow(np.clip(frame, 0, 1))
            axes[0, i].set_title(f'Context {i+1}')
            axes[0, i].axis('off')
        
        # Show predicted frames (use first 3 channels: VIS, WV, SWIR)
        pred_frame1 = predicted[0, :3].cpu().numpy().transpose(1, 2, 0)
        pred_frame2 = predicted[0, 5:8].cpu().numpy().transpose(1, 2, 0)
        
        # Normalize from [-1,1] to [0,1] for display
        pred_frame1 = (pred_frame1 + 1) / 2
        pred_frame2 = (pred_frame2 + 1) / 2
        
        axes[0, 4].imshow(np.clip(pred_frame1, 0, 1))
        axes[0, 4].set_title('Predicted 1')
        axes[0, 4].axis('off')
        
        axes[0, 5].imshow(np.clip(pred_frame2, 0, 1))
        axes[0, 5].set_title('Predicted 2')
        axes[0, 5].axis('off')
        
        # Show ground truth if provided
        if ground_truth_paths:
            gt_frames = self.load_frames(context_paths[:2] + ground_truth_paths)
            gt_frame1 = gt_frames[0, 20:23].cpu().numpy().transpose(1, 2, 0)  # First 3 of 5 channels
            gt_frame2 = gt_frames[0, 25:28].cpu().numpy().transpose(1, 2, 0)  # First 3 of 5 channels
            
            # Normalize from [-1,1] to [0,1] for display
            gt_frame1 = (gt_frame1 + 1) / 2
            gt_frame2 = (gt_frame2 + 1) / 2
            
            # Empty cells for context in second row
            for i in range(4):
                axes[1, i].axis('off')
            
            axes[1, 4].imshow(np.clip(gt_frame1, 0, 1))
            axes[1, 4].set_title('Ground Truth 1')
            axes[1, 4].axis('off')
            
            axes[1, 5].imshow(np.clip(gt_frame2, 0, 1))
            axes[1, 5].set_title('Ground Truth 2')
            axes[1, 5].axis('off')
            
            # Calculate metrics (use all 5 channels)
            gt_tensor_full = gt_frames[0, 20:30].unsqueeze(0)  # Full 10 channels
            
            metrics = calculate_metrics_for_frames(predicted.cpu(), gt_tensor_full.cpu())
            
            plt.suptitle(
                f"PSNR: {metrics['psnr_avg']:.2f} dB | SSIM: {metrics['ssim_avg']:.4f}",
                fontsize=16
            )
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        
        plt.show()


def main():
    # Example usage
    checkpoint_path = 'checkpoints/best_psnr_checkpoint.pt'
    
    # Initialize predictor
    predictor = FramePredictor(checkpoint_path, device='cuda')
    
    # Example: Predict from 4 context frames
    context_paths = [
        'frame_20250401_0130.pt',
        'frame_20250401_0135.pt',  # Replace with your actual files
        'frame_20250401_0140.pt',
        'frame_20250401_0145.pt'
    ]
    
    print("\nGenerating predictions...")
    print("Using DDIM sampling (50 steps) for faster inference")
    
    # Predict frames (DDIM is faster, ~20x speedup)
    frame1, frame2 = predictor.predict_from_paths(context_paths, use_ddim=True)
    
    # Save predicted frames (first 3 channels for RGB visualization)
    frame1_rgb = frame1[:3].transpose(1, 2, 0)
    frame2_rgb = frame2[:3].transpose(1, 2, 0)
    
    # Normalize to [0,1] for saving
    frame1_rgb = (frame1_rgb + 1) / 2
    frame2_rgb = (frame2_rgb + 1) / 2
    
    plt.imsave('predicted_frame1.png', np.clip(frame1_rgb, 0, 1))
    plt.imsave('predicted_frame2.png', np.clip(frame2_rgb, 0, 1))
    
    print("\n✓ Prediction complete!")
    print(f"Frame 1 shape: {frame1.shape} (5 channels: VIS, WV, SWIR, TIR1, TIR2)")
    print(f"Frame 2 shape: {frame2.shape} (5 channels: VIS, WV, SWIR, TIR1, TIR2)")
    print(f"\nSaved visualizations:")
    print(f"  - predicted_frame1.png")
    print(f"  - predicted_frame2.png")
    
    # Optional: Visualize with ground truth for comparison
    # ground_truth_paths = ['frame_20250401_0150.pt', 'frame_20250401_0155.pt']
    # predictor.visualize_prediction(context_paths, 'visualization.png', ground_truth_paths)


if __name__ == '__main__':
    main()
