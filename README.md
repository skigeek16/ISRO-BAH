# 🛰️ Satellite Frame Prediction using Diffusion Models

A PyTorch implementation of a **diffusion-based deep learning model** for predicting future satellite imagery frames from historical observations. Developed as part of the **ISRO-BAH (Bharat AI Hackathon)** challenge.

## 🎯 Overview

This project implements a **UNet-based Diffusion Model** that takes **4 consecutive satellite frames** as input and predicts **2 future frames**. The model is designed to work with **multi-spectral satellite imagery** containing 5 channels:

| Channel | Description |
|---------|-------------|
| **VIS** | Visible spectrum |
| **WV** | Water Vapor |
| **SWIR** | Shortwave Infrared |
| **TIR1** | Thermal Infrared 1 |
| **TIR2** | Thermal Infrared 2 |

## 🏗️ Model Architecture

```
Input: 4 frames × 5 channels = 20 channels (720×720)
       ↓
   Concatenate with 2 noisy target frames (10 channels)
       ↓
   UNet with Attention (30 → 10 channels)
       ↓
Output: 2 predicted frames × 5 channels = 10 channels (720×720)
```

### Key Features

- **~55M parameters** UNet with residual blocks and self-attention
- **DDPM training** with 1000 timesteps
- **DDIM sampling** for faster inference (50-100 steps)
- **Mixed precision training** (FP16) for memory efficiency
- **Gradient accumulation** for larger effective batch sizes
- **PSNR & SSIM metrics** for quality evaluation

## 📁 Project Structure

```
ISRO_BAH/
├── diffusion_model.py    # Core diffusion model architecture
├── dataset.py            # Data loading and preprocessing
├── train.py              # Training script with validation
├── inference.py          # Inference utilities
├── run_inference.py      # Standalone inference script
├── losses.py             # Loss functions (MSE + SSIM)
├── metrics.py            # PSNR and SSIM implementations
├── requirements.txt      # Python dependencies
├── run_training.py       # Training launcher
└── analyze_data.py       # Data analysis utilities
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ISRO_BAH.git
cd ISRO_BAH

# Install dependencies
pip install -r requirements.txt
```

### Data Format

The model expects `.pt` files containing frame data in dictionary format:

```python
{
    'frame_data': torch.Tensor,  # Shape: (5, 720, 720)
    'metadata': dict,             # Contains timestamp, channels, etc.
    'config': dict,               # Channel configurations
    'version': str                # Data version
}
```

### Training

```bash
# Start training (configure paths in train.py)
python train.py
```

**Configuration options:**
- `data_dir`: Directory containing `.pt` files
- `batch_size`: Batch size (default: 4)
- `gradient_accumulation_steps`: For effective batch size (default: 2)
- `num_epochs`: Training epochs (default: 200)
- `learning_rate`: Learning rate (default: 2e-4)

### Inference

```python
from inference import FramePredictor

# Load trained model
predictor = FramePredictor('checkpoints/best_psnr_checkpoint.pt', device='cuda')

# Predict from 4 context frames
context_paths = ['frame1.pt', 'frame2.pt', 'frame3.pt', 'frame4.pt']
frame1, frame2 = predictor.predict_from_paths(context_paths)

# Visualize results
predictor.visualize_prediction(context_paths, output_path='result.png')
```

## 📊 Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **PSNR** | Peak Signal-to-Noise Ratio | Higher is better (20-40 dB) |
| **SSIM** | Structural Similarity Index | Higher is better (0-1) |

## 💡 Training Tips

1. **GPU Memory**: Reduce `batch_size` if you encounter OOM errors
2. **Data Requirements**: Ensure at least 6 consecutive frames per sequence
3. **Checkpointing**: Best models are saved based on PSNR/SSIM metrics
4. **A100 Optimized**: TF32 enabled for faster training on Ampere GPUs

## 🛠️ Customization

### Adjust Diffusion Steps
```python
model = DiffusionModel(timesteps=500)  # Reduce for faster inference
```

### Change Image Resolution
Modify the UNet architecture in `diffusion_model.py` (adjust downsample/upsample layers).

### Different Number of Frames
Modify `in_channels` and `out_channels` in UNet initialization.

## 📚 References

- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)
- [Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502)

## 📄 License

MIT License

## 🙏 Acknowledgments

Developed for the **ISRO Bharat AI Hackathon (BAH)** - India's initiative to leverage AI for space applications.
# ISRO-BAH
