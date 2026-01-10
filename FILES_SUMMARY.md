# Essential Files for Lightning AI Training

## Core Training Files (Required)
1. `train.py` - Main training script with DDIM validation fix
2. `diffusion_model.py` - UNet and diffusion model implementation
3. `dataset.py` - Data loading and preprocessing
4. `metrics.py` - PSNR and SSIM calculations
5. `inference.py` - Model inference and prediction
6. `requirements.txt` - Python dependencies

## Utility Scripts (Optional but Useful)
7. `run_training.py` - Quick training launcher
8. `test_setup.sh` - Environment validation script
9. `analyze_data.py` - Data file analysis tool

## Documentation
10. `README.md` - Project overview and usage
11. `START_HERE.md` - Quick start guide
12. `TRAINING_GUIDE.md` - Detailed training instructions
13. `FIXES_APPLIED.md` - Bug fixes and optimizations
14. `VALIDATION_FIX.md` - Explanation of the DDIM fix
15. `VERIFICATION_REPORT.md` - Pre-deployment verification

## Removed Files
- ✅ `checkpoint_epoch_30.pt` (248 MB - not needed for training)
- ✅ `frame_20250401_0130.pt` (10 MB - test file)
- ✅ `download_checkpoint.py` (utility script)
- ✅ `inspect_checkpoint.py` (utility script)
- ✅ `test_validation_fix.py` (test script)

## Upload to Lightning AI
Upload all remaining files to your Lightning AI studio. The essential files are small (~50 KB total for Python scripts).

## Quick Deploy
```bash
# On Lightning AI:
pip install -r requirements.txt
python train.py
```

Your training will automatically resume from the latest checkpoint if available.
