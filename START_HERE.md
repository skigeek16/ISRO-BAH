# Quick Start Guide 🚀

## Your Setup
- Data location: `data/APR25/`
- Training on: A100 GPU (Lightning AI)

---

## Step-by-Step Commands

### 1️⃣ Install Dependencies (FIRST!)
```bash
pip install -r requirements.txt
```

Wait for installation to complete (~1-2 minutes).

---

### 2️⃣ Test Your Setup (RECOMMENDED)
```bash
bash test_setup.sh
```

This validates:
- ✓ Python environment
- ✓ GPU availability  
- ✓ Data files present
- ✓ Model can initialize
- ✓ Data can be loaded

**Expected output:** All checks should pass with ✓

---

### 3️⃣ Start Training
```bash
python run_training.py
```

Or directly:
```bash
python train.py
```

---

### 4️⃣ Monitor Training

**In terminal:**
```bash
# Watch GPU usage
nvidia-smi -l 1

# Monitor training progress (in another terminal)
tail -f checkpoints/training_history.json
```

**Training will show:**
- Loss decreasing over epochs
- Learning rate schedule
- PSNR and SSIM metrics (every 5 epochs)
- Checkpoint saves

---

### 5️⃣ After Training - Run Inference

Edit `inference.py` to use your frame paths:
```python
context_paths = [
    'data/APR25/frame_20250401_0000.pt',
    'data/APR25/frame_20250401_0030.pt',
    'data/APR25/frame_20250401_0100.pt',
    'data/APR25/frame_20250401_0130.pt'
]
```

Then run:
```bash
python inference.py
```

This generates `predicted_frame1.png` and `predicted_frame2.png`

---

## Expected Training Time

For ~720 files (1 month of data):
- **~715 sequences** (4 input + 2 output)
- **100 epochs**
- **Time: 1.5-2.5 hours on A100**

---

## Output Files

During training:
```
checkpoints/
├── latest_checkpoint.pt          # Resume from here
├── best_psnr_checkpoint.pt        # Best quality
├── best_ssim_checkpoint.pt        # Best similarity  
├── checkpoint_epoch_10.pt         # Periodic saves
├── checkpoint_epoch_20.pt
└── training_history.json          # All metrics
```

---

## Troubleshooting

### Out of Memory?
```python
# In train.py, reduce batch size:
'batch_size': 2
'gradient_accumulation_steps': 4  # Keep effective batch = 8
```

### Training Too Slow?
```python
# Increase batch size (if memory allows):
'batch_size': 8
'num_workers': 12
```

### Resume Interrupted Training?
```python
# In train.py config:
'resume_from': 'checkpoints/latest_checkpoint.pt'
```

---

## What to Expect

**First few epochs:**
- Loss: ~0.5-1.0
- Takes a few minutes per epoch

**After 50 epochs:**
- Loss: ~0.01-0.05
- PSNR: 25-35 dB
- SSIM: 0.85-0.95

**After 100 epochs:**
- Loss: ~0.005-0.02
- PSNR: 30-40 dB
- SSIM: 0.90-0.98

---

## Need Help?

Check these files:
- `TRAINING_GUIDE.md` - Detailed guide
- `FIXES_APPLIED.md` - What was optimized
- `README.md` - Full documentation

---

## Pro Tips

1. **Run test_setup.sh first** - saves hours of debugging
2. **Use screen/tmux** - keep training running if disconnected
3. **Monitor GPU with nvidia-smi** - ensure full utilization
4. **Check history.json** - track progress in real-time
5. **Save checkpoints frequently** - training takes hours

---

🎯 **Ready to go! Start with `./test_setup.sh`**
