# Production Training Guide

## Quick Start on Lightning AI A100

### 1. Upload Your Data
Upload all your April 2025 frame files to the workspace.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Data
```bash
python analyze_data.py
```

### 4. Start Training
```bash
python run_training.py
```

## Configuration for A100

The code is pre-configured for A100 training. Key settings in `train.py`:

```python
config = {
    'batch_size': 4,              # Increase to 8-16 for A100 
    'gradient_accumulation_steps': 2,  # Effective batch = 8
    'learning_rate': 2e-4,
    'num_workers': 8,             # Parallel data loading
    'use_amp': True,              # Mixed precision (FP16)
    'pin_memory': True,           # Faster GPU transfer
    'persistent_workers': True    # Keep workers alive
}
```

### Optimize for Your A100:

**For 40GB A100:**
```python
'batch_size': 8
'gradient_accumulation_steps': 2  # Effective batch = 16
```

**For 80GB A100:**
```python
'batch_size': 16
'gradient_accumulation_steps': 2  # Effective batch = 32
```

## Resume Training

If training stops, resume from last checkpoint:

```python
config = {
    ...
    'resume_from': 'checkpoints/latest_checkpoint.pt'
}
```

## Monitor Training

Training creates:
- `checkpoints/latest_checkpoint.pt` - Most recent
- `checkpoints/best_psnr_checkpoint.pt` - Best quality
- `checkpoints/checkpoint_epoch_N.pt` - Every 10 epochs
- `checkpoints/training_history.json` - Loss/metrics

## Expected Training Time

For 1 month of data (~720 files):
- ~715 sequences (4 input + 2 output frames)
- Batch size 8, 100 epochs
- **~8-12 hours on A100**

## Memory Usage

Model uses ~55M parameters:
- **Training**: ~10-15 GB VRAM (batch_size=8)
- **Inference**: ~2-3 GB VRAM

## Troubleshooting

### Out of Memory
```python
'batch_size': 2  # Reduce batch size
'gradient_accumulation_steps': 4  # Keep effective batch
```

### Slow Data Loading
```python
'num_workers': 16  # Increase workers
'prefetch_factor': 4  # More prefetching
```

### Training Divergence
```python
'learning_rate': 1e-4  # Lower learning rate
'gradient_accumulation_steps': 4  # Larger effective batch
```

## Validation

To add validation split:

1. Split your data into train/val directories
2. In `train.py` main():
```python
val_loader = create_dataloader(
    'path/to/val/data',
    batch_size=config['batch_size'],
    shuffle=False
)
```

## Production Tips

1. **Monitor GPU utilization**: `nvidia-smi -l 1`
2. **Check training progress**: `tail -f checkpoints/training_history.json`
3. **Save checkpoints frequently** - training can take hours
4. **Use screen/tmux** - keep training running if disconnected
5. **Test on small subset first** - verify everything works

## Performance Benchmarks

Expected on A100:
- **Training step**: ~0.3-0.5 sec/batch
- **Validation**: ~2-3 min (full dataset)
- **Epoch time**: ~5-10 min (depends on data size)

## Next Steps

After training:
1. Check `best_psnr_checkpoint.pt` for best quality
2. Use `inference.py` to generate predictions
3. Visualize results with provided tools
