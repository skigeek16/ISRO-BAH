#!/bin/bash

# Quick test script to verify everything works before full training
# Run this first with a small subset of data

echo "==========================================="
echo " Pre-Training Validation"
echo "==========================================="
echo ""

# Check Python and PyTorch
echo "1. Checking Python environment..."
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python3 -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"
if [ $? -eq 0 ]; then
    python3 -c "import torch; print(f'✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
fi
echo ""

# Check data files
echo "2. Checking data files..."
DATA_DIR="data/APR25"
if [ -d "$DATA_DIR" ]; then
    DATA_COUNT=$(ls -1 $DATA_DIR/*.pt 2>/dev/null | wc -l)
    echo "✓ Found $DATA_COUNT .pt files in $DATA_DIR"
    if [ $DATA_COUNT -lt 6 ]; then
        echo "⚠ Warning: Need at least 6 files for training (4 context + 2 target)"
        exit 1
    fi
else
    echo "⚠ Warning: Directory $DATA_DIR not found"
    echo "   Please ensure your data is in $DATA_DIR/"
    exit 1
fi
echo ""

# Analyze first file
echo "3. Analyzing data format..."
FIRST_FILE=$(ls -1 $DATA_DIR/*.pt 2>/dev/null | head -n 1)
if [ ! -z "$FIRST_FILE" ]; then
    python3 analyze_data.py "$FIRST_FILE" | head -n 30
else
    echo "⚠ No .pt files found"
    exit 1
fi
echo ""

# Test data loading
echo "4. Testing data loading..."
python3 -c "
from dataset import FrameSequenceDataset
import torch

try:
    dataset = FrameSequenceDataset('data/APR25', sequence_length=6, validate_data=True)
    print(f'✓ Dataset created: {len(dataset)} sequences')
    
    # Try loading one sample
    context, target = dataset[0]
    print(f'✓ Sample loaded: context={context.shape}, target={target.shape}')
    print(f'✓ Value range: context=[{context.min():.2f}, {context.max():.2f}]')
    print('✓ Data loading test passed!')
except Exception as e:
    print(f'✗ Data loading failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"
echo ""

# Test model creation
echo "5. Testing model initialization..."
python3 -c "
from diffusion_model import DiffusionModel
import torch

try:
    model = DiffusionModel(timesteps=100)  # Small timesteps for test
    print(f'✓ Model created')
    print(f'✓ Parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Test forward pass
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    context = torch.randn(1, 20, 720, 720).to(device)
    target = torch.randn(1, 10, 720, 720).to(device)
    t = torch.randint(0, 100, (1,)).to(device)
    
    # Test forward diffusion
    noisy = model.forward_diffusion(target, t)
    print(f'✓ Forward diffusion: {noisy.shape}')
    
    # Test noise prediction
    noise_pred = model.predict_noise(noisy, context, t)
    print(f'✓ Noise prediction: {noise_pred.shape}')
    
    print('✓ Model test passed!')
except Exception as e:
    print(f'✗ Model test failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"
echo ""

# Check disk space
echo "6. Checking disk space..."
df -h . | tail -n 1
echo ""

echo "==========================================="
echo " ✓ All pre-training checks passed!"
echo "==========================================="
echo ""
echo "Ready to start training with:"
echo "  python run_training.py"
echo ""
