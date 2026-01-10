#!/usr/bin/env python3
"""
Quick Start Training Script for A100 on Lightning AI

This script provides a production-ready training setup with:
- Automatic GPU detection and optimization
- Mixed precision training (AMP)
- Gradient accumulation
- Checkpointing and resume support
- Error handling and recovery
"""

import os
import sys
import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import main

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  Diffusion Model for Satellite Frame Prediction             ║
    ║  Optimized for A100 GPU Training                            ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Verify CUDA availability
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ Warning: CUDA not available, training on CPU")
    
    print()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
