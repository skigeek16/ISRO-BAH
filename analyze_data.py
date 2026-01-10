"""
Script to analyze the .pt data files
"""
import torch
import numpy as np

def analyze_pt_file(filepath):
    """Comprehensive analysis of a .pt file"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {filepath}")
    print(f"{'='*60}\n")
    
    try:
        # Load the data
        data = torch.load(filepath, weights_only=False)
        
        print(f"1. DATA TYPE")
        print(f"   Type: {type(data)}")
        print(f"   Python type: {data.__class__.__name__}\n")
        
        # If it's a dictionary
        if isinstance(data, dict):
            print(f"2. DICTIONARY STRUCTURE")
            print(f"   Number of keys: {len(data.keys())}")
            print(f"   Keys: {list(data.keys())}\n")
            
            print(f"3. DETAILED KEY ANALYSIS")
            for key, value in data.items():
                print(f"\n   Key: '{key}'")
                print(f"   Value type: {type(value)}")
                
                if isinstance(value, torch.Tensor):
                    print(f"   Shape: {value.shape}")
                    print(f"   Dtype: {value.dtype}")
                    print(f"   Device: {value.device}")
                    print(f"   Min value: {value.min().item():.4f}")
                    print(f"   Max value: {value.max().item():.4f}")
                    print(f"   Mean value: {value.mean().item():.4f}")
                    print(f"   Std value: {value.std().item():.4f}")
                    
                    # Check for NaN or Inf
                    if torch.isnan(value).any():
                        print(f"   ⚠️  Contains NaN values!")
                    if torch.isinf(value).any():
                        print(f"   ⚠️  Contains Inf values!")
                        
                elif isinstance(value, np.ndarray):
                    print(f"   Shape: {value.shape}")
                    print(f"   Dtype: {value.dtype}")
                    print(f"   Min value: {value.min():.4f}")
                    print(f"   Max value: {value.max():.4f}")
                    print(f"   Mean value: {value.mean():.4f}")
                    
                elif isinstance(value, (list, tuple)):
                    print(f"   Length: {len(value)}")
                    if len(value) > 0:
                        print(f"   First element type: {type(value[0])}")
                        
                else:
                    print(f"   Value: {value}")
        
        # If it's a tensor
        elif isinstance(data, torch.Tensor):
            print(f"2. TENSOR PROPERTIES")
            print(f"   Shape: {data.shape}")
            print(f"   Dimensions: {data.dim()}")
            print(f"   Dtype: {data.dtype}")
            print(f"   Device: {data.device}")
            print(f"   Total elements: {data.numel()}")
            print(f"   Memory size: {data.element_size() * data.numel() / (1024**2):.2f} MB\n")
            
            print(f"3. STATISTICS")
            print(f"   Min value: {data.min().item():.4f}")
            print(f"   Max value: {data.max().item():.4f}")
            print(f"   Mean value: {data.mean().item():.4f}")
            print(f"   Std value: {data.std().item():.4f}")
            print(f"   Median value: {data.median().item():.4f}\n")
            
            # Check for NaN or Inf
            print(f"4. DATA QUALITY")
            nan_count = torch.isnan(data).sum().item()
            inf_count = torch.isinf(data).sum().item()
            print(f"   NaN values: {nan_count}")
            print(f"   Inf values: {inf_count}")
            
            if data.dtype in [torch.float32, torch.float64, torch.float16]:
                print(f"   Value range: [{data.min().item():.4f}, {data.max().item():.4f}]")
                if data.max().item() > 1.0:
                    print(f"   ℹ️  Values > 1.0 detected (might need normalization)")
            
            # Dimension interpretation
            print(f"\n5. DIMENSION INTERPRETATION")
            if data.dim() == 2:
                print(f"   Likely format: (Height={data.shape[0]}, Width={data.shape[1]})")
                print(f"   Grayscale image")
            elif data.dim() == 3:
                if data.shape[0] == 3 or data.shape[0] == 1:
                    print(f"   Likely format: (Channels={data.shape[0]}, Height={data.shape[1]}, Width={data.shape[2]})")
                    print(f"   RGB image" if data.shape[0] == 3 else "Grayscale image")
                elif data.shape[2] == 3 or data.shape[2] == 1:
                    print(f"   Likely format: (Height={data.shape[0]}, Width={data.shape[1]}, Channels={data.shape[2]})")
                    print(f"   RGB image (HWC format)" if data.shape[2] == 3 else "Grayscale image (HWC format)")
                else:
                    print(f"   Custom format: {data.shape}")
            elif data.dim() == 4:
                print(f"   Likely format: (Batch={data.shape[0]}, Channels={data.shape[1]}, Height={data.shape[2]}, Width={data.shape[3]})")
                print(f"   Batch of images")
            
            # Sample values
            print(f"\n6. SAMPLE VALUES")
            if data.numel() <= 100:
                print(f"   All values:\n{data}")
            else:
                print(f"   First few values (flattened):")
                print(f"   {data.flatten()[:10]}")
                print(f"   Last few values (flattened):")
                print(f"   {data.flatten()[-10:]}")
        
        # If it's a numpy array
        elif isinstance(data, np.ndarray):
            print(f"2. NUMPY ARRAY PROPERTIES")
            print(f"   Shape: {data.shape}")
            print(f"   Dtype: {data.dtype}")
            print(f"   Min: {data.min():.4f}")
            print(f"   Max: {data.max():.4f}")
            print(f"   Mean: {data.mean():.4f}")
        
        # If it's a list or tuple
        elif isinstance(data, (list, tuple)):
            print(f"2. SEQUENCE PROPERTIES")
            print(f"   Length: {len(data)}")
            if len(data) > 0:
                print(f"   First element type: {type(data[0])}")
                print(f"   First element: {data[0]}")
                
                if isinstance(data[0], torch.Tensor):
                    print(f"   First tensor shape: {data[0].shape}")
        
        # Other types
        else:
            print(f"2. RAW DATA")
            print(f"   {data}")
        
        print(f"\n{'='*60}\n")
        
    except Exception as e:
        print(f"❌ Error loading file: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    import glob
    
    # Check if specific file provided
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        analyze_pt_file(filepath)
    else:
        # Analyze all .pt files in current directory
        pt_files = glob.glob("*.pt")
        
        if not pt_files:
            print("No .pt files found in current directory")
        else:
            print(f"Found {len(pt_files)} .pt files")
            
            for pt_file in sorted(pt_files)[:5]:  # Analyze first 5 files
                analyze_pt_file(pt_file)
            
            if len(pt_files) > 5:
                print(f"\nℹ️  Showing first 5 files. Total files: {len(pt_files)}")
                print(f"   Run with filename argument to analyze specific file:")
                print(f"   python analyze_data.py <filename>")
