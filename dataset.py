import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob


class SequenceAugmentation:
    """Apply synchronized augmentation to all frames in a sequence."""
    
    def __init__(self, p_flip_h=0.5, p_flip_v=0.5, p_rotate=0.5):
        self.p_flip_h = p_flip_h
        self.p_flip_v = p_flip_v
        self.p_rotate = p_rotate
    
    def __call__(self, frames):
        do_flip_h = torch.rand(1).item() < self.p_flip_h
        do_flip_v = torch.rand(1).item() < self.p_flip_v
        rotation_k = torch.randint(0, 4, (1,)).item() if torch.rand(1).item() < self.p_rotate else 0
        
        augmented = []
        for frame in frames:
            if do_flip_h:
                frame = torch.flip(frame, dims=[-1])
            if do_flip_v:
                frame = torch.flip(frame, dims=[-2])
            if rotation_k > 0:
                frame = torch.rot90(frame, k=rotation_k, dims=[-2, -1])
            augmented.append(frame)
        return augmented


class FrameSequenceDataset(Dataset):
    """
    Dataset for frame prediction
    Expects data files in format: frame_YYYYMMDD_HHMM.pt
    Each file should contain frames of shape (C, H, W) or sequence of frames
    """
    
    def __init__(self, data_dir, sequence_length=6, transform=None, validate_data=True, augment=False):
        """
        Args:
            data_dir: Directory containing .pt files
            sequence_length: Total number of frames in sequence (4 input + 2 output)
            transform: Optional transform to apply to frames
            validate_data: Whether to validate first file on init
            augment: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.augmentation = SequenceAugmentation() if augment else None
        
        # Load all .pt files
        self.data_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        
        if len(self.data_files) < sequence_length:
            raise ValueError(f"Not enough data files. Need at least {sequence_length}, found {len(self.data_files)}")
        
        print(f"Found {len(self.data_files)} data files")
        print(f"Can create {len(self.data_files) - sequence_length + 1} sequences")
        print(f"Data augmentation: {'enabled' if augment else 'disabled'}")
        
        # Validate first file to ensure correct format
        if validate_data and len(self.data_files) > 0:
            self._validate_data_format(self.data_files[0])
    
    def _validate_data_format(self, file_path):
        """Validate that data file has correct format"""
        try:
            data = torch.load(file_path, weights_only=False)
            
            if isinstance(data, dict):
                frame = data.get('frame_data', None)
                if frame is None:
                    raise ValueError(f"Dictionary missing 'frame_data' key. Found keys: {list(data.keys())}")
            else:
                frame = data
            
            if not isinstance(frame, torch.Tensor):
                frame = torch.tensor(frame)
            
            expected_shape = (5, 720, 720)
            if frame.shape != expected_shape:
                print(f"\u26a0️  Warning: Expected shape {expected_shape}, got {frame.shape}")
                print(f"   Data will be resized/reformatted automatically")
            
            print(f"✓ Data validation passed: shape={frame.shape}, dtype={frame.dtype}, range=[{frame.min():.2f}, {frame.max():.2f}]")
            
        except Exception as e:
            raise ValueError(f"Data validation failed for {file_path}: {e}")
    
    def __len__(self):
        # Number of valid sequences we can create
        return len(self.data_files) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        """
        Returns:
            context: 4 past frames concatenated (20, 720, 720) - 4 frames * 5 channels
            target: 2 future frames concatenated (10, 720, 720) - 2 frames * 5 channels
        """
        # Load sequence of frames with error handling
        frames = []
        for i in range(self.sequence_length):
            frame_path = self.data_files[idx + i]
            
            try:
                frame = torch.load(frame_path, weights_only=False)
                
                # Handle different data formats
                if isinstance(frame, dict):
                    # Extract frame_data from the dictionary structure
                    frame = frame.get('frame_data', frame.get('frame', frame.get('data', None)))
                    if frame is None:
                        raise ValueError(f"Cannot extract frame from dict. Keys: {list(frame.keys()) if isinstance(frame, dict) else 'N/A'}")
                
                # Ensure frame is a tensor
                if not isinstance(frame, torch.Tensor):
                    frame = torch.tensor(frame, dtype=torch.float32)
                
                # Convert to float32 if needed
                if frame.dtype != torch.float32:
                    frame = frame.float()
                
                # Expected shape: (5, 720, 720) - 5 channels at 720x720
                # Channels: VIS, WV, SWIR, TIR1, TIR2
                if len(frame.shape) == 2:
                    raise ValueError(f"Expected 3D tensor (C, H, W), got 2D: {frame.shape}")
                elif len(frame.shape) == 3:
                    # Check if channels first or last
                    if frame.shape[0] > frame.shape[2]:  # Likely (H, W, C)
                        frame = frame.permute(2, 0, 1)
                    # Should have 5 channels
                    if frame.shape[0] != 5:
                        raise ValueError(f"Expected 5 channels (VIS, WV, SWIR, TIR1, TIR2), got {frame.shape[0]}")
                elif len(frame.shape) == 4:
                    # Squeeze batch dimension if present
                    frame = frame.squeeze(0)
                
                # Keep at 720x720 (native resolution)
                target_size = 720
                if frame.shape[1:] != (target_size, target_size):
                    frame = torch.nn.functional.interpolate(
                        frame.unsqueeze(0), 
                        size=(target_size, target_size), 
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                
                # Data is already normalized to [-1, 1], verify range
                if frame.min() < -1.5 or frame.max() > 1.5:
                    # Renormalize if needed
                    if frame.min() >= 0 and frame.max() <= 1:
                        # [0, 1] -> [-1, 1]
                        frame = frame * 2 - 1
                    elif frame.min() >= 0 and frame.max() > 1:
                        # Assume [0, 255] or similar
                        frame = (frame / frame.max()) * 2 - 1
                
                frames.append(frame)
                
            except Exception as e:
                raise RuntimeError(f"Error loading frame {i} from {frame_path}: {str(e)}")
        
        # Apply transforms if any
        if self.transform:
            frames = [self.transform(f) for f in frames]
        
        # Apply synchronized augmentation
        if self.augmentation is not None:
            frames = self.augmentation(frames)
        
        # Split into context (4 frames) and target (2 frames)
        context_frames = frames[:4]  # First 4 frames
        target_frames = frames[4:6]  # Next 2 frames
        
        # Concatenate along channel dimension
        context = torch.cat(context_frames, dim=0)  # Shape: (20, 720, 720)
        target = torch.cat(target_frames, dim=0)     # Shape: (10, 720, 720)
        
        return context, target


def create_dataloader(data_dir, batch_size=4, num_workers=4, shuffle=True, pin_memory=True, persistent_workers=True, augment=False):
    """
    Create DataLoader for training
    
    Args:
        data_dir: Directory containing .pt files
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        pin_memory: Pin memory for faster GPU transfer
        persistent_workers: Keep workers alive between epochs
        augment: Whether to apply data augmentation
    
    Returns:
        DataLoader instance
    """
    dataset = FrameSequenceDataset(data_dir, augment=augment)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    return dataloader
