import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def calculate_psnr(img1, img2, max_val=2.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images using torchmetrics
    
    Args:
        img1: Tensor of shape (B, C, H, W) or (C, H, W)
        img2: Tensor of shape (B, C, H, W) or (C, H, W)
        max_val: Maximum possible pixel value (2.0 for [-1,1] range)
    
    Returns:
        PSNR value in dB
    """
    psnr_metric = PeakSignalNoiseRatio(data_range=max_val).to(img1.device)
    
    # Ensure batch dimension
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    psnr = psnr_metric(img1, img2)
    return psnr.item()


def calculate_ssim(img1, img2, window_size=11, channel=5, max_val=2.0):
    """
    Calculate Structural Similarity Index (SSIM) between two images using torchmetrics
    
    Args:
        img1: Tensor of shape (B, C, H, W)
        img2: Tensor of shape (B, C, H, W)
        window_size: Size of the Gaussian window
        channel: Number of channels (unused, kept for compatibility)
        max_val: Maximum possible pixel value
    
    Returns:
        SSIM value
    """
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=max_val, kernel_size=window_size).to(img1.device)
    
    # Ensure batch dimension
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    ssim = ssim_metric(img1, img2)
    return ssim.item()


def calculate_metrics_for_frames(pred_frames, target_frames):
    """
    Calculate PSNR and SSIM for predicted frames
    
    Args:
        pred_frames: Predicted frames, shape (B, 10, H, W) - 2 frames * 5 channels
        target_frames: Ground truth frames, shape (B, 10, H, W)
    
    Returns:
        Dictionary with metrics
    """
    # Ensure values are in [-1, 1] range
    pred_frames = torch.clamp(pred_frames, -1, 1)
    target_frames = torch.clamp(target_frames, -1, 1)
    
    batch_size = pred_frames.shape[0]
    
    # Calculate metrics for each frame separately
    frame1_pred = pred_frames[:, :5, :, :]  # First frame (5 channels)
    frame1_target = target_frames[:, :5, :, :]
    
    frame2_pred = pred_frames[:, 5:, :, :]  # Second frame (5 channels)
    frame2_target = target_frames[:, 5:, :, :]
    
    # Calculate PSNR
    psnr_frame1 = calculate_psnr(frame1_pred, frame1_target)
    psnr_frame2 = calculate_psnr(frame2_pred, frame2_target)
    psnr_avg = (psnr_frame1 + psnr_frame2) / 2
    
    # Calculate SSIM
    ssim_frame1 = calculate_ssim(frame1_pred, frame1_target)
    ssim_frame2 = calculate_ssim(frame2_pred, frame2_target)
    ssim_avg = (ssim_frame1 + ssim_frame2) / 2
    
    return {
        'psnr_frame1': psnr_frame1,
        'psnr_frame2': psnr_frame2,
        'psnr_avg': psnr_avg,
        'ssim_frame1': ssim_frame1,
        'ssim_frame2': ssim_frame2,
        'ssim_avg': ssim_avg
    }
