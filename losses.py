"""
Advanced loss functions for diffusion model training.
Includes perceptual loss (VGG-based) and SSIM loss components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for better visual quality"""
    
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3']):
        super().__init__()
        # Load pretrained VGG16
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Layer indices for feature extraction
        self.layer_indices = {
            'relu1_2': 4,
            'relu2_2': 9, 
            'relu3_3': 16,
            'relu4_3': 23,
            'relu5_3': 30
        }
        self.selected_indices = [self.layer_indices[l] for l in layers]
        self.max_layer = max(self.selected_indices) + 1
        
        # Only keep layers up to max needed
        self.vgg = nn.Sequential(*list(vgg.children())[:self.max_layer])
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def preprocess(self, x):
        """Convert from [-1,1] 5-channel to [0,1] 3-channel for VGG"""
        x = x[:, :3]
        x = (x + 1) / 2
        x = torch.clamp(x, 0, 1)
        return (x - self.mean) / self.std
    
    def forward(self, pred, target):
        pred = self.preprocess(pred)
        target = self.preprocess(target)
        
        loss = 0.0
        x_pred, x_target = pred, target
        
        for i, layer in enumerate(self.vgg):
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            
            if i in self.selected_indices:
                loss += F.l1_loss(x_pred, x_target)
        
        return loss / len(self.selected_indices)


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss for image quality optimization."""
    
    def __init__(self, window_size=11, channel=5, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.register_buffer('window', self._create_window(window_size, channel))
    
    def _gaussian(self, window_size, sigma):
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()
    
    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _ssim(self, img1, img2):
        channel = img1.size(1)
        
        if channel != self.channel:
            window = self._create_window(self.window_size, channel).to(img1.device)
        else:
            window = self.window
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        L = 2.0
        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, pred, target):
        return 1 - self._ssim(pred, target)


class CombinedLoss(nn.Module):
    """Combined loss function with MSE, perceptual, and SSIM components."""
    
    def __init__(self, mse_weight=1.0, perceptual_weight=0.1, ssim_weight=0.1, 
                 use_perceptual=True, use_ssim=True):
        super().__init__()
        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        
        self.mse = nn.MSELoss()
        
        if use_perceptual and perceptual_weight > 0:
            self.perceptual = PerceptualLoss()
        else:
            self.perceptual = None
            
        if use_ssim and ssim_weight > 0:
            self.ssim = SSIMLoss(channel=10)
        else:
            self.ssim = None
    
    def forward(self, pred_noise, target_noise, pred_x0=None, target_x0=None):
        losses = {}
        losses['mse'] = self.mse(pred_noise, target_noise) * self.mse_weight
        
        if pred_x0 is not None and target_x0 is not None:
            if self.perceptual is not None:
                losses['perceptual'] = self.perceptual(pred_x0, target_x0) * self.perceptual_weight
            if self.ssim is not None:
                losses['ssim'] = self.ssim(pred_x0, target_x0) * self.ssim_weight
        
        losses['total'] = sum(losses.values())
        return losses


class NoisePredictionLoss(nn.Module):
    """Standard noise prediction loss for diffusion training."""
    
    def __init__(self, loss_type='mse'):
        super().__init__()
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'huber':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, pred_noise, target_noise):
        return self.loss_fn(pred_noise, target_noise)
