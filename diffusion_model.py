import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy


class EMA:
    """Exponential Moving Average for model parameters.
    
    EMA maintains a shadow copy of model parameters that is updated
    with exponential moving average. This provides more stable
    inference compared to using the trained model directly.
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
    
    def update(self, model):
        """Update EMA parameters"""
        with torch.no_grad():
            for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)
    
    def get_model(self):
        """Get the EMA model for inference"""
        return self.shadow
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.dropout(h)
        
        # Add time embedding
        time_emb = self.act(self.time_mlp(t_emb))
        h = h + time_emb[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)
        
        attn = torch.bmm(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(attn, v)
        out = out.permute(0, 2, 1).reshape(b, c, h, w)
        out = self.proj(out)
        
        return out + x


class UNet(nn.Module):
    def __init__(self, in_channels=30, out_channels=10, base_channels=64, time_emb_dim=256):
        """
        UNet for diffusion model
        in_channels: 4 context frames (20 ch) + 2 noisy target frames (10 ch) = 30
        out_channels: 2 frames * 5 channels = 10 (predicted noise)
        """
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # Encoder
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        self.down1 = nn.ModuleList([
            ResidualBlock(base_channels, base_channels, time_emb_dim),
            ResidualBlock(base_channels, base_channels, time_emb_dim)
        ])
        self.downsample1 = nn.Conv2d(base_channels, base_channels, 4, stride=2, padding=1)
        
        self.down2 = nn.ModuleList([
            ResidualBlock(base_channels, base_channels * 2, time_emb_dim),
            ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim),
            AttentionBlock(base_channels * 2)  # Added attention at level 2
        ])
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1)
        
        self.down3 = nn.ModuleList([
            ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim),
            ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim),
            AttentionBlock(base_channels * 4)  # Added attention at level 3
        ])
        self.downsample3 = nn.Conv2d(base_channels * 4, base_channels * 4, 4, stride=2, padding=1)
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResidualBlock(base_channels * 4, base_channels * 8, time_emb_dim),
            AttentionBlock(base_channels * 8),
            ResidualBlock(base_channels * 8, base_channels * 8, time_emb_dim)
        ])
        
        # Decoder
        self.upsample3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, stride=2, padding=1)
        self.up3 = nn.ModuleList([
            ResidualBlock(base_channels * 8, base_channels * 4, time_emb_dim),
            ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim),
            AttentionBlock(base_channels * 4)  # Added attention in decoder
        ])
        
        self.upsample2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.up2 = nn.ModuleList([
            ResidualBlock(base_channels * 4, base_channels * 2, time_emb_dim),
            ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim),
            AttentionBlock(base_channels * 2)  # Added attention in decoder
        ])
        
        self.upsample1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.up1 = nn.ModuleList([
            ResidualBlock(base_channels * 2, base_channels, time_emb_dim),
            ResidualBlock(base_channels, base_channels, time_emb_dim)
        ])
        
        self.conv_out = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Encoder
        x = self.conv_in(x)
        
        # Down 1
        for block in self.down1:
            if isinstance(block, AttentionBlock):
                x = block(x)
            else:
                x = block(x, t_emb)
        skip1 = x  # Save BEFORE downsampling
        x = self.downsample1(x)
        
        # Down 2
        for block in self.down2:
            if isinstance(block, AttentionBlock):
                x = block(x)
            else:
                x = block(x, t_emb)
        skip2 = x  # Save BEFORE downsampling
        x = self.downsample2(x)
        
        # Down 3
        for block in self.down3:
            if isinstance(block, AttentionBlock):
                x = block(x)
            else:
                x = block(x, t_emb)
        skip3 = x  # Save BEFORE downsampling
        x = self.downsample3(x)
        
        # Bottleneck
        x = self.bottleneck[0](x, t_emb)
        x = self.bottleneck[1](x)  # Attention
        x = self.bottleneck[2](x, t_emb)
        
        # Decoder
        x = self.upsample3(x)
        x = torch.cat([x, skip3], dim=1)
        for block in self.up3:
            if isinstance(block, AttentionBlock):
                x = block(x)
            else:
                x = block(x, t_emb)
        
        x = self.upsample2(x)
        x = torch.cat([x, skip2], dim=1)
        for block in self.up2:
            if isinstance(block, AttentionBlock):
                x = block(x)
            else:
                x = block(x, t_emb)
        
        x = self.upsample1(x)
        x = torch.cat([x, skip1], dim=1)
        for block in self.up1:
            if isinstance(block, AttentionBlock):
                x = block(x)
            else:
                x = block(x, t_emb)
        
        return self.conv_out(x)


class DiffusionModel(nn.Module):
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.timesteps = timesteps
        
        # Define beta schedule - register as buffers so they move with model
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register as buffers (non-trainable, but move with model.to(device))
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        
        # Model
        self.model = UNet(in_channels=30, out_channels=10)

    def forward_diffusion(self, x_0, t, noise=None):
        """Add noise to the data"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Ensure timesteps are valid
        if isinstance(t, int):
            t = torch.tensor([t], device=x_0.device)
        
        # Index directly on GPU - buffers are already on the same device as model
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_noise(self, x_t, context, t):
        """Predict noise with context (4 past frames)"""
        # Concatenate context frames with noisy target
        # context: (B, 20, H, W) - 4 frames
        # x_t: (B, 10, H, W) - 2 noisy frames
        # x_input: (B, 30, H, W) - total input to UNet
        x_input = torch.cat([context, x_t], dim=1)
        
        # Validate input shape
        expected_channels = 30  # 20 context + 10 target
        if x_input.shape[1] != expected_channels:
            raise ValueError(f"Expected {expected_channels} channels, got {x_input.shape[1]}")
        
        return self.model(x_input, t)

    @torch.no_grad()
    def sample(self, context, device, use_ddim=False, ddim_steps=50):
        """
        Generate 2 future frames given 4 past frames
        context: (B, 20, H, W) - 4 frames * 5 channels
        returns: (B, 10, H, W) - 2 frames * 5 channels
        
        Args:
            use_ddim: Use DDIM sampling for faster generation (50 steps vs 1000)
            ddim_steps: Number of steps for DDIM (only used if use_ddim=True)
        """
        B, _, H, W = context.shape
        x_t = torch.randn(B, 10, H, W).to(device)
        
        if use_ddim:
            # DDIM sampling - much faster
            timestep_seq = torch.linspace(self.timesteps - 1, 0, ddim_steps, dtype=torch.long)
            for i in range(len(timestep_seq)):
                t = timestep_seq[i].item()
                t_batch = torch.full((B,), t, device=device, dtype=torch.long)
                
                # Predict noise
                predicted_noise = self.predict_noise(x_t, context, t_batch)
                
                # DDIM update rule
                alpha_t = self.alphas_cumprod[t].to(device)
                if i < len(timestep_seq) - 1:
                    alpha_prev = self.alphas_cumprod[timestep_seq[i+1].item()].to(device)
                else:
                    alpha_prev = torch.tensor(1.0, device=device)
                
                pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
                pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)  # Clamp to valid range
                direction = torch.sqrt(1 - alpha_prev) * predicted_noise
                x_t = torch.sqrt(alpha_prev) * pred_x0 + direction
        else:
            # Standard DDPM sampling - slower but higher quality
            for t in reversed(range(self.timesteps)):
                t_batch = torch.full((B,), t, device=device, dtype=torch.long)
                
                # Predict noise
                predicted_noise = self.predict_noise(x_t, context, t_batch)
                
                # Compute x_{t-1}
                betas_t = self.betas[t].to(device)
                sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(device)
                sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].to(device)
                
                model_mean = sqrt_recip_alphas_t * (
                    x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
                )
                
                if t > 0:
                    posterior_variance_t = self.posterior_variance[t].to(device)
                    noise = torch.randn_like(x_t)
                    x_t = model_mean + torch.sqrt(posterior_variance_t) * noise
                else:
                    x_t = model_mean
        
        return x_t
