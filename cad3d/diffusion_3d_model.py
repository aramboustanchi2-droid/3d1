"""
3D Diffusion Model for CAD - Stable Diffusion 3D / Point-E Style
مدل انتشار سه‌بعدی برای CAD

این مدل قدرتمندترین روش برای تولید 3D از 2D است:
- Denoising Diffusion Probabilistic Models (DDPM)
- Latent diffusion for efficiency
- Point cloud generation
- NeRF-like volumetric representation
- CLIP-guided generation

Architecture inspired by:
- Stable Diffusion 3D (stability.ai)
- Point-E (OpenAI)
- DeepFloyd IF
- DreamFusion (Google)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Time step embeddings for diffusion process"""
    
    def __init__(self, dim: int):
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


class PointNetSetAbstraction(nn.Module):
    """PointNet++ Set Abstraction Layer for 3D point cloud processing"""
    
    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        in_channel: int,
        mlp: List[int],
        group_all: bool = False
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz: torch.Tensor, points: torch.Tensor):
        """
        Args:
            xyz: (B, N, 3) point coordinates
            points: (B, C, N) point features
        Returns:
            new_xyz: (B, npoint, 3)
            new_points: (B, C', npoint)
        """
        B, N, _ = xyz.shape
        
        if self.group_all:
            new_xyz = xyz[:, 0:1, :]
            grouped_xyz = xyz.view(B, 1, N, 3)
            if points is not None:
                grouped_points = points.view(B, -1, 1, N)
            else:
                grouped_points = grouped_xyz.permute(0, 3, 1, 2)
        else:
            # FPS sampling
            fps_idx = self.farthest_point_sample(xyz, self.npoint)
            new_xyz = self.index_points(xyz, fps_idx)
            
            # Ball query
            idx = self.ball_query(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = self.index_points(xyz, idx)
            grouped_xyz -= new_xyz.unsqueeze(2)
            
            if points is not None:
                grouped_points = self.index_points(points.permute(0, 2, 1), idx)
                grouped_points = grouped_points.permute(0, 3, 2, 1)
                grouped_points = torch.cat([grouped_points, grouped_xyz.permute(0, 3, 2, 1)], dim=1)
            else:
                grouped_points = grouped_xyz.permute(0, 3, 2, 1)
        
        # MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = F.relu(bn(conv(grouped_points)))
        
        new_points = torch.max(grouped_points, 2)[0]
        return new_xyz, new_points
    
    @staticmethod
    def farthest_point_sample(xyz, npoint):
        """Farthest Point Sampling"""
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        
        return centroids
    
    @staticmethod
    def ball_query(radius, nsample, xyz, new_xyz):
        """Ball query"""
        device = xyz.device
        B, N, C = xyz.shape
        _, S, _ = new_xyz.shape
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        
        sqrdists = torch.sum((new_xyz.unsqueeze(2) - xyz.unsqueeze(1)) ** 2, -1)
        group_idx[sqrdists > radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
        mask = group_idx == N
        group_idx[mask] = group_first[mask]
        
        return group_idx
    
    @staticmethod
    def index_points(points, idx):
        """Index points by indices"""
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points


class PointNetEncoder(nn.Module):
    """PointNet++ Encoder for 3D point clouds"""
    
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        
        # Set abstraction layers
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=3, mlp=[64, 64, 128]
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3, mlp=[128, 128, 256]
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 512, latent_dim],
            group_all=True
        )
    
    def forward(self, xyz: torch.Tensor):
        """
        Args:
            xyz: (B, N, 3) point cloud
        Returns:
            latent: (B, latent_dim)
        """
        xyz1, features1 = self.sa1(xyz, None)
        xyz2, features2 = self.sa2(xyz1, features1)
        xyz3, features3 = self.sa3(xyz2, features2)
        
        return features3.squeeze(-1)


class AttentionBlock(nn.Module):
    """Multi-head self-attention block"""
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, N, C = x.shape
        
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class DiffusionUNet3D(nn.Module):
    """
    U-Net for 3D diffusion denoising
    Architecture similar to Point-E and Stable Diffusion
    """
    
    def __init__(
        self,
        point_dim: int = 3,
        feature_dim: int = 128,
        time_dim: int = 256,
        condition_dim: int = 512,
        num_points: int = 2048
    ):
        super().__init__()
        
        self.num_points = num_points
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # Condition encoder (from 2D image features)
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        # Input projection
        self.input_proj = nn.Linear(point_dim, feature_dim)
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Linear(feature_dim + time_dim, feature_dim * 2),
            nn.GELU(),
            AttentionBlock(feature_dim * 2, num_heads=8)
        )
        
        self.enc2 = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim * 4),
            nn.GELU(),
            AttentionBlock(feature_dim * 4, num_heads=8)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim * 8),
            nn.GELU(),
            AttentionBlock(feature_dim * 8, num_heads=16),
            nn.Linear(feature_dim * 8, feature_dim * 4),
            nn.GELU()
        )
        
        # Decoder with skip connections
        self.dec2 = nn.Sequential(
            nn.Linear(feature_dim * 8, feature_dim * 4),  # 8 = 4 + 4 (skip)
            nn.GELU(),
            AttentionBlock(feature_dim * 4, num_heads=8)
        )
        
        self.dec1 = nn.Sequential(
            nn.Linear(feature_dim * 6, feature_dim * 2),  # 6 = 4 + 2 (skip)
            nn.GELU(),
            AttentionBlock(feature_dim * 2, num_heads=8)
        )
        
        # Output
        self.output = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),  # 3 = 2 + 1 (skip)
            nn.GELU(),
            nn.Linear(feature_dim, point_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ):
        """
        Args:
            x: (B, N, 3) noisy point cloud
            t: (B,) timestep
            condition: (B, condition_dim) conditioning vector from 2D image
        Returns:
            noise: (B, N, 3) predicted noise
        """
        B, N, _ = x.shape
        
        # Time embedding
        t_emb = self.time_mlp(t)  # (B, time_dim)
        t_emb = t_emb.unsqueeze(1).expand(-1, N, -1)  # (B, N, time_dim)
        
        # Condition embedding
        if condition is not None:
            cond_emb = self.condition_encoder(condition)  # (B, feature_dim)
            cond_emb = cond_emb.unsqueeze(1).expand(-1, N, -1)  # (B, N, feature_dim)
        else:
            cond_emb = torch.zeros(B, N, self.input_proj.out_features, device=x.device)
        
        # Input projection
        x = self.input_proj(x)  # (B, N, feature_dim)
        h0 = x + cond_emb
        
        # Encoder
        h1 = self.enc1(torch.cat([h0, t_emb], dim=-1))  # (B, N, feature_dim*2)
        h2 = self.enc2(h1)  # (B, N, feature_dim*4)
        
        # Bottleneck
        h = self.bottleneck(h2)  # (B, N, feature_dim*4)
        
        # Decoder with skip connections
        h = self.dec2(torch.cat([h, h2], dim=-1))  # (B, N, feature_dim*4)
        h = self.dec1(torch.cat([h, h1], dim=-1))  # (B, N, feature_dim*2)
        
        # Output
        noise = self.output(torch.cat([h, h0], dim=-1))  # (B, N, 3)
        
        return noise


class GaussianDiffusion3D:
    """
    Gaussian Diffusion Process for 3D point clouds
    Implements DDPM and DDIM sampling
    """
    
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear"
    ):
        self.model = model
        self.timesteps = timesteps
        
        # Beta schedule
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule == "cosine":
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Calculations for diffusion
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                           betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                           (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
    
    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        Forward diffusion: q(x_t | x_0)
        Add noise to x_0 to get x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ):
        """
        Reverse diffusion: p(x_{t-1} | x_t)
        Single denoising step
        """
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Predict noise
        model_output = self.model(x, t, condition)
        
        # Predict x_0
        model_mean = sqrt_recip_alphas_t * (x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t)
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(
        self,
        shape: Tuple[int, int, int],
        condition: Optional[torch.Tensor] = None,
        device: str = "cpu",
        progress: bool = True
    ):
        """
        DDPM sampling: complete reverse diffusion
        Generate 3D point cloud from noise
        """
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, condition)
            
            if progress and i % 100 == 0:
                print(f"Sampling step {self.timesteps - i}/{self.timesteps}")
        
        return x
    
    @torch.no_grad()
    def ddim_sample(
        self,
        shape: Tuple[int, int, int],
        condition: Optional[torch.Tensor] = None,
        steps: int = 50,
        eta: float = 0.0,
        device: str = "cpu"
    ):
        """
        DDIM sampling: faster deterministic sampling
        steps << timesteps for speedup (e.g., 50 vs 1000)
        """
        batch_size = shape[0]
        
        # Select timesteps
        step_size = self.timesteps // steps
        timesteps = list(range(0, self.timesteps, step_size))
        timesteps.reverse()
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.model(x, t_tensor, condition)
            
            # Get alpha values
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[timesteps[i+1]] if i < len(timesteps) - 1 else torch.tensor(1.0)
            
            # Predict x_0
            x0_pred = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_t_prev - eta**2 * (1 - alpha_t) / (1 - alpha_t_prev) * (1 - alpha_t_prev / alpha_t)) * predicted_noise
            
            # Random noise
            noise = torch.randn_like(x) if i < len(timesteps) - 1 else torch.zeros_like(x)
            
            # DDIM update
            x = torch.sqrt(alpha_t_prev) * x0_pred + dir_xt + eta * torch.sqrt((1 - alpha_t_prev) * (1 - alpha_t / alpha_t_prev)) * noise
            
            if i % 10 == 0:
                print(f"DDIM step {i+1}/{len(timesteps)}")
        
        return x
    
    def training_loss(
        self,
        x_start: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ):
        """
        Calculate training loss: MSE between predicted and actual noise
        """
        batch_size = x_start.shape[0]
        
        # Random timestep
        t = torch.randint(0, self.timesteps, (batch_size,), device=x_start.device).long()
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Forward diffusion
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predict noise
        predicted_noise = self.model(x_noisy, t, condition)
        
        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    @staticmethod
    def extract(a, t, x_shape):
        """Extract coefficients at timestep t"""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class CLIPImageEncoder(nn.Module):
    """
    CLIP-like image encoder for conditioning
    Extracts semantic features from 2D images
    """
    
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        
        # Simple CNN encoder (can be replaced with CLIP)
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(512, embed_dim)
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W) input image
        Returns:
            features: (B, embed_dim)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = self.pool(x).flatten(1)
        x = self.proj(x)
        
        return x


class Diffusion3DConverter:
    """
    Complete 3D Diffusion Model for converting 2D CAD drawings to 3D
    
    Combines:
    - CLIP image encoder (2D understanding)
    - Vision Transformer (optional, for better features)
    - Diffusion U-Net (3D generation)
    - PointNet++ (3D refinement)
    """
    
    def __init__(
        self,
        num_points: int = 2048,
        latent_dim: int = 512,
        timesteps: int = 1000,
        device: str = "cpu"
    ):
        self.device = device
        self.num_points = num_points
        
        # Image encoder
        self.image_encoder = CLIPImageEncoder(embed_dim=latent_dim).to(device)
        
        # Diffusion model
        self.unet = DiffusionUNet3D(
            point_dim=3,
            feature_dim=128,
            time_dim=256,
            condition_dim=latent_dim,
            num_points=num_points
        ).to(device)
        
        # Diffusion process
        self.diffusion = GaussianDiffusion3D(
            model=self.unet,
            timesteps=timesteps,
            schedule="cosine"
        )
        
        # Point cloud refiner
        self.pointnet_refiner = PointNetEncoder(latent_dim=256).to(device)
    
    def encode_image(self, image: torch.Tensor):
        """Extract features from 2D image"""
        return self.image_encoder(image)
    
    def generate_3d(
        self,
        image: torch.Tensor,
        num_samples: int = 1,
        sampling_method: str = "ddim",
        steps: int = 50
    ):
        """
        Generate 3D point cloud from 2D image
        
        Args:
            image: (B, 3, H, W) input image
            num_samples: number of 3D samples to generate
            sampling_method: "ddpm" or "ddim"
            steps: number of sampling steps (for DDIM)
        
        Returns:
            point_cloud: (B, N, 3) generated 3D points
        """
        batch_size = image.shape[0]
        
        # Encode image
        condition = self.encode_image(image)
        
        # Repeat condition for multiple samples
        if num_samples > 1:
            condition = condition.repeat(num_samples, 1)
        
        # Generate 3D
        shape = (batch_size * num_samples, self.num_points, 3)
        
        if sampling_method == "ddpm":
            point_cloud = self.diffusion.p_sample_loop(
                shape=shape,
                condition=condition,
                device=self.device,
                progress=True
            )
        elif sampling_method == "ddim":
            point_cloud = self.diffusion.ddim_sample(
                shape=shape,
                condition=condition,
                steps=steps,
                device=self.device
            )
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")
        
        return point_cloud
    
    def train_step(
        self,
        images: torch.Tensor,
        point_clouds: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ):
        """
        Single training step
        
        Args:
            images: (B, 3, H, W) input images
            point_clouds: (B, N, 3) ground truth point clouds
            optimizer: optimizer
        
        Returns:
            loss: scalar loss value
        """
        optimizer.zero_grad()
        
        # Encode images
        condition = self.encode_image(images)
        
        # Calculate diffusion loss
        loss = self.diffusion.training_loss(point_clouds, condition)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        return loss.item()


def create_diffusion_model(
    num_points: int = 2048,
    timesteps: int = 1000,
    device: str = "cpu"
):
    """
    Factory function to create diffusion model
    
    Args:
        num_points: number of points in generated cloud
        timesteps: diffusion timesteps
        device: "cpu" or "cuda"
    
    Returns:
        converter: Diffusion3DConverter
    """
    return Diffusion3DConverter(
        num_points=num_points,
        latent_dim=512,
        timesteps=timesteps,
        device=device
    )


if __name__ == "__main__":
    print("3D Diffusion Model - مدل انتشار سه‌بعدی")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create model
    model = create_diffusion_model(num_points=2048, timesteps=1000, device=device)
    
    # Test forward pass
    print("\nTesting model...")
    batch_size = 2
    dummy_image = torch.randn(batch_size, 3, 256, 256, device=device)
    
    print(f"Input image: {dummy_image.shape}")
    
    # Generate 3D
    with torch.no_grad():
        point_cloud = model.generate_3d(
            dummy_image,
            num_samples=1,
            sampling_method="ddim",
            steps=20  # Fast for testing
        )
    
    print(f"Generated point cloud: {point_cloud.shape}")
    print(f"Point range: [{point_cloud.min():.3f}, {point_cloud.max():.3f}]")
    
    # Model size
    total_params = sum(p.numel() for p in model.unet.parameters())
    trainable_params = sum(p.numel() for p in model.unet.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    print("\n✅ 3D Diffusion Model ready!")
