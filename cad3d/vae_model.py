"""
Variational Autoencoder (VAE) for CAD 3D Conversion
مدل VAE برای تبدیل CAD به 3D

این مدل از Autoencoders برای:
- فشرده‌سازی و بازسازی نقشه‌های 2D
- استخراج ویژگی‌های latent space
- تولید سریع و سبک 3D
- ترکیب با سایر مدل‌ها

مزایا:
- سریع و کارآمد (سبک‌تر از Diffusion)
- یادگیری representation قوی
- قابل ترکیب با ViT و Diffusion
- continuous latent space برای interpolation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class Encoder2D(nn.Module):
    """
    Encoder for 2D CAD drawings
    Compresses image to latent representation
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 512,
        base_channels: int = 64
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder path: 256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )  # 128x128
        
        self.res1 = ResidualBlock(base_channels)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )  # 64x64
        
        self.res2 = ResidualBlock(base_channels * 2)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )  # 32x32
        
        self.res3 = ResidualBlock(base_channels * 4)
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )  # 16x16
        
        self.res4 = ResidualBlock(base_channels * 8)
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )  # 8x8
        
        # Global pooling and latent projection
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # VAE: output both mu and logvar
        self.fc_mu = nn.Linear(base_channels * 8, latent_dim)
        self.fc_logvar = nn.Linear(base_channels * 8, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) input image
        Returns:
            mu: (B, latent_dim) mean
            logvar: (B, latent_dim) log variance
        """
        x = self.conv1(x)
        x = self.res1(x)
        
        x = self.conv2(x)
        x = self.res2(x)
        
        x = self.conv3(x)
        x = self.res3(x)
        
        x = self.conv4(x)
        x = self.res4(x)
        
        x = self.conv5(x)
        
        # Global pooling
        x = self.pool(x).flatten(1)
        
        # VAE parameters
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class Decoder3D(nn.Module):
    """
    Decoder for 3D generation
    Generates 3D representation from latent code
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        output_points: int = 2048,
        output_features: int = 3  # xyz coordinates
    ):
        super().__init__()
        
        self.output_points = output_points
        self.output_features = output_features
        
        # MLP for latent expansion
        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Output projection to 3D points
        self.fc_points = nn.Linear(4096, output_points * output_features)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) latent code
        Returns:
            points: (B, output_points, output_features) 3D points
        """
        x = self.fc1(z)
        x = self.fc2(x)
        x = self.fc3(x)
        
        # Generate points
        points = self.fc_points(x)
        points = points.view(-1, self.output_points, self.output_features)
        
        # Tanh to normalize to [-1, 1]
        points = torch.tanh(points)
        
        return points


class HybridDecoder(nn.Module):
    """
    Hybrid decoder that generates both voxels and points
    For richer 3D representation
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        output_points: int = 2048,
        voxel_resolution: int = 32
    ):
        super().__init__()
        
        self.output_points = output_points
        self.voxel_resolution = voxel_resolution
        
        # Point cloud branch
        self.point_decoder = Decoder3D(
            latent_dim=latent_dim,
            output_points=output_points,
            output_features=3
        )
        
        # Voxel branch (for volumetric representation)
        self.voxel_fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 4096),  # 64 * 4 * 4 * 4 for reshape
            nn.ReLU(inplace=True)
        )
        
        # 3D deconvolution for voxels
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )  # 8 -> 16
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )  # 16 -> 32
        
        self.deconv3 = nn.Conv3d(16, 1, 3, padding=1)  # Output occupancy
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, latent_dim)
        Returns:
            points: (B, N, 3)
            voxels: (B, 1, H, W, D)
        """
        # Point cloud
        points = self.point_decoder(z)
        
        # Voxels
        v = self.voxel_fc(z)
        v = v.view(-1, 64, 4, 4, 4)  # Reshape to 3D (channels, depth, height, width)
        v = self.deconv1(v)
        v = self.deconv2(v)
        voxels = torch.sigmoid(self.deconv3(v))
        
        return points, voxels


class CAD_VAE(nn.Module):
    """Complete Variational Autoencoder for CAD 2D -> 3D (+ optional 2D reconstruction)

    Extended Features:
    - Encode 2D drawings to latent space
    - Optional 2D image reconstruction (for tighter latent supervision)
    - Decode latent to 3D (points + voxels)
    - KL divergence for regularization
    - Smooth latent space for interpolation
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 512,
        output_points: int = 2048,
        voxel_resolution: int = 32,
        use_hybrid_decoder: bool = True,
        enable_image_recon: bool = False
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.use_hybrid_decoder = use_hybrid_decoder
        
        # Encoder
        self.encoder = Encoder2D(
            input_channels=input_channels,
            latent_dim=latent_dim
        )
        
        # 3D Decoder
        if use_hybrid_decoder:
            self.decoder = HybridDecoder(
                latent_dim=latent_dim,
                output_points=output_points,
                voxel_resolution=voxel_resolution
            )
        else:
            self.decoder = Decoder3D(
                latent_dim=latent_dim,
                output_points=output_points
            )

        # 2D Reconstruction Decoder (optional)
        self.enable_image_recon = enable_image_recon
        if enable_image_recon:
            # Simple upsampling decoder from latent vector
            self.img_fc = nn.Sequential(
                nn.Linear(latent_dim, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True)
            )
            # Start with 512 -> 256 channels for spatial seed 16x16 then upscale
            self.img_seed = nn.Linear(512, 256 * 16 * 16)
            self.img_dec = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 32x32
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 64x64
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 128x128
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, input_channels, 4, stride=2, padding=1),  # 256x256
                nn.Tanh()  # output in [-1,1]
            )
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE
        z = mu + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Full forward pass
        
        Args:
            x: (B, 3, H, W) input image
        Returns:
            dict with reconstructions and parameters
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon_img = None
        if self.enable_image_recon:
            h = self.img_fc(z)
            seed = self.img_seed(h).view(-1, 256, 16, 16)
            recon_img = self.img_dec(seed)  # (B, C, 256, 256) range [-1,1]

        if self.use_hybrid_decoder:
            points, voxels = self.decoder(z)
            out = {
                'points': points,
                'voxels': voxels,
                'mu': mu,
                'logvar': logvar,
                'z': z
            }
        else:
            points = self.decoder(z)
            out = {
                'points': points,
                'mu': mu,
                'logvar': logvar,
                'z': z
            }
        if recon_img is not None:
            out['image_recon'] = recon_img
        return out
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent code (deterministic)"""
        mu, logvar = self.encoder(x)
        return mu
    
    def decode(self, z: torch.Tensor):
        """Decode latent code to 3D"""
        return self.decoder(z)
    
    def generate(self, num_samples: int, device: str = "cpu"):
        """Generate random 3D samples"""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
    
    def interpolate(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        steps: int = 10
    ) -> List[torch.Tensor]:
        """
        Interpolate between two images in latent space
        """
        z1 = self.encode(x1)
        z2 = self.encode(x2)
        
        results = []
        for alpha in np.linspace(0, 1, steps):
            z_interp = (1 - alpha) * z1 + alpha * z2
            output = self.decode(z_interp)
            results.append(output)
        
        return results


class VAELoss(nn.Module):
    """Complete loss for VAE training (3D + optional 2D image reconstruction)

    Total = point_loss * point_w + voxel_loss * voxel_w + smooth_loss * smooth_w + kl_loss * kl_w + image_loss * image_w
    """
    
    def __init__(
        self,
        kl_weight: float = 0.001,
        point_weight: float = 1.0,
        voxel_weight: float = 0.5,
        smoothness_weight: float = 0.1,
        image_weight: float = 0.0
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.point_weight = point_weight
        self.voxel_weight = voxel_weight
        self.smoothness_weight = smoothness_weight
        self.image_weight = image_weight
    
    def chamfer_distance(
        self,
        pred_points: torch.Tensor,
        gt_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Chamfer distance for point cloud comparison
        """
        # pred: (B, N1, 3), gt: (B, N2, 3)
        
        # Pairwise distances
        diff = pred_points[:, :, None, :] - gt_points[:, None, :, :]
        dist = torch.sum(diff ** 2, dim=-1)  # (B, N1, N2)
        
        # Forward: for each pred point, find nearest gt point
        min_dist_forward = torch.min(dist, dim=2)[0]  # (B, N1)
        loss_forward = torch.mean(min_dist_forward)
        
        # Backward: for each gt point, find nearest pred point
        min_dist_backward = torch.min(dist, dim=1)[0]  # (B, N2)
        loss_backward = torch.mean(min_dist_backward)
        
        return loss_forward + loss_backward
    
    def kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """KL divergence: KL(q(z|x) || p(z))"""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    def smoothness_loss(self, points: torch.Tensor) -> torch.Tensor:
        """
        Encourage smooth surfaces
        Penalize large differences between neighboring points
        """
        # Sort points by x coordinate
        sorted_points, _ = torch.sort(points, dim=1)
        
        # Compute differences between consecutive points
        diff = sorted_points[:, 1:, :] - sorted_points[:, :-1, :]
        
        # L2 norm of differences
        return torch.mean(torch.sum(diff ** 2, dim=-1))
    
    def forward(
        self,
        outputs: dict,
        target_points: torch.Tensor,
        target_voxels: Optional[torch.Tensor] = None,
        target_image: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Calculate total loss
        
        Args:
            outputs: model outputs (points, voxels, mu, logvar)
            target_points: ground truth points
            target_voxels: ground truth voxels (optional)
        """
        # Point cloud reconstruction loss
        point_loss = self.chamfer_distance(outputs['points'], target_points)
        
        # KL divergence
        kl_loss = self.kl_divergence(outputs['mu'], outputs['logvar'])
        
        # Voxel loss (if available)
        voxel_loss = 0.0
        if 'voxels' in outputs and target_voxels is not None:
            voxel_loss = F.binary_cross_entropy(
                outputs['voxels'],
                target_voxels,
                reduction='mean'
            )
        
        # Smoothness regularization
        smooth_loss = self.smoothness_loss(outputs['points'])

        # Image reconstruction loss (L1)
        image_loss = 0.0
        if self.image_weight > 0 and 'image_recon' in outputs and target_image is not None:
            # target_image expected in [-1,1]
            image_loss = torch.mean(torch.abs(outputs['image_recon'] - target_image))
        
        # Total loss
        total_loss = (
            self.point_weight * point_loss +
            self.kl_weight * kl_loss +
            self.voxel_weight * voxel_loss +
            self.smoothness_weight * smooth_loss +
            self.image_weight * image_loss
        )
        
        return {
            'total': total_loss,
            'point': point_loss,
            'kl': kl_loss,
            'voxel': voxel_loss,
            'smooth': smooth_loss,
            'image': image_loss
        }


def create_cad_vae(
    latent_dim: int = 512,
    output_points: int = 2048,
    device: str = "cpu",
    enable_image_recon: bool = False
) -> CAD_VAE:
    """Factory function to create VAE model"""
    
    model = CAD_VAE(
        input_channels=3,
        latent_dim=latent_dim,
        output_points=output_points,
        voxel_resolution=32,
        use_hybrid_decoder=True,
        enable_image_recon=enable_image_recon
    ).to(device)
    
    return model


if __name__ == "__main__":
    print("CAD VAE - Variational Autoencoder for 3D")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create model
    model = create_cad_vae(latent_dim=512, output_points=2048, device=device)
    
    # Test forward pass
    print("\nTesting model...")
    batch_size = 2
    dummy_image = torch.randn(batch_size, 3, 256, 256, device=device)
    
    with torch.no_grad():
        outputs = model(dummy_image)
    
    print(f"\nOutputs:")
    print(f"  Points: {outputs['points'].shape}")
    if 'voxels' in outputs:
        print(f"  Voxels: {outputs['voxels'].shape}")
    print(f"  Latent mu: {outputs['mu'].shape}")
    print(f"  Latent logvar: {outputs['logvar'].shape}")
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    print("\n✅ VAE model ready!")
