"""
Autoencoders / VAEs for CAD-AI
Lightweight 2D->3D feature learning and reconstruction

- Image encoder -> latent z (mu, logvar)
- Decoders:
  * Image reconstruction (3x256x256)
  * Depth/height map (1x64x64)
  * Point cloud generation (N x 3)
- Losses: L1 image, L1 depth, Chamfer distance (point cloud), KL (beta-VAE)

Designed to be fast and suitable for CAD workflows.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def chamfer_distance(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """Compute symmetric Chamfer Distance between two point clouds.
    p1: (B, N, 3), p2: (B, M, 3)
    returns: scalar (mean over batch)
    """
    B, N, _ = p1.shape
    _, M, _ = p2.shape
    # (B, N, M)
    diff = p1.unsqueeze(2) - p2.unsqueeze(1)
    dist = torch.sum(diff * diff, dim=-1)
    # (B, N)
    min_p1, _ = torch.min(dist, dim=2)
    # (B, M)
    min_p2, _ = torch.min(dist, dim=1)
    return (min_p1.mean(dim=1) + min_p2.mean(dim=1)).mean()


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 256):
        super().__init__()
        ch = [64, 128, 256, 512]
        self.conv1 = nn.Conv2d(in_channels, ch[0], 4, 2, 1)
        self.conv2 = nn.Conv2d(ch[0], ch[1], 4, 2, 1)
        self.conv3 = nn.Conv2d(ch[1], ch[2], 4, 2, 1)
        self.conv4 = nn.Conv2d(ch[2], ch[3], 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ch[0])
        self.bn2 = nn.BatchNorm2d(ch[1])
        self.bn3 = nn.BatchNorm2d(ch[2])
        self.bn4 = nn.BatchNorm2d(ch[3])
        # input 256 -> 128 -> 64 -> 32 -> 16
        self.fc_mu = nn.Linear(ch[3] * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(ch[3] * 16 * 16, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        h = x.flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = reparameterize(mu, logvar)
        return z, mu, logvar


class ImageDecoder(nn.Module):
    def __init__(self, latent_dim: int = 256, out_channels: int = 3):
        super().__init__()
        ch = [512, 256, 128, 64]
        self.fc = nn.Linear(latent_dim, ch[0] * 16 * 16)
        self.deconv1 = nn.ConvTranspose2d(ch[0], ch[1], 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(ch[1], ch[2], 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(ch[2], ch[3], 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(ch[3], out_channels, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ch[1])
        self.bn2 = nn.BatchNorm2d(ch[2])
        self.bn3 = nn.BatchNorm2d(ch[3])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 512, 16, 16)
        x = F.relu(self.bn1(self.deconv1(x)))  # 32
        x = F.relu(self.bn2(self.deconv2(x)))  # 64
        x = F.relu(self.bn3(self.deconv3(x)))  # 128
        x = torch.tanh(self.deconv4(x))        # 256, range [-1,1]
        return x


class DepthDecoder(nn.Module):
    def __init__(self, latent_dim: int = 256, out_size: int = 64):
        super().__init__()
        ch = [256, 128, 64]
        self.fc = nn.Linear(latent_dim, ch[0] * 8 * 8)
        self.deconv1 = nn.ConvTranspose2d(ch[0], ch[1], 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(ch[1], ch[2], 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(ch[2], 1, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ch[1])
        self.bn2 = nn.BatchNorm2d(ch[2])
        self.out_size = out_size

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 256, 8, 8)
        x = F.relu(self.bn1(self.deconv1(x)))  # 16
        x = F.relu(self.bn2(self.deconv2(x)))  # 32
        x = self.deconv3(x)                    # 64
        return x  # not activated, regression


class PointCloudDecoder(nn.Module):
    def __init__(self, latent_dim: int = 256, num_points: int = 2048):
        super().__init__()
        self.num_points = num_points
        hidden = 512
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_points * 3)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.mlp(z)
        x = x.view(-1, self.num_points, 3)
        return x


class ImageVAE2Dto3D(nn.Module):
    def __init__(self, latent_dim: int = 256, num_points: int = 2048):
        super().__init__()
        self.encoder = ConvEncoder(in_channels=3, latent_dim=latent_dim)
        self.img_dec = ImageDecoder(latent_dim=latent_dim, out_channels=3)
        self.depth_dec = DepthDecoder(latent_dim=latent_dim, out_size=64)
        self.pc_dec = PointCloudDecoder(latent_dim=latent_dim, num_points=num_points)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.encoder(x)
        x_rec = self.img_dec(z)
        depth = self.depth_dec(z)
        pc = self.pc_dec(z)
        return x_rec, depth, pc, mu, logvar

    @staticmethod
    def loss_fn(
        x: torch.Tensor,
        x_rec: torch.Tensor,
        depth_gt: Optional[torch.Tensor],
        depth_pred: torch.Tensor,
        pc_gt: Optional[torch.Tensor],
        pc_pred: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        w_img: float = 1.0,
        w_depth: float = 0.5,
        w_pc: float = 1.0,
        beta: float = 0.001,
    ) -> Tuple[torch.Tensor, dict]:
        # Image reconstruction L1
        loss_img = F.l1_loss(x_rec, x)
        # Depth loss if provided
        if depth_gt is not None:
            loss_depth = F.l1_loss(depth_pred, depth_gt)
        else:
            loss_depth = torch.tensor(0.0, device=x.device)
        # Chamfer distance if pc provided
        if pc_gt is not None:
            loss_pc = chamfer_distance(pc_pred, pc_gt)
        else:
            # Encourage small coordinates if no GT
            loss_pc = (pc_pred ** 2).mean()
        # KL divergence
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
        total = w_img * loss_img + w_depth * loss_depth + w_pc * loss_pc + beta * kl
        return total, {
            'loss_total': total.item(),
            'loss_img': loss_img.item(),
            'loss_depth': loss_depth.item() if isinstance(loss_depth, torch.Tensor) else 0.0,
            'loss_pc': loss_pc.item(),
            'kl': kl.item(),
        }


def create_vae_model(latent_dim: int = 256, num_points: int = 2048, device: Optional[str] = None) -> ImageVAE2Dto3D:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageVAE2Dto3D(latent_dim=latent_dim, num_points=num_points).to(device)
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_vae_model(device=device)
    x = torch.randn(2, 3, 256, 256, device=device)
    x_rec, depth, pc, mu, logvar = model(x)
    print("Image rec:", x_rec.shape, "Depth:", depth.shape, "PC:", pc.shape)
