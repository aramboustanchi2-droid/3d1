"""Bridge utilities between CAD_VAE and Diffusion3DConverter.

Purpose:
    Provide helper functions to use a trained CAD_VAE as a prior / conditioner
    for the diffusion model to accelerate convergence and improve structure.

Strategies Implemented:
1. Latent Conditioning Fusion: concatenate VAE latent (mu) with image encoder features,
   projecting back to expected dimension.
2. Prior Initialization: start diffusion sampling from a partially noised VAE point cloud
   instead of pure Gaussian noise.

Usage:
    from cad3d.vae_model import create_cad_vae
    from cad3d.diffusion_3d_model import create_diffusion_model
    from cad3d.vae_diffusion_bridge import DiffusionVAEBridge

    vae = create_cad_vae(device='cuda')
    diff = create_diffusion_model(device='cuda')
    bridge = DiffusionVAEBridge(vae, diff, device='cuda')
    img = torch.randn(1,3,256,256).to('cuda')
    pts = bridge.generate_with_vae_prior(img, steps=30)  # returns (1, N, 3)
"""

from typing import Optional
import torch

from .vae_model import CAD_VAE
from .diffusion_3d_model import Diffusion3DConverter


class DiffusionVAEBridge:
    def __init__(self, vae: CAD_VAE, diffusion: Diffusion3DConverter, device: str = 'cpu'):
        self.vae = vae
        self.diffusion = diffusion
        self.device = device
        self.latent_fusion = torch.nn.Sequential(
            torch.nn.Linear(self.vae.latent_dim + 512, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 512)
        ).to(device)

    def fuse_condition(self, image: torch.Tensor) -> torch.Tensor:
        """Fuse VAE latent mean with diffusion image encoder features."""
        with torch.no_grad():
            mu, _ = self.vae.encoder(image)
            img_feat = self.diffusion.encode_image(image)
        fused = torch.cat([mu, img_feat], dim=-1)
        return self.latent_fusion(fused)

    @torch.no_grad()
    def generate_with_vae_prior(
        self,
        image: torch.Tensor,
        sampling_method: str = 'ddim',
        steps: int = 50,
        prior_strength: float = 0.5
    ) -> torch.Tensor:
        """Generate 3D point cloud using VAE prior initialization.

        Args:
            image: (B,3,H,W)
            sampling_method: 'ddim' or 'ddpm'
            steps: for DDIM
            prior_strength: blend between VAE decoded points and Gaussian noise.
        Returns:
            (B,N,3) point cloud
        """
        B = image.shape[0]
        condition = self.fuse_condition(image)
        # Obtain prior point cloud from VAE decode (using mu deterministic)
        with torch.no_grad():
            mu, _ = self.vae.encoder(image)
            prior_points = self.vae.decode(mu)[0] if self.vae.use_hybrid_decoder else self.vae.decode(mu)
            # prior_points: (B,N,3)

        # Initialize diffusion starting state: blend prior with noise
        shape = (B, self.diffusion.num_points, 3)
        noise = torch.randn(shape, device=self.device)
        init = prior_strength * prior_points + (1 - prior_strength) * noise

        # Custom sampling loop similar to DDIM but starting from init
        if sampling_method == 'ddim':
            steps = max(5, steps)
            step_size = self.diffusion.diffusion.timesteps // steps
            timesteps = list(range(0, self.diffusion.diffusion.timesteps, step_size))
            timesteps.reverse()
            x = init
            for i, t in enumerate(timesteps):
                t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)
                predicted_noise = self.diffusion.unet(x, t_tensor, condition)
                alpha_t = self.diffusion.diffusion.alphas_cumprod[t]
                alpha_t_prev = self.diffusion.diffusion.alphas_cumprod[timesteps[i+1]] if i < len(timesteps) - 1 else torch.tensor(1.0, device=self.device)
                x0_pred = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
                dir_xt = torch.sqrt(1 - alpha_t_prev) * predicted_noise
                noise = torch.zeros_like(x) if i == len(timesteps) - 1 else torch.randn_like(x)
                x = torch.sqrt(alpha_t_prev) * x0_pred + dir_xt + noise * 0.0  # deterministic
            return x
        else:  # ddpm
            x = init
            for i in reversed(range(self.diffusion.diffusion.timesteps)):
                t = torch.full((B,), i, device=self.device, dtype=torch.long)
                betas_t = self.diffusion.diffusion.extract(self.diffusion.diffusion.betas, t, x.shape)
                sqrt_one_minus_alphas_cumprod_t = self.diffusion.diffusion.extract(self.diffusion.diffusion.sqrt_one_minus_alphas_cumprod, t, x.shape)
                sqrt_recip_alphas_t = self.diffusion.diffusion.extract(self.diffusion.diffusion.sqrt_recip_alphas, t, x.shape)
                model_output = self.diffusion.unet(x, t, condition)
                model_mean = sqrt_recip_alphas_t * (x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t)
                if i == 0:
                    x = model_mean
                else:
                    posterior_variance_t = self.diffusion.diffusion.extract(self.diffusion.diffusion.posterior_variance, t, x.shape)
                    x = model_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(x)
            return x


def create_bridge(vae: CAD_VAE, diffusion: Diffusion3DConverter, device: Optional[str] = None) -> DiffusionVAEBridge:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return DiffusionVAEBridge(vae, diffusion, device=device)


if __name__ == "__main__":
    from .vae_model import create_cad_vae
    from .diffusion_3d_model import create_diffusion_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vae = create_cad_vae(device=dev)
    diff = create_diffusion_model(device=dev)
    bridge = DiffusionVAEBridge(vae, diff, device=dev)
    img = torch.randn(1, 3, 256, 256, device=dev)
    pts = bridge.generate_with_vae_prior(img, steps=10)
    print("Bridged sample:", pts.shape, pts.min().item(), pts.max().item())