"""
Deep Hybrid Converter: A Fusion of ViT, VAE, and Diffusion Models.

This module introduces the `DeepHybridConverter`, a sophisticated model that
leverages the strengths of three distinct neural architectures to generate
3D point clouds from 2D images.

The fusion strategy is as follows:
1.  **Vision Transformer (ViT) Encoding**: A `VisionTransformerCAD` model processes
    the input image to extract a rich tensor of semantic and structural features.
    These features are pooled to form a global context vector.
2.  **Variational Autoencoder (VAE) Encoding**: A `CAD_VAE` model encodes the
    same image into a latent space, providing a compressed representation (mu).
    The VAE's decoder output is also used as a structural prior for the diffusion process.
3.  **Diffusion Model Conditioning**: The diffusion model's own image encoder
    provides an additional conditional input.
4.  **Feature Fusion**: The outputs from the ViT, VAE (mu), and diffusion encoder
    are passed through adapter layers to align their distributions, then
    concatenated and projected into a single conditioning vector for the
    diffusion process.
5.  **Prior-Informed Denoising**: The diffusion process starts with a blend of
    the VAE's decoded point cloud and Gaussian noise. The `prior_strength`
    parameter controls this blend.
6.  **DDIM Sampling**: A custom DDIM-like sampling loop denoises the initial
    state over a series of timesteps, guided by the fused conditioning vector,
    to generate the final high-fidelity point cloud.
7.  **Output Normalization**: The final point cloud is normalized to a specified
    CAD unit range (e.g., millimeters).

Key Features:
- `normalize_range`: A tuple `(min_mm, max_mm)` to scale the output from the
  model's native [-1, 1] range to a practical unit range.
- `prior_strength`: Controls the influence of the VAE's output on the initial
  state of the diffusion process.
- `ddim_steps`: The number of steps for the deterministic diffusion sampling,
  allowing a trade-off between speed and quality.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import numpy as np
import cv2
import ezdxf
from ezdxf.document import Drawing

from .vision_transformer_cad import VisionTransformerCAD, VisionTransformerConfig
from .vae_model import create_cad_vae
from .diffusion_3d_model import create_diffusion_model, Diffusion3DConverter


class DeepHybridConverter:
    """
    Orchestrates the ViT, VAE, and Diffusion models to convert images to 3D.
    """
    def __init__(
        self,
        device: str = 'cpu',
        enable_image_recon: bool = True,
        prior_strength: float = 0.5,
        ddim_steps: int = 40,
        normalize_range: Tuple[float, float] = (0.0, 1000.0)
    ):
        """
        Initializes the DeepHybridConverter and its component models.

        Args:
            device: The torch device to run the models on ('cpu' or 'cuda').
            enable_image_recon: Whether the VAE should reconstruct the input image
                (useful for regularization and debugging).
            prior_strength: The weight (0.0 to 1.0) of the VAE's decoded points
                in the initial diffusion state.
            ddim_steps: The number of steps for the DDIM sampling process.
            normalize_range: The target minimum and maximum coordinates for the
                output point cloud in millimeters.
        """
        self.device = device
        self.prior_strength = prior_strength
        self.ddim_steps = int(ddim_steps)
        self.normalize_range = normalize_range

        # --- Component Models ---
        vit_config = VisionTransformerConfig(
            image_size=256, patch_size=16, num_classes=50, dim=768,
            depth=8, heads=12, mlp_dim=2048, dropout=0.1,
            predict_depth=False, predict_height=False, predict_material=True
        )
        self.vit = VisionTransformerCAD(config=vit_config).to(device)

        self.vae = create_cad_vae(
            latent_dim=512, output_points=2048, device=device,
            enable_image_recon=enable_image_recon
        )
        self.diffusion: Diffusion3DConverter = create_diffusion_model(
            num_points=2048, timesteps=1000, device=device
        )

        # --- Feature Adaptation and Fusion Layers ---
        # These layers align the feature distributions from different models.
        self.adapt_mu = nn.Sequential(nn.Linear(512, 512), nn.LayerNorm(512)).to(device)
        self.adapt_diff = nn.Sequential(nn.Linear(512, 512), nn.LayerNorm(512)).to(device)
        self.adapt_vit = nn.Sequential(nn.Linear(512, 512), nn.LayerNorm(512)).to(device)

        # This layer fuses the adapted features into a single conditioning vector.
        self.fusion = nn.Sequential(
            nn.Linear(512 * 3, 512),
            nn.GELU(),
            nn.Linear(512, 512)
        ).to(device)

    def _prepare_image(self, image_path: Path) -> torch.Tensor:
        """Loads, resizes, and normalizes an image for model input."""
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        # Normalize to [-1, 1] range
        img_norm = (img.astype('float32') / 127.5) - 1.0
        tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

    def _encode_and_fuse_features(self, img_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Extracts features from all models and fuses them into a condition tensor."""
        # 1. ViT semantic feature extraction
        vit_out = self.vit(img_tensor)
        vit_global = vit_out['semantic'].mean(dim=1)  # Pool over patches
        if not hasattr(self, 'vit_proj'):
            self.vit_proj = nn.Linear(vit_global.size(-1), 512).to(self.device)
        vit_rep = self.adapt_vit(self.vit_proj(vit_global))

        # 2. VAE latent mean (mu) and decoded points (prior)
        vae_out = self.vae(img_tensor)
        mu = self.adapt_mu(vae_out['mu'])
        prior_points = vae_out['points']

        # 3. Diffusion model's image encoder features
        diff_feat_raw = self.diffusion.encode_image(img_tensor)
        diff_feat = self.adapt_diff(diff_feat_raw)

        # 4. Fusion
        fused = torch.cat([mu, diff_feat, vit_rep], dim=-1)
        condition = self.fusion(fused)

        return condition, prior_points, vae_out.get('image_recon')

    def _run_ddim_sampling(self, condition: torch.Tensor, init_state: torch.Tensor) -> np.ndarray:
        """Performs the DDIM sampling process to generate the point cloud."""
        if self.ddim_steps <= 0:
            return init_state[0].cpu().numpy()

        timesteps_full = self.diffusion.diffusion.timesteps
        step_size = max(1, timesteps_full // self.ddim_steps)
        schedule = list(range(0, timesteps_full, step_size))[::-1]
        
        x = init_state
        for i, t in enumerate(schedule):
            t_tensor = torch.full((x.size(0),), t, device=self.device, dtype=torch.long)
            predicted_noise = self.diffusion.unet(x, t_tensor, condition)
            
            alpha_t = self.diffusion.diffusion.alphas_cumprod[t]
            alpha_t_prev = self.diffusion.diffusion.alphas_cumprod[schedule[i+1]] if i < len(schedule)-1 else torch.tensor(1.0, device=self.device)
            
            # Predict x0 (the original image) from the noise
            x0_pred = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_t_prev) * predicted_noise
            
            # Update x to the previous timestep (deterministic)
            x = torch.sqrt(alpha_t_prev) * x0_pred + dir_xt
            
        return x[0].cpu().numpy()

    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        """Normalizes the point cloud to the target CAD unit range."""
        min_val, max_val = self.normalize_range
        
        # Normalize points from their current range to [-1, 1]
        p_min, p_max = points.min(), points.max()
        p_range = p_max - p_min
        if p_range > 1e-6:
            points_norm = 2.0 * (points - p_min) / p_range - 1.0
        else:
            points_norm = points
        
        # Scale from [-1, 1] to [min_val, max_val]
        points_scaled = ((points_norm * 0.5) + 0.5) * (max_val - min_val) + min_val
        return points_scaled

    def _save_dxf_output(self, points: np.ndarray, output_path: Path) -> None:
        """Saves the generated point cloud as a DXF file with small cubes."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc: Drawing = ezdxf.new(setup=True)
        msp = doc.modelspace()
        
        min_mm, max_mm = self.normalize_range
        cube_size = max(1.0, (max_mm - min_mm) * 0.008)  # Adaptive cube size
        
        # Subsample points to avoid creating an excessively large file
        num_points_to_render = min(600, points.shape[0])
        step = max(1, points.shape[0] // num_points_to_render)

        for i in range(0, points.shape[0], step):
            px, py, pz = points[i]
            verts = [
                (px, py, pz), (px + cube_size, py, pz), (px + cube_size, py + cube_size, pz), (px, py + cube_size, pz),
                (px, py, pz + cube_size), (px + cube_size, py, pz + cube_size), (px + cube_size, py + cube_size, pz + cube_size), (px, py + cube_size, pz + cube_size)
            ]
            mesh = msp.add_mesh()
            with mesh.edit_data() as md:
                md.vertices = verts
                md.faces = [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6], [0,3,7,4], [1,2,6,5]]
        doc.saveas(str(output_path))

    @torch.no_grad()
    def convert(self, image_path: Path, output_dxf: Path) -> Dict[str, Any]:
        """
        Main conversion method to process an image and generate a 3D DXF file.

        Args:
            image_path: Path to the input 2D image file.
            output_dxf: Path where the output 3D DXF file will be saved.

        Returns:
            A dictionary containing metadata about the conversion process.
        """
        img_tensor = self._prepare_image(image_path)

        # 1. Encode inputs and fuse features
        condition, prior_points, recon_img = self._encode_and_fuse_features(img_tensor)

        # 2. Create initial state for diffusion
        noise = torch.randn_like(prior_points)
        init_state = self.prior_strength * prior_points + (1 - self.prior_strength) * noise

        # 3. Run DDIM sampling
        final_points_raw = self._run_ddim_sampling(condition, init_state)

        # 4. Normalize points to the target CAD range
        final_points_mm = self._normalize_points(final_points_raw)

        # 5. Save the 3D model as a DXF file
        self._save_dxf_output(final_points_mm, output_dxf)

        # 6. Save the VAE reconstruction image if available
        recon_path = None
        if recon_img is not None:
            recon_img_np = (recon_img[0].cpu().permute(1, 2, 0).numpy() * 127.5 + 127.5).clip(0, 255).astype('uint8')
            recon_path = output_dxf.with_name(f"{output_dxf.stem}_recon.png")
            cv2.imwrite(str(recon_path), cv2.cvtColor(recon_img_np, cv2.COLOR_RGB2BGR))

        return {
            'dxf_path': str(output_dxf),
            'point_count': final_points_mm.shape[0],
            'prior_strength': self.prior_strength,
            'ddim_steps': self.ddim_steps,
            'normalize_range': self.normalize_range,
            'reconstruction_image_path': str(recon_path) if recon_path else None,
        }


def create_deep_hybrid_converter(
    device: str = 'cpu',
    prior_strength: float = 0.5,
    normalize_range: Tuple[float, float] = (0.0, 1000.0),
    ddim_steps: int = 40,
) -> DeepHybridConverter:
    """
    Factory function to create an instance of the DeepHybridConverter.

    Args:
        device: The torch device ('cpu' or 'cuda').
        prior_strength: The influence of the VAE prior (0.0 to 1.0).
        normalize_range: The target output coordinate range in millimeters.
        ddim_steps: The number of sampling steps.

    Returns:
        An initialized DeepHybridConverter instance.
    """
    return DeepHybridConverter(
        device=device,
        prior_strength=prior_strength,
        normalize_range=normalize_range,
        ddim_steps=ddim_steps
    )


if __name__ == '__main__':
    print("--- Deep Hybrid Converter Demonstration ---")
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {dev}")

    try:
        # 1. Create the converter
        converter = create_deep_hybrid_converter(dev, ddim_steps=20, prior_strength=0.6)

        # 2. Create a synthetic test image
        output_dir = Path('demo_output/deep_hybrid')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        img = np.ones((256, 256, 3), dtype=np.uint8) * 240
        cv2.putText(img, 'HYBRID', (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 2, (20, 20, 20), 4)
        
        test_image_path = output_dir / 'test_image.png'
        cv2.imwrite(str(test_image_path), img)
        print(f"Created test image at: {test_image_path}")

        # 3. Run the conversion
        output_dxf_path = output_dir / 'hybrid_output_3d.dxf'
        results = converter.convert(test_image_path, output_dxf_path)

        # 4. Print results
        print("\n--- Conversion Complete ---")
        for key, value in results.items():
            print(f"  - {key}: {value}")
        print("✅ Demonstration finished successfully.")

    except ImportError as e:
        print(f"\nError: A required library is missing. {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")