"""
Hybrid Vision Transformer and Diffusion Model Integration.

This module defines the `HybridCAD3DConverter`, a state-of-the-art system
for 2D-to-3D conversion. It achieves this by integrating three key concepts:

1.  **Vision Transformer (ViT)**: A powerful deep learning model for understanding
    the semantic and structural content of a 2D image. It extracts rich feature
    vectors representing aspects like object types, materials, and depth cues.

2.  **3D Diffusion Model**: A generative model capable of producing highly
    detailed and coherent 3D point clouds from a conditioning input.

3.  **Continuous Learning**: An online learning mechanism that allows the system
    to improve its performance over time by learning from each conversion it
    performs, using an experience replay buffer.

The pipeline is as follows:
- The ViT analyzes the input image to produce a set of rich feature embeddings.
- These features are fused with the diffusion model's own image encoding to
  create a comprehensive conditioning vector.
- The diffusion model then generates a 3D point cloud guided by this vector.
- The point cloud is enhanced with semantic information (e.g., colors) from the ViT.
- The final 3D model is saved as a DXF file.
- The input image and output point cloud are stored in a replay buffer, and the
  diffusion model is periodically fine-tuned on this data.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import time
from typing import Optional, Tuple, Dict, Any
import cv2
import ezdxf
from ezdxf.document import Drawing

from .vision_transformer_cad import VisionTransformerCAD, VisionTransformerConfig
from .diffusion_3d_model import Diffusion3DConverter, create_diffusion_model
from .diffusion_trainer import DiffusionTrainer, ExperienceReplayBuffer


class HybridCAD3DConverter:
    """
    Orchestrates the ViT and Diffusion Model for 2D-to-3D conversion with
    continuous learning capabilities.
    """

    def __init__(
        self,
        device: str = "cpu",
        vit_model_path: Optional[Path] = None,
        diffusion_model_path: Optional[Path] = None,
        enable_learning: bool = True,
        num_points: int = 4096,
    ):
        """
        Initializes the hybrid converter system.

        Args:
            device: The torch device ('cpu' or 'cuda').
            vit_model_path: Optional path to a pre-trained Vision Transformer model.
            diffusion_model_path: Optional path to a pre-trained Diffusion model.
            enable_learning: If True, enables the continuous learning mechanism.
            num_points: The number of points in the generated 3D point cloud.
        """
        self.device = device
        self.enable_learning = enable_learning
        self.num_points = num_points
        self.conversion_count = 0
        self.learning_updates = 0

        print("Initializing Hybrid CAD 3D Converter...")
        print("=" * 70)

        # 1. Vision Transformer (for rich 2D understanding)
        print("Loading Vision Transformer...")
        vit_config = VisionTransformerConfig(
            image_size=256, patch_size=16, num_classes=50, dim=768,
            depth=12, heads=12, predict_depth=True, predict_height=True,
            predict_material=True
        )
        self.vit = VisionTransformerCAD(vit_config).to(device)

        if vit_model_path and vit_model_path.exists():
            try:
                checkpoint = torch.load(vit_model_path, map_location=device)
                self.vit.load_state_dict(checkpoint['model_state_dict'])
                print(f"  ✅ Loaded ViT from {vit_model_path}")
            except (KeyError, RuntimeError) as e:
                print(f"  ⚠️  Could not load ViT checkpoint: {e}. Using untrained model.")
        else:
            print("  ℹ️  Using untrained ViT (will improve with learning).")

        # 2. Diffusion Model (for detailed 3D generation)
        print("\nLoading Diffusion Model...")
        self.diffusion: Diffusion3DConverter = create_diffusion_model(
            num_points=self.num_points, timesteps=1000, device=device
        )

        if diffusion_model_path and diffusion_model_path.exists():
            try:
                checkpoint = torch.load(diffusion_model_path, map_location=device)
                self.diffusion.image_encoder.load_state_dict(checkpoint['image_encoder_state'])
                self.diffusion.unet.load_state_dict(checkpoint['unet_state'])
                print(f"  ✅ Loaded Diffusion model from {diffusion_model_path}")
            except (KeyError, RuntimeError) as e:
                print(f"  ⚠️  Could not load Diffusion checkpoint: {e}. Using untrained model.")
        else:
            print("  ℹ️  Using untrained Diffusion model (will improve with learning).")

        # 3. Feature Fusion Layer (ViT → Diffusion)
        print("\nCreating feature fusion layer...")
        self.feature_fusion = nn.Sequential(
            nn.Linear(vit_config.dim, 512),  # ViT embed_dim → Diffusion latent_dim
            nn.GELU(),
            nn.Linear(512, 512)
        ).to(device)

        # 4. Continuous Learning Components
        if self.enable_learning:
            print("\nSetting up continuous learning...")
            self.replay_buffer = ExperienceReplayBuffer(capacity=1000)
            self.trainer = DiffusionTrainer(
                model=self.diffusion,
                device=device,
                learning_rate=1e-5  # Lower LR for fine-tuning
            )
            print("  ✅ Experience replay buffer and trainer enabled.")

        print("\n✅ Hybrid Converter is ready!")
        print("=" * 70)

    def _prepare_image(self, image_path: Path) -> torch.Tensor:
        """Loads, resizes, and normalizes an image for model input."""
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img_normalized = (img.astype(np.float32) / 127.5) - 1.0
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        return img_tensor.to(self.device)

    @torch.no_grad()
    def _extract_vit_features(self, image_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extracts rich features using the Vision Transformer."""
        self.vit.eval()
        return self.vit(image_tensor)

    def _generate_3d_from_features(
        self,
        image_tensor: torch.Tensor,
        vit_features: Dict[str, torch.Tensor],
        sampling_steps: int
    ) -> torch.Tensor:
        """Generates a 3D point cloud using the diffusion model with ViT guidance."""
        # Encode image with the diffusion model's own encoder
        diffusion_native_features = self.diffusion.encode_image(image_tensor)

        # Extract a global feature vector from the ViT's semantic output
        # We pool across the spatial dimensions of the patch embeddings
        vit_global_feature = vit_features['semantic'].mean(dim=1)

        # Fuse the ViT feature into the diffusion conditioning space
        fused_vit_feature = self.feature_fusion(vit_global_feature)

        # Combine the diffusion's native encoding with the fused ViT feature
        condition = diffusion_native_features + fused_vit_feature

        # Generate the 3D point cloud using DDIM sampling
        point_cloud = self.diffusion.diffusion.ddim_sample(
            shape=(image_tensor.shape[0], self.num_points, 3),
            condition=condition,
            steps=sampling_steps,
            device=self.device
        )
        return point_cloud

    def _enhance_point_cloud_with_semantics(
        self,
        point_cloud: torch.Tensor,
        vit_features: Dict[str, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhances the point cloud with semantic colors and height information from the ViT.

        Returns:
            A tuple containing:
            - points (np.ndarray): (N, 3) array of points with Z-coordinates adjusted by height.
            - colors (np.ndarray): (N, 3) array of RGB colors based on semantics.
        """
        points_np = point_cloud[0].cpu().numpy()
        N = points_np.shape[0]

        # Extract feature maps from ViT output
        semantic_map = torch.argmax(vit_features['semantic'][0], dim=0).cpu().numpy()
        height_map = vit_features['height'][0].cpu().numpy()
        H, W = semantic_map.shape

        # Project 3D points back to 2D image coordinates to sample features
        # Assumes points are in the range [-1, 1]
        x_2d = np.clip(((points_np[:, 0] + 1) / 2 * (W - 1)), 0, W - 1).astype(int)
        y_2d = np.clip(((points_np[:, 1] + 1) / 2 * (H - 1)), 0, H - 1).astype(int)

        # Sample semantic class and height for each point
        point_semantic_classes = semantic_map[y_2d, x_2d]
        point_height_values = height_map[y_2d, x_2d]

        # Define a color palette for semantic classes
        color_palette = np.array([
            [200, 200, 200], [255, 0, 0], [0, 255, 0], [0, 0, 255],
            [255, 255, 0], [255, 0, 255], [0, 255, 255], [128, 0, 0]
        ])
        num_colors = len(color_palette)
        colors = color_palette[point_semantic_classes % num_colors]

        # Adjust the Z-coordinate of points based on predicted height
        # The scaling factor can be tuned
        points_np[:, 2] = point_height_values * 2.0

        return points_np, colors.astype(np.uint8)

    def _point_cloud_to_dxf(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        output_path: Path,
        scale_factor: float = 1000.0
    ) -> None:
        """Converts the colored point cloud to a DXF file containing a mesh of cubes."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc: Drawing = ezdxf.new(setup=True)
        msp = doc.modelspace()

        points_scaled = points * scale_factor
        cube_size = 10.0  # Size of cubes in millimeters

        # Subsample points to keep the DXF file size manageable
        num_points_to_render = min(1000, points_scaled.shape[0])
        step = max(1, points_scaled.shape[0] // num_points_to_render)

        for i in range(0, points_scaled.shape[0], step):
            pt, col = points_scaled[i], colors[i]
            x, y, z = pt
            
            vertices = [
                (x, y, z), (x + cube_size, y, z), (x + cube_size, y + cube_size, z), (x, y + cube_size, z),
                (x, y, z + cube_size), (x + cube_size, y, z + cube_size),
                (x + cube_size, y + cube_size, z + cube_size), (x, y + cube_size, z + cube_size)
            ]
            
            mesh = msp.add_mesh()
            with mesh.edit_data() as md:
                md.vertices = vertices
                md.faces = [
                    [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                    [1, 2, 6, 5], [2, 3, 7, 6], [0, 3, 7, 4]
                ]
            mesh.dxf.true_color = ezdxf.colors.rgb2int(tuple(col))
        
        doc.saveas(output_path)
        print(f"   ✅ Saved DXF with {num_points_to_render} cubes to {output_path}")

    def convert(
        self,
        image_path: Path,
        output_path: Path,
        sampling_steps: int = 50,
        learn_from_this: bool = True
    ) -> Dict[str, Any]:
        """
        Executes the complete 2D image to 3D DXF conversion pipeline.

        Args:
            image_path: Path to the input image file.
            output_path: Path for the output DXF file.
            sampling_steps: Number of DDIM steps for diffusion (lower is faster).
            learn_from_this: If True, adds the result to the experience replay buffer.

        Returns:
            A dictionary containing statistics and paths related to the conversion.
        """
        print(f"\n{'='*70}\nConverting: {image_path.name} → {output_path.name}\n{'='*70}")
        start_time = time.time()

        # 1. Load and preprocess image
        print("1. Loading and preparing image...")
        image_tensor = self._prepare_image(image_path)

        # 2. Extract features with Vision Transformer
        print("2. Extracting features with Vision Transformer...")
        vit_features = self._extract_vit_features(image_tensor)
        print("   ✅ Extracted semantic, height, and material features.")

        # 3. Generate 3D point cloud
        print(f"3. Generating 3D point cloud (DDIM {sampling_steps} steps)...")
        point_cloud = self._generate_3d_from_features(image_tensor, vit_features, sampling_steps)
        print(f"   ✅ Generated {point_cloud.shape[1]} points.")

        # 4. Enhance point cloud with semantics
        print("4. Enhancing point cloud with semantic information...")
        enhanced_points, colors = self._enhance_point_cloud_with_semantics(point_cloud, vit_features)
        print("   ✅ Applied semantic colors and height adjustments.")

        # 5. Convert to DXF
        print("5. Converting to DXF mesh...")
        self._point_cloud_to_dxf(enhanced_points, colors, output_path)

        # 6. Continuous Learning
        if self.enable_learning and learn_from_this:
            print("6. Storing experience for continuous learning...")
            self.replay_buffer.add(image_tensor, point_cloud.detach())
            
            # Perform a learning update periodically (e.g., every 5 conversions)
            if len(self.replay_buffer) >= 10 and self.conversion_count > 0 and self.conversion_count % 5 == 0:
                print("   🎓 Performing learning update from experience...")
                self.perform_learning_update()

        elapsed = time.time() - start_time
        self.conversion_count += 1
        
        print(f"\n{'='*70}\n✅ Conversion complete in {elapsed:.2f}s\n{'='*70}")
        
        return {
            'input_path': str(image_path),
            'output_path': str(output_path),
            'num_points': self.num_points,
            'conversion_time_seconds': elapsed,
            'total_conversions': self.conversion_count,
            'total_learning_updates': self.learning_updates
        }

    def perform_learning_update(self, num_updates: int = 5, batch_size: int = 4):
        """Performs a learning update by training on samples from the experience replay buffer."""
        if len(self.replay_buffer) < batch_size:
            print("   ⚠️  Not enough experiences in buffer to perform learning update.")
            return

        self.diffusion.unet.train()
        self.diffusion.image_encoder.train()

        total_loss = 0
        for i in range(num_updates):
            batch = self.replay_buffer.sample(batch_size)
            if batch is None:
                break
            
            images, point_clouds = batch
            loss = self.trainer.train_step(images, point_clouds)
            total_loss += loss
            print(f"      Update {i+1}/{num_updates}: Loss = {loss:.6f}")
        
        self.learning_updates += num_updates
        print(f"   ✅ Completed {num_updates} learning updates. Average loss: {total_loss / num_updates:.6f}")

        # Periodically save the continuously trained model
        if self.learning_updates > 0 and self.learning_updates % 20 == 0:
            self._save_continuous_model()

    def _save_continuous_model(self):
        """Saves the state of the continuously learning diffusion model."""
        save_dir = Path("trained_models/diffusion_continuous")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"model_update_{self.learning_updates}.pth"
        
        torch.save({
            'image_encoder_state': self.diffusion.image_encoder.state_dict(),
            'unet_state': self.diffusion.unet.state_dict(),
            'learning_updates': self.learning_updates,
            'conversion_count': self.conversion_count
        }, save_path)
        print(f"      💾 Saved continuously learned model to {save_path}")


def create_hybrid_converter(
    device: str = "cpu",
    enable_learning: bool = True
) -> HybridCAD3DConverter:
    """
    Factory function to create and initialize the HybridCAD3DConverter.

    It automatically looks for pre-trained models in standard project locations.
    """
    vit_path = Path("trained_models/vit/vit_best.pth")
    diffusion_path = Path("trained_models/diffusion/diffusion_best.pth")
    
    return HybridCAD3DConverter(
        device=device,
        vit_model_path=vit_path if vit_path.exists() else None,
        diffusion_model_path=diffusion_path if diffusion_path.exists() else None,
        enable_learning=enable_learning
    )


if __name__ == "__main__":
    print("--- Hybrid CAD 3D Converter Demonstration ---")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}\n")
        
        # 1. Create the converter
        converter = create_hybrid_converter(device=device, enable_learning=True)
        
        # 2. Prepare a test image (create one if it doesn't exist)
        demo_dir = Path("demo_output/hybrid_vit_diffusion")
        demo_dir.mkdir(parents=True, exist_ok=True)
        test_image_path = demo_dir / "test_image.png"

        if not test_image_path.exists():
            print(f"Test image not found, creating a synthetic one at {test_image_path}...")
            img = np.ones((256, 256, 3), dtype=np.uint8) * 230
            cv2.putText(img, 'ViT+Diffusion', (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 30, 30), 3)
            cv2.imwrite(str(test_image_path), img)
            print("  ✅ Synthetic image created.")

        # 3. Run the conversion
        output_dxf_path = demo_dir / "hybrid_output_3d.dxf"
        results = converter.convert(
            image_path=test_image_path,
            output_path=output_dxf_path,
            sampling_steps=20,  # Use fewer steps for a quick demo
            learn_from_this=True
        )
        
        print("\n--- Conversion Results ---")
        for key, value in results.items():
            print(f"  - {key}: {value}")
        print("✅ Demonstration finished successfully.")

    except ImportError as e:
        print(f"\n[ERROR] A required library is missing: {e}")
        print("Please ensure all dependencies are installed.")
    except Exception as e:
        import traceback
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        traceback.print_exc()
