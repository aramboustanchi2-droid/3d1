"""
Advanced 3D Reconstruction using Vision Transformer

This module provides a sophisticated system for reconstructing 3D models from 2D
architectural drawings using a Vision Transformer (ViT).

Features:
- Deep semantic understanding of architectural elements from images.
- Intelligent prediction of heights and depths for elements.
- Material-aware 3D reconstruction capabilities.
- Assembly of structures based on detected relationships.
- Generation of multi-layered 3D models in DXF format.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import cv2
import logging

try:
    import ezdxf
    from ezdxf.math import Vec3
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False
    logging.warning("ezdxf is not installed. DXF generation will not be available.")

from .vision_transformer_cad import CADVisionAnalyzer, TORCH_AVAILABLE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class Element3DConfig:
    """
    Configuration for 3D representation of different element types.

    Attributes:
        name: The name of the element type.
        default_height: Default extrusion height in millimeters.
        default_thickness: Default thickness in millimeters.
        extrusion_type: The method of extrusion ('vertical', 'horizontal', 'custom').
        color: The RGB color tuple for the element's layer.
        layer_name: The target DXF layer name for the element.
    """
    name: str
    default_height: float
    default_thickness: float
    extrusion_type: str
    color: Tuple[int, int, int]
    layer_name: str


# Predefined configurations for common architectural elements
ELEMENT_CONFIGS: Dict[str, Element3DConfig] = {
    "wall": Element3DConfig("wall", 3000, 200, "vertical", (200, 200, 200), "A-WALL-3D"),
    "door": Element3DConfig("door", 2100, 50, "vertical", (139, 69, 19), "A-DOOR-3D"),
    "window": Element3DConfig("window", 1500, 100, "vertical", (135, 206, 250), "A-GLAZ-3D"),
    "column": Element3DConfig("column", 3000, 300, "vertical", (128, 128, 128), "S-COLS-3D"),
    "beam": Element3DConfig("beam", 400, 300, "horizontal", (160, 82, 45), "S-BEAM-3D"),
    "slab": Element3DConfig("slab", 200, 200, "horizontal", (211, 211, 211), "S-SLAB-3D"),
    "stair": Element3DConfig("stair", 3000, 300, "custom", (184, 134, 11), "A-STAIR-3D"),
    "railing": Element3DConfig("railing", 1000, 50, "vertical", (192, 192, 192), "A-RAIL-3D"),
    "furniture": Element3DConfig("furniture", 800, 100, "vertical", (210, 180, 140), "A-FURN-3D"),
}


class Advanced3DReconstructor:
    """
    A system for advanced 3D reconstruction using a Vision Transformer.

    This class orchestrates the process of:
    1. Analyzing a 2D drawing image with a Vision Transformer to understand its semantic content.
    2. Predicting properties like height, depth, and materials for detected elements.
    3. Reconstructing accurate 3D geometry from the analyzed data.
    4. Generating a layered DXF file with appropriate attributes for the 3D model.
    """

    def __init__(self, vit_model_path: Optional[str] = None, device: str = "auto"):
        """
        Initializes the reconstructor.

        Args:
            vit_model_path: Path to the pre-trained Vision Transformer model.
            device: The device to run the model on ('cpu', 'cuda', or 'auto').

        Raises:
            RuntimeError: If PyTorch or ezdxf are not installed.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for Advanced 3D Reconstruction. Please install it.")
        if not EZDXF_AVAILABLE:
            raise RuntimeError("ezdxf is required for DXF generation. Please install it.")

        self.analyzer = CADVisionAnalyzer(model_path=vit_model_path, device=device)
        self.configs = ELEMENT_CONFIGS.copy()
        self.scale_factor = 10.0  # Default: 10mm per pixel

    def set_scale(self, pixels: float, real_world_mm: float):
        """
        Sets the scale factor from a known dimension in the image.

        Args:
            pixels: The length of the known dimension in pixels.
            real_world_mm: The real-world length of the dimension in millimeters.
        """
        if pixels > 0:
            self.scale_factor = real_world_mm / pixels
            logging.info(f"Scale factor set to {self.scale_factor:.2f} mm/pixel")
        else:
            logging.warning("Cannot set scale with zero pixels.")

    def reconstruct_from_image(
        self,
        image: np.ndarray,
        output_dxf_path: str,
        auto_scale: bool = True,
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Executes the full reconstruction pipeline: Image -> Analysis -> 3D Model -> DXF.

        Args:
            image: The input image as an RGB numpy array.
            output_dxf_path: The path for the output DXF file.
            auto_scale: If True, attempts to automatically detect the scale.
            min_confidence: The minimum confidence score for an element to be processed.

        Returns:
            A dictionary containing statistics about the reconstruction process.
        """
        logging.info("Analyzing image with Vision Transformer...")
        analysis = self.analyzer.analyze_image(image)
        logging.info(f"Detected {len(analysis['elements'])} potential elements.")

        if auto_scale:
            detected_scale = self._detect_scale(analysis)
            if detected_scale:
                self.scale_factor = detected_scale
                logging.info(f"Auto-detected scale: {self.scale_factor:.2f} mm/pixel")

        logging.info("Generating 3D structure from analysis...")
        structure_3d = self.analyzer.generate_3d_structure(analysis)

        logging.info("Building 3D geometry and generating DXF...")
        stats = self._build_3d_dxf(structure_3d, output_dxf_path, image.shape[:2], min_confidence)
        
        logging.info(f"3D DXF saved to {output_dxf_path}")
        logging.info(f"Created {stats['total_entities']} 3D entities across {stats['total_layers']} layers.")
        
        return stats

    def _detect_scale(self, analysis: Dict[str, Any]) -> Optional[float]:
        """
        Attempts to auto-detect the drawing scale from the analysis results.

        Current strategies include:
        1. Using standard door sizes (e.g., 2100mm height).
        2. Detecting a scale bar annotation (future implementation).

        Args:
            analysis: The analysis dictionary from the CADVisionAnalyzer.

        Returns:
            The detected scale factor (mm/pixel), or None if not found.
        """
        # Strategy 1: Find doors and assume a standard height
        for elem in analysis.get('elements', []):
            if elem.get('class') == 'door' and elem.get('confidence', 0) > 0.7:
                # Standard door height is ~2100mm
                patch_size = self.analyzer.config.image_size / int(np.sqrt(len(analysis['semantic_map'].flatten())))
                door_height_pixels = patch_size  # This is a rough approximation
                if door_height_pixels > 0:
                    return 2100.0 / door_height_pixels
        
        logging.warning("Could not auto-detect scale. Using default.")
        return None

    def _build_3d_dxf(
        self,
        structure_3d: Dict[str, Any],
        output_path: str,
        image_shape: Tuple[int, int],
        min_confidence: float
    ) -> Dict[str, Any]:
        """
        Builds the 3D DXF file from the structured 3D data.

        Args:
            structure_3d: The 3D structure data from the analyzer.
            output_path: The path to save the DXF file.
            image_shape: The (height, width) of the original image.
            min_confidence: The minimum confidence to process an element.

        Returns:
            A dictionary with statistics about the generated DXF.
        """
        doc = ezdxf.new(dxfversion='R2010', setup=True)
        msp = doc.modelspace()
        
        created_layers = set()
        for config in self.configs.values():
            if config.layer_name not in created_layers:
                layer = doc.layers.new(name=config.layer_name)
                layer.color = ezdxf.colors.rgb2aci(config.color)
                created_layers.add(config.layer_name)
        
        stats: Dict[str, Any] = {'total_entities': 0, 'total_layers': len(created_layers), 'elements_by_class': {}}
        
        image_height, image_width = image_shape
        grid_size = structure_3d['semantic_map'].shape[0]
        patch_size_pixels = image_width / grid_size
        
        for elem in structure_3d.get('elements', []):
            if elem.get('confidence', 0) < min_confidence:
                continue

            class_name = elem.get('class')
            config = self.configs.get(class_name, Element3DConfig(class_name, 1000, 100, "vertical", (150, 150, 150), "A-MISC-3D"))
            
            contours = self._extract_contours_from_mask(elem.get('mask', np.array([])))
            if not contours:
                continue
            
            height_mm = elem.get('height_mm', config.default_height)
            
            for contour in contours:
                contour_mm = [(x * patch_size_pixels * self.scale_factor, (image_height - y) * self.scale_factor) for y, x in contour]
                
                entities = []
                if config.extrusion_type == "vertical":
                    entities = self._create_vertical_extrusion(contour_mm, height_mm, config.layer_name)
                elif config.extrusion_type == "horizontal":
                    entities = self._create_horizontal_slab(contour_mm, config.default_thickness, config.layer_name)
                else:  # custom
                    entities = self._create_custom_geometry(contour_mm, elem, config.layer_name)
                
                for entity in entities:
                    msp.add_entity(entity)
                
                stats['total_entities'] += len(entities)
                stats['elements_by_class'][class_name] = stats['elements_by_class'].get(class_name, 0) + 1
        
        doc.saveas(output_path)
        return stats

    def _extract_contours_from_mask(self, mask: np.ndarray) -> List[np.ndarray]:
        """Extracts contours from a binary mask."""
        if mask.size == 0:
            return []
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c.squeeze() for c in contours if len(c) >= 3 and c.squeeze().ndim == 2]

    def _create_vertical_extrusion(self, footprint: List[Tuple[float, float]], height: float, layer: str) -> List[ezdxf.entities.Face3d]:
        """Creates a vertical extrusion (e.g., for walls, columns) as a list of 3DFACE entities."""
        if len(footprint) < 3:
            return []
            
        bottom_verts = [Vec3(x, y, 0) for x, y in footprint]
        top_verts = [Vec3(x, y, height) for x, y in footprint]
        
        entities = []
        n = len(footprint)
        for i in range(n):
            j = (i + 1) % n
            face = ezdxf.entities.Face3d.new(dxfattribs={'layer': layer})
            face.dxf.vtx0, face.dxf.vtx1, face.dxf.vtx2, face.dxf.vtx3 = bottom_verts[i], bottom_verts[j], top_verts[j], top_verts[i]
            entities.append(face)
        
        # Add caps (simplified triangulation)
        if len(footprint) >= 3:
            bottom_cap = ezdxf.entities.Face3d.new(dxfattribs={'layer': layer})
            bottom_cap.dxf.vtx0, bottom_cap.dxf.vtx1, bottom_cap.dxf.vtx2 = bottom_verts[0], bottom_verts[1], bottom_verts[2]
            bottom_cap.dxf.vtx3 = bottom_verts[2] if len(footprint) == 3 else bottom_verts[3]
            entities.append(bottom_cap)

            top_cap = ezdxf.entities.Face3d.new(dxfattribs={'layer': layer})
            top_cap.dxf.vtx0, top_cap.dxf.vtx1, top_cap.dxf.vtx2 = top_verts[0], top_verts[1], top_verts[2]
            top_cap.dxf.vtx3 = top_verts[2] if len(footprint) == 3 else top_verts[3]
            entities.append(top_cap)
            
        return entities

    def _create_horizontal_slab(self, boundary: List[Tuple[float, float]], thickness: float, layer: str) -> List[ezdxf.entities.Face3d]:
        """Creates a horizontal slab (e.g., for floors, roofs)."""
        return self._create_vertical_extrusion(boundary, thickness, layer)

    def _create_custom_geometry(self, footprint: List[Tuple[float, float]], element_data: Dict, layer: str) -> List[ezdxf.entities.Face3d]:
        """Creates custom geometry for special elements like stairs."""
        # Placeholder: For now, defaults to simple extrusion.
        # A production implementation would have specialized geometry generation logic here.
        height = element_data.get('height_mm', 1000)
        return self._create_vertical_extrusion(footprint, height, layer)

    def visualize_analysis(self, image: np.ndarray, output_path: str):
        """
        Creates and saves a visualization of the Vision Transformer's analysis.

        Args:
            image: The input image that was analyzed.
            output_path: The path to save the visualization image.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logging.error("matplotlib is required for visualization. Please install it.")
            return

        analysis = self.analyzer.analyze_image(image)
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        fig.suptitle("Vision Transformer CAD Analysis", fontsize=16)

        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(analysis.get('semantic_map', np.zeros(image.shape[:2])), cmap='tab20')
        axes[0, 1].set_title(f"Semantic Segmentation ({len(analysis.get('elements', []))} elements)")
        axes[0, 1].axis('off')
        
        im = axes[0, 2].imshow(analysis.get('confidence_map', np.zeros(image.shape[:2])), cmap='hot')
        axes[0, 2].set_title("Confidence Map")
        axes[0, 2].axis('off')
        fig.colorbar(im, ax=axes[0, 2])

        if 'height_map' in analysis:
            im = axes[1, 0].imshow(analysis['height_map'], cmap='viridis')
            axes[1, 0].set_title("Predicted Heights (mm)")
            axes[1, 0].axis('off')
            fig.colorbar(im, ax=axes[1, 0])
        
        if 'depth_map' in analysis:
            im = axes[1, 1].imshow(analysis['depth_map'], cmap='plasma')
            axes[1, 1].set_title("Predicted Depth")
            axes[1, 1].axis('off')
            fig.colorbar(im, ax=axes[1, 1])
        
        if analysis.get('attention_maps'):
            attn = analysis['attention_maps'][-1][0]  # Last layer, first head
            im = axes[1, 2].imshow(attn, cmap='coolwarm')
            axes[1, 2].set_title("Attention Map (Last Layer)")
            axes[1, 2].axis('off')
            fig.colorbar(im, ax=axes[1, 2])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Analysis visualization saved to {output_path}")


def main():
    """Example usage of the Advanced3DReconstructor."""
    logging.info("Running Advanced 3D Reconstruction System Example")
    
    if not TORCH_AVAILABLE or not EZDXF_AVAILABLE:
        logging.error("Missing required libraries (PyTorch or ezdxf). Aborting example.")
        return
    
    try:
        reconstructor = Advanced3DReconstructor()
        
        # Create a dummy image for testing purposes
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        output_dxf = "test_3d_reconstruction.dxf"
        stats = reconstructor.reconstruct_from_image(test_image, output_dxf)
        
        logging.info("\n--- Reconstruction Statistics ---")
        logging.info(f"Total entities created: {stats['total_entities']}")
        logging.info(f"Total layers created: {stats['total_layers']}")
        logging.info("Elements by class:")
        for class_name, count in stats.get('elements_by_class', {}).items():
            logging.info(f"  - {class_name}: {count}")
            
        # Also generate a visualization
        reconstructor.visualize_analysis(test_image, "test_analysis_visualization.png")

    except Exception as e:
        logging.error(f"An error occurred during the example run: {e}", exc_info=True)


if __name__ == "__main__":
    main()
