"""
VAE integration utilities
- Quick conversion from image to coarse 3D point cloud + optional DXF export
- Lightweight and fast; good for CAD-AI pre-processing or previews
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

import ezdxf

from .vae_model import create_cad_vae


class VAEConverter:
    def __init__(self, device: Optional[str] = None, checkpoint: Optional[Path] = None, num_points: int = 2048):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_cad_vae(output_points=num_points, device=self.device)
        self.model.eval()
        if checkpoint and Path(checkpoint).exists():
            ckpt = torch.load(checkpoint, map_location=self.device)
            state = ckpt.get('state_dict', ckpt)
            self.model.load_state_dict(state, strict=False)

    @torch.no_grad()
    def image_to_point_cloud(self, image_np: np.ndarray) -> np.ndarray:
        """Convert RGB image (H,W,3, uint8) in [0,255] to point cloud (N,3) in mm space."""
        img = image_np.astype(np.float32) / 127.5 - 1.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        out = self.model(img)
        pc = out['points'][0].cpu().numpy()  # [-1,1]
        # scale to mm
        pc_mm = pc * 1000.0
        return pc_mm

    @staticmethod
    def point_cloud_to_dxf(points_mm: np.ndarray, output_path: Path, sample: int = 100):
        doc = ezdxf.new(setup=True)
        msp = doc.modelspace()
        cube = 10.0
        for i in range(0, len(points_mm), max(1, len(points_mm)//sample)):
            x, y, z = points_mm[i]
            verts = [
                (x, y, z), (x+cube, y, z), (x+cube, y+cube, z), (x, y+cube, z),
                (x, y, z+cube), (x+cube, y, z+cube), (x+cube, y+cube, z+cube), (x, y+cube, z+cube)
            ]
            mesh = msp.add_mesh()
            with mesh.edit_data() as md:
                md.vertices = verts
                md.faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]]
        doc.saveas(output_path)
        return str(output_path)

    def convert_image_file(self, image_path: Path, output_dxf: Optional[Path] = None) -> dict:
        import cv2
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        pc_mm = self.image_to_point_cloud(img)
        result = {
            'num_points': int(pc_mm.shape[0]),
            'bounds_mm': [float(pc_mm.min()), float(pc_mm.max())]
        }
        if output_dxf is not None:
            Path(output_dxf).parent.mkdir(parents=True, exist_ok=True)
            path = self.point_cloud_to_dxf(pc_mm, output_dxf, sample=200)
            result['dxf'] = path
        return result


def get_vae_service(device: Optional[str] = None) -> VAEConverter:
    # attempt to use best checkpoint if exists
    best = Path("trained_models/vae/vae_best.pth")
    ckpt = best if best.exists() else None
    return VAEConverter(device=device, checkpoint=ckpt)


if __name__ == "__main__":
    conv = get_vae_service()
    # create a synthetic test image
    import numpy as np, cv2
    img = np.ones((256,256,3), dtype=np.uint8)*255
    cv2.rectangle(img,(40,60),(200,180),(0,0,0),3)
    tmp = Path("demo_output/vae/tmp_input.png")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(tmp), img)
    out_path = Path("demo_output/vae/vae_3d.dxf")
    res = conv.convert_image_file(tmp, out_path)
    print(res)
