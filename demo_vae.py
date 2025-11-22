"""Demo for CAD_VAE usage.

Showcases:
1. Model info & parameter counts
2. Single image forward pass (points + optional voxels)
3. Saving generated point cloud to DXF (subset for readability)
4. Latent interpolation between two synthetic CAD-like images
"""

from pathlib import Path
import numpy as np
import cv2
import torch
import ezdxf

from cad3d.vae_model import create_cad_vae


def create_test_image(kind: str = "rects") -> np.ndarray:
    img = np.ones((256, 256, 3), dtype=np.uint8) * 255
    rng = np.random.default_rng()
    if kind == "rects":
        for _ in range(4):
            x, y = rng.integers(20, 200, 2)
            w, h = rng.integers(20, 80, 2)
            cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 0), 2)
    elif kind == "mix":
        for _ in range(3):
            cx, cy = rng.integers(40, 220, 2)
            r = rng.integers(15, 50)
            cv2.circle(img, (int(cx), int(cy)), int(r), (0, 0, 0), 2)
        for _ in range(2):
            x, y = rng.integers(20, 200, 2)
            w, h = rng.integers(30, 90, 2)
            cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 0), 2)
    else:
        cv2.line(img, (30, 30), (220, 220), (0, 0, 0), 3)
    return img


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
    return t


def save_point_cloud_to_dxf(points: torch.Tensor, path: Path, max_points: int = 800):
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = points.detach().cpu().numpy()
    if pts.ndim == 3:  # (B,N,3) take first batch
        pts = pts[0]
    if pts.shape[0] > max_points:
        pts = pts[:max_points]
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    # Represent as polyline chain (order arbitrary)
    pl = msp.add_polyline3d(pts.tolist())
    pl.dxf.layer = "VAE_POINTS"
    doc.saveas(str(path))
    print(f"Saved DXF with {pts.shape[0]} points -> {path}")


def show_info(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=== CAD_VAE Model Info ===")
    print(f"Total params: {total:,} | Trainable: {trainable:,} | Approx size: {total*4/1024/1024:.2f} MB")


def demo_single(model, device: str):
    print("\n[Single Forward]")
    img_np = create_test_image("mix")
    tensor = image_to_tensor(img_np).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    print("Points:", out['points'].shape, "Voxels:" if 'voxels' in out else "(no voxels)")
    save_point_cloud_to_dxf(out['points'], Path("demo_output/vae/sample.dxf"))
    if 'image_recon' in out:
        recon = out['image_recon'][0].cpu().permute(1,2,0).numpy()
        # compute MSE
        orig = tensor[0].cpu().permute(1,2,0).numpy()
        mse = float(np.mean((recon - orig)**2))
        recon_img = ((recon * 127.5) + 127.5).clip(0,255).astype('uint8')
        cv2.imwrite("demo_output/vae/sample_recon.png", cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR))
        print(f"Image reconstruction saved (MSE={mse:.6f})")


def demo_interpolation(model, device: str):
    print("\n[Interpolation]")
    a = image_to_tensor(create_test_image("rects")).unsqueeze(0).to(device)
    b = image_to_tensor(create_test_image("mix")).unsqueeze(0).to(device)
    gallery = model.interpolate(a, b, steps=6)
    out_dir = Path("demo_output/vae/interp")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, pts in enumerate(gallery):
        save_point_cloud_to_dxf(pts, out_dir / f"interp_{i}.dxf", max_points=600)
    print(f"Generated {len(gallery)} interpolated point clouds.")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # enable image reconstruction branch
    model = create_cad_vae(device=device, enable_image_recon=True)
    show_info(model)
    demo_single(model, device)
    demo_interpolation(model, device)


if __name__ == "__main__":
    main()
