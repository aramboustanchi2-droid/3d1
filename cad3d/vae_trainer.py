"""VAE Trainer (adapted to CAD_VAE)

This updated trainer aligns with `cad3d.vae_model.CAD_VAE` and `VAELoss`.
Differences vs previous design:
- Removed image & depth reconstruction branches (not present in current CAD_VAE)
- Uses Chamfer + KL + optional voxel + smoothness losses via VAELoss
- Adds KL annealing schedule (warmup + plateau) configurable
- Optional voxel target generation by point-cloud voxelization
- Experience replay retained for continuous learning
- NEW: Per-epoch JSON logging of KL weight + train/val losses for plotting
"""

from pathlib import Path
from typing import Optional, Tuple
import time
import json

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .vae_model import create_cad_vae, VAELoss, CAD_VAE
from .diffusion_trainer import CAD2D3DDataset, ExperienceReplayBuffer


def voxelize_point_cloud(pc: torch.Tensor, resolution: int = 32) -> torch.Tensor:
    """Convert normalized point cloud [-1,1] to occupancy voxel grid.
    Args:
        pc: (B, N, 3)
        resolution: voxel grid edge size
    Returns:
        vox: (B, 1, R, R, R) occupancy in {0,1}
    """
    B, N, _ = pc.shape
    device = pc.device
    vox = torch.zeros(B, resolution, resolution, resolution, device=device)
    coords = ((pc + 1) * 0.5 * (resolution - 1)).long().clamp(0, resolution - 1)
    x, y, z = coords.unbind(-1)
    for b in range(B):
        vox[b, x[b], y[b], z[b]] = 1
    return vox.unsqueeze(1)


class VAETrainer:
    def __init__(
        self,
        device: Optional[str] = None,
        latent_dim: int = 512,
        num_points: int = 2048,
        lr: float = 2e-4,
        save_dir: Path = Path("trained_models/vae"),
        kl_max: float = 0.001,
        kl_warmup_epochs: int = 10,
        use_voxel_loss: bool = True,
        voxel_resolution: int = 32,
        enable_image_recon: bool = False,
        image_weight: float = 0.5
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: CAD_VAE = create_cad_vae(latent_dim=latent_dim, output_points=num_points, device=self.device, enable_image_recon=enable_image_recon)
        self.opt = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.opt, T_0=10, T_mult=2)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.replay = ExperienceReplayBuffer(capacity=800)
        self.best_loss = float('inf')
        self.loss_history = []
        self.epoch_logs = []  # for JSON logging (epoch, train, val, kl_weight)

        # KL annealing parameters
        self.kl_max = kl_max
        self.kl_warmup_epochs = kl_warmup_epochs
        self.current_epoch = 0

        # Loss object (weights except KL configured here)
        self.loss_fn = VAELoss(
            kl_weight=kl_max,  # will be dynamically rescaled
            point_weight=1.0,
            voxel_weight=0.5 if use_voxel_loss else 0.0,
            smoothness_weight=0.1,
            image_weight=image_weight if enable_image_recon else 0.0
        )
        self.use_voxel_loss = use_voxel_loss
        self.voxel_resolution = voxel_resolution
        self.enable_image_recon = enable_image_recon

    def _compute_kl_weight(self) -> float:
        if self.current_epoch >= self.kl_warmup_epochs:
            return self.kl_max
        return self.kl_max * (self.current_epoch / max(1, self.kl_warmup_epochs))

    def train_epoch(self, loader: DataLoader) -> Tuple[float, dict]:
        self.model.train()
        total = 0.0
        n = 0
        last_parts = {}
        for i, (img, pc_gt) in enumerate(loader):
            img = img.to(self.device)
            pc_gt = pc_gt.to(self.device)
            out = self.model(img)
            target_vox = None
            if self.use_voxel_loss and 'voxels' in out:
                target_vox = voxelize_point_cloud(pc_gt, resolution=self.voxel_resolution)

            # Update dynamic KL weight
            self.loss_fn.kl_weight = self._compute_kl_weight()
            target_img = None
            if self.enable_image_recon:
                # normalize original image to [-1,1]
                target_img = (img * 2.0) - 1.0  # since img was in [0,1] earlier
            losses = self.loss_fn(out, pc_gt, target_vox, target_image=target_img)
            loss = losses['total']

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.opt.step()

            total += loss.item()
            n += 1
            last_parts = losses

            # Replay buffer add few samples
            for b in range(min(img.shape[0], 2)):
                self.replay.add(img[b].detach().cpu(), pc_gt[b].detach().cpu())

            if i % 10 == 0:
                print(
                    f"Epoch {self.current_epoch} | Batch {i}/{len(loader)} | "
                    f"Loss {loss.item():.5f} | point {losses['point']:.4f} "
                    f"vox {losses['voxel']:.4f} kl {losses['kl']:.4f} smooth {losses['smooth']:.4f} "
                    f"KLw {self.loss_fn.kl_weight:.6f}"
                )
        avg = total / max(1, n)
        self.loss_history.append(avg)
        self.scheduler.step()
        return avg, last_parts

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total = 0.0
        n = 0
        for img, pc_gt in loader:
            img = img.to(self.device)
            pc_gt = pc_gt.to(self.device)
            out = self.model(img)
            target_vox = None
            if self.use_voxel_loss and 'voxels' in out:
                target_vox = voxelize_point_cloud(pc_gt, resolution=self.voxel_resolution)
            self.loss_fn.kl_weight = self._compute_kl_weight()  # ensure same weighting
            losses = self.loss_fn(out, pc_gt, target_vox)
            total += losses['total'].item()
            n += 1
        return total / max(1, n)

    def save_checkpoint(self, epoch: int, val_loss: float, parts: dict, is_best: bool = False):
        ckpt = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'loss_history': self.loss_history,
            'last_parts': {k: float(v) for k, v in parts.items()},
            'kl_weight': self.loss_fn.kl_weight,
        }
        # Always save epoch checkpoint
        epoch_path = self.save_dir / f"vae_epoch_{epoch}.pth"
        torch.save(ckpt, epoch_path)
        print(f"Saved checkpoint: {epoch_path}")
        # Additionally save as best if applicable
        if is_best:
            best_path = self.save_dir / "vae_best.pth"
            torch.save(ckpt, best_path)
            print(f"Saved best checkpoint: {best_path}")

    def interpolate_gallery(self, img_a: torch.Tensor, img_b: torch.Tensor, steps: int = 8):
        """Generate latent interpolation between two batch images (B=1 each)."""
        self.model.eval()
        gallery = []
        with torch.no_grad():
            for alpha in torch.linspace(0, 1, steps):
                mu_a, _ = self.model.encoder(img_a)
                mu_b, _ = self.model.encoder(img_b)
                z = (1 - alpha) * mu_a + alpha * mu_b
                pts, vox = self.model.decoder(z) if self.model.use_hybrid_decoder else (self.model.decoder(z), None)
                gallery.append({'alpha': float(alpha), 'points': pts.cpu().numpy(), 'voxels': None if vox is None else vox.cpu().numpy()})
        out_dir = self.save_dir / "interpolations"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"interp_{int(time.time())}.json"
        import json as _json
        serializable = [{"alpha": g['alpha'], "points_shape": list(g['points'].shape)} for g in gallery]
        with open(path, 'w') as f:
            _json.dump(serializable, f, indent=2)
        print(f"Saved interpolation gallery metadata: {path}")
        return gallery

    def train(self, data_dir: Path, epochs: int = 40, batch_size: int = 8):
        dataset = CAD2D3DDataset(data_dir=data_dir, image_size=256, num_points=self.model.decoder.output_points, augment=True)
        if len(dataset) < 2:
            print("Dataset too small; please generate data with cad3d.diffusion_trainer.create_synthetic_training_data")
        train_size = max(1, int(0.9 * len(dataset)))
        val_size = max(1, len(dataset) - train_size)
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

        t0 = time.time()
        epoch_log_path = self.save_dir / "vae_epoch_log.json"
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            tr, parts = self.train_epoch(train_loader)
            val = self.validate(val_loader)
            print(f"Epoch {epoch}/{epochs} | train {tr:.5f} | val {val:.5f} | KLw {self.loss_fn.kl_weight:.6f}")
            self.save_checkpoint(epoch, val, parts, is_best=val < self.best_loss)
            if val < self.best_loss:
                self.best_loss = val
                print(f"✅ New best val loss: {val:.5f}")
            # occasional replay fine-tune
            if epoch % 5 == 0 and len(self.replay) >= batch_size:
                for _ in range(3):
                    sample = self.replay.sample(batch_size)
                    if sample is None:
                        break
                    img, pc_gt = sample
                    img = img.to(self.device)
                    pc_gt = pc_gt.to(self.device)
                    out = self.model(img)
                    target_vox = None
                    if self.use_voxel_loss and 'voxels' in out:
                        target_vox = voxelize_point_cloud(pc_gt, resolution=self.voxel_resolution)
                    self.loss_fn.kl_weight = self._compute_kl_weight()
                    target_img = None
                    if self.enable_image_recon:
                        target_img = (img * 2.0) - 1.0
                    losses = self.loss_fn(out, pc_gt, target_vox, target_image=target_img)
                    self.opt.zero_grad()
                    losses['total'].backward()
                    self.opt.step()
                print("Performed replay fine-tuning")
            # epoch logging (append entry)
            entry = {
                'epoch': epoch,
                'train_loss': float(tr),
                'val_loss': float(val),
                'kl_weight': float(self.loss_fn.kl_weight)
            }
            self.epoch_logs.append(entry)
            try:
                if epoch_log_path.exists():
                    with open(epoch_log_path, 'r', encoding='utf-8') as f:
                        existing = json.load(f)
                        if not isinstance(existing, list):
                            existing = []
                else:
                    existing = []
                existing.append(entry)
                with open(epoch_log_path, 'w', encoding='utf-8') as f:
                    json.dump(existing, f, indent=2)
            except Exception as e:
                print(f"Warning: could not update epoch log: {e}")
        total_time = time.time() - t0
        report = {
            'epochs': epochs,
            'best_val': self.best_loss,
            'time_sec': total_time,
            'loss_history': self.loss_history,
            'device': self.device,
            'kl_max': self.kl_max,
            'kl_warmup_epochs': self.kl_warmup_epochs,
            'epoch_logs': self.epoch_logs
        }
        rp = self.save_dir / "vae_training_report.json"
        with open(rp, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Saved training report: {rp}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = VAETrainer(device=device)
    data_dir = Path("training_data/diffusion_synthetic")
    # short smoke test epochs
    trainer.train(data_dir=data_dir, epochs=2, batch_size=2 if device == 'cuda' else 1)
