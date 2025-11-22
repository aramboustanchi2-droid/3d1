"""
Quick VAE Training: 2 epochs to generate vae_epoch_log.json for verification.
"""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cad3d.vae_trainer import VAETrainer, CAD2D3DDataset


def create_tiny_dataset(data_dir: Path, n_samples=10):
    """Create a minimal synthetic dataset for quick training."""
    img_dir = data_dir / 'images'
    pc_dir = data_dir / 'pointclouds'
    img_dir.mkdir(parents=True, exist_ok=True)
    pc_dir.mkdir(parents=True, exist_ok=True)
    
    import cv2
    for i in range(n_samples):
        # Synthetic image
        img = np.ones((256, 256, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (50+i*5, 50), (150, 150), (0, 0, 0), 2)
        cv2.imwrite(str(img_dir / f'sample_{i:03d}.png'), img)
        
        # Synthetic point cloud
        pc = np.random.randn(512, 3).astype(np.float32) * 0.5
        np.save(pc_dir / f'sample_{i:03d}.npy', pc)
    
    print(f"✓ Created {n_samples} synthetic samples in {data_dir}")


if __name__ == '__main__':
    print("="*70)
    print("QUICK VAE TRAINING (2 epochs to generate KL log)")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create tiny dataset
    data_dir = Path('outputs/quick_vae_train')
    if not (data_dir / 'images').exists():
        create_tiny_dataset(data_dir, n_samples=10)
    
    # Train for 2 epochs
    print("\nTraining VAE for 2 epochs...")
    
    trainer = VAETrainer(device=device, latent_dim=256, num_points=512, save_dir=Path('trained_models/vae_quick'), use_voxel_loss=False)
    trainer.train(data_dir=data_dir, epochs=2, batch_size=2)
    
    # Check log file
    log_path = Path('trained_models/vae_quick/vae_epoch_log.json')
    if log_path.exists():
        import json
        logs = json.loads(log_path.read_text(encoding='utf-8'))
        print(f"\n✓ vae_epoch_log.json exists with {len(logs)} entries")
        for entry in logs:
            print(f"  Epoch {entry['epoch']}: train_loss={entry['train_loss']:.4f}, kl_weight={entry['kl_weight']:.6f}")
        print("\n✅ KL log verification PASSED")
    else:
        print(f"\n⚠️  vae_epoch_log.json not found at {log_path.absolute()}")
