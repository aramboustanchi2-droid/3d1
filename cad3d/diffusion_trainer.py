"""
Diffusion Model Training System
سیستم آموزش مدل انتشار

Features:
- Progressive training (2D → 3D)
- Multi-resolution training
- Continuous learning from conversions
- Experience replay buffer
- Automatic checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
import time
from typing import List, Dict, Optional, Tuple
import cv2
from collections import deque

from .diffusion_3d_model import create_diffusion_model, Diffusion3DConverter


class CAD2D3DDataset(Dataset):
    """Dataset for paired 2D images and 3D point clouds"""
    
    def __init__(
        self,
        data_dir: Path,
        image_size: int = 256,
        num_points: int = 2048,
        augment: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.num_points = num_points
        self.augment = augment
        
        # Find all paired data
        self.pairs = []
        
        image_dir = self.data_dir / "images"
        pointcloud_dir = self.data_dir / "pointclouds"
        
        if image_dir.exists() and pointcloud_dir.exists():
            for img_path in image_dir.glob("*.png"):
                pc_path = pointcloud_dir / f"{img_path.stem}.npy"
                if pc_path.exists():
                    self.pairs.append((img_path, pc_path))
        
        print(f"Found {len(self.pairs)} paired samples")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path, pc_path = self.pairs[idx]
        
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        
        # Augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
            
            # Random brightness/contrast
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.randint(-20, 20)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # Normalize to [-1, 1]
        img = (img.astype(np.float32) / 127.5) - 1.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        
        # Load point cloud
        pc = np.load(pc_path)
        
        # Subsample or pad to num_points
        if pc.shape[0] > self.num_points:
            indices = np.random.choice(pc.shape[0], self.num_points, replace=False)
            pc = pc[indices]
        elif pc.shape[0] < self.num_points:
            # Pad with random points
            padding = np.random.randn(self.num_points - pc.shape[0], 3) * 0.01
            pc = np.vstack([pc, padding])
        
        # Normalize point cloud
        centroid = pc.mean(axis=0)
        pc = pc - centroid
        max_dist = np.abs(pc).max()
        if max_dist > 0:
            pc = pc / max_dist
        
        pc = torch.from_numpy(pc.astype(np.float32))
        
        return img, pc


class ExperienceReplayBuffer:
    """
    Experience replay for continuous learning
    Stores recent conversions for periodic retraining
    """
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, image: torch.Tensor, point_cloud: torch.Tensor):
        """Add a new experience"""
        self.buffer.append((image.cpu(), point_cloud.cpu()))
    
    def sample(self, batch_size: int):
        """Sample a batch from buffer"""
        if len(self.buffer) < batch_size:
            return None
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        images = torch.stack([item[0] for item in batch])
        point_clouds = torch.stack([item[1] for item in batch])
        
        return images, point_clouds
    
    def __len__(self):
        return len(self.buffer)


class DiffusionTrainer:
    """
    Complete training system for 3D Diffusion Model
    
    Training stages:
    1. Pre-training on synthetic data
    2. Fine-tuning on real CAD data
    3. Continuous learning from user conversions
    """
    
    def __init__(
        self,
        model: Diffusion3DConverter,
        device: str = "cpu",
        learning_rate: float = 1e-4,
        save_dir: Path = Path("trained_models/diffusion")
    ):
        self.model = model
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Diffusion model directory ensured: {self.save_dir.absolute()}")
        
        # Optimizer
        self.optimizer = optim.AdamW(
            list(model.image_encoder.parameters()) + list(model.unet.parameters()),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )
        
        # Experience replay
        self.replay_buffer = ExperienceReplayBuffer(capacity=1000)
        
        # Training stats
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.loss_history = []
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> float:
        """Train for one epoch"""
        self.model.unet.train()
        self.model.image_encoder.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, point_clouds) in enumerate(train_loader):
            images = images.to(self.device)
            point_clouds = point_clouds.to(self.device)
            
            # Training step
            loss = self.model.train_step(images, point_clouds, self.optimizer)
            
            total_loss += loss
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss:.6f}")
            
            # Add to replay buffer
            for i in range(images.shape[0]):
                self.replay_buffer.add(images[i], point_clouds[i])
        
        avg_loss = total_loss / num_batches
        self.loss_history.append(avg_loss)
        
        # Update scheduler
        self.scheduler.step()
        
        return avg_loss
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> float:
        """Validation"""
        self.model.unet.eval()
        self.model.image_encoder.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, point_clouds in val_loader:
                images = images.to(self.device)
                point_clouds = point_clouds.to(self.device)
                
                # Encode image
                condition = self.model.encode_image(images)
                
                # Calculate loss
                loss = self.model.diffusion.training_loss(point_clouds, condition)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        epochs: int = 100,
        batch_size: int = 8,
        save_every: int = 10
    ):
        """
        Complete training loop
        
        Args:
            train_dataset: training dataset
            val_dataset: validation dataset (optional)
            epochs: number of epochs
            batch_size: batch size
            save_every: save checkpoint every N epochs
        """
        print("="*70)
        print("Starting Diffusion Model Training")
        print("="*70)
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Windows compatibility
            pin_memory=True if self.device == "cuda" else False
        )
        
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if self.device == "cuda" else False
            )
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*70}")
            
            # Training
            train_loss = self.train_epoch(train_loader, epoch+1)
            
            # Validation
            if val_dataset is not None:
                val_loss = self.validate(val_loader)
                print(f"\nTrain Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
                current_loss = val_loss
            else:
                print(f"\nTrain Loss: {train_loss:.6f}")
                current_loss = train_loss
            
            # Experience replay (every 5 epochs)
            if epoch > 0 and epoch % 5 == 0 and len(self.replay_buffer) >= batch_size:
                print("\nPerforming experience replay...")
                for _ in range(5):  # 5 replay steps
                    replay_data = self.replay_buffer.sample(batch_size)
                    if replay_data is not None:
                        images, point_clouds = replay_data
                        images = images.to(self.device)
                        point_clouds = point_clouds.to(self.device)
                        loss = self.model.train_step(images, point_clouds, self.optimizer)
                        print(f"  Replay loss: {loss:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch+1, current_loss)
            
            # Save best model
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.save_checkpoint(epoch+1, current_loss, is_best=True)
                print(f"✅ New best model saved! Loss: {current_loss:.6f}")
            
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch time: {epoch_time:.2f}s")
            
            self.epoch = epoch + 1
        
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best loss: {self.best_loss:.6f}")
        print(f"Final loss: {current_loss:.6f}")
        
        # Save final report
        self.save_training_report(total_time)
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'image_encoder_state': self.model.image_encoder.state_dict(),
            'unet_state': self.model.unet.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'loss_history': self.loss_history
        }
        
        if is_best:
            path = self.save_dir / "diffusion_best.pth"
            print(f"Saving best checkpoint to {path}")
        else:
            path = self.save_dir / f"diffusion_epoch_{epoch}.pth"
            print(f"Saving checkpoint to {path}")
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        print(f"Loading checkpoint from {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.image_encoder.load_state_dict(checkpoint['image_encoder_state'])
        self.model.unet.load_state_dict(checkpoint['unet_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        self.loss_history = checkpoint['loss_history']
        
        print(f"✅ Loaded checkpoint from epoch {self.epoch}")
    
    def save_training_report(self, total_time: float):
        """Save training report"""
        report = {
            'model': '3D Diffusion Model',
            'architecture': 'DDPM with U-Net + PointNet++',
            'total_epochs': self.epoch,
            'total_steps': self.global_step,
            'best_loss': self.best_loss,
            'final_loss': self.loss_history[-1] if self.loss_history else 0,
            'training_time_hours': total_time / 3600,
            'loss_history': self.loss_history,
            'replay_buffer_size': len(self.replay_buffer),
            'device': self.device,
            'hyperparameters': {
                'timesteps': self.model.diffusion.timesteps,
                'num_points': self.model.num_points,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
        }
        
        report_path = self.save_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Training report saved to {report_path}")


def create_synthetic_training_data(
    output_dir: Path,
    num_samples: int = 100
):
    """
    Create synthetic 2D-3D paired data for initial training
    
    Generates simple CAD-like drawings and corresponding 3D point clouds
    """
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    pc_dir = output_dir / "pointclouds"
    
    image_dir.mkdir(parents=True, exist_ok=True)
    pc_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} synthetic samples...")
    
    for i in range(num_samples):
        # Create 2D drawing
        img = np.ones((512, 512, 3), dtype=np.uint8) * 255
        
        # Generate random shapes
        num_shapes = np.random.randint(3, 10)
        points_3d = []
        
        for _ in range(num_shapes):
            shape_type = np.random.choice(['rect', 'circle', 'line'])
            
            if shape_type == 'rect':
                x, y = np.random.randint(50, 400, 2)
                w, h = np.random.randint(50, 150, 2)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
                
                # Generate 3D points (extrude rectangle)
                height = np.random.uniform(0.2, 1.0)
                for px in np.linspace(x, x+w, 20):
                    for py in np.linspace(y, y+h, 20):
                        # Normalize to [-1, 1]
                        px_norm = (px / 256.0) - 1.0
                        py_norm = (py / 256.0) - 1.0
                        points_3d.append([px_norm, py_norm, 0])
                        points_3d.append([px_norm, py_norm, height])
            
            elif shape_type == 'circle':
                cx, cy = np.random.randint(100, 400, 2)
                r = np.random.randint(30, 80)
                cv2.circle(img, (cx, cy), r, (0, 0, 0), 2)
                
                # Generate 3D points (extrude circle)
                height = np.random.uniform(0.2, 1.0)
                for angle in np.linspace(0, 2*np.pi, 50):
                    px = cx + r * np.cos(angle)
                    py = cy + r * np.sin(angle)
                    px_norm = (px / 256.0) - 1.0
                    py_norm = (py / 256.0) - 1.0
                    points_3d.append([px_norm, py_norm, 0])
                    points_3d.append([px_norm, py_norm, height])
            
            else:  # line
                x1, y1 = np.random.randint(50, 450, 2)
                x2, y2 = np.random.randint(50, 450, 2)
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
                
                # Generate 3D points
                for t in np.linspace(0, 1, 30):
                    px = x1 + t * (x2 - x1)
                    py = y1 + t * (y2 - y1)
                    px_norm = (px / 256.0) - 1.0
                    py_norm = (py / 256.0) - 1.0
                    height = np.random.uniform(0.1, 0.5)
                    points_3d.append([px_norm, py_norm, height])
        
        # Save image
        cv2.imwrite(str(image_dir / f"synthetic_{i:04d}.png"), img)
        
        # Save point cloud
        points_3d = np.array(points_3d, dtype=np.float32)
        np.save(pc_dir / f"synthetic_{i:04d}.npy", points_3d)
        
        if (i+1) % 20 == 0:
            print(f"  Generated {i+1}/{num_samples} samples")
    
    print(f"✅ Generated {num_samples} synthetic samples in {output_dir}")


if __name__ == "__main__":
    print("Diffusion Model Trainer")
    print("="*70)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Generate synthetic data
    data_dir = Path("training_data/diffusion_synthetic")
    if not (data_dir / "images").exists():
        create_synthetic_training_data(data_dir, num_samples=200)
    
    # Create dataset
    dataset = CAD2D3DDataset(
        data_dir=data_dir,
        image_size=256,
        num_points=2048,
        augment=True
    )
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    
    # Create model
    model = create_diffusion_model(
        num_points=2048,
        timesteps=1000,
        device=device
    )
    
    # Create trainer
    trainer = DiffusionTrainer(
        model=model,
        device=device,
        learning_rate=1e-4
    )
    
    # Train
    print("\nStarting training...")
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=50,
        batch_size=4 if device == "cuda" else 2,
        save_every=10
    )
    
    print("\n✅ Training complete!")
