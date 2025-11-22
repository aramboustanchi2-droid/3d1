"""
Vision Transformer Training System
سیستم آموزش Vision Transformer برای نقشه‌های CAD

Features:
- Custom dataset loader for CAD drawings
- Training loop with validation
- Data augmentation for drawings
- Metric tracking and logging
- Model checkpointing
- Transfer learning support
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import json
import time

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .vision_transformer_cad import VisionTransformerCAD, VisionTransformerConfig


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Data
    train_data_dir: str = "data/train"
    val_data_dir: str = "data/val"
    batch_size: int = 8
    num_workers: int = 4
    
    # Training
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    
    # Optimizer
    optimizer: str = "adamw"  # adamw, adam, sgd
    scheduler: str = "cosine"  # cosine, step, plateau
    
    # Loss weights
    semantic_weight: float = 1.0
    depth_weight: float = 0.5
    height_weight: float = 0.5
    material_weight: float = 0.3
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5  # epochs
    keep_best: bool = True
    
    # Device
    device: str = "cuda"  # cuda, cpu, auto
    mixed_precision: bool = True
    
    # Logging
    log_every: int = 10  # batches
    validate_every: int = 1  # epochs


class CADDataset(Dataset):
    """
    Dataset for CAD drawings with annotations
    
    Expected structure:
    data/
      train/
        images/
          drawing_001.png
          drawing_002.png
        annotations/
          drawing_001.json
          drawing_002.json
      val/
        images/
        annotations/
    
    Annotation format (JSON):
    {
      "semantic_map": [[0, 1, 1, ...], ...],  # 2D array of class indices
      "height_map": [[0, 3000, 3000, ...], ...],  # 2D array of heights in mm
      "depth_map": [[0, 0.5, 0.3, ...], ...],  # 2D array of normalized depths
      "material_map": [[0, 1, 1, ...], ...],  # 2D array of material indices
      "metadata": {
        "scale": 10.0,  # mm per pixel
        "drawing_type": "architectural",
        "units": "mm"
      }
    }
    """
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 512,
        augment: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augment = augment
        
        # Find all images
        self.image_dir = self.data_dir / "images"
        self.annotation_dir = self.data_dir / "annotations"
        
        self.image_files = sorted(self.image_dir.glob("*.png")) + \
                          sorted(self.image_dir.glob("*.jpg"))
        
        print(f"Found {len(self.image_files)} images in {data_dir}")
        
        # Transforms
        self.basic_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        import cv2
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotation
        annotation_path = self.annotation_dir / f"{image_path.stem}.json"
        
        if annotation_path.exists():
            with open(annotation_path) as f:
                annotation = json.load(f)
        else:
            # Create dummy annotation if none exists
            annotation = self._create_dummy_annotation()
        
        # Apply transforms
        image_tensor = self.basic_transform(image)
        
        # Prepare targets
        targets = {}
        
        # Semantic segmentation
        semantic_map = np.array(annotation.get('semantic_map', []))
        if semantic_map.size > 0:
            semantic_map = cv2.resize(
                semantic_map.astype(np.float32),
                (32, 32),  # Grid size based on patch size
                interpolation=cv2.INTER_NEAREST
            )
            targets['semantic'] = torch.from_numpy(semantic_map.flatten()).long()
        
        # Height map
        height_map = np.array(annotation.get('height_map', []))
        if height_map.size > 0:
            height_map = cv2.resize(height_map, (32, 32), interpolation=cv2.INTER_LINEAR)
            # Normalize heights to [0, 1] range
            height_map = height_map / 10000.0  # Assume max 10m
            targets['height'] = torch.from_numpy(height_map.flatten()).float().unsqueeze(-1)
        
        # Depth map
        depth_map = np.array(annotation.get('depth_map', []))
        if depth_map.size > 0:
            depth_map = cv2.resize(depth_map, (32, 32), interpolation=cv2.INTER_LINEAR)
            targets['depth'] = torch.from_numpy(depth_map.flatten()).float().unsqueeze(-1)
        
        # Material map
        material_map = np.array(annotation.get('material_map', []))
        if material_map.size > 0:
            material_map = cv2.resize(
                material_map.astype(np.float32),
                (32, 32),
                interpolation=cv2.INTER_NEAREST
            )
            targets['material'] = torch.from_numpy(material_map.flatten()).long()
        
        return image_tensor, targets
    
    def _create_dummy_annotation(self):
        """Create dummy annotation for testing"""
        size = 32
        return {
            'semantic_map': np.zeros((size, size)),
            'height_map': np.zeros((size, size)),
            'depth_map': np.zeros((size, size)),
            'material_map': np.zeros((size, size))
        }


class VisionTransformerTrainer:
    """
    Trainer for Vision Transformer CAD models
    """
    
    def __init__(
        self,
        model_config: VisionTransformerConfig,
        train_config: TrainingConfig
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for training")
        
        self.model_config = model_config
        self.config = train_config
        
        # Setup device
        if train_config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(train_config.device)
        
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = VisionTransformerCAD(model_config).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model: {total_params:,} total parameters, {trainable_params:,} trainable")
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Create scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if train_config.mixed_precision else None
        
        # Tracking
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Create checkpoint directory
        Path(train_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def _create_optimizer(self):
        """Create optimizer"""
        if self.config.optimizer == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=10,
                factor=0.5
            )
        else:
            return None
    
    def compute_loss(self, outputs: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict]:
        """Compute combined loss"""
        losses = {}
        total_loss = 0
        
        # Semantic segmentation loss
        if 'semantic' in outputs and 'semantic' in targets:
            semantic_loss = F.cross_entropy(
                outputs['semantic'].view(-1, self.model_config.num_classes),
                targets['semantic'].view(-1)
            )
            losses['semantic'] = semantic_loss.item()
            total_loss += self.config.semantic_weight * semantic_loss
        
        # Height prediction loss
        if 'height' in outputs and 'height' in targets:
            height_loss = F.mse_loss(outputs['height'], targets['height'])
            losses['height'] = height_loss.item()
            total_loss += self.config.height_weight * height_loss
        
        # Depth prediction loss
        if 'depth' in outputs and 'depth' in targets:
            depth_loss = F.mse_loss(outputs['depth'], targets['depth'])
            losses['depth'] = depth_loss.item()
            total_loss += self.config.depth_weight * depth_loss
        
        # Material classification loss
        if 'material' in outputs and 'material' in targets:
            material_loss = F.cross_entropy(
                outputs['material'].view(-1, 10),
                targets['material'].view(-1)
            )
            losses['material'] = material_loss.item()
            total_loss += self.config.material_weight * material_loss
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses
    
    def train_epoch(self, dataloader: DataLoader) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = []
        epoch_start = time.time()
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            # Move to device
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # Forward pass
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss, losses = self.compute_loss(outputs, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss, losses = self.compute_loss(outputs, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            epoch_losses.append(losses)
            
            # Logging
            if batch_idx % self.config.log_every == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}: Loss = {losses['total']:.4f}")
        
        # Compute average losses
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = np.mean([l[key] for l in epoch_losses])
        
        epoch_time = time.time() - epoch_start
        avg_losses['time'] = epoch_time
        
        return avg_losses
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict:
        """Validate model"""
        self.model.eval()
        
        val_losses = []
        
        for images, targets in dataloader:
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            outputs = self.model(images)
            _, losses = self.compute_loss(outputs, targets)
            
            val_losses.append(losses)
        
        # Average losses
        avg_losses = {}
        for key in val_losses[0].keys():
            avg_losses[key] = np.mean([l[key] for l in val_losses])
        
        return avg_losses
    
    def train(
        self,
        train_dataset: CADDataset,
        val_dataset: Optional[CADDataset] = None
    ):
        """Full training loop"""
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True
            )
        
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        print(f"Train batches: {len(train_loader)}")
        if val_loader:
            print(f"Val batches: {len(val_loader)}")
        print(f"Epochs: {self.config.num_epochs}")
        print("="*60 + "\n")
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            # Train
            train_losses = self.train_epoch(train_loader)
            print(f"  Train Loss: {train_losses['total']:.4f} (time: {train_losses['time']:.1f}s)")
            
            self.history['train_loss'].append(train_losses['total'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Validate
            if val_loader and (epoch + 1) % self.config.validate_every == 0:
                val_losses = self.validate(val_loader)
                print(f"  Val Loss: {val_losses['total']:.4f}")
                self.history['val_loss'].append(val_losses['total'])
                
                # Check if best model
                if val_losses['total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    if self.config.keep_best:
                        self.save_checkpoint("best_model.pth")
                        print(f"  ✓ Best model saved (val_loss: {self.best_val_loss:.4f})")
            
            # Step scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loader:
                        self.scheduler.step(val_losses['total'])
                else:
                    self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")
                print(f"  ✓ Checkpoint saved")
            
            print()
        
        print("Training complete!")
        self.save_checkpoint("final_model.pth")
        self.save_history("training_history.json")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_path = Path(self.config.checkpoint_dir) / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'model_config': self.model_config.__dict__,
            'train_config': self.config.__dict__
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint_path = Path(self.config.checkpoint_dir) / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def save_history(self, filename: str):
        """Save training history"""
        history_path = Path(self.config.checkpoint_dir) / filename
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")


def main():
    """Example training script"""
    if not TORCH_AVAILABLE:
        print("PyTorch not available")
        return
    
    # Create configs
    model_config = VisionTransformerConfig(
        image_size=512,
        patch_size=16,
        num_classes=50,
        dim=768,
        depth=12,
        heads=12
    )
    
    train_config = TrainingConfig(
        train_data_dir="data/train",
        val_data_dir="data/val",
        batch_size=4,
        num_epochs=50,
        learning_rate=1e-4,
        device="auto"
    )
    
    # Create datasets
    train_dataset = CADDataset(train_config.train_data_dir, augment=True)
    val_dataset = CADDataset(train_config.val_data_dir, augment=False)
    
    # Create trainer
    trainer = VisionTransformerTrainer(model_config, train_config)
    
    # Train
    trainer.train(train_dataset, val_dataset)


if __name__ == "__main__":
    main()
