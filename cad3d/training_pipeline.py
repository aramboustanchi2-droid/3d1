"""
Neural Network Training Pipeline - پایپلاین آموزش شبکه‌های عصبی
Training برای:
- Object Detection (Faster R-CNN, YOLO)
- Semantic Segmentation (DeepLab, U-Net)
- Fine-tuning on CAD dataset
"""

from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import json

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.segmentation import deeplabv3_resnet101
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Placeholders when PyTorch not available
    nn = None
    optim = None
    Dataset = object  # Placeholder base class
    DataLoader = None
    transforms = None
    print("⚠️ PyTorch not available")

try:
    from PIL import Image
    import numpy as np
    import cv2
except ImportError:
    Image = None
    np = None
    cv2 = None
    print("⚠️ PIL/numpy/cv2 not available")


class CADDataset(Dataset):
    """Dataset برای CAD drawings در فرمت COCO"""
    
    def __init__(
        self,
        annotations_file: str,
        images_dir: str,
        transform=None,
        target_transform=None
    ):
        """
        Args:
            annotations_file: مسیر فایل COCO JSON
            images_dir: مسیر پوشه تصاویر
            transform: تبدیلات تصویر
            target_transform: تبدیلات target
        """
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.target_transform = target_transform
        
        # ساخت lookup dict
        self.image_id_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_anns:
                self.image_id_to_anns[img_id] = []
            self.image_id_to_anns[img_id].append(ann)
        
        print(f"📦 CAD Dataset loaded:")
        print(f"   Images: {len(self.coco_data['images'])}")
        print(f"   Annotations: {len(self.coco_data['annotations'])}")
        print(f"   Categories: {len(self.coco_data['categories'])}")
    
    def __len__(self):
        return len(self.coco_data['images'])
    
    def __getitem__(self, idx):
        # بارگذاری تصویر
        img_info = self.coco_data['images'][idx]
        img_path = self.images_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # بارگذاری annotations
        img_id = img_info['id']
        anns = self.image_id_to_anns.get(img_id, [])
        
        # تبدیل به PyTorch format
        boxes = []
        labels = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        
        if len(boxes) == 0:
            # تصویر بدون annotation
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id])
        }
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            target = self.target_transform(target)
        
        return image, target


class CADDetectionTrainer:
    """
    Trainer برای Object Detection روی CAD drawings
    """
    
    def __init__(
        self,
        num_classes: int = 15,
        device: str = "auto",
        learning_rate: float = 0.005,
        batch_size: int = 4
    ):
        """
        Args:
            num_classes: تعداد کلاس‌ها (بدون background)
            device: 'cpu', 'cuda', or 'auto'
            learning_rate: نرخ یادگیری
            batch_size: اندازه batch
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for training")
        
        # تعیین device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🎓 CAD Detection Trainer")
        print(f"   Device: {self.device}")
        print(f"   Classes: {num_classes}")
        print(f"   Learning Rate: {learning_rate}")
        print(f"   Batch Size: {batch_size}")
        
        self.num_classes = num_classes
        self.lr = learning_rate
        self.batch_size = batch_size
        
        # ساخت مدل
        self.model = self._build_model()
        self.optimizer = None
        self.lr_scheduler = None
    
    def _build_model(self):
        """ساخت مدل Faster R-CNN"""
        # بارگذاری pre-trained model
        model = fasterrcnn_resnet50_fpn_v2(pretrained=True)
        
        # تطبیق تعداد کلاس‌ها
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        
        # جایگزینی predictor
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features,
            self.num_classes + 1  # +1 for background
        )
        
        model.to(self.device)
        model.train()
        
        print("✅ Model built: Faster R-CNN ResNet50-FPN")
        return model
    
    def setup_optimizer(self, optimizer_type: str = "sgd"):
        """تنظیم optimizer"""
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        if optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                params,
                lr=self.lr,
                momentum=0.9,
                weight_decay=0.0005
            )
        elif optimizer_type == "adam":
            self.optimizer = optim.Adam(
                params,
                lr=self.lr,
                weight_decay=0.0005
            )
        
        # Learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=3,
            gamma=0.1
        )
        
        print(f"✅ Optimizer: {optimizer_type.upper()}")
    
    def train_epoch(
        self,
        dataloader: Any,
        epoch: int
    ) -> Dict[str, float]:
        """آموزش یک epoch"""
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        
        print(f"\n📚 Epoch {epoch}:")
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            # انتقال به device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            
            total_loss += losses.item()
            num_batches += 1
            
            # نمایش پیشرفت
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"   Batch [{batch_idx+1}/{len(dataloader)}] - Loss: {avg_loss:.4f}")
        
        # Update learning rate
        if self.lr_scheduler:
            self.lr_scheduler.step()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"   ✅ Epoch {epoch} complete - Avg Loss: {avg_loss:.4f}")
        
        return {'loss': avg_loss}
    
    def validate(
        self,
        dataloader: Any
    ) -> Dict[str, float]:
        """ارزیابی روی validation set"""
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        print("\n🔍 Validation:")
        
        with torch.no_grad():
            for images, targets in dataloader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                total_loss += losses.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"   Validation Loss: {avg_loss:.4f}")
        
        return {'val_loss': avg_loss}
    
    def train(
        self,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        epochs: int = 10,
        save_dir: Optional[Path] = None
    ):
        """
        Training loop کامل
        
        Args:
            train_dataset: Dataset آموزش
            val_dataset: Dataset validation (اختیاری)
            epochs: تعداد epoch ها
            save_dir: مسیر ذخیره checkpoints
        """
        # Setup optimizer
        if self.optimizer is None:
            self.setup_optimizer()
        
        # Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Windows compatibility
            collate_fn=self._collate_fn
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=self._collate_fn
            )
        
        # مسیر ذخیره
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n🚀 Starting training...")
        print(f"   Epochs: {epochs}")
        print(f"   Train samples: {len(train_dataset)}")
        if val_dataset:
            print(f"   Val samples: {len(val_dataset)}")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            if val_loader:
                val_metrics = self.validate(val_loader)
                
                # ذخیره بهترین مدل
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    if save_dir:
                        checkpoint_path = save_dir / "best_model.pth"
                        self.save_checkpoint(checkpoint_path, epoch, val_metrics)
                        print(f"   💾 Best model saved: {checkpoint_path.name}")
            
            # ذخیره checkpoint
            if save_dir and epoch % 5 == 0:
                checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pth"
                self.save_checkpoint(checkpoint_path, epoch, train_metrics)
        
        print("\n✅ Training complete!")
        if save_dir:
            print(f"   Models saved in: {save_dir}")
    
    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Dict
    ):
        """ذخیره checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'num_classes': self.num_classes
        }, path)
    
    def load_checkpoint(self, path: Path):
        """بارگذاری checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ Checkpoint loaded: {path.name} (Epoch {checkpoint['epoch']})")
        return checkpoint
    
    @staticmethod
    def _collate_fn(batch):
        """Custom collate function برای detection"""
        return tuple(zip(*batch))


# مثال استفاده
if __name__ == "__main__":
    print("🎓 Neural Network Training Pipeline - Demo")
    
    if TORCH_AVAILABLE:
        print("\n✅ PyTorch available")
        print("   Ready for training!")
        print("\nExample usage:")
        print("   trainer = CADDetectionTrainer(num_classes=15)")
        print("   trainer.train(train_dataset, val_dataset, epochs=10)")
    else:
        print("\n❌ PyTorch not available")
        print("   Install: pip install torch torchvision")
