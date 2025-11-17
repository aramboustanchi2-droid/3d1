"""
Vision Transformer (ViT) Integration for CAD Drawing Analysis
یکپارچه‌سازی ویژن ترنسفورمر برای تحلیل نقشه‌های CAD

قابلیت‌ها:
- تحلیل رابطه بین اجزاء نقشه (Attention Mechanism)
- شناسایی الگوهای پیچیده و وابستگی‌های بلند-مدت
- درک ساختار کلی نقشه بهتر از CNN
- تشخیص دقیق‌تر عناصر در نقشه‌های شلوغ
"""

from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import json

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Tensor = Any
    nn = None
    print("⚠️ PyTorch not available for ViT")

try:
    from PIL import Image
    import cv2
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL/cv2 not available")


@dataclass
class ViTConfig:
    """تنظیمات Vision Transformer"""
    image_size: int = 512
    patch_size: int = 16  # هر patch 16x16
    num_channels: int = 3
    num_classes: int = 15  # تعداد کلاس‌های CAD
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1


class PatchEmbedding(nn.Module if TORCH_AVAILABLE else object):
    """تبدیل تصویر به Patch Embeddings"""
    
    def __init__(self, config: ViTConfig):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.patch_size = config.patch_size
        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, channels, height, width)
        Returns:
            (batch_size, num_patches, hidden_size)
        """
        x = self.projection(x)  # (B, hidden_size, H/P, W/P)
        x = x.flatten(2)  # (B, hidden_size, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, hidden_size)
        return x


class MultiHeadSelfAttention(nn.Module if TORCH_AVAILABLE else object):
    """Multi-Head Self-Attention برای تحلیل روابط"""
    
    def __init__(self, config: ViTConfig):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads
        
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
    
    def transpose_for_scores(self, x: Tensor) -> Tensor:
        """تبدیل شکل برای multi-head attention"""
        new_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_length, hidden_size)
        Returns:
            (batch_size, seq_length, hidden_size)
        """
        # محاسبه Q, K, V
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.head_size ** 0.5)
        
        # Attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Weighted sum
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        new_shape = context_layer.size()[:-2] + (self.num_heads * self.head_size,)
        context_layer = context_layer.view(*new_shape)
        
        output = self.output(context_layer)
        return output


class TransformerBlock(nn.Module if TORCH_AVAILABLE else object):
    """یک بلاک Transformer"""
    
    def __init__(self, config: ViTConfig):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.layernorm1 = nn.LayerNorm(config.hidden_size)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.layernorm2 = nn.LayerNorm(config.hidden_size)
    
    def forward(self, x: Tensor) -> Tensor:
        # Self-Attention + Residual
        attention_output = self.attention(x)
        x = self.layernorm1(x + attention_output)
        
        # MLP + Residual
        mlp_output = self.mlp(x)
        x = self.layernorm2(x + mlp_output)
        
        return x


class CADVisionTransformer(nn.Module if TORCH_AVAILABLE else object):
    """Vision Transformer برای تحلیل نقشه‌های CAD"""
    
    def __init__(self, config: ViTConfig):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.config = config
        
        # Patch Embedding
        self.patch_embedding = PatchEmbedding(config)
        
        # تعداد patches
        num_patches = (config.image_size // config.patch_size) ** 2
        
        # CLS token (برای classification)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        
        # Position Embeddings
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.hidden_size)
        )
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        self.layernorm = nn.LayerNorm(config.hidden_size)
        
        # Classification head
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        
        # Detection head (برای Object Detection)
        self.detection_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 4)  # bbox: x, y, w, h
        )
    
    def forward(self, pixel_values: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            pixel_values: (batch_size, channels, height, width)
        Returns:
            Dict with 'logits' and 'bbox_predictions'
        """
        batch_size = pixel_values.shape[0]
        
        # Patch Embeddings
        embeddings = self.patch_embedding(pixel_values)
        
        # اضافه کردن CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)
        
        # اضافه کردن Position Embeddings
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        
        # عبور از Transformer Blocks
        hidden_states = embeddings
        for block in self.blocks:
            hidden_states = block(hidden_states)
        
        hidden_states = self.layernorm(hidden_states)
        
        # CLS token output برای classification
        cls_output = hidden_states[:, 0]
        logits = self.classifier(cls_output)
        
        # Patch outputs برای detection
        patch_outputs = hidden_states[:, 1:]
        bbox_predictions = self.detection_head(patch_outputs)
        
        return {
            'logits': logits,
            'bbox_predictions': bbox_predictions,
            'hidden_states': hidden_states
        }


class CADViTDetector:
    """
    Wrapper برای استفاده از ViT در تشخیص نقشه‌های CAD
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[ViTConfig] = None,
        device: str = 'auto'
    ):
        """
        Args:
            model_path: مسیر مدل ذخیره شده
            config: تنظیمات مدل
            device: 'cpu', 'cuda', یا 'auto'
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for ViT")
        
        self.config = config or ViTConfig()
        
        # تعیین device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # ساخت مدل
        self.model = CADVisionTransformer(self.config)
        
        # بارگذاری weights
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # کلاس‌های CAD (15 discipline)
        self.class_names = [
            'wall', 'door', 'window', 'column', 'beam',
            'hvac', 'plumbing', 'electrical', 'furniture',
            'tree', 'parking', 'fire-alarm', 'elevator',
            'stair', 'dimension'
        ]
    
    def preprocess_image(self, image_path: str) -> Tensor:
        """پیش‌پردازش تصویر نقشه"""
        if not PIL_AVAILABLE:
            raise ImportError("PIL required")
        
        # بارگذاری تصویر
        image = Image.open(image_path).convert('RGB')
        
        # Resize
        image = image.resize((self.config.image_size, self.config.image_size))
        
        # به numpy
        image = np.array(image).astype(np.float32) / 255.0
        
        # Normalize (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # به tensor: (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # اضافه کردن batch dimension
        image = image.unsqueeze(0)
        
        return image.to(self.device)
    
    def detect(self, image_path: str, threshold: float = 0.5) -> List[Dict]:
        """
        تشخیص عناصر در نقشه با ViT
        
        Args:
            image_path: مسیر تصویر نقشه
            threshold: آستانه اطمینان
            
        Returns:
            لیست detection ها: [{'class': str, 'confidence': float, 'bbox': tuple}, ...]
        """
        # پیش‌پردازش
        image = self.preprocess_image(image_path)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(image)
        
        # پردازش نتایج
        logits = outputs['logits']
        bbox_predictions = outputs['bbox_predictions']
        
        # Softmax برای classification
        probs = F.softmax(logits, dim=-1)
        confidences, predicted_classes = torch.max(probs, dim=-1)
        
        detections = []
        
        # برای هر patch
        batch_size, num_patches, _ = bbox_predictions.shape
        for b in range(batch_size):
            for p in range(num_patches):
                confidence = confidences[b].item()
                if confidence > threshold:
                    class_idx = predicted_classes[b].item()
                    bbox = bbox_predictions[b, p].cpu().numpy()
                    
                    detection = {
                        'class': self.class_names[class_idx],
                        'confidence': confidence,
                        'bbox': tuple(bbox),  # (x, y, w, h)
                        'patch_id': p
                    }
                    detections.append(detection)
        
        return detections
    
    def analyze_relationships(self, image_path: str) -> Dict[str, Any]:
        """
        تحلیل روابط بین اجزاء نقشه با Attention Maps
        
        Returns:
            Dict با attention weights و روابط
        """
        image = self.preprocess_image(image_path)
        
        with torch.no_grad():
            outputs = self.model(image)
            hidden_states = outputs['hidden_states']
        
        # استخراج attention weights از آخرین layer
        # (این نیاز به تغییر TransformerBlock دارد تا attention weights را return کند)
        
        return {
            'hidden_states_shape': hidden_states.shape,
            'num_patches': hidden_states.shape[1] - 1,  # -1 برای CLS token
            'feature_dim': hidden_states.shape[2]
        }
    
    def save_model(self, path: str):
        """ذخیره مدل"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__
        }, path)
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path: str):
        """بارگذاری مدل"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from {path}")


def create_vit_for_cad(
    pretrained: bool = False,
    num_classes: int = 15,
    device: str = 'auto'
) -> CADViTDetector:
    """
    ساخت Vision Transformer برای CAD
    
    Args:
        pretrained: استفاده از weights پیش‌آموزش دیده
        num_classes: تعداد کلاس‌ها
        device: دستگاه محاسباتی
        
    Returns:
        CADViTDetector آماده استفاده
    """
    config = ViTConfig(num_classes=num_classes)
    detector = CADViTDetector(config=config, device=device)
    
    if pretrained:
        print("⚠️ Pretrained weights not available yet")
        print("   Train the model first using training_pipeline.py")
    
    return detector


# مثال استفاده
if __name__ == "__main__":
    if TORCH_AVAILABLE:
        print("\n" + "="*60)
        print("Vision Transformer for CAD Analysis")
        print("="*60)
        print("✅ Capabilities:")
        print("   - Attention-based relationship analysis")
        print("   - Long-range dependency modeling")
        print("   - Better understanding of complex layouts")
        print("   - Multi-scale feature extraction")
        print("\n✅ Architecture:")
        print("   - Patch size: 16x16")
        print("   - Hidden size: 768")
        print("   - Attention heads: 12")
        print("   - Transformer layers: 12")
        print("   - Parameters: ~86M")
        print("="*60)
        
        # ساخت مدل
        detector = create_vit_for_cad(num_classes=15)
        print(f"\n✅ Model created on {detector.device}")
    else:
        print("⚠️ PyTorch not available. Install: pip install torch torchvision")
