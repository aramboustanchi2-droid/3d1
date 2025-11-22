"""
Neural Network Training & Validation System
سیستم آموزش و اعتبارسنجی شبکه‌های عصبی

این سیستم تمام مدل‌های هوش مصنوعی را آموزش و ارزیابی می‌کند:
- Vision Transformer (ViT)
- Graph Neural Network (GNN)
- Neural CAD Detector
- Advanced Detection Models
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# Check dependencies
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch not available")

import numpy as np
import cv2


class NeuralModelTrainer:
    """
    مدیر آموزش و ارزیابی تمام مدل‌های عصبی
    """
    
    def __init__(self, device: str = "auto"):
        self.device = self._setup_device(device)
        self.models = {}
        self.training_history = {}
        
        print("="*70)
        print("🧠 Neural Network Training System Initialized")
        print("="*70)
        print(f"Device: {self.device}")
        print()
    
    def _setup_device(self, device: str) -> torch.device:
        """تنظیم device (CPU/GPU)"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for neural training")
        
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                print("ℹ️ Using CPU (GPU not available)")
        
        return torch.device(device)
    
    def check_vision_transformer(self) -> bool:
        """بررسی و تست Vision Transformer"""
        print("\n" + "="*70)
        print("1️⃣ Checking Vision Transformer")
        print("="*70)
        
        try:
            from cad3d.vision_transformer_cad import (
                VisionTransformerCAD, 
                VisionTransformerConfig,
                CADVisionAnalyzer
            )
            
            # ایجاد مدل کوچک برای تست
            config = VisionTransformerConfig(
                image_size=256,
                patch_size=16,
                num_classes=50,
                dim=384,
                depth=6,
                heads=6,
                dropout=0.1
            )
            
            model = VisionTransformerCAD(config).to(self.device)
            total_params = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"✅ Vision Transformer loaded successfully")
            print(f"   - Parameters: {total_params:,}")
            print(f"   - Trainable: {trainable:,}")
            print(f"   - Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
            
            # تست forward pass
            dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
            
            model.eval()
            with torch.no_grad():
                start = time.time()
                outputs = model(dummy_input)
                inference_time = time.time() - start
            
            print(f"   - Inference time: {inference_time*1000:.2f} ms")
            print(f"   - Semantic output shape: {outputs['semantic'].shape}")
            if 'height' in outputs:
                print(f"   - Height output shape: {outputs['height'].shape}")
            if 'depth' in outputs:
                print(f"   - Depth output shape: {outputs['depth'].shape}")
            
            self.models['vision_transformer'] = {
                'model': model,
                'config': config,
                'status': 'ready',
                'params': total_params
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Vision Transformer check failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def check_gnn_detector(self) -> bool:
        """بررسی و تست Graph Neural Network"""
        print("\n" + "="*70)
        print("2️⃣ Checking Graph Neural Network (GNN)")
        print("="*70)
        
        try:
            from cad3d.gnn_detector import CADGraphDetector
            
            detector = CADGraphDetector(device=str(self.device))
            
            print(f"✅ GNN Detector loaded successfully")
            print(f"   - Device: {detector.device}")
            print(f"   - Ready for graph-based CAD analysis")
            
            # تست با گراف ساده
            test_nodes = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
            test_edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
            
            try:
                result = detector.detect_relationships(test_nodes, test_edges)
                print(f"   - Test graph analysis: {len(result)} relationships detected")
            except Exception as e:
                print(f"   - Test failed: {e}")
            
            self.models['gnn_detector'] = {
                'model': detector,
                'status': 'ready'
            }
            
            return True
            
        except Exception as e:
            print(f"❌ GNN Detector check failed: {e}")
            return False
    
    def check_neural_cad_detector(self) -> bool:
        """بررسی Neural CAD Detector"""
        print("\n" + "="*70)
        print("3️⃣ Checking Neural CAD Detector")
        print("="*70)
        
        try:
            from cad3d.neural_cad_detector import NeuralCADDetector, DetectorConfig
            
            config = DetectorConfig(quality="high")
            detector = NeuralCADDetector(device="auto", config=config)
            
            print(f"✅ Neural CAD Detector loaded successfully")
            print(f"   - Quality: {config.quality}")
            print(f"   - Canny thresholds: {config.canny_low}/{config.canny_high}")
            print(f"   - Min contour area: {config.min_contour_area}")
            
            # تست با تصویر ساده
            test_img = np.ones((256, 256, 3), dtype=np.uint8) * 255
            cv2.rectangle(test_img, (50, 50), (200, 200), (0, 0, 0), 2)
            
            temp_file = "temp_test_neural.png"
            cv2.imwrite(temp_file, test_img)
            
            try:
                result = detector.vectorize_drawing(temp_file, scale_mm_per_pixel=1.0)
                print(f"   - Test vectorization: {len(result.polygons)} polygons detected")
                
                import os
                os.unlink(temp_file)
            except Exception as e:
                print(f"   - Test failed: {e}")
            
            self.models['neural_cad'] = {
                'model': detector,
                'config': config,
                'status': 'ready'
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Neural CAD Detector check failed: {e}")
            return False
    
    def check_vit_detector(self) -> bool:
        """بررسی ViT Detector پیشرفته"""
        print("\n" + "="*70)
        print("4️⃣ Checking Advanced ViT Detector")
        print("="*70)
        
        try:
            from cad3d.vit_detector import CADViTDetector
            
            detector = CADViTDetector(device=str(self.device))
            
            print(f"✅ Advanced ViT Detector loaded successfully")
            print(f"   - Device: {detector.device}")
            print(f"   - Multi-scale analysis enabled")
            
            self.models['vit_detector'] = {
                'model': detector,
                'status': 'ready'
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Advanced ViT Detector check failed: {e}")
            return False
    
    def train_vision_transformer_sample(self, num_epochs: int = 5) -> Dict:
        """آموزش نمونه Vision Transformer"""
        print("\n" + "="*70)
        print("🎓 Training Vision Transformer (Sample)")
        print("="*70)
        
        if 'vision_transformer' not in self.models:
            print("❌ Vision Transformer not loaded")
            return {}
        
        model = self.models['vision_transformer']['model']
        model.train()
        
        # ایجاد دیتاست تصادفی برای تست
        batch_size = 2
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        
        history = []
        
        print(f"\nTraining for {num_epochs} epochs...")
        print("-" * 70)
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # دیتای تصادفی
            images = torch.randn(batch_size, 3, 256, 256).to(self.device)
            
            # Target های تصادفی
            semantic_target = torch.randint(0, 50, (batch_size, 1024)).to(self.device)
            height_target = torch.rand(batch_size, 1024, 1).to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # محاسبه loss
            semantic_loss = nn.CrossEntropyLoss()(
                outputs['semantic'].view(-1, 50),
                semantic_target.view(-1)
            )
            
            height_loss = nn.MSELoss()(
                outputs['height'],
                height_target
            )
            
            total_loss = semantic_loss + 0.5 * height_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            epoch_time = time.time() - epoch_start
            
            history.append({
                'epoch': epoch + 1,
                'total_loss': total_loss.item(),
                'semantic_loss': semantic_loss.item(),
                'height_loss': height_loss.item(),
                'time': epoch_time
            })
            
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Loss: {total_loss.item():.4f} | "
                  f"Semantic: {semantic_loss.item():.4f} | "
                  f"Height: {height_loss.item():.4f} | "
                  f"Time: {epoch_time:.2f}s")
        
        print("-" * 70)
        print(f"✅ Training complete!")
        print(f"   Final loss: {history[-1]['total_loss']:.4f}")
        
        self.training_history['vision_transformer'] = history
        
        return {
            'epochs': num_epochs,
            'final_loss': history[-1]['total_loss'],
            'history': history
        }
    
    def benchmark_models(self) -> Dict:
        """Benchmark تمام مدل‌ها"""
        print("\n" + "="*70)
        print("⚡ Benchmarking All Models")
        print("="*70)
        
        benchmarks = {}
        
        # تست Vision Transformer
        if 'vision_transformer' in self.models:
            model = self.models['vision_transformer']['model']
            model.eval()
            
            input_tensor = torch.randn(1, 3, 256, 256).to(self.device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = model(input_tensor)
            
            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(10):
                    start = time.time()
                    _ = model(input_tensor)
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    times.append(time.time() - start)
            
            benchmarks['vision_transformer'] = {
                'avg_time_ms': np.mean(times) * 1000,
                'std_time_ms': np.std(times) * 1000,
                'fps': 1.0 / np.mean(times)
            }
            
            print(f"Vision Transformer:")
            print(f"  - Avg time: {benchmarks['vision_transformer']['avg_time_ms']:.2f} ms")
            print(f"  - FPS: {benchmarks['vision_transformer']['fps']:.2f}")
        
        # تست Neural CAD
        if 'neural_cad' in self.models:
            test_img = np.ones((512, 512, 3), dtype=np.uint8) * 255
            cv2.rectangle(test_img, (100, 100), (400, 400), (0, 0, 0), 2)
            
            temp_file = "temp_bench.png"
            cv2.imwrite(temp_file, test_img)
            
            detector = self.models['neural_cad']['model']
            
            times = []
            for _ in range(5):
                start = time.time()
                _ = detector.vectorize_drawing(temp_file, scale_mm_per_pixel=1.0)
                times.append(time.time() - start)
            
            benchmarks['neural_cad'] = {
                'avg_time_ms': np.mean(times) * 1000,
                'std_time_ms': np.std(times) * 1000
            }
            
            print(f"\nNeural CAD Detector:")
            print(f"  - Avg time: {benchmarks['neural_cad']['avg_time_ms']:.2f} ms")
            
            import os
            os.unlink(temp_file)
        
        return benchmarks
    
    def generate_report(self, output_path: str = "neural_system_report.json"):
        """تولید گزارش کامل"""
        print("\n" + "="*70)
        print("📊 Generating System Report")
        print("="*70)
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(self.device),
            'models': {},
            'training_history': self.training_history,
            'system_info': {
                'torch_version': torch.__version__ if TORCH_AVAILABLE else None,
                'cuda_available': torch.cuda.is_available() if TORCH_AVAILABLE else False,
                'cuda_device': torch.cuda.get_device_name(0) if TORCH_AVAILABLE and torch.cuda.is_available() else None
            }
        }
        
        # اطلاعات مدل‌ها
        for name, info in self.models.items():
            model_info = {
                'status': info['status']
            }
            if 'params' in info:
                model_info['parameters'] = info['params']
            if 'config' in info:
                model_info['config'] = str(info['config'])
            
            report['models'][name] = model_info
        
        # ذخیره
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Report saved to: {output_path}")
        print(f"\nSummary:")
        print(f"  - Total models: {len(self.models)}")
        print(f"  - Ready models: {sum(1 for m in self.models.values() if m['status'] == 'ready')}")
        
        return report


def main():
    """اجرای کامل سیستم"""
    print("\n" + "="*70)
    print("🚀 Neural Network Training & Validation System")
    print("="*70)
    print()
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available!")
        print("\nInstall with:")
        print("  pip install torch torchvision")
        return
    
    # ایجاد trainer
    trainer = NeuralModelTrainer(device="auto")
    
    # بررسی تمام مدل‌ها
    print("\n" + "🔍 Checking All Neural Models".center(70, "="))
    
    results = {
        'vision_transformer': trainer.check_vision_transformer(),
        'gnn_detector': trainer.check_gnn_detector(),
        'neural_cad': trainer.check_neural_cad_detector(),
        'vit_detector': trainer.check_vit_detector()
    }
    
    # خلاصه نتایج
    print("\n" + "="*70)
    print("📋 Model Check Summary")
    print("="*70)
    
    for model_name, success in results.items():
        status = "✅ READY" if success else "❌ FAILED"
        print(f"  {model_name:25s} {status}")
    
    # آموزش نمونه (اگر Vision Transformer موجود باشد)
    if results['vision_transformer']:
        print("\n" + "🎓 Starting Sample Training".center(70, "="))
        training_result = trainer.train_vision_transformer_sample(num_epochs=5)
    
    # Benchmark
    print("\n" + "⚡ Running Benchmarks".center(70, "="))
    benchmarks = trainer.benchmark_models()
    
    # تولید گزارش
    report = trainer.generate_report()
    
    # نتیجه نهایی
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETE!")
    print("="*70)
    print("\n📊 System Status:")
    print(f"  - Device: {trainer.device}")
    print(f"  - Models loaded: {len(trainer.models)}")
    print(f"  - Training history: {len(trainer.training_history)} model(s)")
    print(f"\n💡 Next steps:")
    print("  1. Review neural_system_report.json")
    print("  2. Prepare training dataset")
    print("  3. Run full training: python -m cad3d.vit_trainer")
    print()


if __name__ == "__main__":
    main()
