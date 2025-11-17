"""
Real-world benchmarking examples for CAD detection models.
Demonstrates performance evaluation on actual architectural drawings.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cad3d.benchmark_suite import DetectionBenchmark, compare_models_benchmark
from cad3d.model_optimizer import ModelOptimizer, compare_models


def example_1_benchmark_trained_model():
    """
    Example 1: Benchmark a trained model on test dataset.
    
    Evaluates:
    - Detection accuracy (mAP, Precision, Recall)
    - Inference speed (FPS)
    - Per-category performance
    """
    print("="*70)
    print("Example 1: Benchmark Trained Model")
    print("="*70)
    
    try:
        import torch
        from torch.utils.data import DataLoader
        from cad3d.training_pipeline import CADDataset, CADDetectionTrainer
        
        # Configuration
        model_path = "./models/best_model.pth"
        dataset_dir = "./test_dataset"
        output_dir = "./benchmark_results"
        
        # Load model
        print("\n📂 Loading model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        trainer = CADDetectionTrainer(
            data_dir=".",
            output_dir=".",
            device=device
        )
        
        checkpoint = torch.load(model_path, map_location=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.model.eval()
        
        # Load dataset
        print("📂 Loading test dataset...")
        dataset = CADDataset(
            root_dir=dataset_dir,
            annotation_file=str(Path(dataset_dir) / "annotations.json")
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda x: tuple(zip(*x))
        )
        
        print(f"   Dataset size: {len(dataset)} images")
        
        # Run benchmark
        print("\n🚀 Running benchmark...")
        benchmark = DetectionBenchmark(trainer.model, device=str(device))
        
        overall_metrics, category_metrics = benchmark.evaluate_dataset(
            dataloader,
            max_samples=None  # Evaluate all samples
        )
        
        # Print detailed report
        benchmark.print_detailed_report(overall_metrics, category_metrics)
        
        # Save results
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        benchmark.save_results(
            str(Path(output_dir) / "benchmark_results.json"),
            overall_metrics,
            category_metrics
        )
        
        print(f"\n✅ Benchmark complete!")
        print(f"   Results saved to: {output_dir}")
        
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("This example requires PyTorch. Install with: pip install torch torchvision")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure model and dataset paths are correct.")


def example_2_compare_optimization_formats():
    """
    Example 2: Compare PyTorch vs ONNX vs Quantized models.
    
    Evaluates:
    - Speed improvement
    - Model size reduction
    - Accuracy preservation
    """
    print("\n" + "="*70)
    print("Example 2: Compare Optimization Formats")
    print("="*70)
    
    try:
        import torch
        from cad3d.training_pipeline import CADDetectionTrainer
        
        # Configuration
        model_path = "./models/best_model.pth"
        output_dir = "./optimized_models"
        
        # Load model
        print("\n📂 Loading PyTorch model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        trainer = CADDetectionTrainer(
            data_dir=".",
            output_dir=".",
            device=device
        )
        
        checkpoint = torch.load(model_path, map_location=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.model.eval()
        
        # Create optimizer
        print("\n⚡ Optimizing model...")
        optimizer = ModelOptimizer(device=str(device))
        
        # Run optimization pipeline
        results = optimizer.optimize_full_pipeline(
            model=trainer.model,
            output_dir=output_dir,
            input_shape=(1, 3, 1024, 1024),
            formats=["onnx", "quantized"],
            benchmark=True
        )
        
        # Compare results
        print("\n📊 Comparing models...")
        compare_models(results)
        
        # Print recommendations
        print("\n💡 Recommendations:")
        fastest = min(results, key=lambda r: r.inference_time_ms)
        smallest = min(results, key=lambda r: r.file_size_mb)
        
        print(f"   • For production deployment: {fastest.format}")
        print(f"     - Fastest inference: {fastest.inference_time_ms:.2f} ms")
        print(f"   • For edge devices: {smallest.format}")
        print(f"     - Smallest size: {smallest.file_size_mb:.2f} MB")
        
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Install dependencies: pip install torch onnx onnxruntime")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure model path is correct.")


def example_3_category_performance_analysis():
    """
    Example 3: Analyze per-category performance.
    
    Identifies:
    - Best performing categories
    - Categories needing improvement
    - Common failure patterns
    """
    print("\n" + "="*70)
    print("Example 3: Category Performance Analysis")
    print("="*70)
    
    try:
        import torch
        from torch.utils.data import DataLoader
        from cad3d.training_pipeline import CADDataset, CADDetectionTrainer
        import json
        
        # Configuration
        model_path = "./models/best_model.pth"
        dataset_dir = "./test_dataset"
        
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer = CADDetectionTrainer(data_dir=".", output_dir=".", device=device)
        checkpoint = torch.load(model_path, map_location=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.model.eval()
        
        # Load dataset
        dataset = CADDataset(
            root_dir=dataset_dir,
            annotation_file=str(Path(dataset_dir) / "annotations.json")
        )
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=0, collate_fn=lambda x: tuple(zip(*x))
        )
        
        # Run benchmark
        benchmark = DetectionBenchmark(trainer.model, device=str(device))
        overall_metrics, category_metrics = benchmark.evaluate_dataset(dataloader)
        
        # Analyze categories
        print("\n📊 Category Performance Analysis:")
        print("-" * 70)
        
        # Sort by AP (best to worst)
        sorted_categories = sorted(category_metrics, key=lambda m: m.ap, reverse=True)
        
        print("\n🥇 Top 5 Categories:")
        for i, metric in enumerate(sorted_categories[:5], 1):
            print(f"   {i}. {metric.category_name}: AP={metric.ap:.3f}, "
                  f"P={metric.precision:.3f}, R={metric.recall:.3f}")
        
        print("\n⚠️ Bottom 5 Categories (Need Improvement):")
        for i, metric in enumerate(sorted_categories[-5:], 1):
            print(f"   {i}. {metric.category_name}: AP={metric.ap:.3f}, "
                  f"P={metric.precision:.3f}, R={metric.recall:.3f}")
            
            # Suggest improvements
            if metric.precision < 0.5:
                print(f"      → Too many false positives - increase confidence threshold")
            if metric.recall < 0.5:
                print(f"      → Missing detections - add more training data")
            if metric.num_ground_truth < 10:
                print(f"      → Insufficient test samples ({metric.num_ground_truth})")
        
        # Save detailed analysis
        analysis = {
            "overall": {
                "mAP": overall_metrics.mAP,
                "avg_precision": sum(m.precision for m in category_metrics) / len(category_metrics),
                "avg_recall": sum(m.recall for m in category_metrics) / len(category_metrics)
            },
            "top_categories": [
                {
                    "name": m.category_name,
                    "ap": m.ap,
                    "precision": m.precision,
                    "recall": m.recall
                }
                for m in sorted_categories[:5]
            ],
            "problematic_categories": [
                {
                    "name": m.category_name,
                    "ap": m.ap,
                    "precision": m.precision,
                    "recall": m.recall,
                    "issues": []
                }
                for m in sorted_categories[-5:]
            ]
        }
        
        output_file = "./benchmark_results/category_analysis.json"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Analysis saved to: {output_file}")
        
    except ImportError as e:
        print(f"❌ Error: {e}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")


def example_4_speed_vs_accuracy_tradeoff():
    """
    Example 4: Analyze speed vs accuracy tradeoff.
    
    Tests different:
    - Confidence thresholds
    - Input resolutions
    - Model formats (PyTorch, ONNX, TensorRT)
    """
    print("\n" + "="*70)
    print("Example 4: Speed vs Accuracy Tradeoff")
    print("="*70)
    
    try:
        import torch
        from torch.utils.data import DataLoader
        from cad3d.training_pipeline import CADDataset, CADDetectionTrainer
        import time
        
        model_path = "./models/best_model.pth"
        dataset_dir = "./test_dataset"
        
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer = CADDetectionTrainer(data_dir=".", output_dir=".", device=device)
        checkpoint = torch.load(model_path, map_location=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.model.eval()
        
        # Load dataset
        dataset = CADDataset(
            root_dir=dataset_dir,
            annotation_file=str(Path(dataset_dir) / "annotations.json")
        )
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=0, collate_fn=lambda x: tuple(zip(*x))
        )
        
        # Test different confidence thresholds
        print("\n📊 Testing Confidence Thresholds:")
        print("-" * 70)
        print(f"{'Threshold':<12} {'mAP':<8} {'Precision':<10} {'Recall':<10} {'FPS':<8}")
        print("-" * 70)
        
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        results = []
        
        for threshold in thresholds:
            benchmark = DetectionBenchmark(
                trainer.model,
                device=str(device),
                confidence_threshold=threshold
            )
            
            overall_metrics, _ = benchmark.evaluate_dataset(dataloader, max_samples=50)
            
            results.append({
                'threshold': threshold,
                'mAP': overall_metrics.mAP,
                'precision': overall_metrics.precision,
                'recall': overall_metrics.recall,
                'fps': overall_metrics.fps
            })
            
            print(f"{threshold:<12.1f} {overall_metrics.mAP:<8.3f} "
                  f"{overall_metrics.precision:<10.3f} {overall_metrics.recall:<10.3f} "
                  f"{overall_metrics.fps:<8.1f}")
        
        # Find optimal threshold
        optimal = max(results, key=lambda r: r['mAP'])
        print(f"\n💡 Optimal threshold: {optimal['threshold']:.1f} "
              f"(mAP: {optimal['mAP']:.3f})")
        
        # Test different input sizes
        print("\n📊 Testing Input Resolutions:")
        print("-" * 70)
        print(f"{'Resolution':<15} {'Inference Time (ms)':<20} {'Throughput (FPS)':<18}")
        print("-" * 70)
        
        resolutions = [(512, 512), (768, 768), (1024, 1024), (1280, 1280)]
        
        for res in resolutions:
            # Create dummy input
            dummy_input = torch.randn(1, 3, res[0], res[1]).to(device)
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = trainer.model(dummy_input)
            
            # Measure
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            start = time.time()
            for _ in range(50):
                with torch.no_grad():
                    _ = trainer.model(dummy_input)
                if device.type == "cuda":
                    torch.cuda.synchronize()
            end = time.time()
            
            avg_time = ((end - start) * 1000) / 50
            fps = 1000.0 / avg_time
            
            print(f"{res[0]}x{res[1]:<10} {avg_time:<20.2f} {fps:<18.1f}")
        
        print("\n💡 Recommendation:")
        print("   • For real-time processing (>30 FPS): Use 512x512 or 768x768")
        print("   • For high accuracy: Use 1024x1024 or 1280x1280")
        print("   • Optimal balance: 1024x1024 with confidence threshold 0.5")
        
    except ImportError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


def example_5_baseline_comparison():
    """
    Example 5: Compare custom model against baseline.
    
    Compares:
    - Pre-trained COCO model (baseline)
    - Fine-tuned CAD model (custom)
    - Domain adaptation improvement
    """
    print("\n" + "="*70)
    print("Example 5: Baseline Comparison")
    print("="*70)
    
    print("\n📊 Comparing Models:")
    print("-" * 70)
    
    try:
        import torch
        from torch.utils.data import DataLoader
        from cad3d.training_pipeline import CADDataset, CADDetectionTrainer
        
        dataset_dir = "./test_dataset"
        
        # Load dataset
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = CADDataset(
            root_dir=dataset_dir,
            annotation_file=str(Path(dataset_dir) / "annotations.json")
        )
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=0, collate_fn=lambda x: tuple(zip(*x))
        )
        
        models_to_compare = [
            ("Baseline (Pre-trained COCO)", "./models/pretrained_coco.pth"),
            ("Fine-tuned (CAD Dataset)", "./models/best_model.pth"),
        ]
        
        results = {}
        
        for model_name, model_path in models_to_compare:
            if not Path(model_path).exists():
                print(f"⚠️ Skipping {model_name}: File not found")
                continue
            
            print(f"\n📦 Evaluating: {model_name}")
            
            # Load model
            trainer = CADDetectionTrainer(data_dir=".", output_dir=".", device=device)
            checkpoint = torch.load(model_path, map_location=device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.model.eval()
            
            # Benchmark
            benchmark = DetectionBenchmark(trainer.model, device=str(device))
            overall_metrics, category_metrics = benchmark.evaluate_dataset(
                dataloader, max_samples=100
            )
            
            results[model_name] = overall_metrics
            
            print(f"   mAP: {overall_metrics.mAP:.3f}")
            print(f"   Precision: {overall_metrics.precision:.3f}")
            print(f"   Recall: {overall_metrics.recall:.3f}")
            print(f"   FPS: {overall_metrics.fps:.1f}")
        
        # Calculate improvement
        if len(results) == 2:
            baseline_name, custom_name = list(results.keys())
            baseline = results[baseline_name]
            custom = results[custom_name]
            
            print("\n📈 Improvement Analysis:")
            print("-" * 70)
            
            map_improvement = ((custom.mAP - baseline.mAP) / baseline.mAP) * 100
            precision_improvement = ((custom.precision - baseline.precision) / baseline.precision) * 100
            recall_improvement = ((custom.recall - baseline.recall) / baseline.recall) * 100
            
            print(f"   mAP improvement: {map_improvement:+.1f}%")
            print(f"   Precision improvement: {precision_improvement:+.1f}%")
            print(f"   Recall improvement: {recall_improvement:+.1f}%")
            
            print("\n💡 Conclusion:")
            if map_improvement > 10:
                print("   ✅ Fine-tuning significantly improved CAD detection!")
            elif map_improvement > 0:
                print("   ✅ Fine-tuning provided modest improvements.")
            else:
                print("   ⚠️ Fine-tuning did not improve performance - review training data.")
        
    except ImportError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run all benchmark examples."""
    print("\n" + "="*70)
    print("🚀 REAL-WORLD BENCHMARK EXAMPLES")
    print("="*70)
    
    print("\nThese examples demonstrate how to benchmark CAD detection models")
    print("on real-world architectural drawings.")
    print("\nNote: Examples require trained models and test datasets.")
    print("      Set up your paths in each example function.")
    
    examples = [
        ("Benchmark Trained Model", example_1_benchmark_trained_model),
        ("Compare Optimization Formats", example_2_compare_optimization_formats),
        ("Category Performance Analysis", example_3_category_performance_analysis),
        ("Speed vs Accuracy Tradeoff", example_4_speed_vs_accuracy_tradeoff),
        ("Baseline Comparison", example_5_baseline_comparison),
    ]
    
    print("\n📚 Available Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"   {i}. {name}")
    
    print("\n" + "="*70)
    
    # Run all examples
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*70)


if __name__ == "__main__":
    main()
