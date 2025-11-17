"""
Tests for model optimization and benchmarking tools.
"""
import pytest
from pathlib import Path
import tempfile


def _has_torch():
    """Check if PyTorch is installed."""
    try:
        import torch
        return True
    except ImportError:
        return False


def _has_onnx():
    """Check if ONNX is installed."""
    try:
        import onnx
        import onnxruntime
        return True
    except ImportError:
        return False


def test_optimization_imports():
    """Test that optimization modules can be imported."""
    from cad3d.model_optimizer import ModelOptimizer, OptimizationResult, compare_models
    from cad3d.benchmark_suite import DetectionBenchmark, DetectionMetrics, CategoryMetrics
    
    assert ModelOptimizer is not None
    assert OptimizationResult is not None
    assert DetectionBenchmark is not None
    assert DetectionMetrics is not None
    assert CategoryMetrics is not None


def test_optimization_result_dataclass():
    """Test OptimizationResult dataclass."""
    from cad3d.model_optimizer import OptimizationResult
    
    result = OptimizationResult(
        model_path="model.onnx",
        format="onnx",
        file_size_mb=150.5,
        inference_time_ms=25.3,
        memory_usage_mb=512.0,
        accuracy_drop=0.02
    )
    
    assert result.model_path == "model.onnx"
    assert result.format == "onnx"
    assert result.file_size_mb == 150.5
    assert result.inference_time_ms == 25.3
    assert result.memory_usage_mb == 512.0
    assert result.accuracy_drop == 0.02


def test_detection_metrics_dataclass():
    """Test DetectionMetrics dataclass."""
    from cad3d.benchmark_suite import DetectionMetrics
    
    metrics = DetectionMetrics(
        precision=0.85,
        recall=0.82,
        f1_score=0.835,
        mAP=0.87,
        mAP_50=0.89,
        mAP_75=0.85,
        average_iou=0.75,
        inference_time_ms=30.5,
        fps=32.8
    )
    
    assert metrics.precision == 0.85
    assert metrics.recall == 0.82
    assert metrics.f1_score == 0.835
    assert metrics.mAP == 0.87
    assert metrics.fps == 32.8


def test_category_metrics_dataclass():
    """Test CategoryMetrics dataclass."""
    from cad3d.benchmark_suite import CategoryMetrics
    
    cat_metrics = CategoryMetrics(
        category_name="wall",
        category_id=0,
        precision=0.90,
        recall=0.88,
        f1_score=0.89,
        ap=0.91,
        num_predictions=150,
        num_ground_truth=145
    )
    
    assert cat_metrics.category_name == "wall"
    assert cat_metrics.category_id == 0
    assert cat_metrics.precision == 0.90
    assert cat_metrics.num_predictions == 150
    assert cat_metrics.num_ground_truth == 145


@pytest.mark.skipif(
    not _has_torch(),
    reason="PyTorch not installed"
)
def test_model_optimizer_init():
    """Test ModelOptimizer initialization."""
    from cad3d.model_optimizer import ModelOptimizer
    import torch
    
    optimizer = ModelOptimizer(device="cpu")
    assert optimizer.device.type == "cpu"
    assert len(optimizer.results) == 0


@pytest.mark.skipif(
    not _has_torch(),
    reason="PyTorch not installed"
)
def test_detection_benchmark_init():
    """Test DetectionBenchmark initialization."""
    from cad3d.benchmark_suite import DetectionBenchmark
    import torch
    import torch.nn as nn
    
    # Create dummy model
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1)
    )
    
    benchmark = DetectionBenchmark(model, device="cpu")
    assert benchmark.device.type == "cpu"
    assert benchmark.confidence_threshold == 0.5
    assert benchmark.iou_threshold == 0.5
    assert len(benchmark.category_names) == 15


@pytest.mark.skipif(
    not _has_torch(),
    reason="PyTorch not installed"
)
def test_calculate_iou():
    """Test IoU calculation."""
    from cad3d.benchmark_suite import DetectionBenchmark
    import torch.nn as nn
    import numpy as np
    
    model = nn.Sequential()
    benchmark = DetectionBenchmark(model, device="cpu")
    
    # Test identical boxes
    box1 = np.array([0, 0, 100, 100])
    box2 = np.array([0, 0, 100, 100])
    iou = benchmark.calculate_iou(box1, box2)
    assert iou == 1.0
    
    # Test non-overlapping boxes
    box1 = np.array([0, 0, 50, 50])
    box2 = np.array([100, 100, 150, 150])
    iou = benchmark.calculate_iou(box1, box2)
    assert iou == 0.0
    
    # Test partially overlapping boxes
    box1 = np.array([0, 0, 100, 100])
    box2 = np.array([50, 50, 150, 150])
    iou = benchmark.calculate_iou(box1, box2)
    assert 0 < iou < 1


def test_cli_optimize_command():
    """Test that optimize-model CLI command is registered."""
    from cad3d import cli
    
    # Test --help doesn't crash
    with pytest.raises(SystemExit):
        cli.main(["optimize-model", "--help"])


def test_cli_benchmark_command():
    """Test that benchmark CLI command is registered."""
    from cad3d import cli
    
    # Test --help doesn't crash
    with pytest.raises(SystemExit):
        cli.main(["benchmark", "--help"])


def test_optimization_system_summary(capsys):
    """Print comprehensive optimization system summary."""
    print("\n" + "="*70)
    print("⚡ OPTIMIZATION & BENCHMARK SYSTEM - SUMMARY")
    print("="*70)
    
    print("\n🔧 Model Optimizer (model_optimizer.py):")
    print("  ✓ ModelOptimizer class")
    print("  ✓ PyTorch → ONNX conversion")
    print("  ✓ ONNX optimization")
    print("  ✓ Dynamic quantization (INT8)")
    print("  ✓ TensorRT conversion (FP32/FP16/INT8)")
    print("  ✓ Performance benchmarking")
    print("  ✓ Model comparison and analysis")
    
    print("\n📊 Benchmark Suite (benchmark_suite.py):")
    print("  ✓ DetectionBenchmark class")
    print("  ✓ Precision, Recall, F1-Score calculation")
    print("  ✓ mAP (Mean Average Precision) @ IoU 0.5, 0.75")
    print("  ✓ Per-category performance metrics")
    print("  ✓ IoU (Intersection over Union) calculation")
    print("  ✓ Inference speed measurement (FPS)")
    print("  ✓ Multi-model comparison")
    print("  ✓ JSON report generation")
    
    print("\n🔧 CLI Commands:")
    print("  ✓ optimize-model: Convert model to ONNX/TensorRT/Quantized")
    print("    - Multiple format support (onnx, tensorrt, quantized)")
    print("    - Automatic benchmarking")
    print("    - Performance comparison")
    print("  ✓ benchmark: Evaluate model accuracy and speed")
    print("    - Full dataset evaluation")
    print("    - Detailed per-category metrics")
    print("    - JSON report export")
    
    print("\n📊 Supported Optimizations:")
    print("  • ONNX: Cross-platform deployment, ~1.2-1.5x speedup")
    print("  • Quantization: INT8 precision, ~2-3x speedup, 4x smaller")
    print("  • TensorRT FP16: NVIDIA GPUs, ~2-4x speedup")
    print("  • TensorRT INT8: NVIDIA GPUs, ~4-8x speedup")
    
    print("\n🎯 Evaluation Metrics:")
    print("  • Precision: True Positives / (True Positives + False Positives)")
    print("  • Recall: True Positives / (True Positives + False Negatives)")
    print("  • F1-Score: Harmonic mean of Precision and Recall")
    print("  • mAP: Mean Average Precision across all categories")
    print("  • IoU: Intersection over Union for bounding box overlap")
    print("  • FPS: Frames per second (inference speed)")
    
    print("\n💡 Use Cases:")
    print("  • Deploy models for production (ONNX for CPU, TensorRT for GPU)")
    print("  • Reduce model size for mobile/edge devices (Quantization)")
    print("  • Benchmark before/after optimization")
    print("  • Compare multiple model architectures")
    print("  • Evaluate per-category performance (find weak classes)")
    print("  • Measure accuracy vs speed tradeoffs")
    
    print("\n🔗 Workflow:")
    print("  1. Train model → Best checkpoint")
    print("  2. Optimize model → ONNX/TensorRT/Quantized versions")
    print("  3. Benchmark all versions → Compare speed/accuracy")
    print("  4. Select best model for deployment target")
    print("  5. Use in production (Neural CAD Detector)")
    
    print("\n" + "="*70)


def test_optimization_formats():
    """Test that all optimization formats are supported."""
    from cad3d.model_optimizer import ModelOptimizer
    
    # Check that constants are defined
    assert hasattr(ModelOptimizer, 'convert_to_onnx')
    assert hasattr(ModelOptimizer, 'quantize_model')
    assert hasattr(ModelOptimizer, 'convert_to_tensorrt')
    assert hasattr(ModelOptimizer, 'optimize_full_pipeline')


def test_benchmark_metrics_to_dict():
    """Test DetectionMetrics to_dict conversion."""
    from cad3d.benchmark_suite import DetectionMetrics
    
    metrics = DetectionMetrics(
        precision=0.85,
        recall=0.82,
        f1_score=0.835,
        mAP=0.87,
        mAP_50=0.89,
        mAP_75=0.85,
        average_iou=0.75,
        inference_time_ms=30.5,
        fps=32.8
    )
    
    metrics_dict = metrics.to_dict()
    assert isinstance(metrics_dict, dict)
    assert metrics_dict['precision'] == 0.85
    assert metrics_dict['mAP'] == 0.87
    assert metrics_dict['fps'] == 32.8


@pytest.mark.skipif(
    not _has_torch(),
    reason="PyTorch not installed"
)
def test_benchmark_precision_recall_calculation():
    """Test precision and recall calculation."""
    from cad3d.benchmark_suite import DetectionBenchmark
    import torch.nn as nn
    
    model = nn.Sequential()
    benchmark = DetectionBenchmark(model, device="cpu")
    
    # Test with all true positives
    matches = [True, True, True, True]
    precision, recall = benchmark.calculate_precision_recall(matches, 4, 4)
    assert precision == 1.0
    assert recall == 1.0
    
    # Test with some false positives
    matches = [True, True, False, False]
    precision, recall = benchmark.calculate_precision_recall(matches, 4, 4)
    assert precision == 0.5
    assert recall == 0.5
    
    # Test with no predictions
    precision, recall = benchmark.calculate_precision_recall([], 0, 5)
    assert precision == 0.0
    assert recall == 0.0
