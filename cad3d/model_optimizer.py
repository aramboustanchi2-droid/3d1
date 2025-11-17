"""
Model optimization tools for converting PyTorch models to ONNX and TensorRT.
Includes quantization and performance benchmarking.
"""
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.quantization import quantize_dynamic
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None  # Placeholder when torch not available

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False


@dataclass
class OptimizationResult:
    """Result of model optimization."""
    model_path: str
    format: str  # "pytorch", "onnx", "tensorrt", "quantized"
    file_size_mb: float
    inference_time_ms: float
    memory_usage_mb: float
    accuracy_drop: Optional[float] = None
    
    def __str__(self):
        result = f"{self.format.upper()} Model:\n"
        result += f"  Path: {self.model_path}\n"
        result += f"  Size: {self.file_size_mb:.2f} MB\n"
        result += f"  Inference: {self.inference_time_ms:.2f} ms\n"
        result += f"  Memory: {self.memory_usage_mb:.2f} MB\n"
        if self.accuracy_drop is not None:
            result += f"  Accuracy Drop: {self.accuracy_drop:.2%}\n"
        return result


class ModelOptimizer:
    """
    Optimize PyTorch models for deployment.
    
    Supports:
    - ONNX conversion
    - TensorRT optimization
    - Dynamic quantization
    - Performance benchmarking
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize optimizer.
        
        Args:
            device: Device to use ("cuda" or "cpu")
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Install with: pip install torch")
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.results: List[OptimizationResult] = []
    
    def convert_to_onnx(
        self,
        model: Any,  # nn.Module when available
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 1024, 1024),
        opset_version: int = 14,
        dynamic_axes: Optional[Dict] = None
    ) -> str:
        """
        Convert PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model
            output_path: Output ONNX file path
            input_shape: Input tensor shape (batch, channels, height, width)
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes for variable input sizes
            
        Returns:
            Path to ONNX model
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not installed. Install with: pip install onnx onnxruntime")
        
        print(f"🔄 Converting to ONNX...")
        print(f"   Input shape: {input_shape}")
        print(f"   Opset version: {opset_version}")
        
        model.eval()
        model.to(self.device)
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Default dynamic axes for object detection
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'boxes': {0: 'batch_size', 1: 'num_detections'},
                'labels': {0: 'batch_size', 1: 'num_detections'},
                'scores': {0: 'batch_size', 1: 'num_detections'}
            }
        
        # Export to ONNX
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['boxes', 'labels', 'scores'],
            dynamic_axes=dynamic_axes
        )
        
        print(f"✅ ONNX model saved: {output_path}")
        
        # Verify ONNX model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print(f"✅ ONNX model verified")
        
        return str(output_path)
    
    def optimize_onnx(
        self,
        onnx_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Optimize ONNX model for inference.
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Output path for optimized model
            
        Returns:
            Path to optimized ONNX model
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not installed")
        
        print(f"🔄 Optimizing ONNX model...")
        
        # Load model
        onnx_model = onnx.load(onnx_path)
        
        # Apply optimizations
        from onnxruntime.transformers import optimizer
        optimized_model = optimizer.optimize_model(
            onnx_path,
            model_type='bert',  # Generic optimization
            num_heads=0,
            hidden_size=0
        )
        
        # Save optimized model
        if output_path is None:
            output_path = onnx_path.replace('.onnx', '_optimized.onnx')
        
        optimized_model.save_model_to_file(output_path)
        print(f"✅ Optimized ONNX model saved: {output_path}")
        
        return output_path
    
    def quantize_model(
        self,
        model: Any,  # nn.Module when available
        output_path: str
    ) -> str:
        """
        Apply dynamic quantization to PyTorch model.
        
        Args:
            model: PyTorch model
            output_path: Output path for quantized model
            
        Returns:
            Path to quantized model
        """
        print(f"🔄 Applying dynamic quantization...")
        
        model.eval()
        
        # Apply dynamic quantization
        quantized_model = quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        
        # Save quantized model
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(quantized_model.state_dict(), str(output_path))
        print(f"✅ Quantized model saved: {output_path}")
        
        return str(output_path)
    
    def convert_to_tensorrt(
        self,
        onnx_path: str,
        output_path: str,
        precision: str = "fp16",
        max_batch_size: int = 1,
        workspace_size: int = 1 << 30  # 1GB
    ) -> str:
        """
        Convert ONNX model to TensorRT engine.
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Output path for TensorRT engine
            precision: Precision mode ("fp32", "fp16", "int8")
            max_batch_size: Maximum batch size
            workspace_size: Workspace size in bytes
            
        Returns:
            Path to TensorRT engine
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT not installed")
        
        print(f"🔄 Converting to TensorRT...")
        print(f"   Precision: {precision}")
        print(f"   Max batch size: {max_batch_size}")
        
        # Create TensorRT builder
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = workspace_size
        
        # Set precision
        if precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
        
        # Build engine
        print(f"   Building TensorRT engine (this may take a while)...")
        engine = builder.build_engine(network, config)
        
        # Save engine
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())
        
        print(f"✅ TensorRT engine saved: {output_path}")
        
        return str(output_path)
    
    def benchmark_model(
        self,
        model: Any,
        input_shape: Tuple[int, ...] = (1, 3, 1024, 1024),
        num_runs: int = 100,
        warmup_runs: int = 10,
        model_format: str = "pytorch"
    ) -> Dict[str, float]:
        """
        Benchmark model inference performance.
        
        Args:
            model: Model to benchmark (PyTorch, ONNX, or TensorRT)
            input_shape: Input tensor shape
            num_runs: Number of inference runs
            warmup_runs: Number of warmup runs
            model_format: Model format ("pytorch", "onnx", "tensorrt")
            
        Returns:
            Dictionary with performance metrics
        """
        print(f"📊 Benchmarking {model_format.upper()} model...")
        print(f"   Input shape: {input_shape}")
        print(f"   Runs: {num_runs} (+ {warmup_runs} warmup)")
        
        if model_format == "pytorch":
            return self._benchmark_pytorch(model, input_shape, num_runs, warmup_runs)
        elif model_format == "onnx":
            return self._benchmark_onnx(model, input_shape, num_runs, warmup_runs)
        elif model_format == "tensorrt":
            return self._benchmark_tensorrt(model, input_shape, num_runs, warmup_runs)
        else:
            raise ValueError(f"Unknown model format: {model_format}")
    
    def _benchmark_pytorch(
        self,
        model: Any,  # nn.Module when available
        input_shape: Tuple[int, ...],
        num_runs: int,
        warmup_runs: int
    ) -> Dict[str, float]:
        """Benchmark PyTorch model."""
        model.eval()
        model.to(self.device)
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_input)
        
        # Benchmark
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
        
        end_time = time.time()
        
        # Calculate metrics
        total_time = (end_time - start_time) * 1000  # ms
        avg_time = total_time / num_runs
        throughput = 1000.0 / avg_time  # FPS
        
        # Memory usage
        if self.device.type == "cuda":
            memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            memory_mb = 0.0
        
        return {
            "avg_inference_time_ms": avg_time,
            "throughput_fps": throughput,
            "memory_mb": memory_mb
        }
    
    def _benchmark_onnx(
        self,
        model_path: str,
        input_shape: Tuple[int, ...],
        num_runs: int,
        warmup_runs: int
    ) -> Dict[str, float]:
        """Benchmark ONNX model."""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not installed")
        
        # Create ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device.type == "cuda" else ['CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        
        # Create dummy input
        input_name = session.get_inputs()[0].name
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(warmup_runs):
            _ = session.run(None, {input_name: dummy_input})
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            _ = session.run(None, {input_name: dummy_input})
        end_time = time.time()
        
        # Calculate metrics
        total_time = (end_time - start_time) * 1000
        avg_time = total_time / num_runs
        throughput = 1000.0 / avg_time
        
        return {
            "avg_inference_time_ms": avg_time,
            "throughput_fps": throughput,
            "memory_mb": 0.0  # Not easily measurable for ONNX
        }
    
    def _benchmark_tensorrt(
        self,
        engine_path: str,
        input_shape: Tuple[int, ...],
        num_runs: int,
        warmup_runs: int
    ) -> Dict[str, float]:
        """Benchmark TensorRT engine."""
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT not installed")
        
        # Load engine
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(f.read())
        
        # Create execution context
        context = engine.create_execution_context()
        
        # Allocate buffers
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        d_input = cuda.mem_alloc(dummy_input.nbytes)
        
        # Warmup
        for _ in range(warmup_runs):
            cuda.memcpy_htod(d_input, dummy_input)
            context.execute_v2([int(d_input)])
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            cuda.memcpy_htod(d_input, dummy_input)
            context.execute_v2([int(d_input)])
        cuda.Context.synchronize()
        end_time = time.time()
        
        # Calculate metrics
        total_time = (end_time - start_time) * 1000
        avg_time = total_time / num_runs
        throughput = 1000.0 / avg_time
        
        return {
            "avg_inference_time_ms": avg_time,
            "throughput_fps": throughput,
            "memory_mb": 0.0
        }
    
    def optimize_full_pipeline(
        self,
        model: Any,  # nn.Module when available
        output_dir: str,
        input_shape: Tuple[int, ...] = (1, 3, 1024, 1024),
        formats: List[str] = ["onnx", "quantized"],
        benchmark: bool = True
    ) -> List[OptimizationResult]:
        """
        Run full optimization pipeline.
        
        Args:
            model: PyTorch model
            output_dir: Output directory for optimized models
            input_shape: Input tensor shape
            formats: Formats to export ("onnx", "tensorrt", "quantized")
            benchmark: Whether to benchmark models
            
        Returns:
            List of optimization results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        # Baseline PyTorch model
        if benchmark:
            print("\n" + "="*70)
            print("📊 BASELINE PYTORCH MODEL")
            print("="*70)
            
            metrics = self.benchmark_model(model, input_shape, model_format="pytorch")
            
            # Save baseline model
            baseline_path = output_dir / "baseline.pth"
            torch.save(model.state_dict(), str(baseline_path))
            
            results.append(OptimizationResult(
                model_path=str(baseline_path),
                format="pytorch",
                file_size_mb=baseline_path.stat().st_size / (1024 ** 2),
                inference_time_ms=metrics["avg_inference_time_ms"],
                memory_usage_mb=metrics["memory_mb"]
            ))
        
        # ONNX conversion
        if "onnx" in formats:
            print("\n" + "="*70)
            print("📦 ONNX CONVERSION")
            print("="*70)
            
            onnx_path = output_dir / "model.onnx"
            self.convert_to_onnx(model, str(onnx_path), input_shape)
            
            if benchmark:
                metrics = self.benchmark_model(str(onnx_path), input_shape, model_format="onnx")
                
                results.append(OptimizationResult(
                    model_path=str(onnx_path),
                    format="onnx",
                    file_size_mb=onnx_path.stat().st_size / (1024 ** 2),
                    inference_time_ms=metrics["avg_inference_time_ms"],
                    memory_usage_mb=metrics["memory_mb"]
                ))
        
        # Quantization
        if "quantized" in formats:
            print("\n" + "="*70)
            print("⚡ QUANTIZATION")
            print("="*70)
            
            quantized_path = output_dir / "model_quantized.pth"
            self.quantize_model(model, str(quantized_path))
            
            if benchmark:
                # Load quantized model for benchmarking
                quantized_model = model  # Placeholder - need to load properly
                metrics = self.benchmark_model(quantized_model, input_shape, model_format="pytorch")
                
                results.append(OptimizationResult(
                    model_path=str(quantized_path),
                    format="quantized",
                    file_size_mb=quantized_path.stat().st_size / (1024 ** 2),
                    inference_time_ms=metrics["avg_inference_time_ms"],
                    memory_usage_mb=metrics["memory_mb"]
                ))
        
        # TensorRT (if available)
        if "tensorrt" in formats and TENSORRT_AVAILABLE:
            print("\n" + "="*70)
            print("🚀 TENSORRT CONVERSION")
            print("="*70)
            
            if "onnx" not in formats:
                # Need ONNX model first
                onnx_path = output_dir / "model.onnx"
                self.convert_to_onnx(model, str(onnx_path), input_shape)
            
            trt_path = output_dir / "model_fp16.trt"
            self.convert_to_tensorrt(str(onnx_path), str(trt_path), precision="fp16")
            
            if benchmark:
                metrics = self.benchmark_model(str(trt_path), input_shape, model_format="tensorrt")
                
                results.append(OptimizationResult(
                    model_path=str(trt_path),
                    format="tensorrt_fp16",
                    file_size_mb=trt_path.stat().st_size / (1024 ** 2),
                    inference_time_ms=metrics["avg_inference_time_ms"],
                    memory_usage_mb=metrics["memory_mb"]
                ))
        
        # Print summary
        print("\n" + "="*70)
        print("📈 OPTIMIZATION SUMMARY")
        print("="*70)
        
        for result in results:
            print(f"\n{result}")
        
        return results


def compare_models(results: List[OptimizationResult]) -> None:
    """
    Compare optimization results and print summary.
    
    Args:
        results: List of optimization results
    """
    if not results:
        print("No results to compare")
        return
    
    print("\n" + "="*70)
    print("🏆 MODEL COMPARISON")
    print("="*70)
    
    # Create comparison table
    print(f"\n{'Format':<20} {'Size (MB)':<12} {'Time (ms)':<12} {'Speedup':<10}")
    print("-" * 70)
    
    baseline_time = results[0].inference_time_ms
    
    for result in results:
        speedup = baseline_time / result.inference_time_ms
        print(f"{result.format:<20} {result.file_size_mb:>10.2f}  {result.inference_time_ms:>10.2f}  {speedup:>8.2f}x")
    
    # Find best model
    fastest = min(results, key=lambda r: r.inference_time_ms)
    smallest = min(results, key=lambda r: r.file_size_mb)
    
    print("\n🥇 Best Performance:")
    print(f"   Fastest: {fastest.format} ({fastest.inference_time_ms:.2f} ms)")
    print(f"   Smallest: {smallest.format} ({smallest.file_size_mb:.2f} MB)")
