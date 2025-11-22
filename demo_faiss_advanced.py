"""
Demo: FAISS Advanced - IVF/PQ for Large-Scale Vector Search
نمایش: FAISS پیشرفته برای میلیون‌ها بردار

This demo shows:
1. Index type auto-selection based on dataset size
2. IVF training for approximate search
3. IVFPQ compression for memory efficiency
4. Performance comparison: Flat vs IVF vs IVFPQ
5. GPU acceleration (if available)
"""
import sys
from pathlib import Path
import time
import numpy as np

# Add cad3d to path
sys.path.insert(0, str(Path(__file__).parent))

from cad3d.super_ai.ai_model_database_faiss_advanced import AIModelDatabaseFAISSAdvanced


def benchmark_search(db: AIModelDatabaseFAISSAdvanced, query: str, n_results: int = 10, n_trials: int = 10):
    """Benchmark search performance"""
    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        results = db.search_similar_models(query, n_results)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    return results, avg_time, std_time


def demo_auto_selection():
    """Demo: Auto-select index type based on dataset size"""
    print("\n" + "="*70)
    print("Demo 1: Auto Index Selection")
    print("="*70)
    
    # Small dataset (< 10K) -> Flat
    print("\n1. Small dataset (1000 models) -> IndexFlatL2")
    db = AIModelDatabaseFAISSAdvanced(
        index_path="demo_faiss_data/small",
        dimension=384,
        index_type='auto'
    )
    
    for i in range(1000):
        db.create_model(
            name=f"model_{i}",
            description=f"A small model for classification task {i}",
            architecture="CNN",
            framework="PyTorch"
        )
    
    stats = db.get_statistics()
    print(f"  Index type: {stats['index_info']['models_index_type']}")
    print(f"  Models: {stats['models']}")
    
    results, avg_time, std_time = benchmark_search(db, "classification CNN", 5, 10)
    print(f"  Search time: {avg_time:.2f}±{std_time:.2f} ms")
    print(f"  Top result: {results[0]['name']} (score: {results[0]['similarity_score']:.4f})")
    
    print("\n" + "-"*70)


def demo_ivf_index():
    """Demo: IVF index for medium-scale datasets"""
    print("\n" + "="*70)
    print("Demo 2: IVF Index (Medium Scale)")
    print("="*70)
    
    print("\n2. Medium dataset (50K models) -> IndexIVFFlat")
    db = AIModelDatabaseFAISSAdvanced(
        index_path="demo_faiss_data/medium",
        dimension=384,
        index_type='ivf',
        nlist=1000,
        nprobe=10
    )
    
    print("  Creating 50K models (this may take a few minutes)...")
    architectures = ["CNN", "RNN", "Transformer", "ResNet", "BERT", "GPT"]
    tasks = ["classification", "regression", "detection", "segmentation", "translation"]
    
    for i in range(50_000):
        arch = architectures[i % len(architectures)]
        task = tasks[i % len(tasks)]
        db.create_model(
            name=f"model_{arch}_{i}",
            description=f"A {arch} model for {task} task {i}",
            architecture=arch,
            framework="PyTorch",
            task_type=task
        )
        
        if (i + 1) % 10000 == 0:
            print(f"    Created {i+1}/50000 models...")
    
    stats = db.get_statistics()
    print(f"\n  Index type: {stats['index_info']['models_index_type']}")
    print(f"  Models: {stats['models']}")
    print(f"  Index trained: {stats['index_info']['models_trained']}")
    print(f"  nprobe: {stats['index_info']['nprobe']}")
    
    # Benchmark search
    results, avg_time, std_time = benchmark_search(db, "CNN classification", 10, 20)
    print(f"\n  Search time: {avg_time:.2f}±{std_time:.2f} ms")
    print(f"  Top 3 results:")
    for i, r in enumerate(results[:3], 1):
        print(f"    {i}. {r['name']} (score: {r['similarity_score']:.4f})")
    
    print("\n" + "-"*70)


def demo_ivfpq_index():
    """Demo: IVFPQ index for large-scale datasets"""
    print("\n" + "="*70)
    print("Demo 3: IVFPQ Index (Large Scale)")
    print("="*70)
    
    print("\n3. Large dataset (100K models) -> IndexIVFPQ")
    db = AIModelDatabaseFAISSAdvanced(
        index_path="demo_faiss_data/large",
        dimension=384,
        index_type='ivfpq',
        nlist=4096,
        nprobe=16,
        m=48,  # 384/48 = 8 bytes per vector (compressed)
        nbits=8
    )
    
    print("  Creating 100K models (this will take several minutes)...")
    architectures = ["CNN", "RNN", "LSTM", "GRU", "Transformer", "ResNet", "VGG", "BERT", "GPT", "T5"]
    tasks = ["classification", "regression", "detection", "segmentation", "translation", "summarization"]
    
    for i in range(100_000):
        arch = architectures[i % len(architectures)]
        task = tasks[i % len(tasks)]
        db.create_model(
            name=f"model_{arch}_{task}_{i}",
            description=f"A {arch} model for {task} with advanced features {i}",
            architecture=arch,
            framework="PyTorch" if i % 2 == 0 else "TensorFlow",
            task_type=task
        )
        
        if (i + 1) % 20000 == 0:
            print(f"    Created {i+1}/100000 models...")
    
    stats = db.get_statistics()
    print(f"\n  Index type: {stats['index_info']['models_index_type']}")
    print(f"  Models: {stats['models']}")
    print(f"  Index trained: {stats['index_info']['models_trained']}")
    print(f"  nprobe: {stats['index_info']['nprobe']}")
    
    # Compression ratio
    flat_size = stats['models'] * 384 * 4  # 4 bytes per float32
    compressed_size = stats['models'] * (384 // db.m) * (db.nbits // 8)
    compression_ratio = flat_size / compressed_size
    print(f"\n  Memory usage:")
    print(f"    Flat index: {flat_size / 1024 / 1024:.1f} MB")
    print(f"    IVFPQ index: {compressed_size / 1024 / 1024:.1f} MB")
    print(f"    Compression ratio: {compression_ratio:.1f}x")
    
    # Benchmark search
    results, avg_time, std_time = benchmark_search(db, "Transformer classification", 10, 30)
    print(f"\n  Search time: {avg_time:.2f}±{std_time:.2f} ms")
    print(f"  Top 5 results:")
    for i, r in enumerate(results[:5], 1):
        print(f"    {i}. {r['name']} (score: {r['similarity_score']:.4f})")
    
    print("\n" + "-"*70)


def demo_optimization():
    """Demo: Index optimization"""
    print("\n" + "="*70)
    print("Demo 4: Index Optimization")
    print("="*70)
    
    print("\n4. Creating 15K models, then optimizing...")
    db = AIModelDatabaseFAISSAdvanced(
        index_path="demo_faiss_data/optimize",
        dimension=384,
        index_type='auto'
    )
    
    for i in range(15_000):
        db.create_model(
            name=f"model_{i}",
            description=f"Model for task {i}",
            architecture="CNN" if i % 2 == 0 else "RNN"
        )
    
    stats_before = db.get_statistics()
    print(f"  Before optimization:")
    print(f"    Index type: {stats_before['index_info']['models_index_type']}")
    print(f"    Trained: {stats_before['index_info']['models_trained']}")
    
    print("\n  Running optimization...")
    db.optimize_index('models')
    
    stats_after = db.get_statistics()
    print(f"\n  After optimization:")
    print(f"    Index type: {stats_after['index_info']['models_index_type']}")
    print(f"    Trained: {stats_after['index_info']['models_trained']}")
    
    print("\n" + "-"*70)


def demo_comparison():
    """Demo: Compare Flat vs IVF vs IVFPQ"""
    print("\n" + "="*70)
    print("Demo 5: Performance Comparison")
    print("="*70)
    
    n_vectors = 50_000
    print(f"\nComparing 3 index types with {n_vectors} vectors:")
    
    # Flat
    print("\n1. IndexFlatL2 (Exact search)")
    db_flat = AIModelDatabaseFAISSAdvanced(
        index_path="demo_faiss_data/compare_flat",
        dimension=384,
        index_type='flat'
    )
    
    print("  Creating vectors...")
    for i in range(n_vectors):
        db_flat.create_model(name=f"model_{i}", description=f"Model {i}")
        if (i + 1) % 10000 == 0:
            print(f"    {i+1}/{n_vectors}...")
    
    _, time_flat, std_flat = benchmark_search(db_flat, "classification model", 10, 20)
    print(f"  Search time: {time_flat:.2f}±{std_flat:.2f} ms")
    
    # IVF
    print("\n2. IndexIVFFlat (Approximate search)")
    db_ivf = AIModelDatabaseFAISSAdvanced(
        index_path="demo_faiss_data/compare_ivf",
        dimension=384,
        index_type='ivf',
        nlist=1000,
        nprobe=10
    )
    
    print("  Creating vectors...")
    for i in range(n_vectors):
        db_ivf.create_model(name=f"model_{i}", description=f"Model {i}")
        if (i + 1) % 10000 == 0:
            print(f"    {i+1}/{n_vectors}...")
    
    _, time_ivf, std_ivf = benchmark_search(db_ivf, "classification model", 10, 20)
    print(f"  Search time: {time_ivf:.2f}±{std_ivf:.2f} ms")
    speedup_ivf = time_flat / time_ivf
    print(f"  Speedup vs Flat: {speedup_ivf:.1f}x")
    
    # IVFPQ
    print("\n3. IndexIVFPQ (Compressed approximate search)")
    db_ivfpq = AIModelDatabaseFAISSAdvanced(
        index_path="demo_faiss_data/compare_ivfpq",
        dimension=384,
        index_type='ivfpq',
        nlist=2048,
        nprobe=16,
        m=48,
        nbits=8
    )
    
    print("  Creating vectors...")
    for i in range(n_vectors):
        db_ivfpq.create_model(name=f"model_{i}", description=f"Model {i}")
        if (i + 1) % 10000 == 0:
            print(f"    {i+1}/{n_vectors}...")
    
    _, time_ivfpq, std_ivfpq = benchmark_search(db_ivfpq, "classification model", 10, 20)
    print(f"  Search time: {time_ivfpq:.2f}±{std_ivfpq:.2f} ms")
    speedup_ivfpq = time_flat / time_ivfpq
    print(f"  Speedup vs Flat: {speedup_ivfpq:.1f}x")
    
    # Summary
    print("\n" + "="*70)
    print("Performance Summary:")
    print("="*70)
    print(f"  IndexFlatL2:  {time_flat:.2f} ms  (baseline, exact)")
    print(f"  IndexIVFFlat: {time_ivf:.2f} ms  ({speedup_ivf:.1f}x faster, approximate)")
    print(f"  IndexIVFPQ:   {time_ivfpq:.2f} ms  ({speedup_ivfpq:.1f}x faster, compressed)")
    
    print("\n" + "-"*70)


def main():
    print("\n" + "="*70)
    print("FAISS Advanced Index Demo - IVF/PQ for Millions of Vectors")
    print("نمایش اندیس‌های پیشرفته FAISS برای میلیون‌ها بردار")
    print("="*70)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', type=str, default='all', 
                       choices=['all', 'auto', 'ivf', 'ivfpq', 'optimize', 'compare'],
                       help='Which demo to run')
    args = parser.parse_args()
    
    try:
        if args.demo in ['all', 'auto']:
            demo_auto_selection()
        
        if args.demo in ['all', 'ivf']:
            demo_ivf_index()
        
        if args.demo in ['all', 'ivfpq']:
            demo_ivfpq_index()
        
        if args.demo in ['all', 'optimize']:
            demo_optimization()
        
        if args.demo in ['all', 'compare']:
            demo_comparison()
        
        print("\n" + "="*70)
        print("✓ All demos completed successfully!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
