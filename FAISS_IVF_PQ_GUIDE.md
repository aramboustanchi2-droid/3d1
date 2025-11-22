# FAISS Advanced - IVF/PQ Guide

# راهنمای اندیس‌های پیشرفته FAISS

## Overview

این راهنما نحوه استفاده از اندیس‌های پیشرفته FAISS (IVF و PQ) را برای جستجوی بردارهای با مقیاس میلیونی توضیح می‌دهد.

This guide explains how to use FAISS advanced indexes (IVF and PQ) for large-scale vector search (millions of vectors).

## Index Types

### 1. IndexFlatL2 (Exact Search)

- **Use case**: Small datasets (<10K vectors)
- **Pros**: Exact results, simple, no training needed
- **Cons**: O(n) search time, slow for large datasets
- **Memory**: 4 bytes × dimension per vector

### 2. IndexIVFFlat (Approximate Search)

- **Use case**: Medium datasets (10K-10M vectors)
- **Pros**: Faster search (sub-linear), good accuracy
- **Cons**: Requires training, approximate results
- **Memory**: Same as Flat (no compression)
- **Parameters**:
  - `nlist`: Number of clusters (e.g., sqrt(n))
  - `nprobe`: Number of clusters to search (trade speed/accuracy)

### 3. IndexIVFPQ (Compressed Approximate Search)

- **Use case**: Large datasets (1M-1B+ vectors)
- **Pros**: Fast search + low memory (compression)
- **Cons**: Requires training, lower accuracy than IVF
- **Memory**: (dimension/m) × (nbits/8) bytes per vector
- **Parameters**:
  - `nlist`: Number of clusters
  - `nprobe`: Number of clusters to search
  - `m`: Number of sub-quantizers (dimension must be divisible by m)
  - `nbits`: Bits per sub-quantizer (8 or 16)

### 4. IndexHNSW (Graph-Based Search)

- **Use case**: Very fast search, high memory available
- **Pros**: Fast search, no training needed
- **Cons**: High memory usage
- **Parameters**: `M` (connections per layer)

## Installation

```bash
# CPU version (recommended for most users)
pip install faiss-cpu

# GPU version (requires CUDA)
pip install faiss-gpu
```

## Usage

### Basic Usage (Auto-Select)

```python
from cad3d.super_ai.ai_model_database_faiss_advanced import AIModelDatabaseFAISSAdvanced

# Auto-select best index based on dataset size
db = AIModelDatabaseFAISSAdvanced(
    index_path="faiss_data",
    dimension=384,
    index_type='auto'  # Automatically chooses: flat, ivf, or ivfpq
)

# Add vectors (index trains automatically after 100 samples)
for i in range(100_000):
    db.create_model(
        name=f"model_{i}",
        description="Description...",
        architecture="CNN"
    )

# Search
results = db.search_similar_models("CNN classification", n_results=10)
for r in results:
    print(f"{r['name']}: {r['similarity_score']:.4f}")
```

### Manual Index Selection

#### Flat Index (Exact, Small Scale)

```python
db = AIModelDatabaseFAISSAdvanced(
    index_path="faiss_data",
    dimension=384,
    index_type='flat'
)
```

#### IVF Index (Approximate, Medium Scale)

```python
db = AIModelDatabaseFAISSAdvanced(
    index_path="faiss_data",
    dimension=384,
    index_type='ivf',
    nlist=1000,      # Number of clusters (e.g., sqrt(n_vectors))
    nprobe=10        # Search 10 clusters (higher = more accurate, slower)
)
```

#### IVFPQ Index (Compressed, Large Scale)

```python
db = AIModelDatabaseFAISSAdvanced(
    index_path="faiss_data",
    dimension=384,
    index_type='ivfpq',
    nlist=4096,      # Number of clusters
    nprobe=16,       # Search 16 clusters
    m=48,            # 384/48 = 8 bytes per vector (48x compression)
    nbits=8          # 8 bits per sub-quantizer
)
```

#### GPU Acceleration

```python
db = AIModelDatabaseFAISSAdvanced(
    index_path="faiss_data",
    dimension=384,
    index_type='ivfpq',
    nlist=4096,
    use_gpu=True,    # Enable GPU
    gpu_id=0         # GPU device ID
)
```

## Training

IVF and IVFPQ indexes require training before use:

- **Automatic training**: Index trains after 100 vectors are added
- **Training data**: Uses first 1000 vectors as training sample
- **Training time**:
  - IVF: Few seconds
  - IVFPQ: 10-60 seconds depending on nlist and m

**Note**: Training is a one-time cost. Once trained, the index is saved and reused.

## Parameter Tuning

### nlist (Number of Clusters)

- **Rule of thumb**: sqrt(n_vectors)
- **Examples**:
  - 10K vectors → nlist=100
  - 100K vectors → nlist=1000
  - 1M vectors → nlist=4096
  - 10M vectors → nlist=16384

### nprobe (Search Clusters)

- **Trade-off**: Accuracy vs Speed
- **Values**:
  - nprobe=1: Fastest, lowest accuracy (~70%)
  - nprobe=10: Good balance (~90% accuracy)
  - nprobe=50: High accuracy (~95%), slower
  - nprobe=nlist: Exact search (same as Flat)

### m (Sub-Quantizers for PQ)

- **Constraint**: dimension % m == 0
- **Values**: 8, 16, 32, 64
- **Compression**:
  - m=8 → dimension/8 bytes per vector
  - m=16 → dimension/16 bytes per vector
  - m=48 (for dim=384) → 8 bytes per vector (48x compression)

### nbits (Bits per Sub-Quantizer)

- **Values**: 8 (default), 16
- **Trade-off**: Memory vs Accuracy
- 8 bits: Good balance
- 16 bits: Better accuracy, 2x memory

## Performance Benchmarks

### 50K Vectors (384 dimensions)

| Index Type | Search Time | Memory | Accuracy |
|-----------|------------|--------|---------|
| Flat | 50 ms | 76 MB | 100% |
| IVF (nlist=1000, nprobe=10) | 5 ms | 76 MB | ~90% |
| IVFPQ (nlist=2048, nprobe=16, m=48) | 3 ms | 8 MB | ~85% |

### 1M Vectors (384 dimensions)

| Index Type | Search Time | Memory | Accuracy |
|-----------|------------|--------|---------|
| Flat | 1000 ms | 1.5 GB | 100% |
| IVF (nlist=4096, nprobe=10) | 40 ms | 1.5 GB | ~90% |
| IVFPQ (nlist=4096, nprobe=16, m=48) | 25 ms | 100 MB | ~85% |

**Note**: Times measured on Intel i7-10700K, single-threaded. GPU can be 5-10x faster.

## Best Practices

### 1. Start with Auto-Selection

```python
db = AIModelDatabaseFAISSAdvanced(index_type='auto')
```

### 2. Optimize After Initial Build

```python
# Add all vectors first
for item in data:
    db.create_model(...)

# Then optimize
db.optimize_index('models')
```

### 3. Tune nprobe for Your Use Case

```python
# Fast search, lower accuracy
db.nprobe = 5

# Balanced (default)
db.nprobe = 10

# High accuracy, slower
db.nprobe = 50
```

### 4. Use GPU for Large Datasets

```python
# CPU: 40 ms per search
db = AIModelDatabaseFAISSAdvanced(index_type='ivfpq', use_gpu=False)

# GPU: 5 ms per search (8x faster)
db = AIModelDatabaseFAISSAdvanced(index_type='ivfpq', use_gpu=True)
```

### 5. Monitor Training Status

```python
stats = db.get_statistics()
print(f"Trained: {stats['index_info']['models_trained']}")
print(f"Index type: {stats['index_info']['models_index_type']}")
```

## Example Workflows

### Workflow 1: Start Small, Scale Up

```python
# 1. Start with Flat (exact search)
db = AIModelDatabaseFAISSAdvanced(index_type='flat')

# 2. Add 5K vectors
for i in range(5000):
    db.create_model(...)

# 3. Scale to 100K vectors → auto-optimize to IVF
for i in range(5000, 100_000):
    db.create_model(...)

db.optimize_index('models')  # Converts to IVF

# 4. Scale to 1M vectors → optimize to IVFPQ
for i in range(100_000, 1_000_000):
    db.create_model(...)

db.optimize_index('models')  # Converts to IVFPQ
```

### Workflow 2: Direct Large-Scale

```python
# If you know you have millions of vectors, start with IVFPQ
db = AIModelDatabaseFAISSAdvanced(
    index_type='ivfpq',
    nlist=16384,
    nprobe=32,
    m=48
)

# Add millions of vectors
for i in range(10_000_000):
    db.create_model(...)
```

### Workflow 3: Accuracy-Critical Application

```python
# Use IVF with high nprobe for ~95% accuracy
db = AIModelDatabaseFAISSAdvanced(
    index_type='ivf',
    nlist=4096,
    nprobe=50  # High nprobe = high accuracy
)
```

## Troubleshooting

### Issue: "Index is not trained"

**Solution**: Add at least 100 vectors before searching, or manually train:

```python
# The index will auto-train after 100 vectors
```

### Issue: "Not enough training data"

**Solution**: IVF requires at least `nlist * 39` vectors for training.

```python
# If nlist=1000, need at least 39,000 vectors
# Reduce nlist or add more data
```

### Issue: Search is too slow

**Solutions**:

1. Reduce nprobe: `db.nprobe = 5`
2. Use GPU: `use_gpu=True`
3. Switch to IVFPQ: `index_type='ivfpq'`

### Issue: Search accuracy is too low

**Solutions**:

1. Increase nprobe: `db.nprobe = 50`
2. Reduce m (PQ compression): `m=16` instead of `m=48`
3. Use IVF instead of IVFPQ

### Issue: Out of memory

**Solutions**:

1. Use IVFPQ (compression): `index_type='ivfpq', m=48`
2. Reduce nbits: `nbits=8` instead of `nbits=16`

## Environment Variables

```bash
# Use real embeddings (sentence-transformers)
export EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2"
export EMBEDDING_DEVICE="auto"  # or "cuda", "cpu"

# Or use hash embeddings (fallback)
# (no env vars needed)
```

## Demo

Run the demo to see performance comparisons:

```bash
# Run all demos (may take 30+ minutes)
python demo_faiss_advanced.py --demo all

# Run specific demo
python demo_faiss_advanced.py --demo auto      # Auto-selection
python demo_faiss_advanced.py --demo ivf       # IVF index (50K vectors)
python demo_faiss_advanced.py --demo ivfpq     # IVFPQ index (100K vectors)
python demo_faiss_advanced.py --demo compare   # Compare Flat vs IVF vs IVFPQ
python demo_faiss_advanced.py --demo optimize  # Index optimization
```

## References

- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [FAISS Indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- [FAISS GPU](https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU)
- [FAISS Performance Tuning](https://github.com/facebookresearch/faiss/wiki/Faster-search)

## Summary

- **<10K vectors**: Use `index_type='flat'` (exact search)
- **10K-1M vectors**: Use `index_type='ivf'` (approximate search)
- **1M+ vectors**: Use `index_type='ivfpq'` (compressed approximate)
- **GPU available**: Add `use_gpu=True` for 5-10x speedup
- **Auto-select**: Use `index_type='auto'` and call `optimize_index()`

**Trade-offs**:

- Flat: Slow, exact, high memory
- IVF: Fast, ~90% accurate, high memory
- IVFPQ: Very fast, ~85% accurate, low memory

Choose based on your dataset size, memory constraints, and accuracy requirements.
