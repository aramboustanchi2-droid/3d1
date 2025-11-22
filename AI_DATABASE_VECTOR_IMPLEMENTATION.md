# AI Model Database - Vector Database Implementation Guide

## راهنمای پیاده‌سازی پایگاه داده برداری

## Overview | نمای کلی

This guide covers **ChromaDB** and **FAISS** as the 4th and 5th database backends for storing and searching AI model embeddings.

پایگاه‌های داده برداری برای ذخیره‌سازی embeddings و جستجوی معنایی استفاده می‌شوند.

## Architecture | معماری

```text
AI Model Database Vector Stack
├── ChromaDB (Persistent Vector DB)
│   ├── Collections: models, datasets, predictions
│   ├── Embeddings: 384-dimensional vectors
│   ├── Metadata: JSON storage
│   └── Persistence: chromadb_data/
│
└── FAISS (Fast Similarity Search)
    ├── Indexes: IndexFlatL2 (L2 distance)
    ├── Vectors: NumPy arrays (float32)
    ├── Metadata: JSON files
    └── Persistence: faiss_data/
```text

## Feature Comparison | مقایسه ویژگی‌ها

| Feature | ChromaDB | FAISS |
|---------|----------|-------|
| **Storage** | Persistent (SQLite + files) | Persistent (binary indexes) |
| **Speed** | Good (100-1K QPS) | Excellent (10K+ QPS) |
| **Scalability** | Excellent (billions) | Good (millions optimal) |
| **Memory** | Low (on-disk) | Medium (in-memory indexes) |
| **Metadata** | Rich (JSON support) | Manual (separate files) |
| **Distance** | Cosine, L2, IP | L2, IP, cosine (various) |
| **GPU Support** | No | Yes (faiss-gpu) |
| **Dependencies** | chromadb | faiss-cpu / faiss-gpu |
| **Best For** | Production RAG systems | Research, speed-critical |

## Installation | نصب

### ChromaDB

```bash
# CPU only (lightweight)
pip install chromadb

# With optional dependencies
pip install chromadb[server]  # For client-server mode
```text

### FAISS

```bash
# CPU version (recommended for most)
pip install faiss-cpu

# GPU version (for CUDA-enabled systems)
pip install faiss-gpu

# All vector database dependencies
pip install -r requirements-vector-database.txt
```text

### Requirements File

```txt
chromadb>=0.4.0
faiss-cpu>=1.7.4  # or faiss-gpu for GPU
numpy>=1.24.0
sentence-transformers>=2.2.0  # Optional: for real embeddings
```

## Usage | استفاده

### ChromaDB Example

```python
from cad3d.super_ai.ai_db_factory import create_ai_database

# Create ChromaDB instance
db = create_ai_database(
    backend='chromadb',
    persist_directory='chromadb_data'  # Relative to E:\3d
)

# Create model with embedding
model_id = db.create_model(
    name="ResNet50-Classifier",
    description="Deep residual network for image classification",
    architecture="ResNet50",
    framework="PyTorch",
    task_type="image_classification"
)

# Semantic search: Find similar models
similar_models = db.search_similar_models(
    query="image recognition deep learning",
    n_results=5
)

for model in similar_models:
    print(f"{model['name']}: similarity={model['similarity_score']:.3f}")
```

### FAISS Example

```python
# Create FAISS instance
db = create_ai_database(
    backend='faiss',
    index_path='faiss_data',
    dimension=384  # Must match embedding size
)

# Create model (same API as ChromaDB)
model_id = db.create_model(
    name="BERT-TextClassifier",
    description="Transformer model for NLP text classification",
    architecture="BERT-base",
    framework="Transformers"
)

# Fast similarity search
similar_models = db.search_similar_models(
    query="natural language processing",
    n_results=3
)

# FAISS is faster for large-scale searches
```

### Configuration File

```yaml
# ChromaDB Configuration
backend: chromadb
persist_directory: chromadb_data
echo: false

---

# FAISS Configuration
backend: faiss
index_path: faiss_data
dimension: 384
echo: false
```

## API Reference | مرجع API

All vector databases support the same unified API as other backends:

### Core Methods

```python
# Models
create_model(name, description, architecture, framework, task_type, ...) -> int
get_model(model_id: int) -> Dict
list_models() -> List[Dict]
search_similar_models(query: str, n_results: int) -> List[Dict]  # NEW!

# Datasets
create_dataset(name, description, source_path, format, ...) -> int
get_dataset(dataset_id: int) -> Dict
search_similar_datasets(query: str, n_results: int) -> List[Dict]  # NEW!

# Predictions
log_prediction(model_version_id, input_data, output_data, confidence, ...)
get_predictions(model_version_id: int, limit: int) -> List[Dict]
search_similar_predictions(query: str, n_results: int) -> List[Dict]  # NEW!

# Model Versions
create_model_version(model_id, version, checkpoint_path, config, ...) -> int
get_model_versions(model_id: int) -> List[Dict]

# Training Runs
create_training_run(model_version_id, dataset_id, run_name, ...) -> int
update_training_run(run_id, status, final_loss, ...)
get_training_run(run_id: int) -> Dict

# Hyperparameters & Metrics
log_hyperparameters(training_run_id, params: Dict)
log_metric(training_run_id, metric_name, metric_value, epoch, step, split)
get_hyperparameters(training_run_id: int) -> Dict
get_metrics(training_run_id, metric_name) -> List[Dict]

# Experiments
create_experiment(name, description, hypothesis, ...) -> int
add_experiment_run(experiment_id, training_run_id, variant_name, notes)
get_experiment_runs(experiment_id: int) -> List[Dict]

# Statistics
get_statistics() -> Dict
```

### New Vector Search Methods

```python
# Semantic similarity search (added to vector databases)
search_similar_models(query: str, n_results: int = 5) -> List[Dict]
search_similar_datasets(query: str, n_results: int = 5) -> List[Dict]
search_similar_predictions(query: str, n_results: int = 10) -> List[Dict]

# Returns: List of items with similarity_score (0.0 to 1.0)
# Higher score = more similar
```

## Embedding Generation | تولید Embedding

### Demo (Hash-based, for testing)

Both vector databases use a simple hash-based embedding for demonstration:

```python
def _generate_embedding(text: str) -> np.ndarray:
    """Simple hash-based embedding (384-dim)"""
    import hashlib
    hash_obj = hashlib.sha256(text.encode())
    hash_bytes = hash_obj.digest()
    
    embedding = []
    for i in range(384):
        embedding.append(float(hash_bytes[i % len(hash_bytes)]) / 255.0)
    
    return np.array(embedding, dtype=np.float32)
```

### Production (Real Embeddings)

For production, replace with real embedding models:

```python
# Option 1: sentence-transformers (best for general use)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions
embedding = model.encode(text)

# Option 2: OpenAI Embeddings
import openai
response = openai.Embedding.create(
    model="text-embedding-ada-002",  # 1536 dimensions
    input=text
)
embedding = response['data'][0]['embedding']

# Option 3: Hugging Face Transformers
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)
embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
```

### Environment Activation (Real Embeddings)

To enable real sentence-transformers embeddings (instead of the default hash demo), set the following variables in your `.env` file:

```dotenv
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2  # Any SentenceTransformer model
EMBEDDING_DEVICE=auto                  # cpu | cuda | auto
EMBEDDING_NORMALIZE=true               # Use unit-normalized embeddings (recommended)
```

Behavior:

1. On first embedding request the system lazily attempts to load `sentence-transformers`.
2. If loading succeeds, `_embedding_backend` switches to `transformer` and all subsequent embeddings are cached and generated via the model.
3. If loading fails (package missing, model download error, GPU issue) it silently falls back to the hash-based demo embeddings (`_embedding_backend == 'hash'`).
4. FAISS will auto-adjust index dimension on the very first transformer embedding if no vectors have been inserted yet; otherwise it pads/truncates to the original dimension to preserve index integrity.

You can confirm activation by running:

```bash
python demo_real_embeddings_vector.py
```

Expected output snippet when active:

```text
ChromaDB embedding backend: transformer (model=all-MiniLM-L6-v2)
FAISS embedding backend: transformer (model=all-MiniLM-L6-v2)
```

If not active (fallback):

```text
ChromaDB embedding backend: hash (model=all-MiniLM-L6-v2)
```

## Data Flow | جریان داده

```
1. User Creates Model
   ├── Model metadata: name, architecture, framework
   ├── Generate text: "name architecture framework task_type"
   ├── Create embedding: _generate_embedding(text)
   └── Store: ChromaDB.add() / FAISS.add()

2. User Searches Models
   ├── Query text: "image classification deep learning"
   ├── Generate query embedding
   ├── Vector search: find nearest neighbors
   ├── Calculate similarity scores
   └── Return: models sorted by similarity

3. Persistence
   ├── ChromaDB: Auto-persist to SQLite + files
   └── FAISS: Manual save via write_index()
```

## Performance Benchmarks | معیارهای عملکرد

### ChromaDB Performance

```
Operation               Time (ms)   Throughput (ops/sec)
─────────────────────   ─────────   ───────────────────
Insert 1 vector         5-10 ms     100-200 ops/sec
Insert 1000 vectors     200-500 ms  2-5K ops/sec (batch)
Search (1-10 results)   10-50 ms    20-100 QPS
Search (100+ results)   50-200 ms   5-20 QPS

Memory: ~100 MB + (vectors * 4 bytes * dimension)
Disk: ~500 MB + vectors storage
```

### FAISS Performance

```
Operation               Time (ms)   Throughput (ops/sec)
─────────────────────   ─────────   ───────────────────
Insert 1 vector         0.5-1 ms    1K-2K ops/sec
Insert 1000 vectors     50-100 ms   10-20K ops/sec (batch)
Search (1-10 results)   1-5 ms      200-1K QPS
Search (100+ results)   5-20 ms     50-200 QPS

Memory: ~200 MB + (vectors * 4 bytes * dimension)
Disk: ~100 MB + index size

GPU (faiss-gpu):
Search: 0.1-1 ms (10-100x faster)
```

### Scalability

| Vectors | ChromaDB | FAISS CPU | FAISS GPU |
|---------|----------|-----------|-----------|
| 1K      | Excellent | Excellent | Overkill  |
| 10K     | Excellent | Excellent | Excellent |
| 100K    | Good      | Good      | Excellent |
| 1M      | Good      | Medium    | Excellent |
| 10M+    | Excellent | Slow      | Good      |

## Use Cases | موارد استفاده

### ChromaDB Best For

1. **Production RAG Systems**
   - Long-term persistent storage
   - Rich metadata queries
   - Multi-user applications

2. **Model Registry**
   - Semantic model search
   - Find similar architectures
   - Model recommendation

3. **Dataset Discovery**
   - Find similar datasets
   - Data quality analysis
   - Duplicate detection

4. **Prediction Analysis**
   - Cluster similar predictions
   - Anomaly detection
   - Quality monitoring

### FAISS Best For

1. **High-Speed Search**
   - Real-time recommendations
   - Live model selection
   - Low-latency APIs

2. **Research & Experimentation**
   - Quick prototyping
   - Algorithm testing
   - Parameter tuning

3. **Large-Scale Search**
   - Millions of vectors
   - GPU acceleration
   - Approximate nearest neighbors

4. **Batch Processing**
   - Offline analysis
   - Similarity clustering
   - Deduplication

## Integration with Existing Backends | یکپارچگی با Backend‌های موجود

Vector databases complement existing backends:

```python
# Use Case 1: Hybrid - SQLite for metadata + ChromaDB for search
sqlite_db = create_ai_database(backend='sqlite')
chroma_db = create_ai_database(backend='chromadb')

# Store metadata in SQLite (fast CRUD)
model_id = sqlite_db.create_model(name="ResNet50", ...)

# Store embeddings in ChromaDB (fast search)
chroma_db.create_model(name="ResNet50", ...)

# Use Case 2: Redis for real-time + FAISS for similarity
redis_db = create_ai_database(backend='redis')
faiss_db = create_ai_database(backend='faiss')

# Real-time metrics in Redis
redis_db.log_metric(run_id, "loss", 0.05)

# Semantic search in FAISS
similar = faiss_db.search_similar_models("image classification")
```

## Advanced Features | ویژگی‌های پیشرفته

### ChromaDB Advanced

```python
# 1. Collections (already implemented)
db.models_collection       # For model embeddings
db.datasets_collection     # For dataset embeddings
db.predictions_collection  # For prediction embeddings

# 2. Filter by metadata (future enhancement)
results = db.models_collection.query(
    query_embeddings=[embedding],
    where={"framework": "PyTorch"},  # Filter condition
    n_results=5
)

# 3. Custom distance metrics
collection = db.client.create_collection(
    name="custom_models",
    metadata={"hnsw:space": "cosine"}  # or "l2", "ip"
)
```

### FAISS Advanced

```python
import faiss

# 1. Different index types
index_flat = faiss.IndexFlatL2(dimension)         # Exact search
index_ivf = faiss.IndexIVFFlat(index_flat, dimension, 100)  # Faster, approximate
index_pq = faiss.IndexPQ(dimension, 8, 8)         # Compressed

# 2. GPU acceleration
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # Move to GPU 0

# 3. Multi-GPU
gpu_index = faiss.index_cpu_to_all_gpus(index)

# 4. Distance metrics
index = faiss.IndexFlatIP(dimension)  # Inner product (cosine)
```

## Troubleshooting | عیب‌یابی

### ChromaDB Issues

```bash
# Error: "Cannot connect to ChromaDB"
# Solution: Check permissions, disk space

# Error: "Collection already exists"
# Solution: Use get_or_create_collection()

# Performance slow?
# Solution: Batch inserts, use smaller embeddings
```

### FAISS Issues

```bash
# Error: "Index size mismatch"
# Solution: Ensure dimension matches exactly

# Error: "Out of memory"
# Solution: Use IndexIVF (approximate) or smaller batches

# Error: "GPU not available"
# Solution: Check CUDA, use faiss-cpu instead
```

## Migration Guide | راهنمای مهاجرت

### From SQLite to ChromaDB

```python
# Export from SQLite
sqlite_db = create_ai_database(backend='sqlite')
models = sqlite_db.list_models()

# Import to ChromaDB
chroma_db = create_ai_database(backend='chromadb')

for model in models:
    chroma_db.create_model(
        name=model['name'],
        description=model.get('description', ''),
        architecture=model.get('architecture', ''),
        framework=model.get('framework', '')
    )

print(f"Migrated {len(models)} models")
```

### From ChromaDB to FAISS

```python
# Export from ChromaDB
chroma_db = create_ai_database(backend='chromadb')
models = chroma_db.list_models()

# Import to FAISS
faiss_db = create_ai_database(backend='faiss', dimension=384)

for model in models:
    faiss_db.create_model(
        name=model['name'],
        description=model.get('description', ''),
        architecture=model.get('architecture', ''),
        framework=model.get('framework', '')
    )

print(f"Migrated {len(models)} models to FAISS")
```

## Demo Scripts | اسکریپت‌های نمایشی

```bash
# Run vector database demo
python demo_ai_database_vector.py

# Demos included:
# 1. ChromaDB - Persistent vector storage
# 2. FAISS - Fast similarity search
# 3. Comparison - ChromaDB vs FAISS
```

## FAISS Advanced - IVF/PQ for Large-Scale | اندیس‌های پیشرفته FAISS

### Overview

For datasets with **millions to billions** of vectors, standard FAISS IndexFlatL2 becomes slow and memory-intensive. FAISS provides advanced index types that use:

- **IVF (Inverted File Index)**: Clustering for approximate search
- **PQ (Product Quantization)**: Vector compression for memory efficiency

📖 **See full guide**: [FAISS_IVF_PQ_GUIDE.md](FAISS_IVF_PQ_GUIDE.md)

### Advanced FAISS Backend

```python
from cad3d.super_ai.ai_model_database_faiss_advanced import AIModelDatabaseFAISSAdvanced

# Auto-select best index based on dataset size
db = AIModelDatabaseFAISSAdvanced(
    index_path="faiss_data",
    dimension=384,
    index_type='auto'  # Automatically chooses: flat, ivf, or ivfpq
)

# For large datasets (1M+ vectors), use IVFPQ
db = AIModelDatabaseFAISSAdvanced(
    index_path="faiss_data",
    dimension=384,
    index_type='ivfpq',
    nlist=4096,      # Number of clusters
    nprobe=16,       # Search 16 clusters (trade speed/accuracy)
    m=48,            # 384/48 = 8 bytes per vector (48x compression!)
    nbits=8          # 8 bits per sub-quantizer
)

# GPU acceleration
db = AIModelDatabaseFAISSAdvanced(
    index_path="faiss_data",
    dimension=384,
    index_type='ivfpq',
    use_gpu=True     # 5-10x faster search
)
```

### Index Selection Guide

| Dataset Size | Recommended Index | Search Time | Memory |
|--------------|------------------|-------------|--------|
| <10K vectors | `flat` | 10-50 ms | High |
| 10K-100K | `ivf` | 5-20 ms | High |
| 100K-1M | `ivf` | 10-40 ms | High |
| 1M-10M | `ivfpq` | 20-60 ms | Low (compressed) |
| 10M+ | `ivfpq` + GPU | 5-20 ms | Low (compressed) |

### Performance Example (1M vectors, 384 dim)

```python
# Standard FAISS (IndexFlatL2)
db_flat = AIModelDatabaseFAISS(dimension=384)
# Search time: ~1000 ms
# Memory: 1.5 GB

# IVF (Approximate)
db_ivf = AIModelDatabaseFAISSAdvanced(
    index_type='ivf',
    nlist=4096,
    nprobe=10
)
# Search time: ~40 ms (25x faster!)
# Memory: 1.5 GB
# Accuracy: ~90%

# IVFPQ (Compressed Approximate)
db_ivfpq = AIModelDatabaseFAISSAdvanced(
    index_type='ivfpq',
    nlist=4096,
    nprobe=16,
    m=48
)
# Search time: ~25 ms (40x faster!)
# Memory: 100 MB (15x smaller!)
# Accuracy: ~85%
```

### Key Parameters

**nlist** (Number of clusters)

- Rule of thumb: `sqrt(n_vectors)`
- 10K vectors → nlist=100
- 1M vectors → nlist=4096
- 10M vectors → nlist=16384

**nprobe** (Search clusters)

- Trade-off: Accuracy vs Speed
- nprobe=1: Fastest (~70% accuracy)
- nprobe=10: Balanced (~90% accuracy)
- nprobe=50: High accuracy (~95%)

**m** (Sub-quantizers for PQ)

- Constraint: `dimension % m == 0`
- Higher m → More compression
- m=48 (for dim=384) → 48x compression!

### Training

IVF/PQ indexes require one-time training:

```python
# Automatic training after 100 vectors
db = AIModelDatabaseFAISSAdvanced(index_type='ivfpq')

for i in range(100_000):
    db.create_model(name=f"model_{i}", ...)
    
    if i == 100:
        # Index automatically trains here
        print("Index trained!")

# Check training status
stats = db.get_statistics()
print(f"Trained: {stats['index_info']['models_trained']}")
```

### Optimization

```python
# Start with any index type
db = AIModelDatabaseFAISSAdvanced(index_type='auto')

# Add vectors
for item in data:
    db.create_model(...)

# Optimize for current dataset size
db.optimize_index('models')

# Converts:
# <10K vectors → Flat (exact)
# 10K-1M → IVF (approximate)
# 1M+ → IVFPQ (compressed)
```

### Demo

```bash
# Run advanced FAISS demo
python demo_faiss_advanced.py

# Options:
python demo_faiss_advanced.py --demo auto      # Auto-selection
python demo_faiss_advanced.py --demo ivf       # IVF (50K vectors)
python demo_faiss_advanced.py --demo ivfpq     # IVFPQ (100K vectors)
python demo_faiss_advanced.py --demo compare   # Performance comparison
```

### When to Use Advanced FAISS

✅ **Use IVFPQ when**:

- Dataset has 1M+ vectors
- Memory is limited
- ~85% accuracy is acceptable
- Need fast search (<50 ms)

✅ **Use IVF when**:

- Dataset has 100K-10M vectors
- Need ~90% accuracy
- Memory is available
- Need fast search (<100 ms)

✅ **Use Flat when**:

- Dataset has <10K vectors
- Need 100% accuracy (exact search)
- Memory and speed are not critical

### Resources

- **Full Guide**: [FAISS_IVF_PQ_GUIDE.md](FAISS_IVF_PQ_GUIDE.md)
- **Demo**: `demo_faiss_advanced.py`
- **Implementation**: `cad3d/super_ai/ai_model_database_faiss_advanced.py`

---

## Summary | خلاصه

✅ **ChromaDB**: Best for production, persistent storage, rich metadata
✅ **FAISS (Basic)**: Best for speed, medium-scale search (up to 1M vectors)
✅ **FAISS (IVF/PQ)**: Best for large-scale search (millions to billions of vectors)
✅ **Unified API**: Same interface as SQLite, PostgreSQL, MySQL, Redis
✅ **Semantic Search**: Find similar models, datasets, predictions
✅ **Easy Integration**: Drop-in replacement via factory pattern

---

**Total Database Backends**: 7

- SQLite (default, local)
- PostgreSQL (production SQL)
- MySQL (production SQL)
- Redis (in-memory, ultra-fast)
- ChromaDB (vector, persistent)
- FAISS (vector, speed - basic & advanced)
- MongoDB (document store)

**FAISS Variants**:

- **Basic**: `AIModelDatabaseFAISS` (IndexFlatL2, exact search, <1M vectors)
- **Advanced**: `AIModelDatabaseFAISSAdvanced` (IVF/PQ, approximate, 1M-1B+ vectors)

Choose based on your needs! 🚀
