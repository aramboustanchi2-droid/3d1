# Quick Reference: AI Model Database

راهنمای سریع: پایگاه داده مدل‌های هوش مصنوعی

## 🚀 Quick Setup

### SQLite (Zero Config)

```powershell
python demo_ai_database.py
```

### PostgreSQL

```powershell
# 1. Install PostgreSQL & create database
createdb aimodels

# 2. Install dependencies
pip install -r requirements-sql-database.txt

# 3. Run demo
python demo_ai_database_sql.py --backend postgresql --user postgres --password yourpass
```

### MySQL

```powershell
# 1. Install MySQL & create database
mysql -u root -p
CREATE DATABASE aimodels;

# 2. Install dependencies
pip install -r requirements-sql-database.txt

# 3. Run demo
python demo_ai_database_sql.py --backend mysql --user root --password yourpass
```

### Redis

```powershell
# 1. Install & start Redis server
# Windows: Download from https://github.com/microsoftarchive/redis/releases
# Docker: docker run -d -p 6379:6379 redis
# WSL: sudo apt install redis-server && redis-server

# 2. Install dependencies
pip install -r requirements-redis-database.txt

# 3. Run demo
python demo_ai_database_redis.py
```

### ChromaDB (Vector - Persistent Embeddings)

```powershell
# 1. Install dependencies
pip install chromadb

# Or full vector stack
pip install -r requirements-vector-database.txt

# 2. Run vector demo (includes ChromaDB)
python demo_ai_database_vector.py

# 3. Direct usage
python - <<'PY'
from cad3d.super_ai.ai_db_factory import create_ai_database
db = create_ai_database(backend='chromadb', persist_directory='chromadb_data')
model_id = db.create_model(name='ResNet50', description='Image classification deep CNN', architecture='ResNet50')
print(db.search_similar_models('image recognition deep learning', n_results=3))
PY
```

### FAISS (Vector - High-Speed Similarity)

```powershell
# 1. Install dependencies (CPU)
pip install faiss-cpu

# (Optional GPU)
pip install faiss-gpu

# Or full vector stack
pip install -r requirements-vector-database.txt

# 2. Run vector demo (includes FAISS)
python demo_ai_database_vector.py

# 3. Direct usage
python - <<'PY'
from cad3d.super_ai.ai_db_factory import create_ai_database
db = create_ai_database(backend='faiss', index_path='faiss_data', dimension=384)
model_id = db.create_model(name='BERT-Classifier', description='Transformer model for NLP classification', architecture='BERT-base')
print(db.search_similar_models('natural language processing transformer', n_results=3))
PY
```

## 📝 Configuration File

Create `ai_db_config.yaml` in E:\3d:

### SQLite

```yaml
backend: sqlite
db_path: ai_models.db
echo: false
```

### PostgreSQL Config

```yaml
backend: postgresql
connection:
  host: localhost
  port: 5432
  database: aimodels
  user: postgres
  password: yourpass
pool_size: 5
echo: false
```

### MySQL Config

```yaml
backend: mysql
connection:
  host: localhost
  port: 3306
  database: aimodels
  user: root
  password: yourpass
pool_size: 5
echo: false
```

### Redis Config

```yaml
backend: redis
connection:
  host: localhost
  port: 6379
  db: 0
  password: null  # Optional
echo: false
```

### ChromaDB Config

```yaml
backend: chromadb
persist_directory: chromadb_data  # Relative path
echo: false
```

### FAISS Config

```yaml
backend: faiss
index_path: faiss_data
dimension: 384
echo: false
```

## 💻 Code Examples

### Option 1: Direct Backend Selection

```python
from cad3d.super_ai.ai_db_factory import create_ai_database

# SQLite
db = create_ai_database(backend='sqlite', db_path='ai_models.db')

# PostgreSQL
db = create_ai_database(
    backend='postgresql',
    connection_string='postgresql://postgres:pass@localhost/aimodels'
)

# MySQL
db = create_ai_database(
    backend='mysql',
    connection_string='mysql+pymysql://root:pass@localhost/aimodels'
)

# Redis
db = create_ai_database(
    backend='redis',
    redis_host='localhost',
    redis_port=6379,
    redis_db=0
)

# ChromaDB (Vector persistent)
db = create_ai_database(
    backend='chromadb',
    persist_directory='chromadb_data'
)

# FAISS (Vector fast search)
db = create_ai_database(
    backend='faiss',
    index_path='faiss_data',
    dimension=384
)
```

### Option 2: From Config File

```python
from cad3d.super_ai.ai_db_factory import create_ai_database_from_config

db = create_ai_database_from_config('ai_db_config.yaml')
```

### Option 3: Auto-Detect (Recommended)

```python
from cad3d.super_ai.ai_db_factory import ai_db

# Checks: ai_db_config.yaml → AI_DB_CONFIG env var → SQLite fallback
db = ai_db()
```

## 🔧 Common Operations

### Create Model

```python
model_id = db.create_model(
    name="MyModel",
    description="My ML model",
    architecture="ResNet50",
    framework="PyTorch",
    task_type="classification"
)
```

### Create Version

```python
version_id = db.create_model_version(
    model_id=model_id,
    version="v1.0.0",
    checkpoint_path="models/checkpoint.pth",
    config={"layers": 50, "pretrained": True},
    status="active"
)
```

### Create Dataset

```python
dataset_id = db.create_dataset(
    name="MyDataset",
    description="Training dataset",
    source_path="/data/images",
    format="ImageFolder",
    num_samples=10000,
    split_info={"train": 8000, "val": 1000, "test": 1000}
)
```

### Training Run

```python
# Start run
run_id = db.create_training_run(
    model_version_id=version_id,
    dataset_id=dataset_id,
    run_name="baseline_run",
    status="running"
)

# Log hyperparameters
db.log_hyperparameters(run_id, {
    "learning_rate": 1e-4,
    "batch_size": 32,
    "num_epochs": 100,
    "optimizer": "AdamW"
})

# Log metrics
for epoch in range(1, 101):
    train_loss = ...  # Your training loop
    val_loss = ...
    db.log_metric(run_id, "loss", train_loss, epoch=epoch, split="train")
    db.log_metric(run_id, "loss", val_loss, epoch=epoch, split="val")

# Complete run
from datetime import datetime
db.update_training_run(
    run_id,
    status="completed",
    completed_at=datetime.utcnow().isoformat(),
    final_loss=0.15,
    best_metric_value=0.92
)
```

### Predictions

```python
db.log_prediction(
    model_version_id=version_id,
    input_data={"image": "cat.jpg"},
    output_data={"class": "cat", "confidence": 0.95},
    confidence=0.95,
    inference_time_ms=25.3
)
```

### Experiments

```python
exp_id = db.create_experiment(
    name="LR_Tuning",
    description="Test different learning rates",
    hypothesis="Lower LR improves accuracy",
    status="active"
)

db.add_experiment_run(exp_id, run_id, variant_name="lr_1e-4")
```

### Queries

```python
# Get model info
model = db.get_model(model_id)

# Get training run
run = db.get_training_run(run_id)

# Get metrics
metrics = db.get_metrics(run_id, metric_name="loss")

# Get predictions
predictions = db.get_predictions(version_id, limit=10)

# Get statistics
stats = db.get_statistics()
print(f"Models: {stats['models']}")
print(f"Training runs: {stats['training_runs']}")
```

## 🎯 Decision Guide

| Use Case | Recommended Backend |
|----------|---------------------|
| Local development | SQLite |
| Team collaboration | PostgreSQL |
| Production (scalability) | PostgreSQL |
| Production (web hosting) | MySQL |
| **Real-time inference** | **Redis** ⚡ |
| **Live metrics dashboard** | **Redis** ⚡ |
| Semantic model search | ChromaDB |
| High-speed similarity (large scale) | FAISS |
| Document-style flexible storage | MongoDB |
| Cloud deployment | PostgreSQL (RDS/Azure/GCP) |
| Prototyping | SQLite |
| Experiment tracking | SQLite or PostgreSQL |
| MLOps pipeline | PostgreSQL |
| **A/B testing** | **Redis** ⚡ |
| Model serving cache | Redis |
| Persistent embeddings (RAG) | ChromaDB |
| GPU-accelerated vector search | FAISS (GPU) |

## 📊 Backend Comparison

| Feature | SQLite | PostgreSQL | MySQL | **Redis** | ChromaDB | FAISS |
|---------|--------|------------|-------|-----------|----------|-------|---------|
| Setup | Zero config | Server required | Server required | Simple | Simple | Simple | Server required |
| **Latency** | ~10ms | ~5ms | ~5ms | **~0.1ms** ⚡ | ~10-50ms | ~1-5ms | ~5-15ms |
| Concurrency | Limited | Excellent | Good | Excellent | Excellent | Excellent | Excellent |
| Max data | ~1M rows | Billions | Billions | Memory-limited | Billions | Millions+ | Billions |
| JSON | Text serialization | Native JSONB | Native JSON | Native | Rich metadata | External (files) | Native document |
| Full-text search | No | Yes | Yes | Yes | Via embeddings | Via embeddings | Yes (Atlas Search) |
| Cloud hosting | No | Yes | Yes | Yes | Yes | Yes | Yes |
| Connection pooling | N/A | Yes | Yes | Built-in | Internal | N/A | Yes |
| **Persistence** | ✅ Always | ✅ Always | ✅ Always | ⚠️ Optional | ✅ Always | ✅ Manual | ✅ Always |
| **Pub/Sub** | ❌ No | ✅ LISTEN/NOTIFY | ❌ No | ✅ Native | ❌ No | ❌ No | ❌ (Use Change Streams) |
| **Best For** | Dev/local | Production | Web apps | Real-time | Semantic search | High-speed similarity | Flexible document storage |

## 🔐 Security Checklist

- [ ] Never commit passwords to Git
- [ ] Use `.env` files for local development
- [ ] Use environment variables in production
- [ ] Enable SSL/TLS for remote connections
- [ ] Restrict database user permissions
- [ ] Use connection pooling (SQL backends)
- [ ] Regular backups (automated)
- [ ] Monitor connection pool exhaustion
- [ ] Keep dependencies updated

## 🛠️ Troubleshooting

### Connection Failed

```powershell
# PostgreSQL
psql -U postgres -d aimodels  # Test connection
# Check: server running, firewall, pg_hba.conf

# MySQL
mysql -u root -p aimodels  # Test connection
# Check: server running, firewall, user permissions
```

### Import Errors

```powershell
pip install -r requirements-sql-database.txt
# Or individually:
pip install sqlalchemy psycopg2-binary pymysql pyyaml
```

### Database Doesn't Exist

```powershell
# PostgreSQL
createdb aimodels

# MySQL
mysql -u root -p
CREATE DATABASE aimodels;
```

### Permission Denied

```sql
-- PostgreSQL
GRANT ALL PRIVILEGES ON DATABASE aimodels TO postgres;

-- MySQL
GRANT ALL PRIVILEGES ON aimodels.* TO 'root'@'localhost';
FLUSH PRIVILEGES;
```

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `ai_models.db` | SQLite database (auto-created) |
| `ai_db_config.yaml` | Database configuration |
| `ai_db_config.yaml.template` | Config template |
| `demo_ai_database.py` | SQLite demo |
| `demo_ai_database_sql.py` | PostgreSQL/MySQL demo |
| `demo_ai_database_redis.py` | **Redis demo** |
| `demo_ai_database_vector.py` | Vector DB demo (ChromaDB & FAISS) |
| `demo_ai_database_mongo.py` | MongoDB demo |
| `requirements-sql-database.txt` | SQL dependencies |
| `requirements-redis-database.txt` | **Redis dependencies** |
| `requirements-vector-database.txt` | Vector DB dependencies |
| `requirements-mongo-database.txt` | MongoDB dependencies |
| `cad3d/super_ai/ai_model_database.py` | SQLite implementation |
| `cad3d/super_ai/ai_model_database_sql.py` | PostgreSQL/MySQL implementation |
| `cad3d/super_ai/ai_model_database_redis.py` | **Redis implementation** |
| `cad3d/super_ai/ai_model_database_vector.py` | **Vector (ChromaDB/FAISS) implementation** |
| `cad3d/super_ai/ai_model_database_mongo.py` | **MongoDB implementation** |
| `cad3d/super_ai/ai_db_factory.py` | Backend factory |

## 🧪 Testing Commands

```powershell
# SQLite
python demo_ai_database.py

# PostgreSQL
python demo_ai_database_sql.py --backend postgresql --user postgres --password yourpass

# MySQL
python demo_ai_database_sql.py --backend mysql --user root --password yourpass

# Redis
python demo_ai_database_redis.py
python demo_ai_database_redis.py --host localhost --port 6379 --db 0
python demo_ai_database_redis.py --password yourpass

# From config
python demo_ai_database_sql.py --config ai_db_config.yaml
python demo_ai_database_redis.py --config ai_db_config.yaml

# With SQL logging
python demo_ai_database_sql.py --backend postgresql --echo

# Redis monitoring
redis-cli ping
redis-cli monitor
redis-cli info memory

# Unit tests
pytest tests/test_ai_database.py -v
```

## 💡 Tips

1. **Start simple**: Use SQLite for initial development
2. **Real-time needs**: Use Redis for inference logging and live metrics
3. **Test locally**: Validate your code with SQLite before moving to production
4. **Config files**: Use `ai_db_config.yaml` for easy switching
5. **Connection pooling**: Tune `pool_size` based on concurrent users (SQL backends)
6. **Redis persistence**: Enable RDB + AOF for important data
7. **Monitor performance**: Use `--echo` flag (SQL) or `redis-cli monitor` (Redis)
8. **Backup regularly**: Automate with cron/scheduled tasks
9. **Use transactions**: SQLAlchemy handles this automatically
10. **Hybrid approach**: PostgreSQL for storage + Redis for real-time
11. **Index wisely**: Schema already includes strategic indexes
12. **Version control**: Track model versions and experiments
13. **Document experiments**: Use clear names and descriptions

## 🚀 Next Steps

1. Run SQLite demo: `python demo_ai_database.py`
2. Review schema: Check 9 tables in database
3. Explore API: Try all CRUD operations
4. Setup SQL backend: Install PostgreSQL/MySQL
5. Create config: Copy template and customize
6. Run SQL demo: Test PostgreSQL/MySQL connection
7. Integrate: Use in your training scripts
8. Monitor: Check statistics regularly
9. Backup: Setup automated backups
10. Scale: Move to cloud when ready

---

**All files in**: `E:\3d`  
**Quick help**: Check `README.md` sections "🤖 AI Model Database" and "🗄️ AI Model Database - SQL Backend"  
**Implementation details**: See `AI_DATABASE_SQL_IMPLEMENTATION.md`
