# AI Model Database - MongoDB Backend Implementation Guide

MongoDB adds a **flexible document-oriented storage** layer to the AI Model Database system as the sixth backend (after SQLite, PostgreSQL/MySQL, Redis, ChromaDB, FAISS).

## ✅ Key Features

- Unified API (same method names as other backends)
- Atomic integer IDs via a `counters` collection
- Collections mapped 1:1 to logical entities
- Indexes for performance (model_id, training_run_id, timestamps)
- Optional `mongomock` in-memory mode for development/testing without a running server
- Easy integration through the factory (`create_ai_database(backend='mongodb', ...)`)

## 🧱 Collections

| Collection | Purpose | Indexes |
|------------|---------|---------|
| models | Model metadata | id (unique), created_at |
| model_versions | Version info per model | id (unique), model_id |
| datasets | Dataset metadata | id (unique) |
| training_runs | Training lifecycle | id (unique), model_version_id |
| hyperparams | Stored hyperparameters | training_run_id |
| metrics | Time-series & performance metrics | training_run_id+metric_name+epoch |
| predictions | Inference logging | model_version_id, timestamp |
| experiments | Experiment definitions | id (unique) |
| experiment_runs | Mapping experiments → runs | experiment_id |
| counters | Atomic ID generator | _id (counter name) |

## 🔢 ID Generation Strategy

Uses a `counters` collection with atomic `$inc` via `find_one_and_update` for each entity type.

```python
def _next_id(self, counter_name: str) -> int:
    doc = self.col_counters.find_one_and_update(
        {'_id': counter_name},
        {'$inc': {'seq': 1}},
        upsert=True,
        return_document=True
    )
    return int(doc['seq'])
```

This ensures consistent integer IDs across documents even though MongoDB normally uses ObjectIds.

## 🏗️ Initialization

```python
from cad3d.super_ai.ai_db_factory import create_ai_database

# Direct host/port
mdb = create_ai_database(
    backend='mongodb',
    mongo_host='localhost',
    mongo_port=27017,
    mongo_database='aidb'
)

# Connection string
mdb = create_ai_database(
    backend='mongodb',
    connection_string='mongodb://user:pass@localhost:27017/aidb'
)

# In-memory mock (no server required)
mdb = create_ai_database(backend='mongodb', mongo_use_mock=True)
```

## 💾 Configuration (ai_db_config.yaml)

```yaml
backend: mongodb
connection:
  host: localhost
  port: 27017
  database: aidb
  user: myuser      # optional
  password: secret  # optional
use_mock: false
```

Or direct string:

```yaml
backend: mongodb
connection_string: "mongodb://myuser:secret@localhost:27017/aidb"
use_mock: false
```

## 🧪 Demo

Run the included demo:

```powershell
python demo_ai_database_mongo.py --host localhost --port 27017 --db aidb
python demo_ai_database_mongo.py --mock  # mongomock
```

## 🔁 Unified API (Selected Methods)

```python
model_id = mdb.create_model(name="ResNet50", architecture="ResNet50", framework="PyTorch")
version_id = mdb.create_model_version(model_id, version="1.0.0")
dataset_id = mdb.create_dataset(name="ImageNet", format="JPEG")
run_id = mdb.create_training_run(model_version_id=version_id, dataset_id=dataset_id)
mdb.log_hyperparameters(run_id, {"lr": 1e-3})
mdb.log_metric(run_id, "loss", 0.42, epoch=1)
mdb.log_prediction(version_id, {"image": "cat.jpg"}, {"class": "cat"}, confidence=0.93)
exp_id = mdb.create_experiment(name="LR_Sweep")
mdb.add_experiment_run(exp_id, run_id, variant_name="lr_1e-3")
metrics = mdb.get_metrics(run_id, metric_name="loss")
stats = mdb.get_statistics()
```

## 📊 Statistics

`get_statistics()` aggregates counts across all collections for quick dashboard summaries.

## ⚙️ Indexes

Automatically created on initialization (`create_indexes=True`). Adjust or extend for advanced queries (e.g., compound index on metrics for faster range scans).

## 🛡️ Error Handling & Health

- `ping()` executes an admin command to verify connectivity.
- All operations assume connectivity; wrap calls in try/except for production resilience.

## 🧪 Testing Strategies

- Use `mongo_use_mock=True` to test logic without a running MongoDB.
- For integration tests, spin up a temporary container:

```powershell
docker run -d -p 27017:27017 --name aidb-mongo mongo:7
```

## 🚀 Performance Considerations

| Operation | Expected Latency |
|-----------|------------------|
| Insert single doc | ~5–15 ms local |
| Bulk insert (100 docs) | ~50–120 ms |
| Indexed query | ~5–20 ms |
| Aggregation (simple) | ~10–50 ms |

For higher throughput:

- Use bulk writes for metrics/predictions.
- Consider sharding for very large scale.
- Use change streams for reactive processing.

## 🔄 Comparison vs Other Backends

| Backend | Strength |
|---------|----------|
| SQLite | Simplicity/local dev |
| PostgreSQL | Relational integrity, advanced queries |
| MySQL | Ubiquity in web hosting |
| Redis | Real-time, in-memory speed |
| ChromaDB | Persistent vector embeddings |
| FAISS | High-speed similarity search |
| MongoDB | Flexible document modeling + horizontal scaling |

## 📝 When To Choose MongoDB

- Need flexible schema evolution
- High write rates with variable document shapes
- Store JSON-like structures directly (no serialization layer)
- Combine with Redis (real-time) and ChromaDB/FAISS (semantic) for hybrid architecture

## 🔐 Security Notes

- Use SCRAM authentication (default in modern MongoDB)
- Enable TLS/SSL in production
- Principle of least privilege for database users
- Enable backups (mongodump/mongorestore or cloud snapshots)

## 🧹 Cleanup

Drop the database (CAREFUL):

```python
mdb.drop_database()
```

## ✅ Summary

MongoDB backend fully integrated with unified API, offering flexible, schema-less storage that complements existing relational, in-memory, and vector backends.

---
Sixth backend completed ✔
