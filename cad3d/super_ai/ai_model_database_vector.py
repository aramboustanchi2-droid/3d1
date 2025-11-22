"""
AI Model Database - Vector Database Adapter (ChromaDB & FAISS)
پایگاه داده برداری برای ذخیره embeddings و جستجوی معنایی

Features:
- ChromaDB & FAISS as 4th & 5th database backends
- Store model embeddings for semantic search
- Vector similarity queries
- Perfect for RAG (Retrieval-Augmented Generation)
- Model version comparison via embeddings
- Dataset similarity analysis
- Same unified API

Installation:
    # ChromaDB
    pip install chromadb
    
    # FAISS
    pip install faiss-cpu  # or faiss-gpu for GPU support

Use Cases:
- Store model architecture embeddings
- Find similar models by description
- Semantic search across datasets
- Model recommendation system
- Duplicate detection
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import numpy as np
from pathlib import Path
import os


class AIModelDatabaseChroma:
    """مدیریت پایگاه داده ChromaDB برای embeddings مدل‌های AI"""
    
    def __init__(self, persist_directory: str = "chromadb_data"):
        """
        Args:
            persist_directory: Directory for persistent storage (relative to E:\3d)
        """
        import chromadb
        from chromadb.config import Settings
        
        # Make path absolute relative to project root
        if not Path(persist_directory).is_absolute():
            project_root = Path(__file__).parent.parent.parent  # E:\3d
            persist_directory = str(project_root / persist_directory)
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Create collections for different entity types
        self.models_collection = self.client.get_or_create_collection(
            name="ai_models",
            metadata={"description": "AI model metadata and embeddings"}
        )
        
        self.datasets_collection = self.client.get_or_create_collection(
            name="ai_datasets",
            metadata={"description": "Dataset metadata and embeddings"}
        )
        
        self.predictions_collection = self.client.get_or_create_collection(
            name="ai_predictions",
            metadata={"description": "Prediction embeddings for analysis"}
        )
        
        # Store other data in memory (non-vector data)
        self._versions = {}
        self._runs = {}
        self._hyperparams = {}
        self._metrics = {}
        self._experiments = {}
        self._experiment_runs = {}
        
        # ID counters
        self._next_model_id = 1
        self._next_version_id = 1
        self._next_dataset_id = 1
        self._next_run_id = 1
        self._next_experiment_id = 1
        self._next_prediction_id = 1

        # Embedding configuration (lazy real embedding support)
        self._embedding_backend = 'hash'  # 'transformer' once successfully loaded
        self._embedding_cache: Dict[str, List[float]] = {}
        self._embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        self._embedding_device = os.getenv("EMBEDDING_DEVICE", "auto")  # cpu | cuda | auto
        # Real embedding will be used automatically if sentence-transformers installed.
    
    def _ensure_transformer(self):
        """Attempt to load sentence-transformers model lazily."""
        if self._embedding_backend == 'transformer':
            return
        try:
            from sentence_transformers import SentenceTransformer  # local import
            if self._embedding_device == 'auto':
                model = SentenceTransformer(self._embedding_model_name)
            else:
                model = SentenceTransformer(self._embedding_model_name, device=self._embedding_device)
            self._transformer_model = model
            self._embedding_backend = 'transformer'
        except Exception:
            # Fallback silently to hash embeddings
            self._embedding_backend = 'hash'

    def _hash_embedding(self, text: str, dim: int = 384) -> List[float]:
        import hashlib
        hb = hashlib.sha256(text.encode()).digest()
        return [float(hb[i % len(hb)]) / 255.0 for i in range(dim)]

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding with optional sentence-transformers fallback to hash.
        Caches results to avoid recomputation.
        """
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        # Try real embeddings
        if self._embedding_backend != 'transformer':
            self._ensure_transformer()
        if self._embedding_backend == 'transformer':
            try:
                emb = self._transformer_model.encode([text], normalize_embeddings=True)[0]
                emb_list = emb.tolist()
                self._embedding_cache[text] = emb_list
                return emb_list
            except Exception:
                self._embedding_backend = 'hash'
        # Fallback hash
        emb_list = self._hash_embedding(text)
        self._embedding_cache[text] = emb_list
        return emb_list
    
    # Models
    def create_model(self, name: str, description: str = "", architecture: str = "", 
                     framework: str = "", task_type: str = "", 
                     input_shape: str = "", output_shape: str = "") -> int:
        """ثبت مدل جدید با embedding"""
        model_id = self._next_model_id
        self._next_model_id += 1
        
        now = datetime.utcnow().isoformat()
        
        # Create embedding from model description and architecture
        text_for_embedding = f"{name} {description} {architecture} {framework} {task_type}"
        embedding = self._generate_embedding(text_for_embedding)
        
        metadata = {
            'id': str(model_id),
            'name': name,
            'description': description,
            'architecture': architecture,
            'framework': framework,
            'task_type': task_type,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'created_at': now,
            'updated_at': now
        }
        
        # Store in ChromaDB with embedding
        self.models_collection.add(
            ids=[f"model_{model_id}"],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[text_for_embedding]
        )
        
        return model_id
    
    def get_model(self, model_id: int) -> Optional[Dict]:
        """دریافت اطلاعات مدل"""
        try:
            result = self.models_collection.get(
                ids=[f"model_{model_id}"],
                include=['metadatas']
            )
            if result['metadatas']:
                metadata = result['metadatas'][0]
                metadata['id'] = int(metadata['id'])
                return metadata
        except:
            pass
        return None
    
    def list_models(self) -> List[Dict]:
        """لیست همه مدل‌ها"""
        result = self.models_collection.get(include=['metadatas'])
        models = []
        for metadata in result['metadatas']:
            models.append({
                'id': int(metadata['id']),
                'name': metadata['name'],
                'architecture': metadata.get('architecture', ''),
                'framework': metadata.get('framework', ''),
                'task_type': metadata.get('task_type', ''),
                'created_at': metadata.get('created_at', '')
            })
        return sorted(models, key=lambda x: x['id'])
    
    def search_similar_models(self, query: str, n_results: int = 5) -> List[Dict]:
        """جستجوی معنایی مدل‌های مشابه"""
        query_embedding = self._generate_embedding(query)
        
        results = self.models_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['metadatas', 'distances']
        )
        
        similar_models = []
        if results['metadatas']:
            for i, metadata in enumerate(results['metadatas'][0]):
                similar_models.append({
                    'id': int(metadata['id']),
                    'name': metadata['name'],
                    'architecture': metadata.get('architecture', ''),
                    'similarity_score': 1.0 - results['distances'][0][i],  # Convert distance to similarity
                    'description': metadata.get('description', '')
                })
        
        return similar_models
    
    # Model Versions
    def create_model_version(self, model_id: int, version: str, 
                             checkpoint_path: str = "", config: Dict = None,
                             status: str = "active") -> int:
        """ایجاد نسخه جدید مدل"""
        version_id = self._next_version_id
        self._next_version_id += 1
        
        now = datetime.utcnow().isoformat()
        
        self._versions[version_id] = {
            'id': version_id,
            'model_id': model_id,
            'version': version,
            'checkpoint_path': checkpoint_path,
            'config': config or {},
            'status': status,
            'created_at': now
        }
        
        return version_id
    
    def get_model_versions(self, model_id: int) -> List[Dict]:
        """دریافت نسخه‌های یک مدل"""
        versions = [v for v in self._versions.values() if v['model_id'] == model_id]
        return sorted(versions, key=lambda x: x['id'], reverse=True)
    
    # Datasets
    def create_dataset(self, name: str, description: str = "", source_path: str = "",
                       format: str = "", size_bytes: int = 0, num_samples: int = 0,
                       split_info: Dict = None, preprocessing: Dict = None) -> int:
        """ثبت دیتاست با embedding"""
        dataset_id = self._next_dataset_id
        self._next_dataset_id += 1
        
        now = datetime.utcnow().isoformat()
        
        # Create embedding from dataset description
        text_for_embedding = f"{name} {description} {format}"
        embedding = self._generate_embedding(text_for_embedding)
        
        metadata = {
            'id': str(dataset_id),
            'name': name,
            'description': description,
            'source_path': source_path,
            'format': format,
            'size_bytes': str(size_bytes),
            'num_samples': str(num_samples),
            'split_info': json.dumps(split_info or {}),
            'preprocessing': json.dumps(preprocessing or {}),
            'created_at': now
        }
        
        # Store in ChromaDB
        self.datasets_collection.add(
            ids=[f"dataset_{dataset_id}"],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[text_for_embedding]
        )
        
        return dataset_id
    
    def get_dataset(self, dataset_id: int) -> Optional[Dict]:
        """دریافت اطلاعات دیتاست"""
        try:
            result = self.datasets_collection.get(
                ids=[f"dataset_{dataset_id}"],
                include=['metadatas']
            )
            if result['metadatas']:
                metadata = result['metadatas'][0]
                return {
                    'id': int(metadata['id']),
                    'name': metadata['name'],
                    'description': metadata['description'],
                    'source_path': metadata.get('source_path', ''),
                    'format': metadata.get('format', ''),
                    'size_bytes': int(metadata.get('size_bytes', 0)),
                    'num_samples': int(metadata.get('num_samples', 0)),
                    'split_info': json.loads(metadata.get('split_info', '{}')),
                    'preprocessing': json.loads(metadata.get('preprocessing', '{}')),
                    'created_at': metadata.get('created_at', '')
                }
        except:
            pass
        return None
    
    def search_similar_datasets(self, query: str, n_results: int = 5) -> List[Dict]:
        """جستجوی معنایی دیتاست‌های مشابه"""
        query_embedding = self._generate_embedding(query)
        
        results = self.datasets_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['metadatas', 'distances']
        )
        
        similar_datasets = []
        if results['metadatas']:
            for i, metadata in enumerate(results['metadatas'][0]):
                similar_datasets.append({
                    'id': int(metadata['id']),
                    'name': metadata['name'],
                    'format': metadata.get('format', ''),
                    'similarity_score': 1.0 - results['distances'][0][i],
                    'description': metadata.get('description', '')
                })
        
        return similar_datasets
    
    # Training Runs
    def create_training_run(self, model_version_id: int, dataset_id: int = None,
                            run_name: str = "", status: str = "started") -> int:
        """شروع آموزش جدید"""
        run_id = self._next_run_id
        self._next_run_id += 1
        
        now = datetime.utcnow().isoformat()
        
        self._runs[run_id] = {
            'id': run_id,
            'model_version_id': model_version_id,
            'dataset_id': dataset_id,
            'run_name': run_name,
            'status': status,
            'started_at': now,
            'completed_at': None,
            'duration_seconds': None,
            'final_loss': None,
            'best_metric_value': None,
            'artifacts': {}
        }
        
        return run_id
    
    def update_training_run(self, run_id: int, status: str = None, 
                            completed_at: str = None, duration_seconds: float = None,
                            final_loss: float = None, best_metric_value: float = None,
                            artifacts: Dict = None):
        """به‌روزرسانی وضعیت آموزش"""
        if run_id in self._runs:
            if status:
                self._runs[run_id]['status'] = status
            if completed_at:
                self._runs[run_id]['completed_at'] = completed_at
            if duration_seconds is not None:
                self._runs[run_id]['duration_seconds'] = duration_seconds
            if final_loss is not None:
                self._runs[run_id]['final_loss'] = final_loss
            if best_metric_value is not None:
                self._runs[run_id]['best_metric_value'] = best_metric_value
            if artifacts:
                self._runs[run_id]['artifacts'] = artifacts
    
    def get_training_run(self, run_id: int) -> Optional[Dict]:
        """دریافت اطلاعات یک آموزش"""
        return self._runs.get(run_id)
    
    # Hyperparameters
    def log_hyperparameters(self, training_run_id: int, params: Dict[str, Any]):
        """ثبت هایپرپارامترها"""
        self._hyperparams[training_run_id] = {k: str(v) for k, v in params.items()}
    
    def get_hyperparameters(self, training_run_id: int) -> Dict[str, str]:
        """دریافت هایپرپارامترها"""
        return self._hyperparams.get(training_run_id, {})
    
    # Metrics
    def log_metric(self, training_run_id: int, metric_name: str, metric_value: float,
                   epoch: int = None, step: int = None, split: str = "train"):
        """ثبت متریک"""
        if training_run_id not in self._metrics:
            self._metrics[training_run_id] = []
        
        self._metrics[training_run_id].append({
            'epoch': epoch,
            'step': step,
            'metric_name': metric_name,
            'metric_value': metric_value,
            'split': split,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def get_metrics(self, training_run_id: int, metric_name: str = None) -> List[Dict]:
        """دریافت متریک‌ها"""
        metrics = self._metrics.get(training_run_id, [])
        if metric_name:
            metrics = [m for m in metrics if m['metric_name'] == metric_name]
        return metrics
    
    # Experiments
    def create_experiment(self, name: str, description: str = "", 
                          hypothesis: str = "", status: str = "active") -> int:
        """ایجاد آزمایش جدید"""
        exp_id = self._next_experiment_id
        self._next_experiment_id += 1
        
        now = datetime.utcnow().isoformat()
        
        self._experiments[exp_id] = {
            'id': exp_id,
            'name': name,
            'description': description,
            'hypothesis': hypothesis,
            'status': status,
            'created_at': now,
            'completed_at': None,
            'results': {}
        }
        
        return exp_id
    
    def add_experiment_run(self, experiment_id: int, training_run_id: int,
                           variant_name: str = "", notes: str = ""):
        """اضافه کردن run به آزمایش"""
        if experiment_id not in self._experiment_runs:
            self._experiment_runs[experiment_id] = []
        
        self._experiment_runs[experiment_id].append({
            'training_run_id': training_run_id,
            'variant_name': variant_name,
            'notes': notes
        })
    
    def get_experiment_runs(self, experiment_id: int) -> List[Dict]:
        """دریافت runهای یک آزمایش"""
        exp_runs = self._experiment_runs.get(experiment_id, [])
        results = []
        
        for er in exp_runs:
            run = self.get_training_run(er['training_run_id'])
            if run:
                results.append({
                    'training_run_id': er['training_run_id'],
                    'variant_name': er['variant_name'],
                    'status': run['status'],
                    'final_loss': run['final_loss'],
                    'best_metric_value': run['best_metric_value']
                })
        
        return results
    
    # Predictions
    def log_prediction(self, model_version_id: int, input_data: Any, output_data: Any,
                       confidence: float = None, inference_time_ms: float = None,
                       metadata: Dict = None):
        """ثبت پیش‌بینی با embedding"""
        pred_id = self._next_prediction_id
        self._next_prediction_id += 1
        
        now = datetime.utcnow().isoformat()
        
        # Create embedding from prediction output
        output_text = json.dumps(output_data) if isinstance(output_data, dict) else str(output_data)
        embedding = self._generate_embedding(output_text)
        
        pred_metadata = {
            'id': str(pred_id),
            'model_version_id': str(model_version_id),
            'input': json.dumps(input_data if isinstance(input_data, dict) else {"data": input_data}),
            'output': json.dumps(output_data if isinstance(output_data, dict) else {"data": output_data}),
            'confidence': str(confidence or 0),
            'inference_time_ms': str(inference_time_ms or 0),
            'timestamp': now,
            'metadata': json.dumps(metadata or {})
        }
        
        # Store in ChromaDB
        self.predictions_collection.add(
            ids=[f"pred_{pred_id}"],
            embeddings=[embedding],
            metadatas=[pred_metadata],
            documents=[output_text]
        )
    
    def get_predictions(self, model_version_id: int, limit: int = 100) -> List[Dict]:
        """دریافت پیش‌بینی‌های یک مدل"""
        # Get all predictions (ChromaDB doesn't have perfect filtering, so we filter in Python)
        result = self.predictions_collection.get(include=['metadatas'])
        
        predictions = []
        for metadata in result['metadatas']:
            if int(metadata['model_version_id']) == model_version_id:
                predictions.append({
                    'input': json.loads(metadata['input']),
                    'output': json.loads(metadata['output']),
                    'confidence': float(metadata.get('confidence', 0)) or None,
                    'inference_time_ms': float(metadata.get('inference_time_ms', 0)) or None,
                    'timestamp': metadata['timestamp']
                })
        
        # Sort by timestamp (reverse) and limit
        predictions.sort(key=lambda x: x['timestamp'], reverse=True)
        return predictions[:limit]
    
    def search_similar_predictions(self, query: str, n_results: int = 10) -> List[Dict]:
        """جستجوی معنایی پیش‌بینی‌های مشابه"""
        query_embedding = self._generate_embedding(query)
        
        results = self.predictions_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['metadatas', 'distances']
        )
        
        similar_predictions = []
        if results['metadatas']:
            for i, metadata in enumerate(results['metadatas'][0]):
                similar_predictions.append({
                    'output': json.loads(metadata['output']),
                    'confidence': float(metadata.get('confidence', 0)),
                    'similarity_score': 1.0 - results['distances'][0][i],
                    'timestamp': metadata['timestamp']
                })
        
        return similar_predictions
    
    # Statistics
    def get_statistics(self) -> Dict[str, Any]:
        """آمار کلی پایگاه داده"""
        models_count = self.models_collection.count()
        datasets_count = self.datasets_collection.count()
        predictions_count = self.predictions_collection.count()
        
        completed_runs = sum(1 for r in self._runs.values() if r['status'] == 'completed')
        
        return {
            'models': models_count,
            'model_versions': len(self._versions),
            'datasets': datasets_count,
            'training_runs': len(self._runs),
            'completed_runs': completed_runs,
            'experiments': len(self._experiments),
            'predictions': predictions_count
        }


class AIModelDatabaseFAISS:
    """مدیریت پایگاه داده FAISS برای embeddings مدل‌های AI"""
    
    def __init__(self, index_path: str = "faiss_data", dimension: int = 384):
        """
        Args:
            index_path: Directory for FAISS index storage (relative to E:\3d)
            dimension: Embedding dimension (default: 384)
        """
        import faiss
        
        # Make path absolute
        if not Path(index_path).is_absolute():
            project_root = Path(__file__).parent.parent.parent
            index_path = str(project_root / index_path)
        
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        
        # Initialize FAISS indexes (one per entity type)
        self.models_index = faiss.IndexFlatL2(dimension)
        self.datasets_index = faiss.IndexFlatL2(dimension)
        self.predictions_index = faiss.IndexFlatL2(dimension)
        
        # Metadata storage (FAISS only stores vectors, not metadata)
        self._models_meta = []
        self._datasets_meta = []
        self._predictions_meta = []
        
        # Other data structures (same as ChromaDB)
        self._versions = {}
        self._runs = {}
        self._hyperparams = {}
        self._metrics = {}
        self._experiments = {}
        self._experiment_runs = {}
        
        # ID counters
        self._next_model_id = 1
        self._next_version_id = 1
        self._next_dataset_id = 1
        self._next_run_id = 1
        self._next_experiment_id = 1
        self._next_prediction_id = 1

        # Embedding configuration (shared with Chroma implementation)
        self._embedding_backend = 'hash'
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        self._embedding_device = os.getenv("EMBEDDING_DEVICE", "auto")
        
        # Load existing indexes if available
        self._load_indexes()
    
    def _load_indexes(self):
        """Load FAISS indexes from disk"""
        import faiss
        
        models_index_file = self.index_path / "models.index"
        datasets_index_file = self.index_path / "datasets.index"
        predictions_index_file = self.index_path / "predictions.index"
        
        if models_index_file.exists():
            self.models_index = faiss.read_index(str(models_index_file))
        
        if datasets_index_file.exists():
            self.datasets_index = faiss.read_index(str(datasets_index_file))
        
        if predictions_index_file.exists():
            self.predictions_index = faiss.read_index(str(predictions_index_file))
        
        # Load metadata
        models_meta_file = self.index_path / "models_meta.json"
        if models_meta_file.exists():
            with open(models_meta_file, 'r') as f:
                self._models_meta = json.load(f)
                if self._models_meta:
                    self._next_model_id = max(m['id'] for m in self._models_meta) + 1
        
        datasets_meta_file = self.index_path / "datasets_meta.json"
        if datasets_meta_file.exists():
            with open(datasets_meta_file, 'r') as f:
                self._datasets_meta = json.load(f)
                if self._datasets_meta:
                    self._next_dataset_id = max(d['id'] for d in self._datasets_meta) + 1
    
    def _save_indexes(self):
        """Save FAISS indexes to disk"""
        import faiss
        
        faiss.write_index(self.models_index, str(self.index_path / "models.index"))
        faiss.write_index(self.datasets_index, str(self.index_path / "datasets.index"))
        faiss.write_index(self.predictions_index, str(self.index_path / "predictions.index"))
        
        # Save metadata
        with open(self.index_path / "models_meta.json", 'w') as f:
            json.dump(self._models_meta, f)
        
        with open(self.index_path / "datasets_meta.json", 'w') as f:
            json.dump(self._datasets_meta, f)
    
    def _ensure_transformer(self):
        if self._embedding_backend == 'transformer':
            return
        try:
            from sentence_transformers import SentenceTransformer
            if self._embedding_device == 'auto':
                model = SentenceTransformer(self._embedding_model_name)
            else:
                model = SentenceTransformer(self._embedding_model_name, device=self._embedding_device)
            self._transformer_model = model
            self._embedding_backend = 'transformer'
        except Exception:
            self._embedding_backend = 'hash'

    def _hash_embedding(self, text: str) -> np.ndarray:
        import hashlib
        hb = hashlib.sha256(text.encode()).digest()
        vals = [float(hb[i % len(hb)]) / 255.0 for i in range(self.dimension)]
        return np.array(vals, dtype=np.float32)

    def _generate_embedding(self, text: str) -> np.ndarray:
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        if self._embedding_backend != 'transformer':
            self._ensure_transformer()
        if self._embedding_backend == 'transformer':
            try:
                emb = self._transformer_model.encode([text], normalize_embeddings=True)[0]
                # Adjust dimension if first time and indexes empty
                emb_len = len(emb)
                if emb_len != self.dimension:
                    if self.models_index.ntotal == 0 and self.datasets_index.ntotal == 0 and self.predictions_index.ntotal == 0:
                        # Recreate indexes with new dimension
                        import faiss
                        self.dimension = emb_len
                        self.models_index = faiss.IndexFlatL2(self.dimension)
                        self.datasets_index = faiss.IndexFlatL2(self.dimension)
                        self.predictions_index = faiss.IndexFlatL2(self.dimension)
                    else:
                        # Pad or truncate
                        if emb_len > self.dimension:
                            emb = emb[:self.dimension]
                        else:
                            emb = np.concatenate([emb, np.zeros(self.dimension - emb_len)], axis=0)
                emb_arr = np.array(emb, dtype=np.float32)
                self._embedding_cache[text] = emb_arr
                return emb_arr
            except Exception:
                self._embedding_backend = 'hash'
        emb_arr = self._hash_embedding(text)
        self._embedding_cache[text] = emb_arr
        return emb_arr
    
    # Models (same API as ChromaDB)
    def create_model(self, name: str, description: str = "", architecture: str = "", 
                     framework: str = "", task_type: str = "", 
                     input_shape: str = "", output_shape: str = "") -> int:
        model_id = self._next_model_id
        self._next_model_id += 1
        
        now = datetime.utcnow().isoformat()
        
        text_for_embedding = f"{name} {description} {architecture} {framework} {task_type}"
        embedding = self._generate_embedding(text_for_embedding)
        
        metadata = {
            'id': model_id,
            'name': name,
            'description': description,
            'architecture': architecture,
            'framework': framework,
            'task_type': task_type,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'created_at': now,
            'updated_at': now
        }
        
        # Add to FAISS index
        self.models_index.add(embedding.reshape(1, -1))
        self._models_meta.append(metadata)
        
        self._save_indexes()
        
        return model_id
    
    def get_model(self, model_id: int) -> Optional[Dict]:
        for model in self._models_meta:
            if model['id'] == model_id:
                return model.copy()
        return None
    
    def list_models(self) -> List[Dict]:
        return [{
            'id': m['id'],
            'name': m['name'],
            'architecture': m.get('architecture', ''),
            'framework': m.get('framework', ''),
            'task_type': m.get('task_type', ''),
            'created_at': m.get('created_at', '')
        } for m in self._models_meta]
    
    def search_similar_models(self, query: str, n_results: int = 5) -> List[Dict]:
        if self.models_index.ntotal == 0:
            return []
        
        query_embedding = self._generate_embedding(query).reshape(1, -1)
        distances, indices = self.models_index.search(query_embedding, min(n_results, self.models_index.ntotal))
        
        similar_models = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self._models_meta):
                model = self._models_meta[idx]
                similar_models.append({
                    'id': model['id'],
                    'name': model['name'],
                    'architecture': model.get('architecture', ''),
                    'similarity_score': 1.0 / (1.0 + distances[0][i]),  # Convert L2 distance to similarity
                    'description': model.get('description', '')
                })
        
        return similar_models
    
    # Datasets
    def create_dataset(self, name: str, description: str = "", source_path: str = "",
                       format: str = "", size_bytes: int = 0, num_samples: int = 0,
                       split_info: Dict = None, preprocessing: Dict = None) -> int:
        dataset_id = self._next_dataset_id
        self._next_dataset_id += 1
        
        now = datetime.utcnow().isoformat()
        
        text_for_embedding = f"{name} {description} {format}"
        embedding = self._generate_embedding(text_for_embedding)
        
        metadata = {
            'id': dataset_id,
            'name': name,
            'description': description,
            'source_path': source_path,
            'format': format,
            'size_bytes': size_bytes,
            'num_samples': num_samples,
            'split_info': split_info or {},
            'preprocessing': preprocessing or {},
            'created_at': now
        }
        
        self.datasets_index.add(embedding.reshape(1, -1))
        self._datasets_meta.append(metadata)
        
        self._save_indexes()
        
        return dataset_id
    
    def get_dataset(self, dataset_id: int) -> Optional[Dict]:
        for dataset in self._datasets_meta:
            if dataset['id'] == dataset_id:
                return dataset.copy()
        return None
    
    def search_similar_datasets(self, query: str, n_results: int = 5) -> List[Dict]:
        if self.datasets_index.ntotal == 0:
            return []
        
        query_embedding = self._generate_embedding(query).reshape(1, -1)
        distances, indices = self.datasets_index.search(query_embedding, min(n_results, self.datasets_index.ntotal))
        
        similar_datasets = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self._datasets_meta):
                dataset = self._datasets_meta[idx]
                similar_datasets.append({
                    'id': dataset['id'],
                    'name': dataset['name'],
                    'format': dataset.get('format', ''),
                    'similarity_score': 1.0 / (1.0 + distances[0][i]),
                    'description': dataset.get('description', '')
                })
        
        return similar_datasets
    
    # All other methods (same API as ChromaDB)
    def create_model_version(self, model_id: int, version: str, checkpoint_path: str = "", config: Dict = None, status: str = "active") -> int:
        version_id = self._next_version_id
        self._next_version_id += 1
        self._versions[version_id] = {'id': version_id, 'model_id': model_id, 'version': version, 'checkpoint_path': checkpoint_path, 'config': config or {}, 'status': status, 'created_at': datetime.utcnow().isoformat()}
        return version_id
    
    def get_model_versions(self, model_id: int) -> List[Dict]:
        return sorted([v for v in self._versions.values() if v['model_id'] == model_id], key=lambda x: x['id'], reverse=True)
    
    def create_training_run(self, model_version_id: int, dataset_id: int = None, run_name: str = "", status: str = "started") -> int:
        run_id = self._next_run_id
        self._next_run_id += 1
        self._runs[run_id] = {'id': run_id, 'model_version_id': model_version_id, 'dataset_id': dataset_id, 'run_name': run_name, 'status': status, 'started_at': datetime.utcnow().isoformat(), 'completed_at': None, 'duration_seconds': None, 'final_loss': None, 'best_metric_value': None, 'artifacts': {}}
        return run_id
    
    def update_training_run(self, run_id: int, status: str = None, completed_at: str = None, duration_seconds: float = None, final_loss: float = None, best_metric_value: float = None, artifacts: Dict = None):
        if run_id in self._runs:
            if status: self._runs[run_id]['status'] = status
            if completed_at: self._runs[run_id]['completed_at'] = completed_at
            if duration_seconds is not None: self._runs[run_id]['duration_seconds'] = duration_seconds
            if final_loss is not None: self._runs[run_id]['final_loss'] = final_loss
            if best_metric_value is not None: self._runs[run_id]['best_metric_value'] = best_metric_value
            if artifacts: self._runs[run_id]['artifacts'] = artifacts
    
    def get_training_run(self, run_id: int) -> Optional[Dict]:
        return self._runs.get(run_id)
    
    def log_hyperparameters(self, training_run_id: int, params: Dict[str, Any]):
        self._hyperparams[training_run_id] = {k: str(v) for k, v in params.items()}
    
    def get_hyperparameters(self, training_run_id: int) -> Dict[str, str]:
        return self._hyperparams.get(training_run_id, {})
    
    def log_metric(self, training_run_id: int, metric_name: str, metric_value: float, epoch: int = None, step: int = None, split: str = "train"):
        if training_run_id not in self._metrics:
            self._metrics[training_run_id] = []
        self._metrics[training_run_id].append({'epoch': epoch, 'step': step, 'metric_name': metric_name, 'metric_value': metric_value, 'split': split, 'timestamp': datetime.utcnow().isoformat()})
    
    def get_metrics(self, training_run_id: int, metric_name: str = None) -> List[Dict]:
        metrics = self._metrics.get(training_run_id, [])
        if metric_name:
            metrics = [m for m in metrics if m['metric_name'] == metric_name]
        return metrics
    
    def create_experiment(self, name: str, description: str = "", hypothesis: str = "", status: str = "active") -> int:
        exp_id = self._next_experiment_id
        self._next_experiment_id += 1
        self._experiments[exp_id] = {'id': exp_id, 'name': name, 'description': description, 'hypothesis': hypothesis, 'status': status, 'created_at': datetime.utcnow().isoformat(), 'completed_at': None, 'results': {}}
        return exp_id
    
    def add_experiment_run(self, experiment_id: int, training_run_id: int, variant_name: str = "", notes: str = ""):
        if experiment_id not in self._experiment_runs:
            self._experiment_runs[experiment_id] = []
        self._experiment_runs[experiment_id].append({'training_run_id': training_run_id, 'variant_name': variant_name, 'notes': notes})
    
    def get_experiment_runs(self, experiment_id: int) -> List[Dict]:
        exp_runs = self._experiment_runs.get(experiment_id, [])
        results = []
        for er in exp_runs:
            run = self.get_training_run(er['training_run_id'])
            if run:
                results.append({'training_run_id': er['training_run_id'], 'variant_name': er['variant_name'], 'status': run['status'], 'final_loss': run['final_loss'], 'best_metric_value': run['best_metric_value']})
        return results
    
    def log_prediction(self, model_version_id: int, input_data: Any, output_data: Any, confidence: float = None, inference_time_ms: float = None, metadata: Dict = None):
        pred_id = self._next_prediction_id
        self._next_prediction_id += 1
        
        output_text = json.dumps(output_data) if isinstance(output_data, dict) else str(output_data)
        embedding = self._generate_embedding(output_text)
        
        pred_metadata = {
            'id': pred_id,
            'model_version_id': model_version_id,
            'input': input_data if isinstance(input_data, dict) else {"data": input_data},
            'output': output_data if isinstance(output_data, dict) else {"data": output_data},
            'confidence': confidence,
            'inference_time_ms': inference_time_ms,
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': metadata or {}
        }
        
        self.predictions_index.add(embedding.reshape(1, -1))
        self._predictions_meta.append(pred_metadata)
    
    def get_predictions(self, model_version_id: int, limit: int = 100) -> List[Dict]:
        predictions = [p for p in self._predictions_meta if p['model_version_id'] == model_version_id]
        predictions.sort(key=lambda x: x['timestamp'], reverse=True)
        return predictions[:limit]
    
    def search_similar_predictions(self, query: str, n_results: int = 10) -> List[Dict]:
        if self.predictions_index.ntotal == 0:
            return []
        
        query_embedding = self._generate_embedding(query).reshape(1, -1)
        distances, indices = self.predictions_index.search(query_embedding, min(n_results, self.predictions_index.ntotal))
        
        similar_predictions = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self._predictions_meta):
                pred = self._predictions_meta[idx]
                similar_predictions.append({
                    'output': pred['output'],
                    'confidence': pred.get('confidence'),
                    'similarity_score': 1.0 / (1.0 + distances[0][i]),
                    'timestamp': pred['timestamp']
                })
        
        return similar_predictions
    
    def get_statistics(self) -> Dict[str, Any]:
        completed_runs = sum(1 for r in self._runs.values() if r['status'] == 'completed')
        return {
            'models': len(self._models_meta),
            'model_versions': len(self._versions),
            'datasets': len(self._datasets_meta),
            'training_runs': len(self._runs),
            'completed_runs': completed_runs,
            'experiments': len(self._experiments),
            'predictions': len(self._predictions_meta)
        }
