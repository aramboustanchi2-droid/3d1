"""
FAISS Advanced Indexes - IVF (Inverted File) and PQ (Product Quantization)
اندیس‌های پیشرفته FAISS برای میلیون‌ها بردار

Purpose:
- Handle millions to billions of vectors efficiently
- Reduce memory footprint with compression (PQ)
- Faster approximate search with clustering (IVF)
- Trade accuracy for speed (configurable)

Index Types:
1. IndexFlatL2: Exact search, small scale (<100K vectors)
2. IndexIVFFlat: Approximate, medium scale (100K-10M vectors)
3. IndexIVFPQ: Compressed approximate, large scale (10M-1B+ vectors)
4. IndexHNSW: Graph-based, very fast search

Features:
- Auto-select index type based on dataset size
- GPU acceleration support (optional)
- Training on sample data
- Incremental index building
- Index optimization and tuning

Installation:
    pip install faiss-cpu  # or faiss-gpu for GPU
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import os

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore


class AIModelDatabaseFAISSAdvanced:
    """
    Advanced FAISS backend with IVF/PQ indexes for large-scale vector search
    
    Index Selection Strategy:
    - <10K vectors: IndexFlatL2 (exact, fast)
    - 10K-100K: IndexIVFFlat (approximate, nlist=100)
    - 100K-1M: IndexIVFFlat (approximate, nlist=1000)
    - 1M-10M: IndexIVFPQ (compressed, nlist=4096, m=8)
    - 10M+: IndexIVFPQ (compressed, nlist=16384, m=16)
    """
    
    def __init__(
        self,
        index_path: str = "faiss_advanced_data",
        dimension: int = 384,
        index_type: str = "auto",
        nlist: Optional[int] = None,
        nprobe: int = 8,
        m: Optional[int] = None,
        nbits: int = 8,
        use_gpu: bool = False,
        gpu_id: int = 0
    ):
        """
        Args:
            index_path: Directory for index storage
            dimension: Embedding dimension
            index_type: 'auto', 'flat', 'ivf', 'ivfpq', 'hnsw'
            nlist: Number of clusters for IVF (default: sqrt(n_vectors))
            nprobe: Number of clusters to search (trade speed/accuracy)
            m: Number of sub-quantizers for PQ (dimension must be divisible by m)
            nbits: Bits per sub-quantizer (default: 8)
            use_gpu: Use GPU acceleration (requires faiss-gpu)
            gpu_id: GPU device ID
        """
        if faiss is None:
            raise ImportError("FAISS not installed. Run: pip install faiss-cpu")
        
        # Make path absolute
        if not Path(index_path).is_absolute():
            project_root = Path(__file__).parent.parent.parent
            index_path = str(project_root / index_path)
        
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        self.index_type = index_type
        self.nprobe = nprobe
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        
        # Auto-compute nlist and m if not provided
        self.nlist = nlist
        self.m = m or self._auto_m()
        self.nbits = nbits
        
        # Initialize indexes
        self.models_index = None
        self.datasets_index = None
        self.predictions_index = None
        
        # Training state
        self._models_trained = False
        self._datasets_trained = False
        self._predictions_trained = False
        
        # Metadata storage
        self._models_meta = []
        self._datasets_meta = []
        self._predictions_meta = []
        
        # Other data structures
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
        
        # Embedding configuration
        self._embedding_backend = 'hash'
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        self._embedding_device = os.getenv("EMBEDDING_DEVICE", "auto")
        
        # GPU resources
        self.gpu_res = None
        if use_gpu:
            self._init_gpu()
        
        # Load or create indexes
        self._load_or_create_indexes()
    
    def _auto_m(self) -> int:
        """Auto-compute m (sub-quantizers) based on dimension"""
        # m must divide dimension evenly
        # Common choices: 8, 16, 32, 64
        for m in [64, 32, 16, 8]:
            if self.dimension % m == 0:
                return m
        return 8  # fallback
    
    def _init_gpu(self):
        """Initialize GPU resources"""
        try:
            self.gpu_res = faiss.StandardGpuResources()
        except Exception as e:
            print(f"Warning: GPU initialization failed: {e}")
            self.use_gpu = False
    
    def _create_index(self, index_type: str, n_vectors: int = 0) -> Any:
        """
        Create FAISS index based on type and dataset size
        
        Args:
            index_type: 'flat', 'ivf', 'ivfpq', 'hnsw', 'auto'
            n_vectors: Current number of vectors (for auto-selection)
        
        Returns:
            FAISS index
        """
        # Auto-select index type
        if index_type == 'auto':
            if n_vectors < 10_000:
                index_type = 'flat'
            elif n_vectors < 100_000:
                index_type = 'ivf'
            else:
                index_type = 'ivfpq'
        
        # Create quantizer (for IVF-based indexes)
        quantizer = faiss.IndexFlatL2(self.dimension)
        
        if index_type == 'flat':
            # Exact search (brute force)
            index = faiss.IndexFlatL2(self.dimension)
        
        elif index_type == 'ivf':
            # Approximate search with clustering
            nlist = self.nlist or min(4096, max(100, int(np.sqrt(n_vectors))))
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            index.nprobe = self.nprobe
        
        elif index_type == 'ivfpq':
            # Compressed approximate search
            nlist = self.nlist or min(16384, max(1000, int(np.sqrt(n_vectors))))
            index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, self.m, self.nbits)
            index.nprobe = self.nprobe
        
        elif index_type == 'hnsw':
            # Graph-based search (very fast, high memory)
            M = 32  # number of connections per layer
            index = faiss.IndexHNSWFlat(self.dimension, M)
        
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Move to GPU if requested
        if self.use_gpu and self.gpu_res:
            try:
                index = faiss.index_cpu_to_gpu(self.gpu_res, self.gpu_id, index)
            except Exception as e:
                print(f"Warning: GPU transfer failed: {e}")
        
        return index
    
    def _load_or_create_indexes(self):
        """Load existing indexes or create new ones"""
        models_index_file = self.index_path / "models_advanced.index"
        datasets_index_file = self.index_path / "datasets_advanced.index"
        predictions_index_file = self.index_path / "predictions_advanced.index"
        
        # Load metadata first to know dataset size
        self._load_metadata()
        
        # Load or create indexes
        if models_index_file.exists():
            self.models_index = faiss.read_index(str(models_index_file))
            self._models_trained = True
        else:
            self.models_index = self._create_index(self.index_type, len(self._models_meta))
        
        if datasets_index_file.exists():
            self.datasets_index = faiss.read_index(str(datasets_index_file))
            self._datasets_trained = True
        else:
            self.datasets_index = self._create_index(self.index_type, len(self._datasets_meta))
        
        if predictions_index_file.exists():
            self.predictions_index = faiss.read_index(str(predictions_index_file))
            self._predictions_trained = True
        else:
            self.predictions_index = self._create_index(self.index_type, len(self._predictions_meta))
    
    def _load_metadata(self):
        """Load metadata from disk"""
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
        
        predictions_meta_file = self.index_path / "predictions_meta.json"
        if predictions_meta_file.exists():
            with open(predictions_meta_file, 'r') as f:
                self._predictions_meta = json.load(f)
                if self._predictions_meta:
                    self._next_prediction_id = max(p['id'] for p in self._predictions_meta) + 1
    
    def _save_indexes(self):
        """Save indexes and metadata to disk"""
        # Convert GPU index to CPU before saving
        models_index = faiss.index_gpu_to_cpu(self.models_index) if self.use_gpu else self.models_index
        datasets_index = faiss.index_gpu_to_cpu(self.datasets_index) if self.use_gpu else self.datasets_index
        predictions_index = faiss.index_gpu_to_cpu(self.predictions_index) if self.use_gpu else self.predictions_index
        
        faiss.write_index(models_index, str(self.index_path / "models_advanced.index"))
        faiss.write_index(datasets_index, str(self.index_path / "datasets_advanced.index"))
        faiss.write_index(predictions_index, str(self.index_path / "predictions_advanced.index"))
        
        # Save metadata
        with open(self.index_path / "models_meta.json", 'w') as f:
            json.dump(self._models_meta, f)
        
        with open(self.index_path / "datasets_meta.json", 'w') as f:
            json.dump(self._datasets_meta, f)
        
        with open(self.index_path / "predictions_meta.json", 'w') as f:
            json.dump(self._predictions_meta, f)
    
    def _train_index(self, index: Any, vectors: np.ndarray):
        """
        Train IVF/PQ index on sample data
        
        Args:
            index: FAISS index to train
            vectors: Training vectors (numpy array, shape: [n_samples, dimension])
        """
        if not hasattr(index, 'is_trained') or index.is_trained:
            return
        
        # Need at least nlist*39 vectors for training (FAISS requirement)
        nlist = getattr(index, 'nlist', 0)
        min_train = max(1000, nlist * 39)
        
        if len(vectors) < min_train:
            print(f"Warning: Not enough vectors for training ({len(vectors)} < {min_train}). Using available data.")
        
        index.train(vectors)
    
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
                emb_arr = np.array(emb, dtype=np.float32)
                
                # Ensure dimension matches
                if len(emb_arr) != self.dimension:
                    if len(emb_arr) > self.dimension:
                        emb_arr = emb_arr[:self.dimension]
                    else:
                        emb_arr = np.concatenate([emb_arr, np.zeros(self.dimension - len(emb_arr))], axis=0)
                
                self._embedding_cache[text] = emb_arr
                return emb_arr
            except Exception:
                self._embedding_backend = 'hash'
        
        emb_arr = self._hash_embedding(text)
        self._embedding_cache[text] = emb_arr
        return emb_arr
    
    # ========== Models ==========
    
    def create_model(
        self,
        name: str,
        description: str = "",
        architecture: str = "",
        framework: str = "",
        task_type: str = "",
        input_shape: str = "",
        output_shape: str = ""
    ) -> int:
        """Create model with embedding and add to index"""
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
        
        # Train index if needed (IVF/PQ requires training)
        if not self._models_trained and hasattr(self.models_index, 'is_trained'):
            # Collect training data (use first batch)
            if len(self._models_meta) == 0:
                # First vector, can't train yet
                pass
            elif len(self._models_meta) >= 100:  # Train after 100 samples
                # Gather training vectors
                train_vectors = []
                for m in self._models_meta[:min(1000, len(self._models_meta))]:
                    text = f"{m['name']} {m.get('description', '')} {m.get('architecture', '')}"
                    vec = self._generate_embedding(text)
                    train_vectors.append(vec)
                
                train_vectors = np.array(train_vectors, dtype=np.float32)
                self._train_index(self.models_index, train_vectors)
                self._models_trained = True
        
        # Add to index
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
        
        # Search
        n = min(n_results, self.models_index.ntotal)
        distances, indices = self.models_index.search(query_embedding, n)
        
        similar_models = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self._models_meta) and idx >= 0:
                model = self._models_meta[idx]
                similar_models.append({
                    'id': model['id'],
                    'name': model['name'],
                    'architecture': model.get('architecture', ''),
                    'similarity_score': 1.0 / (1.0 + distances[0][i]),
                    'description': model.get('description', '')
                })
        
        return similar_models
    
    # ========== Datasets ==========
    
    def create_dataset(
        self,
        name: str,
        description: str = "",
        source_path: str = "",
        format: str = "",
        size_bytes: int = 0,
        num_samples: int = 0,
        split_info: Dict = None,
        preprocessing: Dict = None
    ) -> int:
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
        
        # Train if needed
        if not self._datasets_trained and hasattr(self.datasets_index, 'is_trained') and len(self._datasets_meta) >= 100:
            train_vectors = []
            for d in self._datasets_meta[:1000]:
                text = f"{d['name']} {d.get('description', '')} {d.get('format', '')}"
                vec = self._generate_embedding(text)
                train_vectors.append(vec)
            train_vectors = np.array(train_vectors, dtype=np.float32)
            self._train_index(self.datasets_index, train_vectors)
            self._datasets_trained = True
        
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
        n = min(n_results, self.datasets_index.ntotal)
        distances, indices = self.datasets_index.search(query_embedding, n)
        
        similar_datasets = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self._datasets_meta) and idx >= 0:
                dataset = self._datasets_meta[idx]
                similar_datasets.append({
                    'id': dataset['id'],
                    'name': dataset['name'],
                    'format': dataset.get('format', ''),
                    'similarity_score': 1.0 / (1.0 + distances[0][i]),
                    'description': dataset.get('description', '')
                })
        
        return similar_datasets
    
    # ========== Remaining methods (same API as basic FAISS) ==========
    
    def create_model_version(self, model_id: int, version: str, checkpoint_path: str = "", config: Dict = None, status: str = "active") -> int:
        version_id = self._next_version_id
        self._next_version_id += 1
        self._versions[version_id] = {
            'id': version_id,
            'model_id': model_id,
            'version': version,
            'checkpoint_path': checkpoint_path,
            'config': config or {},
            'status': status,
            'created_at': datetime.utcnow().isoformat()
        }
        return version_id
    
    def get_model_versions(self, model_id: int) -> List[Dict]:
        return sorted([v for v in self._versions.values() if v['model_id'] == model_id], key=lambda x: x['id'], reverse=True)
    
    def create_training_run(self, model_version_id: int, dataset_id: int = None, run_name: str = "", status: str = "started") -> int:
        run_id = self._next_run_id
        self._next_run_id += 1
        self._runs[run_id] = {
            'id': run_id,
            'model_version_id': model_version_id,
            'dataset_id': dataset_id,
            'run_name': run_name,
            'status': status,
            'started_at': datetime.utcnow().isoformat(),
            'completed_at': None,
            'duration_seconds': None,
            'final_loss': None,
            'best_metric_value': None,
            'artifacts': {}
        }
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
        self._metrics[training_run_id].append({
            'epoch': epoch,
            'step': step,
            'metric_name': metric_name,
            'metric_value': metric_value,
            'split': split,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def get_metrics(self, training_run_id: int, metric_name: str = None) -> List[Dict]:
        metrics = self._metrics.get(training_run_id, [])
        if metric_name:
            metrics = [m for m in metrics if m['metric_name'] == metric_name]
        return metrics
    
    def create_experiment(self, name: str, description: str = "", hypothesis: str = "", status: str = "active") -> int:
        exp_id = self._next_experiment_id
        self._next_experiment_id += 1
        self._experiments[exp_id] = {
            'id': exp_id,
            'name': name,
            'description': description,
            'hypothesis': hypothesis,
            'status': status,
            'created_at': datetime.utcnow().isoformat(),
            'completed_at': None,
            'results': {}
        }
        return exp_id
    
    def add_experiment_run(self, experiment_id: int, training_run_id: int, variant_name: str = "", notes: str = ""):
        if experiment_id not in self._experiment_runs:
            self._experiment_runs[experiment_id] = []
        self._experiment_runs[experiment_id].append({
            'training_run_id': training_run_id,
            'variant_name': variant_name,
            'notes': notes
        })
    
    def get_experiment_runs(self, experiment_id: int) -> List[Dict]:
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
        
        # Train predictions index if needed
        if not self._predictions_trained and hasattr(self.predictions_index, 'is_trained') and len(self._predictions_meta) >= 100:
            train_vectors = []
            for p in self._predictions_meta[:1000]:
                text = json.dumps(p.get('output', {}))
                vec = self._generate_embedding(text)
                train_vectors.append(vec)
            train_vectors = np.array(train_vectors, dtype=np.float32)
            self._train_index(self.predictions_index, train_vectors)
            self._predictions_trained = True
        
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
        n = min(n_results, self.predictions_index.ntotal)
        distances, indices = self.predictions_index.search(query_embedding, n)
        
        similar_predictions = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self._predictions_meta) and idx >= 0:
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
        
        # Index info
        index_info = {
            'models_index_type': type(self.models_index).__name__,
            'models_trained': self._models_trained,
            'datasets_index_type': type(self.datasets_index).__name__,
            'datasets_trained': self._datasets_trained,
            'predictions_index_type': type(self.predictions_index).__name__,
            'predictions_trained': self._predictions_trained,
            'nprobe': self.nprobe,
            'use_gpu': self.use_gpu
        }
        
        return {
            'models': len(self._models_meta),
            'model_versions': len(self._versions),
            'datasets': len(self._datasets_meta),
            'training_runs': len(self._runs),
            'completed_runs': completed_runs,
            'experiments': len(self._experiments),
            'predictions': len(self._predictions_meta),
            'index_info': index_info
        }
    
    def optimize_index(self, collection: str = 'models'):
        """
        Optimize index for better search performance
        
        Args:
            collection: 'models', 'datasets', or 'predictions'
        """
        if collection == 'models':
            index = self.models_index
            meta = self._models_meta
            trained_flag = '_models_trained'
        elif collection == 'datasets':
            index = self.datasets_index
            meta = self._datasets_meta
            trained_flag = '_datasets_trained'
        elif collection == 'predictions':
            index = self.predictions_index
            meta = self._predictions_meta
            trained_flag = '_predictions_trained'
        else:
            raise ValueError(f"Unknown collection: {collection}")
        
        n_vectors = len(meta)
        
        # If small dataset, convert to Flat
        if n_vectors < 10_000 and not isinstance(index, faiss.IndexFlat):
            print(f"Optimizing {collection}: Converting to IndexFlatL2 (n={n_vectors})")
            new_index = faiss.IndexFlatL2(self.dimension)
            
            # Rebuild index
            all_vectors = []
            for item in meta:
                if collection == 'models':
                    text = f"{item['name']} {item.get('description', '')}"
                elif collection == 'datasets':
                    text = f"{item['name']} {item.get('description', '')}"
                else:
                    text = json.dumps(item.get('output', {}))
                vec = self._generate_embedding(text)
                all_vectors.append(vec)
            
            all_vectors = np.array(all_vectors, dtype=np.float32)
            new_index.add(all_vectors)
            
            # Replace index
            if collection == 'models':
                self.models_index = new_index
            elif collection == 'datasets':
                self.datasets_index = new_index
            else:
                self.predictions_index = new_index
            
            self._save_indexes()
        
        # If large dataset, ensure trained and tuned
        elif n_vectors >= 10_000 and hasattr(index, 'is_trained'):
            if not getattr(self, trained_flag):
                print(f"Training {collection} index (n={n_vectors})...")
                # Gather all vectors for training
                all_vectors = []
                for item in meta:
                    if collection == 'models':
                        text = f"{item['name']} {item.get('description', '')}"
                    elif collection == 'datasets':
                        text = f"{item['name']} {item.get('description', '')}"
                    else:
                        text = json.dumps(item.get('output', {}))
                    vec = self._generate_embedding(text)
                    all_vectors.append(vec)
                
                all_vectors = np.array(all_vectors, dtype=np.float32)
                self._train_index(index, all_vectors)
                setattr(self, trained_flag, True)
                self._save_indexes()
                print(f"  ✓ {collection} index trained")
