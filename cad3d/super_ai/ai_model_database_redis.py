"""
AI Model Database - Redis Adapter
پایگاه داده مدل‌های هوش مصنوعی با پشتیبانی Redis

Features:
- Redis as third database backend (alongside SQLite, PostgreSQL, MySQL)
- In-memory storage with optional persistence
- High-speed access (microsecond latency)
- JSON serialization for complex data
- Key-value + Hash + Sorted Set patterns
- Perfect for real-time predictions and metrics
- Same API as other backends

Installation:
    pip install redis

Redis Setup:
    # Windows: Download from https://github.com/microsoftarchive/redis/releases
    # Or use Docker: docker run -d -p 6379:6379 redis
    # Or WSL: sudo apt install redis-server && redis-server
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import redis
from pathlib import Path


class AIModelDatabaseRedis:
    """مدیریت پایگاه داده Redis برای مدل‌های هوش مصنوعی"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, 
                 db: int = 0, password: Optional[str] = None,
                 decode_responses: bool = True):
        """
        Args:
            host: Redis server host
            port: Redis server port (default: 6379)
            db: Redis database number (0-15)
            password: Redis password (if auth enabled)
            decode_responses: Auto-decode bytes to strings
        """
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=decode_responses
        )
        # Test connection
        self.client.ping()
        
        # Key prefixes for namespacing
        self.PREFIX_MODEL = "aidb:model:"
        self.PREFIX_VERSION = "aidb:version:"
        self.PREFIX_DATASET = "aidb:dataset:"
        self.PREFIX_RUN = "aidb:run:"
        self.PREFIX_HYPERPARAM = "aidb:hyperparam:"
        self.PREFIX_METRIC = "aidb:metric:"
        self.PREFIX_EXPERIMENT = "aidb:experiment:"
        self.PREFIX_EXPRUN = "aidb:exprun:"
        self.PREFIX_PREDICTION = "aidb:prediction:"
        
        # Index keys (sorted sets for listings)
        self.IDX_MODELS = "aidb:idx:models"
        self.IDX_VERSIONS = "aidb:idx:versions:"  # + model_id
        self.IDX_DATASETS = "aidb:idx:datasets"
        self.IDX_RUNS = "aidb:idx:runs"
        self.IDX_EXPERIMENTS = "aidb:idx:experiments"
        self.IDX_PREDICTIONS = "aidb:idx:predictions:"  # + version_id
        
        # Counters for auto-increment IDs
        self.COUNTER_MODEL = "aidb:counter:model"
        self.COUNTER_VERSION = "aidb:counter:version"
        self.COUNTER_DATASET = "aidb:counter:dataset"
        self.COUNTER_RUN = "aidb:counter:run"
        self.COUNTER_EXPERIMENT = "aidb:counter:experiment"
        self.COUNTER_PREDICTION = "aidb:counter:prediction"
    
    def _get_next_id(self, counter_key: str) -> int:
        """Get next auto-increment ID"""
        return self.client.incr(counter_key)
    
    def _json_encode(self, obj: Any) -> str:
        """Encode object to JSON string"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        return json.dumps(obj, default=str)
    
    def _json_decode(self, data: str) -> Any:
        """Decode JSON string to object"""
        if data is None:
            return None
        return json.loads(data)
    
    # Models
    def create_model(self, name: str, description: str = "", architecture: str = "", 
                     framework: str = "", task_type: str = "", 
                     input_shape: str = "", output_shape: str = "") -> int:
        """ثبت مدل جدید"""
        model_id = self._get_next_id(self.COUNTER_MODEL)
        now = datetime.utcnow().isoformat()
        
        model_data = {
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
        
        # Store as hash
        key = f"{self.PREFIX_MODEL}{model_id}"
        self.client.hset(key, mapping={k: v for k, v in model_data.items()})
        
        # Add to index (sorted by ID for ordering)
        self.client.zadd(self.IDX_MODELS, {model_id: model_id})
        
        # Store name->id mapping for lookups
        self.client.hset(f"{self.PREFIX_MODEL}name_to_id", name, model_id)
        
        return model_id
    
    def get_model(self, model_id: int) -> Optional[Dict]:
        """دریافت اطلاعات مدل"""
        key = f"{self.PREFIX_MODEL}{model_id}"
        data = self.client.hgetall(key)
        
        if not data:
            return None
        
        # Convert ID fields to int
        if 'id' in data:
            data['id'] = int(data['id'])
        
        return data
    
    def list_models(self) -> List[Dict]:
        """لیست همه مدل‌ها"""
        # Get all model IDs from index (ordered)
        model_ids = self.client.zrange(self.IDX_MODELS, 0, -1)
        
        models = []
        for model_id in model_ids:
            model = self.get_model(int(model_id))
            if model:
                # Return summary fields
                models.append({
                    'id': model['id'],
                    'name': model['name'],
                    'architecture': model.get('architecture', ''),
                    'framework': model.get('framework', ''),
                    'task_type': model.get('task_type', ''),
                    'created_at': model.get('created_at', '')
                })
        
        return models
    
    # Model Versions
    def create_model_version(self, model_id: int, version: str, 
                             checkpoint_path: str = "", config: Dict = None,
                             status: str = "active") -> int:
        """ایجاد نسخه جدید مدل"""
        version_id = self._get_next_id(self.COUNTER_VERSION)
        now = datetime.utcnow().isoformat()
        
        version_data = {
            'id': version_id,
            'model_id': model_id,
            'version': version,
            'checkpoint_path': checkpoint_path,
            'config': self._json_encode(config or {}),
            'status': status,
            'created_at': now
        }
        
        # Store as hash
        key = f"{self.PREFIX_VERSION}{version_id}"
        self.client.hset(key, mapping=version_data)
        
        # Add to model's version index
        idx_key = f"{self.IDX_VERSIONS}{model_id}"
        self.client.zadd(idx_key, {version_id: version_id})
        
        return version_id
    
    def get_model_versions(self, model_id: int) -> List[Dict]:
        """دریافت نسخه‌های یک مدل"""
        idx_key = f"{self.IDX_VERSIONS}{model_id}"
        version_ids = self.client.zrevrange(idx_key, 0, -1)  # Reverse order (newest first)
        
        versions = []
        for version_id in version_ids:
            key = f"{self.PREFIX_VERSION}{version_id}"
            data = self.client.hgetall(key)
            if data:
                versions.append({
                    'id': int(data['id']),
                    'version': data['version'],
                    'checkpoint_path': data.get('checkpoint_path', ''),
                    'status': data.get('status', ''),
                    'created_at': data.get('created_at', '')
                })
        
        return versions
    
    # Datasets
    def create_dataset(self, name: str, description: str = "", source_path: str = "",
                       format: str = "", size_bytes: int = 0, num_samples: int = 0,
                       split_info: Dict = None, preprocessing: Dict = None) -> int:
        """ثبت دیتاست"""
        dataset_id = self._get_next_id(self.COUNTER_DATASET)
        now = datetime.utcnow().isoformat()
        
        dataset_data = {
            'id': dataset_id,
            'name': name,
            'description': description,
            'source_path': source_path,
            'format': format,
            'size_bytes': size_bytes,
            'num_samples': num_samples,
            'split_info': self._json_encode(split_info or {}),
            'preprocessing': self._json_encode(preprocessing or {}),
            'created_at': now
        }
        
        # Store as hash
        key = f"{self.PREFIX_DATASET}{dataset_id}"
        self.client.hset(key, mapping=dataset_data)
        
        # Add to index
        self.client.zadd(self.IDX_DATASETS, {dataset_id: dataset_id})
        
        return dataset_id
    
    def get_dataset(self, dataset_id: int) -> Optional[Dict]:
        """دریافت اطلاعات دیتاست"""
        key = f"{self.PREFIX_DATASET}{dataset_id}"
        data = self.client.hgetall(key)
        
        if not data:
            return None
        
        # Parse JSON fields
        data['id'] = int(data['id'])
        data['size_bytes'] = int(data.get('size_bytes', 0))
        data['num_samples'] = int(data.get('num_samples', 0))
        data['split_info'] = self._json_decode(data.get('split_info', '{}'))
        data['preprocessing'] = self._json_decode(data.get('preprocessing', '{}'))
        
        return data
    
    # Training Runs
    def create_training_run(self, model_version_id: int, dataset_id: int = None,
                            run_name: str = "", status: str = "started") -> int:
        """شروع آموزش جدید"""
        run_id = self._get_next_id(self.COUNTER_RUN)
        now = datetime.utcnow().isoformat()
        
        run_data = {
            'id': run_id,
            'model_version_id': model_version_id,
            'dataset_id': dataset_id or 0,
            'run_name': run_name,
            'status': status,
            'started_at': now,
            'completed_at': '',
            'duration_seconds': 0,
            'final_loss': 0,
            'best_metric_value': 0,
            'artifacts': '{}'
        }
        
        # Store as hash
        key = f"{self.PREFIX_RUN}{run_id}"
        self.client.hset(key, mapping=run_data)
        
        # Add to index
        self.client.zadd(self.IDX_RUNS, {run_id: run_id})
        
        return run_id
    
    def update_training_run(self, run_id: int, status: str = None, 
                            completed_at: str = None, duration_seconds: float = None,
                            final_loss: float = None, best_metric_value: float = None,
                            artifacts: Dict = None):
        """به‌روزرسانی وضعیت آموزش"""
        key = f"{self.PREFIX_RUN}{run_id}"
        
        updates = {}
        if status:
            updates['status'] = status
        if completed_at:
            updates['completed_at'] = completed_at
        if duration_seconds is not None:
            updates['duration_seconds'] = duration_seconds
        if final_loss is not None:
            updates['final_loss'] = final_loss
        if best_metric_value is not None:
            updates['best_metric_value'] = best_metric_value
        if artifacts:
            updates['artifacts'] = self._json_encode(artifacts)
        
        if updates:
            self.client.hset(key, mapping=updates)
    
    def get_training_run(self, run_id: int) -> Optional[Dict]:
        """دریافت اطلاعات یک آموزش"""
        key = f"{self.PREFIX_RUN}{run_id}"
        data = self.client.hgetall(key)
        
        if not data:
            return None
        
        # Convert numeric fields
        data['id'] = int(data['id'])
        data['model_version_id'] = int(data['model_version_id'])
        data['dataset_id'] = int(data.get('dataset_id', 0)) or None
        data['duration_seconds'] = float(data.get('duration_seconds', 0)) or None
        data['final_loss'] = float(data.get('final_loss', 0)) or None
        data['best_metric_value'] = float(data.get('best_metric_value', 0)) or None
        data['artifacts'] = self._json_decode(data.get('artifacts', '{}'))
        
        # Handle empty string for completed_at
        if not data.get('completed_at'):
            data['completed_at'] = None
        
        return data
    
    # Hyperparameters
    def log_hyperparameters(self, training_run_id: int, params: Dict[str, Any]):
        """ثبت هایپرپارامترها"""
        key = f"{self.PREFIX_HYPERPARAM}{training_run_id}"
        # Store all params as hash fields
        self.client.hset(key, mapping={k: str(v) for k, v in params.items()})
    
    def get_hyperparameters(self, training_run_id: int) -> Dict[str, str]:
        """دریافت هایپرپارامترها"""
        key = f"{self.PREFIX_HYPERPARAM}{training_run_id}"
        return self.client.hgetall(key)
    
    # Metrics
    def log_metric(self, training_run_id: int, metric_name: str, metric_value: float,
                   epoch: int = None, step: int = None, split: str = "train"):
        """ثبت متریک"""
        # Store metric in sorted set (score = epoch or step for ordering)
        # Key format: aidb:metric:{run_id}:{metric_name}:{split}
        key = f"{self.PREFIX_METRIC}{training_run_id}:{metric_name}:{split}"
        
        score = epoch if epoch is not None else (step if step is not None else 0)
        
        # Value is JSON with all details
        metric_data = {
            'epoch': epoch,
            'step': step,
            'metric_name': metric_name,
            'metric_value': metric_value,
            'split': split,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Store in sorted set (score for ordering, JSON as member)
        self.client.zadd(key, {self._json_encode(metric_data): score})
    
    def get_metrics(self, training_run_id: int, metric_name: str = None) -> List[Dict]:
        """دریافت متریک‌ها"""
        metrics = []
        
        if metric_name:
            # Get specific metric for all splits
            for split in ['train', 'val', 'test']:
                key = f"{self.PREFIX_METRIC}{training_run_id}:{metric_name}:{split}"
                # Get all members (ordered by score)
                entries = self.client.zrange(key, 0, -1)
                for entry in entries:
                    metrics.append(self._json_decode(entry))
        else:
            # Get all metrics (scan all keys for this run)
            pattern = f"{self.PREFIX_METRIC}{training_run_id}:*"
            for key in self.client.scan_iter(match=pattern):
                entries = self.client.zrange(key, 0, -1)
                for entry in entries:
                    metrics.append(self._json_decode(entry))
        
        return metrics
    
    # Experiments
    def create_experiment(self, name: str, description: str = "", 
                          hypothesis: str = "", status: str = "active") -> int:
        """ایجاد آزمایش جدید"""
        exp_id = self._get_next_id(self.COUNTER_EXPERIMENT)
        now = datetime.utcnow().isoformat()
        
        exp_data = {
            'id': exp_id,
            'name': name,
            'description': description,
            'hypothesis': hypothesis,
            'status': status,
            'created_at': now,
            'completed_at': '',
            'results': '{}'
        }
        
        # Store as hash
        key = f"{self.PREFIX_EXPERIMENT}{exp_id}"
        self.client.hset(key, mapping=exp_data)
        
        # Add to index
        self.client.zadd(self.IDX_EXPERIMENTS, {exp_id: exp_id})
        
        return exp_id
    
    def add_experiment_run(self, experiment_id: int, training_run_id: int,
                           variant_name: str = "", notes: str = ""):
        """اضافه کردن run به آزمایش"""
        # Store experiment run as hash
        key = f"{self.PREFIX_EXPRUN}{experiment_id}:{training_run_id}"
        data = {
            'experiment_id': experiment_id,
            'training_run_id': training_run_id,
            'variant_name': variant_name,
            'notes': notes
        }
        self.client.hset(key, mapping=data)
        
        # Add to experiment's run list (set)
        runs_key = f"{self.PREFIX_EXPRUN}{experiment_id}:runs"
        self.client.sadd(runs_key, training_run_id)
    
    def get_experiment_runs(self, experiment_id: int) -> List[Dict]:
        """دریافت runهای یک آزمایش"""
        # Get all run IDs for this experiment
        runs_key = f"{self.PREFIX_EXPRUN}{experiment_id}:runs"
        run_ids = self.client.smembers(runs_key)
        
        results = []
        for run_id in run_ids:
            # Get experiment run details
            key = f"{self.PREFIX_EXPRUN}{experiment_id}:{run_id}"
            exp_run = self.client.hgetall(key)
            
            # Get training run details
            training_run = self.get_training_run(int(run_id))
            
            if exp_run and training_run:
                results.append({
                    'training_run_id': int(run_id),
                    'variant_name': exp_run.get('variant_name', ''),
                    'status': training_run.get('status', ''),
                    'final_loss': training_run.get('final_loss'),
                    'best_metric_value': training_run.get('best_metric_value')
                })
        
        return results
    
    # Predictions
    def log_prediction(self, model_version_id: int, input_data: Any, output_data: Any,
                       confidence: float = None, inference_time_ms: float = None,
                       metadata: Dict = None):
        """ثبت پیش‌بینی"""
        pred_id = self._get_next_id(self.COUNTER_PREDICTION)
        now = datetime.utcnow().isoformat()
        
        pred_data = {
            'id': pred_id,
            'model_version_id': model_version_id,
            'input': self._json_encode(input_data if isinstance(input_data, dict) else {"data": input_data}),
            'output': self._json_encode(output_data if isinstance(output_data, dict) else {"data": output_data}),
            'confidence': confidence or 0,
            'inference_time_ms': inference_time_ms or 0,
            'timestamp': now,
            'metadata': self._json_encode(metadata or {})
        }
        
        # Store as hash
        key = f"{self.PREFIX_PREDICTION}{pred_id}"
        self.client.hset(key, mapping=pred_data)
        
        # Add to version's prediction index (sorted by timestamp for recent queries)
        idx_key = f"{self.IDX_PREDICTIONS}{model_version_id}"
        # Use negative timestamp as score for reverse chronological order
        import time
        score = -time.time()
        self.client.zadd(idx_key, {pred_id: score})
    
    def get_predictions(self, model_version_id: int, limit: int = 100) -> List[Dict]:
        """دریافت پیش‌بینی‌های یک مدل"""
        idx_key = f"{self.IDX_PREDICTIONS}{model_version_id}"
        # Get most recent predictions (lowest scores = most recent)
        pred_ids = self.client.zrange(idx_key, 0, limit - 1)
        
        predictions = []
        for pred_id in pred_ids:
            key = f"{self.PREFIX_PREDICTION}{pred_id}"
            data = self.client.hgetall(key)
            if data:
                predictions.append({
                    'input': self._json_decode(data.get('input', '{}')),
                    'output': self._json_decode(data.get('output', '{}')),
                    'confidence': float(data.get('confidence', 0)) or None,
                    'inference_time_ms': float(data.get('inference_time_ms', 0)) or None,
                    'timestamp': data.get('timestamp', '')
                })
        
        return predictions
    
    # Statistics
    def get_statistics(self) -> Dict[str, Any]:
        """آمار کلی پایگاه داده"""
        stats = {
            'models': self.client.zcard(self.IDX_MODELS),
            'model_versions': int(self.client.get(self.COUNTER_VERSION) or 0),
            'datasets': self.client.zcard(self.IDX_DATASETS),
            'training_runs': self.client.zcard(self.IDX_RUNS),
            'completed_runs': 0,  # Count runs with status=completed
            'experiments': self.client.zcard(self.IDX_EXPERIMENTS),
            'predictions': int(self.client.get(self.COUNTER_PREDICTION) or 0)
        }
        
        # Count completed runs
        run_ids = self.client.zrange(self.IDX_RUNS, 0, -1)
        for run_id in run_ids:
            run = self.get_training_run(int(run_id))
            if run and run.get('status') == 'completed':
                stats['completed_runs'] += 1
        
        return stats
    
    def close(self):
        """بستن اتصال Redis"""
        self.client.close()
