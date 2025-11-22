"""
AI Model Database - MongoDB Backend (پایگاه داده مدل MongoDB)
Sixth backend complementing: SQLite, PostgreSQL/MySQL, Redis, ChromaDB, FAISS.

Features:
 - Unified API (same method names/semantics)
 - Collections: models, model_versions, datasets, training_runs, hyperparams, metrics, predictions, experiments, experiment_runs
 - Atomic integer IDs via counters collection (find_one_and_update with $inc)
 - Indexes for frequent queries
 - Optional in-memory mock via mongomock for local development (use_mock=True)

Installation:
    pip install -r requirements-mongo-database.txt

Connection:
    # Direct string
    mongodb://user:pass@localhost:27017/aidb

    # Factory usage
    create_ai_database(backend='mongodb', host='localhost', port=27017, database='aidb')

Note: MongoDB stores its data outside the project folder (managed by the server). Code, config and requirements reside inside the project as requested.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    import pymongo
except ImportError:  # Allow reading file without dependency installed
    pymongo = None  # type: ignore


class AIModelDatabaseMongo:
    """MongoDB backend for AI model database."""

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 27017,
        database: str = 'aidb',
        username: Optional[str] = None,
        password: Optional[str] = None,
        connection_string: Optional[str] = None,
        use_mock: bool = False,
        create_indexes: bool = True
    ):
        if use_mock:
            # In-memory mock (file-system if mongomock implements) for demos without real server
            try:
                import mongomock  # type: ignore
                self._client = mongomock.MongoClient()
            except ImportError:
                raise ImportError("mongomock not installed. Run: pip install mongomock")
        else:
            if pymongo is None:
                raise ImportError("pymongo not installed. Run: pip install pymongo")
            if connection_string:
                self._client = pymongo.MongoClient(connection_string)
            else:
                self._client = pymongo.MongoClient(host=host, port=port, username=username, password=password)

        self._db = self._client[database]

        # Collections
        self.col_models = self._db['models']
        self.col_versions = self._db['model_versions']
        self.col_datasets = self._db['datasets']
        self.col_runs = self._db['training_runs']
        self.col_hparams = self._db['hyperparams']
        self.col_metrics = self._db['metrics']
        self.col_predictions = self._db['predictions']
        self.col_experiments = self._db['experiments']
        self.col_experiment_runs = self._db['experiment_runs']
        self.col_counters = self._db['counters']  # { _id: 'models', seq: int }

        if create_indexes:
            self._ensure_indexes()

    # ------------------------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------------------------
    def _ensure_indexes(self):
        self.col_models.create_index('id', unique=True)
        self.col_models.create_index([('created_at', 1)])
        self.col_versions.create_index('id', unique=True)
        self.col_versions.create_index('model_id')
        self.col_datasets.create_index('id', unique=True)
        self.col_runs.create_index('id', unique=True)
        self.col_runs.create_index('model_version_id')
        self.col_metrics.create_index([('training_run_id', 1), ('metric_name', 1), ('epoch', 1)])
        self.col_predictions.create_index('model_version_id')
        self.col_predictions.create_index('timestamp')
        self.col_experiments.create_index('id', unique=True)
        self.col_experiment_runs.create_index('experiment_id')

    def _next_id(self, counter_name: str) -> int:
        doc = self.col_counters.find_one_and_update(
            {'_id': counter_name},
            {'$inc': {'seq': 1}},
            upsert=True,
            return_document=True
        )
        return int(doc['seq'])

    def _utcnow(self) -> str:
        return datetime.utcnow().isoformat()

    # ------------------------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------------------------
    def create_model(
        self,
        name: str,
        description: str = '',
        architecture: str = '',
        framework: str = '',
        task_type: str = '',
        input_shape: str = '',
        output_shape: str = ''
    ) -> int:
        model_id = self._next_id('models')
        now = self._utcnow()
        doc = {
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
        self.col_models.insert_one(doc)
        return model_id

    def get_model(self, model_id: int) -> Optional[Dict[str, Any]]:
        return self.col_models.find_one({'id': model_id}, {'_id': 0})

    def list_models(self) -> List[Dict[str, Any]]:
        return list(self.col_models.find({}, {'_id': 0}).sort('id', 1))

    # ------------------------------------------------------------------------------------
    # Model Versions
    # ------------------------------------------------------------------------------------
    def create_model_version(
        self,
        model_id: int,
        version: str,
        checkpoint_path: str = '',
        config: Dict[str, Any] = None,
        status: str = 'active'
    ) -> int:
        version_id = self._next_id('model_versions')
        doc = {
            'id': version_id,
            'model_id': model_id,
            'version': version,
            'checkpoint_path': checkpoint_path,
            'config': config or {},
            'status': status,
            'created_at': self._utcnow()
        }
        self.col_versions.insert_one(doc)
        return version_id

    def get_model_versions(self, model_id: int) -> List[Dict[str, Any]]:
        return list(self.col_versions.find({'model_id': model_id}, {'_id': 0}).sort('id', -1))

    # ------------------------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------------------------
    def create_dataset(
        self,
        name: str,
        description: str = '',
        source_path: str = '',
        format: str = '',
        size_bytes: int = 0,
        num_samples: int = 0,
        split_info: Dict[str, Any] = None,
        preprocessing: Dict[str, Any] = None
    ) -> int:
        dataset_id = self._next_id('datasets')
        doc = {
            'id': dataset_id,
            'name': name,
            'description': description,
            'source_path': source_path,
            'format': format,
            'size_bytes': size_bytes,
            'num_samples': num_samples,
            'split_info': split_info or {},
            'preprocessing': preprocessing or {},
            'created_at': self._utcnow()
        }
        self.col_datasets.insert_one(doc)
        return dataset_id

    def get_dataset(self, dataset_id: int) -> Optional[Dict[str, Any]]:
        return self.col_datasets.find_one({'id': dataset_id}, {'_id': 0})

    # ------------------------------------------------------------------------------------
    # Training Runs
    # ------------------------------------------------------------------------------------
    def create_training_run(
        self,
        model_version_id: int,
        dataset_id: Optional[int] = None,
        run_name: str = '',
        status: str = 'started'
    ) -> int:
        run_id = self._next_id('training_runs')
        doc = {
            'id': run_id,
            'model_version_id': model_version_id,
            'dataset_id': dataset_id,
            'run_name': run_name,
            'status': status,
            'started_at': self._utcnow(),
            'completed_at': None,
            'duration_seconds': None,
            'final_loss': None,
            'best_metric_value': None,
            'artifacts': {}
        }
        self.col_runs.insert_one(doc)
        return run_id

    def update_training_run(
        self,
        run_id: int,
        status: Optional[str] = None,
        completed_at: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        final_loss: Optional[float] = None,
        best_metric_value: Optional[float] = None,
        artifacts: Optional[Dict[str, Any]] = None
    ) -> None:
        update = {}
        if status is not None:
            update['status'] = status
        if completed_at is not None:
            update['completed_at'] = completed_at
        if duration_seconds is not None:
            update['duration_seconds'] = duration_seconds
        if final_loss is not None:
            update['final_loss'] = final_loss
        if best_metric_value is not None:
            update['best_metric_value'] = best_metric_value
        if artifacts is not None:
            update['artifacts'] = artifacts
        if update:
            self.col_runs.update_one({'id': run_id}, {'$set': update})

    def get_training_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        return self.col_runs.find_one({'id': run_id}, {'_id': 0})

    # ------------------------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------------------------
    def log_hyperparameters(self, training_run_id: int, params: Dict[str, Any]) -> None:
        self.col_hparams.update_one(
            {'training_run_id': training_run_id},
            {'$set': {'params': {k: str(v) for k, v in params.items()}}},
            upsert=True
        )

    def get_hyperparameters(self, training_run_id: int) -> Dict[str, str]:
        doc = self.col_hparams.find_one({'training_run_id': training_run_id}, {'_id': 0})
        return doc.get('params', {}) if doc else {}

    # ------------------------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------------------------
    def log_metric(
        self,
        training_run_id: int,
        metric_name: str,
        metric_value: float,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        split: str = 'train'
    ) -> None:
        doc = {
            'training_run_id': training_run_id,
            'metric_name': metric_name,
            'metric_value': metric_value,
            'epoch': epoch,
            'step': step,
            'split': split,
            'timestamp': self._utcnow()
        }
        self.col_metrics.insert_one(doc)

    def get_metrics(self, training_run_id: int, metric_name: Optional[str] = None) -> List[Dict[str, Any]]:
        query = {'training_run_id': training_run_id}
        if metric_name:
            query['metric_name'] = metric_name
        cursor = self.col_metrics.find(query, {'_id': 0}).sort('timestamp', 1)
        return list(cursor)

    # ------------------------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------------------------
    def create_experiment(
        self,
        name: str,
        description: str = '',
        hypothesis: str = '',
        status: str = 'active'
    ) -> int:
        exp_id = self._next_id('experiments')
        doc = {
            'id': exp_id,
            'name': name,
            'description': description,
            'hypothesis': hypothesis,
            'status': status,
            'created_at': self._utcnow(),
            'completed_at': None,
            'results': {}
        }
        self.col_experiments.insert_one(doc)
        return exp_id

    def add_experiment_run(
        self,
        experiment_id: int,
        training_run_id: int,
        variant_name: str = '',
        notes: str = ''
    ) -> None:
        doc = {
            'experiment_id': experiment_id,
            'training_run_id': training_run_id,
            'variant_name': variant_name,
            'notes': notes
        }
        self.col_experiment_runs.insert_one(doc)

    def get_experiment_runs(self, experiment_id: int) -> List[Dict[str, Any]]:
        runs = list(self.col_experiment_runs.find({'experiment_id': experiment_id}, {'_id': 0}))
        results: List[Dict[str, Any]] = []
        for r in runs:
            run_doc = self.get_training_run(r['training_run_id'])
            if run_doc:
                results.append({
                    'training_run_id': r['training_run_id'],
                    'variant_name': r.get('variant_name', ''),
                    'status': run_doc.get('status'),
                    'final_loss': run_doc.get('final_loss'),
                    'best_metric_value': run_doc.get('best_metric_value')
                })
        return results

    # ------------------------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------------------------
    def log_prediction(
        self,
        model_version_id: int,
        input_data: Any,
        output_data: Any,
        confidence: Optional[float] = None,
        inference_time_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        pred_id = self._next_id('predictions')
        doc = {
            'id': pred_id,
            'model_version_id': model_version_id,
            'input': input_data if isinstance(input_data, dict) else {'data': input_data},
            'output': output_data if isinstance(output_data, dict) else {'data': output_data},
            'confidence': confidence,
            'inference_time_ms': inference_time_ms,
            'metadata': metadata or {},
            'timestamp': self._utcnow()
        }
        self.col_predictions.insert_one(doc)

    def get_predictions(self, model_version_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        cursor = self.col_predictions.find({'model_version_id': model_version_id}, {'_id': 0}).sort('timestamp', -1).limit(limit)
        return list(cursor)

    # ------------------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------------------
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'models': self.col_models.count_documents({}),
            'model_versions': self.col_versions.count_documents({}),
            'datasets': self.col_datasets.count_documents({}),
            'training_runs': self.col_runs.count_documents({}),
            'completed_runs': self.col_runs.count_documents({'status': 'completed'}),
            'experiments': self.col_experiments.count_documents({}),
            'predictions': self.col_predictions.count_documents({})
        }

    # ------------------------------------------------------------------------------------
    # Health & Utility
    # ------------------------------------------------------------------------------------
    def ping(self) -> bool:
        try:
            # For mongomock or real client
            self._client.admin.command('ping')
            return True
        except Exception:
            return False

    def drop_database(self) -> None:
        name = self._db.name
        self._client.drop_database(name)

__all__ = ["AIModelDatabaseMongo"]
