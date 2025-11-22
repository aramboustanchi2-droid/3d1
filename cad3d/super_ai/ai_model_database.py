"""
AI Model Database - SQLite Storage for AI/ML Model Tracking
پایگاه داده مدیریت مدل‌های هوش مصنوعی

Features:
- Model registry (architecture, framework, versions)
- Training run tracking (hyperparameters, metrics, artifacts)
- Dataset management (splits, preprocessing info)
- Experiment tracking (A/B tests, comparisons)
- Prediction logging (inference history, monitoring)
- Performance metrics (accuracy, loss, custom metrics)
- Model versioning (checkpoints, rollback)
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import threading

DB_FILE = Path("ai_models.db")
_lock = threading.Lock()

SCHEMA = {
    "models": """
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            architecture TEXT,
            framework TEXT,
            task_type TEXT,
            input_shape TEXT,
            output_shape TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """,
    "model_versions": """
        CREATE TABLE IF NOT EXISTS model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER NOT NULL,
            version TEXT NOT NULL,
            checkpoint_path TEXT,
            config JSON,
            status TEXT,
            created_at TEXT,
            FOREIGN KEY(model_id) REFERENCES models(id),
            UNIQUE(model_id, version)
        )
    """,
    "datasets": """
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            source_path TEXT,
            format TEXT,
            size_bytes INTEGER,
            num_samples INTEGER,
            split_info JSON,
            preprocessing JSON,
            created_at TEXT
        )
    """,
    "training_runs": """
        CREATE TABLE IF NOT EXISTS training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version_id INTEGER NOT NULL,
            dataset_id INTEGER,
            run_name TEXT,
            status TEXT,
            started_at TEXT,
            completed_at TEXT,
            duration_seconds REAL,
            final_loss REAL,
            best_metric_value REAL,
            artifacts JSON,
            FOREIGN KEY(model_version_id) REFERENCES model_versions(id),
            FOREIGN KEY(dataset_id) REFERENCES datasets(id)
        )
    """,
    "hyperparameters": """
        CREATE TABLE IF NOT EXISTS hyperparameters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            training_run_id INTEGER NOT NULL,
            param_name TEXT NOT NULL,
            param_value TEXT NOT NULL,
            FOREIGN KEY(training_run_id) REFERENCES training_runs(id)
        )
    """,
    "metrics": """
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            training_run_id INTEGER NOT NULL,
            epoch INTEGER,
            step INTEGER,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            split TEXT,
            timestamp TEXT,
            FOREIGN KEY(training_run_id) REFERENCES training_runs(id)
        )
    """,
    "experiments": """
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            hypothesis TEXT,
            status TEXT,
            created_at TEXT,
            completed_at TEXT,
            results JSON
        )
    """,
    "experiment_runs": """
        CREATE TABLE IF NOT EXISTS experiment_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            training_run_id INTEGER NOT NULL,
            variant_name TEXT,
            notes TEXT,
            FOREIGN KEY(experiment_id) REFERENCES experiments(id),
            FOREIGN KEY(training_run_id) REFERENCES training_runs(id)
        )
    """,
    "predictions": """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version_id INTEGER NOT NULL,
            input_data JSON,
            output_data JSON,
            confidence REAL,
            inference_time_ms REAL,
            timestamp TEXT,
            metadata JSON,
            FOREIGN KEY(model_version_id) REFERENCES model_versions(id)
        )
    """
}

class AIModelDatabase:
    """مدیریت پایگاه داده مدل‌های هوش مصنوعی"""
    
    def __init__(self, db_path: Path = DB_FILE):
        self.db_path = db_path
        self._init_db()
    
    def _connect(self):
        return sqlite3.connect(self.db_path)
    
    def _init_db(self):
        """ایجاد جداول پایگاه داده"""
        with _lock:
            conn = self._connect()
            cur = conn.cursor()
            for name, ddl in SCHEMA.items():
                cur.execute(ddl)
            conn.commit()
            conn.close()
    
    # Models
    def create_model(self, name: str, description: str = "", architecture: str = "", 
                     framework: str = "", task_type: str = "", 
                     input_shape: str = "", output_shape: str = "") -> int:
        """ثبت مدل جدید"""
        with _lock:
            conn = self._connect()
            cur = conn.cursor()
            now = datetime.utcnow().isoformat()
            cur.execute("""
                INSERT INTO models(name, description, architecture, framework, task_type, 
                                   input_shape, output_shape, created_at, updated_at)
                VALUES(?,?,?,?,?,?,?,?,?)
            """, (name, description, architecture, framework, task_type, 
                  input_shape, output_shape, now, now))
            model_id = cur.lastrowid
            conn.commit()
            conn.close()
            return model_id
    
    def get_model(self, model_id: int) -> Optional[Dict]:
        """دریافت اطلاعات مدل"""
        with _lock:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("SELECT * FROM models WHERE id=?", (model_id,))
            row = cur.fetchone()
            conn.close()
            if row:
                return {
                    'id': row[0], 'name': row[1], 'description': row[2],
                    'architecture': row[3], 'framework': row[4], 'task_type': row[5],
                    'input_shape': row[6], 'output_shape': row[7],
                    'created_at': row[8], 'updated_at': row[9]
                }
            return None
    
    def list_models(self) -> List[Dict]:
        """لیست همه مدل‌ها"""
        with _lock:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("SELECT id, name, architecture, framework, task_type, created_at FROM models")
            rows = cur.fetchall()
            conn.close()
            return [
                {
                    'id': r[0], 'name': r[1], 'architecture': r[2],
                    'framework': r[3], 'task_type': r[4], 'created_at': r[5]
                } for r in rows
            ]
    
    # Model Versions
    def create_model_version(self, model_id: int, version: str, 
                             checkpoint_path: str = "", config: Dict = None,
                             status: str = "active") -> int:
        """ایجاد نسخه جدید مدل"""
        with _lock:
            conn = self._connect()
            cur = conn.cursor()
            now = datetime.utcnow().isoformat()
            config_json = json.dumps(config or {})
            cur.execute("""
                INSERT INTO model_versions(model_id, version, checkpoint_path, config, status, created_at)
                VALUES(?,?,?,?,?,?)
            """, (model_id, version, checkpoint_path, config_json, status, now))
            version_id = cur.lastrowid
            conn.commit()
            conn.close()
            return version_id
    
    def get_model_versions(self, model_id: int) -> List[Dict]:
        """دریافت نسخه‌های یک مدل"""
        with _lock:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("SELECT id, version, checkpoint_path, status, created_at FROM model_versions WHERE model_id=? ORDER BY id DESC", (model_id,))
            rows = cur.fetchall()
            conn.close()
            return [
                {
                    'id': r[0], 'version': r[1], 'checkpoint_path': r[2],
                    'status': r[3], 'created_at': r[4]
                } for r in rows
            ]
    
    # Datasets
    def create_dataset(self, name: str, description: str = "", source_path: str = "",
                       format: str = "", size_bytes: int = 0, num_samples: int = 0,
                       split_info: Dict = None, preprocessing: Dict = None) -> int:
        """ثبت دیتاست"""
        with _lock:
            conn = self._connect()
            cur = conn.cursor()
            now = datetime.utcnow().isoformat()
            split_json = json.dumps(split_info or {})
            prep_json = json.dumps(preprocessing or {})
            cur.execute("""
                INSERT INTO datasets(name, description, source_path, format, size_bytes, 
                                     num_samples, split_info, preprocessing, created_at)
                VALUES(?,?,?,?,?,?,?,?,?)
            """, (name, description, source_path, format, size_bytes, num_samples,
                  split_json, prep_json, now))
            dataset_id = cur.lastrowid
            conn.commit()
            conn.close()
            return dataset_id
    
    def get_dataset(self, dataset_id: int) -> Optional[Dict]:
        """دریافت اطلاعات دیتاست"""
        with _lock:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,))
            row = cur.fetchone()
            conn.close()
            if row:
                return {
                    'id': row[0], 'name': row[1], 'description': row[2],
                    'source_path': row[3], 'format': row[4], 'size_bytes': row[5],
                    'num_samples': row[6], 'split_info': json.loads(row[7]),
                    'preprocessing': json.loads(row[8]), 'created_at': row[9]
                }
            return None
    
    # Training Runs
    def create_training_run(self, model_version_id: int, dataset_id: int = None,
                            run_name: str = "", status: str = "started") -> int:
        """شروع آموزش جدید"""
        with _lock:
            conn = self._connect()
            cur = conn.cursor()
            now = datetime.utcnow().isoformat()
            cur.execute("""
                INSERT INTO training_runs(model_version_id, dataset_id, run_name, status, started_at)
                VALUES(?,?,?,?,?)
            """, (model_version_id, dataset_id, run_name, status, now))
            run_id = cur.lastrowid
            conn.commit()
            conn.close()
            return run_id
    
    def update_training_run(self, run_id: int, status: str = None, 
                            completed_at: str = None, duration_seconds: float = None,
                            final_loss: float = None, best_metric_value: float = None,
                            artifacts: Dict = None):
        """به‌روزرسانی وضعیت آموزش"""
        with _lock:
            conn = self._connect()
            cur = conn.cursor()
            updates = []
            params = []
            if status:
                updates.append("status=?")
                params.append(status)
            if completed_at:
                updates.append("completed_at=?")
                params.append(completed_at)
            if duration_seconds is not None:
                updates.append("duration_seconds=?")
                params.append(duration_seconds)
            if final_loss is not None:
                updates.append("final_loss=?")
                params.append(final_loss)
            if best_metric_value is not None:
                updates.append("best_metric_value=?")
                params.append(best_metric_value)
            if artifacts:
                updates.append("artifacts=?")
                params.append(json.dumps(artifacts))
            
            if updates:
                params.append(run_id)
                sql = f"UPDATE training_runs SET {', '.join(updates)} WHERE id=?"
                cur.execute(sql, tuple(params))
                conn.commit()
            conn.close()
    
    def get_training_run(self, run_id: int) -> Optional[Dict]:
        """دریافت اطلاعات یک آموزش"""
        with _lock:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("SELECT * FROM training_runs WHERE id=?", (run_id,))
            row = cur.fetchone()
            conn.close()
            if row:
                return {
                    'id': row[0], 'model_version_id': row[1], 'dataset_id': row[2],
                    'run_name': row[3], 'status': row[4], 'started_at': row[5],
                    'completed_at': row[6], 'duration_seconds': row[7],
                    'final_loss': row[8], 'best_metric_value': row[9],
                    'artifacts': json.loads(row[10]) if row[10] else {}
                }
            return None
    
    # Hyperparameters
    def log_hyperparameters(self, training_run_id: int, params: Dict[str, Any]):
        """ثبت هایپرپارامترها"""
        with _lock:
            conn = self._connect()
            cur = conn.cursor()
            for name, value in params.items():
                cur.execute("""
                    INSERT INTO hyperparameters(training_run_id, param_name, param_value)
                    VALUES(?,?,?)
                """, (training_run_id, name, str(value)))
            conn.commit()
            conn.close()
    
    def get_hyperparameters(self, training_run_id: int) -> Dict[str, str]:
        """دریافت هایپرپارامترها"""
        with _lock:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("SELECT param_name, param_value FROM hyperparameters WHERE training_run_id=?", (training_run_id,))
            rows = cur.fetchall()
            conn.close()
            return {r[0]: r[1] for r in rows}
    
    # Metrics
    def log_metric(self, training_run_id: int, metric_name: str, metric_value: float,
                   epoch: int = None, step: int = None, split: str = "train"):
        """ثبت متریک"""
        with _lock:
            conn = self._connect()
            cur = conn.cursor()
            now = datetime.utcnow().isoformat()
            cur.execute("""
                INSERT INTO metrics(training_run_id, epoch, step, metric_name, metric_value, split, timestamp)
                VALUES(?,?,?,?,?,?,?)
            """, (training_run_id, epoch, step, metric_name, metric_value, split, now))
            conn.commit()
            conn.close()
    
    def get_metrics(self, training_run_id: int, metric_name: str = None) -> List[Dict]:
        """دریافت متریک‌ها"""
        with _lock:
            conn = self._connect()
            cur = conn.cursor()
            if metric_name:
                cur.execute("""
                    SELECT epoch, step, metric_name, metric_value, split, timestamp 
                    FROM metrics WHERE training_run_id=? AND metric_name=? ORDER BY id
                """, (training_run_id, metric_name))
            else:
                cur.execute("""
                    SELECT epoch, step, metric_name, metric_value, split, timestamp 
                    FROM metrics WHERE training_run_id=? ORDER BY id
                """, (training_run_id,))
            rows = cur.fetchall()
            conn.close()
            return [
                {
                    'epoch': r[0], 'step': r[1], 'metric_name': r[2],
                    'metric_value': r[3], 'split': r[4], 'timestamp': r[5]
                } for r in rows
            ]
    
    # Experiments
    def create_experiment(self, name: str, description: str = "", 
                          hypothesis: str = "", status: str = "active") -> int:
        """ایجاد آزمایش جدید"""
        with _lock:
            conn = self._connect()
            cur = conn.cursor()
            now = datetime.utcnow().isoformat()
            cur.execute("""
                INSERT INTO experiments(name, description, hypothesis, status, created_at)
                VALUES(?,?,?,?,?)
            """, (name, description, hypothesis, status, now))
            exp_id = cur.lastrowid
            conn.commit()
            conn.close()
            return exp_id
    
    def add_experiment_run(self, experiment_id: int, training_run_id: int,
                           variant_name: str = "", notes: str = ""):
        """اضافه کردن run به آزمایش"""
        with _lock:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO experiment_runs(experiment_id, training_run_id, variant_name, notes)
                VALUES(?,?,?,?)
            """, (experiment_id, training_run_id, variant_name, notes))
            conn.commit()
            conn.close()
    
    def get_experiment_runs(self, experiment_id: int) -> List[Dict]:
        """دریافت runهای یک آزمایش"""
        with _lock:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("""
                SELECT er.training_run_id, er.variant_name, tr.status, tr.final_loss, tr.best_metric_value
                FROM experiment_runs er
                JOIN training_runs tr ON er.training_run_id = tr.id
                WHERE er.experiment_id=?
            """, (experiment_id,))
            rows = cur.fetchall()
            conn.close()
            return [
                {
                    'training_run_id': r[0], 'variant_name': r[1],
                    'status': r[2], 'final_loss': r[3], 'best_metric_value': r[4]
                } for r in rows
            ]
    
    # Predictions
    def log_prediction(self, model_version_id: int, input_data: Any, output_data: Any,
                       confidence: float = None, inference_time_ms: float = None,
                       metadata: Dict = None):
        """ثبت پیش‌بینی"""
        with _lock:
            conn = self._connect()
            cur = conn.cursor()
            now = datetime.utcnow().isoformat()
            input_json = json.dumps(input_data)
            output_json = json.dumps(output_data)
            meta_json = json.dumps(metadata or {})
            cur.execute("""
                INSERT INTO predictions(model_version_id, input_data, output_data, 
                                        confidence, inference_time_ms, timestamp, metadata)
                VALUES(?,?,?,?,?,?,?)
            """, (model_version_id, input_json, output_json, confidence, 
                  inference_time_ms, now, meta_json))
            conn.commit()
            conn.close()
    
    def get_predictions(self, model_version_id: int, limit: int = 100) -> List[Dict]:
        """دریافت پیش‌بینی‌های یک مدل"""
        with _lock:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("""
                SELECT input_data, output_data, confidence, inference_time_ms, timestamp
                FROM predictions WHERE model_version_id=? ORDER BY id DESC LIMIT ?
            """, (model_version_id, limit))
            rows = cur.fetchall()
            conn.close()
            return [
                {
                    'input': json.loads(r[0]), 'output': json.loads(r[1]),
                    'confidence': r[2], 'inference_time_ms': r[3], 'timestamp': r[4]
                } for r in rows
            ]
    
    # Statistics
    def get_statistics(self) -> Dict[str, Any]:
        """آمار کلی پایگاه داده"""
        with _lock:
            conn = self._connect()
            cur = conn.cursor()
            
            cur.execute("SELECT COUNT(*) FROM models")
            models_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM model_versions")
            versions_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM datasets")
            datasets_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM training_runs")
            runs_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM training_runs WHERE status='completed'")
            completed_runs = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM experiments")
            experiments_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM predictions")
            predictions_count = cur.fetchone()[0]
            
            conn.close()
            
            return {
                'models': models_count,
                'model_versions': versions_count,
                'datasets': datasets_count,
                'training_runs': runs_count,
                'completed_runs': completed_runs,
                'experiments': experiments_count,
                'predictions': predictions_count
            }

# Singleton instance
ai_db = AIModelDatabase()
