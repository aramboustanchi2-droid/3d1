"""
AI Model Database - PostgreSQL/MySQL Adapter
پایگاه داده مدل‌های هوش مصنوعی با پشتیبانی PostgreSQL/MySQL

Features:
- PostgreSQL و MySQL در کنار SQLite
- Connection pooling برای بهبود عملکرد
- JSON fields برای متادیتا
- Full-text search
- Better concurrency & transactions
- همان API با SQLite

Installation:
    pip install psycopg2-binary pymysql sqlalchemy
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text, 
    DateTime, ForeignKey, JSON, UniqueConstraint, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.pool import QueuePool

Base = declarative_base()

# Models
class Model(Base):
    __tablename__ = 'models'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(Text)
    architecture = Column(String(255))
    framework = Column(String(100))
    task_type = Column(String(100))
    input_shape = Column(String(255))
    output_shape = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    versions = relationship("ModelVersion", back_populates="model", cascade="all, delete-orphan")

class ModelVersion(Base):
    __tablename__ = 'model_versions'
    
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('models.id', ondelete='CASCADE'), nullable=False)
    version = Column(String(100), nullable=False)
    checkpoint_path = Column(Text)
    config = Column(JSON)
    status = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    model = relationship("Model", back_populates="versions")
    training_runs = relationship("TrainingRun", back_populates="model_version", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="model_version", cascade="all, delete-orphan")
    
    __table_args__ = (
        UniqueConstraint('model_id', 'version', name='uq_model_version'),
        Index('idx_model_version_status', 'model_id', 'status'),
    )

class Dataset(Base):
    __tablename__ = 'datasets'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(Text)
    source_path = Column(Text)
    format = Column(String(50))
    size_bytes = Column(Integer)
    num_samples = Column(Integer)
    split_info = Column(JSON)
    preprocessing = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    training_runs = relationship("TrainingRun", back_populates="dataset")

class TrainingRun(Base):
    __tablename__ = 'training_runs'
    
    id = Column(Integer, primary_key=True)
    model_version_id = Column(Integer, ForeignKey('model_versions.id', ondelete='CASCADE'), nullable=False)
    dataset_id = Column(Integer, ForeignKey('datasets.id', ondelete='SET NULL'))
    run_name = Column(String(255))
    status = Column(String(50), index=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    duration_seconds = Column(Float)
    final_loss = Column(Float)
    best_metric_value = Column(Float)
    artifacts = Column(JSON)
    
    model_version = relationship("ModelVersion", back_populates="training_runs")
    dataset = relationship("Dataset", back_populates="training_runs")
    hyperparameters = relationship("Hyperparameter", back_populates="training_run", cascade="all, delete-orphan")
    metrics = relationship("Metric", back_populates="training_run", cascade="all, delete-orphan")
    experiment_runs = relationship("ExperimentRun", back_populates="training_run")
    
    __table_args__ = (
        Index('idx_training_run_status_date', 'status', 'started_at'),
    )

class Hyperparameter(Base):
    __tablename__ = 'hyperparameters'
    
    id = Column(Integer, primary_key=True)
    training_run_id = Column(Integer, ForeignKey('training_runs.id', ondelete='CASCADE'), nullable=False)
    param_name = Column(String(255), nullable=False)
    param_value = Column(Text, nullable=False)
    
    training_run = relationship("TrainingRun", back_populates="hyperparameters")
    
    __table_args__ = (
        Index('idx_hyperparameter_run_name', 'training_run_id', 'param_name'),
    )

class Metric(Base):
    __tablename__ = 'metrics'
    
    id = Column(Integer, primary_key=True)
    training_run_id = Column(Integer, ForeignKey('training_runs.id', ondelete='CASCADE'), nullable=False)
    epoch = Column(Integer)
    step = Column(Integer)
    metric_name = Column(String(255), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    split = Column(String(50))
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    training_run = relationship("TrainingRun", back_populates="metrics")
    
    __table_args__ = (
        Index('idx_metric_run_name_epoch', 'training_run_id', 'metric_name', 'epoch'),
    )

class Experiment(Base):
    __tablename__ = 'experiments'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(Text)
    hypothesis = Column(Text)
    status = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    results = Column(JSON)
    
    experiment_runs = relationship("ExperimentRun", back_populates="experiment", cascade="all, delete-orphan")

class ExperimentRun(Base):
    __tablename__ = 'experiment_runs'
    
    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey('experiments.id', ondelete='CASCADE'), nullable=False)
    training_run_id = Column(Integer, ForeignKey('training_runs.id', ondelete='CASCADE'), nullable=False)
    variant_name = Column(String(255))
    notes = Column(Text)
    
    experiment = relationship("Experiment", back_populates="experiment_runs")
    training_run = relationship("TrainingRun", back_populates="experiment_runs")

class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    model_version_id = Column(Integer, ForeignKey('model_versions.id', ondelete='CASCADE'), nullable=False)
    input_data = Column(JSON)
    output_data = Column(JSON)
    confidence = Column(Float)
    inference_time_ms = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    metadata = Column(JSON)
    
    model_version = relationship("ModelVersion", back_populates="predictions")
    
    __table_args__ = (
        Index('idx_prediction_model_time', 'model_version_id', 'timestamp'),
    )


class AIModelDatabaseSQL:
    """مدیریت پایگاه داده SQL (PostgreSQL/MySQL) برای مدل‌های هوش مصنوعی"""
    
    def __init__(self, connection_string: str, echo: bool = False):
        """
        Args:
            connection_string: SQLAlchemy connection string
                PostgreSQL: "postgresql://user:pass@localhost/dbname"
                MySQL: "mysql+pymysql://user:pass@localhost/dbname"
            echo: Log SQL queries
        """
        self.engine = create_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=echo
        )
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def _get_session(self) -> Session:
        return self.SessionLocal()
    
    # Models
    def create_model(self, name: str, description: str = "", architecture: str = "", 
                     framework: str = "", task_type: str = "", 
                     input_shape: str = "", output_shape: str = "") -> int:
        """ثبت مدل جدید"""
        session = self._get_session()
        try:
            model = Model(
                name=name, description=description, architecture=architecture,
                framework=framework, task_type=task_type,
                input_shape=input_shape, output_shape=output_shape
            )
            session.add(model)
            session.commit()
            return model.id
        finally:
            session.close()
    
    def get_model(self, model_id: int) -> Optional[Dict]:
        """دریافت اطلاعات مدل"""
        session = self._get_session()
        try:
            model = session.query(Model).filter(Model.id == model_id).first()
            if model:
                return {
                    'id': model.id, 'name': model.name, 'description': model.description,
                    'architecture': model.architecture, 'framework': model.framework,
                    'task_type': model.task_type, 'input_shape': model.input_shape,
                    'output_shape': model.output_shape, 'created_at': model.created_at.isoformat(),
                    'updated_at': model.updated_at.isoformat()
                }
            return None
        finally:
            session.close()
    
    def list_models(self) -> List[Dict]:
        """لیست همه مدل‌ها"""
        session = self._get_session()
        try:
            models = session.query(Model).all()
            return [
                {
                    'id': m.id, 'name': m.name, 'architecture': m.architecture,
                    'framework': m.framework, 'task_type': m.task_type,
                    'created_at': m.created_at.isoformat()
                } for m in models
            ]
        finally:
            session.close()
    
    # Model Versions
    def create_model_version(self, model_id: int, version: str, 
                             checkpoint_path: str = "", config: Dict = None,
                             status: str = "active") -> int:
        """ایجاد نسخه جدید مدل"""
        session = self._get_session()
        try:
            model_version = ModelVersion(
                model_id=model_id, version=version,
                checkpoint_path=checkpoint_path, config=config or {},
                status=status
            )
            session.add(model_version)
            session.commit()
            return model_version.id
        finally:
            session.close()
    
    def get_model_versions(self, model_id: int) -> List[Dict]:
        """دریافت نسخه‌های یک مدل"""
        session = self._get_session()
        try:
            versions = session.query(ModelVersion).filter(
                ModelVersion.model_id == model_id
            ).order_by(ModelVersion.id.desc()).all()
            return [
                {
                    'id': v.id, 'version': v.version, 'checkpoint_path': v.checkpoint_path,
                    'status': v.status, 'created_at': v.created_at.isoformat()
                } for v in versions
            ]
        finally:
            session.close()
    
    # Datasets
    def create_dataset(self, name: str, description: str = "", source_path: str = "",
                       format: str = "", size_bytes: int = 0, num_samples: int = 0,
                       split_info: Dict = None, preprocessing: Dict = None) -> int:
        """ثبت دیتاست"""
        session = self._get_session()
        try:
            dataset = Dataset(
                name=name, description=description, source_path=source_path,
                format=format, size_bytes=size_bytes, num_samples=num_samples,
                split_info=split_info or {}, preprocessing=preprocessing or {}
            )
            session.add(dataset)
            session.commit()
            return dataset.id
        finally:
            session.close()
    
    def get_dataset(self, dataset_id: int) -> Optional[Dict]:
        """دریافت اطلاعات دیتاست"""
        session = self._get_session()
        try:
            dataset = session.query(Dataset).filter(Dataset.id == dataset_id).first()
            if dataset:
                return {
                    'id': dataset.id, 'name': dataset.name, 'description': dataset.description,
                    'source_path': dataset.source_path, 'format': dataset.format,
                    'size_bytes': dataset.size_bytes, 'num_samples': dataset.num_samples,
                    'split_info': dataset.split_info, 'preprocessing': dataset.preprocessing,
                    'created_at': dataset.created_at.isoformat()
                }
            return None
        finally:
            session.close()
    
    # Training Runs
    def create_training_run(self, model_version_id: int, dataset_id: int = None,
                            run_name: str = "", status: str = "started") -> int:
        """شروع آموزش جدید"""
        session = self._get_session()
        try:
            training_run = TrainingRun(
                model_version_id=model_version_id, dataset_id=dataset_id,
                run_name=run_name, status=status
            )
            session.add(training_run)
            session.commit()
            return training_run.id
        finally:
            session.close()
    
    def update_training_run(self, run_id: int, status: str = None, 
                            completed_at: str = None, duration_seconds: float = None,
                            final_loss: float = None, best_metric_value: float = None,
                            artifacts: Dict = None):
        """به‌روزرسانی وضعیت آموزش"""
        session = self._get_session()
        try:
            run = session.query(TrainingRun).filter(TrainingRun.id == run_id).first()
            if run:
                if status:
                    run.status = status
                if completed_at:
                    run.completed_at = datetime.fromisoformat(completed_at)
                if duration_seconds is not None:
                    run.duration_seconds = duration_seconds
                if final_loss is not None:
                    run.final_loss = final_loss
                if best_metric_value is not None:
                    run.best_metric_value = best_metric_value
                if artifacts:
                    run.artifacts = artifacts
                session.commit()
        finally:
            session.close()
    
    def get_training_run(self, run_id: int) -> Optional[Dict]:
        """دریافت اطلاعات یک آموزش"""
        session = self._get_session()
        try:
            run = session.query(TrainingRun).filter(TrainingRun.id == run_id).first()
            if run:
                return {
                    'id': run.id, 'model_version_id': run.model_version_id,
                    'dataset_id': run.dataset_id, 'run_name': run.run_name,
                    'status': run.status, 'started_at': run.started_at.isoformat(),
                    'completed_at': run.completed_at.isoformat() if run.completed_at else None,
                    'duration_seconds': run.duration_seconds, 'final_loss': run.final_loss,
                    'best_metric_value': run.best_metric_value, 'artifacts': run.artifacts or {}
                }
            return None
        finally:
            session.close()
    
    # Hyperparameters
    def log_hyperparameters(self, training_run_id: int, params: Dict[str, Any]):
        """ثبت هایپرپارامترها"""
        session = self._get_session()
        try:
            for name, value in params.items():
                hyperparam = Hyperparameter(
                    training_run_id=training_run_id,
                    param_name=name,
                    param_value=str(value)
                )
                session.add(hyperparam)
            session.commit()
        finally:
            session.close()
    
    def get_hyperparameters(self, training_run_id: int) -> Dict[str, str]:
        """دریافت هایپرپارامترها"""
        session = self._get_session()
        try:
            params = session.query(Hyperparameter).filter(
                Hyperparameter.training_run_id == training_run_id
            ).all()
            return {p.param_name: p.param_value for p in params}
        finally:
            session.close()
    
    # Metrics
    def log_metric(self, training_run_id: int, metric_name: str, metric_value: float,
                   epoch: int = None, step: int = None, split: str = "train"):
        """ثبت متریک"""
        session = self._get_session()
        try:
            metric = Metric(
                training_run_id=training_run_id, epoch=epoch, step=step,
                metric_name=metric_name, metric_value=metric_value, split=split
            )
            session.add(metric)
            session.commit()
        finally:
            session.close()
    
    def get_metrics(self, training_run_id: int, metric_name: str = None) -> List[Dict]:
        """دریافت متریک‌ها"""
        session = self._get_session()
        try:
            query = session.query(Metric).filter(Metric.training_run_id == training_run_id)
            if metric_name:
                query = query.filter(Metric.metric_name == metric_name)
            metrics = query.order_by(Metric.id).all()
            return [
                {
                    'epoch': m.epoch, 'step': m.step, 'metric_name': m.metric_name,
                    'metric_value': m.metric_value, 'split': m.split,
                    'timestamp': m.timestamp.isoformat()
                } for m in metrics
            ]
        finally:
            session.close()
    
    # Experiments
    def create_experiment(self, name: str, description: str = "", 
                          hypothesis: str = "", status: str = "active") -> int:
        """ایجاد آزمایش جدید"""
        session = self._get_session()
        try:
            experiment = Experiment(
                name=name, description=description,
                hypothesis=hypothesis, status=status
            )
            session.add(experiment)
            session.commit()
            return experiment.id
        finally:
            session.close()
    
    def add_experiment_run(self, experiment_id: int, training_run_id: int,
                           variant_name: str = "", notes: str = ""):
        """اضافه کردن run به آزمایش"""
        session = self._get_session()
        try:
            exp_run = ExperimentRun(
                experiment_id=experiment_id, training_run_id=training_run_id,
                variant_name=variant_name, notes=notes
            )
            session.add(exp_run)
            session.commit()
        finally:
            session.close()
    
    def get_experiment_runs(self, experiment_id: int) -> List[Dict]:
        """دریافت runهای یک آزمایش"""
        session = self._get_session()
        try:
            exp_runs = session.query(ExperimentRun, TrainingRun).join(
                TrainingRun, ExperimentRun.training_run_id == TrainingRun.id
            ).filter(ExperimentRun.experiment_id == experiment_id).all()
            
            return [
                {
                    'training_run_id': er.training_run_id,
                    'variant_name': er.variant_name,
                    'status': tr.status,
                    'final_loss': tr.final_loss,
                    'best_metric_value': tr.best_metric_value
                } for er, tr in exp_runs
            ]
        finally:
            session.close()
    
    # Predictions
    def log_prediction(self, model_version_id: int, input_data: Any, output_data: Any,
                       confidence: float = None, inference_time_ms: float = None,
                       metadata: Dict = None):
        """ثبت پیش‌بینی"""
        session = self._get_session()
        try:
            prediction = Prediction(
                model_version_id=model_version_id,
                input_data=input_data if isinstance(input_data, dict) else {"data": input_data},
                output_data=output_data if isinstance(output_data, dict) else {"data": output_data},
                confidence=confidence,
                inference_time_ms=inference_time_ms,
                metadata=metadata or {}
            )
            session.add(prediction)
            session.commit()
        finally:
            session.close()
    
    def get_predictions(self, model_version_id: int, limit: int = 100) -> List[Dict]:
        """دریافت پیش‌بینی‌های یک مدل"""
        session = self._get_session()
        try:
            predictions = session.query(Prediction).filter(
                Prediction.model_version_id == model_version_id
            ).order_by(Prediction.id.desc()).limit(limit).all()
            return [
                {
                    'input': p.input_data, 'output': p.output_data,
                    'confidence': p.confidence, 'inference_time_ms': p.inference_time_ms,
                    'timestamp': p.timestamp.isoformat()
                } for p in predictions
            ]
        finally:
            session.close()
    
    # Statistics
    def get_statistics(self) -> Dict[str, Any]:
        """آمار کلی پایگاه داده"""
        session = self._get_session()
        try:
            stats = {
                'models': session.query(Model).count(),
                'model_versions': session.query(ModelVersion).count(),
                'datasets': session.query(Dataset).count(),
                'training_runs': session.query(TrainingRun).count(),
                'completed_runs': session.query(TrainingRun).filter(
                    TrainingRun.status == 'completed'
                ).count(),
                'experiments': session.query(Experiment).count(),
                'predictions': session.query(Prediction).count()
            }
            return stats
        finally:
            session.close()
