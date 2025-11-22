"""
Test Suite for AI Model Database
تست‌های پایگاه داده مدل‌های هوش مصنوعی

Run:
    pytest tests/test_ai_database.py -v
"""
import pytest
from pathlib import Path
import tempfile
import json
from datetime import datetime

from cad3d.super_ai.ai_model_database import AIModelDatabase

@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    
    db = AIModelDatabase(db_path)
    yield db
    
    # Cleanup
    if db_path.exists():
        db_path.unlink()

class TestModelOperations:
    """تست عملیات مربوط به مدل‌ها"""
    
    def test_create_model(self, temp_db):
        """تست ساخت مدل جدید"""
        model_id = temp_db.create_model(
            name="TestModel",
            description="A test model",
            architecture="ResNet50",
            framework="PyTorch",
            task_type="classification"
        )
        assert model_id > 0
        
        model = temp_db.get_model(model_id)
        assert model is not None
        assert model['name'] == "TestModel"
        assert model['architecture'] == "ResNet50"
        assert model['framework'] == "PyTorch"
    
    def test_list_models(self, temp_db):
        """تست لیست کردن مدل‌ها"""
        temp_db.create_model("Model1", architecture="ViT")
        temp_db.create_model("Model2", architecture="VAE")
        
        models = temp_db.list_models()
        assert len(models) == 2
        assert any(m['name'] == "Model1" for m in models)
        assert any(m['name'] == "Model2" for m in models)
    
    def test_duplicate_model_name(self, temp_db):
        """تست نام تکراری برای مدل"""
        temp_db.create_model("UniqueModel")
        
        with pytest.raises(Exception):
            temp_db.create_model("UniqueModel")

class TestModelVersions:
    """تست نسخه‌های مدل"""
    
    def test_create_version(self, temp_db):
        """تست ساخت نسخه جدید"""
        model_id = temp_db.create_model("VersionedModel")
        
        version_id = temp_db.create_model_version(
            model_id=model_id,
            version="v1.0.0",
            checkpoint_path="/path/to/checkpoint.pth",
            config={"layers": 12},
            status="active"
        )
        assert version_id > 0
    
    def test_list_versions(self, temp_db):
        """تست لیست نسخه‌ها"""
        model_id = temp_db.create_model("MultiVersionModel")
        
        temp_db.create_model_version(model_id, "v1.0.0")
        temp_db.create_model_version(model_id, "v1.1.0")
        temp_db.create_model_version(model_id, "v2.0.0")
        
        versions = temp_db.get_model_versions(model_id)
        assert len(versions) == 3
        assert versions[0]['version'] == "v2.0.0"  # Most recent first

class TestDatasets:
    """تست دیتاست‌ها"""
    
    def test_create_dataset(self, temp_db):
        """تست ساخت دیتاست"""
        dataset_id = temp_db.create_dataset(
            name="TestDataset",
            description="Test data",
            source_path="/data/test",
            format="CSV",
            size_bytes=1000000,
            num_samples=10000,
            split_info={"train": 8000, "test": 2000},
            preprocessing={"normalize": True}
        )
        assert dataset_id > 0
        
        dataset = temp_db.get_dataset(dataset_id)
        assert dataset is not None
        assert dataset['name'] == "TestDataset"
        assert dataset['num_samples'] == 10000
        assert dataset['split_info']['train'] == 8000
        assert dataset['preprocessing']['normalize'] is True

class TestTrainingRuns:
    """تست آموزش‌های مدل"""
    
    def test_create_training_run(self, temp_db):
        """تست ساخت run آموزش"""
        model_id = temp_db.create_model("TrainedModel")
        version_id = temp_db.create_model_version(model_id, "v1.0.0")
        dataset_id = temp_db.create_dataset("TrainDataset", num_samples=1000)
        
        run_id = temp_db.create_training_run(
            model_version_id=version_id,
            dataset_id=dataset_id,
            run_name="Test Run",
            status="started"
        )
        assert run_id > 0
        
        run = temp_db.get_training_run(run_id)
        assert run is not None
        assert run['status'] == "started"
        assert run['run_name'] == "Test Run"
    
    def test_update_training_run(self, temp_db):
        """تست به‌روزرسانی run آموزش"""
        model_id = temp_db.create_model("UpdateModel")
        version_id = temp_db.create_model_version(model_id, "v1.0.0")
        run_id = temp_db.create_training_run(version_id, status="started")
        
        temp_db.update_training_run(
            run_id,
            status="completed",
            final_loss=0.5,
            best_metric_value=0.95,
            duration_seconds=3600.0
        )
        
        run = temp_db.get_training_run(run_id)
        assert run['status'] == "completed"
        assert run['final_loss'] == 0.5
        assert run['best_metric_value'] == 0.95
        assert run['duration_seconds'] == 3600.0

class TestHyperparameters:
    """تست هایپرپارامترها"""
    
    def test_log_hyperparameters(self, temp_db):
        """تست ثبت هایپرپارامترها"""
        model_id = temp_db.create_model("HyperModel")
        version_id = temp_db.create_model_version(model_id, "v1.0.0")
        run_id = temp_db.create_training_run(version_id)
        
        params = {
            "learning_rate": 1e-4,
            "batch_size": 32,
            "optimizer": "Adam"
        }
        temp_db.log_hyperparameters(run_id, params)
        
        retrieved = temp_db.get_hyperparameters(run_id)
        assert retrieved["learning_rate"] == str(1e-4)
        assert retrieved["batch_size"] == "32"
        assert retrieved["optimizer"] == "Adam"

class TestMetrics:
    """تست متریک‌ها"""
    
    def test_log_metrics(self, temp_db):
        """تست ثبت متریک"""
        model_id = temp_db.create_model("MetricModel")
        version_id = temp_db.create_model_version(model_id, "v1.0.0")
        run_id = temp_db.create_training_run(version_id)
        
        temp_db.log_metric(run_id, "loss", 2.5, epoch=1, split="train")
        temp_db.log_metric(run_id, "loss", 1.8, epoch=2, split="train")
        temp_db.log_metric(run_id, "accuracy", 0.75, epoch=1, split="val")
        
        loss_metrics = temp_db.get_metrics(run_id, "loss")
        assert len(loss_metrics) == 2
        assert loss_metrics[0]['metric_value'] == 2.5
        assert loss_metrics[1]['metric_value'] == 1.8
        
        all_metrics = temp_db.get_metrics(run_id)
        assert len(all_metrics) == 3

class TestExperiments:
    """تست آزمایش‌ها"""
    
    def test_create_experiment(self, temp_db):
        """تست ساخت آزمایش"""
        exp_id = temp_db.create_experiment(
            name="TestExperiment",
            description="Testing experiments",
            hypothesis="Hypothesis here",
            status="active"
        )
        assert exp_id > 0
    
    def test_experiment_runs(self, temp_db):
        """تست runهای آزمایش"""
        exp_id = temp_db.create_experiment("MultiRunExperiment")
        
        model_id = temp_db.create_model("ExpModel")
        version_id = temp_db.create_model_version(model_id, "v1.0.0")
        
        run1_id = temp_db.create_training_run(version_id, run_name="Variant A")
        run2_id = temp_db.create_training_run(version_id, run_name="Variant B")
        
        temp_db.update_training_run(run1_id, status="completed", final_loss=0.5, best_metric_value=0.9)
        temp_db.update_training_run(run2_id, status="completed", final_loss=0.4, best_metric_value=0.92)
        
        temp_db.add_experiment_run(exp_id, run1_id, variant_name="A", notes="Baseline")
        temp_db.add_experiment_run(exp_id, run2_id, variant_name="B", notes="Improved")
        
        runs = temp_db.get_experiment_runs(exp_id)
        assert len(runs) == 2
        assert runs[0]['variant_name'] == "A"
        assert runs[1]['variant_name'] == "B"

class TestPredictions:
    """تست پیش‌بینی‌ها"""
    
    def test_log_predictions(self, temp_db):
        """تست ثبت پیش‌بینی"""
        model_id = temp_db.create_model("PredModel")
        version_id = temp_db.create_model_version(model_id, "v1.0.0")
        
        temp_db.log_prediction(
            model_version_id=version_id,
            input_data={"image": "test.jpg"},
            output_data={"class": 5, "prob": 0.95},
            confidence=0.95,
            inference_time_ms=12.5
        )
        
        preds = temp_db.get_predictions(version_id, limit=10)
        assert len(preds) == 1
        assert preds[0]['input']['image'] == "test.jpg"
        assert preds[0]['output']['class'] == 5
        assert preds[0]['confidence'] == 0.95

class TestStatistics:
    """تست آمار کلی"""
    
    def test_statistics(self, temp_db):
        """تست آمار پایگاه داده"""
        # Create some data
        model_id = temp_db.create_model("StatsModel")
        version_id = temp_db.create_model_version(model_id, "v1.0.0")
        dataset_id = temp_db.create_dataset("StatsDataset", num_samples=100)
        run_id = temp_db.create_training_run(version_id, dataset_id)
        temp_db.update_training_run(run_id, status="completed")
        exp_id = temp_db.create_experiment("StatsExp")
        temp_db.log_prediction(version_id, {"x": 1}, {"y": 2})
        
        stats = temp_db.get_statistics()
        
        assert stats['models'] >= 1
        assert stats['model_versions'] >= 1
        assert stats['datasets'] >= 1
        assert stats['training_runs'] >= 1
        assert stats['completed_runs'] >= 1
        assert stats['experiments'] >= 1
        assert stats['predictions'] >= 1

class TestDataIntegrity:
    """تست یکپارچگی داده"""
    
    def test_foreign_key_model_version(self, temp_db):
        """تست foreign key برای نسخه مدل"""
        # Try to create version for non-existent model
        with pytest.raises(Exception):
            temp_db.create_model_version(model_id=999999, version="v1.0.0")
    
    def test_json_serialization(self, temp_db):
        """تست سریالیزه کردن JSON"""
        dataset_id = temp_db.create_dataset(
            "JSONTest",
            split_info={"train": 80, "val": 10, "test": 10},
            preprocessing={"ops": ["resize", "normalize"], "params": {"size": 224}}
        )
        
        dataset = temp_db.get_dataset(dataset_id)
        assert isinstance(dataset['split_info'], dict)
        assert dataset['split_info']['train'] == 80
        assert isinstance(dataset['preprocessing'], dict)
        assert dataset['preprocessing']['ops'][0] == "resize"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
