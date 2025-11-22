"""
Demo: AI Model Database - MongoDB Backend
نمایش پایگاه داده MongoDB (ششمین backend)

Usage:
    python demo_ai_database_mongo.py --host localhost --port 27017 --db aidb
    python demo_ai_database_mongo.py --connection "mongodb://user:pass@localhost:27017/aidb"
    python demo_ai_database_mongo.py --mock   # Use mongomock in-memory

Install:
    pip install -r requirements-mongo-database.txt
"""
from __future__ import annotations
import argparse
from cad3d.super_ai.ai_db_factory import create_ai_database
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser(description="MongoDB AI Model Database Demo")
    p.add_argument('--host', default='localhost')
    p.add_argument('--port', type=int, default=27017)
    p.add_argument('--db', default='aidb')
    p.add_argument('--user', default=None)
    p.add_argument('--password', default=None)
    p.add_argument('--connection', default=None, help='Full MongoDB connection string')
    p.add_argument('--mock', action='store_true', help='Use mongomock (in-memory)')
    return p.parse_args()


def main():
    args = parse_args()

    print("\n🚀 MongoDB Backend Demo")
    print("==========================================\n")

    db = create_ai_database(
        backend='mongodb',
        connection_string=args.connection,
        mongo_host=args.host,
        mongo_port=args.port,
        mongo_database=args.db,
        mongo_user=args.user,
        mongo_password=args.password,
        mongo_use_mock=args.mock
    )

    if not db.ping():
        print("❌ Failed to ping MongoDB server. Check connection parameters.")
        return

    print("✅ Connected to MongoDB (mock mode:" , args.mock, ")")

    # Create sample model
    model_id = db.create_model(
        name="ResNet50-Classifier",
        description="Residual network for image classification",
        architecture="ResNet50",
        framework="PyTorch",
        task_type="image_classification"
    )
    print(f"→ Created model ID={model_id}")

    # Model version
    version_id = db.create_model_version(
        model_id=model_id,
        version="1.0.0",
        checkpoint_path="/models/resnet50_v1.pth",
        config={"lr": 1e-3, "batch_size": 32}
    )
    print(f"→ Created model version ID={version_id}")

    # Dataset
    dataset_id = db.create_dataset(
        name="ImageNet-Subset",
        description="Subset of ImageNet for quick experiments",
        source_path="/data/imagenet_subset",
        format="JPEG",
        num_samples=5000,
        split_info={"train": 4000, "val": 1000}
    )
    print(f"→ Created dataset ID={dataset_id}")

    # Training run
    run_id = db.create_training_run(
        model_version_id=version_id,
        dataset_id=dataset_id,
        run_name="resnet50_baseline"
    )
    print(f"→ Started training run ID={run_id}")

    # Log hyperparameters
    db.log_hyperparameters(run_id, {"optimizer": "AdamW", "weight_decay": 0.01})

    # Log metrics
    for epoch in range(1, 4):
        db.log_metric(run_id, "loss", 0.9 / epoch, epoch=epoch, split="train")
        db.log_metric(run_id, "accuracy", 0.5 + 0.1 * epoch, epoch=epoch, split="val")

    # Complete run
    db.update_training_run(
        run_id,
        status="completed",
        completed_at=datetime.utcnow().isoformat(),
        final_loss=0.3,
        best_metric_value=0.78
    )

    # Experiment
    exp_id = db.create_experiment(
        name="LR_Sweep",
        description="Try different learning rates",
        hypothesis="Lower LR improves validation accuracy"
    )
    db.add_experiment_run(exp_id, run_id, variant_name="lr_1e-3")

    # Predictions
    for i in range(3):
        db.log_prediction(
            model_version_id=version_id,
            input_data={"image_id": i},
            output_data={"class": "dog", "confidence": 0.8 + 0.05 * i},
            confidence=0.8 + 0.05 * i,
            inference_time_ms=12.0
        )

    # Query back data
    model = db.get_model(model_id)
    versions = db.get_model_versions(model_id)
    metrics_loss = db.get_metrics(run_id, metric_name="loss")
    predictions = db.get_predictions(version_id, limit=5)
    exp_runs = db.get_experiment_runs(exp_id)
    stats = db.get_statistics()

    print("\n📄 Model:", model)
    print("\n🧬 Versions:", versions)
    print("\n📈 Loss Metrics:", metrics_loss)
    print("\n🔮 Predictions:", predictions)
    print("\n🧪 Experiment Runs:", exp_runs)
    print("\n📊 Statistics:", stats)

    print("\n✅ MongoDB Demo Complete!\n")


if __name__ == '__main__':
    main()
