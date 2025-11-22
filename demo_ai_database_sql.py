"""
Demo: AI Model Database - SQL Backend (PostgreSQL/MySQL)
نمایش استفاده از پایگاه داده SQL برای مدل‌های هوش مصنوعی

This demo shows:
- PostgreSQL/MySQL connection
- Same API as SQLite version
- Connection pooling
- JSON field support
- Better concurrency

Requirements:
    pip install psycopg2-binary pymysql sqlalchemy
    
Setup PostgreSQL:
    createdb aimodels
    
Setup MySQL:
    mysql -u root -p
    CREATE DATABASE aimodels;
    
Usage:
    # PostgreSQL
    python demo_ai_database_sql.py --backend postgresql
    
    # MySQL
    python demo_ai_database_sql.py --backend mysql
    
    # From config file
    python demo_ai_database_sql.py --config ai_db_config.yaml
"""
from __future__ import annotations
import argparse
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cad3d.super_ai.ai_db_factory import create_ai_database, create_ai_database_from_config


def demo_sql_database(db, backend_name: str = "SQL"):
    """نمایش استفاده از پایگاه داده SQL"""
    print(f"=== AI Model Database Demo - {backend_name} Backend ===\n")
    
    # 1. Create Models
    print("📦 Creating models...")
    model1_id = db.create_model(
        name="VisionTransformer-CAD",
        description="Vision Transformer for CAD drawing analysis",
        architecture="ViT-B/16",
        framework="PyTorch",
        task_type="image_classification",
        input_shape="(3, 224, 224)",
        output_shape="(100,)"
    )
    print(f"✓ Created VisionTransformer-CAD (ID: {model1_id})")
    
    model2_id = db.create_model(
        name="VAE-3D-Generator",
        description="Variational Autoencoder for 3D model generation",
        architecture="VAE-ConvNet",
        framework="PyTorch",
        task_type="generation",
        input_shape="(512,)",
        output_shape="(3, 64, 64, 64)"
    )
    print(f"✓ Created VAE-3D-Generator (ID: {model2_id})")
    
    model3_id = db.create_model(
        name="GNN-Structural-Analysis",
        description="Graph Neural Network for structural analysis",
        architecture="GraphSAGE",
        framework="PyTorch Geometric",
        task_type="node_classification",
        input_shape="graph",
        output_shape="(5,)"
    )
    print(f"✓ Created GNN-Structural-Analysis (ID: {model3_id})")
    
    # 2. Create Model Versions
    print("\n📋 Creating model versions...")
    version1_id = db.create_model_version(
        model_id=model1_id,
        version="v1.0.0",
        checkpoint_path="models/vit_cad_v1.pth",
        config={
            "num_layers": 12,
            "hidden_size": 768,
            "num_heads": 12,
            "patch_size": 16
        },
        status="active"
    )
    print(f"✓ Created version v1.0.0 for VisionTransformer-CAD")
    
    version2_id = db.create_model_version(
        model_id=model2_id,
        version="v2.1.0",
        checkpoint_path="models/vae_3d_v2.pth",
        config={
            "latent_dim": 512,
            "encoder_layers": [64, 128, 256, 512],
            "decoder_layers": [512, 256, 128, 64]
        },
        status="active"
    )
    print(f"✓ Created version v2.1.0 for VAE-3D-Generator")
    
    # 3. Create Datasets
    print("\n💾 Creating datasets...")
    dataset1_id = db.create_dataset(
        name="CAD-Drawings-2024",
        description="Large-scale CAD drawings dataset",
        source_path="/data/cad_drawings_2024",
        format="DXF",
        size_bytes=1073741824,  # 1GB
        num_samples=50000,
        split_info={
            "train": 40000,
            "val": 5000,
            "test": 5000
        },
        preprocessing={
            "resize": [224, 224],
            "normalize": True,
            "augmentation": ["flip", "rotate", "scale"]
        }
    )
    print(f"✓ Created CAD-Drawings-2024 dataset")
    
    dataset2_id = db.create_dataset(
        name="3D-Models-Structural",
        description="3D structural models for training",
        source_path="/data/3d_structural",
        format="OBJ/STL",
        size_bytes=2147483648,  # 2GB
        num_samples=10000,
        split_info={
            "train": 8000,
            "val": 1000,
            "test": 1000
        },
        preprocessing={
            "voxel_size": 64,
            "normalize": True,
            "center": True
        }
    )
    print(f"✓ Created 3D-Models-Structural dataset")
    
    # 4. Create Training Runs
    print("\n🚀 Creating training runs...")
    run1_id = db.create_training_run(
        model_version_id=version1_id,
        dataset_id=dataset1_id,
        run_name="VisionTransformer training - baseline",
        status="running"
    )
    print(f"✓ Started training run for VisionTransformer-CAD")
    
    run2_id = db.create_training_run(
        model_version_id=version2_id,
        dataset_id=dataset2_id,
        run_name="VAE training - improved loss",
        status="running"
    )
    print(f"✓ Started training run for VAE-3D-Generator")
    
    # 5. Log Hyperparameters
    print("\n⚙️ Logging hyperparameters...")
    db.log_hyperparameters(run1_id, {
        "learning_rate": 0.0001,
        "batch_size": 32,
        "num_epochs": 100,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "warmup_steps": 1000
    })
    print(f"✓ Logged hyperparameters for run {run1_id}")
    
    db.log_hyperparameters(run2_id, {
        "learning_rate": 0.0005,
        "batch_size": 16,
        "num_epochs": 200,
        "optimizer": "Adam",
        "beta_kl": 0.5,
        "beta_reconstruction": 1.0
    })
    print(f"✓ Logged hyperparameters for run {run2_id}")
    
    # 6. Log Metrics (simulate training)
    print("\n📊 Logging training metrics...")
    for epoch in range(1, 6):
        # VisionTransformer metrics
        train_loss = 2.5 - (epoch * 0.3)
        train_acc = 0.4 + (epoch * 0.1)
        db.log_metric(run1_id, "loss", train_loss, epoch=epoch, split="train")
        db.log_metric(run1_id, "accuracy", train_acc, epoch=epoch, split="train")
        
        val_loss = 2.7 - (epoch * 0.25)
        val_acc = 0.35 + (epoch * 0.12)
        db.log_metric(run1_id, "loss", val_loss, epoch=epoch, split="val")
        db.log_metric(run1_id, "accuracy", val_acc, epoch=epoch, split="val")
    
    print(f"✓ Logged 5 epochs of metrics for VisionTransformer")
    
    for epoch in range(1, 6):
        # VAE metrics
        train_loss = 150 - (epoch * 20)
        recon_loss = 100 - (epoch * 15)
        kl_loss = 50 - (epoch * 5)
        db.log_metric(run2_id, "total_loss", train_loss, epoch=epoch, split="train")
        db.log_metric(run2_id, "reconstruction_loss", recon_loss, epoch=epoch, split="train")
        db.log_metric(run2_id, "kl_loss", kl_loss, epoch=epoch, split="train")
    
    print(f"✓ Logged 5 epochs of metrics for VAE")
    
    # 7. Complete Training Run
    print("\n✅ Completing training run...")
    db.update_training_run(
        run1_id,
        status="completed",
        completed_at=datetime.utcnow().isoformat(),
        duration_seconds=3600.5,
        final_loss=1.2,
        best_metric_value=0.89,
        artifacts={
            "checkpoint": "models/vit_cad_v1_best.pth",
            "logs": "logs/vit_train_run1.log",
            "tensorboard": "runs/vit_run1"
        }
    )
    print(f"✓ Training run {run1_id} completed")
    
    # 8. Log Predictions
    print("\n🔮 Logging predictions...")
    for i in range(5):
        db.log_prediction(
            model_version_id=version1_id,
            input_data={"image": f"test_image_{i}.jpg"},
            output_data={"class": i % 3, "label": ["wall", "column", "beam"][i % 3]},
            confidence=0.85 + (i * 0.02),
            inference_time_ms=25.5 + (i * 2.3),
            metadata={"device": "cuda", "batch_size": 1}
        )
    print(f"✓ Logged 5 predictions")
    
    # 9. Create Experiment
    print("\n🧪 Creating experiment...")
    exp_id = db.create_experiment(
        name="VisionTransformer-Hyperparameter-Tuning",
        description="Testing different learning rates and batch sizes",
        hypothesis="Larger batch size improves convergence",
        status="active"
    )
    db.add_experiment_run(exp_id, run1_id, variant_name="baseline", notes="Standard hyperparameters")
    print(f"✓ Created experiment and added baseline run")
    
    # 10. Query Statistics
    print("\n📈 Database Statistics:")
    stats = db.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # 11. Query Examples
    print("\n🔍 Query Examples:")
    
    # Get model info
    model = db.get_model(model1_id)
    print(f"\n   Model: {model['name']}")
    print(f"   Architecture: {model['architecture']}")
    print(f"   Framework: {model['framework']}")
    
    # Get training run info
    run = db.get_training_run(run1_id)
    print(f"\n   Training Run: {run['run_name']}")
    print(f"   Status: {run['status']}")
    print(f"   Duration: {run['duration_seconds']:.1f}s")
    print(f"   Final Loss: {run['final_loss']}")
    print(f"   Best Metric: {run['best_metric_value']}")
    
    # Get metrics
    metrics = db.get_metrics(run1_id, "accuracy")
    print(f"\n   Accuracy History:")
    for m in metrics[-3:]:  # Last 3 epochs
        print(f"      Epoch {m['epoch']} ({m['split']}): {m['metric_value']:.3f}")
    
    # Get predictions
    predictions = db.get_predictions(version1_id, limit=3)
    print(f"\n   Recent Predictions:")
    for p in predictions:
        print(f"      Output: {p['output']}, Confidence: {p['confidence']:.2f}")
    
    print(f"\n✅ Demo completed successfully! Using {backend_name} backend.\n")


def main():
    parser = argparse.ArgumentParser(description="AI Model Database SQL Demo")
    parser.add_argument('--backend', choices=['postgresql', 'mysql'], 
                        default='postgresql',
                        help='Database backend (default: postgresql)')
    parser.add_argument('--config', type=str,
                        help='Path to config file (YAML/JSON)')
    parser.add_argument('--host', default='localhost',
                        help='Database host (default: localhost)')
    parser.add_argument('--port', type=int,
                        help='Database port (5432 for PostgreSQL, 3306 for MySQL)')
    parser.add_argument('--database', default='aimodels',
                        help='Database name (default: aimodels)')
    parser.add_argument('--user', default='postgres',
                        help='Database user (default: postgres)')
    parser.add_argument('--password', default='',
                        help='Database password')
    parser.add_argument('--echo', action='store_true',
                        help='Echo SQL queries')
    
    args = parser.parse_args()
    
    try:
        if args.config:
            # Load from config file
            print(f"Loading configuration from: {args.config}\n")
            db = create_ai_database_from_config(args.config)
            backend_name = "SQL (from config)"
        else:
            # Build connection string
            if args.port is None:
                args.port = 5432 if args.backend == 'postgresql' else 3306
            
            if args.backend == 'postgresql':
                connection_string = f"postgresql://{args.user}:{args.password}@{args.host}:{args.port}/{args.database}"
                backend_name = "PostgreSQL"
            else:  # mysql
                connection_string = f"mysql+pymysql://{args.user}:{args.password}@{args.host}:{args.port}/{args.database}"
                backend_name = "MySQL"
            
            print(f"Connecting to {backend_name}...")
            print(f"Host: {args.host}:{args.port}")
            print(f"Database: {args.database}\n")
            
            db = create_ai_database(
                backend=args.backend,
                connection_string=connection_string,
                echo=args.echo
            )
        
        # Run demo
        demo_sql_database(db, backend_name)
        
    except ImportError as e:
        print(f"❌ Error: Missing required package")
        print(f"   {e}")
        print("\n💡 Install required packages:")
        print("   pip install psycopg2-binary pymysql sqlalchemy")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure:")
        print("   1. Database server is running")
        print("   2. Database exists (createdb aimodels or CREATE DATABASE aimodels)")
        print("   3. Credentials are correct")
        print("   4. Required packages are installed (pip install psycopg2-binary pymysql sqlalchemy)")
        sys.exit(1)


if __name__ == '__main__':
    main()
