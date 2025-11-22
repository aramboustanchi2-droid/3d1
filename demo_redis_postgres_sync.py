"""
Redis → PostgreSQL Sync - Demo Script
نمونه اجرای همگام‌سازی Redis به PostgreSQL

This script demonstrates:
1. Populate Redis with sample data
2. Run sync to PostgreSQL
3. Verify data transfer
4. Show statistics

Run:
    python demo_redis_postgres_sync.py

Requirements:
    - PostgreSQL server running
    - POSTGRES_URL environment variable set
    - Redis server running (or use REDIS_URL)
"""
import os
from cad3d.super_ai.ai_db_factory import create_ai_database
from cad3d.super_ai.redis_postgres_sync import RedisPgSync

def populate_redis(redis_db):
    """Add sample data to Redis"""
    print("\n[1] Populating Redis with sample data...")
    
    # Models
    model1_id = redis_db.create_model(
        name="ResNet50-Redis",
        description="Deep residual network for image classification",
        architecture="ResNet50",
        framework="PyTorch",
        task_type="image_classification"
    )
    print(f"  ✓ Model 1: ResNet50-Redis (ID: {model1_id})")
    
    model2_id = redis_db.create_model(
        name="BERT-Redis",
        description="Transformer model for NLP",
        architecture="BERT-base",
        framework="Transformers",
        task_type="text_classification"
    )
    print(f"  ✓ Model 2: BERT-Redis (ID: {model2_id})")
    
    # Datasets
    dataset1_id = redis_db.create_dataset(
        name="ImageNet-Redis",
        description="Image classification dataset",
        source_path="/data/imagenet",
        format="JPEG",
        size_bytes=150_000_000_000,
        num_samples=1_281_167
    )
    print(f"  ✓ Dataset 1: ImageNet-Redis (ID: {dataset1_id})")
    
    dataset2_id = redis_db.create_dataset(
        name="IMDB-Redis",
        description="Text classification dataset",
        source_path="/data/imdb",
        format="CSV",
        size_bytes=84_000_000,
        num_samples=50_000
    )
    print(f"  ✓ Dataset 2: IMDB-Redis (ID: {dataset2_id})")
    
    # Version
    version_id = redis_db.create_model_version(
        model_id=model1_id,
        version="1.0.0",
        checkpoint_path="/models/resnet50_v1.pth",
        config={"lr": 0.001, "batch_size": 32}
    )
    print(f"  ✓ Version: 1.0.0 (ID: {version_id})")
    
    # Training run
    run_id = redis_db.create_training_run(
        model_version_id=version_id,
        dataset_id=dataset1_id,
        run_name="ResNet50-Training",
        status="completed"
    )
    print(f"  ✓ Training Run (ID: {run_id})")
    
    # Hyperparameters
    redis_db.log_hyperparameters(run_id, {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10
    })
    print(f"  ✓ Hyperparameters logged")
    
    # Metrics
    for epoch in range(5):
        redis_db.log_metric(run_id, "loss", 0.5 - epoch * 0.08, epoch=epoch, split="train")
        redis_db.log_metric(run_id, "accuracy", 0.6 + epoch * 0.05, epoch=epoch, split="val")
    print(f"  ✓ Metrics logged (5 epochs)")
    
    # Predictions
    for i in range(10):
        redis_db.log_prediction(
            model_version_id=version_id,
            input_data={"image_id": i},
            output_data={"class": f"class_{i % 3}"},
            confidence=0.85 + i * 0.01,
            inference_time_ms=12.5
        )
    print(f"  ✓ Predictions logged (10 samples)")
    
    print("\n  Redis populated successfully!")
    return redis_db.get_statistics()


def verify_postgres(postgres_db):
    """Verify data in PostgreSQL"""
    print("\n[3] Verifying PostgreSQL data...")
    
    stats = postgres_db.get_statistics()
    print(f"  Models: {stats['models']}")
    print(f"  Datasets: {stats['datasets']}")
    print(f"  Versions: {stats['model_versions']}")
    print(f"  Runs: {stats['training_runs']}")
    print(f"  Experiments: {stats['experiments']}")
    
    # List models
    models = postgres_db.list_models()
    if models:
        print(f"\n  Models in PostgreSQL:")
        for m in models:
            print(f"    - {m['name']} ({m['architecture']})")
    
    # List datasets
    datasets = postgres_db.list_datasets()
    if datasets:
        print(f"\n  Datasets in PostgreSQL:")
        for d in datasets:
            print(f"    - {d['name']} ({d['format']})")
    
    return stats


def main():
    print("\n" + "="*70)
    print("Redis → PostgreSQL Sync Demo")
    print("="*70)
    
    # Check environment
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    postgres_url = os.getenv('POSTGRES_URL')
    
    if not postgres_url:
        print("\n❌ Error: POSTGRES_URL not set")
        print("\nSet environment variable:")
        print("  $env:POSTGRES_URL='postgresql://user:password@localhost:5432/ai_models'")
        print("\nOr create a test database:")
        print("  CREATE DATABASE ai_models;")
        return 1
    
    try:
        # Connect to databases
        print(f"\nConnecting to Redis: {redis_url}")
        redis_db = create_ai_database(backend='redis', connection_string=redis_url)
        print("✓ Redis connected")
        
        print(f"\nConnecting to PostgreSQL: {postgres_url[:30]}...")
        postgres_db = create_ai_database(backend='postgresql', connection_string=postgres_url)
        print("✓ PostgreSQL connected")
        
        # Populate Redis
        redis_stats_before = populate_redis(redis_db)
        print(f"\nRedis stats before sync:")
        print(f"  Models: {redis_stats_before['models']}")
        print(f"  Datasets: {redis_stats_before['datasets']}")
        print(f"  Versions: {redis_stats_before['model_versions']}")
        
        # Run sync
        print("\n" + "="*70)
        print("[2] Running Redis → PostgreSQL Sync")
        print("="*70)
        
        sync = RedisPgSync(redis_db, postgres_db)
        sync_stats = sync.full_sync()
        
        # Verify PostgreSQL
        print("\n" + "="*70)
        pg_stats = verify_postgres(postgres_db)
        
        # Summary
        print("\n" + "="*70)
        print("Summary")
        print("="*70)
        print(f"\nSync Results:")
        print(f"  Models synced: {sync_stats['models']}")
        print(f"  Datasets synced: {sync_stats['datasets']}")
        print(f"  Errors: {sync_stats['errors']}")
        
        print(f"\nPostgreSQL Final State:")
        print(f"  Models: {pg_stats['models']}")
        print(f"  Datasets: {pg_stats['datasets']}")
        
        print("\n✅ Demo completed successfully!")
        print("\nNote: Full sync (versions, runs, predictions) requires ID mapping")
        print("      Current demo syncs models and datasets only")
        
        return 0
    
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("\nInstall required packages:")
        print("  pip install redis psycopg2-binary sqlalchemy")
        return 1
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
