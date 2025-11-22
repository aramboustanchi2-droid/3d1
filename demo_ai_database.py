"""
AI Model Database Demo & Initialization
نمایش و راه‌اندازی پایگاه داده مدل‌های هوش مصنوعی

Usage:
    python demo_ai_database.py
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cad3d.super_ai.ai_model_database import ai_db

print("\n" + "="*80)
print("  🤖 AI MODEL DATABASE DEMO")
print("  سیستم مدیریت پایگاه داده مدل‌های هوش مصنوعی")
print("="*80 + "\n")

# 1. Create Models
print("📦 Creating AI Models...")
model1_id = ai_db.create_model(
    name="VisionTransformer-CAD",
    description="Vision Transformer for CAD drawing analysis",
    architecture="ViT-Base/16",
    framework="PyTorch",
    task_type="image_classification",
    input_shape="(3, 224, 224)",
    output_shape="(14,)"
)
print(f"  ✓ Created model ID: {model1_id}")

model2_id = ai_db.create_model(
    name="VAE-3D-Generator",
    description="Variational Autoencoder for 3D mesh generation",
    architecture="VAE",
    framework="PyTorch",
    task_type="generation",
    input_shape="(256,)",
    output_shape="(N, 3)"
)
print(f"  ✓ Created model ID: {model2_id}")

model3_id = ai_db.create_model(
    name="GNN-Structural-Analysis",
    description="Graph Neural Network for structural relationship analysis",
    architecture="GCN",
    framework="PyTorch",
    task_type="graph_classification",
    input_shape="graph",
    output_shape="(5,)"
)
print(f"  ✓ Created model ID: {model3_id}\n")

# 2. Create Model Versions
print("🔢 Creating Model Versions...")
version1_id = ai_db.create_model_version(
    model_id=model1_id,
    version="v1.0.0",
    checkpoint_path="checkpoints/vit_cad_v1.pth",
    config={"hidden_dim": 768, "num_layers": 12, "num_heads": 12},
    status="active"
)
print(f"  ✓ Created version ID: {version1_id}")

version2_id = ai_db.create_model_version(
    model_id=model2_id,
    version="v1.0.0",
    checkpoint_path="checkpoints/vae_3d_v1.pth",
    config={"latent_dim": 256, "encoder_layers": [512, 256], "decoder_layers": [256, 512]},
    status="active"
)
print(f"  ✓ Created version ID: {version2_id}\n")

# 3. Create Datasets
print("📊 Creating Datasets...")
dataset1_id = ai_db.create_dataset(
    name="CAD-Drawing-Dataset",
    description="2D CAD drawings from multiple industries",
    source_path="datasets/cad_drawings",
    format="DXF",
    size_bytes=5_000_000_000,
    num_samples=50000,
    split_info={"train": 40000, "val": 5000, "test": 5000},
    preprocessing={"resize": [224, 224], "normalize": True, "augmentation": ["rotate", "flip"]}
)
print(f"  ✓ Created dataset ID: {dataset1_id}")

dataset2_id = ai_db.create_dataset(
    name="3D-Mesh-Dataset",
    description="3D mesh models for generation training",
    source_path="datasets/3d_meshes",
    format="OBJ",
    size_bytes=10_000_000_000,
    num_samples=25000,
    split_info={"train": 20000, "val": 2500, "test": 2500},
    preprocessing={"normalize": True, "subsample": 2048}
)
print(f"  ✓ Created dataset ID: {dataset2_id}\n")

# 4. Create Training Runs
print("🏋️ Creating Training Runs...")
run1_id = ai_db.create_training_run(
    model_version_id=version1_id,
    dataset_id=dataset1_id,
    run_name="ViT-CAD-Experiment-1",
    status="started"
)
print(f"  ✓ Created training run ID: {run1_id}")

# Log hyperparameters
ai_db.log_hyperparameters(run1_id, {
    "learning_rate": 1e-4,
    "batch_size": 32,
    "num_epochs": 50,
    "optimizer": "AdamW",
    "weight_decay": 0.01,
    "warmup_steps": 1000
})
print(f"  ✓ Logged hyperparameters")

# Simulate training metrics
print(f"  📈 Logging training metrics...")
for epoch in range(1, 6):  # Simulate 5 epochs
    train_loss = 2.5 - (epoch * 0.3)
    val_loss = 2.6 - (epoch * 0.25)
    train_acc = 0.3 + (epoch * 0.12)
    val_acc = 0.28 + (epoch * 0.11)
    
    ai_db.log_metric(run1_id, "loss", train_loss, epoch=epoch, split="train")
    ai_db.log_metric(run1_id, "loss", val_loss, epoch=epoch, split="val")
    ai_db.log_metric(run1_id, "accuracy", train_acc, epoch=epoch, split="train")
    ai_db.log_metric(run1_id, "accuracy", val_acc, epoch=epoch, split="val")

print(f"  ✓ Logged metrics for 5 epochs")

# Complete training run
ai_db.update_training_run(
    run1_id,
    status="completed",
    completed_at=datetime.utcnow().isoformat(),
    duration_seconds=3600.5,
    final_loss=1.0,
    best_metric_value=0.85,
    artifacts={"best_checkpoint": "checkpoints/vit_cad_v1_best.pth", "tensorboard_logs": "logs/run1"}
)
print(f"  ✓ Training run completed\n")

# 5. Create second training run
run2_id = ai_db.create_training_run(
    model_version_id=version2_id,
    dataset_id=dataset2_id,
    run_name="VAE-3D-Generation-1",
    status="started"
)
print(f"  ✓ Created second training run ID: {run2_id}")

ai_db.log_hyperparameters(run2_id, {
    "learning_rate": 5e-4,
    "batch_size": 64,
    "num_epochs": 100,
    "optimizer": "Adam",
    "beta": 0.5,
    "kl_weight": 0.001
})

for epoch in range(1, 4):
    recon_loss = 0.8 - (epoch * 0.1)
    kl_loss = 0.05 + (epoch * 0.01)
    
    ai_db.log_metric(run2_id, "reconstruction_loss", recon_loss, epoch=epoch, split="train")
    ai_db.log_metric(run2_id, "kl_divergence", kl_loss, epoch=epoch, split="train")

ai_db.update_training_run(
    run2_id,
    status="running",
    artifacts={"checkpoint": "checkpoints/vae_3d_v1_epoch3.pth"}
)
print(f"  ✓ Second training run in progress\n")

# 6. Create Experiment
print("🔬 Creating Experiment...")
exp1_id = ai_db.create_experiment(
    name="ViT-Architecture-Comparison",
    description="Compare different ViT architectures for CAD analysis",
    hypothesis="Larger patch size improves efficiency without sacrificing accuracy",
    status="active"
)
print(f"  ✓ Created experiment ID: {exp1_id}")

ai_db.add_experiment_run(exp1_id, run1_id, variant_name="ViT-Base/16", notes="Baseline model")
print(f"  ✓ Added run to experiment\n")

# 7. Log Predictions
print("🎯 Logging Predictions...")
for i in range(5):
    ai_db.log_prediction(
        model_version_id=version1_id,
        input_data={"image_path": f"test_images/sample_{i}.dxf", "preprocessing": "standard"},
        output_data={"class": i % 14, "probabilities": [0.1 + (0.05 * j) for j in range(14)]},
        confidence=0.85 + (i * 0.02),
        inference_time_ms=15.3 + (i * 0.5),
        metadata={"device": "cuda:0", "batch_size": 1}
    )
print(f"  ✓ Logged 5 predictions\n")

# 8. Display Statistics
print("="*80)
print("  📊 DATABASE STATISTICS")
print("="*80 + "\n")

stats = ai_db.get_statistics()
for key, value in stats.items():
    key_display = key.replace('_', ' ').title()
    print(f"  {key_display:<25}: {value}")

print("\n" + "="*80)
print("  🔍 QUERY EXAMPLES")
print("="*80 + "\n")

# List all models
print("📦 All Models:")
models = ai_db.list_models()
for m in models:
    print(f"  • {m['name']} ({m['framework']}) - {m['task_type']}")

# Get model versions
print(f"\n🔢 Versions of '{models[0]['name']}':")
versions = ai_db.get_model_versions(model1_id)
for v in versions:
    print(f"  • Version {v['version']} - {v['status']} (created: {v['created_at'][:10]})")

# Get training run details
print(f"\n🏋️ Training Run Details (ID: {run1_id}):")
run_info = ai_db.get_training_run(run1_id)
print(f"  Run Name: {run_info['run_name']}")
print(f"  Status: {run_info['status']}")
print(f"  Duration: {run_info['duration_seconds']:.1f}s")
print(f"  Final Loss: {run_info['final_loss']}")
print(f"  Best Metric: {run_info['best_metric_value']}")

# Get hyperparameters
print(f"\n⚙️  Hyperparameters:")
hypers = ai_db.get_hyperparameters(run1_id)
for name, value in hypers.items():
    print(f"  {name}: {value}")

# Get metrics
print(f"\n📈 Training Metrics (last 3 records):")
metrics = ai_db.get_metrics(run1_id, "accuracy")[-3:]
for m in metrics:
    print(f"  Epoch {m['epoch']} ({m['split']}): {m['metric_value']:.3f}")

# Get experiment runs
print(f"\n🔬 Experiment Runs:")
exp_runs = ai_db.get_experiment_runs(exp1_id)
for er in exp_runs:
    print(f"  Variant: {er['variant_name']} | Status: {er['status']} | Best: {er['best_metric_value']}")

# Get predictions
print(f"\n🎯 Recent Predictions (last 3):")
preds = ai_db.get_predictions(version1_id, limit=3)
for p in preds:
    print(f"  Class: {p['output']['class']} | Confidence: {p['confidence']:.2f} | Time: {p['inference_time_ms']:.1f}ms")

print("\n" + "="*80)
print("  ✅ DATABASE DEMO COMPLETED")
print("="*80)
print(f"\n📁 Database file: ai_models.db")
print(f"📚 Module: cad3d/super_ai/ai_model_database.py\n")
