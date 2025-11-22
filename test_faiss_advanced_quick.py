"""
Quick test for FAISS Advanced (IVF/PQ) implementation
تست سریع برای پیاده‌سازی FAISS پیشرفته
"""
import sys
from pathlib import Path

# Add cad3d to path
sys.path.insert(0, str(Path(__file__).parent))

from cad3d.super_ai.ai_model_database_faiss_advanced import AIModelDatabaseFAISSAdvanced


def test_flat_index():
    """Test IndexFlatL2 (exact search)"""
    print("\n" + "="*70)
    print("Test 1: IndexFlatL2 (Exact Search)")
    print("="*70)
    
    db = AIModelDatabaseFAISSAdvanced(
        index_path="test_faiss_data/flat",
        dimension=384,
        index_type='flat'
    )
    
    # Add 100 models
    for i in range(100):
        db.create_model(
            name=f"model_{i}",
            description=f"A CNN model for classification task {i}",
            architecture="CNN"
        )
    
    # Search
    results = db.search_similar_models("CNN classification", n_results=5)
    
    stats = db.get_statistics()
    print(f"✓ Created {stats['models']} models")
    print(f"✓ Index type: {stats['index_info']['models_index_type']}")
    print(f"✓ Top result: {results[0]['name']} (score: {results[0]['similarity_score']:.4f})")
    
    assert len(results) == 5, "Should return 5 results"
    assert results[0]['similarity_score'] > 0, "Should have positive similarity"
    
    print("✓ Test passed!")


def test_ivf_index():
    """Test IndexIVFFlat (approximate search)"""
    print("\n" + "="*70)
    print("Test 2: IndexIVFFlat (Approximate Search)")
    print("="*70)
    
    db = AIModelDatabaseFAISSAdvanced(
        index_path="test_faiss_data/ivf",
        dimension=384,
        index_type='ivf',
        nlist=100,
        nprobe=10
    )
    
    # Add 1000 models (triggers training at 100)
    print("Adding 1000 models (training at 100)...")
    for i in range(1000):
        db.create_model(
            name=f"model_{i}",
            description=f"A model for task {i}",
            architecture="ResNet" if i % 2 == 0 else "BERT"
        )
    
    # Search
    results = db.search_similar_models("ResNet classification", n_results=5)
    
    stats = db.get_statistics()
    print(f"✓ Created {stats['models']} models")
    print(f"✓ Index type: {stats['index_info']['models_index_type']}")
    print(f"✓ Index trained: {stats['index_info']['models_trained']}")
    print(f"✓ nprobe: {stats['index_info']['nprobe']}")
    print(f"✓ Top result: {results[0]['name']} (score: {results[0]['similarity_score']:.4f})")
    
    assert stats['index_info']['models_trained'], "Index should be trained"
    assert len(results) == 5, "Should return 5 results"
    
    print("✓ Test passed!")


def test_ivfpq_index():
    """Test IndexIVFPQ (compressed approximate search)"""
    print("\n" + "="*70)
    print("Test 3: IndexIVFPQ (Compressed Approximate Search)")
    print("="*70)
    
    db = AIModelDatabaseFAISSAdvanced(
        index_path="test_faiss_data/ivfpq",
        dimension=384,
        index_type='ivfpq',
        nlist=200,
        nprobe=10,
        m=48,
        nbits=8
    )
    
    # Add 2000 models
    print("Adding 2000 models (training at 100)...")
    for i in range(2000):
        db.create_model(
            name=f"model_{i}",
            description=f"Model {i}",
            architecture="Transformer"
        )
    
    # Search
    results = db.search_similar_models("Transformer model", n_results=5)
    
    stats = db.get_statistics()
    print(f"✓ Created {stats['models']} models")
    print(f"✓ Index type: {stats['index_info']['models_index_type']}")
    print(f"✓ Index trained: {stats['index_info']['models_trained']}")
    print(f"✓ Compression: m={db.m}, nbits={db.nbits}")
    
    # Calculate compression
    flat_size = stats['models'] * 384 * 4  # 4 bytes per float32
    compressed_size = stats['models'] * (384 // db.m) * (db.nbits // 8)
    compression_ratio = flat_size / compressed_size
    print(f"✓ Memory: {flat_size / 1024 / 1024:.1f} MB → {compressed_size / 1024 / 1024:.1f} MB ({compression_ratio:.1f}x compression)")
    
    assert stats['index_info']['models_trained'], "Index should be trained"
    assert len(results) == 5, "Should return 5 results"
    
    print("✓ Test passed!")


def test_auto_selection():
    """Test auto index selection"""
    print("\n" + "="*70)
    print("Test 4: Auto Index Selection")
    print("="*70)
    
    db = AIModelDatabaseFAISSAdvanced(
        index_path="test_faiss_data/auto",
        dimension=384,
        index_type='auto'
    )
    
    # Add 500 models (should use Flat for <10K)
    print("Adding 500 models (should use Flat)...")
    for i in range(500):
        db.create_model(name=f"model_{i}", description=f"Model {i}")
    
    stats = db.get_statistics()
    print(f"✓ Created {stats['models']} models")
    print(f"✓ Auto-selected index: {stats['index_info']['models_index_type']}")
    
    # Should be Flat for small dataset
    assert 'Flat' in stats['index_info']['models_index_type'], "Should use Flat for <10K vectors"
    
    print("✓ Test passed!")


def test_all_collections():
    """Test models, datasets, and predictions"""
    print("\n" + "="*70)
    print("Test 5: All Collections (Models, Datasets, Predictions)")
    print("="*70)
    
    db = AIModelDatabaseFAISSAdvanced(
        index_path="test_faiss_data/all",
        dimension=384,
        index_type='flat'
    )
    
    # Models
    model_id = db.create_model(
        name="TestModel",
        description="A test model",
        architecture="CNN"
    )
    print(f"✓ Created model {model_id}")
    
    # Datasets
    dataset_id = db.create_dataset(
        name="TestDataset",
        description="A test dataset",
        format="CSV"
    )
    print(f"✓ Created dataset {dataset_id}")
    
    # Model version
    version_id = db.create_model_version(
        model_id=model_id,
        version="1.0.0",
        checkpoint_path="/path/to/model.pth"
    )
    print(f"✓ Created model version {version_id}")
    
    # Predictions
    db.log_prediction(
        model_version_id=version_id,
        input_data={"image": "cat.jpg"},
        output_data={"class": "cat", "confidence": 0.95},
        confidence=0.95,
        inference_time_ms=50.0
    )
    print("✓ Logged prediction")
    
    # Search all collections
    model_results = db.search_similar_models("CNN test", n_results=1)
    dataset_results = db.search_similar_datasets("test CSV", n_results=1)
    pred_results = db.search_similar_predictions("cat", n_results=1)
    
    print(f"✓ Model search: {len(model_results)} results")
    print(f"✓ Dataset search: {len(dataset_results)} results")
    print(f"✓ Prediction search: {len(pred_results)} results")
    
    stats = db.get_statistics()
    print(f"\nStatistics:")
    print(f"  Models: {stats['models']}")
    print(f"  Datasets: {stats['datasets']}")
    print(f"  Predictions: {stats['predictions']}")
    
    assert stats['models'] == 1, "Should have 1 model"
    assert stats['datasets'] == 1, "Should have 1 dataset"
    assert stats['predictions'] == 1, "Should have 1 prediction"
    
    print("✓ Test passed!")


def main():
    print("\n" + "="*70)
    print("FAISS Advanced (IVF/PQ) - Quick Tests")
    print("تست‌های سریع برای FAISS پیشرفته")
    print("="*70)
    
    try:
        test_flat_index()
        test_ivf_index()
        test_ivfpq_index()
        test_auto_selection()
        test_all_collections()
        
        print("\n" + "="*70)
        print("✓ All tests passed!")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
