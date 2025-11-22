"""
Demo: AI Model Database - Vector Database (ChromaDB & FAISS)
نمایش قابلیت‌های پایگاه داده برداری

Features:
- Store model embeddings for semantic search
- Find similar models by description
- Dataset similarity search
- Prediction similarity analysis
- Both ChromaDB (persistent) and FAISS (fast)
"""
import time
from cad3d.super_ai.ai_db_factory import create_ai_database

def demo_chromadb():
    """ChromaDB Demo - Persistent vector database"""
    print("\n" + "="*70)
    print("Demo 1: ChromaDB - Persistent Vector Database")
    print("="*70 + "\n")
    
    # Create ChromaDB instance
    db = create_ai_database(backend='chromadb', persist_directory='chromadb_demo')
    
    print("✓ Connected to ChromaDB (persistent storage)\n")
    
    # Create models with semantic descriptions
    print("📦 Creating models with embeddings...\n")
    
    model1_id = db.create_model(
        name="ResNet50-ImageClassifier",
        description="Deep residual network for image classification, trained on ImageNet with 50 layers",
        architecture="ResNet50",
        framework="PyTorch",
        task_type="image_classification",
        input_shape="3x224x224",
        output_shape="1000"
    )
    print(f"   → Model 1: ResNet50 (ID: {model1_id})")
    
    model2_id = db.create_model(
        name="VGG16-ObjectDetection",
        description="Visual Geometry Group network for object detection in images with 16 layers",
        architecture="VGG16",
        framework="PyTorch",
        task_type="object_detection",
        input_shape="3x224x224",
        output_shape="80"
    )
    print(f"   → Model 2: VGG16 (ID: {model2_id})")
    
    model3_id = db.create_model(
        name="BERT-TextClassification",
        description="Bidirectional transformer model for natural language text classification tasks",
        architecture="BERT-base",
        framework="Transformers",
        task_type="text_classification",
        input_shape="512",
        output_shape="10"
    )
    print(f"   → Model 3: BERT (ID: {model3_id})")
    
    model4_id = db.create_model(
        name="GPT2-TextGeneration",
        description="Generative pre-trained transformer for creative text generation and completion",
        architecture="GPT-2",
        framework="Transformers",
        task_type="text_generation",
        input_shape="1024",
        output_shape="vocab_size"
    )
    print(f"   → Model 4: GPT-2 (ID: {model4_id})")
    
    model5_id = db.create_model(
        name="EfficientNet-ImageSegmentation",
        description="Efficient neural network for semantic image segmentation with compound scaling",
        architecture="EfficientNet-B0",
        framework="TensorFlow",
        task_type="image_segmentation",
        input_shape="3x224x224",
        output_shape="21x224x224"
    )
    print(f"   → Model 5: EfficientNet (ID: {model5_id})")
    
    # Create datasets
    print("\n📂 Creating datasets with embeddings...\n")
    
    dataset1_id = db.create_dataset(
        name="ImageNet-2012",
        description="Large-scale image classification dataset with 1000 categories of natural objects",
        source_path="/data/imagenet",
        format="JPEG",
        size_bytes=150_000_000_000,
        num_samples=1_281_167,
        split_info={"train": 1281167, "val": 50000}
    )
    print(f"   → Dataset 1: ImageNet (ID: {dataset1_id})")
    
    dataset2_id = db.create_dataset(
        name="COCO-2017",
        description="Object detection dataset with complex scenes, multiple objects, and segmentation masks",
        source_path="/data/coco",
        format="JPEG+JSON",
        size_bytes=25_000_000_000,
        num_samples=118_287,
        split_info={"train": 118287, "val": 5000}
    )
    print(f"   → Dataset 2: COCO (ID: {dataset2_id})")
    
    dataset3_id = db.create_dataset(
        name="IMDB-Reviews",
        description="Text classification dataset with movie reviews for sentiment analysis tasks",
        source_path="/data/imdb",
        format="CSV",
        size_bytes=84_000_000,
        num_samples=50_000,
        split_info={"train": 25000, "test": 25000}
    )
    print(f"   → Dataset 3: IMDB (ID: {dataset3_id})")
    
    # Semantic Search: Find similar models
    print("\n🔍 Semantic Search #1: Find models for 'image recognition'...\n")
    
    similar_models = db.search_similar_models("image recognition deep learning", n_results=3)
    
    for i, model in enumerate(similar_models, 1):
        print(f"   {i}. {model['name']} ({model['architecture']})")
        print(f"      Similarity: {model['similarity_score']:.3f}")
        print(f"      Description: {model['description'][:80]}...")
        print()
    
    # Semantic Search: Find models for NLP
    print("🔍 Semantic Search #2: Find models for 'natural language processing'...\n")
    
    similar_models = db.search_similar_models("natural language processing NLP text", n_results=3)
    
    for i, model in enumerate(similar_models, 1):
        print(f"   {i}. {model['name']} ({model['architecture']})")
        print(f"      Similarity: {model['similarity_score']:.3f}")
        print(f"      Task: {similar_models[i-1].get('description', '')[:60]}...")
        print()
    
    # Semantic Search: Find similar datasets
    print("🔍 Semantic Search #3: Find datasets for 'object detection images'...\n")
    
    similar_datasets = db.search_similar_datasets("object detection multiple objects images", n_results=2)
    
    for i, dataset in enumerate(similar_datasets, 1):
        print(f"   {i}. {dataset['name']} ({dataset['format']})")
        print(f"      Similarity: {dataset['similarity_score']:.3f}")
        print(f"      Description: {dataset['description'][:80]}...")
        print()
    
    # Create training run and log predictions
    print("🏋️ Creating training run and logging predictions...\n")
    
    version_id = db.create_model_version(
        model_id=model1_id,
        version="1.0.0",
        checkpoint_path="/models/resnet50_v1.pth",
        config={"learning_rate": 0.001, "batch_size": 32}
    )
    
    run_id = db.create_training_run(
        model_version_id=version_id,
        dataset_id=dataset1_id,
        run_name="ResNet50-ImageNet-Training",
        status="completed"
    )
    
    # Log some predictions with embeddings
    predictions = [
        {"image_id": 1, "predicted_class": "golden_retriever", "confidence": 0.95},
        {"image_id": 2, "predicted_class": "persian_cat", "confidence": 0.89},
        {"image_id": 3, "predicted_class": "sports_car", "confidence": 0.92},
        {"image_id": 4, "predicted_class": "mountain_bike", "confidence": 0.88},
        {"image_id": 5, "predicted_class": "beach_scene", "confidence": 0.91}
    ]
    
    for pred in predictions:
        db.log_prediction(
            model_version_id=version_id,
            input_data={"image_id": pred["image_id"]},
            output_data={"predicted_class": pred["predicted_class"]},
            confidence=pred["confidence"],
            inference_time_ms=12.5
        )
    
    print(f"   ✓ Logged {len(predictions)} predictions with embeddings\n")
    
    # Search similar predictions
    print("🔍 Semantic Search #4: Find similar predictions to 'dog pet animal'...\n")
    
    similar_predictions = db.search_similar_predictions("dog pet animal", n_results=3)
    
    for i, pred in enumerate(similar_predictions, 1):
        print(f"   {i}. Predicted: {pred['output']['predicted_class']}")
        print(f"      Confidence: {pred['confidence']:.2f}")
        print(f"      Similarity: {pred['similarity_score']:.3f}")
        print()
    
    # Statistics
    print("📊 Database Statistics:\n")
    stats = db.get_statistics()
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print("\n✓ ChromaDB Demo Complete!")
    print("   Data persisted to: chromadb_demo/\n")


def demo_faiss():
    """FAISS Demo - Fast similarity search"""
    print("\n" + "="*70)
    print("Demo 2: FAISS - Fast Similarity Search")
    print("="*70 + "\n")
    
    # Create FAISS instance
    db = create_ai_database(backend='faiss', index_path='faiss_demo', dimension=384)
    
    print("✓ Connected to FAISS (in-memory + disk persistence)\n")
    
    # Create models (same as ChromaDB for comparison)
    print("📦 Creating models with FAISS indexing...\n")
    
    model_ids = []
    
    model_ids.append(db.create_model(
        name="YOLO-v5-ObjectDetection",
        description="You Only Look Once real-time object detection system with bounding boxes",
        architecture="YOLO-v5",
        framework="PyTorch",
        task_type="object_detection"
    ))
    print(f"   → Model 1: YOLO-v5 (ID: {model_ids[-1]})")
    
    model_ids.append(db.create_model(
        name="U-Net-MedicalSegmentation",
        description="Convolutional network for biomedical image segmentation in medical imaging",
        architecture="U-Net",
        framework="TensorFlow",
        task_type="medical_segmentation"
    ))
    print(f"   → Model 2: U-Net (ID: {model_ids[-1]})")
    
    model_ids.append(db.create_model(
        name="T5-TextSummarization",
        description="Text-to-Text Transfer Transformer for document summarization and generation",
        architecture="T5-base",
        framework="Transformers",
        task_type="text_summarization"
    ))
    print(f"   → Model 3: T5 (ID: {model_ids[-1]})")
    
    model_ids.append(db.create_model(
        name="Faster-RCNN-Detection",
        description="Region-based Convolutional Neural Network for accurate object detection",
        architecture="Faster-RCNN",
        framework="PyTorch",
        task_type="object_detection"
    ))
    print(f"   → Model 4: Faster-RCNN (ID: {model_ids[-1]})")
    
    # Performance test: FAISS similarity search speed
    print("\n⚡ Performance Test: Search speed...\n")
    
    queries = [
        "real-time object detection",
        "medical image analysis",
        "text summarization NLP",
        "accurate detection bounding boxes"
    ]
    
    total_time = 0
    for query in queries:
        start = time.perf_counter()
        results = db.search_similar_models(query, n_results=3)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        total_time += elapsed
        
        print(f"   Query: '{query}'")
        print(f"   Time: {elapsed:.2f} ms")
        if results:
            print(f"   Top Result: {results[0]['name']} (similarity: {results[0]['similarity_score']:.3f})")
        print()
    
    avg_time = total_time / len(queries)
    print(f"   Average search time: {avg_time:.2f} ms")
    print(f"   Throughput: ~{1000/avg_time:.0f} queries/second\n")
    
    # List all models
    print("📋 All Models in FAISS:\n")
    all_models = db.list_models()
    for model in all_models:
        print(f"   - {model['name']} ({model['architecture']})")
    
    # Statistics
    print("\n📊 Database Statistics:\n")
    stats = db.get_statistics()
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print("\n✓ FAISS Demo Complete!")
    print("   Indexes saved to: faiss_demo/\n")


def demo_comparison():
    """Compare ChromaDB vs FAISS"""
    print("\n" + "="*70)
    print("Demo 3: ChromaDB vs FAISS Comparison")
    print("="*70 + "\n")
    
    print("📊 Backend Comparison:\n")
    
    comparison = """
    ┌─────────────────┬──────────────────┬──────────────────┐
    │ Feature         │ ChromaDB         │ FAISS            │
    ├─────────────────┼──────────────────┼──────────────────┤
    │ Storage         │ Persistent       │ Persistent       │
    │ Speed           │ Good             │ Excellent        │
    │ Scalability     │ Excellent        │ Good             │
    │ Memory Usage    │ Low              │ Medium           │
    │ Features        │ Rich metadata    │ Fast search      │
    │ Best For        │ Production       │ Research/Speed   │
    │ Dependencies    │ chromadb         │ faiss-cpu/gpu    │
    └─────────────────┴──────────────────┴──────────────────┘
    """
    print(comparison)
    
    print("\n💡 Use Cases:\n")
    print("   ChromaDB:")
    print("   • Production RAG systems")
    print("   • Rich metadata queries")
    print("   • Multi-user applications")
    print("   • Long-term persistence\n")
    
    print("   FAISS:")
    print("   • High-speed similarity search")
    print("   • Large-scale vector search (millions of vectors)")
    print("   • Research and experimentation")
    print("   • GPU acceleration support\n")
    
    print("\n✓ Comparison Complete!\n")


if __name__ == '__main__':
    print("\n🚀 AI Model Database - Vector Database Demo")
    print("=" * 70)
    
    try:
        # Demo 1: ChromaDB
        demo_chromadb()
        
        # Demo 2: FAISS
        demo_faiss()
        
        # Demo 3: Comparison
        demo_comparison()
        
        print("\n" + "="*70)
        print("✅ All Demos Completed Successfully!")
        print("="*70 + "\n")
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\nInstall required packages:")
        print("   pip install chromadb faiss-cpu")
        print("\nOr for GPU support:")
        print("   pip install chromadb faiss-gpu\n")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
