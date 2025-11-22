"""
Demo: Real Embeddings Integration (Sentence-Transformers) for Vector Backends

Shows upgraded semantic similarity using sentence-transformers if available.
Falls back automatically to hash-based embeddings if the model is not installed.

Environment Variables (configure in .env):
  EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
  EMBEDDING_DEVICE=auto  # cpu | cuda | auto

Run:
  python demo_real_embeddings_vector.py
"""
import os
from cad3d.super_ai.ai_db_factory import create_ai_database

QUERY_SET = [
    "image classification deep learning",
    "natural language understanding",
    "object detection bounding boxes",
    "semantic segmentation medical imaging"
]

def describe_backend(db):
    backend = getattr(db, '_embedding_backend', 'unknown')
    model = getattr(db, '_embedding_model_name', 'n/a')
    return backend, model


def load_and_populate(db, label: str):
    print(f"\n[{label}] Creating sample models...")
    samples = [
        ("ResNet50", "deep residual network for image classification", "ResNet50", "PyTorch", "image_classification"),
        ("BERT-base", "bidirectional transformer for NLP tasks", "BERT-base", "Transformers", "text_classification"),
        ("YOLOv5", "real-time object detection model with bounding boxes", "YOLOv5", "PyTorch", "object_detection"),
        ("U-Net", "segmentation model for biomedical images", "U-Net", "TensorFlow", "image_segmentation"),
    ]
    ids = []
    for name, desc, arch, fw, task in samples:
        ids.append(db.create_model(name=name, description=desc, architecture=arch, framework=fw, task_type=task))
    print(f"Inserted {len(ids)} models.")


def run_queries(db, label: str):
    print(f"\n[{label}] Semantic queries:")
    for q in QUERY_SET:
        results = db.search_similar_models(q, n_results=2)
        if not results:
            print(f"  Query '{q}' => (no results)")
            continue
        top = results[0]
        print(f"  Query '{q}' => Top: {top['name']} (similarity={top['similarity_score']:.3f})")


def main():
    print("\n🚀 Real Embeddings Demo (Sentence-Transformers)")
    print("================================================\n")
    print("Configuration:")
    print(f"  EMBEDDING_MODEL_NAME={os.getenv('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')}")
    print(f"  EMBEDDING_DEVICE={os.getenv('EMBEDDING_DEVICE', 'auto')}")

    # ChromaDB
    chroma = create_ai_database(backend='chromadb', persist_directory='chromadb_real_demo')
    load_and_populate(chroma, 'ChromaDB')
    backend_type, model_name = describe_backend(chroma)
    print(f"\nChromaDB embedding backend: {backend_type} (model={model_name})")
    run_queries(chroma, 'ChromaDB')

    # FAISS
    faiss_db = create_ai_database(backend='faiss', index_path='faiss_real_demo', dimension=384)
    load_and_populate(faiss_db, 'FAISS')
    backend_type_f, model_name_f = describe_backend(faiss_db)
    print(f"\nFAISS embedding backend: {backend_type_f} (model={model_name_f})")
    run_queries(faiss_db, 'FAISS')

    print("\n✅ Demo complete. If backend shows 'transformer' you are using real sentence-transformers embeddings.\n")

if __name__ == '__main__':
    try:
        main()
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("Install with: pip install sentence-transformers chromadb faiss-cpu")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback; traceback.print_exc()
