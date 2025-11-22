"""
RAG (Retrieval-Augmented Generation) System for KURDO-AI
Fourth complementary method alongside Fine-Tuning, LoRA, and Prompt Engineering

RAG dynamically retrieves relevant information from knowledge bases and documents
to augment AI responses with accurate, up-to-date context.

Features:
- Document indexing and vectorization
- Semantic search with embeddings
- Multiple retrieval strategies
- Integration with Fine-Tuning/LoRA/Prompt Engineering
- Architectural knowledge base management
"""

import logging
import os
import json
import pickle
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss not available. Install with: pip install faiss-cpu")


class Document:
    """A document in the knowledge base."""
    
    def __init__(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any] = None,
        embedding: np.ndarray = None
    ):
        self.doc_id = doc_id
        self.content = content
        self.metadata = metadata or {}
        self.embedding = embedding
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "has_embedding": self.embedding is not None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Document':
        """Create from dictionary."""
        return cls(
            doc_id=data["doc_id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
        )


class VectorStore:
    """Vector store for semantic search."""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.documents: List[Document] = []
        self.index = None
        
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(embedding_dim)
            logger.info(f"FAISS index initialized with dimension {embedding_dim}")
        else:
            logger.warning("FAISS not available. Using simple similarity search.")
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store."""
        new_embeddings = []
        
        for doc in documents:
            if doc.embedding is not None:
                self.documents.append(doc)
                new_embeddings.append(doc.embedding)
        
        if new_embeddings and FAISS_AVAILABLE and self.index is not None:
            embeddings_array = np.array(new_embeddings).astype('float32')
            self.index.add(embeddings_array)
            logger.info(f"Added {len(new_embeddings)} documents to vector store")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents."""
        if not self.documents:
            return []
        
        if FAISS_AVAILABLE and self.index is not None:
            # FAISS search
            query_array = np.array([query_embedding]).astype('float32')
            distances, indices = self.index.search(query_array, min(top_k, len(self.documents)))
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    similarity = 1 / (1 + distance)  # Convert distance to similarity
                    results.append((doc, float(similarity)))
            
            return results
        else:
            # Simple cosine similarity
            results = []
            for doc in self.documents:
                if doc.embedding is not None:
                    similarity = self._cosine_similarity(query_embedding, doc.embedding)
                    results.append((doc, similarity))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def save(self, filepath: str):
        """Save vector store to disk."""
        data = {
            "embedding_dim": self.embedding_dim,
            "documents": [doc.to_dict() for doc in self.documents]
        }
        
        # Save metadata
        with open(filepath + ".json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Save embeddings
        if self.documents:
            embeddings = np.array([doc.embedding for doc in self.documents if doc.embedding is not None])
            np.save(filepath + "_embeddings.npy", embeddings)
        
        # Save FAISS index
        if FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, filepath + ".faiss")
        
        logger.info(f"Vector store saved to {filepath}")
    
    def load(self, filepath: str):
        """Load vector store from disk."""
        # Load metadata
        with open(filepath + ".json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.embedding_dim = data["embedding_dim"]
        
        # Load embeddings
        if os.path.exists(filepath + "_embeddings.npy"):
            embeddings = np.load(filepath + "_embeddings.npy")
            
            # Reconstruct documents
            self.documents = []
            for i, doc_data in enumerate(data["documents"]):
                doc = Document.from_dict(doc_data)
                if i < len(embeddings):
                    doc.embedding = embeddings[i]
                self.documents.append(doc)
        
        # Load FAISS index
        if FAISS_AVAILABLE and os.path.exists(filepath + ".faiss"):
            self.index = faiss.read_index(filepath + ".faiss")
        
        logger.info(f"Vector store loaded from {filepath}")


class RAGSystem:
    """
    Retrieval-Augmented Generation System.
    
    Combines document retrieval with generation for accurate, context-aware responses.
    """
    
    def __init__(
        self,
        storage_dir: str = "models/rag",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.vector_store = None
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                self.vector_store = VectorStore(embedding_dim=embedding_dim)
                logger.info(f"RAG System initialized with {embedding_model}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self.embedding_model = None
        
        # Statistics
        self.retrieval_stats = {
            "total_retrievals": 0,
            "avg_retrieval_time": 0.0,
            "documents_indexed": 0
        }
        
        # Load architectural knowledge base
        self._initialize_architectural_knowledge()
    
    def _initialize_architectural_knowledge(self):
        """Initialize with built-in architectural knowledge."""
        architectural_docs = [
            {
                "doc_id": "arch_001",
                "content": "محاسبه مساحت اتاق: مساحت = طول × عرض. برای اتاق با ابعاد 5 متر در 4 متر، مساحت = 5 × 4 = 20 متر مربع. برای اتاق‌های مسکونی، مساحت استاندارد اتاق خواب بین 12 تا 20 متر مربع است.",
                "metadata": {"category": "calculations", "topic": "area", "language": "fa"}
            },
            {
                "doc_id": "arch_002",
                "content": "محاسبه حجم اتاق: حجم = طول × عرض × ارتفاع. برای اتاق 6×4×2.8 متر، حجم = 67.2 متر مکعب. این حجم نیاز به تهویه مناسب دارد (حداقل 2 تعویض هوا در ساعت).",
                "metadata": {"category": "calculations", "topic": "volume", "language": "fa"}
            },
            {
                "doc_id": "arch_003",
                "content": "تخمین آجر: برای هر متر مربع دیوار، حدود 60 آجر استاندارد نیاز است. برای دیوار 10 متری با ارتفاع 3 متر: مساحت = 30 متر مربع، آجر مورد نیاز = 1800 عدد. همیشه 10% ضریب اتلاف اضافه کنید.",
                "metadata": {"category": "materials", "topic": "brick", "language": "fa"}
            },
            {
                "doc_id": "arch_004",
                "content": "استاندارد ارتفاع سقف: طبق مبحث 19 مقررات ملی ساختمان ایران، حداقل ارتفاع سقف اتاق‌های اصلی 2.4 متر، راهروها 2.1 متر، و سرویس‌های بهداشتی 2.1 متر است. ارتفاع پیشنهادی برای احساس فضای بهتر: 2.6-2.8 متر.",
                "metadata": {"category": "codes", "topic": "ceiling_height", "language": "fa", "standard": "مبحث 19"}
            },
            {
                "doc_id": "arch_005",
                "content": "طراحی پی ساختمان: عمق پی باید حداقل 1.5 متر زیر تراز یخبندان باشد. برای ساختمان 3 طبقه در تهران: عمق پی حداقل 1.5-2 متر، پهنای پی حداقل 80 سانتی‌متر. حتماً آزمایش خاک انجام شود.",
                "metadata": {"category": "foundation", "topic": "depth", "language": "fa"}
            },
            {
                "doc_id": "arch_006",
                "content": "محاسبه پله: رابطه بلون برای طراحی پله: 2h + d = 63 سانتی‌متر، که h ارتفاع پله (17-18 سانتی‌متر) و d عرض پله (28-30 سانتی‌متر) است. عرض راه پله حداقل 90 سانتی‌متر، ارتفاع نرده 90-100 سانتی‌متر.",
                "metadata": {"category": "design", "topic": "stairs", "language": "fa"}
            },
            {
                "doc_id": "arch_007",
                "content": "Room area calculation: Area = length × width. For a 5m × 4m room, area = 20 square meters. Standard bedroom size: 12-20 sqm. Master bedroom: 15-25 sqm.",
                "metadata": {"category": "calculations", "topic": "area", "language": "en"}
            },
            {
                "doc_id": "arch_008",
                "content": "مقاومت در برابر زلزله: استفاده از دیوارهای برشی، اتصالات قوی، توزیع متقارن جرم، اسکلت بتنی یا فلزی مناسب، رعایت استاندارد 2800 ایران. برای ساختمان‌های بالای 5 طبقه، مشاور سازه الزامی است.",
                "metadata": {"category": "structural", "topic": "earthquake", "language": "fa", "standard": "2800"}
            },
            {
                "doc_id": "arch_009",
                "content": "نسبت پنجره به کف: حداقل 1/8 (12.5٪) مساحت کف. برای اتاق 20 متری، حداقل 2.5 متر مربع پنجره. پیشنهادی: 1/6 تا 1/5 (16-20٪) برای نور کافی.",
                "metadata": {"category": "design", "topic": "window", "language": "fa"}
            },
            {
                "doc_id": "arch_010",
                "content": "شیب لوله فاضلاب: لوله‌های 50-100 میلی‌متر: شیب 2-3٪ (2-3 سانتی‌متر در متر). لوله‌های بزرگ‌تر: 1-2٪. حداقل مطلق: 1٪. برای لوله 5 متری، حداقل 5 سانتی‌متر اختلاف ارتفاع.",
                "metadata": {"category": "plumbing", "topic": "drainage", "language": "fa"}
            },
            {
                "doc_id": "arch_011",
                "content": "ابعاد پارکینگ: عمودی 2.5×5 متر، موازی 2×6 متر، جانبی 45 درجه 2.5×5.5 متر. عرض راهرو دسترسی حداقل 6 متر. ارتفاع سقف پارکینگ حداقل 2.2 متر.",
                "metadata": {"category": "parking", "topic": "dimensions", "language": "fa"}
            },
            {
                "doc_id": "arch_012",
                "content": "بهینه‌سازی انرژی: عایق‌کاری دیوار (5 سانتی‌متر)، پنجره‌های دوجداره (کاهش 30-40٪ اتلاف)، عایق سقف و کف، پنل‌های خورشیدی، جهت‌گیری مناسب ساختمان به سمت جنوب.",
                "metadata": {"category": "energy", "topic": "efficiency", "language": "fa"}
            }
        ]
        
        # Add documents
        for doc_data in architectural_docs:
            self.add_document(
                content=doc_data["content"],
                doc_id=doc_data["doc_id"],
                metadata=doc_data["metadata"]
            )
    
    def add_document(
        self,
        content: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Document:
        """Add a document to the knowledge base."""
        if not self.embedding_model:
            logger.error("Embedding model not available")
            return None
        
        if doc_id is None:
            doc_id = f"doc_{len(self.vector_store.documents) + 1:04d}"
        
        # Generate embedding
        embedding = self.embedding_model.encode(content)
        
        # Create document
        doc = Document(
            doc_id=doc_id,
            content=content,
            metadata=metadata or {},
            embedding=embedding
        )
        
        # Add to vector store
        if self.vector_store:
            self.vector_store.add_documents([doc])
            self.retrieval_stats["documents_indexed"] += 1
        
        logger.info(f"Document added: {doc_id}")
        return doc
    
    def add_documents_from_file(self, filepath: str) -> int:
        """Add documents from a text file (one document per line)."""
        added = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    self.add_document(
                        content=line,
                        doc_id=f"file_{Path(filepath).stem}_{i:04d}",
                        metadata={"source": filepath, "line": i}
                    )
                    added += 1
        
        logger.info(f"Added {added} documents from {filepath}")
        return added
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        filter_metadata: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            filter_metadata: Filter by metadata (e.g., {"category": "calculations"})
        
        Returns:
            List of (document, similarity_score) tuples
        """
        if not self.embedding_model or not self.vector_store:
            logger.error("RAG system not properly initialized")
            return []
        
        import time
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Search
        results = self.vector_store.search(query_embedding, top_k=top_k * 2)  # Get more for filtering
        
        # Filter by metadata if specified
        if filter_metadata:
            filtered_results = []
            for doc, score in results:
                match = all(
                    doc.metadata.get(key) == value
                    for key, value in filter_metadata.items()
                )
                if match:
                    filtered_results.append((doc, score))
            results = filtered_results[:top_k]
        else:
            results = results[:top_k]
        
        # Update stats
        retrieval_time = time.time() - start_time
        self.retrieval_stats["total_retrievals"] += 1
        
        # Update average retrieval time
        n = self.retrieval_stats["total_retrievals"]
        old_avg = self.retrieval_stats["avg_retrieval_time"]
        self.retrieval_stats["avg_retrieval_time"] = (old_avg * (n - 1) + retrieval_time) / n
        
        logger.info(f"Retrieved {len(results)} documents in {retrieval_time:.3f}s")
        return results
    
    def generate_rag_prompt(
        self,
        query: str,
        top_k: int = 3,
        instruction: str = None
    ) -> str:
        """
        Generate a RAG-enhanced prompt.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            instruction: Optional instruction for the model
        
        Returns:
            Prompt with retrieved context
        """
        # Retrieve relevant documents
        results = self.retrieve(query, top_k=top_k)
        
        # Build context from retrieved documents
        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            context_parts.append(f"[Context {i}] (Relevance: {score:.2f})\n{doc.content}")
        
        context = "\n\n".join(context_parts)
        
        # Build prompt
        if instruction is None:
            instruction = "You are KURDO-AI, an expert architectural assistant. Use the provided context to answer accurately."
        
        prompt = f"""{instruction}

# Retrieved Context

{context}

# User Query

{query}

# Your Response

Provide a detailed, accurate answer based on the context above. If the context doesn't contain enough information, acknowledge it and provide general guidance."""
        
        return prompt
    
    def generate_rag_response(
        self,
        query: str,
        top_k: int = 3,
        generation_method: str = "prompt_engineering"
    ) -> Dict[str, Any]:
        """
        Generate a complete RAG response.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            generation_method: "prompt_engineering", "lora", or "fine_tuning"
        
        Returns:
            Response with retrieved context and generated answer
        """
        # Retrieve documents
        results = self.retrieve(query, top_k=top_k)
        
        # Generate prompt
        prompt = self.generate_rag_prompt(query, top_k=top_k)
        
        # Prepare response
        response = {
            "query": query,
            "retrieved_documents": [
                {
                    "doc_id": doc.doc_id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "relevance_score": float(score)
                }
                for doc, score in results
            ],
            "prompt": prompt,
            "generation_method": generation_method,
            "num_documents_retrieved": len(results)
        }
        
        return response
    
    def hybrid_rag_retrieve(
        self,
        query: str,
        top_k: int = 3,
        use_keywords: bool = True
    ) -> List[Tuple[Document, float]]:
        """
        Hybrid retrieval: semantic + keyword-based.
        
        Args:
            query: Search query
            top_k: Number of results
            use_keywords: Also use keyword matching
        
        Returns:
            List of (document, score) tuples
        """
        # Semantic retrieval
        semantic_results = self.retrieve(query, top_k=top_k * 2)
        
        if not use_keywords:
            return semantic_results[:top_k]
        
        # Keyword matching
        query_words = set(query.lower().split())
        keyword_scores = {}
        
        for doc, semantic_score in semantic_results:
            doc_words = set(doc.content.lower().split())
            overlap = len(query_words & doc_words)
            keyword_score = overlap / max(len(query_words), 1)
            
            # Combine scores
            combined_score = 0.7 * semantic_score + 0.3 * keyword_score
            keyword_scores[doc.doc_id] = (doc, combined_score)
        
        # Sort by combined score
        results = list(keyword_scores.values())
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def save_knowledge_base(self, name: str = "default"):
        """Save the knowledge base to disk."""
        filepath = os.path.join(self.storage_dir, name)
        
        if self.vector_store:
            self.vector_store.save(filepath)
        
        # Save stats
        stats_file = filepath + "_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.retrieval_stats, f, indent=2)
        
        logger.info(f"Knowledge base saved: {name}")
    
    def load_knowledge_base(self, name: str = "default"):
        """Load a knowledge base from disk."""
        filepath = os.path.join(self.storage_dir, name)
        
        if self.vector_store:
            self.vector_store.load(filepath)
        
        # Load stats
        stats_file = filepath + "_stats.json"
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                self.retrieval_stats = json.load(f)
        
        logger.info(f"Knowledge base loaded: {name}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        return {
            "documents_indexed": self.retrieval_stats["documents_indexed"],
            "total_retrievals": self.retrieval_stats["total_retrievals"],
            "avg_retrieval_time_ms": self.retrieval_stats["avg_retrieval_time"] * 1000,
            "embedding_model": self.embedding_model_name,
            "vector_store_size": len(self.vector_store.documents) if self.vector_store else 0,
            "faiss_available": FAISS_AVAILABLE,
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE
        }
    
    def compare_with_other_methods(self) -> Dict[str, Any]:
        """Compare RAG with other training methods."""
        return {
            "rag": {
                "type": "Retrieval-Augmented Generation",
                "setup_time": "Minutes (indexing)",
                "cost": "$0 (local)",
                "gpu_required": False,
                "quality": "Excellent (for knowledge-based tasks)",
                "flexibility": "Very High",
                "best_for": [
                    "Large knowledge bases",
                    "Frequently updated information",
                    "Fact-based queries",
                    "Domain-specific knowledge",
                    "Multi-document reasoning"
                ],
                "limitations": [
                    "Requires good retrieval",
                    "Context window limits",
                    "Depends on document quality",
                    "May retrieve irrelevant docs"
                ],
                "advantages_over_fine_tuning": [
                    "No training needed",
                    "Easily updatable (add new docs)",
                    "Transparent (shows sources)",
                    "Lower cost",
                    "No model drift"
                ],
                "combines_well_with": [
                    "Prompt Engineering (RAG + Few-shot)",
                    "LoRA (Fine-tuned retrieval)",
                    "Fine-Tuning (RAG + Specialized model)"
                ]
            }
        }


# Global instance
rag_system = RAGSystem()
