"""
Embeddings module for semantic similarity and vector operations.
Provides embedding generation and vector store management for semantic retrieval.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate dense vector embeddings for text passages."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Sentence Transformer model to use (default: lightweight, fast model)
        """
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"EmbeddingGenerator initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.model = None
    
    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for a single text passage.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        if not self.model or not text:
            return None
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not self.model:
            return [None] * len(texts)
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return [embeddings[i] for i in range(len(texts))]
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            return [None] * len(texts)
    
    def get_embedding_dim(self) -> int:
        """Get dimensionality of embeddings."""
        if not self.model:
            return 0
        return self.model.get_sentence_embedding_dimension()


class VectorStore:
    """Manage and search vector embeddings for documents."""
    
    def __init__(self, embedding_generator: Optional[EmbeddingGenerator] = None):
        """
        Initialize vector store.
        
        Args:
            embedding_generator: EmbeddingGenerator instance
        """
        self.embedding_gen = embedding_generator or EmbeddingGenerator()
        self.vectors: List[np.ndarray] = []
        self.documents: List[Dict] = []
        self.index: Dict[int, int] = {}  # doc_id -> index mapping
        logger.info("VectorStore initialized")
    
    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict] = None) -> bool:
        """
        Add a document to the vector store.
        
        Args:
            doc_id: Unique document identifier
            text: Document text
            metadata: Optional metadata dict
            
        Returns:
            True if successful
        """
        embedding = self.embedding_gen.embed_text(text)
        if embedding is None:
            logger.warning(f"Failed to embed document {doc_id}")
            return False
        
        idx = len(self.documents)
        self.documents.append({
            'id': doc_id,
            'text': text,
            'metadata': metadata or {}
        })
        self.vectors.append(embedding)
        self.index[int(doc_id.split('_')[-1]) if '_' in str(doc_id) else hash(doc_id) % 10000] = idx
        
        logger.debug(f"Added document {doc_id} to vector store")
        return True
    
    def add_documents_batch(self, documents: List[Dict]) -> int:
        """
        Add multiple documents to vector store.
        
        Args:
            documents: List of dicts with 'id', 'text', 'metadata' keys
            
        Returns:
            Number of successfully added documents
        """
        texts = [doc.get('text', '') for doc in documents]
        embeddings = self.embedding_gen.embed_batch(texts)
        
        added_count = 0
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            if embedding is not None:
                idx = len(self.documents)
                self.documents.append(doc)
                self.vectors.append(embedding)
                added_count += 1
        
        logger.info(f"Added {added_count}/{len(documents)} documents to vector store")
        return added_count
    
    def search(self, query: str, top_k: int = 5, threshold: float = 0.3) -> List[Dict]:
        """
        Search for similar documents using semantic similarity.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            threshold: Minimum similarity score
            
        Returns:
            List of results with doc info and similarity scores
        """
        if not self.vectors:
            return []
        
        query_embedding = self.embedding_gen.embed_text(query)
        if query_embedding is None:
            return []
        
        # Compute cosine similarity
        similarities = cosine_similarity([query_embedding], self.vectors)[0]
        
        # Get top-k results above threshold
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                doc = self.documents[idx]
                results.append({
                    'id': doc['id'],
                    'text': doc['text'],
                    'similarity': score,
                    'metadata': doc.get('metadata', {})
                })
        
        logger.debug(f"Search for '{query[:50]}...' returned {len(results)} results")
        return results
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get document by ID."""
        for doc in self.documents:
            if doc['id'] == doc_id:
                return doc
        return None
    
    def clear(self):
        """Clear all documents from store."""
        self.vectors.clear()
        self.documents.clear()
        self.index.clear()
        logger.info("Vector store cleared")
    
    def size(self) -> int:
        """Get number of documents in store."""
        return len(self.documents)


class SemanticSimilarityScorer:
    """Score semantic similarity between text passages."""
    
    def __init__(self, embedding_generator: Optional[EmbeddingGenerator] = None):
        """Initialize scorer."""
        self.embedding_gen = embedding_generator or EmbeddingGenerator()
    
    def score_pair(self, text1: str, text2: str) -> float:
        """
        Score similarity between two text passages (0-1).
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        embed1 = self.embedding_gen.embed_text(text1)
        embed2 = self.embedding_gen.embed_text(text2)
        
        if embed1 is None or embed2 is None:
            return 0.0
        
        # Cosine similarity
        similarity = cosine_similarity([embed1], [embed2])[0][0]
        return max(0.0, min(1.0, float(similarity)))  # Clamp to [0, 1]
    
    def score_batch(self, text1: str, texts2: List[str]) -> List[float]:
        """
        Score similarity between one text and multiple texts.
        
        Args:
            text1: Reference text
            texts2: List of texts to compare
            
        Returns:
            List of similarity scores
        """
        embed1 = self.embedding_gen.embed_text(text1)
        if embed1 is None:
            return [0.0] * len(texts2)
        
        embeddings2 = self.embedding_gen.embed_batch(texts2)
        scores = []
        
        for embed2 in embeddings2:
            if embed2 is not None:
                score = cosine_similarity([embed1], [embed2])[0][0]
                scores.append(float(score))
            else:
                scores.append(0.0)
        
        return scores
