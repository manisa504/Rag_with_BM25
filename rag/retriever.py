"""
Hybrid retrieval combining vector search with BM25 re-ranking.
"""

from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank-bm25 is not installed. Please install it with: pip install rank-bm25")

from .vectorstore.qdrant_store import QdrantStore, SearchResult
from .embeddings.base import BaseEmbeddingProvider
from .settings import settings

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """Represents a hybrid search result with both vector and BM25 scores."""
    id: str
    text: str
    metadata: Dict[str, Any]
    vector_score: float
    bm25_score: float
    final_score: float


class HybridRetriever:
    """Hybrid retriever combining vector search with BM25 re-ranking."""
    
    def __init__(
        self,
        vector_store: QdrantStore,
        embedding_provider: BaseEmbeddingProvider,
        top_k: int = None,
        rerank_k: int = None
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Qdrant vector store instance
            embedding_provider: Embedding provider for query encoding
            top_k: Number of candidates to retrieve from vector search
            rerank_k: Number of final results after re-ranking
        """
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.top_k = top_k or settings.top_k
        self.rerank_k = rerank_k or settings.rerank_k
        
        # BM25 will be built on-demand for retrieved candidates
        self._bm25_index = None
        self._candidate_texts = []
        
        logger.info(f"Initialized hybrid retriever with top_k={self.top_k}, rerank_k={self.rerank_k}")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        rerank_k: Optional[int] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[HybridResult]:
        """
        Perform hybrid retrieval with vector search followed by BM25 re-ranking.
        
        Args:
            query: Search query
            top_k: Override default top_k for vector search
            rerank_k: Override default rerank_k for final results
            filter_conditions: Optional filters for vector search
            
        Returns:
            List of hybrid search results
        """
        effective_top_k = top_k or self.top_k
        effective_rerank_k = rerank_k or self.rerank_k
        
        logger.info(f"Starting hybrid retrieval for query: '{query[:50]}...'")
        
        # Step 1: Vector search to get candidates
        vector_results = self._vector_search(query, effective_top_k, filter_conditions)
        
        if not vector_results:
            logger.warning("No vector search results found")
            return []
        
        # Step 2: BM25 re-ranking on candidates
        hybrid_results = self._bm25_rerank(query, vector_results, effective_rerank_k)
        
        logger.info(f"Hybrid retrieval completed. Final results: {len(hybrid_results)}")
        return hybrid_results
    
    def _vector_search(
        self,
        query: str,
        top_k: int,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform vector search using the embedding provider and vector store."""
        try:
            # Encode query
            query_vector = self.embedding_provider.embed_query(query)
            
            # Search vector store
            results = self.vector_store.search(
                query_vector=query_vector,
                top_k=top_k,
                filter_conditions=filter_conditions
            )
            
            logger.info(f"Vector search returned {len(results)} candidates")
            return results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            raise
    
    def _bm25_rerank(
        self,
        query: str,
        vector_results: List[SearchResult],
        rerank_k: int
    ) -> List[HybridResult]:
        """Re-rank vector search results using BM25."""
        if not vector_results:
            return []
        
        try:
            # Prepare texts for BM25
            candidate_texts = [result.text for result in vector_results]
            
            # Tokenize texts for BM25 (simple split by space)
            tokenized_texts = [text.lower().split() for text in candidate_texts]
            
            # Build BM25 index
            bm25 = BM25Okapi(tokenized_texts)
            
            # Tokenize query
            tokenized_query = query.lower().split()
            
            # Get BM25 scores
            bm25_scores = bm25.get_scores(tokenized_query)
            
            # Combine scores and create hybrid results
            hybrid_results = []
            for i, (vector_result, bm25_score) in enumerate(zip(vector_results, bm25_scores)):
                # Normalize BM25 score (simple min-max normalization)
                # Note: BM25 scores are typically non-negative
                normalized_bm25 = max(0.0, min(1.0, bm25_score / (max(bm25_scores) + 1e-8)))
                
                # Calculate final score (weighted combination)
                # Vector score is already normalized (cosine similarity)
                # Weight: 0.7 for BM25, 0.3 for vector similarity
                final_score = 0.7 * normalized_bm25 + 0.3 * vector_result.score
                
                hybrid_result = HybridResult(
                    id=vector_result.id,
                    text=vector_result.text,
                    metadata=vector_result.metadata,
                    vector_score=vector_result.score,
                    bm25_score=bm25_score,
                    final_score=final_score
                )
                hybrid_results.append(hybrid_result)
            
            # Sort by final score (descending)
            hybrid_results.sort(key=lambda x: x.final_score, reverse=True)
            
            # Return top rerank_k results
            final_results = hybrid_results[:rerank_k]
            
            logger.info(f"BM25 re-ranking completed. Top {len(final_results)} results selected")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in BM25 re-ranking: {e}")
            # Fallback: return vector results as hybrid results
            return self._fallback_to_vector_results(vector_results, rerank_k)
    
    def _fallback_to_vector_results(
        self,
        vector_results: List[SearchResult],
        rerank_k: int
    ) -> List[HybridResult]:
        """Fallback to vector results if BM25 re-ranking fails."""
        logger.warning("Falling back to vector-only results")
        
        hybrid_results = []
        for result in vector_results[:rerank_k]:
            hybrid_result = HybridResult(
                id=result.id,
                text=result.text,
                metadata=result.metadata,
                vector_score=result.score,
                bm25_score=0.0,  # No BM25 score available
                final_score=result.score  # Use vector score as final
            )
            hybrid_results.append(hybrid_result)
        
        return hybrid_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        vector_store_info = self.vector_store.get_collection_info()
        
        return {
            "vector_store": vector_store_info,
            "embedding_model": self.embedding_provider.model_name,
            "embedding_dimension": self.embedding_provider.dimension,
            "top_k": self.top_k,
            "rerank_k": self.rerank_k,
        }
