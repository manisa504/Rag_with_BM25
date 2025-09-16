"""
Google embeddings provider using text-embedding-004.
"""

from typing import List
import logging
import time
from ..settings import settings

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError("google-generativeai is not installed. Please install it with: pip install google-generativeai")

from .base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class GoogleEmbeddingProvider(BaseEmbeddingProvider):
    """Google embeddings provider using text-embedding-004."""
    
    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Initialize Google embedding provider.
        
        Args:
            api_key: Google API key (defaults to settings)
            model_name: Model name (defaults to settings)
        """
        self.api_key = api_key or settings.google_api_key
        self.model_name_str = model_name or settings.google_embed_model
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        
        # Validate model
        self._validate_model()
        
        logger.info(f"Initialized Google embeddings with model: {self.model_name_str}")
    
    def _validate_model(self) -> None:
        """Validate that the model is available."""
        try:
            # Test with a small embedding
            test_result = genai.embed_content(
                model=f"models/{self.model_name_str}",
                content="test",
                task_type="retrieval_document"
            )
            self._dimension = len(test_result['embedding'])
            logger.info(f"Model validation successful. Dimension: {self._dimension}")
        except Exception as e:
            logger.error(f"Failed to validate Google embedding model: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        embeddings = []
        batch_size = 100  # Google API limit
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._embed_batch(batch)
            embeddings.extend(batch_embeddings)
            
            # Rate limiting
            if i + batch_size < len(texts):
                time.sleep(0.1)
        
        logger.info(f"Embedded {len(texts)} texts")
        return embeddings
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        try:
            # Handle single text vs batch
            if len(texts) == 1:
                result = genai.embed_content(
                    model=f"models/{self.model_name_str}",
                    content=texts[0],
                    task_type="retrieval_document"
                )
                return [result['embedding']]
            else:
                result = genai.embed_content(
                    model=f"models/{self.model_name_str}",
                    content=texts,
                    task_type="retrieval_document"
                )
                return result['embedding']
                
        except Exception as e:
            logger.error(f"Error embedding batch: {e}")
            # Retry with individual texts if batch fails
            if len(texts) > 1:
                return [self._embed_single(text) for text in texts]
            else:
                raise
    
    def _embed_single(self, text: str) -> List[float]:
        """Embed a single text with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = genai.embed_content(
                    model=f"models/{self.model_name_str}",
                    content=text,
                    task_type="retrieval_document"
                )
                return result['embedding']
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to embed text after {max_retries} attempts: {e}")
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector
        """
        try:
            result = genai.embed_content(
                model=f"models/{self.model_name_str}",
                content=query,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return getattr(self, '_dimension', 768)  # Default for text-embedding-004
    
    @property
    def model_name(self) -> str:
        """Return model name."""
        return self.model_name_str
