"""
Base embedding provider protocol.
"""

from typing import List, Protocol
from abc import ABC, abstractmethod


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts and return their vector representations.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            List of embedding vectors (list of floats)
        """
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            query: Query string to embed
            
        Returns:
            Embedding vector as list of floats
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of embeddings produced by this provider."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name used by this provider."""
        pass


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model name."""
        pass
