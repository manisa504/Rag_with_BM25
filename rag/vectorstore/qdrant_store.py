"""
Qdrant vector store implementation with cosine similarity and HNSW indexing.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import uuid
import warnings
from dataclasses import dataclass

# Suppress Qdrant version compatibility warnings
warnings.filterwarnings("ignore", message=".*Qdrant client version.*incompatible.*")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
except ImportError:
    raise ImportError("qdrant-client is not installed. Please install it with: pip install qdrant-client")

from ..settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result from Qdrant."""
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float


class QdrantStore:
    """Qdrant vector store with cosine similarity and HNSW indexing."""
    
    def __init__(
        self,
        url: str = None,
        api_key: str = None,
        collection_name: str = None
    ):
        """
        Initialize Qdrant client and collection.
        
        Args:
            url: Qdrant server URL
            api_key: API key for authentication
            collection_name: Name of the collection
        """
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key
        self.collection_name = collection_name or settings.qdrant_collection
        
        # Initialize client
        if self.api_key:
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
        else:
            self.client = QdrantClient(url=self.url)
        
        logger.info(f"Initialized Qdrant client for {self.url}")
    
    def create_collection(self, dimension: int, force_recreate: bool = False) -> None:
        """
        Create collection with HNSW configuration.
        
        Args:
            dimension: Vector dimension
            force_recreate: Whether to recreate if collection exists
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(
                collection.name == self.collection_name 
                for collection in collections.collections
            )
            
            if collection_exists:
                if force_recreate:
                    logger.info(f"Deleting existing collection: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                else:
                    logger.info(f"Collection {self.collection_name} already exists")
                    return
            
            # Create collection with HNSW parameters
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=32,  # Number of bi-directional links for every new element
                    ef_construct=128,  # Size of the dynamic candidate list
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=2,
                    max_segment_size=None,
                    memmap_threshold=None,
                    indexing_threshold=10000,
                    flush_interval_sec=5,
                    max_optimization_threads=None,
                )
            )
            
            logger.info(f"Created collection: {self.collection_name} with dimension {dimension}")
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def upsert_points(
        self,
        texts: List[str],
        vectors: List[List[float]],
        metadata_list: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Upsert points to the collection.
        
        Args:
            texts: List of text content
            vectors: List of embedding vectors
            metadata_list: List of metadata dictionaries
            
        Returns:
            List of point IDs
        """
        if len(texts) != len(vectors) != len(metadata_list):
            raise ValueError("texts, vectors, and metadata_list must have the same length")
        
        points = []
        point_ids = []
        
        for text, vector, metadata in zip(texts, vectors, metadata_list):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            
            # Combine text and metadata in payload
            payload = {
                "text": text,
                **metadata
            }
            
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
            points.append(point)
        
        try:
            # Upsert in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            
            logger.info(f"Upserted {len(points)} points to {self.collection_name}")
            return point_ids
            
        except Exception as e:
            logger.error(f"Error upserting points: {e}")
            raise
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_conditions: Optional filter conditions
            
        Returns:
            List of search results
        """
        try:
            # Prepare filter if provided
            query_filter = None
            if filter_conditions:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                        for key, value in filter_conditions.items()
                    ]
                )
            
            # Perform search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )
            
            # Convert to SearchResult objects
            results = []
            for hit in search_result:
                result = SearchResult(
                    id=str(hit.id),
                    text=hit.payload["text"],
                    metadata={k: v for k, v in hit.payload.items() if k != "text"},
                    score=hit.score
                )
                results.append(result)
            
            logger.info(f"Found {len(results)} results for search query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Collection information dictionary
        """
        try:
            info = self.client.get_collection(self.collection_name)
            
            # Handle optimizer status safely
            optimizer_ok = False
            try:
                if hasattr(info.optimizer_status, 'ok'):
                    optimizer_ok = info.optimizer_status.ok
                elif hasattr(info.optimizer_status, 'status'):
                    optimizer_ok = info.optimizer_status.status == "ok"
                else:
                    optimizer_ok = str(info.optimizer_status).lower() == "ok"
            except:
                optimizer_ok = False
            
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance.value,
                "status": info.status.value,
                "optimizer_status": optimizer_ok,
                "indexed_vectors_count": info.indexed_vectors_count,
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                "name": self.collection_name,
                "points_count": 0,
                "segments_count": 0,
                "vector_size": 768,
                "distance": "Cosine",
                "status": "Unknown",
                "optimizer_status": False,
                "indexed_vectors_count": 0,
            }
    
    def delete_collection(self) -> None:
        """Delete the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise
    
    def count_points(self) -> int:
        """
        Count total points in collection.
        
        Returns:
            Number of points
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except Exception as e:
            logger.error(f"Error counting points: {e}")
            return 0
