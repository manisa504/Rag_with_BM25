"""
CLI script for checking Qdrant collection status.
"""

import sys
from pathlib import Path
import logging
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from rag.pipeline import RAGPipeline
from rag.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_bytes(bytes_value: int) -> str:
    """Format bytes in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def print_status():
    """Print RAG system status."""
    try:
        print("=" * 60)
        print("RAG SYSTEM STATUS")
        print("=" * 60)
        
        # Initialize pipeline
        pipeline = RAGPipeline()
        
        # Get status
        status = pipeline.get_status()
        
        # Print configuration
        print("\n📋 CONFIGURATION:")
        print(f"  Qdrant URL: {settings.qdrant_url}")
        print(f"  Collection: {settings.qdrant_collection}")
        print(f"  Embedding Backend: {settings.embedding_backend}")
        print(f"  Embedding Model: {status['embedding_provider']['model']}")
        print(f"  Generation Model: {status['generator_model']}")
        print(f"  Top-K: {settings.top_k}")
        print(f"  Rerank-K: {settings.rerank_k}")
        
        # Print collection info
        collection_info = status.get("collection_info", {})
        
        print("\n🗄️  VECTOR COLLECTION:")
        if collection_info:
            print(f"  Name: {collection_info.get('name', 'Unknown')}")
            print(f"  Points Count: {collection_info.get('points_count', 0):,}")
            print(f"  Segments: {collection_info.get('segments_count', 0)}")
            print(f"  Vector Size: {collection_info.get('vector_size', 'Unknown')}")
            print(f"  Distance Metric: {collection_info.get('distance', 'Unknown')}")
            print(f"  Status: {collection_info.get('status', 'Unknown')}")
            print(f"  Optimizer OK: {collection_info.get('optimizer_status', 'Unknown')}")
            print(f"  Indexed Vectors: {collection_info.get('indexed_vectors_count', 0):,}")
        else:
            print("  ❌ Collection not found or not accessible")
        
        # Print retriever stats
        print("\n🔍 RETRIEVER CONFIG:")
        retriever_stats = pipeline.retriever.get_stats()
        print(f"  Embedding Dimension: {retriever_stats['embedding_dimension']}")
        print(f"  Vector Top-K: {retriever_stats['top_k']}")
        print(f"  BM25 Rerank-K: {retriever_stats['rerank_k']}")
        
        # Memory and performance estimates
        if collection_info and collection_info.get('points_count', 0) > 0:
            points_count = collection_info['points_count']
            vector_size = collection_info.get('vector_size', 768)
            
            # Estimate memory usage (rough calculation)
            vector_memory = points_count * vector_size * 4  # 4 bytes per float32
            metadata_memory = points_count * 1024  # Rough estimate for metadata
            total_memory = vector_memory + metadata_memory
            
            print("\n💾 ESTIMATED MEMORY USAGE:")
            print(f"  Vector Data: {format_bytes(vector_memory)}")
            print(f"  Metadata: {format_bytes(metadata_memory)}")
            print(f"  Total Estimated: {format_bytes(total_memory)}")
        
        # Check if processed chunks exist
        processed_chunks_path = Path("data/processed/chunks.jsonl")
        if processed_chunks_path.exists():
            try:
                chunk_count = sum(1 for _ in open(processed_chunks_path, 'r'))
                file_size = processed_chunks_path.stat().st_size
                print(f"\n📝 PROCESSED CHUNKS:")
                print(f"  File: {processed_chunks_path}")
                print(f"  Chunks: {chunk_count:,}")
                print(f"  File Size: {format_bytes(file_size)}")
            except Exception as e:
                print(f"\n📝 PROCESSED CHUNKS: Error reading file - {e}")
        else:
            print(f"\n📝 PROCESSED CHUNKS: No processed chunks found")
        
        print("\n✅ Status check completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error checking status: {e}")
        logger.exception("Detailed error:")
        sys.exit(1)


def main():
    """Main function."""
    print_status()


if __name__ == "__main__":
    main()
