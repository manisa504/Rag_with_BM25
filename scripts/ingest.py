"""
CLI script for ingesting documents into the RAG system.
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import List

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


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ingest.py --paths data/raw/*.pdf
  python scripts/ingest.py --paths data/raw/doc1.pdf data/raw/doc2.html
  python scripts/ingest.py --paths data/raw/*.pdf --force-recreate
        """
    )
    
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="Paths to documents to ingest (supports glob patterns)"
    )
    
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Force recreation of the vector collection"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def expand_paths(paths: List[str]) -> List[Path]:
    """Expand glob patterns and validate paths."""
    expanded_paths = []
    
    for path_str in paths:
        path = Path(path_str)
        
        # Check if it's a glob pattern
        if "*" in str(path) or "?" in str(path):
            # Expand glob pattern
            parent = path.parent
            pattern = path.name
            
            if parent.exists():
                matches = list(parent.glob(pattern))
                expanded_paths.extend(matches)
            else:
                logger.warning(f"Parent directory does not exist: {parent}")
        else:
            # Regular path
            if path.exists():
                expanded_paths.append(path)
            else:
                logger.warning(f"Path does not exist: {path}")
    
    # Filter for supported file types
    supported_extensions = {'.pdf', '.html', '.htm', '.md', '.markdown', '.jsonl', '.json'}
    filtered_paths = []
    
    for path in expanded_paths:
        if path.is_file() and path.suffix.lower() in supported_extensions:
            filtered_paths.append(path)
        elif path.is_file():
            logger.warning(f"Unsupported file type: {path}")
        elif path.is_dir():
            logger.warning(f"Path is a directory, not a file: {path}")
    
    return filtered_paths


def main():
    """Main function."""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting document ingestion")
    logger.info(f"Qdrant URL: {settings.qdrant_url}")
    logger.info(f"Collection: {settings.qdrant_collection}")
    logger.info(f"Embedding backend: {settings.embedding_backend}")
    
    try:
        # Expand and validate paths
        file_paths = expand_paths(args.paths)
        
        if not file_paths:
            logger.error("No valid files found to ingest")
            sys.exit(1)
        
        logger.info(f"Found {len(file_paths)} files to ingest:")
        for path in file_paths:
            logger.info(f"  - {path}")
        
        # Initialize pipeline
        logger.info("Initializing RAG pipeline...")
        pipeline = RAGPipeline()
        
        # Force recreate collection if requested
        if args.force_recreate:
            logger.info("Force recreating vector collection...")
            pipeline.vector_store.delete_collection()
            dimension = pipeline.embedding_provider.dimension
            pipeline.vector_store.create_collection(dimension, force_recreate=True)
        
        # Ingest documents
        logger.info("Starting ingestion process...")
        pipeline.ingest([str(path) for path in file_paths])
        
        # Show final status
        status = pipeline.get_status()
        collection_info = status.get("collection_info", {})
        
        logger.info("Ingestion completed successfully!")
        logger.info(f"Collection points: {collection_info.get('points_count', 'Unknown')}")
        logger.info(f"Vector dimension: {collection_info.get('vector_size', 'Unknown')}")
        
    except KeyboardInterrupt:
        logger.info("Ingestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
