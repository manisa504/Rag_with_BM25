"""
Minimal chunking module for apps that don't need document processing.
Used when documents are already processed and stored.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """A chunk of text with metadata."""
    text: str
    metadata: Dict[str, Any]
    token_count: Optional[int] = None

class DocumentProcessor:
    """Minimal document processor that can only load existing chunks."""
    
    def __init__(self):
        logger.info("Initialized minimal document processor (no document processing capability)")
    
    def load_chunks(self, chunks_file: Path) -> List[Chunk]:
        """Load chunks from existing JSONL file."""
        chunks = []
        if not chunks_file.exists():
            logger.warning(f"Chunks file not found: {chunks_file}")
            return chunks
            
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    chunk = Chunk(
                        text=data['text'],
                        metadata=data['metadata'],
                        token_count=data.get('token_count')
                    )
                    chunks.append(chunk)
                except Exception as e:
                    logger.warning(f"Error loading chunk at line {line_num}: {e}")
        
        logger.info(f"Loaded {len(chunks)} chunks from {chunks_file}")
        return chunks
    
    def process_files(self, file_paths: List[Path]) -> List[Chunk]:
        """Not implemented in minimal version."""
        raise NotImplementedError("Document processing not available in minimal version. Documents must be pre-processed.")
