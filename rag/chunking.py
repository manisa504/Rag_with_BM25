"""
Document chunking using Docling for parsing and semantic chunking.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import logging
from dataclasses import dataclass
import tiktoken

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
except ImportError:
    raise ImportError("Docling is not installed. Please install it with: pip install docling")

from .settings import settings

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    metadata: Dict[str, Any]
    token_count: int


class DoclingLoader:
    """Loader for documents using Docling."""
    
    def __init__(self):
        self.converter = DocumentConverter()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def load_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Load and parse a PDF file."""
        try:
            result = self.converter.convert(str(file_path))
            
            # Extract structured content
            content = {
                "text": result.document.export_to_markdown(),
                "metadata": {
                    "source_path": str(file_path),
                    "doc_type": "pdf",
                    "title": getattr(result.document, "title", file_path.stem),
                    "page_count": len(getattr(result.document, "pages", [])),
                }
            }
            
            return content
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise
    
    def load_html(self, file_path: Path) -> Dict[str, Any]:
        """Load and parse an HTML file."""
        try:
            result = self.converter.convert(str(file_path))
            
            content = {
                "text": result.document.export_to_markdown(),
                "metadata": {
                    "source_path": str(file_path),
                    "doc_type": "html",
                    "title": getattr(result.document, "title", file_path.stem),
                }
            }
            
            return content
            
        except Exception as e:
            logger.error(f"Error loading HTML {file_path}: {e}")
            raise
    
    def load_markdown(self, file_path: Path) -> Dict[str, Any]:
        """Load a markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            content = {
                "text": text,
                "metadata": {
                    "source_path": str(file_path),
                    "doc_type": "markdown",
                    "title": file_path.stem,
                }
            }
            
            return content
            
        except Exception as e:
            logger.error(f"Error loading Markdown {file_path}: {e}")
            raise


class SemanticChunker:
    """Semantic chunker that creates meaningful text chunks."""
    
    def __init__(self, chunk_tokens: int = None, overlap_tokens: int = None):
        self.chunk_tokens = chunk_tokens or settings.chunk_tokens
        self.overlap_tokens = overlap_tokens or settings.chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def split_by_sections(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Split text into chunks by sections with overlap."""
        chunks = []
        
        # Split by markdown headers or paragraphs
        sections = self._split_into_sections(text)
        
        current_chunk = ""
        current_tokens = 0
        
        for section in sections:
            section_tokens = self.count_tokens(section)
            
            # If section alone exceeds chunk size, split it further
            if section_tokens > self.chunk_tokens:
                if current_chunk:
                    # Save current chunk
                    chunks.append(self._create_chunk(current_chunk, metadata))
                    current_chunk = ""
                    current_tokens = 0
                
                # Split large section
                subsections = self._split_large_section(section)
                for subsection in subsections:
                    chunks.append(self._create_chunk(subsection, metadata))
                
            elif current_tokens + section_tokens > self.chunk_tokens:
                # Save current chunk and start new one with overlap
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, metadata))
                    
                    # Create overlap
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + "\n\n" + section
                    current_tokens = self.count_tokens(current_chunk)
                else:
                    current_chunk = section
                    current_tokens = section_tokens
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + section
                else:
                    current_chunk = section
                current_tokens += section_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, metadata))
        
        return chunks
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into logical sections."""
        # Split by double newlines first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        sections = []
        current_section = ""
        
        for para in paragraphs:
            # Check for headers (lines starting with #)
            if para.startswith('#') and current_section:
                sections.append(current_section.strip())
                current_section = para
            else:
                if current_section:
                    current_section += "\n\n" + para
                else:
                    current_section = para
        
        if current_section:
            sections.append(current_section.strip())
        
        return sections
    
    def _split_large_section(self, section: str) -> List[str]:
        """Split a large section into smaller chunks."""
        sentences = section.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if not sentence.endswith('.') and sentence != sentences[-1]:
                sentence += '.'
            
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if self.count_tokens(test_chunk) > self.chunk_tokens and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = test_chunk
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= self.overlap_tokens:
            return text
        
        overlap_tokens = tokens[-self.overlap_tokens:]
        return self.tokenizer.decode(overlap_tokens)
    
    def _create_chunk(self, text: str, base_metadata: Dict[str, Any]) -> Chunk:
        """Create a chunk object with metadata."""
        token_count = self.count_tokens(text)
        
        # Extract topic/section info from text start
        first_line = text.split('\n')[0].strip()
        topic = first_line if first_line.startswith('#') else "general"
        
        metadata = {
            **base_metadata,
            "topic": topic,
            "chunk_tokens": token_count,
        }
        
        return Chunk(text=text, metadata=metadata, token_count=token_count)


class DocumentProcessor:
    """Main document processor combining loading and chunking."""
    
    def __init__(self):
        self.loader = DoclingLoader()
        self.chunker = SemanticChunker()
    
    def process_file(self, file_path: Path) -> List[Chunk]:
        """Process a single file into chunks."""
        logger.info(f"Processing file: {file_path}")
        
        # Determine file type and load
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            content = self.loader.load_pdf(file_path)
        elif suffix in ['.html', '.htm']:
            content = self.loader.load_html(file_path)
        elif suffix in ['.md', '.markdown']:
            content = self.loader.load_markdown(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        # Chunk the content
        chunks = self.chunker.split_by_sections(content["text"], content["metadata"])
        
        logger.info(f"Created {len(chunks)} chunks from {file_path}")
        return chunks
    
    def process_files(self, file_paths: List[Path]) -> List[Chunk]:
        """Process multiple files into chunks."""
        all_chunks = []
        
        for file_path in file_paths:
            try:
                chunks = self.process_file(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def save_chunks(self, chunks: List[Chunk], output_path: Path) -> None:
        """Save chunks to JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                chunk_data = {
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "token_count": chunk.token_count
                }
                f.write(json.dumps(chunk_data, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")
    
    def load_chunks(self, input_path: Path) -> List[Chunk]:
        """Load chunks from JSONL file."""
        chunks = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                chunk = Chunk(
                    text=data["text"],
                    metadata=data["metadata"],
                    token_count=data["token_count"]
                )
                chunks.append(chunk)
        
        logger.info(f"Loaded {len(chunks)} chunks from {input_path}")
        return chunks
