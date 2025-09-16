"""
High-level RAG pipeline combining all components.
"""

from typing import List, Dict, Any, Optional
import logging
import time
from pathlib import Path
import pandas as pd
from dataclasses import asdict

from .chunking_minimal import DocumentProcessor, Chunk
from .embeddings.google_embeddings import GoogleEmbeddingProvider
from .embeddings.gemma_embeddings import GemmaEmbeddingProvider
from .vectorstore.qdrant_store import QdrantStore
from .retriever import HybridRetriever
from .generator import AnswerGenerator, GenerationResult
from .judge import LLMJudge, JudgeResult
from .settings import settings

logger = logging.getLogger(__name__)


class RAGPipeline:
    """High-level RAG pipeline orchestrating all components."""
    
    def __init__(self):
        """Initialize RAG pipeline with all components."""
        logger.info("Initializing RAG pipeline...")
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.embedding_provider = self._create_embedding_provider()
        self.vector_store = QdrantStore()
        self.retriever = HybridRetriever(self.vector_store, self.embedding_provider)
        self.generator = AnswerGenerator()
        self.judge = LLMJudge()
        
        # Ensure collection exists
        self._ensure_collection_exists()
        
        logger.info("RAG pipeline initialized successfully")
    
    def _create_embedding_provider(self):
        """Create embedding provider based on settings."""
        if settings.embedding_backend == "google":
            return GoogleEmbeddingProvider()
        elif settings.embedding_backend == "gemma":
            return GemmaEmbeddingProvider()
        else:
            raise ValueError(f"Unsupported embedding backend: {settings.embedding_backend}")
    
    def _ensure_collection_exists(self):
        """Ensure Qdrant collection exists."""
        try:
            # Try to get collection info directly from Qdrant client
            collection_info = self.vector_store.client.get_collection(settings.qdrant_collection)
            if collection_info:
                logger.info(f"Using existing collection: {settings.qdrant_collection}")
                return
        except Exception as e:
            # Collection doesn't exist, create it
            logger.info(f"Collection doesn't exist, creating it. Error: {e}")
        
        # Create collection
        dimension = self.embedding_provider.dimension
        self.vector_store.create_collection(dimension)
        logger.info(f"Created new collection: {settings.qdrant_collection}")
    
    def ingest(self, paths: List[str]) -> None:
        """
        Ingest documents into the vector store.
        
        Args:
            paths: List of file paths to ingest
        """
        logger.info(f"Starting ingestion of {len(paths)} files")
        
        # Convert paths to Path objects
        file_paths = [Path(path) for path in paths]
        
        # Check if we have preprocessed chunks (.jsonl files)
        chunks = []
        for file_path in file_paths:
            if file_path.suffix.lower() == '.jsonl':
                # Load preprocessed chunks
                logger.info(f"Loading preprocessed chunks from {file_path}")
                chunks.extend(self.document_processor.load_chunks(file_path))
            else:
                # Process raw documents into chunks
                logger.info(f"Processing raw document: {file_path}")
                file_chunks = self.document_processor.process_files([file_path])
                chunks.extend(file_chunks)
        
        if not chunks:
            logger.warning("No chunks created from input files")
            return
        
        # Extract texts and metadata
        texts = [chunk.text for chunk in chunks]
        metadata_list = [chunk.metadata for chunk in chunks]
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedding_provider.embed_texts(texts)
        
        # Upsert to vector store
        logger.info("Upserting to vector store...")
        point_ids = self.vector_store.upsert_points(texts, embeddings, metadata_list)
        
        # Save processed chunks (only if we processed raw files)
        if any(path.suffix.lower() != '.jsonl' for path in file_paths):
            output_path = Path("data/processed/chunks.jsonl")
            if hasattr(self.document_processor, 'save_chunks'):
                self.document_processor.save_chunks(chunks, output_path)
        
        logger.info(f"Ingestion completed. {len(point_ids)} points added to vector store")
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        rerank_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: User question
            top_k: Override default top_k for retrieval
            rerank_k: Override default rerank_k for final results
            
        Returns:
            Dictionary containing answer, citations, and metadata
        """
        start_time = time.time()
        
        logger.info(f"Processing query: '{question[:50]}...'")
        
        try:
            # Retrieve relevant context
            retrieval_start = time.time()
            context_results = self.retriever.retrieve(
                query=question,
                top_k=top_k,
                rerank_k=rerank_k
            )
            retrieval_time = time.time() - retrieval_start
            
            if not context_results:
                return {
                    "answer": "I don't have enough information to answer this question.",
                    "citations": [],
                    "latency_s": time.time() - start_time,
                    "retrieval_time_s": retrieval_time,
                    "generation_time_s": 0,
                    "context_found": False
                }
            
            # Generate answer
            generation_start = time.time()
            generation_result = self.generator.generate_answer(question, context_results)
            generation_time = time.time() - generation_start
            
            # Check if answer indicates "not in corpus"
            not_in_corpus = self.generator.check_not_in_corpus(generation_result.answer)
            
            total_time = time.time() - start_time
            
            result = {
                "answer": generation_result.answer,
                "citations": generation_result.citations,
                "latency_s": total_time,
                "retrieval_time_s": retrieval_time,
                "generation_time_s": generation_time,
                "context_found": True,
                "not_in_corpus": not_in_corpus,
                "token_usage": {
                    "total_tokens": generation_result.total_tokens,
                    "prompt_tokens": generation_result.prompt_tokens,
                    "completion_tokens": generation_result.completion_tokens
                },
                "retrieval_debug": [
                    {
                        "text_snippet": result.text[:200] + "..." if len(result.text) > 200 else result.text,
                        "metadata": result.metadata,
                        "vector_score": result.vector_score,
                        "bm25_score": result.bm25_score,
                        "final_score": result.final_score
                    }
                    for result in context_results
                ]
            }
            
            logger.info(f"Query completed in {total_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": "I apologize, but I encountered an error processing your question. Please try again.",
                "citations": [],
                "latency_s": time.time() - start_time,
                "error": str(e)
            }
    
    def evaluate(self, questions: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Evaluate the RAG system on a set of questions.
        
        Args:
            questions: List of QA pairs, each containing:
                - question: str
                - reference_answer: str (optional)
                
        Returns:
            DataFrame with evaluation results
        """
        logger.info(f"Starting evaluation on {len(questions)} questions")
        
        results = []
        
        for i, qa in enumerate(questions):
            logger.info(f"Evaluating question {i+1}/{len(questions)}")
            
            question = qa["question"]
            reference_answer = qa.get("reference_answer")
            
            # Get RAG response
            start_time = time.time()
            rag_result = self.query(question)
            query_time = time.time() - start_time
            
            # Prepare context for judge
            context = [
                {
                    "source_title": citation["source_title"],
                    "snippet": citation["snippet"],
                    "text": citation["snippet"]
                }
                for citation in rag_result.get("citations", [])
            ]
            
            # Evaluate with LLM judge
            judge_start = time.time()
            judge_result = self.judge.evaluate(
                question=question,
                answer=rag_result["answer"],
                context=context,
                reference_answer=reference_answer
            )
            judge_time = time.time() - judge_start
            
            # Compile results
            result_row = {
                "question": question,
                "answer": rag_result["answer"],
                "reference_answer": reference_answer or "",
                "accuracy": judge_result.accuracy,
                "relevance": judge_result.relevance,
                "completeness": judge_result.completeness,
                "source_quality": judge_result.source_quality,
                "justification": judge_result.justification,
                "query_time_s": query_time,
                "judge_time_s": judge_time,
                "total_time_s": query_time + judge_time,
                "not_in_corpus": rag_result.get("not_in_corpus", False),
                "num_citations": len(rag_result.get("citations", [])),
                "token_usage": rag_result.get("token_usage", {}),
                "retrieval_time_s": rag_result.get("retrieval_time_s", 0),
                "generation_time_s": rag_result.get("generation_time_s", 0)
            }
            
            results.append(result_row)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Calculate overall score
        df["overall_score"] = (
            0.35 * df["accuracy"] +
            0.25 * df["relevance"] +
            0.25 * df["completeness"] +
            0.15 * df["source_quality"]
        )
        
        # Save results
        output_path = Path("evaluation/results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output_path, orient="records", indent=2)
        
        logger.info(f"Evaluation completed. Results saved to {output_path}")
        
        # Log summary statistics
        avg_scores = df[["accuracy", "relevance", "completeness", "source_quality", "overall_score"]].mean()
        logger.info(f"Average scores: {avg_scores.to_dict()}")
        
        return df
    
    def get_status(self) -> Dict[str, Any]:
        """Get status information about the RAG system."""
        try:
            collection_info = self.vector_store.get_collection_info()
            retriever_stats = self.retriever.get_stats()
            
            status = {
                "collection_name": settings.qdrant_collection,
                "collection_info": collection_info,
                "embedding_provider": {
                    "backend": settings.embedding_backend,
                    "model": self.embedding_provider.model_name,
                    "dimension": self.embedding_provider.dimension
                },
                "retriever_config": {
                    "top_k": settings.top_k,
                    "rerank_k": settings.rerank_k
                },
                "generator_model": settings.gemini_model_text,
                "qdrant_url": settings.qdrant_url
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {"error": str(e)}
