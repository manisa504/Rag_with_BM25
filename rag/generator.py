"""
Answer generation using Gemini 2.5 Flash with strict grounding and citations.
"""

from typing import List, Dict, Any, Optional
import logging
import json
from dataclasses import dataclass

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError("google-generativeai is not installed. Please install it with: pip install google-generativeai")

from .retriever import HybridResult
from .settings import settings

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Represents the result of answer generation."""
    answer: str
    citations: List[Dict[str, Any]]
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int


class AnswerGenerator:
    """Answer generator using Gemini 2.5 Flash with grounding and citations."""
    
    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Initialize answer generator.
        
        Args:
            api_key: Google API key
            model_name: Gemini model name
        """
        self.api_key = api_key or settings.google_api_key
        self.model_name = model_name or settings.gemini_model_text
        
        # Configure API
        genai.configure(api_key=self.api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(self.model_name)
        
        logger.info(f"Initialized answer generator with model: {self.model_name}")
    
    def generate_answer(
        self,
        question: str,
        context_results: List[HybridResult],
        temperature: float = 0.2,
        max_tokens: int = 2000
    ) -> GenerationResult:
        """
        Generate answer based on question and retrieved context.
        
        Args:
            question: User question
            context_results: Retrieved context from hybrid search
            temperature: Generation temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            GenerationResult with answer and citations
        """
        logger.info(f"Generating answer for question: '{question[:50]}...'")
        
        # Build prompt with context
        prompt = self._build_prompt(question, context_results)
        
        try:
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    top_p=0.8,
                    top_k=20
                )
            )
            
            # Parse response - handle both old and new API formats
            try:
                answer_text = response.text
            except Exception:
                # New API format - extract text from parts
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        answer_text = ''.join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                    else:
                        answer_text = str(candidate.content)
                elif hasattr(response, 'parts'):
                    answer_text = ''.join(part.text for part in response.parts if hasattr(part, 'text'))
                else:
                    raise Exception("Unable to extract text from response")
            
            # Extract citations from context
            citations = self._extract_citations(context_results, answer_text)
            
            # Get token usage (if available)
            token_info = self._get_token_usage(response)
            
            result = GenerationResult(
                answer=answer_text,
                citations=citations,
                total_tokens=token_info.get("total_tokens", 0),
                prompt_tokens=token_info.get("prompt_tokens", 0),
                completion_tokens=token_info.get("completion_tokens", 0)
            )
            
            logger.info(f"Answer generated successfully. Length: {len(answer_text)} chars")
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            # Return fallback response
            return self._generate_fallback_response(question)
    
    def _build_prompt(self, question: str, context_results: List[HybridResult]) -> str:
        """Build prompt with question and context."""
        
        # System instruction
        system_prompt = """You are an expert airline domain assistant. Your role is to provide accurate, helpful answers based strictly on the provided context.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the provided CONTEXT
2. If the answer is not in the context, respond with "Not in corpus."
3. Cite sources using [source_title] format in your answer
4. Provide detailed, comprehensive explanations with examples when available
5. Use aviation terminology appropriately and explain technical concepts
6. Prioritize safety-related information when relevant
7. Structure your response with clear explanations and supporting details

FORMAT:
- Provide a thorough, detailed answer to the question
- Include relevant citations in [source_title] format
- If multiple sources support a point, cite all relevant ones
- Explain technical concepts clearly with context
- Use bullet points or numbered lists when helpful for clarity"""

        # Build context section
        context_section = "CONTEXT:\n"
        for i, result in enumerate(context_results, 1):
            source_title = result.metadata.get("title", f"Document_{i}")
            context_section += f"\n[{source_title}]\n{result.text}\n"
            context_section += f"Vector Score: {result.vector_score:.3f}, BM25 Score: {result.bm25_score:.3f}\n"
        
        # Build question section
        question_section = f"\nQUESTION: {question}\n"
        
        # Build instruction section
        instruction_section = """
INSTRUCTIONS:
1. Review the context carefully
2. Answer the question using only the provided information
3. Include citations in [source_title] format
4. If the information is not available, respond with "Not in corpus."
5. Be accurate and concise

ANSWER:"""
        
        # Combine all sections
        full_prompt = system_prompt + "\n\n" + context_section + question_section + instruction_section
        
        return full_prompt
    
    def _extract_citations(
        self,
        context_results: List[HybridResult],
        answer_text: str
    ) -> List[Dict[str, Any]]:
        """Extract citations from context that appear in the answer."""
        citations = []
        
        for result in context_results:
            source_title = result.metadata.get("title", "Unknown")
            
            # Check if this source is referenced in the answer
            if f"[{source_title}]" in answer_text:
                citation = {
                    "source_title": source_title,
                    "source_path": result.metadata.get("source_path", ""),
                    "snippet": result.text[:200] + "..." if len(result.text) > 200 else result.text,
                    "vector_score": result.vector_score,
                    "bm25_score": result.bm25_score,
                    "final_score": result.final_score,
                    "doc_type": result.metadata.get("doc_type", "unknown"),
                    "topic": result.metadata.get("topic", "general")
                }
                citations.append(citation)
        
        # If no explicit citations found, include top sources anyway
        if not citations and context_results:
            for result in context_results[:3]:  # Top 3 sources
                source_title = result.metadata.get("title", "Unknown")
                citation = {
                    "source_title": source_title,
                    "source_path": result.metadata.get("source_path", ""),
                    "snippet": result.text[:200] + "..." if len(result.text) > 200 else result.text,
                    "vector_score": result.vector_score,
                    "bm25_score": result.bm25_score,
                    "final_score": result.final_score,
                    "doc_type": result.metadata.get("doc_type", "unknown"),
                    "topic": result.metadata.get("topic", "general"),
                    "implicit_citation": True
                }
                citations.append(citation)
        
        return citations
    
    def _get_token_usage(self, response) -> Dict[str, int]:
        """Extract token usage from response if available."""
        try:
            # Gemini API may not always provide token counts
            usage_metadata = getattr(response, 'usage_metadata', None)
            if usage_metadata:
                return {
                    "prompt_tokens": getattr(usage_metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(usage_metadata, 'candidates_token_count', 0),
                    "total_tokens": getattr(usage_metadata, 'total_token_count', 0)
                }
        except Exception as e:
            logger.warning(f"Could not extract token usage: {e}")
        
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    def _generate_fallback_response(self, question: str) -> GenerationResult:
        """Generate fallback response when main generation fails."""
        fallback_answer = (
            "I apologize, but I'm unable to generate a response at this time. "
            "This could be due to a temporary service issue or the question "
            "requiring information not available in the current knowledge base. "
            "Please try rephrasing your question or contact support if the issue persists."
        )
        
        return GenerationResult(
            answer=fallback_answer,
            citations=[],
            total_tokens=0,
            prompt_tokens=0,
            completion_tokens=0
        )
    
    def check_not_in_corpus(self, answer: str) -> bool:
        """Check if the answer indicates information is not in corpus."""
        not_in_corpus_phrases = [
            "not in corpus",
            "not available in the provided context",
            "information is not present",
            "cannot be found in the context",
            "not mentioned in the context"
        ]
        
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in not_in_corpus_phrases)
