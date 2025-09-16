"""
LLM-as-a-Judge evaluation using Gemini 2.5 Flash.
"""

from typing import Dict, Any, Optional, List
import logging
import json
import re
from dataclasses import dataclass

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError("google-generativeai is not installed. Please install it with: pip install google-generativeai")

from .settings import settings

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    """Result from LLM-as-a-Judge evaluation."""
    accuracy: float
    relevance: float
    completeness: float
    source_quality: float
    justification: str
    raw_response: str


class LLMJudge:
    """LLM-as-a-Judge for evaluating RAG responses using Gemini 2.5 Flash."""
    
    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Initialize LLM judge.
        
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
        
        logger.info(f"Initialized LLM judge with model: {self.model_name}")
    
    def evaluate(
        self,
        question: str,
        answer: str,
        context: List[Dict[str, Any]],
        reference_answer: Optional[str] = None
    ) -> JudgeResult:
        """
        Evaluate a RAG response using LLM-as-a-Judge.
        
        Args:
            question: Original question
            answer: Generated answer to evaluate
            context: Retrieved context used for generation
            reference_answer: Optional reference answer for comparison
            
        Returns:
            JudgeResult with scores and justification
        """
        logger.info(f"Evaluating answer for question: '{question[:50]}...'")
        
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(question, answer, context, reference_answer)
        
        try:
            # Generate evaluation
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistent evaluation
                    max_output_tokens=800,
                    top_p=0.8
                )
            )
            
            # Parse response - handle both old and new API formats
            try:
                response_text = response.text
            except Exception as e:
                # New API format - extract text from parts
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        response_text = ''.join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                    else:
                        response_text = str(candidate.content)
                elif hasattr(response, 'parts'):
                    response_text = ''.join(part.text for part in response.parts if hasattr(part, 'text'))
                else:
                    logger.error("Judge: Unable to extract text from response")
                    # Check if response was blocked
                    if hasattr(response, 'prompt_feedback'):
                        logger.error(f"Prompt feedback: {response.prompt_feedback}")
                    if hasattr(response, 'candidates') and response.candidates:
                        logger.error(f"Candidate finish reason: {response.candidates[0].finish_reason if response.candidates else 'No candidates'}")
                    raise Exception("Unable to extract text from response")
            
            # Parse evaluation results
            result = self._parse_evaluation_response(response_text)
            result.raw_response = response_text
            
            logger.info("Evaluation completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM evaluation: {e}")
            return self._generate_fallback_evaluation()
    
    def _build_evaluation_prompt(
        self,
        question: str,
        answer: str,
        context: List[Dict[str, Any]],
        reference_answer: Optional[str] = None
    ) -> str:
        """Build evaluation prompt for LLM judge."""
        
        # Base system prompt
        system_prompt = """You are an expert evaluator for airline domain question-answering systems. 
Your task is to evaluate the quality of answers based on specific criteria.

EVALUATION CRITERIA (score each from 0.0 to 1.0):

1. ACCURACY (0.0-1.0): How factually correct is the answer?
   - 1.0: Completely accurate with no errors
   - 0.8: Mostly accurate with minor issues
   - 0.6: Generally accurate but some concerns
   - 0.4: Some accurate elements but notable errors
   - 0.2: Largely inaccurate
   - 0.0: Completely wrong or misleading

2. RELEVANCE (0.0-1.0): How well does the answer address the question?
   - 1.0: Directly and fully addresses the question
   - 0.8: Addresses the question well with minor gaps
   - 0.6: Partially addresses the question
   - 0.4: Somewhat related but misses key aspects
   - 0.2: Marginally relevant
   - 0.0: Not relevant to the question

3. COMPLETENESS (0.0-1.0): How complete is the answer?
   - 1.0: Comprehensive, covers all important aspects
   - 0.8: Covers most important aspects
   - 0.6: Covers some key aspects
   - 0.4: Basic coverage, missing important details
   - 0.2: Incomplete, major gaps
   - 0.0: Very incomplete or too brief

4. SOURCE_QUALITY (0.0-1.0): How well are sources used and cited?
   - 1.0: Excellent source usage with proper citations
   - 0.8: Good source usage, mostly well-cited
   - 0.6: Adequate source usage
   - 0.4: Limited source usage or poor citations
   - 0.2: Minimal source usage
   - 0.0: No source usage or completely unsupported claims

OUTPUT FORMAT - IMPORTANT: You must respond with ONLY valid JSON, no other text:
{
    "accuracy": 0.8,
    "relevance": 0.9,
    "completeness": 0.7,
    "source_quality": 0.6,
    "justification": "Brief explanation of scores (2-3 sentences)"
}

DO NOT include any text before or after the JSON. Start your response with { and end with }."""

        # Build context section
        context_section = "\nCONTEXT PROVIDED:\n"
        for i, ctx in enumerate(context, 1):
            source_title = ctx.get("source_title", f"Source_{i}")
            snippet = ctx.get("snippet", ctx.get("text", ""))
            context_section += f"{i}. [{source_title}]: {snippet}\n"
        
        # Build reference section if provided
        reference_section = ""
        if reference_answer:
            reference_section = f"\nREFERENCE ANSWER:\n{reference_answer}\n"
        
        # Build main content
        content_section = f"""
QUESTION: {question}

ANSWER TO EVALUATE: {answer}
{context_section}{reference_section}
TASK: Evaluate the answer according to the four criteria above. 

RESPOND WITH ONLY VALID JSON (no explanation, no markdown, just JSON):"""
        
        return system_prompt + content_section
        
        return system_prompt + content_section
    
    def _parse_evaluation_response(self, response_text: str) -> JudgeResult:
        """Parse evaluation response and extract scores."""
        try:
            # Handle empty or whitespace-only responses
            if not response_text or not response_text.strip():
                # This is normal for some complex evaluations - just use fallback
                return self._parse_fallback_scores("")
            
            # Clean the response text
            response_text = response_text.strip()
            
            # Try to extract JSON from response
            json_match = re.search(r'\{[^}]*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                eval_data = json.loads(json_str)
            else:
                # Fallback: try to parse the entire response as JSON
                eval_data = json.loads(response_text)
            
            # Extract scores with validation
            accuracy = self._validate_score(eval_data.get("accuracy", 0.5))
            relevance = self._validate_score(eval_data.get("relevance", 0.5))
            completeness = self._validate_score(eval_data.get("completeness", 0.5))
            source_quality = self._validate_score(eval_data.get("source_quality", 0.5))
            justification = eval_data.get("justification", "No justification provided")
            
            return JudgeResult(
                accuracy=accuracy,
                relevance=relevance,
                completeness=completeness,
                source_quality=source_quality,
                justification=justification,
                raw_response=response_text
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # This is normal - just use fallback parsing
            return self._parse_fallback_scores(response_text)
    
    def _parse_fallback_scores(self, response_text: str) -> JudgeResult:
        """Fallback parsing for when JSON parsing fails."""
        # Try to extract scores using regex
        score_patterns = {
            "accuracy": r"accuracy[:\s]+([0-9.]+)",
            "relevance": r"relevance[:\s]+([0-9.]+)",
            "completeness": r"completeness[:\s]+([0-9.]+)",
            "source_quality": r"source[_\s]quality[:\s]+([0-9.]+)"
        }
        
        scores = {}
        for metric, pattern in score_patterns.items():
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    scores[metric] = self._validate_score(score)
                except ValueError:
                    scores[metric] = 0.5
            else:
                scores[metric] = 0.5
        
        return JudgeResult(
            accuracy=scores["accuracy"],
            relevance=scores["relevance"],
            completeness=scores["completeness"],
            source_quality=scores["source_quality"],
            justification="Scores extracted from unstructured response",
            raw_response=response_text
        )
    
    def _validate_score(self, score: float) -> float:
        """Validate and clamp score to [0.0, 1.0] range."""
        try:
            score = float(score)
            return max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            return 0.5  # Default to middle score if invalid
    
    def _generate_fallback_evaluation(self) -> JudgeResult:
        """Generate fallback evaluation when LLM evaluation fails."""
        return JudgeResult(
            accuracy=0.5,
            relevance=0.5,
            completeness=0.5,
            source_quality=0.5,
            justification="Evaluation failed, returning default scores",
            raw_response="Error: Unable to complete evaluation"
        )
    
    def batch_evaluate(
        self,
        evaluations: List[Dict[str, Any]]
    ) -> List[JudgeResult]:
        """
        Evaluate multiple QA pairs in batch.
        
        Args:
            evaluations: List of evaluation requests, each containing:
                - question: str
                - answer: str
                - context: List[Dict]
                - reference_answer: Optional[str]
                
        Returns:
            List of JudgeResult objects
        """
        results = []
        
        for i, eval_req in enumerate(evaluations):
            logger.info(f"Evaluating batch item {i+1}/{len(evaluations)}")
            
            result = self.evaluate(
                question=eval_req["question"],
                answer=eval_req["answer"],
                context=eval_req["context"],
                reference_answer=eval_req.get("reference_answer")
            )
            results.append(result)
        
        logger.info(f"Batch evaluation completed for {len(evaluations)} items")
        return results
