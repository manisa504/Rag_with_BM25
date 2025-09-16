"""
Evaluation metrics including LLM-based and traditional metrics.
"""

from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from rouge_score import rouge_scorer
except ImportError:
    raise ImportError("rouge-score is not installed. Please install it with: pip install rouge-score")

try:
    import sacrebleu
except ImportError:
    raise ImportError("sacrebleu is not installed. Please install it with: pip install sacrebleu")

from rag.judge import LLMJudge, JudgeResult

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result container for evaluation metrics."""
    accuracy_llm: float
    relevance_llm: float
    completeness_llm: float
    source_quality_llm: float
    rouge_l: float
    bleu: float
    overall_score: float
    justification: str


class EvaluationMetrics:
    """Comprehensive evaluation metrics for RAG systems."""
    
    def __init__(self, judge: Optional[LLMJudge] = None):
        """
        Initialize evaluation metrics.
        
        Args:
            judge: LLM judge instance for scoring
        """
        self.judge = judge or LLMJudge()
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        logger.info("Initialized evaluation metrics")
    
    def evaluate_response(
        self,
        question: str,
        answer: str,
        context: List[Dict[str, Any]],
        reference_answer: Optional[str] = None
    ) -> MetricResult:
        """
        Evaluate a single RAG response across all metrics.
        
        Args:
            question: Original question
            answer: Generated answer
            context: Retrieved context used for generation
            reference_answer: Optional reference answer
            
        Returns:
            MetricResult with all evaluation scores
        """
        logger.debug(f"Evaluating response for question: '{question[:50]}...'")
        
        # LLM-based evaluation
        judge_result = self.judge.evaluate(question, answer, context, reference_answer)
        
        # Traditional metrics (only if reference answer provided)
        rouge_l_score = 0.0
        bleu_score = 0.0
        
        if reference_answer:
            rouge_l_score = self.compute_rouge_l(reference_answer, answer)
            bleu_score = self.compute_bleu(reference_answer, answer)
        
        # Calculate overall score
        overall_score = self.calculate_overall_score(
            judge_result.accuracy,
            judge_result.relevance,
            judge_result.completeness,
            judge_result.source_quality
        )
        
        return MetricResult(
            accuracy_llm=judge_result.accuracy,
            relevance_llm=judge_result.relevance,
            completeness_llm=judge_result.completeness,
            source_quality_llm=judge_result.source_quality,
            rouge_l=rouge_l_score,
            bleu=bleu_score,
            overall_score=overall_score,
            justification=judge_result.justification
        )
    
    def accuracy_llm(
        self,
        question: str,
        answer: str,
        context: List[Dict[str, Any]],
        reference_answer: Optional[str] = None
    ) -> float:
        """Get accuracy score from LLM judge."""
        result = self.judge.evaluate(question, answer, context, reference_answer)
        return result.accuracy
    
    def relevance_llm(
        self,
        question: str,
        answer: str,
        context: List[Dict[str, Any]],
        reference_answer: Optional[str] = None
    ) -> float:
        """Get relevance score from LLM judge."""
        result = self.judge.evaluate(question, answer, context, reference_answer)
        return result.relevance
    
    def completeness_llm(
        self,
        question: str,
        answer: str,
        context: List[Dict[str, Any]],
        reference_answer: Optional[str] = None
    ) -> float:
        """Get completeness score from LLM judge."""
        result = self.judge.evaluate(question, answer, context, reference_answer)
        return result.completeness
    
    def source_quality_llm(
        self,
        question: str,
        answer: str,
        context: List[Dict[str, Any]],
        reference_answer: Optional[str] = None
    ) -> float:
        """Get source quality score from LLM judge."""
        result = self.judge.evaluate(question, answer, context, reference_answer)
        return result.source_quality
    
    def compute_rouge_l(self, reference: str, answer: str) -> float:
        """
        Compute ROUGE-L score between reference and answer.
        
        Args:
            reference: Reference answer
            answer: Generated answer
            
        Returns:
            ROUGE-L F1 score
        """
        try:
            scores = self.rouge_scorer.score(reference, answer)
            return scores['rougeL'].fmeasure
        except Exception as e:
            logger.warning(f"Error computing ROUGE-L: {e}")
            return 0.0
    
    def compute_bleu(self, reference: str, answer: str) -> float:
        """
        Compute BLEU score between reference and answer.
        
        Args:
            reference: Reference answer
            answer: Generated answer
            
        Returns:
            BLEU score (0-1 normalized)
        """
        try:
            # Use sacrebleu sentence_bleu function
            bleu_score = sacrebleu.sentence_bleu(answer, [reference])
            
            # Return normalized score (0-1)
            return bleu_score.score / 100.0
            
        except Exception as e:
            logger.warning(f"Error computing BLEU: {e}")
            return 0.0
    
    def calculate_overall_score(
        self,
        accuracy: float,
        relevance: float,
        completeness: float,
        source_quality: float
    ) -> float:
        """
        Calculate weighted overall score.
        
        Args:
            accuracy: Accuracy score (0-1)
            relevance: Relevance score (0-1)
            completeness: Completeness score (0-1)
            source_quality: Source quality score (0-1)
            
        Returns:
            Weighted overall score
        """
        # Weights: 35% accuracy, 25% relevance, 25% completeness, 15% source quality
        overall = (
            0.35 * accuracy +
            0.25 * relevance +
            0.25 * completeness +
            0.15 * source_quality
        )
        
        return overall
    
    def batch_evaluate(
        self,
        evaluations: List[Dict[str, Any]]
    ) -> List[MetricResult]:
        """
        Evaluate multiple responses in batch.
        
        Args:
            evaluations: List of evaluation requests, each containing:
                - question: str
                - answer: str
                - context: List[Dict]
                - reference_answer: Optional[str]
                
        Returns:
            List of MetricResult objects
        """
        results = []
        
        for i, eval_req in enumerate(evaluations):
            logger.info(f"Evaluating batch item {i+1}/{len(evaluations)}")
            
            result = self.evaluate_response(
                question=eval_req["question"],
                answer=eval_req["answer"],
                context=eval_req["context"],
                reference_answer=eval_req.get("reference_answer")
            )
            results.append(result)
        
        logger.info(f"Batch evaluation completed for {len(evaluations)} items")
        return results
    
    def compute_aggregate_metrics(self, results: List[MetricResult]) -> Dict[str, float]:
        """
        Compute aggregate metrics across multiple results.
        
        Args:
            results: List of MetricResult objects
            
        Returns:
            Dictionary of aggregated metrics
        """
        if not results:
            return {}
        
        # Calculate means
        aggregates = {
            "mean_accuracy": sum(r.accuracy_llm for r in results) / len(results),
            "mean_relevance": sum(r.relevance_llm for r in results) / len(results),
            "mean_completeness": sum(r.completeness_llm for r in results) / len(results),
            "mean_source_quality": sum(r.source_quality_llm for r in results) / len(results),
            "mean_rouge_l": sum(r.rouge_l for r in results) / len(results),
            "mean_bleu": sum(r.bleu for r in results) / len(results),
            "mean_overall": sum(r.overall_score for r in results) / len(results),
        }
        
        # Calculate standard deviations
        n = len(results)
        if n > 1:
            mean_acc = aggregates["mean_accuracy"]
            mean_rel = aggregates["mean_relevance"]
            mean_comp = aggregates["mean_completeness"]
            mean_sq = aggregates["mean_source_quality"]
            
            aggregates.update({
                "std_accuracy": (sum((r.accuracy_llm - mean_acc) ** 2 for r in results) / (n - 1)) ** 0.5,
                "std_relevance": (sum((r.relevance_llm - mean_rel) ** 2 for r in results) / (n - 1)) ** 0.5,
                "std_completeness": (sum((r.completeness_llm - mean_comp) ** 2 for r in results) / (n - 1)) ** 0.5,
                "std_source_quality": (sum((r.source_quality_llm - mean_sq) ** 2 for r in results) / (n - 1)) ** 0.5,
            })
        
        return aggregates
