"""
CLI script for running batch evaluation and saving results.
"""

import argparse
import sys
from pathlib import Path
import logging
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from rag.pipeline import RAGPipeline
from evaluation.datasets import get_evaluation_dataset, get_sample_questions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run batch evaluation on the RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/eval_run.py
  python scripts/eval_run.py --sample 5
  python scripts/eval_run.py --output-file my_results.json
        """
    )
    
    parser.add_argument(
        "--sample",
        type=int,
        help="Run evaluation on a sample of N questions instead of full dataset"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default="evaluation/results.json",
        help="Output file for results (default: evaluation/results.json)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting batch evaluation")
    
    try:
        # Get evaluation dataset
        if args.sample:
            logger.info(f"Using sample of {args.sample} questions")
            questions = get_sample_questions(args.sample)
        else:
            logger.info("Using full evaluation dataset")
            questions = get_evaluation_dataset()
        
        logger.info(f"Evaluating {len(questions)} questions")
        
        # Initialize pipeline
        logger.info("Initializing RAG pipeline...")
        pipeline = RAGPipeline()
        
        # Check system status
        status = pipeline.get_status()
        collection_info = status.get("collection_info", {})
        points_count = collection_info.get("points_count", 0)
        
        if points_count == 0:
            logger.warning("No documents found in vector store. Please run ingestion first.")
            print("\nTo ingest documents, run:")
            print("python scripts/ingest.py --paths data/raw/*.pdf")
            sys.exit(1)
        
        logger.info(f"Found {points_count:,} documents in vector store")
        
        # Run evaluation
        logger.info("Starting evaluation...")
        results_df = pipeline.evaluate(questions)
        
        # Save results
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        results_dict = results_df.to_dict(orient="records")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        avg_metrics = results_df[[
            'accuracy', 'relevance', 'completeness', 'source_quality', 'overall_score'
        ]].mean()
        
        print(f"\nAverage Scores:")
        print(f"  Accuracy:       {avg_metrics['accuracy']:.3f}")
        print(f"  Relevance:      {avg_metrics['relevance']:.3f}")
        print(f"  Completeness:   {avg_metrics['completeness']:.3f}")
        print(f"  Source Quality: {avg_metrics['source_quality']:.3f}")
        print(f"  Overall Score:  {avg_metrics['overall_score']:.3f}")
        
        # Performance metrics
        avg_time = results_df['total_time_s'].mean()
        not_in_corpus_rate = results_df['not_in_corpus'].mean()
        
        print(f"\nPerformance Metrics:")
        print(f"  Avg Response Time: {avg_time:.2f}s")
        print(f"  Not in Corpus Rate: {not_in_corpus_rate:.1%}")
        print(f"  Total Questions: {len(results_df)}")
        
        # Time breakdown if available
        time_cols = ['retrieval_time_s', 'generation_time_s', 'judge_time_s']
        available_time_cols = [col for col in time_cols if col in results_df.columns]
        
        if available_time_cols:
            print(f"\nTime Breakdown:")
            for col in available_time_cols:
                avg_time = results_df[col].mean()
                label = col.replace('_time_s', '').replace('_', ' ').title()
                print(f"  {label}: {avg_time:.2f}s")
        
        # Quality distribution
        print(f"\nQuality Distribution:")
        overall_scores = results_df['overall_score']
        print(f"  Excellent (>0.8): {(overall_scores > 0.8).sum()} questions")
        print(f"  Good (0.6-0.8):   {((overall_scores > 0.6) & (overall_scores <= 0.8)).sum()} questions")
        print(f"  Fair (0.4-0.6):   {((overall_scores > 0.4) & (overall_scores <= 0.6)).sum()} questions")
        print(f"  Poor (<0.4):      {(overall_scores <= 0.4).sum()} questions")
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📊 View results in Streamlit dashboard: streamlit run app.py")
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
