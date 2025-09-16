
import sys
import os
import json
import pandas as pd

# Ensure the rag module is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.pipeline import RAGPipeline
from evaluation.datasets import get_sample_questions

def run_test(test_name, test_function):
    """Helper to run a test and print status."""
    print(f"--- Running test: {test_name} ---")
    try:
        test_function()
        print(f"✅ PASSED: {test_name}\n")
        return True
    except Exception as e:
        print(f"❌ FAILED: {test_name}")
        print(f"   Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_initialization():
    """Tests if the RAGPipeline initializes without errors."""
    pipeline = RAGPipeline()
    assert pipeline is not None, "Pipeline object is None"
    assert pipeline.retriever is not None, "Retriever is None"
    assert pipeline.generator is not None, "Generator is None"
    assert pipeline.judge is not None, "Judge is None"
    print("   Pipeline initialized successfully.")

def test_simple_query():
    """Tests a simple query and checks the output structure."""
    pipeline = RAGPipeline()
    query = "What is BADA?"
    result = pipeline.query(query)
    
    assert "answer" in result, "Result missing 'answer' key"
    assert "citations" in result, "Result missing 'citations' key"
    assert result["answer"], "Answer is empty"
    assert len(result["answer"]) > 50, "Answer is too short"
    assert len(result["citations"]) > 0, "No citations found"
    
    print(f"   Query: '{query}'")
    print(f"   Answer: {result['answer'][:100]}...")
    print(f"   Citations found: {len(result['citations'])}")

def test_out_of_domain_query():
    """Tests a query that is not in the document corpus."""
    pipeline = RAGPipeline()
    query = "What is the capital of France?"
    result = pipeline.query(query)
    
    assert "answer" in result, "Result missing 'answer' key"
    assert result["answer"], "Answer is empty"
    assert "not in our corpus" in result["answer"].lower() or \
           "i don't know" in result["answer"].lower() or \
           "i cannot answer" in result["answer"].lower(), \
           "Answer does not indicate out-of-domain"
           
    print(f"   Query: '{query}'")
    print(f"   Response indicates out-of-domain: '{result['answer'][:100]}...'")

def test_evaluation_run():
    """Tests the full evaluation pipeline on a sample question."""
    pipeline = RAGPipeline()
    questions = get_sample_questions(1)
    
    eval_results = pipeline.evaluate(questions)
    
    assert isinstance(eval_results, pd.DataFrame), "Evaluation result is not a DataFrame"
    assert not eval_results.empty, "Evaluation DataFrame is empty"
    assert "question" in eval_results.columns
    assert "answer" in eval_results.columns
    assert "accuracy" in eval_results.columns
    assert "relevance" in eval_results.columns
    
    print("   Evaluation ran successfully.")
    print(f"   Result columns: {list(eval_results.columns)}")
    print(f"   Sample accuracy: {eval_results.iloc[0]['accuracy']}")

def main():
    """Main function to run all tests."""
    print("🚀 Starting RAG System Comprehensive Test 🚀\n")
    
    tests = {
        "Pipeline Initialization": test_pipeline_initialization,
        "Simple Query": test_simple_query,
        "Out-of-Domain Query": test_out_of_domain_query,
        "Evaluation Run": test_evaluation_run,
    }
    
    results = {name: run_test(name, func) for name, func in tests.items()}
    
    print("--- Test Summary ---")
    passed_count = sum(results.values())
    total_count = len(results)
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")
        
    print(f"\n🏁 Test Complete: {passed_count}/{total_count} tests passed. 🏁")
    
    if passed_count != total_count:
        sys.exit(1)

if __name__ == "__main__":
    main()
