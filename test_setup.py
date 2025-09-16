"""
Quick test script to verify the RAG system setup.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from rag.settings import settings
        print("✅ Settings loaded successfully")
        print(f"   - Embedding backend: {settings.embedding_backend}")
        print(f"   - Qdrant URL: {settings.qdrant_url}")
        print(f"   - Collection: {settings.qdrant_collection}")
    except Exception as e:
        print(f"❌ Settings import failed: {e}")
        return False
    
    try:
        from rag.chunking import DocumentProcessor
        print("✅ Document processor imported successfully")
    except Exception as e:
        print(f"❌ Document processor import failed: {e}")
        return False
    
    try:
        from rag.embeddings.google_embeddings import GoogleEmbeddingProvider
        print("✅ Google embeddings imported successfully")
    except Exception as e:
        print(f"❌ Google embeddings import failed: {e}")
        return False
    
    try:
        from rag.vectorstore.qdrant_store import QdrantStore
        print("✅ Qdrant store imported successfully")
    except Exception as e:
        print(f"❌ Qdrant store import failed: {e}")
        return False
    
    try:
        from rag.retriever import HybridRetriever
        print("✅ Hybrid retriever imported successfully")
    except Exception as e:
        print(f"❌ Hybrid retriever import failed: {e}")
        return False
    
    try:
        from rag.generator import AnswerGenerator
        print("✅ Answer generator imported successfully")
    except Exception as e:
        print(f"❌ Answer generator import failed: {e}")
        return False
    
    try:
        from rag.judge import LLMJudge
        print("✅ LLM judge imported successfully")
    except Exception as e:
        print(f"❌ LLM judge import failed: {e}")
        return False
    
    try:
        from rag.pipeline import RAGPipeline
        print("✅ RAG pipeline imported successfully")
    except Exception as e:
        print(f"❌ RAG pipeline import failed: {e}")
        return False
    
    return True


def check_data_files():
    """Check that data files are in place."""
    print("\nChecking data files...")
    
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {data_dir}")
        return False
    
    print(f"✅ Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        print(f"   - {pdf_file.name}")
    
    return True


def check_environment():
    """Check environment variables."""
    print("\nChecking environment...")
    
    required_vars = ["GOOGLE_API_KEY"]
    optional_vars = ["QDRANT_URL", "QDRANT_API_KEY"]
    
    all_good = True
    
    for var in required_vars:
        if var in os.environ and os.environ[var]:
            print(f"✅ {var} is set")
        else:
            print(f"❌ {var} is not set (required)")
            all_good = False
    
    for var in optional_vars:
        if var in os.environ and os.environ[var]:
            print(f"✅ {var} is set")
        else:
            print(f"⚠️  {var} is not set (optional)")
    
    return all_good


def main():
    """Run all tests."""
    print("🧪 RAG System Setup Test")
    print("=" * 40)
    
    # Check environment
    env_ok = check_environment()
    
    # Check data files
    data_ok = check_data_files()
    
    # Test imports
    imports_ok = test_imports()
    
    print("\n" + "=" * 40)
    print("📋 SUMMARY")
    print("=" * 40)
    
    if env_ok and data_ok and imports_ok:
        print("✅ All checks passed! System is ready.")
        print("\n🚀 Next steps:")
        print("1. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        print("2. Ingest documents: python scripts/ingest.py --paths data/raw/*.pdf")
        print("3. Launch app: streamlit run app.py")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        if not env_ok:
            print("\n📋 Environment setup:")
            print("1. Copy .env.example to .env")
            print("2. Add your GOOGLE_API_KEY to .env")
        if not data_ok:
            print("\n📁 Data setup:")
            print("1. Add PDF files to data/raw/ directory")
        if not imports_ok:
            print("\n📦 Dependencies:")
            print("1. Run: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
