#!/bin/bash

# Aviation RAG System Setup Script
echo "✈️  Setting up Aviation LLM Assistant..."
echo "=================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python version $python_version is compatible"
else
    echo "❌ Python $required_version or higher required. Found: $python_version"
    exit 1
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Setup environment
if [ ! -f .env ]; then
    echo "📋 Creating environment file..."
    cp .env.example .env
    echo "✅ Created .env file from template"
    echo "⚠️  Please edit .env and add your GOOGLE_API_KEY"
else
    echo "✅ Environment file already exists"
fi

# Check for data files
pdf_count=$(find data/raw -name "*.pdf" 2>/dev/null | wc -l)
if [ "$pdf_count" -gt 0 ]; then
    echo "✅ Found $pdf_count PDF files in data/raw/"
else
    echo "⚠️  No PDF files found in data/raw/"
    echo "   Place your PDF documents in data/raw/ for ingestion"
fi

# Check if Qdrant is running
echo "🔍 Checking Qdrant connection..."
if curl -s http://localhost:6333/collections >/dev/null 2>&1; then
    echo "✅ Qdrant is running and accessible"
else
    echo "⚠️  Qdrant not running on localhost:6333"
    echo "   Start with: docker run -p 6333:6333 qdrant/qdrant"
fi

echo ""
echo "🚀 Setup Complete! Next steps:"
echo "=================================="
echo "1. Edit .env and add your GOOGLE_API_KEY"
echo "2. Start Qdrant (if not running):"
echo "   docker run -p 6333:6333 qdrant/qdrant"
echo "3. Test the setup:"
echo "   python test_setup.py"
echo "4. Ingest documents:"
echo "   python scripts/ingest.py --paths data/raw/*.pdf"
echo "5. Check system status:"
echo "   python scripts/status.py"
echo "6. Launch the app:"
echo "   streamlit run app.py"
echo ""
echo "📚 See README.md for detailed usage instructions"
