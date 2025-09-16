"""
Streamlit app for Aviation LLM Assistant with Chat and Evaluation tabs.
"""

import streamlit as st
import pandas as pd
import json
import time
from pathlib import Path
import logging
import sys

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from rag.pipeline import RAGPipeline
from rag.settings import settings
from evaluation.datasets import get_evaluation_dataset, get_sample_questions
from evaluation.dashboard import (
    create_radar_chart, create_metrics_table, create_performance_chart,
    create_category_breakdown, create_time_analysis, display_evaluation_summary
)

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Aviation LLM Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    .citation-box {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .debug-info {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.9rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_pipeline():
    """Initialize and cache the RAG pipeline."""
    try:
        pipeline = RAGPipeline()
        return pipeline, None
    except Exception as e:
        return None, str(e)


def display_system_status():
    """Display system status in sidebar."""
    st.sidebar.header("🔧 System Status")
    
    pipeline, error = initialize_pipeline()
    
    if error:
        st.sidebar.error(f"Failed to initialize: {error}")
        return None
    
    try:
        status = pipeline.get_status()
        collection_info = status.get("collection_info", {})
        
        # Collection status
        points_count = collection_info.get("points_count", 0)
        if points_count > 0:
            st.sidebar.success(f"📚 Documents: {points_count:,}")
        else:
            st.sidebar.warning("📚 No documents loaded")
        
        # Embedding info
        embedding_info = status.get("embedding_provider", {})
        st.sidebar.info(f"🔤 Embeddings: {embedding_info.get('backend', 'Unknown')}")
        st.sidebar.info(f"📐 Dimension: {embedding_info.get('dimension', 'Unknown')}")
        
        # Vector store info
        st.sidebar.info(f"🗄️ Collection: {status.get('collection_name', 'Unknown')}")
        
        return pipeline
        
    except Exception as e:
        st.sidebar.error(f"Status error: {e}")
        return None


def chat_tab():
    """Chat assistant tab."""
    st.markdown('<h1 class="main-header">✈️ Aviation LLM Assistant</h1>', unsafe_allow_html=True)
    
    pipeline = display_system_status()
    
    if not pipeline:
        st.error("System not available. Please check configuration and try again.")
        return
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    
    top_k = st.sidebar.slider("Vector Top-K", 1, 20, settings.top_k, 
                             help="Number of candidates from vector search")
    rerank_k = st.sidebar.slider("BM25 Rerank-K", 1, 10, settings.rerank_k,
                                help="Number of final results after BM25 re-ranking")
    debug_mode = st.sidebar.checkbox("Debug Mode", 
                                   help="Show retrieval details and scores")
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show citations if available
            if "citations" in message and message["citations"]:
                with st.expander("📎 Sources"):
                    for i, citation in enumerate(message["citations"], 1):
                        st.markdown(f'<div class="citation-box">', unsafe_allow_html=True)
                        st.markdown(f"**{i}. {citation['source_title']}**")
                        st.markdown(f"*{citation['snippet']}*")
                        if debug_mode:
                            st.markdown(f"Vector: {citation.get('vector_score', 0):.3f}, "
                                      f"BM25: {citation.get('bm25_score', 0):.3f}, "
                                      f"Final: {citation.get('final_score', 0):.3f}")
                        st.markdown('</div>', unsafe_allow_html=True)
            
            # Show debug info if available
            if debug_mode and "debug_info" in message:
                with st.expander("🔍 Debug Information"):
                    debug_info = message["debug_info"]
                    st.markdown('<div class="debug-info">', unsafe_allow_html=True)
                    st.json(debug_info)
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask about airline operations, delays, or aviation topics..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base and generating response..."):
                start_time = time.time()
                
                try:
                    result = pipeline.query(prompt, top_k=top_k, rerank_k=rerank_k)
                    response_time = time.time() - start_time
                    
                    answer = result.get("answer", "I apologize, but I couldn't generate a response.")
                    citations = result.get("citations", [])
                    
                    # Display answer
                    if answer and answer.strip():
                        st.markdown("### 🤖 Response:")
                        st.markdown(answer)
                    else:
                        st.error("Empty or invalid response received")
                    
                    # Show performance metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Response Time", f"{response_time:.2f}s")
                    with col2:
                        st.metric("Sources Found", len(citations))
                    with col3:
                        not_in_corpus = result.get("not_in_corpus", False)
                        status_emoji = "❌" if not_in_corpus else "✅"
                        st.metric("Status", f"{status_emoji} {'Not in corpus' if not_in_corpus else 'Found'}")
                    
                    # Prepare message data
                    message_data = {
                        "role": "assistant",
                        "content": answer,
                        "citations": citations
                    }
                    
                    # Add debug info if debug mode is enabled
                    if debug_mode:
                        debug_info = {
                            "retrieval_time": result.get("retrieval_time_s", 0),
                            "generation_time": result.get("generation_time_s", 0),
                            "total_latency": result.get("latency_s", 0),
                            "token_usage": result.get("token_usage", {}),
                            "retrieval_debug": result.get("retrieval_debug", [])
                        }
                        message_data["debug_info"] = debug_info
                    
                    # Add assistant message
                    st.session_state.messages.append(message_data)
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "I apologize, but I encountered an error processing your question. Please try again."
                    })
    
    # Suggested questions
    if not st.session_state.messages:
        st.markdown("### 💡 Try asking about:")
        suggested_questions = [
            "What are the key differences between BADA 3 and BADA 4 models?",
            "How does Cost Index affect aircraft cruise speed?",
            "What is Maximum Range Cruise and how does it work?",
            "Explain Maximum Endurance Cruise for holding operations",
            "How does aircraft weight affect optimum altitude?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(suggested_questions):
            col = cols[i % 2]
            if col.button(question, key=f"suggest_{i}"):
                # Trigger the question as if user typed it
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()


def evaluation_tab():
    """Evaluation dashboard tab."""
    st.markdown('<h1 class="main-header">📊 Evaluation Dashboard</h1>', unsafe_allow_html=True)
    
    pipeline = display_system_status()
    
    if not pipeline:
        st.error("System not available. Please check configuration and try again.")
        return
    
    # Load existing results if available
    results_file = Path("evaluation/results.json")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### Evaluate RAG System Performance")
    
    with col2:
        sample_size = st.selectbox("Sample Size", [5, 10, "All"], index=0)
    
    with col3:
        run_evaluation = st.button("🚀 Run Evaluation", type="primary")
    
    # Run evaluation if button clicked
    if run_evaluation:
        with st.spinner("Running evaluation..."):
            try:
                # Get questions
                if sample_size == "All":
                    questions = get_evaluation_dataset()
                else:
                    questions = get_sample_questions(sample_size)
                
                st.info(f"Evaluating {len(questions)} questions...")
                
                # Run evaluation
                results_df = pipeline.evaluate(questions)
                
                # Save results
                results_file.parent.mkdir(parents=True, exist_ok=True)
                results_df.to_json(results_file, orient="records", indent=2)
                
                st.success("Evaluation completed!")
                
                # Store in session state
                st.session_state.evaluation_results = results_df
                
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                return
    
    # Load results (from session state or file)
    results_df = None
    
    if "evaluation_results" in st.session_state:
        results_df = st.session_state.evaluation_results
    elif results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results_data = json.load(f)
            results_df = pd.DataFrame(results_data)
            st.info(f"Loaded cached results from {results_file}")
        except Exception as e:
            st.warning(f"Could not load cached results: {e}")
    
    if results_df is not None and not results_df.empty:
        # Display summary metrics
        display_evaluation_summary(results_df)
        
        # Create two columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Radar chart
            st.subheader("📈 Metric Overview")
            
            avg_metrics = {
                "Accuracy": results_df['accuracy'].mean(),
                "Relevance": results_df['relevance'].mean(), 
                "Completeness": results_df['completeness'].mean(),
                "Source Quality": results_df['source_quality'].mean()
            }
            
            if 'rouge_l' in results_df.columns:
                avg_metrics["ROUGE-L"] = results_df['rouge_l'].mean()
            if 'bleu' in results_df.columns:
                avg_metrics["BLEU"] = results_df['bleu'].mean()
            
            radar_fig = create_radar_chart(avg_metrics)
            st.plotly_chart(radar_fig, use_container_width=True)
        
        with col2:
            # Performance distribution
            st.subheader("📊 Score Distribution")
            perf_fig = create_performance_chart(results_df)
            st.plotly_chart(perf_fig, use_container_width=True)
        
        # Additional charts
        col3, col4 = st.columns(2)
        
        with col3:
            # Category breakdown
            st.subheader("🏷️ Performance by Category")
            category_fig = create_category_breakdown(results_df)
            st.plotly_chart(category_fig, use_container_width=True)
        
        with col4:
            # Time analysis
            st.subheader("⏱️ Response Time Analysis")
            time_fig = create_time_analysis(results_df)
            st.plotly_chart(time_fig, use_container_width=True)
        
        # Detailed results table
        st.subheader("📋 Detailed Results")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            score_filter = st.selectbox(
                "Filter by Score",
                ["All", "Excellent (>0.8)", "Good (0.6-0.8)", "Fair (0.4-0.6)", "Poor (<0.4)"]
            )
        
        with col2:
            if 'category' in results_df.columns:
                categories = ["All"] + list(results_df['category'].unique())
                category_filter = st.selectbox("Filter by Category", categories)
            else:
                category_filter = "All"
        
        with col3:
            show_details = st.checkbox("Show Full Details", value=False)
        
        # Apply filters
        filtered_df = results_df.copy()
        
        if score_filter != "All":
            if score_filter == "Excellent (>0.8)":
                filtered_df = filtered_df[filtered_df['overall_score'] > 0.8]
            elif score_filter == "Good (0.6-0.8)":
                filtered_df = filtered_df[(filtered_df['overall_score'] > 0.6) & (filtered_df['overall_score'] <= 0.8)]
            elif score_filter == "Fair (0.4-0.6)":
                filtered_df = filtered_df[(filtered_df['overall_score'] > 0.4) & (filtered_df['overall_score'] <= 0.6)]
            elif score_filter == "Poor (<0.4)":
                filtered_df = filtered_df[filtered_df['overall_score'] <= 0.4]
        
        if category_filter != "All" and 'category' in results_df.columns:
            filtered_df = filtered_df[filtered_df['category'] == category_filter]
        
        # Display table
        if show_details:
            st.dataframe(filtered_df, use_container_width=True, height=400)
        else:
            display_df = create_metrics_table(filtered_df)
            st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="evaluation_results.csv",
            mime="text/csv"
        )
        
    else:
        st.info("No evaluation results available. Click 'Run Evaluation' to start.")
        
        # Show sample questions
        st.subheader("📝 Sample Evaluation Questions")
        sample_questions = get_sample_questions(5)
        for i, q in enumerate(sample_questions, 1):
            with st.expander(f"Question {i}: {q['question'][:80]}..."):
                st.markdown(f"**Category:** {q['category']}")
                st.markdown(f"**Difficulty:** {q['difficulty']}")
                st.markdown(f"**Question:** {q['question']}")
                st.markdown(f"**Reference Answer:** {q['reference_answer']}")


def main():
    """Main application."""
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    # Tab selection
    tab = st.sidebar.radio(
        "Select Tab",
        ["💬 Chat Assistant", "📊 Evaluation Dashboard"],
        index=0
    )
    
    # Main content
    if tab == "💬 Chat Assistant":
        chat_tab()
    elif tab == "📊 Evaluation Dashboard":
        evaluation_tab()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    **Aviation LLM Assistant**
    
    A production-ready RAG system for airline domain questions featuring:
    - 🔍 Hybrid retrieval (Vector + BM25)
    - 🤖 Gemini 2.5 Flash generation
    - 📊 LLM-as-a-Judge evaluation
    - 🗄️ Qdrant vector storage
    
    Built with Streamlit, Docling, and Google AI.
    """)


if __name__ == "__main__":
    main()
