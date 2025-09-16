"""
Dashboard utilities for Streamlit evaluation visualization.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List
import streamlit as st


def create_radar_chart(metrics: Dict[str, float], title: str = "Evaluation Metrics") -> go.Figure:
    """
    Create a radar chart for evaluation metrics.
    
    Args:
        metrics: Dictionary of metric names and values
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Prepare data for radar chart
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    # Close the radar chart by adding the first value at the end
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        line=dict(color='rgb(0, 123, 255)', width=2),
        fillcolor='rgba(0, 123, 255, 0.25)',
        name='Metrics'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']
            )
        ),
        showlegend=False,
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=16)
        ),
        height=400
    )
    
    return fig


def create_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a formatted metrics table for display.
    
    Args:
        df: DataFrame with evaluation results
        
    Returns:
        Formatted DataFrame for display
    """
    # Select key columns for display
    display_columns = [
        'question',
        'accuracy',
        'relevance', 
        'completeness',
        'source_quality',
        'overall_score',
        'total_time_s',
        'not_in_corpus'
    ]
    
    # Filter to available columns
    available_columns = [col for col in display_columns if col in df.columns]
    display_df = df[available_columns].copy()
    
    # Format numeric columns
    numeric_columns = ['accuracy', 'relevance', 'completeness', 'source_quality', 'overall_score']
    for col in numeric_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(3)
    
    if 'total_time_s' in display_df.columns:
        display_df['total_time_s'] = display_df['total_time_s'].round(2)
    
    # Truncate questions for display
    if 'question' in display_df.columns:
        display_df['question'] = display_df['question'].str[:80] + "..."
    
    # Rename columns for display
    column_rename = {
        'accuracy': 'Accuracy',
        'relevance': 'Relevance',
        'completeness': 'Completeness', 
        'source_quality': 'Source Quality',
        'overall_score': 'Overall Score',
        'total_time_s': 'Time (s)',
        'not_in_corpus': 'Not in Corpus',
        'question': 'Question'
    }
    
    display_df = display_df.rename(columns=column_rename)
    
    return display_df


def create_performance_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a performance distribution chart.
    
    Args:
        df: DataFrame with evaluation results
        
    Returns:
        Plotly figure object
    """
    # Create histogram of overall scores
    fig = px.histogram(
        df,
        x='overall_score',
        nbins=10,
        title='Distribution of Overall Scores',
        labels={'overall_score': 'Overall Score', 'count': 'Number of Questions'}
    )
    
    fig.update_layout(
        xaxis_title='Overall Score',
        yaxis_title='Count',
        height=300
    )
    
    return fig


def create_category_breakdown(df: pd.DataFrame) -> go.Figure:
    """
    Create a breakdown of performance by category if available.
    
    Args:
        df: DataFrame with evaluation results
        
    Returns:
        Plotly figure object
    """
    if 'category' not in df.columns:
        # Create a simple bar chart of average metrics
        metrics = ['accuracy', 'relevance', 'completeness', 'source_quality']
        available_metrics = [m for m in metrics if m in df.columns]
        
        if not available_metrics:
            return go.Figure()
        
        avg_scores = df[available_metrics].mean()
        
        fig = px.bar(
            x=avg_scores.index,
            y=avg_scores.values,
            title='Average Metric Scores',
            labels={'x': 'Metric', 'y': 'Average Score'}
        )
        
        fig.update_layout(height=300)
        return fig
    
    # Group by category and calculate means
    category_means = df.groupby('category')[['accuracy', 'relevance', 'completeness', 'source_quality', 'overall_score']].mean()
    
    # Create grouped bar chart
    fig = go.Figure()
    
    metrics = ['accuracy', 'relevance', 'completeness', 'source_quality', 'overall_score']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for i, metric in enumerate(metrics):
        if metric in category_means.columns:
            fig.add_trace(go.Bar(
                name=metric.title(),
                x=category_means.index,
                y=category_means[metric],
                marker_color=colors[i % len(colors)]
            ))
    
    fig.update_layout(
        title='Performance by Category',
        xaxis_title='Category',
        yaxis_title='Score',
        barmode='group',
        height=400
    )
    
    return fig


def create_time_analysis(df: pd.DataFrame) -> go.Figure:
    """
    Create analysis of response times.
    
    Args:
        df: DataFrame with evaluation results
        
    Returns:
        Plotly figure object
    """
    time_columns = ['retrieval_time_s', 'generation_time_s', 'judge_time_s']
    available_time_columns = [col for col in time_columns if col in df.columns]
    
    if not available_time_columns:
        # Fallback to total time if available
        if 'total_time_s' in df.columns:
            fig = px.histogram(
                df,
                x='total_time_s',
                nbins=10,
                title='Response Time Distribution',
                labels={'total_time_s': 'Total Time (seconds)', 'count': 'Count'}
            )
            fig.update_layout(height=300)
            return fig
        else:
            return go.Figure()
    
    # Create stacked bar chart of time components
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    labels = ['Retrieval', 'Generation', 'Evaluation']
    
    for i, col in enumerate(available_time_columns):
        label = labels[i] if i < len(labels) else col
        fig.add_trace(go.Bar(
            name=label,
            x=list(range(len(df))),
            y=df[col],
            marker_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        title='Response Time Breakdown',
        xaxis_title='Question Index',
        yaxis_title='Time (seconds)',
        barmode='stack',
        height=300
    )
    
    return fig


def display_evaluation_summary(df: pd.DataFrame) -> None:
    """
    Display evaluation summary statistics.
    
    Args:
        df: DataFrame with evaluation results
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'accuracy' in df.columns:
            avg_accuracy = df['accuracy'].mean()
            st.metric("Avg Accuracy", f"{avg_accuracy:.3f}")
    
    with col2:
        if 'relevance' in df.columns:
            avg_relevance = df['relevance'].mean()
            st.metric("Avg Relevance", f"{avg_relevance:.3f}")
    
    with col3:
        if 'completeness' in df.columns:
            avg_completeness = df['completeness'].mean()
            st.metric("Avg Completeness", f"{avg_completeness:.3f}")
    
    with col4:
        if 'overall_score' in df.columns:
            avg_overall = df['overall_score'].mean()
            st.metric("Avg Overall", f"{avg_overall:.3f}")
    
    # Additional metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        if 'source_quality' in df.columns:
            avg_source_quality = df['source_quality'].mean()
            st.metric("Avg Source Quality", f"{avg_source_quality:.3f}")
    
    with col6:
        if 'total_time_s' in df.columns:
            avg_time = df['total_time_s'].mean()
            st.metric("Avg Response Time", f"{avg_time:.2f}s")
    
    with col7:
        if 'not_in_corpus' in df.columns:
            not_in_corpus_rate = df['not_in_corpus'].mean()
            st.metric("Not in Corpus Rate", f"{not_in_corpus_rate:.1%}")
    
    with col8:
        total_questions = len(df)
        st.metric("Total Questions", total_questions)
