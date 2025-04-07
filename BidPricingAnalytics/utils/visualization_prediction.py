"""
Prediction visualization functionality for the CPI Analysis & Prediction Dashboard.
Contains model visualization and prediction-related charts.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging
from typing import Dict, List, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import constants from visualization_basic if available, or define here
try:
    from .visualization_basic import WON_COLOR, LOST_COLOR
except ImportError:
    # Define color constants if import fails
    WON_COLOR = '#3288bd'  # Blue
    LOST_COLOR = '#f58518'  # Orange

def create_feature_importance_chart(feature_importance: pd.DataFrame) -> go.Figure:
    """
    Create a horizontal bar chart of feature importance.
    
    Args:
        feature_importance (pd.DataFrame): DataFrame with Feature and Importance columns
    
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        # Create horizontal bar chart
        fig = px.bar(
            feature_importance, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            title='Feature Importance Analysis',
            color='Importance',
            color_continuous_scale='Viridis',  # Color-blind friendly
            text=feature_importance['Importance'].map(lambda x: f"{x:.4f}")
        )
        
        # Update layout
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Relative Importance",
            yaxis_title="Feature",
            height=500,
            # Improved accessibility
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="black"
            )
        )
        
        # Add hover template
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            textposition="outside"
        )
        
        # Add annotation explaining feature importance
        fig.add_annotation(
            x=feature_importance['Importance'].max() * 0.95,
            y=0,
            text="Higher values indicate<br>stronger influence on CPI",
            showarrow=False,
            align="right",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error in create_feature_importance_chart: {e}", exc_info=True)
        # Return empty figure
        return go.Figure()

def create_prediction_comparison_chart(predictions: dict, won_avg: float, lost_avg: float) -> go.Figure:
    """
    Create a chart comparing model predictions with won/lost averages.
    
    Args:
        predictions (dict): Dictionary of model predictions
        won_avg (float): Average CPI for won bids
        lost_avg (float): Average CPI for lost bids
    
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        # Prepare data
        models = list(predictions.keys())
        values = list(predictions.values())
        
        # Add average prediction
        avg_prediction = sum(values) / len(values)
        models.append('Average Prediction')
        values.append(avg_prediction)
        
        # Add reference values
        reference_models = ['Won Avg', 'Lost Avg']
        reference_values = [won_avg, lost_avg]
        
        # Create figure
        fig = go.Figure()
        
        # Add prediction bars
        fig.add_trace(go.Bar(
            x=models,
            y=values,
            marker_color='#4292c6',  # Blue
            name='Predictions',
            text=[f"${v:.2f}" for v in values],
            textposition='auto',
            hovertemplate="<b>%{x}</b><br>CPI: $%{y:.2f}<extra></extra>"
        ))
        
        # Add reference lines
        fig.add_trace(go.Scatter(
            x=[models[0], models[-1]],
            y=[won_avg, won_avg],
            mode='lines',
            line=dict(color=WON_COLOR, width=2, dash='dot'),
            name='Won Avg',
            hovertemplate=f"Won Avg: ${won_avg:.2f}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=[models[0], models[-1]],
            y=[lost_avg, lost_avg],
            mode='lines',
            line=dict(color=LOST_COLOR, width=2, dash='dot'),
            name='Lost Avg',
            hovertemplate=f"Lost Avg: ${lost_avg:.2f}<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title='CPI Predictions Comparison',
            xaxis_title='Model',
            yaxis_title='Predicted CPI ($)',
            height=500,
            # Improved accessibility
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="black"
            ),
            yaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                gridwidth=1,
                tickprefix='', # Add dollar sign to y-axis
            )
        )
        
        # Add annotations for won/lost avg
        fig.add_annotation(
            x=models[-1],
            y=won_avg,
            text=f"Won Avg: ${won_avg:.2f}",
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=20,
            font=dict(color=WON_COLOR)
        )
        
        fig.add_annotation(
            x=models[-1],
            y=lost_avg,
            text=f"Lost Avg: ${lost_avg:.2f}",
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=-20,
            font=dict(color=LOST_COLOR)
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error in create_prediction_comparison_chart: {e}", exc_info=True)
        # Return empty figure
        return go.Figure()