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

from config import COLOR_SYSTEM, TYPOGRAPHY
from ui_components import apply_chart_styling, add_insights_annotation, add_data_point_annotation

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from .visualization_basic import WON_COLOR, LOST_COLOR
except ImportError:
    WON_COLOR = COLOR_SYSTEM['CHARTS']['WON']
    LOST_COLOR = COLOR_SYSTEM['CHARTS']['LOST']

def create_feature_importance_chart(feature_importance: pd.DataFrame) -> go.Figure:
    """
    Create a horizontal bar chart of feature importance.
    
    Args:
        feature_importance (pd.DataFrame): DataFrame with 'Feature' and 'Importance' columns.
        
    Returns:
        go.Figure: Plotly figure.
    """
    try:
        # Create the bar chart
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance Analysis',
            color='Importance',
            color_continuous_scale='Viridis',
            text=feature_importance['Importance'].map(lambda x: f"{x:.4f}")
        )
        
        # Apply dark theme styling
        fig = apply_chart_styling(fig, title='Feature Importance Analysis', height=500)
        
        # Update axis titles and ordering
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Relative Importance",
            yaxis_title="Feature"
        )
        
        # Update trace styling
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            textposition="outside"
        )
        
        # Add insights annotation
        fig = add_insights_annotation(
            fig,
            "Higher values indicate stronger influence on CPI.",
            0.01, 
            0.95, 
            width=220
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error in create_feature_importance_chart: {e}", exc_info=True)
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            x=0.5, 
            y=0.5, 
            showarrow=False,
            font=dict(
                family=TYPOGRAPHY['FONT_FAMILY'], 
                size=14, 
                color=COLOR_SYSTEM['ACCENT']['RED']
            ),
            bgcolor=COLOR_SYSTEM['BACKGROUND']['CARD'],
            bordercolor=COLOR_SYSTEM['NEUTRAL']['LIGHT']
        )
        fig = apply_chart_styling(fig)
        return fig

def create_prediction_comparison_chart(predictions: dict, won_avg: float, lost_avg: float) -> go.Figure:
    """
    Create a chart comparing model predictions with historical averages.
    
    Args:
        predictions (dict): Dictionary of model predictions.
        won_avg (float): Average CPI for won bids.
        lost_avg (float): Average CPI for lost bids.
        
    Returns:
        go.Figure: Plotly figure.
    """
    try:
        # Prepare data
        models = list(predictions.keys())
        values = list(predictions.values())
        
        # Add average prediction
        avg_prediction = sum(values) / len(values)
        models.append('Average Prediction')
        values.append(avg_prediction)
        
        # Create figure
        fig = go.Figure()
        
        # Add prediction bars
        fig.add_trace(go.Bar(
            x=models,
            y=values,
            marker_color=COLOR_SYSTEM['ACCENT']['BLUE'],
            name='Predictions',
            text=[f"${v:.2f}" for v in values],
            textposition='auto',
            hovertemplate="<b>%{x}</b><br>CPI: $%{y:.2f}<extra></extra>"
        ))
        
        # Add reference lines for won/lost averages
        fig.add_trace(go.Scatter(
            x=[models[0], models[-1]],
            y=[won_avg, won_avg],
            mode='lines',
            line=dict(color=COLOR_SYSTEM['CHARTS']['WON'], width=2, dash='dot'),
            name='Won Avg',
            hovertemplate=f"<b>Won Avg</b>: ${won_avg:.2f}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=[models[0], models[-1]],
            y=[lost_avg, lost_avg],
            mode='lines',
            line=dict(color=COLOR_SYSTEM['CHARTS']['LOST'], width=2, dash='dot'),
            name='Lost Avg',
            hovertemplate=f"<b>Lost Avg</b>: ${lost_avg:.2f}<extra></extra>"
        ))
        
        # Apply dark theme styling
        fig = apply_chart_styling(fig, title='CPI Predictions Comparison', height=500)
        
        # Update layout
        fig.update_layout(
            xaxis_title='Model',
            yaxis_title='Predicted CPI ($)'
        )
        
        # Add annotations for reference lines
        fig = add_data_point_annotation(
            fig, 
            x=models[-1], 
            y=won_avg,
            text=f"Won Avg: ${won_avg:.2f}", 
            direction="up", 
            color=COLOR_SYSTEM['CHARTS']['WON']
        )
        
        fig = add_data_point_annotation(
            fig, 
            x=models[-1], 
            y=lost_avg,
            text=f"Lost Avg: ${lost_avg:.2f}", 
            direction="down", 
            color=COLOR_SYSTEM['CHARTS']['LOST']
        )
        
        # Add insights annotation
        fig = add_insights_annotation(
            fig,
            "Compare model predictions with historical averages. Prices far above 'Lost Avg' may be uncompetitive.",
            0.01, 
            0.95, 
            width=220
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error in create_prediction_comparison_chart: {e}", exc_info=True)
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            x=0.5, 
            y=0.5, 
            showarrow=False,
            font=dict(
                family=TYPOGRAPHY['FONT_FAMILY'], 
                size=14, 
                color=COLOR_SYSTEM['ACCENT']['RED']
            ),
            bgcolor=COLOR_SYSTEM['BACKGROUND']['CARD'],
            bordercolor=COLOR_SYSTEM['NEUTRAL']['LIGHT']
        )
        fig = apply_chart_styling(fig)
        return fig
