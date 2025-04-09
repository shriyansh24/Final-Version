"""
Compatibility module for visualization functions.
This module re-exports visualization functions with added error handling.
"""

from .visualization_basic import (
    WON_COLOR,
    LOST_COLOR,
    HEATMAP_COLORSCALE_WON,
    HEATMAP_COLORSCALE_LOST,
    COLORBLIND_PALETTE
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import streamlit as st
from typing import Dict, List, Optional, Union, Tuple

from config import COLOR_SYSTEM, TYPOGRAPHY
from ui_components import apply_chart_styling, add_insights_annotation

logger = logging.getLogger(__name__)

def create_type_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create a pie chart showing the distribution of Won vs Lost bids with error handling."""
    try:
        if len(df) < 5:
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough data for visualization",
                x=0.5, y=0.5, 
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
            
        df = df.copy()
        if 'Type' not in df.columns:
            logger.error("Type column not found in dataframe")
            fig = go.Figure()
            fig.add_annotation(
                text="Missing required data",
                x=0.5, y=0.5, 
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
            
        from .visualization_basic import create_type_distribution_chart as original_func
        return original_func(df)
        
    except Exception as e:
        logger.error(f"Error in create_type_distribution_chart: {e}", exc_info=True)
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            x=0.5, y=0.5, 
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

def create_cpi_distribution_boxplot(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """Create a boxplot comparing CPI distributions with error handling."""
    try:
        if len(won_data) < 5 or len(lost_data) < 5:
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough data for visualization",
                x=0.5, y=0.5, 
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
            
        for df in [won_data, lost_data]:
            if 'CPI' not in df.columns:
                fig = go.Figure()
                fig.add_annotation(
                    text="Missing CPI column in data",
                    x=0.5, y=0.5, 
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
                
            df['CPI'] = pd.to_numeric(df['CPI'], errors='coerce')
            
        from .visualization_basic import create_cpi_distribution_boxplot as original_func
        return original_func(won_data, lost_data)
        
    except Exception as e:
        logger.error(f"Error in create_cpi_distribution_boxplot: {e}", exc_info=True)
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            x=0.5, y=0.5, 
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

def create_cpi_histogram_comparison(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """Create a side-by-side histogram comparison with error handling."""
    try:
        if len(won_data) < 5 or len(lost_data) < 5:
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough data for visualization",
                x=0.5, y=0.5, 
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
            
        won_data = won_data.copy()
        lost_data = lost_data.copy()
        
        for df in [won_data, lost_data]:
            if 'CPI' not in df.columns:
                fig = go.Figure()
                fig.add_annotation(
                    text="Missing CPI column in data",
                    x=0.5, y=0.5, 
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
                
            df['CPI'] = pd.to_numeric(df['CPI'], errors='coerce')
            df = df.dropna(subset=['CPI'])
            
        from .visualization_basic import create_cpi_histogram_comparison as original_func
        return original_func(won_data, lost_data)
        
    except Exception as e:
        logger.error(f"Error in create_cpi_histogram_comparison: {e}", exc_info=True)
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            x=0.5, y=0.5, 
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

def create_cpi_efficiency_chart(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """Create CPI efficiency chart with error handling."""
    try:
        if len(won_data) < 5 or len(lost_data) < 5:
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough data for visualization",
                x=0.5, y=0.5, 
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
            
        won_data = won_data.copy()
        lost_data = lost_data.copy()
        
        required_cols = ['CPI', 'IR', 'LOI', 'Completes']
        for df in [won_data, lost_data]:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Missing columns: {', '.join(missing_cols)}",
                    x=0.5, y=0.5, 
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
                
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            if (df['LOI'] == 0).any():
                df.loc[df['LOI'] == 0, 'LOI'] = 0.1
                
            df['CPI_Efficiency'] = (df['IR'] / 100) * (1 / df['LOI']) * df['Completes']
            
        from .visualization_basic import create_cpi_efficiency_chart as original_func
        return original_func(won_data, lost_data)
        
    except Exception as e:
        logger.error(f"Error in create_cpi_efficiency_chart: {e}", exc_info=True)
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            x=0.5, y=0.5, 
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

def create_cpi_vs_ir_scatter(won_data: pd.DataFrame, lost_data: pd.DataFrame, add_trend_line: bool = True) -> go.Figure:
    """Create scatter plot of CPI vs IR with error handling."""
    try:
        if len(won_data) < 5 or len(lost_data) < 5:
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough data for visualization",
                x=0.5, y=0.5, 
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
            
        won_data = won_data.copy()
        lost_data = lost_data.copy()
        
        required_cols = ['CPI', 'IR', 'LOI', 'Completes']
        for df in [won_data, lost_data]:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Missing columns: {', '.join(missing_cols)}",
                    x=0.5, y=0.5, 
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
                
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            df.dropna(subset=required_cols, inplace=True)
            
        from .visualization_analysis import create_cpi_vs_ir_scatter as original_func
        return original_func(won_data, lost_data, add_trend_line)
        
    except Exception as e:
        logger.error(f"Error in create_cpi_vs_ir_scatter: {e}", exc_info=True)
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            x=0.5, y=0.5, 
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

def create_bar_chart_by_bin(won_data: pd.DataFrame, lost_data: pd.DataFrame,
                          bin_column: str, value_column: str = 'CPI', title: str = None) -> go.Figure:
    """
    Create a bar chart comparing a value across bins between Won and Lost bids.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids.
        lost_data (pd.DataFrame): DataFrame of Lost bids.
        bin_column (str): Column with bin categories.
        value_column (str, optional): Column to aggregate. Defaults to 'CPI'.
        title (str, optional): Chart title.
        
    Returns:
        go.Figure: Plotly figure object.
    """
    try:
        if len(won_data) < 5 or len(lost_data) < 5:
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough data for visualization",
                x=0.5, y=0.5, 
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
        
        for df in [won_data, lost_data]:
            if bin_column not in df.columns or value_column not in df.columns:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Missing column(s) in data: {bin_column} or {value_column}",
                    x=0.5, y=0.5, 
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
                
            df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
            df = df.dropna(subset=[bin_column, value_column])
        
        from .visualization_analysis import create_bar_chart_by_bin as original_func
        return original_func(won_data, lost_data, bin_column, value_column, title)
        
    except Exception as e:
        logger.error(f"Error in create_bar_chart_by_bin: {e}", exc_info=True)
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            x=0.5, y=0.5, 
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

def create_heatmap(pivot_data: pd.DataFrame, title: str, colorscale: str = None) -> go.Figure:
    """
    Create a heatmap from pivot table data with robust error handling.
    
    Args:
        pivot_data (pd.DataFrame): Pivot table DataFrame.
        title (str): Chart title.
        colorscale (str, optional): Colorscale to use. Defaults to HEATMAP_COLORSCALE_WON.
        
    Returns:
        go.Figure: Plotly figure object.
    """
    try:
        if pivot_data.empty or (pivot_data == 0).all().all():
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for heatmap visualization",
                x=0.5, y=0.5, 
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
            
        if colorscale is None:
            colorscale = HEATMAP_COLORSCALE_WON
            
        from .visualization_basic import create_heatmap as original_func
        return original_func(pivot_data, title, colorscale)
        
    except Exception as e:
        logger.error(f"Error in create_heatmap: {e}", exc_info=True)
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating heatmap: {str(e)}",
            x=0.5, y=0.5, 
            showarrow=False,
            font=dict(
                family=TYPOGRAPHY['FONT_FAMILY'], 
                size=14, 
                color=COLOR_SYSTEM['ACCENT']['RED']
            ),
            bgcolor=COLOR_SYSTEM['BACKGROUND']['CARD'],
            bordercolor=COLOR_SYSTEM['NEUTRAL']['LIGHT']
        )
        fig.update_layout(title=title)
        fig = apply_chart_styling(fig)
        return fig

def create_feature_importance_chart(feature_importance: pd.DataFrame) -> go.Figure:
    """
    Create a horizontal bar chart of feature importance.
    
    Args:
        feature_importance (pd.DataFrame): DataFrame with 'Feature' and 'Importance' columns.
        
    Returns:
        go.Figure: Plotly figure.
    """
    try:
        from .visualization_prediction import create_feature_importance_chart as original_func
        return original_func(feature_importance)
    except Exception as e:
        logger.error(f"Error in create_feature_importance_chart: {e}", exc_info=True)
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating feature importance chart: {str(e)}",
            x=0.5, y=0.5, 
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
        from .visualization_prediction import create_prediction_comparison_chart as original_func
        return original_func(predictions, won_avg, lost_avg)
    except Exception as e:
        logger.error(f"Error in create_prediction_comparison_chart: {e}", exc_info=True)
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating prediction comparison chart: {str(e)}",
            x=0.5, y=0.5, 
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
