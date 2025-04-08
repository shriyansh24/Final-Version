"""
Compatibility module for visualization functions.
This module re-exports visualization functions with added error handling.
"""

# Re-export functions from visualization_basic
from .visualization_basic import (
    WON_COLOR,
    LOST_COLOR,
    HEATMAP_COLORSCALE_WON,
    HEATMAP_COLORSCALE_LOST,
    COLORBLIND_PALETTE
)

# Import necessary libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import streamlit as st
from typing import Dict, List, Optional, Union, Tuple

# Import configuration and UI components
from config import COLOR_SYSTEM, TYPOGRAPHY
from ui_components import apply_chart_styling, add_insights_annotation

# Configure logging
logger = logging.getLogger(__name__)

# Define robust visualization functions that handle errors gracefully
def create_type_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create a pie chart showing the distribution of Won vs Lost bids with error handling."""
    try:
        if len(df) < 5:
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough data for visualization", 
                x=0.5, 
                y=0.5, 
                showarrow=False,
                font=dict(
                    family=TYPOGRAPHY['FONT_FAMILY'],
                    size=14,
                    color=COLOR_SYSTEM['ACCENT']['RED']
                )
            )
            return fig
            
        # Clean up data
        df = df.copy()
        if 'Type' not in df.columns:
            logger.error("Type column not found in dataframe")
            fig = go.Figure()
            fig.add_annotation(
                text="Missing required data", 
                x=0.5, 
                y=0.5, 
                showarrow=False,
                font=dict(
                    family=TYPOGRAPHY['FONT_FAMILY'],
                    size=14,
                    color=COLOR_SYSTEM['ACCENT']['RED']
                )
            )
            return fig
            
        # Proceed with original implementation
        from .visualization_basic import create_type_distribution_chart as original_func
        return original_func(df)
    except Exception as e:
        logger.error(f"Error in create_type_distribution_chart: {e}", exc_info=True)
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
            )
        )
        return fig

def create_cpi_distribution_boxplot(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """Create a boxplot comparing CPI distribution with error handling."""
    try:
        # Validate input data
        if len(won_data) < 5 or len(lost_data) < 5:
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough data for visualization", 
                x=0.5, 
                y=0.5, 
                showarrow=False,
                font=dict(
                    family=TYPOGRAPHY['FONT_FAMILY'],
                    size=14,
                    color=COLOR_SYSTEM['ACCENT']['RED']
                )
            )
            return fig
        
        # Check for CPI column and numeric values
        for df in [won_data, lost_data]:
            if 'CPI' not in df.columns:
                fig = go.Figure()
                fig.add_annotation(
                    text="Missing CPI column in data", 
                    x=0.5, 
                    y=0.5, 
                    showarrow=False,
                    font=dict(
                        family=TYPOGRAPHY['FONT_FAMILY'],
                        size=14,
                        color=COLOR_SYSTEM['ACCENT']['RED']
                    )
                )
                return fig
                
            # Convert to numeric and drop NaN values
            df['CPI'] = pd.to_numeric(df['CPI'], errors='coerce')
            
        # Proceed with original implementation
        from .visualization_basic import create_cpi_distribution_boxplot as original_func
        return original_func(won_data, lost_data)
    except Exception as e:
        logger.error(f"Error in create_cpi_distribution_boxplot: {e}", exc_info=True)
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
            )
        )
        return fig

def create_cpi_histogram_comparison(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """Create a side-by-side histogram comparison with error handling."""
    try:
        # Validate input data
        if len(won_data) < 5 or len(lost_data) < 5:
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough data for visualization", 
                x=0.5, 
                y=0.5, 
                showarrow=False,
                font=dict(
                    family=TYPOGRAPHY['FONT_FAMILY'],
                    size=14,
                    color=COLOR_SYSTEM['ACCENT']['RED']
                )
            )
            return fig
        
        # Clean up data
        won_data = won_data.copy()
        lost_data = lost_data.copy()
        
        # Process CPI column
        for df in [won_data, lost_data]:
            if 'CPI' not in df.columns:
                fig = go.Figure()
                fig.add_annotation(
                    text="Missing CPI column in data", 
                    x=0.5, 
                    y=0.5, 
                    showarrow=False,
                    font=dict(
                        family=TYPOGRAPHY['FONT_FAMILY'],
                        size=14,
                        color=COLOR_SYSTEM['ACCENT']['RED']
                    )
                )
                return fig
                
            # Convert to numeric and handle NaN values
            df['CPI'] = pd.to_numeric(df['CPI'], errors='coerce')
            df = df.dropna(subset=['CPI'])
            
        # Proceed with original implementation
        from .visualization_basic import create_cpi_histogram_comparison as original_func
        return original_func(won_data, lost_data)
    except Exception as e:
        logger.error(f"Error in create_cpi_histogram_comparison: {e}", exc_info=True)
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
            )
        )
        return fig

def create_cpi_efficiency_chart(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """Create CPI efficiency chart with error handling."""
    try:
        # Validate input data
        if len(won_data) < 5 or len(lost_data) < 5:
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough data for visualization", 
                x=0.5, 
                y=0.5, 
                showarrow=False,
                font=dict(
                    family=TYPOGRAPHY['FONT_FAMILY'],
                    size=14,
                    color=COLOR_SYSTEM['ACCENT']['RED']
                )
            )
            return fig
            
        # Clean up data
        won_data = won_data.copy()
        lost_data = lost_data.copy()
        
        # Check required columns
        required_cols = ['CPI', 'IR', 'LOI', 'Completes']
        for df in [won_data, lost_data]:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Missing columns: {', '.join(missing_cols)}", 
                    x=0.5, 
                    y=0.5, 
                    showarrow=False,
                    font=dict(
                        family=TYPOGRAPHY['FONT_FAMILY'],
                        size=14,
                        color=COLOR_SYSTEM['ACCENT']['RED']
                    )
                )
                return fig
                
            # Convert to numeric and handle issues
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Replace zeros with small values to avoid division by zero
            if (df['LOI'] == 0).any():
                df.loc[df['LOI'] == 0, 'LOI'] = 0.1
                
            # Calculate efficiency metric safely
            df['CPI_Efficiency'] = (df['IR'] / 100) * (1 / df['LOI']) * df['Completes']
            
        # Proceed with original implementation
        from .visualization_basic import create_cpi_efficiency_chart as original_func
        return original_func(won_data, lost_data)
    except Exception as e:
        logger.error(f"Error in create_cpi_efficiency_chart: {e}", exc_info=True)
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
            )
        )
        return fig

def create_cpi_vs_ir_scatter(won_data: pd.DataFrame, lost_data: pd.DataFrame, add_trend_line: bool = True) -> go.Figure:
    """Create scatter plot of CPI vs IR with error handling."""
    try:
        # Validate input data
        if len(won_data) < 5 or len(lost_data) < 5:
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough data for visualization", 
                x=0.5, 
                y=0.5, 
                showarrow=False,
                font=dict(
                    family=TYPOGRAPHY['FONT_FAMILY'],
                    size=14,
                    color=COLOR_SYSTEM['ACCENT']['RED']
                )
            )
            return fig
            
        # Clean up data
        won_data = won_data.copy()
        lost_data = lost_data.copy()
        
        # Check required columns
        required_cols = ['CPI', 'IR', 'LOI', 'Completes']
        for df in [won_data, lost_data]:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Missing columns: {', '.join(missing_cols)}", 
                    x=0.5, 
                    y=0.5, 
                    showarrow=False,
                    font=dict(
                        family=TYPOGRAPHY['FONT_FAMILY'],
                        size=14,
                        color=COLOR_SYSTEM['ACCENT']['RED']
                    )
                )
                return fig
                
            # Convert to numeric and handle issues
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop NaN values
            df.dropna(subset=required_cols, inplace=True)
        
        # Proceed with original implementation from visualization_analysis
        from .visualization_analysis import create_cpi_vs_ir_scatter as original_func
        return original_func(won_data, lost_data, add_trend_line)
    except Exception as e:
        logger.error(f"Error in create_cpi_vs_ir_scatter: {e}", exc_info=True)
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
            )
        )
        return fig

def create_bar_chart_by_bin(won_data: pd.DataFrame, lost_data: pd.DataFrame, bin_column: str, 
                          value_column: str = 'CPI', title: str = None) -> go.Figure:
    """Create a bar chart comparing values across bins with error handling."""
    try:
        # Validate input data
        if len(won_data) < 5 or len(lost_data) < 5:
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough data for visualization", 
                x=0.5, 
                y=0.5, 
                showarrow=False,
                font=dict(
                    family=TYPOGRAPHY['FONT_FAMILY'],
                    size=14,
                    color=COLOR_SYSTEM['ACCENT']['RED']
                )
            )
            return fig
            
        # Check required columns
        if bin_column not in won_data.columns or bin_column not in lost_data.columns:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Missing {bin_column} column in data", 
                x=0.5, 
                y=0.5, 
                showarrow=False,
                font=dict(
                    family=TYPOGRAPHY['FONT_FAMILY'],
                    size=14,
                    color=COLOR_SYSTEM['ACCENT']['RED']
                )
            )
            return fig
            
        if value_column not in won_data.columns or value_column not in lost_data.columns:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Missing {value_column} column in data", 
                x=0.5, 
                y=0.5, 
                showarrow=False,
                font=dict(
                    family=TYPOGRAPHY['FONT_FAMILY'],
                    size=14,
                    color=COLOR_SYSTEM['ACCENT']['RED']
                )
            )
            return fig
            
        # Generate aggregated data safely
        won_data = won_data.copy()
        lost_data = lost_data.copy()
        
        # Convert value column to numeric
        won_data[value_column] = pd.to_numeric(won_data[value_column], errors='coerce')
        lost_data[value_column] = pd.to_numeric(lost_data[value_column], errors='coerce')
        
        # Drop NaN values
        won_data = won_data.dropna(subset=[bin_column, value_column])
        lost_data = lost_data.dropna(subset=[bin_column, value_column])
        
        # Need minimum values in each group
        if len(won_data[bin_column].unique()) < 2 or len(lost_data[bin_column].unique()) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough data points across bins", 
                x=0.5, 
                y=0.5, 
                showarrow=False,
                font=dict(
                    family=TYPOGRAPHY['FONT_FAMILY'],
                    size=14,
                    color=COLOR_SYSTEM['ACCENT']['RED']
                )
            )
            return fig
            
        # Proceed with original implementation from visualization_analysis
        from .visualization_analysis import create_bar_chart_by_bin as original_func
        return original_func(won_data, lost_data, bin_column, value_column, title)
        
    except Exception as e:
        logger.error(f"Error in create_bar_chart_by_bin: {e}", exc_info=True)
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
            )
        )
        return fig

def create_heatmap(pivot_data: pd.DataFrame, title: str, colorscale: str = None) -> go.Figure:
    """Create a heatmap from pivot table data with robust error handling."""
    try:
        # Check if pivot data is empty or all zero
        if pivot_data.empty or (pivot_data == 0).all().all():
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for heatmap visualization", 
                x=0.5, 
                y=0.5, 
                showarrow=False,
                font=dict(
                    family=TYPOGRAPHY['FONT_FAMILY'],
                    size=14,
                    color=COLOR_SYSTEM['ACCENT']['RED']
                )
            )
            return fig
        
        # Proceed with original implementation from visualization_analysis
        from .visualization_analysis import create_heatmap as original_func
        return original_func(pivot_data, title, colorscale)
        
    except Exception as e:
        logger.error(f"Error in create_heatmap: {e}", exc_info=True)
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating heatmap: {str(e)}", 
            x=0.5, 
            y=0.5, 
            showarrow=False,
            font=dict(
                family=TYPOGRAPHY['FONT_FAMILY'],
                size=14,
                color=COLOR_SYSTEM['ACCENT']['RED']
            )
        )
        return fig

def create_feature_importance_chart(feature_importance: pd.DataFrame) -> go.Figure:
    """Create feature importance chart with error handling."""
    try:
        # Check if data is valid
        if feature_importance.empty or 'Feature' not in feature_importance.columns or 'Importance' not in feature_importance.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="No feature importance data available", 
                x=0.5, 
                y=0.5, 
                showarrow=False,
                font=dict(
                    family=TYPOGRAPHY['FONT_FAMILY'],
                    size=14,
                    color=COLOR_SYSTEM['ACCENT']['RED']
                )
            )
            return fig
            
        # Forward to enhanced version
        from .visualization_prediction import create_feature_importance_chart as original_func
        
        # Get result from original function
        fig = original_func(feature_importance)
        
        # Apply consistent styling
        fig = apply_chart_styling(
            fig,
            title='Feature Importance Analysis',
            height=500
        )
        
        # Add insights annotation
        fig = add_insights_annotation(
            fig,
            "This chart shows the relative importance of each feature in the prediction model. Features with higher values have stronger influence on the predicted CPI.",
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
            )
        )
        return fig

def create_prediction_comparison_chart(predictions: dict, won_avg: float, lost_avg: float) -> go.Figure:
    """Create prediction comparison chart with error handling."""
    try:
        # Check if data is valid
        if not predictions:
            fig = go.Figure()
            fig.add_annotation(
                text="No prediction data available", 
                x=0.5, 
                y=0.5, 
                showarrow=False,
                font=dict(
                    family=TYPOGRAPHY['FONT_FAMILY'],
                    size=14,
                    color=COLOR_SYSTEM['ACCENT']['RED']
                )
            )
            return fig
            
        # Forward to original enhanced function
        from .visualization_prediction import create_prediction_comparison_chart as original_func
        
        # Get result from original function
        fig = original_func(predictions, won_avg, lost_avg)
        
        # Apply consistent styling
        fig = apply_chart_styling(
            fig,
            title='CPI Predictions Comparison',
            height=500
        )
        
        # Add insights annotation
        fig = add_insights_annotation(
            fig,
            "This chart compares the CPI predictions from different models with the average CPIs for won and lost bids. The closer to 'Won Avg', the more competitive the pricing.",
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
            )
        )
        return fig