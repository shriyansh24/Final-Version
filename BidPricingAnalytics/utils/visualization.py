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

# Configure logging
logger = logging.getLogger(__name__)

# Define robust visualization functions that handle errors gracefully
def create_type_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create a pie chart showing the distribution of Won vs Lost bids with error handling."""
    try:
        if len(df) < 5:
            fig = go.Figure()
            fig.add_annotation(text="Not enough data for visualization", x=0.5, y=0.5, showarrow=False)
            return fig
            
        # Clean up data
        df = df.copy()
        if 'Type' not in df.columns:
            logger.error("Type column not found in dataframe")
            fig = go.Figure()
            fig.add_annotation(text="Missing required data", x=0.5, y=0.5, showarrow=False)
            return fig
            
        # Proceed with original implementation
        from .visualization_basic import create_type_distribution_chart as original_func
        return original_func(df)
    except Exception as e:
        logger.error(f"Error in create_type_distribution_chart: {e}", exc_info=True)
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating chart: {str(e)}", x=0.5, y=0.5, showarrow=False)
        return fig

def create_cpi_distribution_boxplot(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """Create a boxplot comparing CPI distribution with error handling."""
    try:
        # Validate input data
        if len(won_data) < 5 or len(lost_data) < 5:
            fig = go.Figure()
            fig.add_annotation(text="Not enough data for visualization", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Check for CPI column and numeric values
        for df in [won_data, lost_data]:
            if 'CPI' not in df.columns:
                fig = go.Figure()
                fig.add_annotation(text="Missing CPI column in data", x=0.5, y=0.5, showarrow=False)
                return fig
                
            # Convert to numeric and drop NaN values
            df['CPI'] = pd.to_numeric(df['CPI'], errors='coerce')
            
        # Proceed with original implementation
        from .visualization_basic import create_cpi_distribution_boxplot as original_func
        return original_func(won_data, lost_data)
    except Exception as e:
        logger.error(f"Error in create_cpi_distribution_boxplot: {e}", exc_info=True)
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating chart: {str(e)}", x=0.5, y=0.5, showarrow=False)
        return fig

def create_cpi_histogram_comparison(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """Create a side-by-side histogram comparison with error handling."""
    try:
        # Validate input data
        if len(won_data) < 5 or len(lost_data) < 5:
            fig = go.Figure()
            fig.add_annotation(text="Not enough data for visualization", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Clean up data
        won_data = won_data.copy()
        lost_data = lost_data.copy()
        
        # Process CPI column
        for df in [won_data, lost_data]:
            if 'CPI' not in df.columns:
                fig = go.Figure()
                fig.add_annotation(text="Missing CPI column in data", x=0.5, y=0.5, showarrow=False)
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
        fig.add_annotation(text=f"Error creating chart: {str(e)}", x=0.5, y=0.5, showarrow=False)
        return fig

def create_cpi_efficiency_chart(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """Create CPI efficiency chart with error handling."""
    try:
        # Validate input data
        if len(won_data) < 5 or len(lost_data) < 5:
            fig = go.Figure()
            fig.add_annotation(text="Not enough data for visualization", x=0.5, y=0.5, showarrow=False)
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
                fig.add_annotation(text=f"Missing columns: {', '.join(missing_cols)}", x=0.5, y=0.5, showarrow=False)
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
        fig.add_annotation(text=f"Error creating chart: {str(e)}", x=0.5, y=0.5, showarrow=False)
        return fig

def create_cpi_vs_ir_scatter(won_data: pd.DataFrame, lost_data: pd.DataFrame, add_trend_line: bool = True) -> go.Figure:
    """Create scatter plot of CPI vs IR with error handling."""
    try:
        # Validate input data
        if len(won_data) < 5 or len(lost_data) < 5:
            fig = go.Figure()
            fig.add_annotation(text="Not enough data for visualization", x=0.5, y=0.5, showarrow=False)
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
                fig.add_annotation(text=f"Missing columns: {', '.join(missing_cols)}", x=0.5, y=0.5, showarrow=False)
                return fig
                
            # Convert to numeric and handle issues
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop NaN values
            df.dropna(subset=required_cols, inplace=True)
            
        # Create a safe trend line function that handles errors
        def safe_trend_line(x, y):
            try:
                # Calculate trend line only if we have enough points
                if len(x) < 3 or len(y) < 3:
                    return None, None
                    
                # Use polynomial fit of degree 1 (linear)
                coeffs = np.polyfit(x, y, 1)
                trend_x = np.linspace(min(x), max(x), 100)
                trend_y = np.polyval(coeffs, trend_x)
                return trend_x, trend_y
            except Exception:
                return None, None
        
        # Create figure
        fig = go.Figure()
        
        # Add Won data
        fig.add_trace(go.Scatter(
            x=won_data['IR'], 
            y=won_data['CPI'], 
            mode='markers',
            marker=dict(
                color=WON_COLOR, 
                size=8, 
                opacity=0.6,
                line=dict(width=1, color='black')
            ),
            name="Won",
            hovertemplate='<b>Won Bid</b><br>IR: %{x:.1f}%<br>CPI: $%{y:.2f}<br>LOI: %{customdata[0]:.1f} min<br>Completes: %{customdata[1]}<extra></extra>',
            customdata=won_data[['LOI', 'Completes']]
        ))
        
        # Add Lost data
        fig.add_trace(go.Scatter(
            x=lost_data['IR'], 
            y=lost_data['CPI'], 
            mode='markers',
            marker=dict(
                color=LOST_COLOR, 
                size=8, 
                opacity=0.6,
                line=dict(width=1, color='black')
            ),
            name="Lost",
            hovertemplate='<b>Lost Bid</b><br>IR: %{x:.1f}%<br>CPI: $%{y:.2f}<br>LOI: %{customdata[0]:.1f} min<br>Completes: %{customdata[1]}<extra></extra>',
            customdata=lost_data[['LOI', 'Completes']]
        ))
        
        # Add trend lines if requested
        if add_trend_line:
            # Won trend line
            trend_x, trend_y = safe_trend_line(won_data['IR'], won_data['CPI'])
            if trend_x is not None and trend_y is not None:
                fig.add_trace(go.Scatter(
                    x=trend_x,
                    y=trend_y,
                    mode='lines',
                    line=dict(color=WON_COLOR, width=2),
                    name='Won Trend',
                    hoverinfo='skip'
                ))
            
            # Lost trend line
            trend_x, trend_y = safe_trend_line(lost_data['IR'], lost_data['CPI'])
            if trend_x is not None and trend_y is not None:
                fig.add_trace(go.Scatter(
                    x=trend_x,
                    y=trend_y,
                    mode='lines',
                    line=dict(color=LOST_COLOR, width=2),
                    name='Lost Trend',
                    hoverinfo='skip'
                ))
        
        # Update layout
        fig.update_layout(
            title_text="CPI vs Incidence Rate (IR) Relationship",
            height=600,
            xaxis_title="Incidence Rate (%)",
            yaxis_title="CPI ($)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error in create_cpi_vs_ir_scatter: {e}", exc_info=True)
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating chart: {str(e)}", x=0.5, y=0.5, showarrow=False)
        return fig

def create_bar_chart_by_bin(won_data: pd.DataFrame, lost_data: pd.DataFrame, bin_column: str, 
                          value_column: str = 'CPI', title: str = None) -> go.Figure:
    """Create a bar chart comparing values across bins with error handling."""
    try:
        # Validate input data
        if len(won_data) < 5 or len(lost_data) < 5:
            fig = go.Figure()
            fig.add_annotation(text="Not enough data for visualization", x=0.5, y=0.5, showarrow=False)
            return fig
            
        # Check required columns
        if bin_column not in won_data.columns or bin_column not in lost_data.columns:
            fig = go.Figure()
            fig.add_annotation(text=f"Missing {bin_column} column in data", x=0.5, y=0.5, showarrow=False)
            return fig
            
        if value_column not in won_data.columns or value_column not in lost_data.columns:
            fig = go.Figure()
            fig.add_annotation(text=f"Missing {value_column} column in data", x=0.5, y=0.5, showarrow=False)
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
        
        # Aggregate data
        won_agg = won_data.groupby(bin_column)[value_column].mean().reset_index()
        lost_agg = lost_data.groupby(bin_column)[value_column].mean().reset_index()
        
        # Need minimum values in each group
        if len(won_agg) < 2 or len(lost_agg) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Not enough data points across bins", x=0.5, y=0.5, showarrow=False)
            return fig
            
        # Create figure
        fig = go.Figure()
        
        # Add Won bars
        fig.add_trace(go.Bar(
            x=won_agg[bin_column],
            y=won_agg[value_column],
            name='Won',
            marker_color=WON_COLOR,
            opacity=0.8,
            text=won_agg[value_column].map('${:.2f}'.format),
            textposition='auto',
            hovertemplate='<b>Won Bids</b><br>%{x}<br>Avg ' + value_column + ': $%{y:.2f}<extra></extra>'
        ))
        
        # Add Lost bars
        fig.add_trace(go.Bar(
            x=lost_agg[bin_column],
            y=lost_agg[value_column],
            name='Lost',
            marker_color=LOST_COLOR,
            opacity=0.8,
            text=lost_agg[value_column].map('${:.2f}'.format),
            textposition='auto',
            hovertemplate='<b>Lost Bids</b><br>%{x}<br>Avg ' + value_column + ': $%{y:.2f}<extra></extra>'
        ))
        
        # Generate automatic title if not provided
        if title is None:
            title = f'Average {value_column} by {bin_column}'
        
        # Determine x-axis title based on bin_column
        if bin_column == 'IR_Bin':
            xaxis_title = 'Incidence Rate Bin (%)'
        elif bin_column == 'LOI_Bin':
            xaxis_title = 'Length of Interview Bin'
        elif bin_column == 'Completes_Bin':
            xaxis_title = 'Sample Size Bin'
        else:
            xaxis_title = bin_column
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=f'Average {value_column} ($)' if value_column == 'CPI' else f'Average {value_column}',
            barmode='group',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            yaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                gridwidth=1,
                tickprefix='$' if value_column == 'CPI' else '',
            ),
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)'
        )
        
        # Add a percentage difference annotation
        for i in range(len(won_agg)):
            bin_name = won_agg[bin_column].iloc[i]
            won_val = won_agg[value_column].iloc[i]
            
            # Find matching lost value
            try:
                lost_val = lost_agg[lost_agg[bin_column] == bin_name][value_column].iloc[0]
                percent_diff = ((lost_val - won_val) / won_val) * 100
                
                if abs(percent_diff) > 5:  # Only annotate significant differences
                    fig.add_annotation(
                        x=bin_name,
                        y=max(won_val, lost_val) * 1.05,
                        text=f"{percent_diff:+.1f}%",
                        showarrow=False,
                        font=dict(
                            size=10,
                            color="black"
                        )
                    )
            except (IndexError, KeyError, ZeroDivisionError):
                pass
        
        return fig
        
    except Exception as e:
        logger.error(f"Error in create_bar_chart_by_bin: {e}", exc_info=True)
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating chart: {str(e)}", x=0.5, y=0.5, showarrow=False)
        return fig

def create_heatmap(pivot_data: pd.DataFrame, title: str, colorscale: str = None) -> go.Figure:
    """Create a heatmap from pivot table data with robust error handling."""
    try:
        # Check if pivot data is empty or all zero
        if pivot_data.empty or (pivot_data == 0).all().all():
            fig = go.Figure()
            fig.add_annotation(text="No data available for heatmap visualization", x=0.5, y=0.5, showarrow=False)
            return fig
            
        # Set default colorscale if not specified
        if colorscale is None:
            colorscale = HEATMAP_COLORSCALE_WON
            
        # Handle potential SVD issues by using a simpler approach to heatmap
        z_values = pivot_data.fillna(0).values
        x_labels = pivot_data.columns.tolist()
        y_labels = pivot_data.index.tolist()
        
        # Create figure directly with go.Heatmap instead of px.imshow to avoid SVD
        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            hovertemplate='IR Bin: %{y}<br>LOI Bin: %{x}<br>Avg CPI: $%{z:.2f}<extra></extra>',
            text=[[f"${val:.2f}" for val in row] for row in z_values],
            texttemplate="%{text}",
            colorbar=dict(
                title="Avg CPI ($)",
                tickprefix="$",
                len=0.75
            )
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            height=600,
            xaxis_title="LOI Bin",
            yaxis_title="IR Bin",
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="black"
            )
        )
        
        # Update xaxis properties
        fig.update_xaxes(
            tickangle=45,
            title_font=dict(size=14),
            title_standoff=25
        )
        
        # Update yaxis properties
        fig.update_yaxes(
            title_font=dict(size=14),
            title_standoff=25
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error in create_heatmap: {e}", exc_info=True)
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating heatmap: {str(e)}", x=0.5, y=0.5, showarrow=False)
        return fig

def create_feature_importance_chart(feature_importance: pd.DataFrame) -> go.Figure:
    """Create feature importance chart with error handling."""
    try:
        # Check if data is valid
        if feature_importance.empty or 'Feature' not in feature_importance.columns or 'Importance' not in feature_importance.columns:
            fig = go.Figure()
            fig.add_annotation(text="No feature importance data available", x=0.5, y=0.5, showarrow=False)
            return fig
            
        # Forward to original function
        from .visualization_prediction import create_feature_importance_chart as original_func
        return original_func(feature_importance)
        
    except Exception as e:
        logger.error(f"Error in create_feature_importance_chart: {e}", exc_info=True)
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating chart: {str(e)}", x=0.5, y=0.5, showarrow=False)
        return fig

def create_prediction_comparison_chart(predictions: dict, won_avg: float, lost_avg: float) -> go.Figure:
    """Create prediction comparison chart with error handling."""
    try:
        # Check if data is valid
        if not predictions:
            fig = go.Figure()
            fig.add_annotation(text="No prediction data available", x=0.5, y=0.5, showarrow=False)
            return fig
            
        # Forward to original function
        from .visualization_prediction import create_prediction_comparison_chart as original_func
        return original_func(predictions, won_avg, lost_avg)
        
    except Exception as e:
        logger.error(f"Error in create_prediction_comparison_chart: {e}", exc_info=True)
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating chart: {str(e)}", x=0.5, y=0.5, showarrow=False)
        return fig