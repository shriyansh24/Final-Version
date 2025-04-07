"""
Basic visualization functionality for the CPI Analysis & Prediction Dashboard.
Contains overview charts and basic visualizations used across the dashboard.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define color-blind friendly palettes
# Using a blue-orange palette which is more distinguishable for most color vision deficiencies
COLORBLIND_PALETTE = {
    'qualitative': ['#3288bd', '#d53e4f', '#66c2a5', '#fee08b', '#e6f598', '#abdda4'],
    'sequential': ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594'],
    'diverging': ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4']
}

# Colors for Won vs Lost (blue-orange contrast)
WON_COLOR = '#3288bd'  # Blue
LOST_COLOR = '#f58518'  # Orange

# Color scales for heatmaps (color-blind friendly)
HEATMAP_COLORSCALE_WON = 'Viridis'  # Good color-blind friendly option for sequential data
HEATMAP_COLORSCALE_LOST = 'Plasma'  # Another good color-blind friendly option

def create_type_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a pie chart showing the distribution of Won vs Lost bids.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'Type' column
    
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        fig = px.pie(
            df, 
            names='Type', 
            title='Distribution of Won vs Lost Bids',
            color='Type',
            color_discrete_map={'Won': WON_COLOR, 'Lost': LOST_COLOR},
            hole=0.4
        )
        
        # Add data labels
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}'
        )
        
        # Add a more descriptive hover tooltip
        fig.update_layout(
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        
        # Add annotations for accessibility
        fig.update_layout(
            annotations=[
                dict(
                    text=f"Won: {(df['Type'] == 'Won').sum()} bids",
                    x=0.5,
                    y=1.1,
                    font_size=12,
                    showarrow=False
                ),
                dict(
                    text=f"Lost: {(df['Type'] == 'Lost').sum()} bids",
                    x=0.5,
                    y=1.05,
                    font_size=12,
                    showarrow=False
                )
            ]
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error in create_type_distribution_chart: {e}", exc_info=True)
        # Return empty figure
        return go.Figure()

def create_cpi_distribution_boxplot(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """
    Create a boxplot comparing CPI distribution between Won and Lost bids.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
    
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        fig = go.Figure()
        
        # Add Won trace
        fig.add_trace(go.Box(
            y=won_data['CPI'],
            name='Won',
            marker_color=WON_COLOR,
            boxmean=True,  # Show mean as a dashed line
            line=dict(width=2),
            jitter=0.3,  # Add some jitter to points for better visualization
            pointpos=-1.8,  # Offset points to the left
            boxpoints='outliers'  # Only show outliers
        ))
        
        # Add Lost trace
        fig.add_trace(go.Box(
            y=lost_data['CPI'],
            name='Lost',
            marker_color=LOST_COLOR,
            boxmean=True,  # Show mean as a dashed line
            line=dict(width=2),
            jitter=0.3,  # Add some jitter to points for better visualization
            pointpos=-1.8,  # Offset points to the left
            boxpoints='outliers'  # Only show outliers
        ))
        
        # Add mean annotations for better accessibility
        fig.add_annotation(
            x=0,  # x-position (Won)
            y=won_data['CPI'].mean(),
            text=f"Mean: ${won_data['CPI'].mean():.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            ax=50,
            ay=-30
        )
        
        fig.add_annotation(
            x=1,  # x-position (Lost)
            y=lost_data['CPI'].mean(),
            text=f"Mean: ${lost_data['CPI'].mean():.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            ax=50,
            ay=-30
        )
        
        # Update layout
        fig.update_layout(
            title='CPI Distribution: Won vs Lost',
            yaxis_title='CPI ($)',
            xaxis_title='Bid Type',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            # Add grid for easier reading of values
            yaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                gridwidth=1
            ),
            # Add hover information template
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            ),
            # Improved accessibility with contrasting colors
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="black"
            )
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error in create_cpi_distribution_boxplot: {e}", exc_info=True)
        # Return empty figure
        return go.Figure()

def create_cpi_histogram_comparison(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """
    Create a side-by-side histogram comparison of CPI distributions.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
    
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        fig = make_subplots(
            rows=1, 
            cols=2, 
            subplot_titles=("Won Bids CPI Distribution", "Lost Bids CPI Distribution"),
            shared_yaxes=True,
            shared_xaxes=True
        )
        
        # Add histograms
        fig.add_trace(
            go.Histogram(
                x=won_data['CPI'], 
                name="Won", 
                marker_color=WON_COLOR, 
                opacity=0.7,
                histnorm='percent',  # Show as percentage for easier comparison
                hovertemplate='CPI: $%{x:.2f}<br>Percentage: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=lost_data['CPI'], 
                name="Lost", 
                marker_color=LOST_COLOR, 
                opacity=0.7,
                histnorm='percent',  # Show as percentage for easier comparison
                hovertemplate='CPI: $%{x:.2f}<br>Percentage: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Add vertical lines for means
        fig.add_shape(
            type="line",
            x0=won_data['CPI'].mean(), x1=won_data['CPI'].mean(),
            y0=0, y1=1,
            yref="paper",
            line=dict(color="black", width=2, dash="dash"),
            row=1, col=1
        )
        
        fig.add_shape(
            type="line",
            x0=lost_data['CPI'].mean(), x1=lost_data['CPI'].mean(),
            y0=0, y1=1,
            yref="paper",
            line=dict(color="black", width=2, dash="dash"),
            row=1, col=2
        )
        
        # Add annotations for mean values
        fig.add_annotation(
            x=won_data['CPI'].mean(),
            y=0.95,
            text=f"Mean: ${won_data['CPI'].mean():.2f}",
            showarrow=True,
            arrowhead=2,
            yref="paper",
            row=1, col=1
        )
        
        fig.add_annotation(
            x=lost_data['CPI'].mean(),
            y=0.95,
            text=f"Mean: ${lost_data['CPI'].mean():.2f}",
            showarrow=True,
            arrowhead=2,
            yref="paper",
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=500,
            showlegend=False,
            title_text="CPI Distribution Comparison (Won vs Lost)",
            # Improved accessibility with contrasting colors
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="black"
            )
        )
        
        fig.update_xaxes(title_text="CPI ($)")
        fig.update_yaxes(title_text="Percentage of Bids (%)")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error in create_cpi_histogram_comparison: {e}", exc_info=True)
        # Return empty figure
        return go.Figure()

def create_cpi_efficiency_chart(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """
    Create a visualization showing CPI efficiency (IR/LOI/Completes).
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
    
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        # Calculate CPI efficiency metric if not already present
        if 'CPI_Efficiency' not in won_data.columns:
            won_data['CPI_Efficiency'] = (won_data['IR'] / 100) * (1 / won_data['LOI']) * won_data['Completes']
        
        if 'CPI_Efficiency' not in lost_data.columns:
            lost_data['CPI_Efficiency'] = (lost_data['IR'] / 100) * (1 / lost_data['LOI']) * lost_data['Completes']
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot for Won data
        fig.add_trace(go.Scatter(
            x=won_data['CPI_Efficiency'],
            y=won_data['CPI'],
            mode='markers',
            marker=dict(
                color=WON_COLOR,
                size=10,
                opacity=0.6,
                line=dict(width=1, color='black')
            ),
            name='Won',
            hovertemplate='<b>Won Bid</b><br>Efficiency: %{x:.1f}<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<br>LOI: %{customdata[1]:.1f} min<br>Completes: %{customdata[2]}<extra></extra>',
            customdata=won_data[['IR', 'LOI', 'Completes']]
        ))
        
        # Add scatter plot for Lost data
        fig.add_trace(go.Scatter(
            x=lost_data['CPI_Efficiency'],
            y=lost_data['CPI'],
            mode='markers',
            marker=dict(
                color=LOST_COLOR,
                size=10,
                opacity=0.6,
                line=dict(width=1, color='black')
            ),
            name='Lost',
            hovertemplate='<b>Lost Bid</b><br>Efficiency: %{x:.1f}<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<br>LOI: %{customdata[1]:.1f} min<br>Completes: %{customdata[2]}<extra></extra>',
            customdata=lost_data[['IR', 'LOI', 'Completes']]
        ))
        
        # Add trend lines
        # Won trend line
        x_range = np.linspace(won_data['CPI_Efficiency'].min(), won_data['CPI_Efficiency'].max(), 100)
        coeffs = np.polyfit(won_data['CPI_Efficiency'], won_data['CPI'], 1)
        trend_y = np.polyval(coeffs, x_range)
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=trend_y,
            mode='lines',
            line=dict(color=WON_COLOR, width=2),
            name='Won Trend',
            hoverinfo='skip'
        ))
        
        # Lost trend line
        x_range = np.linspace(lost_data['CPI_Efficiency'].min(), lost_data['CPI_Efficiency'].max(), 100)
        coeffs = np.polyfit(lost_data['CPI_Efficiency'], lost_data['CPI'], 1)
        trend_y = np.polyval(coeffs, x_range)
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=trend_y,
            mode='lines',
            line=dict(color=LOST_COLOR, width=2),
            name='Lost Trend',
            hoverinfo='skip'
        ))
        
        # Update layout
        fig.update_layout(
            title='CPI vs Efficiency Metric',
            xaxis_title='Efficiency Metric ((IR/100) × (1/LOI) × Completes)',
            yaxis_title='CPI ($)',
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
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
                tickprefix='',  # Add dollar sign to y-axis
            ),
            xaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                gridwidth=1
            )
        )
        
        # Add annotation explaining the efficiency metric
        fig.add_annotation(
            x=0.02,
            y=0.95,
            xref="paper",
            yref="paper",
            text="Efficiency Metric combines IR, LOI, and<br>Sample Size into a single value.<br>Higher values indicate more<br>efficient survey parameters.",
            showarrow=False,
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error in create_cpi_efficiency_chart: {e}", exc_info=True)
        # Return empty figure
        return go.Figure()