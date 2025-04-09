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

from config import COLOR_SYSTEM, TYPOGRAPHY
from ui_components import apply_chart_styling, add_insights_annotation

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for backward compatibility - updated for dark theme
WON_COLOR = COLOR_SYSTEM['CHARTS']['WON']  # Bright cyan blue
LOST_COLOR = COLOR_SYSTEM['CHARTS']['LOST']  # Vivid orange
HEATMAP_COLORSCALE_WON = 'Viridis'  # Color-blind friendly
HEATMAP_COLORSCALE_LOST = 'Plasma'  # Color-blind friendly

# Color-blind friendly palette for dark backgrounds
COLORBLIND_PALETTE = {
    'qualitative': [
        COLOR_SYSTEM['ACCENT']['BLUE'],    # Bright blue
        COLOR_SYSTEM['ACCENT']['ORANGE'],  # Orange
        COLOR_SYSTEM['ACCENT']['GREEN'],   # Neon green
        COLOR_SYSTEM['ACCENT']['YELLOW'],  # Yellow
        COLOR_SYSTEM['ACCENT']['PURPLE'],  # Purple
        COLOR_SYSTEM['ACCENT']['RED']      # Red
    ],
    'sequential': ['#084594', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#f7fbff'],
    'diverging': ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4']
}

def create_type_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a pie chart showing the distribution of Won vs Lost bids.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Type' column.
        
    Returns:
        go.Figure: Plotly figure.
    """
    try:
        fig = px.pie(
            df,
            names='Type',
            title='Distribution of Won vs Lost Bids',
            color='Type',
            color_discrete_map={'Won': COLOR_SYSTEM['CHARTS']['WON'], 'Lost': COLOR_SYSTEM['CHARTS']['LOST']},
            hole=0.4
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}',
            textfont=dict(color=COLOR_SYSTEM['BACKGROUND']['MAIN'])  # Black text for contrast on colored sections
        )
        
        # Annotations with counts
        fig.update_layout(
            annotations=[
                dict(text=f"Won: {(df['Type'] == 'Won').sum()} bids", 
                     x=0.5, y=1.1, 
                     font=dict(size=12, color=COLOR_SYSTEM['PRIMARY']['MAIN']),
                     showarrow=False),
                dict(text=f"Lost: {(df['Type'] == 'Lost').sum()} bids", 
                     x=0.5, y=1.05, 
                     font=dict(size=12, color=COLOR_SYSTEM['PRIMARY']['MAIN']),
                     showarrow=False)
            ]
        )
        
        # Apply dark theme styling
        fig = apply_chart_styling(fig, title="Distribution of Won vs Lost Bids", height=400, show_legend=True)
        
        return fig
    except Exception as e:
        logger.error(f"Error in create_type_distribution_chart: {e}", exc_info=True)
        return go.Figure()

def create_cpi_distribution_boxplot(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """
    Create a boxplot comparing CPI distributions between Won and Lost bids.
    
    Args:
        won_data (pd.DataFrame): Won bids.
        lost_data (pd.DataFrame): Lost bids.
        
    Returns:
        go.Figure: Plotly figure.
    """
    try:
        fig = go.Figure()
        
        fig.add_trace(go.Box(
            y=won_data['CPI'],
            name='Won',
            marker_color=COLOR_SYSTEM['CHARTS']['WON'],
            boxmean=True,
            line=dict(width=2),
            jitter=0.3,
            pointpos=-1.8,
            boxpoints='outliers'
        ))
        
        fig.add_trace(go.Box(
            y=lost_data['CPI'],
            name='Lost',
            marker_color=COLOR_SYSTEM['CHARTS']['LOST'],
            boxmean=True,
            line=dict(width=2),
            jitter=0.3,
            pointpos=-1.8,
            boxpoints='outliers'
        ))
        
        # Annotations for means - bright text on dark background
        fig.add_annotation(
            x=0,
            y=won_data['CPI'].mean(),
            text=f"Mean: ${won_data['CPI'].mean():.2f}",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1,
            ax=50, ay=-30,
            font=dict(
                family=TYPOGRAPHY['FONT_FAMILY'], 
                size=11, 
                color=COLOR_SYSTEM['PRIMARY']['MAIN']
            ),
            bgcolor=COLOR_SYSTEM['BACKGROUND']['CARD'],
            bordercolor=COLOR_SYSTEM['NEUTRAL']['LIGHTER'],
            borderwidth=1
        )
        
        fig.add_annotation(
            x=1,
            y=lost_data['CPI'].mean(),
            text=f"Mean: ${lost_data['CPI'].mean():.2f}",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1,
            ax=50, ay=-30,
            font=dict(
                family=TYPOGRAPHY['FONT_FAMILY'], 
                size=11, 
                color=COLOR_SYSTEM['PRIMARY']['MAIN']
            ),
            bgcolor=COLOR_SYSTEM['BACKGROUND']['CARD'],
            bordercolor=COLOR_SYSTEM['NEUTRAL']['LIGHTER'],
            borderwidth=1
        )
        
        # Apply dark theme styling
        fig = apply_chart_styling(fig, title='CPI Distribution: Won vs Lost', height=500)
        fig.update_layout(yaxis_title='CPI ($)', xaxis_title='Bid Type')
        
        # Add context annotation
        fig = add_insights_annotation(
            fig,
            text="This boxplot shows the CPI distribution; boxes represent IQR, the line is the median, and dashed line is the mean.",
            x_pos=0.01, y_pos=0.95, width=220
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error in create_cpi_distribution_boxplot: {e}", exc_info=True)
        return go.Figure()

def create_cpi_histogram_comparison(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """
    Create side-by-side histograms comparing CPI distributions.
    
    Args:
        won_data (pd.DataFrame): Won bids.
        lost_data (pd.DataFrame): Lost bids.
        
    Returns:
        go.Figure: Plotly figure.
    """
    try:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Won Bids CPI Distribution", "Lost Bids CPI Distribution"),
            shared_yaxes=True, shared_xaxes=True
        )
        
        fig.add_trace(
            go.Histogram(
                x=won_data['CPI'],
                name="Won",
                marker_color=COLOR_SYSTEM['CHARTS']['WON'],
                opacity=0.7,
                histnorm='percent',
                hovertemplate='CPI: $%{x:.2f}<br>Percentage: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=lost_data['CPI'],
                name="Lost",
                marker_color=COLOR_SYSTEM['CHARTS']['LOST'],
                opacity=0.7,
                histnorm='percent',
                hovertemplate='CPI: $%{x:.2f}<br>Percentage: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Add mean line indicators
        fig.add_shape(
            type="line",
            x0=won_data['CPI'].mean(), x1=won_data['CPI'].mean(),
            y0=0, y1=1, yref="paper",
            line=dict(color=COLOR_SYSTEM['PRIMARY']['MAIN'], width=2, dash="dash"),
            row=1, col=1
        )
        
        fig.add_shape(
            type="line",
            x0=lost_data['CPI'].mean(), x1=lost_data['CPI'].mean(),
            y0=0, y1=1, yref="paper",
            line=dict(color=COLOR_SYSTEM['PRIMARY']['MAIN'], width=2, dash="dash"),
            row=1, col=2
        )
        
        # Add mean annotations
        fig.add_annotation(
            x=won_data['CPI'].mean(),
            y=0.95,
            text=f"Mean: ${won_data['CPI'].mean():.2f}",
            showarrow=True, arrowhead=2, yref="paper",
            font=dict(family=TYPOGRAPHY['FONT_FAMILY'], size=11, color=COLOR_SYSTEM['PRIMARY']['MAIN']),
            bgcolor=COLOR_SYSTEM['BACKGROUND']['CARD'],
            bordercolor=COLOR_SYSTEM['NEUTRAL']['LIGHTER'],
            row=1, col=1
        )
        
        fig.add_annotation(
            x=lost_data['CPI'].mean(),
            y=0.95,
            text=f"Mean: ${lost_data['CPI'].mean():.2f}",
            showarrow=True, arrowhead=2, yref="paper",
            font=dict(family=TYPOGRAPHY['FONT_FAMILY'], size=11, color=COLOR_SYSTEM['PRIMARY']['MAIN']),
            bgcolor=COLOR_SYSTEM['BACKGROUND']['CARD'],
            bordercolor=COLOR_SYSTEM['NEUTRAL']['LIGHTER'],
            row=1, col=2
        )
        
        # Apply dark theme styling
        fig = apply_chart_styling(fig)
        
        # Update specific layout features
        fig.update_layout(
            title_text="CPI Distribution Comparison (Won vs Lost)",
            height=500,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="CPI ($)")
        fig.update_yaxes(title_text="Percentage of Bids (%)")
        
        return fig
    except Exception as e:
        logger.error(f"Error in create_cpi_histogram_comparison: {e}", exc_info=True)
        return go.Figure()

def create_cpi_efficiency_chart(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """
    Create a scatter plot visualizing CPI efficiency.
    
    Args:
        won_data (pd.DataFrame): Won bids.
        lost_data (pd.DataFrame): Lost bids.
        
    Returns:
        go.Figure: Plotly figure.
    """
    try:
        # Calculate efficiency if not already available
        if 'CPI_Efficiency' not in won_data.columns:
            won_data = won_data.copy()
            won_data['CPI_Efficiency'] = (won_data['IR'] / 100) * (1 / won_data['LOI']) * won_data['Completes']
            
        if 'CPI_Efficiency' not in lost_data.columns:
            lost_data = lost_data.copy()
            lost_data['CPI_Efficiency'] = (lost_data['IR'] / 100) * (1 / lost_data['LOI']) * lost_data['Completes']
        
        fig = go.Figure()
        
        # Won bids scatter
        fig.add_trace(go.Scatter(
            x=won_data['CPI_Efficiency'],
            y=won_data['CPI'],
            mode='markers',
            marker=dict(
                color=COLOR_SYSTEM['CHARTS']['WON'],
                size=10,
                opacity=0.7,
                line=dict(width=1, color=COLOR_SYSTEM['NEUTRAL']['LIGHTEST'])
            ),
            name='Won',
            hovertemplate='<b>Won Bid</b><br>Efficiency: %{x:.1f}<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<br>LOI: %{customdata[1]:.1f} min<br>Completes: %{customdata[2]}<extra></extra>',
            customdata=won_data[['IR', 'LOI', 'Completes']]
        ))
        
        # Lost bids scatter
        fig.add_trace(go.Scatter(
            x=lost_data['CPI_Efficiency'],
            y=lost_data['CPI'],
            mode='markers',
            marker=dict(
                color=COLOR_SYSTEM['CHARTS']['LOST'],
                size=10,
                opacity=0.7,
                line=dict(width=1, color=COLOR_SYSTEM['NEUTRAL']['LIGHTEST'])
            ),
            name='Lost',
            hovertemplate='<b>Lost Bid</b><br>Efficiency: %{x:.1f}<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<br>LOI: %{customdata[1]:.1f} min<br>Completes: %{customdata[2]}<extra></extra>',
            customdata=lost_data[['IR', 'LOI', 'Completes']]
        ))
        
        # Add trend lines
        x_range_won = np.linspace(min(won_data['CPI_Efficiency']), max(won_data['CPI_Efficiency']), 100)
        try:
            coeffs_won = np.polyfit(won_data['CPI_Efficiency'], won_data['CPI'], 1)
            y_trend_won = np.polyval(coeffs_won, x_range_won)
            
            fig.add_trace(go.Scatter(
                x=x_range_won,
                y=y_trend_won,
                mode='lines',
                line=dict(color=COLOR_SYSTEM['CHARTS']['WON'], width=2, dash='dash'),
                name='Won Trend',
                hoverinfo='skip'
            ))
        except:
            logger.warning("Could not calculate trend line for Won data")
        
        x_range_lost = np.linspace(min(lost_data['CPI_Efficiency']), max(lost_data['CPI_Efficiency']), 100)
        try:
            coeffs_lost = np.polyfit(lost_data['CPI_Efficiency'], lost_data['CPI'], 1)
            y_trend_lost = np.polyval(coeffs_lost, x_range_lost)
            
            fig.add_trace(go.Scatter(
                x=x_range_lost,
                y=y_trend_lost,
                mode='lines',
                line=dict(color=COLOR_SYSTEM['CHARTS']['LOST'], width=2, dash='dash'),
                name='Lost Trend',
                hoverinfo='skip'
            ))
        except:
            logger.warning("Could not calculate trend line for Lost data")
        
        # Apply dark theme styling
        fig = apply_chart_styling(
            fig, 
            title="CPI vs Efficiency Metric",
            height=500
        )
        
        # Update axes titles
        fig.update_layout(
            xaxis_title="Efficiency Metric ((IR/100) × (1/LOI) × Completes)",
            yaxis_title="CPI ($)"
        )
        
        # Add explanatory annotation
        fig = add_insights_annotation(
            fig,
            "Higher efficiency values indicate more cost-effective bids. The efficiency metric combines IR, LOI, and sample size.",
            x_pos=0.01, 
            y_pos=0.95,
            width=250
        )
        
        return fig
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
            )
        )
        return fig

def create_heatmap(pivot_data: pd.DataFrame, title: str, colorscale: str = None) -> go.Figure:
    """
    Create a heatmap visualization.
    
    Args:
        pivot_data (pd.DataFrame): Pivot table data to display.
        title (str): Chart title.
        colorscale (str, optional): Name of the colorscale to use.
        
    Returns:
        go.Figure: Plotly figure.
    """
    try:
        if colorscale is None:
            colorscale = HEATMAP_COLORSCALE_WON
            
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale=colorscale,
            colorbar=dict(
                title="CPI ($)",
                titleside="right",
                titlefont=dict(
                    family=TYPOGRAPHY['FONT_FAMILY'],
                    size=12,
                    color=COLOR_SYSTEM['PRIMARY']['MAIN']
                ),
                tickfont=dict(
                    family=TYPOGRAPHY['FONT_FAMILY'],
                    size=10,
                    color=COLOR_SYSTEM['PRIMARY']['MAIN']
                )
            ),
            hovertemplate=(
                "IR: %{y}<br>" +
                "LOI: %{x}<br>" +
                "CPI: $%{z:.2f}<extra></extra>"
            )
        ))
        
        # Apply dark theme styling
        fig = apply_chart_styling(fig, title=title, height=500)
        
        # Update axes
        fig.update_layout(
            xaxis_title="LOI Bin",
            yaxis_title="IR Bin",
        )
        
        # Add explanatory annotation
        fig = add_insights_annotation(
            fig,
            "This heatmap shows CPI values across different IR/LOI combinations. Darker colors indicate higher values.",
            x_pos=0.01,
            y_pos=0.95,
            width=220
        )
        
        return fig
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
            )
        )
        fig.update_layout(title=title)
        return fig
