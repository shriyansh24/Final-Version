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

# Import configuration and UI components
from config import COLOR_SYSTEM, TYPOGRAPHY
from ui_components import apply_chart_styling, add_insights_annotation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants for backward compatibility
WON_COLOR = COLOR_SYSTEM['CHARTS']['WON']
LOST_COLOR = COLOR_SYSTEM['CHARTS']['LOST']
HEATMAP_COLORSCALE_WON = 'Viridis'
HEATMAP_COLORSCALE_LOST = 'Plasma'
COLORBLIND_PALETTE = {
    'qualitative': [COLOR_SYSTEM['ACCENT']['BLUE'], COLOR_SYSTEM['ACCENT']['RED'], 
                    COLOR_SYSTEM['ACCENT']['GREEN'], COLOR_SYSTEM['ACCENT']['YELLOW'], 
                    COLOR_SYSTEM['ACCENT']['PURPLE'], COLOR_SYSTEM['ACCENT']['ORANGE']],
    'sequential': ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594'],
    'diverging': ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4']
}

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
            color_discrete_map={'Won': COLOR_SYSTEM['CHARTS']['WON'], 'Lost': COLOR_SYSTEM['CHARTS']['LOST']},
            hole=0.4
        )
        
        # Add data labels
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}'
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
        
        # Apply consistent styling
        fig = apply_chart_styling(
            fig,
            title="Distribution of Won vs Lost Bids",
            height=400,
            show_legend=True
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
            marker_color=COLOR_SYSTEM['CHARTS']['WON'],
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
            marker_color=COLOR_SYSTEM['CHARTS']['LOST'],
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
            ay=-30,
            font=dict(
                family=TYPOGRAPHY['FONT_FAMILY'],
                size=11,
                color=COLOR_SYSTEM['PRIMARY']['MAIN']
            ),
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
            ay=-30,
            font=dict(
                family=TYPOGRAPHY['FONT_FAMILY'],
                size=11,
                color=COLOR_SYSTEM['PRIMARY']['MAIN']
            ),
        )
        
        # Apply consistent styling
        fig = apply_chart_styling(
            fig,
            title='CPI Distribution: Won vs Lost',
            height=500
        )
        
        # Update axes with more details
        fig.update_layout(
            yaxis_title='CPI ($)',
            xaxis_title='Bid Type',
        )
        
        # Add insights annotation
        fig = add_insights_annotation(
            fig,
            text="This boxplot shows the distribution of CPI values for Won vs Lost bids. The boxes represent the interquartile range (IQR), the line inside the box is the median, and the dashed line is the mean.",
            x_pos=0.01,
            y_pos=0.95,
            width=220
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
                marker_color=COLOR_SYSTEM['CHARTS']['WON'], 
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
                marker_color=COLOR_SYSTEM['CHARTS']['LOST'], 
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
            line=dict(color=COLOR_SYSTEM['PRIMARY']['MAIN'], width=2, dash="dash"),
            row=1, col=1
        )
        
        fig.add_shape(
            type="line",
            x0=lost_data['CPI'].mean(), x1=lost_data['CPI'].mean(),
            y0=0, y1=1,
            yref="paper",
            line=dict(color=COLOR_SYSTEM['PRIMARY']['MAIN'], width=2, dash="dash"),
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
            font=dict(
                family=TYPOGRAPHY['FONT_FAMILY'],
                size=11,
                color=COLOR_SYSTEM['PRIMARY']['MAIN']
            ),
            row=1, col=1
        )
        
        fig.add_annotation(
            x=lost_data['CPI'].mean(),
            y=0.95,
            text=f"Mean: ${lost_data['CPI'].mean():.2f}",
            showarrow=True,
            arrowhead=2,
            yref="paper",
            font=dict(
                family=TYPOGRAPHY['FONT_FAMILY'],
                size=11,
                color=COLOR_SYSTEM['PRIMARY']['MAIN']
            ),
            row=1, col=2
        )
        
        # Apply consistent styling
        fig.update_layout(
            title_text="CPI Distribution Comparison (Won vs Lost)",
            height=500,
            font=dict(
                family=TYPOGRAPHY['FONT_FAMILY'],
                size=12,
                color=COLOR_SYSTEM['PRIMARY']['MAIN']
            ),
            plot_bgcolor=COLOR_SYSTEM['NEUTRAL']['WHITE'],
            paper_bgcolor=COLOR_SYSTEM['NEUTRAL']['WHITE'],
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=False,
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
            won_data = won_data.copy()
            won_data['CPI_Efficiency'] = (won_data['IR'] / 100) * (1 / won_data['LOI']) * won_data['Completes']
        
        if 'CPI_Efficiency' not in lost_data.columns:
            lost_data = lost_data.copy()
            lost_data['CPI_Efficiency'] = (lost_data['IR'] / 100) * (1 / lost_data['LOI']) * lost_data['Completes']
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot for Won data
        fig.add_trace(go.Scatter(
            x=won_data['CPI_Efficiency'],
            y=won_data['CPI'],
            mode='markers',
            marker=dict(
                color=COLOR_SYSTEM['CHARTS']['WON'],
                size=10,
                opacity=0.7,
                line=dict(width=1, color=COLOR_SYSTEM['NEUTRAL']['WHITE'])
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
                color=COLOR_SYSTEM['CHARTS']['LOST'],
                size=10,
                opacity=0.7,
                line=dict(width=1, color=COLOR_SYSTEM['NEUTRAL']['WHITE'])
            ),
            name='Lost',
            hovertemplate='<b>Lost Bid</b><br>Efficiency: %{x:.1f}<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<br>LOI: %{customdata[1]:.1f} min<br>Completes: %{customdata[2]}<extra></extra>',
            customdata=lost_data[['IR', 'LOI', 'Completes']]
        ))
        
        # Add trend lines with robust error handling
        try:
            # Won trend line
            valid_won = won_data.dropna(subset=['CPI_Efficiency', 'CPI'])
            if len(valid_won) > 1:
                x_range = np.linspace(valid_won['CPI_Efficiency'].min(), valid_won['CPI_Efficiency'].max(), 100)
                try:
                    coeffs = np.polyfit(valid_won['CPI_Efficiency'], valid_won['CPI'], 1)
                    trend_y = np.polyval(coeffs, x_range)
                    
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=trend_y,
                        mode='lines',
                        line=dict(color=COLOR_SYSTEM['CHARTS']['WON'], width=2),
                        name='Won Trend',
                        hoverinfo='skip'
                    ))
                except np.linalg.LinAlgError:
                    logger.warning("Could not calculate Won trend line due to numerical issues")
        except Exception as e:
            logger.warning(f"Error adding Won trend line: {e}")

        try:
            # Lost trend line
            valid_lost = lost_data.dropna(subset=['CPI_Efficiency', 'CPI'])
            if len(valid_lost) > 1:
                x_range = np.linspace(valid_lost['CPI_Efficiency'].min(), valid_lost['CPI_Efficiency'].max(), 100)
                try:
                    coeffs = np.polyfit(valid_lost['CPI_Efficiency'], valid_lost['CPI'], 1)
                    trend_y = np.polyval(coeffs, x_range)
                    
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=trend_y,
                        mode='lines',
                        line=dict(color=COLOR_SYSTEM['CHARTS']['LOST'], width=2),
                        name='Lost Trend',
                        hoverinfo='skip'
                    ))
                except np.linalg.LinAlgError:
                    logger.warning("Could not calculate Lost trend line due to numerical issues")
        except Exception as e:
            logger.warning(f"Error adding Lost trend line: {e}")
        
        # Apply consistent styling
        fig = apply_chart_styling(
            fig,
            title='CPI vs Efficiency Metric',
            height=600
        )
        
        # Update axis titles with more descriptive labels
        fig.update_layout(
            xaxis_title='Efficiency Metric ((IR/100) × (1/LOI) × Completes)',
            yaxis_title='CPI ($)',
        )
        
        # Add insights annotation
        fig = add_insights_annotation(
            fig,
            "Efficiency Metric combines IR, LOI, and Sample Size into a single value. Higher values indicate more efficient survey parameters.",
            0.01,
            0.95,
            width=220
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error in create_cpi_efficiency_chart: {e}", exc_info=True)
        # Return empty figure
        return go.Figure()