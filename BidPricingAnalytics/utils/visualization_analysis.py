"""
Analysis visualization functionality for the CPI Analysis & Prediction Dashboard.
Contains detailed analysis charts and visualizations for exploring relationships.
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
    from .visualization_basic import WON_COLOR, LOST_COLOR, HEATMAP_COLORSCALE_WON, HEATMAP_COLORSCALE_LOST
except ImportError:
    WON_COLOR = COLOR_SYSTEM['CHARTS']['WON']
    LOST_COLOR = COLOR_SYSTEM['CHARTS']['LOST']
    HEATMAP_COLORSCALE_WON = 'Viridis'
    HEATMAP_COLORSCALE_LOST = 'Plasma'

def create_cpi_vs_ir_scatter(won_data: pd.DataFrame, lost_data: pd.DataFrame, add_trend_line: bool = True) -> go.Figure:
    """
    Create a scatter plot of CPI vs IR with trend lines.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids.
        lost_data (pd.DataFrame): DataFrame of Lost bids.
        add_trend_line (bool): Whether to add trend lines.
        
    Returns:
        go.Figure: Plotly figure.
    """
    try:
        fig = go.Figure()
        
        # Add Won data points
        fig.add_trace(go.Scatter(
            x=won_data['IR'],
            y=won_data['CPI'],
            mode='markers',
            marker=dict(
                color=COLOR_SYSTEM['CHARTS']['WON'],
                size=10, 
                opacity=0.7,
                line=dict(width=1, color=COLOR_SYSTEM['NEUTRAL']['LIGHTEST'])
            ),
            name="Won",
            hovertemplate='<b>Won Bid</b><br>IR: %{x:.1f}%<br>CPI: $%{y:.2f}<br>LOI: %{customdata[0]:.1f} min<br>Completes: %{customdata[1]}<extra></extra>',
            customdata=won_data[['LOI', 'Completes']]
        ))
        
        # Add Lost data points
        fig.add_trace(go.Scatter(
            x=lost_data['IR'],
            y=lost_data['CPI'],
            mode='markers',
            marker=dict(
                color=COLOR_SYSTEM['CHARTS']['LOST'],
                size=10, 
                opacity=0.7,
                line=dict(width=1, color=COLOR_SYSTEM['NEUTRAL']['LIGHTEST'])
            ),
            name="Lost",
            hovertemplate='<b>Lost Bid</b><br>IR: %{x:.1f}%<br>CPI: $%{y:.2f}<br>LOI: %{customdata[0]:.1f} min<br>Completes: %{customdata[1]}<extra></extra>',
            customdata=lost_data[['LOI', 'Completes']]
        ))
        
        # Add trend lines if requested
        if add_trend_line:
            # Won trend line
            x_range_won = np.linspace(won_data['IR'].min(), won_data['IR'].max(), 100)
            coeffs_won = np.polyfit(won_data['IR'], won_data['CPI'], 1)
            y_trend_won = np.polyval(coeffs_won, x_range_won)
            
            fig.add_trace(go.Scatter(
                x=x_range_won,
                y=y_trend_won,
                mode='lines',
                line=dict(color=COLOR_SYSTEM['CHARTS']['WON'], width=2, dash='solid'),
                name='Won Trend',
                hoverinfo='skip'
            ))
            
            # Lost trend line
            x_range_lost = np.linspace(lost_data['IR'].min(), lost_data['IR'].max(), 100)
            coeffs_lost = np.polyfit(lost_data['IR'], lost_data['CPI'], 1)
            y_trend_lost = np.polyval(coeffs_lost, x_range_lost)
            
            fig.add_trace(go.Scatter(
                x=x_range_lost,
                y=y_trend_lost,
                mode='lines',
                line=dict(color=COLOR_SYSTEM['CHARTS']['LOST'], width=2, dash='solid'),
                name='Lost Trend',
                hoverinfo='skip'
            ))
        
        # Apply dark theme styling
        fig = apply_chart_styling(
            fig, 
            title="Relationship Between Incidence Rate (IR) and CPI", 
            height=600
        )
        
        # Update axis titles with more descriptive labels
        fig.update_layout(
            xaxis_title="Incidence Rate (%) - Percentage of people who qualify for survey",
            yaxis_title="Cost Per Interview ($)"
        )
        
        # Add insights annotations
        fig = add_insights_annotation(
            fig,
            "Lower incidence rates typically require higher CPI due to increased screening effort.",
            0.01, 
            0.95, 
            width=220
        )
        
        # Calculate and annotate the convergence point (if any)
        try:
            if add_trend_line and coeffs_won[0] != coeffs_lost[0]:
                # Find where trend lines intersect
                intersection_x = (coeffs_lost[1] - coeffs_won[1]) / (coeffs_won[0] - coeffs_lost[0])
                intersection_y = coeffs_won[0] * intersection_x + coeffs_won[1]
                
                # Only annotate if intersection is within realistic IR range
                if 0 <= intersection_x <= 100:
                    fig = add_data_point_annotation(
                        fig, 
                        intersection_x, 
                        intersection_y,
                        f"Trend lines converge at IR={intersection_x:.1f}%<br>price sensitivity changes",
                        direction="up"
                    )
        except Exception as e:
            logger.warning(f"Could not calculate intersection point: {e}")
        
        # Add median lines
        won_median_cpi = won_data['CPI'].median()
        lost_median_cpi = lost_data['CPI'].median()
        
        fig.add_shape(
            type="line",
            x0=0, 
            y0=won_median_cpi, 
            x1=100, 
            y1=won_median_cpi,
            line=dict(
                color=COLOR_SYSTEM['CHARTS']['WON'], 
                width=1, 
                dash="dot"
            )
        )
        
        fig.add_shape(
            type="line",
            x0=0, 
            y0=lost_median_cpi, 
            x1=100, 
            y1=lost_median_cpi,
            line=dict(
                color=COLOR_SYSTEM['CHARTS']['LOST'], 
                width=1, 
                dash="dot"
            )
        )
        
        # Add annotations for median lines
        fig.add_annotation(
            x=5, 
            y=won_median_cpi,
            text=f"Won Median: ${won_median_cpi:.2f}",
            showarrow=False,
            font=dict(
                family=TYPOGRAPHY['FONT_FAMILY'], 
                size=10, 
                color=COLOR_SYSTEM['CHARTS']['WON']
            ),
            bgcolor=COLOR_SYSTEM['BACKGROUND']['CARD'],
            bordercolor=COLOR_SYSTEM['NEUTRAL']['LIGHT'],
            borderpad=2
        )
        
        fig.add_annotation(
            x=5, 
            y=lost_median_cpi,
            text=f"Lost Median: ${lost_median_cpi:.2f}",
            showarrow=False,
            font=dict(
                family=TYPOGRAPHY['FONT_FAMILY'], 
                size=10, 
                color=COLOR_SYSTEM['CHARTS']['LOST']
            ),
            bgcolor=COLOR_SYSTEM['BACKGROUND']['CARD'],
            bordercolor=COLOR_SYSTEM['NEUTRAL']['LIGHT'],
            borderpad=2
        )
        
        return fig
    
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

def create_bar_chart_by_bin(won_data: pd.DataFrame, lost_data: pd.DataFrame, 
                          bin_column: str, value_column: str = 'CPI', title: str = None) -> go.Figure:
    """
    Create a bar chart comparing a value across bins for Won vs Lost bids.
    
    Args:
        won_data (pd.DataFrame): Won bids.
        lost_data (pd.DataFrame): Lost bids.
        bin_column (str): Column name for bin categories.
        value_column (str): Column to aggregate (default 'CPI').
        title (str, optional): Chart title.
        
    Returns:
        go.Figure: Plotly figure.
    """
    try:
        # Aggregate data by bin
        won_agg = won_data.groupby(bin_column)[value_column].mean().reset_index()
        lost_agg = lost_data.groupby(bin_column)[value_column].mean().reset_index()
        
        fig = go.Figure()
        
        # Add bars for Won data
        fig.add_trace(go.Bar(
            x=won_agg[bin_column],
            y=won_agg[value_column],
            name='Won',
            marker_color=COLOR_SYSTEM['CHARTS']['WON'],
            opacity=0.8,
            text=won_agg[value_column].map('${:.2f}'.format) if value_column == 'CPI' else won_agg[value_column].map('{:.2f}'.format),
            textposition='auto',
            hovertemplate='<b>Won Bids</b><br>%{x}<br>Avg ' + value_column + ': $%{y:.2f}<extra></extra>'
        ))
        
        # Add bars for Lost data
        fig.add_trace(go.Bar(
            x=lost_agg[bin_column],
            y=lost_agg[value_column],
            name='Lost',
            marker_color=COLOR_SYSTEM['CHARTS']['LOST'],
            opacity=0.8,
            text=lost_agg[value_column].map('${:.2f}'.format) if value_column == 'CPI' else lost_agg[value_column].map('{:.2f}'.format),
            textposition='auto',
            hovertemplate='<b>Lost Bids</b><br>%{x}<br>Avg ' + value_column + ': $%{y:.2f}<extra></extra>'
        ))
        
        # Set appropriate title and axis labels
        if title is None:
            title = f'Average {value_column} by {bin_column}'
        
        if bin_column == 'IR_Bin':
            xaxis_title = 'Incidence Rate Bin (%)'
        elif bin_column == 'LOI_Bin':
            xaxis_title = 'Length of Interview Bin'
        elif bin_column == 'Completes_Bin':
            xaxis_title = 'Sample Size Bin'
        else:
            xaxis_title = bin_column
        
        # Apply dark theme styling
        fig = apply_chart_styling(fig, title=title, height=500)
        fig.update_layout(
            xaxis_title=xaxis_title,
            yaxis_title=f'Average {value_column} ($)' if value_column == 'CPI' else f'Average {value_column}',
            barmode='group'
        )
        
        # Add percentage difference annotations for significant differences
        for i in range(len(won_agg)):
            bin_name = won_agg[bin_column].iloc[i]
            won_val = won_agg[value_column].iloc[i]
            
            try:
                lost_val = lost_agg[lost_agg[bin_column] == bin_name][value_column].iloc[0]
                percent_diff = ((lost_val - won_val) / won_val) * 100
                
                if abs(percent_diff) > 10:  # Only show significant differences
                    fig.add_annotation(
                        x=bin_name,
                        y=max(won_val, lost_val) * 1.05,
                        text=f"{percent_diff:+.1f}%",
                        showarrow=False,
                        font=dict(
                            family=TYPOGRAPHY['FONT_FAMILY'], 
                            size=10, 
                            color=COLOR_SYSTEM['PRIMARY']['MAIN']
                        ),
                        bgcolor=COLOR_SYSTEM['BACKGROUND']['CARD'],
                        bordercolor=COLOR_SYSTEM['NEUTRAL']['LIGHT'],
                        borderwidth=1,
                        borderpad=2
                    )
            except (IndexError, KeyError):
                pass
        
        # Add insights annotation
        if value_column == 'CPI':
            bin_type = ("incidence rate" if bin_column == "IR_Bin"
                      else "interview length" if bin_column == "LOI_Bin"
                      else "sample size" if bin_column == "Completes_Bin" 
                      else bin_column)
            
            fig = add_insights_annotation(
                fig,
                f"This chart shows how average CPI varies by {bin_type} between won and lost bids. Larger differences indicate pricing sensitivity.",
                0.01, 
                0.95, 
                width=220
            )
        
        return fig
    
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
    """
    Create a heatmap from pivot table data with robust error handling.
    
    Args:
        pivot_data (pd.DataFrame): Pivot table DataFrame.
        title (str): Chart title.
        colorscale (str, optional): Colorscale to use. Defaults to HEATMAP_COLORSCALE_WON.
        
    Returns:
        go.Figure: Plotly figure.
    """
    try:
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
                ),
                bgcolor=COLOR_SYSTEM['BACKGROUND']['CARD'],
                bordercolor=COLOR_SYSTEM['NEUTRAL']['LIGHT']
            )
            fig = apply_chart_styling(fig, title=title)
            return fig
        
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
        fig = apply_chart_styling(fig, title=title, height=600)
        
        # Update axes
        fig.update_layout(
            xaxis_title="LOI Bin",
            yaxis_title="IR Bin",
        )
        
        # Adjust axis styling
        fig.update_xaxes(tickangle=45, title_standoff=25)
        fig.update_yaxes(title_standoff=25)
        
        # Add explanatory annotation
        fig = add_insights_annotation(
            fig,
            text="This heatmap shows how average CPI varies across different combinations of IR and LOI. Darker colors indicate higher CPI values.",
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
        fig.update_layout(title=title)
        fig = apply_chart_styling(fig)
        return fig
