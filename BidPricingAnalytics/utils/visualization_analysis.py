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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import constants from visualization_basic if available, or define here
try:
    from .visualization_basic import WON_COLOR, LOST_COLOR, HEATMAP_COLORSCALE_WON, HEATMAP_COLORSCALE_LOST
except ImportError:
    # Define color constants if import fails
    WON_COLOR = '#3288bd'  # Blue
    LOST_COLOR = '#f58518'  # Orange
    HEATMAP_COLORSCALE_WON = 'Viridis'
    HEATMAP_COLORSCALE_LOST = 'Plasma'

def create_cpi_vs_ir_scatter(won_data: pd.DataFrame, lost_data: pd.DataFrame, add_trend_line: bool = True) -> go.Figure:
    """
    Create a scatter plot of CPI vs IR with trend lines.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
        add_trend_line (bool, optional): Whether to add trend lines. Defaults to True.
    
    Returns:
        go.Figure: Plotly figure object
    """
    try:
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
                line=dict(width=1, color='black')  # Add border for better visibility
            ),
            name="Won",
            hovertemplate='<b>Won Bid</b><br>IR: %{x:.1f}%<br>CPI: $%{y:.2f}<br>LOI: %{customdata[0]:.1f} min<br>Completes: %{customdata[1]}<extra></extra>',
            customdata=won_data[['LOI', 'Completes']]  # Include additional data for hover
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
                line=dict(width=1, color='black')  # Add border for better visibility
            ),
            name="Lost",
            hovertemplate='<b>Lost Bid</b><br>IR: %{x:.1f}%<br>CPI: $%{y:.2f}<br>LOI: %{customdata[0]:.1f} min<br>Completes: %{customdata[1]}<extra></extra>',
            customdata=lost_data[['LOI', 'Completes']]  # Include additional data for hover
        ))
        
        # Add trend lines
        if add_trend_line:
            # Add a trend line for Won bids
            won_trend = go.Scatter(
                x=won_data['IR'],
                y=won_data['IR'].map(lambda x: 
                    np.polyval(np.polyfit(won_data['IR'], won_data['CPI'], 1), x)
                ),
                mode='lines',
                line=dict(color=WON_COLOR, width=2, dash='solid'),
                name='Won Trend',
                hoverinfo='skip'
            )
            fig.add_trace(won_trend)
            
            # Add a trend line for Lost bids
            lost_trend = go.Scatter(
                x=lost_data['IR'],
                y=lost_data['IR'].map(lambda x: 
                    np.polyval(np.polyfit(lost_data['IR'], lost_data['CPI'], 1), x)
                ),
                mode='lines',
                line=dict(color=LOST_COLOR, width=2, dash='solid'),
                name='Lost Trend',
                hoverinfo='skip'
            )
            fig.add_trace(lost_trend)
        
        # Update layout with improved accessibility
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
            # Add grid for easier reading of values
            xaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                gridwidth=1,
                tickformat='.0f'  # No decimal places for IR
            ),
            yaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                gridwidth=1,
                tickprefix='$',  # Add dollar sign to y-axis
            ),
            # Improved accessibility with contrasting colors
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="black"
            ),
            # Add hover information template
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        
        # Add annotations for context
        fig.add_annotation(
            x=won_data['IR'].min() + (won_data['IR'].max() - won_data['IR'].min()) * 0.05,
            y=won_data['CPI'].max() - (won_data['CPI'].max() - won_data['CPI'].min()) * 0.05,
            text="Lower IR typically requires<br>higher CPI due to<br>increased difficulty finding<br>qualified respondents",
            showarrow=False,
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error in create_cpi_vs_ir_scatter: {e}", exc_info=True)
        # Return empty figure
        return go.Figure()

def create_bar_chart_by_bin(won_data: pd.DataFrame, lost_data: pd.DataFrame, bin_column: str, 
                          value_column: str = 'CPI', title: str = None) -> go.Figure:
    """
    Create a bar chart comparing a value across bins between Won and Lost bids.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
        bin_column (str): Column name containing bin categories
        value_column (str, optional): Column to aggregate. Defaults to 'CPI'.
        title (str, optional): Chart title. Defaults to auto-generated title.
    
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        # Generate aggregated data
        won_agg = won_data.groupby(bin_column)[value_column].mean().reset_index()
        lost_agg = lost_data.groupby(bin_column)[value_column].mean().reset_index()
        
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
        
        # Update layout with improved accessibility
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
            # Add grid for easier reading of values
            yaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                gridwidth=1,
                tickprefix='$' if value_column == 'CPI' else '',  # Add dollar sign if CPI
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
        
        # Add a percentage difference annotation
        for i in range(len(won_agg)):
            bin_name = won_agg[bin_column].iloc[i]
            won_val = won_agg[value_column].iloc[i]
            
            # Find matching lost value
            try:
                lost_val = lost_agg[lost_agg[bin_column] == bin_name][value_column].iloc[0]
                percent_diff = ((lost_val - won_val) / won_val) * 100
                
                if abs(percent_diff) > 10:  # Only annotate significant differences
                    fig.add_annotation(
                        x=bin_name,
                        y=max(won_val, lost_val) + 1,
                        text=f"{percent_diff:+.1f}%",
                        showarrow=False,
                        font=dict(
                            size=10,
                            color="black"
                        )
                    )
            except:
                pass
        
        return fig
    
    except Exception as e:
        logger.error(f"Error in create_bar_chart_by_bin: {e}", exc_info=True)
        # Return empty figure
        return go.Figure()

def create_heatmap(pivot_data: pd.DataFrame, title: str, colorscale: str = None) -> go.Figure:
    """
    Create a heatmap from pivot table data.
    
    Args:
        pivot_data (pd.DataFrame): Pivot table DataFrame
        title (str): Chart title
        colorscale (str, optional): Colorscale to use. Defaults to HEATMAP_COLORSCALE_WON.
    
    Returns:
        go.Figure: Plotly figure object
    """
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
        # Return a simple empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating heatmap: {str(e)}",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(color="red", size=14)
        )
        fig.update_layout(title=title)
        return fig