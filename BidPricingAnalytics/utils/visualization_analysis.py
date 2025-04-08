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

# Import configuration and UI components
from config import COLOR_SYSTEM, TYPOGRAPHY
from ui_components import apply_chart_styling, add_insights_annotation, add_data_point_annotation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import constants from visualization_basic if available, or define here
try:
    from .visualization_basic import WON_COLOR, LOST_COLOR, HEATMAP_COLORSCALE_WON, HEATMAP_COLORSCALE_LOST
except ImportError:
    # Define color constants if import fails
    WON_COLOR = COLOR_SYSTEM['CHARTS']['WON']
    LOST_COLOR = COLOR_SYSTEM['CHARTS']['LOST']
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
                color=COLOR_SYSTEM['CHARTS']['WON'], 
                size=10, 
                opacity=0.7,
                line=dict(width=1, color=COLOR_SYSTEM['NEUTRAL']['WHITE'])
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
                color=COLOR_SYSTEM['CHARTS']['LOST'], 
                size=10, 
                opacity=0.7,
                line=dict(width=1, color=COLOR_SYSTEM['NEUTRAL']['WHITE'])
            ),
            name="Lost",
            hovertemplate='<b>Lost Bid</b><br>IR: %{x:.1f}%<br>CPI: $%{y:.2f}<br>LOI: %{customdata[0]:.1f} min<br>Completes: %{customdata[1]}<extra></extra>',
            customdata=lost_data[['LOI', 'Completes']]  # Include additional data for hover
        ))
        
        # Add trend lines
        if add_trend_line:
            # Add a trend line for Won bids
            x_range = np.linspace(min(won_data['IR']), max(won_data['IR']), 100)
            coeffs = np.polyfit(won_data['IR'], won_data['CPI'], 1)
            y_trend = np.polyval(coeffs, x_range)
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_trend,
                mode='lines',
                line=dict(color=COLOR_SYSTEM['CHARTS']['WON'], width=2, dash='solid'),
                name='Won Trend',
                hoverinfo='skip'
            ))
            
            # Add a trend line for Lost bids
            x_range = np.linspace(min(lost_data['IR']), max(lost_data['IR']), 100)
            coeffs = np.polyfit(lost_data['IR'], lost_data['CPI'], 1)
            y_trend = np.polyval(coeffs, x_range)
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_trend,
                mode='lines',
                line=dict(color=COLOR_SYSTEM['CHARTS']['LOST'], width=2, dash='solid'),
                name='Lost Trend',
                hoverinfo='skip'
            ))
        
        # Apply consistent styling
        fig = apply_chart_styling(
            fig,
            title="Relationship Between Incidence Rate (IR) and CPI",
            height=600
        )
        
        # Update axis titles with more descriptive labels
        fig.update_layout(
            xaxis_title="Incidence Rate (%) - Percentage of people who qualify for survey",
            yaxis_title="Cost Per Interview ($)",
        )
        
        # Add insights annotations
        fig = add_insights_annotation(
            fig,
            "Lower incidence rates typically require higher CPI due to increased screening effort to find qualified respondents.",
            0.01,
            0.95,
            width=220
        )
        
        # Calculate and annotate the convergence point (if any)
        try:
            # Find where trend lines intersect
            won_coeffs = np.polyfit(won_data['IR'], won_data['CPI'], 1)
            lost_coeffs = np.polyfit(lost_data['IR'], lost_data['CPI'], 1)
            
            # Solve for intersection: m1*x + b1 = m2*x + b2
            # x = (b2 - b1) / (m1 - m2)
            if won_coeffs[0] != lost_coeffs[0]:  # Ensure slopes are different
                intersection_x = (lost_coeffs[1] - won_coeffs[1]) / (won_coeffs[0] - lost_coeffs[0])
                intersection_y = won_coeffs[0] * intersection_x + won_coeffs[1]
                
                # Only annotate if intersection is within realistic IR range
                if 0 <= intersection_x <= 100:
                    fig = add_data_point_annotation(
                        fig,
                        intersection_x,
                        intersection_y,
                        f"Trend lines converge at IR={intersection_x:.1f}%<br>suggesting price sensitivity<br>changes at this threshold",
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
                dash="dot",
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
                dash="dot",
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
            bgcolor="rgba(255, 255, 255, 0.8)",
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
            bgcolor="rgba(255, 255, 255, 0.8)",
            borderpad=2
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
            marker_color=COLOR_SYSTEM['CHARTS']['WON'],
            opacity=0.8,
            text=won_agg[value_column].map('${:.2f}'.format) if value_column == 'CPI' else won_agg[value_column].map('{:.2f}'.format),
            textposition='auto',
            hovertemplate='<b>Won Bids</b><br>%{x}<br>Avg ' + value_column + ': $%{y:.2f}<extra></extra>' if value_column == 'CPI' else 
                          '<b>Won Bids</b><br>%{x}<br>Avg ' + value_column + ': %{y:.2f}<extra></extra>'
        ))
        
        # Add Lost bars
        fig.add_trace(go.Bar(
            x=lost_agg[bin_column],
            y=lost_agg[value_column],
            name='Lost',
            marker_color=COLOR_SYSTEM['CHARTS']['LOST'],
            opacity=0.8,
            text=lost_agg[value_column].map('${:.2f}'.format) if value_column == 'CPI' else lost_agg[value_column].map('{:.2f}'.format),
            textposition='auto',
            hovertemplate='<b>Lost Bids</b><br>%{x}<br>Avg ' + value_column + ': $%{y:.2f}<extra></extra>' if value_column == 'CPI' else
                          '<b>Lost Bids</b><br>%{x}<br>Avg ' + value_column + ': %{y:.2f}<extra></extra>'
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
        
        # Apply consistent styling
        fig = apply_chart_styling(
            fig,
            title=title,
            height=500
        )
        
        # Update layout with custom axes
        fig.update_layout(
            xaxis_title=xaxis_title,
            yaxis_title=f'Average {value_column} ($)' if value_column == 'CPI' else f'Average {value_column}',
            barmode='group',
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
                        y=max(won_val, lost_val) * 1.05,
                        text=f"{percent_diff:+.1f}%",
                        showarrow=False,
                        font=dict(
                            family=TYPOGRAPHY['FONT_FAMILY'],
                            size=10,
                            color=COLOR_SYSTEM['PRIMARY']['MAIN']
                        ),
                        bgcolor="rgba(255, 255, 255, 0.8)",
                        bordercolor=COLOR_SYSTEM['NEUTRAL']['LIGHT'],
                        borderwidth=1,
                        borderpad=2
                    )
            except (IndexError, KeyError):
                pass
        
        # Add an insights annotation if CPI is being analyzed
        if value_column == 'CPI':
            bin_type = "incidence rate" if bin_column == "IR_Bin" else \
                       "interview length" if bin_column == "LOI_Bin" else \
                       "sample size" if bin_column == "Completes_Bin" else bin_column
                       
            fig = add_insights_annotation(
                fig,
                f"This chart shows how average CPI varies by {bin_type} between won and lost bids. Larger percentage differences indicate pricing sensitivity in those segments.",
                0.01, 
                0.95,
                width=220
            )
        
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
                len=0.75,
                title_font=dict(
                    family=TYPOGRAPHY['FONT_FAMILY'],
                    size=12,
                    color=COLOR_SYSTEM['PRIMARY']['MAIN']
                ),
                tickfont=dict(
                    family=TYPOGRAPHY['FONT_FAMILY'],
                    size=10,
                    color=COLOR_SYSTEM['PRIMARY']['MAIN']
                )
            )
        ))
        
        # Apply consistent styling
        fig.update_layout(
            title=title,
            height=600,
            xaxis_title="LOI Bin",
            yaxis_title="IR Bin",
            font=dict(
                family=TYPOGRAPHY['FONT_FAMILY'],
                size=12,
                color=COLOR_SYSTEM['PRIMARY']['MAIN']
            ),
            plot_bgcolor=COLOR_SYSTEM['NEUTRAL']['WHITE'],
            paper_bgcolor=COLOR_SYSTEM['NEUTRAL']['WHITE'],
            margin=dict(l=10, r=10, t=50, b=10),
        )
        
        # Update xaxis properties
        fig.update_xaxes(
            tickangle=45,
            title_font=dict(
                family=TYPOGRAPHY['FONT_FAMILY'],
                size=14,
                color=COLOR_SYSTEM['PRIMARY']['MAIN']
            ),
            tickfont=dict(
                family=TYPOGRAPHY['FONT_FAMILY'],
                size=10,
                color=COLOR_SYSTEM['PRIMARY']['MAIN']
            ),
            title_standoff=25
        )
        
        # Update yaxis properties
        fig.update_yaxes(
            title_font=dict(
                family=TYPOGRAPHY['FONT_FAMILY'],
                size=14,
                color=COLOR_SYSTEM['PRIMARY']['MAIN']
            ),
            tickfont=dict(
                family=TYPOGRAPHY['FONT_FAMILY'],
                size=10,
                color=COLOR_SYSTEM['PRIMARY']['MAIN']
            ),
            title_standoff=25
        )
        
        # Add insights annotation
        fig = add_insights_annotation(
            fig,
            text="This heatmap shows how average CPI varies across different combinations of Incidence Rate (IR) and Length of Interview (LOI). Darker colors indicate higher CPI values.",
            x_pos=0.01,
            y_pos=0.95,
            width=220
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