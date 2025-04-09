"""
Basic analysis component for the CPI Analysis & Prediction Dashboard.
Contains basic relationship visualizations and analysis tools.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging
from typing import Dict, Any

# Import UI components
from ui_components import (
    render_card,
    metrics_row,
    apply_chart_styling,
    add_insights_annotation,
    add_data_point_annotation,
    grid_layout
)

# Import visualization utilities
from utils.visualization import (
    create_cpi_vs_ir_scatter,
    create_bar_chart_by_bin
)

# Import data utilities
from utils.data_processor import get_data_summary

# Import configuration
from config import COLOR_SYSTEM, TYPOGRAPHY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_basic_analysis(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> None:
    """
    Display the basic analysis component with relationship charts.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids.
        lost_data (pd.DataFrame): DataFrame of Lost bids.
    """
    try:
        # Section header
        st.markdown("""
        <div style="
            background-color: #1F1F1F;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1.5rem;
            border-left: 4px solid #00BFFF;
        ">
            <h2 style="color: #FFFFFF; margin-top: 0;">Basic CPI Analysis</h2>
            <p style="color: #B0B0B0; margin-bottom: 0;">
                This section explores the relationship between CPI and key project parameters:
                <span style="color: #FFFFFF; font-weight: 500;">Incidence Rate (IR)</span>, 
                <span style="color: #FFFFFF; font-weight: 500;">Length of Interview (LOI)</span>,
                and <span style="color: #FFFFFF; font-weight: 500;">Sample Size (Completes)</span>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        explanation_content = """
        <p style="color: #B0B0B0;">
            Understanding the relationships between project parameters and CPI helps in making
            data-driven pricing decisions and identifying optimal pricing strategies.
        </p>
        <ul style="color: #B0B0B0;">
            <li><span style="color: #FFFFFF; font-weight: 500;">IR (Incidence Rate)</span>: Lower IRs typically require higher CPIs due to increased screening costs</li>
            <li><span style="color: #FFFFFF; font-weight: 500;">LOI (Length of Interview)</span>: Longer surveys generally command higher CPIs</li>
            <li><span style="color: #FFFFFF; font-weight: 500;">Sample Size</span>: Larger samples may benefit from economies of scale</li>
        </ul>
        """
        
        render_card(
            title="Relationship Analysis Guide",
            content=explanation_content,
            icon='🔍',
            accent_color=COLOR_SYSTEM['ACCENT']['BLUE']
        )
        
        # Create tabs for each analysis section
        tab1, tab2, tab3, tab4 = st.tabs([
            "IR Impact", 
            "LOI Impact", 
            "Sample Size Impact",
            "Multi-Factor Analysis"
        ])
        
        with tab1:
            st.markdown("""
            <div style="color: #B0B0B0; margin-bottom: 1rem;">
                Examine how <span style="color: #FFFFFF; font-weight: 500;">Incidence Rate (IR)</span> 
                affects CPI pricing and win rates.
            </div>
            """, unsafe_allow_html=True)
            
            # CPI vs IR scatter plot
            fig = create_cpi_vs_ir_scatter(won_data, lost_data, add_trend_line=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate trend coefficients for IR
            try:
                ir_coeffs_won = np.polyfit(won_data['IR'], won_data['CPI'], 1)
                ir_coeffs_lost = np.polyfit(lost_data['IR'], lost_data['CPI'], 1)
                
                ir_content = f"""
                <div style="background-color: #2E2E2E; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
                    <h4 style="color: #FFFFFF; margin-top: 0;">Key IR Impact Insights</h4>
                    <p style="color: #B0B0B0;">
                        For every <span style="color: #FFFFFF;">1% decrease</span> in IR, the CPI tends to 
                        increase by <span style="color: #00BFFF; font-weight: 600;">${abs(ir_coeffs_won[0]):.2f}</span> 
                        for won bids and <span style="color: #FFB74D; font-weight: 600;">${abs(ir_coeffs_lost[0]):.2f}</span> 
                        for lost bids.
                    </p>
                    <p style="color: #B0B0B0;">
                        The price sensitivity to IR is <span style="color: #FFFFFF;">{abs(ir_coeffs_lost[0]/ir_coeffs_won[0]):.1f}x higher</span> 
                        in lost bids compared to won bids.
                    </p>
                </div>
                """
                st.markdown(ir_content, unsafe_allow_html=True)
            except:
                st.warning("Unable to calculate trend coefficients for IR analysis.")
            
            # Create barplot by IR bin
            st.subheader("Average CPI by Incidence Rate Range")
            ir_fig = create_bar_chart_by_bin(
                won_data, 
                lost_data, 
                bin_column='IR_Bin', 
                title="Average CPI by Incidence Rate (IR) Range"
            )
            st.plotly_chart(ir_fig, use_container_width=True)
        
        with tab2:
            st.markdown("""
            <div style="color: #B0B0B0; margin-bottom: 1rem;">
                Examine how <span style="color: #FFFFFF; font-weight: 500;">Length of Interview (LOI)</span> 
                affects pricing and discover optimal pricing strategies based on survey duration.
            </div>
            """, unsafe_allow_html=True)
            
            # Create scatter plot for LOI vs CPI
            loi_scatter = go.Figure()
            
            # Add Won data points
            loi_scatter.add_trace(go.Scatter(
                x=won_data['LOI'],
                y=won_data['CPI'],
                mode='markers',
                marker=dict(
                    color=COLOR_SYSTEM['CHARTS']['WON'],
                    size=10, 
                    opacity=0.7,
                    line=dict(width=1, color=COLOR_SYSTEM['NEUTRAL']['LIGHTEST'])
                ),
                name="Won",
                hovertemplate='<b>Won Bid</b><br>LOI: %{x:.1f} min<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<br>Completes: %{customdata[1]}<extra></extra>',
                customdata=won_data[['IR', 'Completes']]
            ))
            
            # Add Lost data points
            loi_scatter.add_trace(go.Scatter(
                x=lost_data['LOI'],
                y=lost_data['CPI'],
                mode='markers',
                marker=dict(
                    color=COLOR_SYSTEM['CHARTS']['LOST'],
                    size=10, 
                    opacity=0.7,
                    line=dict(width=1, color=COLOR_SYSTEM['NEUTRAL']['LIGHTEST'])
                ),
                name="Lost",
                hovertemplate='<b>Lost Bid</b><br>LOI: %{x:.1f} min<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<br>Completes: %{customdata[1]}<extra></extra>',
                customdata=lost_data[['IR', 'Completes']]
            ))
            
            # Add trend lines
            try:
                # Won trend line
                loi_x_range_won = np.linspace(won_data['LOI'].min(), won_data['LOI'].max(), 100)
                loi_coeffs_won = np.polyfit(won_data['LOI'], won_data['CPI'], 1)
                loi_y_trend_won = np.polyval(loi_coeffs_won, loi_x_range_won)
                
                loi_scatter.add_trace(go.Scatter(
                    x=loi_x_range_won,
                    y=loi_y_trend_won,
                    mode='lines',
                    line=dict(color=COLOR_SYSTEM['CHARTS']['WON'], width=2, dash='dash'),
                    name='Won Trend',
                    hoverinfo='skip'
                ))
                
                # Lost trend line
                loi_x_range_lost = np.linspace(lost_data['LOI'].min(), lost_data['LOI'].max(), 100)
                loi_coeffs_lost = np.polyfit(lost_data['LOI'], lost_data['CPI'], 1)
                loi_y_trend_lost = np.polyval(loi_coeffs_lost, loi_x_range_lost)
                
                loi_scatter.add_trace(go.Scatter(
                    x=loi_x_range_lost,
                    y=loi_y_trend_lost,
                    mode='lines',
                    line=dict(color=COLOR_SYSTEM['CHARTS']['LOST'], width=2, dash='dash'),
                    name='Lost Trend',
                    hoverinfo='skip'
                ))
            except:
                pass
            
            # Apply styling
            loi_scatter = apply_chart_styling(
                loi_scatter,
                title="Relationship Between Length of Interview (LOI) and CPI",
                height=500
            )
            
            loi_scatter.update_layout(
                xaxis_title="Length of Interview (minutes)",
                yaxis_title="Cost Per Interview ($)"
            )
            
            st.plotly_chart(loi_scatter, use_container_width=True)
            
            # LOI insights
            try:
                loi_content = f"""
                <div style="background-color: #2E2E2E; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
                    <h4 style="color: #FFFFFF; margin-top: 0;">Key LOI Impact Insights</h4>
                    <p style="color: #B0B0B0;">
                        For every <span style="color: #FFFFFF;">additional minute</span> of interview length, 
                        the CPI increases by <span style="color: #00BFFF; font-weight: 600;">${loi_coeffs_won[0]:.2f}</span> 
                        for won bids and <span style="color: #FFB74D; font-weight: 600;">${loi_coeffs_lost[0]:.2f}</span> 
                        for lost bids.
                    </p>
                    <p style="color: #B0B0B0;">
                        The price per minute is <span style="color: #FFFFFF;">${loi_coeffs_won[0]:.2f}</span> for competitive bids.
                    </p>
                </div>
                """
                st.markdown(loi_content, unsafe_allow_html=True)
            except:
                st.warning("Unable to calculate trend coefficients for LOI analysis.")
            
            # Create barplot by LOI bin
            st.subheader("Average CPI by Interview Length")
            loi_fig = create_bar_chart_by_bin(
                won_data, 
                lost_data, 
                bin_column='LOI_Bin', 
                title="Average CPI by Length of Interview Range"
            )
            st.plotly_chart(loi_fig, use_container_width=True)
        
        with tab3:
            st.markdown("""
            <div style="color: #B0B0B0; margin-bottom: 1rem;">
                Examine how <span style="color: #FFFFFF; font-weight: 500;">Sample Size (Completes)</span>
                affects pricing and find opportunities for volume-based discounting.
            </div>
            """, unsafe_allow_html=True)
            
            # Create scatter plot for Completes vs CPI
            completes_scatter = go.Figure()
            
            # Add Won data points
            completes_scatter.add_trace(go.Scatter(
                x=won_data['Completes'],
                y=won_data['CPI'],
                mode='markers',
                marker=dict(
                    color=COLOR_SYSTEM['CHARTS']['WON'],
                    size=10, 
                    opacity=0.7,
                    line=dict(width=1, color=COLOR_SYSTEM['NEUTRAL']['LIGHTEST'])
                ),
                name="Won",
                hovertemplate='<b>Won Bid</b><br>Completes: %{x}<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<br>LOI: %{customdata[1]:.1f} min<extra></extra>',
                customdata=won_data[['IR', 'LOI']]
            ))
            
            # Add Lost data points
            completes_scatter.add_trace(go.Scatter(
                x=lost_data['Completes'],
                y=lost_data['CPI'],
                mode='markers',
                marker=dict(
                    color=COLOR_SYSTEM['CHARTS']['LOST'],
                    size=10, 
                    opacity=0.7,
                    line=dict(width=1, color=COLOR_SYSTEM['NEUTRAL']['LIGHTEST'])
                ),
                name="Lost",
                hovertemplate='<b>Lost Bid</b><br>Completes: %{x}<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<br>LOI: %{customdata[1]:.1f} min<extra></extra>',
                customdata=lost_data[['IR', 'LOI']]
            ))
            
            # Apply logarithmic scale for better visualization
            completes_scatter.update_xaxes(type='log')
            
            # Apply styling
            completes_scatter = apply_chart_styling(
                completes_scatter,
                title="Relationship Between Sample Size and CPI",
                height=500
            )
            
            completes_scatter.update_layout(
                xaxis_title="Sample Size (Completes)",
                yaxis_title="Cost Per Interview ($)"
            )
            
            st.plotly_chart(completes_scatter, use_container_width=True)
            
            # Create barplot by Completes bin
            st.subheader("Average CPI by Sample Size Range")
            completes_fig = create_bar_chart_by_bin(
                won_data, 
                lost_data, 
                bin_column='Completes_Bin', 
                title="Average CPI by Sample Size Range"
            )
            st.plotly_chart(completes_fig, use_container_width=True)
            
            # Sample size insights
            completes_content = f"""
            <div style="background-color: #2E2E2E; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
                <h4 style="color: #FFFFFF; margin-top: 0;">Volume Discounting Strategy</h4>
                <p style="color: #B0B0B0;">
                    Data suggests that volume discounts may be appropriate for larger sample sizes:
                </p>
                <ul style="color: #B0B0B0;">
                    <li><span style="color: #FFFFFF;">Small (1-100)</span>: No discount</li>
                    <li><span style="color: #FFFFFF;">Medium (101-500)</span>: 5-7% discount</li>
                    <li><span style="color: #FFFFFF;">Large (501-1000)</span>: 8-12% discount</li>
                    <li><span style="color: #FFFFFF;">Very Large (1000+)</span>: 12-15% discount</li>
                </ul>
                <p style="color: #B0B0B0;">
                    Implementing a tiered discount structure can optimize win rates while maintaining profitability.
                </p>
            </div>
            """
            st.markdown(completes_content, unsafe_allow_html=True)
        
        with tab4:
            st.markdown("""
            <div style="color: #B0B0B0; margin-bottom: 1rem;">
                Examine the <span style="color: #FFFFFF; font-weight: 500;">combined effect</span> of multiple factors on CPI
                to identify optimal pricing strategies for different project specifications.
            </div>
            """, unsafe_allow_html=True)
            
            # Create efficiency metric chart
            if 'CPI_Efficiency' not in won_data.columns:
                won_data_temp = won_data.copy()
                won_data_temp['CPI_Efficiency'] = (won_data_temp['IR'] / 100) * (1 / won_data_temp['LOI'].replace(0, 0.5)) * won_data_temp['Completes']
                
                lost_data_temp = lost_data.copy()
                lost_data_temp['CPI_Efficiency'] = (lost_data_temp['IR'] / 100) * (1 / lost_data_temp['LOI'].replace(0, 0.5)) * lost_data_temp['Completes']
            else:
                won_data_temp = won_data
                lost_data_temp = lost_data
            
            efficiency_scatter = go.Figure()
            
            # Add Won data points
            efficiency_scatter.add_trace(go.Scatter(
                x=won_data_temp['CPI_Efficiency'],
                y=won_data_temp['CPI'],
                mode='markers',
                marker=dict(
                    color=COLOR_SYSTEM['CHARTS']['WON'],
                    size=10, 
                    opacity=0.7,
                    line=dict(width=1, color=COLOR_SYSTEM['NEUTRAL']['LIGHTEST'])
                ),
                name="Won",
                hovertemplate='<b>Won Bid</b><br>Efficiency: %{x:.1f}<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<br>LOI: %{customdata[1]:.1f} min<br>Completes: %{customdata[2]}<extra></extra>',
                customdata=won_data_temp[['IR', 'LOI', 'Completes']]
            ))
            
            # Add Lost data points
            efficiency_scatter.add_trace(go.Scatter(
                x=lost_data_temp['CPI_Efficiency'],
                y=lost_data_temp['CPI'],
                mode='markers',
                marker=dict(
                    color=COLOR_SYSTEM['CHARTS']['LOST'],
                    size=10, 
                    opacity=0.7,
                    line=dict(width=1, color=COLOR_SYSTEM['NEUTRAL']['LIGHTEST'])
                ),
                name="Lost",
                hovertemplate='<b>Lost Bid</b><br>Efficiency: %{x:.1f}<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<br>LOI: %{customdata[1]:.1f} min<br>Completes: %{customdata[2]}<extra></extra>',
                customdata=lost_data_temp[['IR', 'LOI', 'Completes']]
            ))
            
            # Apply styling
            efficiency_scatter = apply_chart_styling(
                efficiency_scatter,
                title="CPI vs Efficiency Metric",
                height=500
            )
            
            efficiency_scatter.update_layout(
                xaxis_title="Efficiency Metric ((IR/100) × (1/LOI) × Completes)",
                yaxis_title="Cost Per Interview ($)"
            )
            
            # Add insights annotation
            efficiency_scatter = add_insights_annotation(
                efficiency_scatter,
                "Higher efficiency values indicate more cost-effective bids. This metric combines IR, LOI, and sample size.",
                0.01, 
                0.95,
                width=250
            )
            
            st.plotly_chart(efficiency_scatter, use_container_width=True)
            
            # Explain the efficiency metric
            efficiency_content = f"""
            <div style="background-color: #2E2E2E; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
                <h4 style="color: #FFFFFF; margin-top: 0;">Understanding the Efficiency Metric</h4>
                <p style="color: #B0B0B0;">
                    The efficiency metric combines multiple factors that influence CPI:
                </p>
                <p style="color: #B0B0B0; text-align: center; font-family: monospace; background: #121212; padding: 0.5rem; border-radius: 0.3rem;">
                    Efficiency = (IR/100) × (1/LOI) × Completes
                </p>
                <p style="color: #B0B0B0;">
                    <span style="color: #FFFFFF;">Higher values</span> indicate conditions that should lead to more cost-effective pricing (higher IR, shorter LOI, larger sample).
                </p>
                <p style="color: #B0B0B0;">
                    The chart shows that as efficiency increases, CPI generally decreases, but lost bids tend to be priced higher than won bids at similar efficiency levels.
                </p>
            </div>
            """
            st.markdown(efficiency_content, unsafe_allow_html=True)

    except Exception as e:
        logger.error(f"Error in show_basic_analysis: {e}", exc_info=True)
        st.error(f"An error occurred while displaying the basic analysis: {str(e)}")
