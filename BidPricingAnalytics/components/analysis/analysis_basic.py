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
    render_card, metrics_row, apply_chart_styling,
    add_insights_annotation, add_data_point_annotation,
    grid_layout, render_icon_tabs
)

# Import visualization utilities
from utils.visualization import (
    create_cpi_vs_ir_scatter,
    create_bar_chart_by_bin
)

# Import data utilities
from utils.data_processor import get_data_summary

# Import color system
from config import COLOR_SYSTEM, TYPOGRAPHY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_basic_analysis(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> None:
    """
    Display the basic analysis component with relationship charts.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
    """
    try:
        # Add section header
        st.markdown(f"""
        <h2 style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            font-size: {TYPOGRAPHY['HEADING']['H2']['size']};
            font-weight: {TYPOGRAPHY['HEADING']['H2']['weight']};
            margin-bottom: 1rem;
        ">Basic CPI Relationship Analysis</h2>
        """, unsafe_allow_html=True)
        
        # Add explanation card
        explanation_content = f"""
        <p style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
        ">
            This section explores the relationship between CPI and key project parameters:
            <strong>Incidence Rate (IR)</strong>, <strong>Length of Interview (LOI)</strong>,
            and <strong>Sample Size (Completes)</strong>.
        </p>
        <p style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            margin-top: 0.5rem;
        ">
            Understanding these relationships helps in making data-driven pricing decisions
            and identifying optimal pricing strategies for different project specifications.
        </p>
        """
        
        render_card(
            title="Relationship Analysis Guide", 
            content=explanation_content,
            icon=f'<span style="font-size: 1.5rem; color: {COLOR_SYSTEM["ACCENT"]["BLUE"]};">🔍</span>'
        )
        
        # Create custom tabs with icons
        def render_ir_analysis():
            # Add CPI vs IR scatter plot
            fig = create_cpi_vs_ir_scatter(won_data, lost_data, add_trend_line=True)
            
            # Apply enhanced styling
            st.plotly_chart(fig, use_container_width=True, height=600)
            
            # Add explanation card
            ir_content = f"""
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                <strong>Incidence Rate (IR)</strong> is the percentage of people who qualify for a survey. 
                It significantly impacts CPI due to screening costs.
            </p>
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-top: 0.5rem;
            ">
                <strong>Key observations:</strong>
            </p>
            <ul style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                <li>Lower IR values generally require higher CPI due to increased screening cost</li>
                <li>The relationship is typically non-linear, with steeper CPI increases below 20% IR</li>
                <li>Won bids tend to have a more favorable CPI-to-IR ratio than lost bids</li>
            </ul>
            """
            
            render_card(
                title="IR-CPI Relationship Insights", 
                content=ir_content,
                accent_color=COLOR_SYSTEM['ACCENT']['BLUE']
            )
            
            # Add IR bin analysis
            st.markdown(f"""
            <h3 style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                font-size: {TYPOGRAPHY['HEADING']['H3']['size']};
                font-weight: {TYPOGRAPHY['HEADING']['H3']['weight']};
                margin: 1.5rem 0 1rem 0;
            ">CPI by IR Bin Analysis</h3>
            """, unsafe_allow_html=True)
            
            # Add bar chart for IR bins
            fig = create_bar_chart_by_bin(
                won_data, 
                lost_data, 
                bin_column='IR_Bin', 
                value_column='CPI',
                title='Average CPI by Incidence Rate (IR) Bin'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate optimal IR bin
            ir_bins = won_data.groupby('IR_Bin')['CPI'].mean().reset_index()
            lost_ir_bins = lost_data.groupby('IR_Bin')['CPI'].mean().reset_index()
            
            # Merge to calculate CPI difference
            ir_comparison = pd.merge(
                ir_bins, 
                lost_ir_bins, 
                on='IR_Bin', 
                suffixes=('_won', '_lost')
            )
            
            ir_comparison['diff'] = ir_comparison['CPI_lost'] - ir_comparison['CPI_won']
            ir_comparison['diff_pct'] = (ir_comparison['CPI_lost'] / ir_comparison['CPI_won'] - 1) * 100
            
            # Find the bin with the biggest advantage for won bids
            if not ir_comparison.empty:
                optimal_ir_bin = ir_comparison.loc[ir_comparison['diff'].idxmax()]
                
                # Display the optimal IR bin information
                col1, col2 = st.columns(2)
                
                with col1:
                    render_card(
                        title="Optimal IR Range", 
                        content=f"""
                        <div style="text-align: center; font-size: 1.5rem; font-weight: bold; color: {COLOR_SYSTEM['ACCENT']['GREEN']};">
                            {optimal_ir_bin['IR_Bin']}%
                        </div>
                        <div style="text-align: center; font-size: 0.9rem; color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};">
                            IR range with largest won/lost differential
                        </div>
                        """,
                        accent_color=COLOR_SYSTEM['ACCENT']['GREEN']
                    )
                
                with col2:
                    render_card(
                        title="Price Advantage", 
                        content=f"""
                        <div style="text-align: center; font-size: 1.5rem; font-weight: bold; color: {COLOR_SYSTEM['ACCENT']['GREEN']};">
                            ${optimal_ir_bin['diff']:.2f} ({optimal_ir_bin['diff_pct']:.1f}%)
                        </div>
                        <div style="text-align: center; font-size: 0.9rem; color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};">
                            CPI difference between won and lost bids
                        </div>
                        """,
                        accent_color=COLOR_SYSTEM['ACCENT']['GREEN']
                    )
            
            return st.container()

        def render_loi_analysis():
            # Add bar chart for LOI bins
            fig = create_bar_chart_by_bin(
                won_data, 
                lost_data, 
                bin_column='LOI_Bin', 
                value_column='CPI',
                title='Average CPI by Length of Interview (LOI) Bin'
            )
            
            # Add loi vs CPI scatter plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Create a scatter plot for LOI vs CPI
            fig = go.Figure()
            
            # Add Won data
            fig.add_trace(go.Scatter(
                x=won_data['LOI'], 
                y=won_data['CPI'], 
                mode='markers',
                marker=dict(
                    color=COLOR_SYSTEM['CHARTS']['WON'], 
                    size=10, 
                    opacity=0.7,
                    line=dict(width=1, color=COLOR_SYSTEM['NEUTRAL']['WHITE'])
                ),
                name="Won",
                hovertemplate='<b>Won Bid</b><br>LOI: %{x:.1f} min<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<br>Completes: %{customdata[1]}<extra></extra>',
                customdata=won_data[['IR', 'Completes']]
            ))
            
            # Add Lost data
            fig.add_trace(go.Scatter(
                x=lost_data['LOI'], 
                y=lost_data['CPI'], 
                mode='markers',
                marker=dict(
                    color=COLOR_SYSTEM['CHARTS']['LOST'], 
                    size=10, 
                    opacity=0.7,
                    line=dict(width=1, color=COLOR_SYSTEM['NEUTRAL']['WHITE'])
                ),
                name="Lost",
                hovertemplate='<b>Lost Bid</b><br>LOI: %{x:.1f} min<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<br>Completes: %{customdata[1]}<extra></extra>',
                customdata=lost_data[['IR', 'Completes']]
            ))
            
            # Add trend lines
            # Won trend line
            x_range = np.linspace(min(won_data['LOI']), max(won_data['LOI']), 100)
            coeffs = np.polyfit(won_data['LOI'], won_data['CPI'], 1)
            y_trend = np.polyval(coeffs, x_range)
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_trend,
                mode='lines',
                line=dict(color=COLOR_SYSTEM['CHARTS']['WON'], width=2, dash='solid'),
                name='Won Trend',
                hoverinfo='skip'
            ))
            
            # Lost trend line
            x_range = np.linspace(min(lost_data['LOI']), max(lost_data['LOI']), 100)
            coeffs = np.polyfit(lost_data['LOI'], lost_data['CPI'], 1)
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
                title="Relationship Between Length of Interview (LOI) and CPI",
                height=600
            )
            
            # Update axis titles with more descriptive labels
            fig.update_layout(
                xaxis_title="Length of Interview (minutes)",
                yaxis_title="Cost Per Interview ($)",
            )
            
            # Add insights annotations
            fig = add_insights_annotation(
                fig,
                "Longer interviews typically require higher CPI due to increased respondent compensation and dropout rates.",
                0.01,
                0.95,
                width=220
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation card
            loi_content = f"""
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                <strong>Length of Interview (LOI)</strong> affects CPI because longer surveys require:
            </p>
            <ul style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                <li>Higher respondent incentives</li>
                <li>Increased dropout rates (raising overall costs)</li>
                <li>More complex programming and quality control</li>
            </ul>
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-top: 0.5rem;
            ">
                <strong>Pricing strategy:</strong> For every additional minute of LOI, expect to increase CPI by 
                approximately ${coeffs[0]:.2f} for competitive bids.
            </p>
            """
            
            render_card(
                title="LOI-CPI Relationship Insights", 
                content=loi_content,
                accent_color=COLOR_SYSTEM['ACCENT']['ORANGE']
            )
            
            # Calculate CPI per minute metric
            st.markdown(f"""
            <h3 style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                font-size: {TYPOGRAPHY['HEADING']['H3']['size']};
                font-weight: {TYPOGRAPHY['HEADING']['H3']['weight']};
                margin: 1.5rem 0 1rem 0;
            ">CPI per Minute Analysis</h3>
            """, unsafe_allow_html=True)
            
            # Create CPI per minute column
            won_data_copy = won_data.copy()
            lost_data_copy = lost_data.copy()
            
            won_data_copy['CPI_per_Min'] = won_data_copy['CPI'] / won_data_copy['LOI']
            lost_data_copy['CPI_per_Min'] = lost_data_copy['CPI'] / lost_data_copy['LOI']
            
            # Calculate average CPI per minute
            won_cpi_per_min = won_data_copy['CPI_per_Min'].mean()
            lost_cpi_per_min = lost_data_copy['CPI_per_Min'].mean()
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                render_card(
                    title="Won Bids CPI per Minute", 
                    content=f"""
                    <div style="text-align: center; font-size: 1.5rem; font-weight: bold; color: {COLOR_SYSTEM['CHARTS']['WON']};">
                        ${won_cpi_per_min:.2f}
                    </div>
                    <div style="text-align: center; font-size: 0.9rem; color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};">
                        Average cost per minute for won bids
                    </div>
                    """,
                    accent_color=COLOR_SYSTEM['CHARTS']['WON']
                )
            
            with col2:
                render_card(
                    title="Lost Bids CPI per Minute", 
                    content=f"""
                    <div style="text-align: center; font-size: 1.5rem; font-weight: bold; color: {COLOR_SYSTEM['CHARTS']['LOST']};">
                        ${lost_cpi_per_min:.2f}
                    </div>
                    <div style="text-align: center; font-size: 0.9rem; color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};">
                        Average cost per minute for lost bids
                    </div>
                    """,
                    accent_color=COLOR_SYSTEM['CHARTS']['LOST']
                )
            
            with col3:
                diff_pct = (lost_cpi_per_min / won_cpi_per_min - 1) * 100
                render_card(
                    title="Difference", 
                    content=f"""
                    <div style="text-align: center; font-size: 1.5rem; font-weight: bold; color: {COLOR_SYSTEM['ACCENT']['PURPLE']};">
                        {diff_pct:.1f}%
                    </div>
                    <div style="text-align: center; font-size: 0.9rem; color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};">
                        Cost premium for lost bids per minute
                    </div>
                    """,
                    accent_color=COLOR_SYSTEM['ACCENT']['PURPLE']
                )
                
            return st.container()

        def render_completes_analysis():
            # Add bar chart for Completes bins
            fig = create_bar_chart_by_bin(
                won_data, 
                lost_data, 
                bin_column='Completes_Bin', 
                value_column='CPI',
                title='Average CPI by Sample Size (Completes) Bin'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create a scatter plot for Completes vs CPI
            fig = go.Figure()
            
            # Add Won data
            fig.add_trace(go.Scatter(
                x=won_data['Completes'], 
                y=won_data['CPI'], 
                mode='markers',
                marker=dict(
                    color=COLOR_SYSTEM['CHARTS']['WON'], 
                    size=10, 
                    opacity=0.7,
                    line=dict(width=1, color=COLOR_SYSTEM['NEUTRAL']['WHITE'])
                ),
                name="Won",
                hovertemplate='<b>Won Bid</b><br>Completes: %{x}<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<br>LOI: %{customdata[1]:.1f} min<extra></extra>',
                customdata=won_data[['IR', 'LOI']]
            ))
            
            # Add Lost data
            fig.add_trace(go.Scatter(
                x=lost_data['Completes'], 
                y=lost_data['CPI'], 
                mode='markers',
                marker=dict(
                    color=COLOR_SYSTEM['CHARTS']['LOST'], 
                    size=10, 
                    opacity=0.7,
                    line=dict(width=1, color=COLOR_SYSTEM['NEUTRAL']['WHITE'])
                ),
                name="Lost",
                hovertemplate='<b>Lost Bid</b><br>Completes: %{x}<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<br>LOI: %{customdata[1]:.1f} min<extra></extra>',
                customdata=lost_data[['IR', 'LOI']]
            ))
            
            # Add log-transformed trend lines for better fit with scale effects
            # Won trend line
            won_log_completes = np.log1p(won_data['Completes'])
            coeffs = np.polyfit(won_log_completes, won_data['CPI'], 1)
            
            x_range = np.linspace(min(won_log_completes), max(won_log_completes), 100)
            y_trend = np.polyval(coeffs, x_range)
            
            # Convert back to original scale for plotting
            fig.add_trace(go.Scatter(
                x=np.expm1(x_range),
                y=y_trend,
                mode='lines',
                line=dict(color=COLOR_SYSTEM['CHARTS']['WON'], width=2, dash='solid'),
                name='Won Trend',
                hoverinfo='skip'
            ))
            
            # Lost trend line
            lost_log_completes = np.log1p(lost_data['Completes'])
            coeffs = np.polyfit(lost_log_completes, lost_data['CPI'], 1)
            
            x_range = np.linspace(min(lost_log_completes), max(lost_log_completes), 100)
            y_trend = np.polyval(coeffs, x_range)
            
            # Convert back to original scale for plotting
            fig.add_trace(go.Scatter(
                x=np.expm1(x_range),
                y=y_trend,
                mode='lines',
                line=dict(color=COLOR_SYSTEM['CHARTS']['LOST'], width=2, dash='solid'),
                name='Lost Trend',
                hoverinfo='skip'
            ))
            
            # Apply consistent styling
            fig = apply_chart_styling(
                fig,
                title="Relationship Between Sample Size (Completes) and CPI",
                height=600
            )
            
            # Update axis titles with more descriptive labels
            fig.update_layout(
                xaxis_title="Sample Size (Number of Completes)",
                yaxis_title="Cost Per Interview ($)",
                xaxis_type="log"  # Log scale for better visualization of scale effects
            )
            
            # Add insights annotations
            fig = add_insights_annotation(
                fig,
                "Larger sample sizes often show economies of scale, with lower CPI for higher volumes. This effect is typically logarithmic rather than linear.",
                0.01,
                0.95,
                width=220
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation card
            completes_content = f"""
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                <strong>Sample Size (Completes)</strong> impacts CPI through:
            </p>
            <ul style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                <li><strong>Economies of Scale</strong>: Fixed costs are spread across more completes</li>
                <li><strong>Volume Discounts</strong>: Panel providers often offer lower prices for larger samples</li>
                <li><strong>Operational Efficiency</strong>: Larger projects gain efficiency in project management</li>
            </ul>
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-top: 0.5rem;
            ">
                <strong>Pricing strategy:</strong> Consider offering volume discounts for larger sample sizes,
                especially when competing for high-volume projects.
            </p>
            """
            
            render_card(
                title="Sample Size-CPI Relationship Insights", 
                content=completes_content,
                accent_color=COLOR_SYSTEM['ACCENT']['PURPLE']
            )
            
            return st.container()

        def render_combined_analysis():
            # Add explanation for the combined analysis
            explanation_content = f"""
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                This analysis examines the <strong>combined effect</strong> of multiple factors on CPI.
                It helps identify which combinations of IR, LOI, and Sample Size lead to the most 
                competitive pricing.
            </p>
            """
            
            render_card(
                title="Multi-Factor Analysis", 
                content=explanation_content,
                icon=f'<span style="font-size: 1.5rem; color: {COLOR_SYSTEM["ACCENT"]["YELLOW"]};">🔄</span>',
                accent_color=COLOR_SYSTEM['ACCENT']['YELLOW']
            )
            
            # Create a table of average CPI by IR bin and LOI bin
            pivot_won = pd.pivot_table(
                won_data,
                values='CPI',
                index='IR_Bin',
                columns='LOI_Bin',
                aggfunc='mean'
            )
            
            pivot_lost = pd.pivot_table(
                lost_data,
                values='CPI',
                index='IR_Bin',
                columns='LOI_Bin',
                aggfunc='mean'
            )
            
            # Calculate the difference between lost and won
            pivot_diff = pivot_lost - pivot_won
            pivot_diff_pct = (pivot_lost / pivot_won - 1) * 100
            
            # Create heatmaps
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <h3 style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                    font-size: {TYPOGRAPHY['HEADING']['H3']['size']};
                    font-weight: {TYPOGRAPHY['HEADING']['H3']['weight']};
                    margin: 1rem 0;
                    text-align: center;
                ">Won Bids: Avg CPI by IR and LOI</h3>
                """, unsafe_allow_html=True)
                
                # Create and style the heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=pivot_won.values,
                    x=pivot_won.columns,
                    y=pivot_won.index,
                    colorscale=px.colors.sequential.Blues,
                    hovertemplate='IR Bin: %{y}<br>LOI Bin: %{x}<br>Avg CPI: $%{z:.2f}<extra></extra>',
                    text=[[f"${val:.2f}" if not np.isnan(val) else "" for val in row] for row in pivot_won.values],
                    texttemplate="%{text}",
                ))
                
                # Apply consistent styling
                fig = apply_chart_styling(
                    fig,
                    height=450,
                    show_legend=True
                )
                
                # Update layout
                fig.update_layout(
                    coloraxis_colorbar=dict(title="Avg CPI ($)"),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown(f"""
                <h3 style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                    font-size: {TYPOGRAPHY['HEADING']['H3']['size']};
                    font-weight: {TYPOGRAPHY['HEADING']['H3']['weight']};
                    margin: 1rem 0;
                    text-align: center;
                ">Lost Bids: Avg CPI by IR and LOI</h3>
                """, unsafe_allow_html=True)
                
                # Create and style the heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=pivot_lost.values,
                    x=pivot_lost.columns,
                    y=pivot_lost.index,
                    colorscale=px.colors.sequential.Oranges,
                    hovertemplate='IR Bin: %{y}<br>LOI Bin: %{x}<br>Avg CPI: $%{z:.2f}<extra></extra>',
                    text=[[f"${val:.2f}" if not np.isnan(val) else "" for val in row] for row in pivot_lost.values],
                    texttemplate="%{text}",
                ))
                
                # Apply consistent styling
                fig = apply_chart_styling(
                    fig,
                    height=450,
                    show_legend=True
                )
                
                # Update layout
                fig.update_layout(
                    coloraxis_colorbar=dict(title="Avg CPI ($)"),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Display the difference heatmap
            st.markdown(f"""
            <h3 style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                font-size: {TYPOGRAPHY['HEADING']['H3']['size']};
                font-weight: {TYPOGRAPHY['HEADING']['H3']['weight']};
                margin: 1.5rem 0 1rem 0;
                text-align: center;
            ">CPI Difference Between Lost and Won Bids</h3>
            """, unsafe_allow_html=True)
            
            # Create and style the heatmap for the difference
            fig = go.Figure(data=go.Heatmap(
                z=pivot_diff.values,
                x=pivot_diff.columns,
                y=pivot_diff.index,
                colorscale=px.colors.diverging.RdBu_r,  # Red-Blue diverging colorscale
                hovertemplate='IR Bin: %{y}<br>LOI Bin: %{x}<br>CPI Diff: $%{z:.2f}<extra></extra>',
                text=[[f"${val:.2f}" if not np.isnan(val) else "" for val in row] for row in pivot_diff.values],
                texttemplate="%{text}",
                zmid=0  # Set midpoint for diverging scale
            ))
            
            # Apply consistent styling
            fig = apply_chart_styling(
                fig,
                height=500,
                show_legend=True
            )
            
            # Update layout
            fig.update_layout(
                coloraxis_colorbar=dict(title="CPI Difference ($)"),
            )
            
            # Add insights annotations
            fig = add_insights_annotation(
                fig,
                "Positive values (red) indicate areas where lost bids priced significantly higher than won bids. These are potential areas for more competitive pricing.",
                0.01,
                0.95,
                width=250
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Find the most extreme pricing differential
            # Flatten the pivot tables to find max difference
            pivot_diff_flat = pivot_diff.stack().dropna()
            
            if not pivot_diff_flat.empty:
                max_diff_idx = pivot_diff_flat.idxmax()
                max_diff_value = pivot_diff_flat[max_diff_idx]
                
                ir_bin, loi_bin = max_diff_idx
                
                # Show the insight card
                max_diff_content = f"""
                <p style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                ">
                    The largest pricing differential occurs with projects having:
                </p>
                <div style="
                    display: flex;
                    justify-content: space-between;
                    margin: 0.75rem 0;
                ">
                    <div style="text-align: center; flex: 1;">
                        <div style="font-weight: bold; font-size: 1.1rem;">IR Bin</div>
                        <div style="font-size: 1.3rem; color: {COLOR_SYSTEM['ACCENT']['BLUE']};">{ir_bin}</div>
                    </div>
                    <div style="text-align: center; flex: 1;">
                        <div style="font-weight: bold; font-size: 1.1rem;">LOI Bin</div>
                        <div style="font-size: 1.3rem; color: {COLOR_SYSTEM['ACCENT']['ORANGE']};">{loi_bin}</div>
                    </div>
                </div>
                <p style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                ">
                    For this combination, <strong>lost bids</strong> priced <strong>${max_diff_value:.2f}</strong> higher 
                    than <strong>won bids</strong> on average.
                </p>
                <p style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                    margin-top: 0.5rem;
                ">
                    <strong>Pricing strategy:</strong> Pay special attention to competitive 
                    pricing for projects with these specifications to maximize win probability.
                </p>
                """
                
                render_card(
                    title="Maximum Pricing Differential", 
                    content=max_diff_content,
                    icon=f'<span style="font-size: 1.5rem; color: {COLOR_SYSTEM["ACCENT"]["RED"]};">🎯</span>',
                    accent_color=COLOR_SYSTEM['ACCENT']['RED']
                )
            
            return st.container()
        
        # Create the tabs with icons
        tabs = render_icon_tabs([
            {
                "icon": f'<span style="font-size: 1.3rem; color: {COLOR_SYSTEM["ACCENT"]["BLUE"]};">📈</span>',
                "label": "IR Analysis",
                "content_func": render_ir_analysis
            },
            {
                "icon": f'<span style="font-size: 1.3rem; color: {COLOR_SYSTEM["ACCENT"]["ORANGE"]};">⏱️</span>',
                "label": "LOI Analysis",
                "content_func": render_loi_analysis
            },
            {
                "icon": f'<span style="font-size: 1.3rem; color: {COLOR_SYSTEM["ACCENT"]["PURPLE"]};">📊</span>',
                "label": "Sample Size",
                "content_func": render_completes_analysis
            },
            {
                "icon": f'<span style="font-size: 1.3rem; color: {COLOR_SYSTEM["ACCENT"]["YELLOW"]};">🔄</span>',
                "label": "Combined Analysis",
                "content_func": render_combined_analysis
            }
        ])
    
    except Exception as e:
        # Log error
        logger.error(f"Error in show_basic_analysis: {e}", exc_info=True)
        
        # Display user-friendly error message
        st.error(f"An error occurred while displaying the analysis: {str(e)}")
        st.markdown(f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            border-radius: 0.5rem;
            padding: 1rem;
            margin-top: 1rem;
            border-left: 4px solid {COLOR_SYSTEM['ACCENT']['YELLOW']};
        ">
            <h4 style="margin-top: 0;">Troubleshooting</h4>
            <p>Please try the following:</p>
            <ul>
                <li>Refresh the page</li>
                <li>Check that your data contains sufficient records</li>
                <li>Ensure all required columns (IR, LOI, Completes, CPI) are present</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)won', '_lost')
            )