"""
Advanced analysis component for the CPI Analysis & Prediction Dashboard.
Contains 3D visualizations, heatmaps, and advanced analytical tools.
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
    add_insights_annotation, grid_layout, render_icon_tabs
)

# Import visualization utilities
from utils.visualization import (
    create_heatmap
)

# Import data utilities
from utils.data_processor import get_data_summary
from utils.data_processor import engineer_features

# Import color system
from config import COLOR_SYSTEM, TYPOGRAPHY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_advanced_analysis(won_data: pd.DataFrame, lost_data: pd.DataFrame, combined_data: pd.DataFrame) -> None:
    """
    Display the advanced analysis component with 3D visualizations and heatmaps.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
        combined_data (pd.DataFrame): Combined DataFrame of both Won and Lost bids
    """
    try:
        # Add page header
        st.markdown(f"""
        <h2 style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            font-size: {TYPOGRAPHY['HEADING']['H2']['size']};
            font-weight: {TYPOGRAPHY['HEADING']['H2']['weight']};
            margin-bottom: 1rem;
        ">Advanced CPI Analysis</h2>
        """, unsafe_allow_html=True)
        
        # Add introduction card
        intro_content = f"""
        <p style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
        ">
            This section provides advanced visualizations and analysis techniques to uncover 
            deeper insights into the relationships between project parameters and CPI.
        </p>
        <p style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            margin-top: 0.5rem;
        ">
            Explore 3D visualizations, heatmaps, and multi-factor analyses to optimize your 
            pricing strategy based on complex parameter interactions.
        </p>
        """
        
        render_card(
            title="Advanced Analysis Tools", 
            content=intro_content,
            icon=f'<span style="font-size: 1.5rem; color: {COLOR_SYSTEM["ACCENT"]["PURPLE"]};">🔬</span>'
        )
        
        # Create tabs for different advanced analyses
        def render_3d_visualization():
            # Add 3D scatter plot showing IR, LOI, and CPI
            st.markdown(f"""
            <h3 style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                font-size: {TYPOGRAPHY['HEADING']['H3']['size']};
                font-weight: {TYPOGRAPHY['HEADING']['H3']['weight']};
                margin: 1rem 0;
            ">3D Relationship: IR, LOI and CPI</h3>
            """, unsafe_allow_html=True)
            
            # Create the 3D scatter plot
            fig = go.Figure()
            
            # Add Won data points
            fig.add_trace(go.Scatter3d(
                x=won_data['IR'],
                y=won_data['LOI'],
                z=won_data['CPI'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=COLOR_SYSTEM['CHARTS']['WON'],
                    opacity=0.7,
                    line=dict(width=0.5, color=COLOR_SYSTEM['NEUTRAL']['WHITE'])
                ),
                name='Won Bids',
                hovertemplate='<b>Won Bid</b><br>IR: %{x:.1f}%<br>LOI: %{y:.1f} min<br>CPI: $%{z:.2f}<br>Completes: %{customdata[0]}<extra></extra>',
                customdata=won_data[['Completes']]
            ))
            
            # Add Lost data points
            fig.add_trace(go.Scatter3d(
                x=lost_data['IR'],
                y=lost_data['LOI'],
                z=lost_data['CPI'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=COLOR_SYSTEM['CHARTS']['LOST'],
                    opacity=0.7,
                    line=dict(width=0.5, color=COLOR_SYSTEM['NEUTRAL']['WHITE'])
                ),
                name='Lost Bids',
                hovertemplate='<b>Lost Bid</b><br>IR: %{x:.1f}%<br>LOI: %{y:.1f} min<br>CPI: $%{z:.2f}<br>Completes: %{customdata[0]}<extra></extra>',
                customdata=lost_data[['Completes']]
            ))
            
            # Update layout with improved styling
            fig.update_layout(
                title='3D Relationship Between IR, LOI, and CPI',
                scene=dict(
                    xaxis_title='Incidence Rate (%)',
                    yaxis_title='Length of Interview (min)',
                    zaxis_title='CPI ($)',
                    xaxis=dict(
                        gridcolor=COLOR_SYSTEM['NEUTRAL']['LIGHT'],
                        zerolinecolor=COLOR_SYSTEM['NEUTRAL']['MEDIUM'],
                        showbackground=True,
                        backgroundcolor=COLOR_SYSTEM['NEUTRAL']['LIGHTEST']
                    ),
                    yaxis=dict(
                        gridcolor=COLOR_SYSTEM['NEUTRAL']['LIGHT'],
                        zerolinecolor=COLOR_SYSTEM['NEUTRAL']['MEDIUM'],
                        showbackground=True,
                        backgroundcolor=COLOR_SYSTEM['NEUTRAL']['LIGHTEST']
                    ),
                    zaxis=dict(
                        gridcolor=COLOR_SYSTEM['NEUTRAL']['LIGHT'],
                        zerolinecolor=COLOR_SYSTEM['NEUTRAL']['MEDIUM'],
                        showbackground=True,
                        backgroundcolor=COLOR_SYSTEM['NEUTRAL']['LIGHTEST']
                    )
                ),
                width=800,
                height=700,
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(
                        family=TYPOGRAPHY['FONT_FAMILY'],
                        size=12,
                        color=COLOR_SYSTEM['PRIMARY']['MAIN']
                    )
                ),
                font=dict(
                    family=TYPOGRAPHY['FONT_FAMILY'],
                    size=12,
                    color=COLOR_SYSTEM['PRIMARY']['MAIN']
                )
            )
            
            # Add buttons for different perspectives
            camera_buttons = [
                dict(
                    args=[{'scene.camera': {'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5}}}],
                    label='3D View',
                    method='relayout'
                ),
                dict(
                    args=[{'scene.camera': {'eye': {'x': 0, 'y': 0, 'z': 2.5}}}],
                    label='Top View (IR-LOI Plane)',
                    method='relayout'
                ),
                dict(
                    args=[{'scene.camera': {'eye': {'x': 0, 'y': 2.5, 'z': 0}}}],
                    label='Front View (IR-CPI Plane)',
                    method='relayout'
                ),
                dict(
                    args=[{'scene.camera': {'eye': {'x': 2.5, 'y': 0, 'z': 0}}}],
                    label='Side View (LOI-CPI Plane)',
                    method='relayout'
                )
            ]
            
            fig.update_layout(
                updatemenus=[
                    dict(
                        type='buttons',
                        showactive=True,
                        buttons=camera_buttons,
                        direction='right',
                        pad={'r': 10, 't': 10},
                        x=0.1,
                        xanchor='left',
                        y=1.1,
                        yanchor='top',
                        font=dict(
                            family=TYPOGRAPHY['FONT_FAMILY'],
                            size=12,
                            color=COLOR_SYSTEM['PRIMARY']['MAIN']
                        ),
                        bgcolor=COLOR_SYSTEM['NEUTRAL']['LIGHTEST'],
                        bordercolor=COLOR_SYSTEM['NEUTRAL']['LIGHT'],
                        borderwidth=1
                    )
                ]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanatory card
            explanation_content = f"""
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                This 3D visualization shows the <strong>combined effect</strong> of IR and LOI on CPI values.
            </p>
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-top: 0.5rem;
            ">
                <strong>How to use this chart:</strong>
            </p>
            <ul style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                <li>Use the view buttons to switch perspectives</li>
                <li>Click and drag to rotate the 3D space</li>
                <li>Scroll to zoom in and out</li>
                <li>Hover over points to see detailed information</li>
            </ul>
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-top: 0.5rem;
            ">
                <strong>Key insight:</strong> Notice the <strong>price surface</strong> pattern formed by 
                these points. The difference between won and lost bids forms a separation boundary in this 
                three-dimensional space.
            </p>
            """
            
            render_card(
                title="Understanding the 3D Visualization", 
                content=explanation_content,
                accent_color=COLOR_SYSTEM['ACCENT']['PURPLE']
            )
            
            # Add a second 3D visualization with Sample Size as the third dimension
            st.markdown(f"""
            <h3 style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                font-size: {TYPOGRAPHY['HEADING']['H3']['size']};
                font-weight: {TYPOGRAPHY['HEADING']['H3']['weight']};
                margin: 1.5rem 0 1rem 0;
            ">3D Relationship: IR, LOI and Sample Size</h3>
            """, unsafe_allow_html=True)
            
            # Create a bubble chart to show 4 dimensions: IR, LOI, Completes (size), and CPI (color)
            fig = go.Figure()
            
            # Set bubble size scale factor based on data
            max_completes = max(
                max(won_data['Completes']) if not won_data.empty else 0,
                max(lost_data['Completes']) if not lost_data.empty else 0
            )
            bubble_size_factor = 50 / (max_completes if max_completes > 0 else 1)
            
            # Add Won data points
            fig.add_trace(go.Scatter3d(
                x=won_data['IR'],
                y=won_data['LOI'],
                z=won_data['CPI'],
                mode='markers',
                marker=dict(
                    size=won_data['Completes'] * bubble_size_factor,  # Scale by sample size
                    sizemin=4,  # Minimum marker size
                    sizemode='diameter',
                    color=won_data['CPI'],
                    colorscale='Blues',
                    opacity=0.7,
                    line=dict(width=0.5, color=COLOR_SYSTEM['NEUTRAL']['WHITE']),
                    colorbar=dict(
                        title='CPI ($)',
                        x=0.9,
                        thickness=15
                    )
                ),
                name='Won Bids',
                hovertemplate='<b>Won Bid</b><br>IR: %{x:.1f}%<br>LOI: %{y:.1f} min<br>CPI: $%{z:.2f}<br>Completes: %{customdata[0]}<extra></extra>',
                customdata=won_data[['Completes']]
            ))
            
            # Update layout with improved styling
            fig.update_layout(
                title='4D Visualization: IR, LOI, CPI, and Sample Size',
                scene=dict(
                    xaxis_title='Incidence Rate (%)',
                    yaxis_title='Length of Interview (min)',
                    zaxis_title='CPI ($)',
                    xaxis=dict(
                        gridcolor=COLOR_SYSTEM['NEUTRAL']['LIGHT'],
                        zerolinecolor=COLOR_SYSTEM['NEUTRAL']['MEDIUM'],
                        showbackground=True,
                        backgroundcolor=COLOR_SYSTEM['NEUTRAL']['LIGHTEST']
                    ),
                    yaxis=dict(
                        gridcolor=COLOR_SYSTEM['NEUTRAL']['LIGHT'],
                        zerolinecolor=COLOR_SYSTEM['NEUTRAL']['MEDIUM'],
                        showbackground=True,
                        backgroundcolor=COLOR_SYSTEM['NEUTRAL']['LIGHTEST']
                    ),
                    zaxis=dict(
                        gridcolor=COLOR_SYSTEM['NEUTRAL']['LIGHT'],
                        zerolinecolor=COLOR_SYSTEM['NEUTRAL']['MEDIUM'],
                        showbackground=True,
                        backgroundcolor=COLOR_SYSTEM['NEUTRAL']['LIGHTEST']
                    )
                ),
                width=800,
                height=700,
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(
                        family=TYPOGRAPHY['FONT_FAMILY'],
                        size=12,
                        color=COLOR_SYSTEM['PRIMARY']['MAIN']
                    )
                ),
                font=dict(
                    family=TYPOGRAPHY['FONT_FAMILY'],
                    size=12,
                    color=COLOR_SYSTEM['PRIMARY']['MAIN']
                )
            )
            
            # Add buttons for different perspectives
            camera_buttons = [
                dict(
                    args=[{'scene.camera': {'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5}}}],
                    label='3D View',
                    method='relayout'
                ),
                dict(
                    args=[{'scene.camera': {'eye': {'x': 0, 'y': 0, 'z': 2.5}}}],
                    label='Top View (IR-LOI Plane)',
                    method='relayout'
                ),
                dict(
                    args=[{'scene.camera': {'eye': {'x': 0, 'y': 2.5, 'z': 0}}}],
                    label='Front View (IR-CPI Plane)',
                    method='relayout'
                ),
                dict(
                    args=[{'scene.camera': {'eye': {'x': 2.5, 'y': 0, 'z': 0}}}],
                    label='Side View (LOI-CPI Plane)',
                    method='relayout'
                )
            ]
            
            fig.update_layout(
                updatemenus=[
                    dict(
                        type='buttons',
                        showactive=True,
                        buttons=camera_buttons,
                        direction='right',
                        pad={'r': 10, 't': 10},
                        x=0.1,
                        xanchor='left',
                        y=1.1,
                        yanchor='top',
                        font=dict(
                            family=TYPOGRAPHY['FONT_FAMILY'],
                            size=12,
                            color=COLOR_SYSTEM['PRIMARY']['MAIN']
                        ),
                        bgcolor=COLOR_SYSTEM['NEUTRAL']['LIGHTEST'],
                        bordercolor=COLOR_SYSTEM['NEUTRAL']['LIGHT'],
                        borderwidth=1
                    )
                ]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanatory card
            explanation_content = f"""
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                This 4D visualization displays:
            </p>
            <ul style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                <li><strong>X-axis:</strong> Incidence Rate (%)</li>
                <li><strong>Y-axis:</strong> Length of Interview (min)</li>
                <li><strong>Z-axis:</strong> CPI ($)</li>
                <li><strong>Bubble Size:</strong> Sample Size (number of completes)</li>
                <li><strong>Color Intensity:</strong> CPI value</li>
            </ul>
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-top: 0.5rem;
            ">
                <strong>Key insight:</strong> Larger bubbles (bigger sample sizes) tend to have lower CPI values,
                demonstrating the economy of scale effect. This visualization helps identify optimal combinations
                of all three parameters.
            </p>
            """
            
            render_card(
                title="Understanding the 4D Visualization", 
                content=explanation_content,
                accent_color=COLOR_SYSTEM['ACCENT']['BLUE']
            )
            
            return st.container()

        def render_heatmap_analysis():
            # Advanced heatmap analysis for identifying pricing opportunities
            st.markdown(f"""
            <h3 style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                font-size: {TYPOGRAPHY['HEADING']['H3']['size']};
                font-weight: {TYPOGRAPHY['HEADING']['H3']['weight']};
                margin: 1rem 0;
            ">CPI Heatmap Analysis</h3>
            """, unsafe_allow_html=True)
            
            # Create pivot tables for IR vs LOI
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
            
            # Create enhanced heatmaps
            col1, col2 = st.columns(2)
            
            with col1:
                won_title = "Won Bids: Average CPI by IR and LOI"
                fig = create_heatmap(
                    pivot_won, 
                    won_title,
                    colorscale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                lost_title = "Lost Bids: Average CPI by IR and LOI"
                fig = create_heatmap(
                    pivot_lost, 
                    lost_title,
                    colorscale='Oranges'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Calculate the differential heatmap
            st.markdown(f"""
            <h3 style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                font-size: {TYPOGRAPHY['HEADING']['H3']['size']};
                font-weight: {TYPOGRAPHY['HEADING']['H3']['weight']};
                margin: 1.5rem 0 1rem 0;
            ">Pricing Differential Analysis</h3>
            """, unsafe_allow_html=True)
            
            # Calculate the absolute and percentage differences
            pivot_diff = pivot_lost - pivot_won
            pivot_diff_pct = ((pivot_lost / pivot_won) - 1) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                diff_title = "Absolute CPI Difference (Lost - Won)"
                fig = create_heatmap(
                    pivot_diff, 
                    diff_title,
                    colorscale='RdBu_r'
                )
                
                # Add annotations to highlight the largest differences
                fig = add_insights_annotation(
                    fig,
                    "Red cells indicate areas where lost bids priced significantly higher than won bids. These represent potential pricing opportunities.",
                    0.01,
                    0.95,
                    width=220
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                pct_diff_title = "Percentage CPI Difference (%)"
                fig = create_heatmap(
                    pivot_diff_pct, 
                    pct_diff_title,
                    colorscale='RdBu_r'
                )
                
                # Add annotations to highlight the largest percentage differences
                fig = add_insights_annotation(
                    fig,
                    "This view shows the percentage premium of lost bids over won bids, identifying areas with the greatest pricing disparity.",
                    0.01,
                    0.95,
                    width=220
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Add opportunity areas card
            opportunity_content = f"""
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                Heatmap analysis reveals these key pricing opportunity areas:
            </p>
            """
            
            # Find the top 3 largest differentials
            if not pivot_diff.empty:
                # Flatten the pivot tables to find top differences
                pivot_diff_flat = pivot_diff.stack().dropna()
                
                if not pivot_diff_flat.empty:
                    # Get top 3 differentials
                    top_diff_indices = pivot_diff_flat.nlargest(3).index.tolist()
                    
                    opportunity_content += f"""
                    <table style="width: 100%; border-collapse: collapse; margin-top: 0.5rem;">
                        <tr style="background-color: {COLOR_SYSTEM['NEUTRAL']['LIGHTER']};">
                            <th style="padding: 8px; text-align: left; border-bottom: 1px solid {COLOR_SYSTEM['NEUTRAL']['LIGHT']};">IR Bin</th>
                            <th style="padding: 8px; text-align: left; border-bottom: 1px solid {COLOR_SYSTEM['NEUTRAL']['LIGHT']};">LOI Bin</th>
                            <th style="padding: 8px; text-align: right; border-bottom: 1px solid {COLOR_SYSTEM['NEUTRAL']['LIGHT']};">Difference</th>
                            <th style="padding: 8px; text-align: right; border-bottom: 1px solid {COLOR_SYSTEM['NEUTRAL']['LIGHT']};">% Difference</th>
                        </tr>
                    """
                    
                    for idx in top_diff_indices:
                        ir_bin, loi_bin = idx
                        abs_diff = pivot_diff.loc[ir_bin, loi_bin]
                        pct_diff = pivot_diff_pct.loc[ir_bin, loi_bin]
                        
                        opportunity_content += f"""
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid {COLOR_SYSTEM['NEUTRAL']['LIGHT']};">{ir_bin}</td>
                            <td style="padding: 8px; border-bottom: 1px solid {COLOR_SYSTEM['NEUTRAL']['LIGHT']};">{loi_bin}</td>
                            <td style="padding: 8px; text-align: right; border-bottom: 1px solid {COLOR_SYSTEM['NEUTRAL']['LIGHT']};">${abs_diff:.2f}</td>
                            <td style="padding: 8px; text-align: right; border-bottom: 1px solid {COLOR_SYSTEM['NEUTRAL']['LIGHT']};">{pct_diff:.1f}%</td>
                        </tr>
                        """
                    
                    opportunity_content += """
                    </table>
                    """
                    
                    opportunity_content += f"""
                    <p style="
                        font-family: {TYPOGRAPHY['FONT_FAMILY']};
                        font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                        color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                        margin-top: 0.75rem;
                    ">
                        <strong>Pricing recommendation:</strong> For projects falling in these parameter ranges, 
                        maintain a moderate pricing premium above won bid levels but below lost bid levels to 
                        maximize both win rate and profit margin.
                    </p>
                    """
            
            render_card(
                title="Pricing Opportunity Areas", 
                content=opportunity_content,
                icon=f'<span style="font-size: 1.5rem; color: {COLOR_SYSTEM["ACCENT"]["GREEN"]};">💰</span>',
                accent_color=COLOR_SYSTEM['ACCENT']['GREEN']
            )
            
            # Create additional heatmaps for different combinations
            st.markdown(f"""
            <h3 style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                font-size: {TYPOGRAPHY['HEADING']['H3']['size']};
                font-weight: {TYPOGRAPHY['HEADING']['H3']['weight']};
                margin: 1.5rem 0 1rem 0;
            ">Additional Parameter Relationships</h3>
            """, unsafe_allow_html=True)
            
            # Create pivot tables for IR vs Completes
            pivot_ir_completes_won = pd.pivot_table(
                won_data,
                values='CPI',
                index='IR_Bin',
                columns='Completes_Bin',
                aggfunc='mean'
            )
            
            pivot_ir_completes_lost = pd.pivot_table(
                lost_data,
                values='CPI',
                index='IR_Bin',
                columns='Completes_Bin',
                aggfunc='mean'
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                ir_completes_title = "Won Bids: CPI by IR and Sample Size"
                fig = create_heatmap(
                    pivot_ir_completes_won, 
                    ir_completes_title,
                    colorscale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                ir_completes_title = "Lost Bids: CPI by IR and Sample Size"
                fig = create_heatmap(
                    pivot_ir_completes_lost, 
                    ir_completes_title,
                    colorscale='Oranges'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Create pivot tables for LOI vs Completes
            pivot_loi_completes_won = pd.pivot_table(
                won_data,
                values='CPI',
                index='LOI_Bin',
                columns='Completes_Bin',
                aggfunc='mean'
            )
            
            pivot_loi_completes_lost = pd.pivot_table(
                lost_data,
                values='CPI',
                index='LOI_Bin',
                columns='Completes_Bin',
                aggfunc='mean'
            )
            
            st.markdown("<hr>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                loi_completes_title = "Won Bids: CPI by LOI and Sample Size"
                fig = create_heatmap(
                    pivot_loi_completes_won, 
                    loi_completes_title,
                    colorscale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                loi_completes_title = "Lost Bids: CPI by LOI and Sample Size"
                fig = create_heatmap(
                    pivot_loi_completes_lost, 
                    loi_completes_title,
                    colorscale='Oranges'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            return st.container()
            
        def render_feature_importance():
            # Generate engineered features
            st.markdown(f"""
            <h3 style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                font-size: {TYPOGRAPHY['HEADING']['H3']['size']};
                font-weight: {TYPOGRAPHY['HEADING']['H3']['weight']};
                margin: 1rem 0;
            ">Feature Importance Analysis</h3>
            """, unsafe_allow_html=True)
            
            # Create explanation card
            explanation_content = f"""
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                This analysis examines how different factors impact CPI by using advanced
                feature engineering and statistical techniques.
            </p>
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-top: 0.5rem;
            ">
                We engineer additional features beyond the basic parameters to understand 
                more complex relationships and interactions between variables.
            </p>
            """
            
            render_card(
                title="Understanding Feature Importance", 
                content=explanation_content,
                icon=f'<span style="font-size: 1.5rem; color: {COLOR_SYSTEM["ACCENT"]["RED"]};">🔍</span>'
            )
            
            # Engineer features
            # First ensure won/lost data has a Type column
            won_data_copy = won_data.copy()
            lost_data_copy = lost_data.copy()
            
            # Make sure Type column exists
            if 'Type' not in won_data_copy.columns:
                won_data_copy['Type'] = 'Won'
            if 'Type' not in lost_data_copy.columns:
                lost_data_copy['Type'] = 'Lost'
            
            # Combine the data for feature engineering
            combined_for_analysis = pd.concat([won_data_copy, lost_data_copy], ignore_index=True)
            
            # Apply feature engineering
            engineered_data = engineer_features(combined_for_analysis)
            
            # Create correlation heatmap
            st.subheader("Feature Correlation Heatmap")
            
            # Select numeric columns for correlation
            numeric_cols = [
                'IR', 'LOI', 'Completes', 'CPI',
                'IR_LOI_Ratio', 'IR_Completes_Ratio', 'LOI_Completes_Ratio',
                'CPI_Efficiency'
            ]
            
            # Add additional engineered features if they exist
            additional_features = [
                'IR_LOI_Product', 'CPI_per_Minute', 'Log_Completes',
                'Log_IR', 'Log_LOI', 'Log_CPI',
                'IR_Normalized', 'LOI_Normalized', 'Completes_Normalized'
            ]
            
            for feature in additional_features:
                if feature in engineered_data.columns:
                    numeric_cols.append(feature)
            
            # Calculate correlation matrix
            corr_matrix = engineered_data[numeric_cols].corr()
            
            # Create correlation heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu_r',
                zmid=0,  # Center the color scale at 0
                text=[[f"{val:.2f}" for val in row] for row in corr_matrix.values],
                texttemplate="%{text}",
                hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
            ))
            
            # Apply consistent styling
            fig = apply_chart_styling(
                fig,
                title="Feature Correlation Matrix",
                height=700
            )
            
            # Add insights annotation
            fig = add_insights_annotation(
                fig,
                "This heatmap shows correlations between features. Strong positive correlations appear in blue, strong negative correlations in red. The diagonal is always 1 (perfect correlation with itself).",
                0.01,
                0.95,
                width=250
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate feature correlation with CPI
            cpi_corr = corr_matrix['CPI'].drop('CPI').sort_values(ascending=False)
            
            # Create a horizontal bar chart of CPI correlations
            fig = go.Figure()
            
            # Add bars
            fig.add_trace(go.Bar(
                y=cpi_corr.index,
                x=cpi_corr.values,
                orientation='h',
                marker=dict(
                    color=[
                        COLOR_SYSTEM['ACCENT']['BLUE'] if x > 0 else COLOR_SYSTEM['ACCENT']['RED']
                        for x in cpi_corr.values
                    ],
                    opacity=0.8
                ),
                text=[f"{x:.3f}" for x in cpi_corr.values],
                textposition='auto',
                hovertemplate='%{y}<br>Correlation with CPI: %{x:.3f}<extra></extra>'
            ))
            
            # Apply consistent styling
            fig = apply_chart_styling(
                fig,
                title="Feature Correlations with CPI",
                height=500
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Correlation Coefficient",
                yaxis_title="Feature",
                yaxis=dict(
                    categoryorder='total ascending'
                )
            )
            
            # Add zero line
            fig.add_shape(
                type="line",
                x0=0,
                y0=-0.5,
                x1=0,
                y1=len(cpi_corr) - 0.5,
                line=dict(
                    color=COLOR_SYSTEM['NEUTRAL']['DARKER'],
                    width=1,
                    dash="dash"
                )
            )
            
            # Add insights annotation
            fig = add_insights_annotation(
                fig,
                "Blue bars show features positively correlated with CPI (as they increase, CPI tends to increase). Red bars show negative correlations (as they increase, CPI tends to decrease).",
                0.01,
                0.95,
                width=250
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create insights about feature importance
            top_positive = cpi_corr[cpi_corr > 0].head(3)
            top_negative = cpi_corr[cpi_corr < 0].head(3)
            
            insights_content = f"""
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                <strong>Top features that increase CPI:</strong>
            </p>
            <ul style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
            """
            
            for feature, value in top_positive.items():
                insights_content += f"<li><strong>{feature}</strong>: {value:.3f} correlation</li>"
            
            insights_content += f"""
            </ul>
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-top: 0.5rem;
            ">
                <strong>Top features that decrease CPI:</strong>
            </p>
            <ul style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
            """
            
            for feature, value in top_negative.items():
                insights_content += f"<li><strong>{feature}</strong>: {value:.3f} correlation</li>"
            
            insights_content += f"""
            </ul>
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-top: 0.75rem;
            ">
                <strong>Strategic insight:</strong> Focus on project parameters that have the strongest
                favorable impact on CPI to optimize pricing efficiency.
            </p>
            """
            
            render_card(
                title="Key Feature Insights", 
                content=insights_content,
                accent_color=COLOR_SYSTEM['ACCENT']['YELLOW']
            )
            
            return st.container()
            
        # Create tabs for different advanced analyses
        tabs = render_icon_tabs([
            {
                "icon": f'<span style="font-size: 1.3rem; color: {COLOR_SYSTEM["ACCENT"]["PURPLE"]};">🔮</span>',
                "label": "3D Visualization",
                "content_func": render_3d_visualization
            },
            {
                "icon": f'<span style="font-size: 1.3rem; color: {COLOR_SYSTEM["ACCENT"]["GREEN"]};">🔥</span>',
                "label": "Heatmap Analysis",
                "content_func": render_heatmap_analysis
            },
            {
                "icon": f'<span style="font-size: 1.3rem; color: {COLOR_SYSTEM["ACCENT"]["RED"]};">📊</span>',
                "label": "Feature Importance",
                "content_func": render_feature_importance
            }
        ])
        
    except Exception as e:
        # Log error
        logger.error(f"Error in show_advanced_analysis: {e}", exc_info=True)
        
        # Display user-friendly error message
        st.error(f"An error occurred while displaying the advanced analysis: {str(e)}")
        st.markdown(f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            border-radius: 0.5rem;
            padding: 1rem;
            margin-top: 1rem;
            border-left: 4px solid {COLOR_SYSTEM['ACCENT']['YELLOW']};
        ">
            <h4 style="margin-top: 0;">Troubleshooting</h4>
            <p>Advanced analysis requires more computing resources. Please try the following:</p>
            <ul>
                <li>Refresh the page</li>
                <li>Reduce the data size by applying more filters</li>
                <li>Try the basic analysis section if this persists</li>
                <li>Check that your system has sufficient memory available</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)