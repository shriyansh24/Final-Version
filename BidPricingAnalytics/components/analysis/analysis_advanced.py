y=y_trend,
                    mode='lines',
                    line=dict(color=COLOR_SYSTEM['CHARTS']['LOST'], width=2, dash='solid'),
                    name='Lost Trend',
                    hoverinfo='skip'
                ))
        except Exception as e:
            logger.warning(f"Error adding trend lines: {e}")
        
        # Update layout with consistent styling
        fig.update_layout(
            title={
                'text': 'CPI vs Sample Size (Completes)',
                'font': {
                    'family': TYPOGRAPHY['FONT_FAMILY'],
                    'size': 20,
                    'color': COLOR_SYSTEM['PRIMARY']['MAIN']
                },
                'x': 0.01,
                'xanchor': 'left'
            },
            xaxis_title='Sample Size (Number of Completes)',
            yaxis_title='Cost Per Interview ($)',
            xaxis=dict(type='log'),  # Logarithmic x-axis for better visualization
            legend={
                'font': {
                    'family': TYPOGRAPHY['FONT_FAMILY'],
                    'size': 12,
                    'color': COLOR_SYSTEM['PRIMARY']['MAIN']
                },
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.02,
                'xanchor': 'right',
                'x': 1
            },
            font={
                'family': TYPOGRAPHY['FONT_FAMILY'],
                'size': 12,
                'color': COLOR_SYSTEM['PRIMARY']['MAIN']
            },
            plot_bgcolor=COLOR_SYSTEM['NEUTRAL']['WHITE'],
            paper_bgcolor=COLOR_SYSTEM['NEUTRAL']['WHITE'],
            height=600
        )
        
        # Render scatter plot card
        render_card(
            "CPI vs Sample Size", 
            st.plotly_chart(fig, use_container_width=True)
        )
        
        # CPI by Completes Bin bar chart
        render_card(
            "Average CPI by Sample Size Bin", 
            st.plotly_chart(
                create_bar_chart_by_bin(won_data, lost_data, 'Completes_Bin', 'CPI',
                                      title='Average CPI by Sample Size Bin'), 
                use_container_width=True
            )
        )
        
        # Interpretation section
        with st.expander("📊 Interpretation", expanded=False):
            st.markdown(f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                border-radius: 0.5rem;
                padding: 1rem;
                border-left: 4px solid {COLOR_SYSTEM['ACCENT']['GREEN']};
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
            ">
                <h3 style="
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                    margin-bottom: 0.75rem;
                    font-size: 1.1rem;
                    font-weight: 600;
                ">Understanding the Sample Size-CPI Relationship</h3>
                
                <ul style="
                    color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};
                    line-height: 1.6;
                    padding-left: 1.5rem;
                ">
                    <li>Sample size significantly impacts Cost Per Interview (CPI).</li>
                    <li>Larger sample sizes typically benefit from volume discounts.</li>
                    <li>The scatter plot reveals a logarithmic decrease in CPI as sample size increases.</li>
                    <li>Lost bids show less efficient pricing across different sample sizes.</li>
                    <li>The bar chart highlights CPI variations across sample size bins.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

def show_multi_factor_analysis(won_data: pd.DataFrame, lost_data: pd.DataFrame, tab) -> None:
    """
    Display the Multi-Factor Analysis in the given tab.
    """
    with tab:
        # Enhanced header
        st.markdown(f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 4px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
        ">
            <h2 style="
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin: 0 0 0.5rem 0;
                font-size: 1.25rem;
                font-weight: 600;
            ">Multi-Factor CPI Analysis</h2>
            
            <p style="
                color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};
                margin: 0;
                font-size: 0.9rem;
            ">
                Explore the complex interactions between Incidence Rate (IR), 
                Length of Interview (LOI), and Sample Size in determining Cost Per Interview.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check for sufficient data
        if len(won_data) < 5 or len(lost_data) < 5:
            st.warning("Not enough data for multi-factor analysis. Need at least 5 data points in each category.")
            return
        
        # Heatmap section
        st.subheader("IR and LOI Combined Influence on CPI")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Won bids heatmap
            if 'IR_Bin' in won_data.columns and 'LOI_Bin' in won_data.columns:
                won_grouped = won_data.groupby(['IR_Bin', 'LOI_Bin'])['CPI'].agg(['mean', 'count']).reset_index()
                won_grouped = won_grouped[won_grouped['count'] >= 2]
                
                if len(won_grouped) >= 4:
                    won_pivot = won_grouped.pivot(index='IR_Bin', columns='LOI_Bin', values='mean')
                    render_card(
                        "Won Deals: Average CPI by IR and LOI", 
                        st.plotly_chart(
                            create_heatmap(won_pivot, "Won Deals: Average CPI by IR and LOI", 'Viridis'), 
                            use_container_width=True
                        )
                    )
                else:
                    st.info("Not enough Won data combinations for a meaningful heatmap.")
            else:
                st.info("Missing required binned columns in Won data.")
        
        with col2:
            # Lost bids heatmap
            if 'IR_Bin' in lost_data.columns and 'LOI_Bin' in lost_data.columns:
                lost_grouped = lost_data.groupby(['IR_Bin', 'LOI_Bin'])['CPI'].agg(['mean', 'count']).reset_index()
                lost_grouped = lost_grouped[lost_grouped['count'] >= 2]
                
                if len(lost_grouped) >= 4:
                    lost_pivot = lost_grouped.pivot(index='IR_Bin', columns='LOI_Bin', values='mean')
                    render_card(
                        "Lost Deals: Average CPI by IR and LOI", 
                        st.plotly_chart(
                            create_heatmap(lost_pivot, "Lost Deals: Average CPI by IR and LOI", 'Plasma'), 
                            use_container_width=True
                        )
                    )
                else:
                    st.info("Not enough Lost data combinations for a meaningful heatmap.")
            else:
                st.info("Missing required binned columns in Lost data.")
        
        # 3D Visualization of Total Project Cost
        st.subheader("Combined Factor Impact on Total Project Cost")
        
        # Prepare data
        won_data_copy = won_data.copy()
        lost_data_copy = lost_data.copy()
        won_data_copy['Total_Cost'] = won_data_copy['CPI'] * won_data_copy['Completes']
        lost_data_copy['Total_Cost'] = lost_data_copy['CPI'] * lost_data_copy['Completes']
        
        # 3D Scatter Plot
        fig = go.Figure()
        
        # Add Won data
        fig.add_trace(go.Scatter3d(
            x=won_data_copy['IR'],
            y=won_data_copy['LOI'],
            z=won_data_copy['Total_Cost'],
            mode='markers',
            marker=dict(
                size=won_data_copy['Completes'] / 50,
                color=COLOR_SYSTEM['CHARTS']['WON'],
                opacity=0.7
            ),
            name='Won Bids',
            hovertemplate='<b>Won Bid</b><br>IR: %{x:.1f}%<br>LOI: %{y:.1f} min<br>Total Cost: $%{z:.2f}<br>Completes: %{customdata[0]}<br>CPI: $%{customdata[1]:.2f}<extra></extra>',
            customdata=won_data_copy[['Completes', 'CPI']]
        ))
        
        # Add Lost data
        fig.add_trace(go.Scatter3d(
            x=lost_data_copy['IR'],
            y=lost_data_copy['LOI'],
            z=lost_data_copy['Total_Cost'],
            mode='markers',
            marker=dict(
                size=lost_data_copy['Completes'] / 50,
                color=COLOR_SYSTEM['CHARTS']['LOST'],
                opacity=0.7
            ),
            name='Lost Bids',
            hovertemplate='<b>Lost Bid</b><br>IR: %{x:.1f}%<br>LOI: %{y:.1f} min<br>Total Cost: $%{z:.2f}<br>Completes: %{customdata[0]}<br>CPI: $%{customdata[1]:.2f}<extra></extra>',
            customdata=lost_data_copy[['Completes', 'CPI']]
        ))
        
        # Update 3D plot layout
        fig.update_layout(
            scene=dict(
                xaxis_title='Incidence Rate (%)',
                yaxis_title='Length of Interview (min)',
                zaxis_title='Total Project Cost ($)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            title='3D Visualization: IR, LOI, and Total Project Cost',
            height=700,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        # Render 3D plot
        render_card(
            "3D Visualization of Project Parameters", 
            st.plotly_chart(fig, use_container_width=True)
        )
        
        # Interpretation section
        with st.expander("📊 Interpretation", expanded=False):
            st.markdown(f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                border-radius: 0.5rem;
                padding: 1rem;
                border-left: 4px solid {COLOR_SYSTEM['ACCENT']['GREEN']};
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
            ">
                <h3 style="
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                    margin-bottom: 0.75rem;
                    font-size: 1.1rem;
                    font-weight: 600;
                ">Understanding Multi-Factor Interactions</h3>
                
                <ul style="
                    color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};
                    line-height: 1.6;
                    padding-left: 1.5rem;
                ">
                    <li>CPI is not determined by a single factor but by complex interactions.</li>
                    <li>Heatmaps reveal how Incidence Rate and LOI jointly influence pricing.</li>
                    <li>3D visualization shows the combined impact on total project cost.</li>
                    <li>Marker sizes represent sample size, highlighting volume effects.</li>
                    <li>Won and lost bids show distinct patterns in multi-dimensional space.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
"""
Advanced CPI Analysis component for the CPI Analysis & Prediction Dashboard.
Handles advanced analyses (LOI, sample size, and multi-factor).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
from typing import Dict, List, Tuple, Any, Optional

# Import configuration
from config import COLOR_SYSTEM, TYPOGRAPHY

# Import UI components
from ui_components import render_card, add_tooltip

# Import visualization utilities
from utils.visualization import (
    create_bar_chart_by_bin,
    create_heatmap
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_loi_analysis(won_data: pd.DataFrame, lost_data: pd.DataFrame, tab) -> None:
    """
    Display the Length of Interview (LOI) analysis in the given tab.
    """
    with tab:
        # Enhanced header
        st.markdown(f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 4px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
        ">
            <h2 style="
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin: 0 0 0.5rem 0;
                font-size: 1.25rem;
                font-weight: 600;
            ">CPI Analysis by Length of Interview (LOI)</h2>
            
            <p style="
                color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};
                margin: 0;
                font-size: 0.9rem;
            ">
                Explore how survey duration impacts Cost Per Interview. 
                Understand the relationship between interview length and pricing dynamics.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create CPI vs LOI scatter plot
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
            name="Won Bids",
            hovertemplate='<b>Won Bid</b><br>LOI: %{x:.1f} min<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<extra></extra>',
            customdata=won_data[['IR']]
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
            name="Lost Bids",
            hovertemplate='<b>Lost Bid</b><br>LOI: %{x:.1f} min<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<extra></extra>',
            customdata=lost_data[['IR']]
        ))
        
        # Add trend lines
        try:
            # Won trend line
            valid_won = won_data.dropna(subset=['LOI', 'CPI'])
            if len(valid_won) > 1:
                won_coeffs = np.polyfit(valid_won['LOI'], valid_won['CPI'], 1)
                x_range_won = np.linspace(valid_won['LOI'].min(), valid_won['LOI'].max(), 100)
                y_trend_won = np.polyval(won_coeffs, x_range_won)
                
                fig.add_trace(go.Scatter(
                    x=x_range_won,
                    y=y_trend_won,
                    mode='lines',
                    line=dict(color=COLOR_SYSTEM['CHARTS']['WON'], width=2, dash='solid'),
                    name='Won Trend',
                    hoverinfo='skip'
                ))
            
            # Lost trend line
            valid_lost = lost_data.dropna(subset=['LOI', 'CPI'])
            if len(valid_lost) > 1:
                lost_coeffs = np.polyfit(valid_lost['LOI'], valid_lost['CPI'], 1)
                x_range_lost = np.linspace(valid_lost['LOI'].min(), valid_lost['LOI'].max(), 100)
                y_trend_lost = np.polyval(lost_coeffs, x_range_lost)
                
                fig.add_trace(go.Scatter(
                    x=x_range_lost,
                    y=y_trend_lost,
                    mode='lines',
                    line=dict(color=COLOR_SYSTEM['CHARTS']['LOST'], width=2, dash='solid'),
                    name='Lost Trend',
                    hoverinfo='skip'
                ))
        except Exception as e:
            logger.warning(f"Error adding trend lines: {e}")
        
        # Update layout with consistent styling
        fig.update_layout(
            title={
                'text': 'CPI vs Length of Interview (LOI)',
                'font': {
                    'family': TYPOGRAPHY['FONT_FAMILY'],
                    'size': 20,
                    'color': COLOR_SYSTEM['PRIMARY']['MAIN']
                },
                'x': 0.01,
                'xanchor': 'left'
            },
            xaxis_title='Length of Interview (minutes)',
            yaxis_title='Cost Per Interview ($)',
            legend={
                'font': {
                    'family': TYPOGRAPHY['FONT_FAMILY'],
                    'size': 12,
                    'color': COLOR_SYSTEM['PRIMARY']['MAIN']
                },
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.02,
                'xanchor': 'right',
                'x': 1
            },
            font={
                'family': TYPOGRAPHY['FONT_FAMILY'],
                'size': 12,
                'color': COLOR_SYSTEM['PRIMARY']['MAIN']
            },
            plot_bgcolor=COLOR_SYSTEM['NEUTRAL']['WHITE'],
            paper_bgcolor=COLOR_SYSTEM['NEUTRAL']['WHITE'],
            height=600
        )
        
        # Render scatter plot card
        render_card(
            "CPI vs Length of Interview", 
            st.plotly_chart(fig, use_container_width=True)
        )
        
        # CPI by LOI Bin bar chart
        render_card(
            "Average CPI by LOI Bin", 
            st.plotly_chart(
                create_bar_chart_by_bin(won_data, lost_data, 'LOI_Bin', 'CPI',
                                      title='Average CPI by Length of Interview Bin'), 
                use_container_width=True
            )
        )
        
        # Interpretation section
        with st.expander("📊 Interpretation", expanded=False):
            st.markdown(f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                border-radius: 0.5rem;
                padding: 1rem;
                border-left: 4px solid {COLOR_SYSTEM['ACCENT']['GREEN']};
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
            ">
                <h3 style="
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                    margin-bottom: 0.75rem;
                    font-size: 1.1rem;
                    font-weight: 600;
                ">Understanding the LOI-CPI Relationship</h3>
                
                <ul style="
                    color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};
                    line-height: 1.6;
                    padding-left: 1.5rem;
                ">
                    <li>Length of Interview (LOI) directly impacts Cost Per Interview (CPI).</li>
                    <li>Longer surveys typically command higher prices to compensate for respondent time.</li>
                    <li>The scatter plot reveals a positive correlation between LOI and CPI.</li>
                    <li>Lost bids show a steeper increase in CPI with longer interview lengths.</li>
                    <li>The bar chart highlights pricing variations across different LOI bins.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

def show_sample_size_analysis(won_data: pd.DataFrame, lost_data: pd.DataFrame, tab) -> None:
    """
    Display the Sample Size analysis in the given tab.
    """
    with tab:
        # Enhanced header
        st.markdown(f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 4px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
        ">
            <h2 style="
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin: 0 0 0.5rem 0;
                font-size: 1.25rem;
                font-weight: 600;
            ">CPI Analysis by Sample Size (Completes)</h2>
            
            <p style="
                color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};
                margin: 0;
                font-size: 0.9rem;
            ">
                Examine how project scale impacts Cost Per Interview. 
                Discover volume discounting and pricing dynamics across different sample sizes.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create CPI vs Completes scatter plot with logarithmic characteristics
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
            name="Won Bids",
            hovertemplate='<b>Won Bid</b><br>Completes: %{x}<br>CPI: $%{y:.2f}<br>LOI: %{customdata[0]:.1f} min<extra></extra>',
            customdata=won_data[['LOI']]
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
            name="Lost Bids",
            hovertemplate='<b>Lost Bid</b><br>Completes: %{x}<br>CPI: $%{y:.2f}<br>LOI: %{customdata[0]:.1f} min<extra></extra>',
            customdata=lost_data[['LOI']]
        ))
        
        # Add trend lines with logarithmic transformation
        try:
            # Won trend line (log-transformed)
            valid_won = won_data[won_data['Completes'] > 0].dropna(subset=['Completes', 'CPI'])
            if len(valid_won) > 1:
                log_x = np.log(valid_won['Completes'])
                won_coeffs = np.polyfit(log_x, valid_won['CPI'], 1)
                x_range = np.linspace(valid_won['Completes'].min(), valid_won['Completes'].max(), 100)
                y_trend = won_coeffs[0] * np.log(x_range) + won_coeffs[1]
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=y_trend,
                    mode='lines',
                    line=dict(color=COLOR_SYSTEM['CHARTS']['WON'], width=2, dash='solid'),
                    name='Won Trend',
                    hoverinfo='skip'
                ))
            
            # Lost trend line (log-transformed)
            valid_lost = lost_data[lost_data['Completes'] > 0].dropna(subset=['Completes', 'CPI'])
            if len(valid_lost) > 1:
                log_x = np.log(valid_lost['Completes'])
                lost_coeffs = np.polyfit(log_x, valid_lost['CPI'], 1)
                x_range = np.linspace(valid_lost['Completes'].min(), valid_lost['Completes'].max(), 100)
                y_trend = lost_coeffs[0] * np.log(x_range) + lost_coeffs[1]
                
                fig.add_trace(go.Scatter(
                    x=x_range