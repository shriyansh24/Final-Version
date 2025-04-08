"""
Strategic Insights Component for Market Research Bid Pricing Dashboard

This module provides comprehensive, actionable insights derived from 
historical bid data, helping market research teams optimize their 
pricing strategies and improve win rates.

Key Objectives:
- Analyze pricing patterns
- Identify competitive strategies
- Provide data-driven recommendations
- Visualize complex market dynamics
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any, Optional

# UI and Configuration Imports
from ui_components import (
    render_card, 
    metrics_row, 
    grid_layout
)
from config import COLOR_SYSTEM, TYPOGRAPHY

# Configure logging for tracking and debugging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def show_insights(
    won_data: pd.DataFrame, 
    lost_data: pd.DataFrame, 
    combined_data: pd.DataFrame
) -> None:
    """
    Generate a comprehensive insights dashboard with strategic pricing recommendations.
    
    This function creates an interactive Streamlit interface that provides:
    1. Key performance metrics
    2. Pricing strategy insights
    3. Visualization of market dynamics
    4. Actionable recommendations
    
    Args:
        won_data (pd.DataFrame): Historical data for won bids
        lost_data (pd.DataFrame): Historical data for lost bids
        combined_data (pd.DataFrame): Combined dataset of all bids
    """
    try:
        # Engaging Dashboard Header
        st.markdown(f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            border-left: 4px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
        ">
            <h1 style="
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-bottom: 1rem;
                font-size: 2rem;
                font-weight: 700;
            ">Strategic Pricing Insights</h1>
            
            <p style="
                color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};
                font-size: 1rem;
                line-height: 1.6;
            ">
                🔍 Transform Raw Data into Competitive Pricing Strategies
                
                Uncover hidden patterns in your bid pricing, understand 
                market dynamics, and develop data-driven approaches 
                to improve win rates and profitability.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Calculate Key Performance Metrics
        won_avg_cpi = won_data['CPI'].mean()
        lost_avg_cpi = lost_data['CPI'].mean()
        cpi_diff = lost_avg_cpi - won_avg_cpi
        cpi_diff_pct = (cpi_diff / won_avg_cpi) * 100
        
        current_win_rate = len(won_data) / (len(won_data) + len(lost_data)) * 100
        avg_project_revenue = won_data['Revenue'].mean() if 'Revenue' in won_data.columns else 0
        
        # Key Findings Section
        st.header("🔬 Key Performance Findings")
        
        # Metrics Grid
        col1, col2, col3 = st.columns(3)
        
        with col1:
            render_card(
                "Average CPI (Won Bids)", 
                f"""
                <div style="text-align: center;">
                    <h2 style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']}; margin-bottom: 0.5rem;">${won_avg_cpi:.2f}</h2>
                    <p style="color: {COLOR_SYSTEM['NEUTRAL']['DARKER']}; margin: 0;">
                        {f"<span style='color: {'green' if cpi_diff_pct < 0 else 'red'}'>({cpi_diff_pct:+.1f}%)</span> vs Lost Bids" if cpi_diff_pct != 0 else ""}
                    </p>
                </div>
                """
            )
        
        with col2:
            render_card(
                "Win Rate", 
                f"""
                <div style="text-align: center;">
                    <h2 style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']}; margin-bottom: 0.5rem;">{current_win_rate:.1f}%</h2>
                    <p style="color: {COLOR_SYSTEM['NEUTRAL']['DARKER']}; margin: 0;">
                        Total Bids: {len(won_data) + len(lost_data)}
                    </p>
                </div>
                """
            )
        
        with col3:
            render_card(
                "Avg Project Revenue", 
                f"""
                <div style="text-align: center;">
                    <h2 style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']}; margin-bottom: 0.5rem;">${avg_project_revenue:.2f}</h2>
                    <p style="color: {COLOR_SYSTEM['NEUTRAL']['DARKER']}; margin: 0;">
                        Per Successful Bid
                    </p>
                </div>
                """
            )
        
        # CPI Comparison Visualization
        st.header("📊 CPI Distribution Analysis")
        
        # Prepare data for visualization
        ir_grouped_won = won_data.groupby('IR_Bin')['CPI'].agg(['mean', 'median', 'count']).reset_index()
        ir_grouped_lost = lost_data.groupby('IR_Bin')['CPI'].agg(['mean', 'median', 'count']).reset_index()
        
        # Create comparison figure
        fig = go.Figure()
        
        # Won bids average CPI
        fig.add_trace(go.Bar(
            x=ir_grouped_won['IR_Bin'],
            y=ir_grouped_won['mean'],
            name='Won Bids Avg CPI',
            marker_color=COLOR_SYSTEM['CHARTS']['WON'],
            opacity=0.7,
            text=[f'${x:.2f}' for x in ir_grouped_won['mean']],
            textposition='auto'
        ))
        
        # Lost bids average CPI
        fig.add_trace(go.Bar(
            x=ir_grouped_lost['IR_Bin'],
            y=ir_grouped_lost['mean'],
            name='Lost Bids Avg CPI',
            marker_color=COLOR_SYSTEM['CHARTS']['LOST'],
            opacity=0.7,
            text=[f'${x:.2f}' for x in ir_grouped_lost['mean']],
            textposition='auto'
        ))
        
        # Update layout
        fig.update_layout(
            title='Average CPI by Incidence Rate Bin',
            xaxis_title='Incidence Rate Bin',
            yaxis_title='Average CPI ($)',
            barmode='group',
            legend_title_text='Bid Outcome'
        )
        
        # Render CPI comparison chart
        render_card(
            "CPI Comparison by Incidence Rate", 
            st.plotly_chart(fig, use_container_width=True)
        )
        
        # Strategic Recommendations Section
        st.header("🚀 Strategic Pricing Recommendations")
        
        # Pricing Tiers Recommendations
        st.markdown(f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin-bottom: 1rem;
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
        ">
            <h3 style="
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-bottom: 1rem;
                font-size: 1.25rem;
                font-weight: 600;
            ">Pricing Strategy Framework</h3>
            
            <div style="color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};">
                <h4>1. Incidence Rate (IR) Based Pricing</h4>
                <ul>
                    <li><strong>Low IR (0-20%):</strong> Add a 15-20% premium to base CPI</li>
                    <li><strong>Medium IR (21-50%):</strong> Add a 5-10% premium to base CPI</li>
                    <li><strong>High IR (51-100%):</strong> No additional IR premium</li>
                </ul>
                
                <h4>2. Length of Interview (LOI) Adjustments</h4>
                <ul>
                    <li><strong>Short LOI (1-10 min):</strong> Base CPI</li>
                    <li><strong>Medium LOI (11-20 min):</strong> 1.3x base CPI</li>
                    <li><strong>Long LOI (21+ min):</strong> 1.5x base CPI</li>
                </ul>
                
                <h4>3. Sample Size Discounts</h4>
                <ul>
                    <li><strong>Small (1-100 completes):</strong> No discount</li>
                    <li><strong>Medium (101-500 completes):</strong> 5-10% discount</li>
                    <li><strong>Large (501-1000 completes):</strong> 10-15% discount</li>
                    <li><strong>Very Large (1000+ completes):</strong> 15-20% discount</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Implementation Roadmap
        st.header("📋 Implementation Roadmap")
        st.markdown(f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin-bottom: 1rem;
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
        ">
            <h3 style="
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-bottom: 1rem;
                font-size: 1.25rem;
                font-weight: 600;
            ">Strategic Action Plan</h3>
            
            <ol style="
                color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};
                line-height: 1.6;
                padding-left: 1.5rem;
            ">
                <li><strong>Pricing Tool Development:</strong> Create an internal CPI prediction calculator</li>
                <li><strong>Guideline Documentation:</strong> Develop comprehensive pricing strategy guidelines</li>
                <li><strong>Monthly Performance Review:</strong> Analyze bid win rates and pricing effectiveness</li>
                <li><strong>Continuous Model Refinement:</strong> Update predictive models with new bid data</li>
                <li><strong>Competitive Intelligence:</strong> Monitor market pricing trends and adjust strategy</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Final Call to Action
        st.markdown(f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin-top: 1rem;
            text-align: center;
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
        ">
            <h3 style="
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-bottom: 1rem;
                font-size: 1.25rem;
                font-weight: 600;
            ">Ready to Optimize Your Pricing Strategy?</h3>
            
            <p style="
                color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};
                margin-bottom: 1rem;
            ">
                Use the CPI Prediction tool to apply these insights to your next bid.
            </p>
            
            <a href="#cpi-prediction" style="
                background-color: {COLOR_SYSTEM['ACCENT']['BLUE']};
                color: white;
                padding: 0.75rem 1.5rem;
                text-decoration: none;
                border-radius: 0.5rem;
                font-weight: 600;
                display: inline-block;
                transition: background-color 0.3s;
            ">
                Go to CPI Prediction Tool
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        logger.error(f"Error in insights component: {e}", exc_info=True)
        st.error(f"An error occurred while generating insights: {str(e)}")