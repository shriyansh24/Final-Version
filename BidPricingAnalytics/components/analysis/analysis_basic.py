"""
Basic CPI Analysis component for the CPI Analysis & Prediction Dashboard.
Handles the main analysis structure and basic analyses (distribution and IR).
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional

# Import UI components
from ui_components import (
    render_card, 
    render_icon_tabs, 
    add_tooltip
)

# Import configuration
from config import COLOR_SYSTEM, TYPOGRAPHY

# Import visualization utilities
from utils.visualization import (
    create_cpi_histogram_comparison,
    create_cpi_vs_ir_scatter,
    create_bar_chart_by_bin
)

# Import advanced analysis functions
from .analysis_advanced import (
    show_loi_analysis,
    show_sample_size_analysis,
    show_multi_factor_analysis
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_cpi_distribution(won_data: pd.DataFrame, lost_data: pd.DataFrame, tab) -> None:
    """
    Display the CPI Distribution analysis in the given tab.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
        tab: Streamlit tab object
    """
    with tab:
        # Enhanced header with tooltips
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
            ">CPI Distribution Comparison</h2>
            
            <p style="
                color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};
                margin: 0;
                font-size: 0.9rem;
            ">
                Explore the distribution of Cost Per Interview (CPI) for won and lost bids. 
                This visualization helps identify pricing patterns and thresholds.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create and display CPI histogram comparison
        render_card(
            "CPI Distribution Histogram", 
            st.plotly_chart(
                create_cpi_histogram_comparison(won_data, lost_data), 
                use_container_width=True
            )
        )
        
        # Create columns for CPI statistics
        col1, col2 = st.columns(2)
        
        with col1:
            # Won bids statistics card
            render_card(
                "Won Bids CPI Statistics", 
                f"""
                <div style="font-family: {TYPOGRAPHY['FONT_FAMILY']};">
                <ul style="
                    color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};
                    list-style-type: none;
                    padding: 0;
                ">
                    <li>• <strong>Minimum:</strong> ${won_data['CPI'].min():.2f}</li>
                    <li>• <strong>Maximum:</strong> ${won_data['CPI'].max():.2f}</li>
                    <li>• <strong>Mean:</strong> ${won_data['CPI'].mean():.2f}</li>
                    <li>• <strong>Median:</strong> ${won_data['CPI'].median():.2f}</li>
                    <li>• <strong>Standard Deviation:</strong> ${won_data['CPI'].std():.2f}</li>
                </ul>
                </div>
                """
            )
        
        with col2:
            # Lost bids statistics card
            render_card(
                "Lost Bids CPI Statistics", 
                f"""
                <div style="font-family: {TYPOGRAPHY['FONT_FAMILY']};">
                <ul style="
                    color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};
                    list-style-type: none;
                    padding: 0;
                ">
                    <li>• <strong>Minimum:</strong> ${lost_data['CPI'].min():.2f}</li>
                    <li>• <strong>Maximum:</strong> ${lost_data['CPI'].max():.2f}</li>
                    <li>• <strong>Mean:</strong> ${lost_data['CPI'].mean():.2f}</li>
                    <li>• <strong>Median:</strong> ${lost_data['CPI'].median():.2f}</li>
                    <li>• <strong>Standard Deviation:</strong> ${lost_data['CPI'].std():.2f}</li>
                </ul>
                </div>
                """
            )
        
        # Interpretation section with enhanced styling
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
                ">Understanding the CPI Distribution</h3>
                
                <ul style="
                    color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};
                    line-height: 1.6;
                    padding-left: 1.5rem;
                ">
                    <li>The histograms reveal the distribution of CPI values for won and lost bids.</li>
                    <li>Won bids typically have a lower and more concentrated CPI range compared to lost bids.</li>
                    <li>The distribution shape indicates pricing patterns and variability across different projects.</li>
                    <li>Overlapping regions suggest areas where pricing becomes a critical differentiator.</li>
                    <li>Standard deviation highlights the spread of CPI values within each category.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

def show_ir_analysis(won_data: pd.DataFrame, lost_data: pd.DataFrame, tab) -> None:
    """
    Display the Incidence Rate (IR) analysis in the given tab.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
        tab: Streamlit tab object
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
            ">CPI Analysis by Incidence Rate (IR)</h2>
            
            <p style="
                color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};
                margin: 0;
                font-size: 0.9rem;
            ">
                Explore how Incidence Rate (IR) impacts Cost Per Interview. 
                Discover the relationship between survey qualification rates and pricing.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # CPI vs IR Scatter Plot
        render_card(
            "CPI vs Incidence Rate (IR)", 
            st.plotly_chart(
                create_cpi_vs_ir_scatter(won_data, lost_data), 
                use_container_width=True
            )
        )
        
        # CPI by IR Bin Bar Chart
        render_card(
            "Average CPI by IR Bin", 
            st.plotly_chart(
                create_bar_chart_by_bin(won_data, lost_data, 'IR_Bin', 'CPI',
                                      title='Average CPI by Incidence Rate Bin'), 
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
                ">Understanding the IR-CPI Relationship</h3>
                
                <ul style="
                    color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};
                    line-height: 1.6;
                    padding-left: 1.5rem;
                ">
                    <li>Incidence Rate (IR) represents the percentage of people who qualify for a survey.</li>
                    <li>Lower IR values correlate with higher CPIs due to increased screening effort.</li>
                    <li>The scatter plot reveals an inverse relationship between IR and CPI.</li>
                    <li>Lost bids consistently show higher CPIs across different IR ranges.</li>
                    <li>The bar chart highlights pricing variations across IR bins.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

def show_analysis(won_data: pd.DataFrame, lost_data: pd.DataFrame, combined_data: pd.DataFrame) -> None:
    """
    Display the CPI analysis dashboard showing detailed breakdown by different factors.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
        combined_data (pd.DataFrame): Combined DataFrame of won and lost bids
    """
    try:
        st.markdown(f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
        ">
            <h1 style="
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-bottom: 1rem;
                font-size: 2rem;
                font-weight: 700;
            ">CPI Analysis: Won vs. Lost Bids</h1>
            
            <p style="
                color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};
                font-size: 1rem;
                line-height: 1.6;
            ">
                Dive deep into the Cost Per Interview (CPI) analysis. 
                Explore how different factors impact pricing and identify key insights 
                that can help optimize your bidding strategy.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Define tab configuration for icon-based tabs
        tabs_config = [
            {
                'icon': '📊', 
                'label': 'CPI Distribution', 
                'content_func': lambda: show_cpi_distribution(won_data, lost_data, st)
            },
            {
                'icon': '📈', 
                'label': 'By Incidence Rate', 
                'content_func': lambda: show_ir_analysis(won_data, lost_data, st)
            },
            {
                'icon': '⏱️', 
                'label': 'By Length of Interview', 
                'content_func': lambda: show_loi_analysis(won_data, lost_data, st)
            },
            {
                'icon': '📝', 
                'label': 'By Sample Size', 
                'content_func': lambda: show_sample_size_analysis(won_data, lost_data, st)
            },
            {
                'icon': '🔍', 
                'label': 'Multi-Factor Analysis', 
                'content_func': lambda: show_multi_factor_analysis(won_data, lost_data, st)
            }
        ]
        
        # Render custom icon-based tabs
        render_icon_tabs(tabs_config)
        
    except Exception as e:
        logger.error(f"Error in analysis component: {e}", exc_info=True)
        st.error(f"An error occurred while rendering the analysis component: {str(e)}")