"""
CPI Analysis & Prediction Dashboard
Main application file that orchestrates the dashboard components.

This application provides a comprehensive analysis of Cost Per Interview (CPI)
for market research projects, including visualization, analysis, and prediction tools.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Any, Optional

# Import configuration
import config

# Import UI components
from ui_components import (
    setup_ui, render_header, render_card, 
    metrics_row, apply_chart_styling,
    COLOR_SYSTEM
)

# Import utility modules
from utils.data_loader import load_data
from utils.data_processor import apply_all_bins, engineer_features, get_data_summary

# Import components
from components.overview import show_overview
from components.analysis import show_analysis
from components.prediction import show_prediction
from components.insights import show_insights

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main application function to orchestrate the dashboard."""
    # Setup UI components
    setup_ui()
    
    # Add app title and description in sidebar
    st.sidebar.title("CPI Analysis & Prediction")
    
    # Navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Choose a mode",
        ["Overview", "CPI Analysis", "CPI Prediction", "Insights & Recommendations"]
    )
    
    # Render page header
    render_header(current_page=app_mode)
    
    # Load data
    with st.spinner("Loading data..."):
        try:
            data = load_data()
            won_df = data['won']
            won_df_filtered = data['won_filtered']
            lost_df = data['lost']
            lost_df_filtered = data['lost_filtered']
            combined_df = data['combined']
            combined_df_filtered = data['combined_filtered']
            
            # Log data shapes
            logger.info(f"Won deals: {won_df.shape}")
            logger.info(f"Lost deals: {lost_df.shape}")
            logger.info(f"Combined: {combined_df.shape}")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.error("Please check that all required data files are in the correct location.")
            st.stop()
    
    # Process data - add bins to all dataframes
    try:
        won_df = apply_all_bins(won_df)
        won_df_filtered = apply_all_bins(won_df_filtered)
        lost_df = apply_all_bins(lost_df)
        lost_df_filtered = apply_all_bins(lost_df_filtered)
        combined_df = apply_all_bins(combined_df)
        combined_df_filtered = apply_all_bins(combined_df_filtered)
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.stop()
    
    # Add sidebar filters with enhanced styling
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div style="
        background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
    ">
        <h3 style="margin-top: 0;">Filtering Options</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Filter for extreme values
    show_filtered = st.sidebar.checkbox(
        "Filter out extreme values (>95th percentile)", 
        value=True,
        help="Remove outliers with very high CPI values to focus on typical cases"
    )
    
    # Choose datasets based on filtering option
    if show_filtered:
        won_data = won_df_filtered
        lost_data = lost_df_filtered
        combined_data = combined_df_filtered
    else:
        won_data = won_df
        lost_data = lost_df
        combined_data = combined_df
    
    # Display metrics in sidebar with enhanced styling
    data_summary = get_data_summary(combined_data)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div style="
        background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid {COLOR_SYSTEM['ACCENT']['PURPLE']};
    ">
        <h3 style="margin-top: 0;">Data Summary</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Create metrics data for display
    metrics_data = []
    
    if 'Won' in data_summary:
        metrics_data.append({
            "label": "Won Bids Avg CPI",
            "value": f"${data_summary['Won']['Avg_CPI']:.2f}"
        })
    
    if 'Lost' in data_summary:
        metrics_data.append({
            "label": "Lost Bids Avg CPI",
            "value": f"${data_summary['Lost']['Avg_CPI']:.2f}"
        })
    
    if 'Won' in data_summary and 'Lost' in data_summary:
        diff = data_summary['Lost']['Avg_CPI'] - data_summary['Won']['Avg_CPI']
        metrics_data.append({
            "label": "CPI Difference",
            "value": f"${diff:.2f}",
            "delta": f"{diff:.2f}"
        })
    
    # Show metrics in sidebar
    for metric in metrics_data:
        st.sidebar.metric(
            label=metric["label"],
            value=metric["value"],
            delta=metric.get("delta", None)
        )
    
    # Add footer with enhanced styling
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div style="
        background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
        border-radius: 0.5rem;
        padding: 1rem;
        font-size: 0.9rem;
        color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};
    ">
        <p style="margin: 0;">
            This dashboard provides analysis and prediction tools for 
            Cost Per Interview (CPI) pricing in market research projects.
        </p>
        <p style="margin: 0.5rem 0 0 0;">
            <span style="color: {COLOR_SYSTEM['ACCENT']['BLUE']};">●</span> <strong>Data updated:</strong> April 2025
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show selected component based on app_mode
    if app_mode == "Overview":
        show_overview(won_data, lost_data, combined_data)
    
    elif app_mode == "CPI Analysis":
        show_analysis(won_data, lost_data, combined_data)
    
    elif app_mode == "CPI Prediction":
        # Engineer features for the prediction model
        combined_data_engineered = engineer_features(combined_data)
        show_prediction(combined_data_engineered, won_data, lost_data)
    
    elif app_mode == "Insights & Recommendations":
        show_insights(won_data, lost_data, combined_data)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.error(f"Unexpected error in main application: {e}", exc_info=True)