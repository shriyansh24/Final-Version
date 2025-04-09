"""
Overview component for the CPI Analysis & Prediction Dashboard.
Displays summary metrics, key charts, and an introductory overview of the data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging
from typing import Dict, Any

# Import UI components and visualization utilities
from ui_components import render_card, metrics_row, apply_chart_styling, add_insights_annotation, grid_layout
from utils.visualization import (
    create_type_distribution_chart,
    create_cpi_distribution_boxplot,
    create_cpi_histogram_comparison,
    create_cpi_efficiency_chart
)
from utils.data_processor import get_data_summary
from config import COLOR_SYSTEM, TYPOGRAPHY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_overview(won_data: pd.DataFrame, lost_data: pd.DataFrame, combined_data: pd.DataFrame) -> None:
    """
    Display the overview dashboard with key metrics and charts.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids.
        lost_data (pd.DataFrame): DataFrame of Lost bids.
        combined_data (pd.DataFrame): Combined DataFrame.
    """
    try:
        # Section header with dark theme styling
        st.markdown(f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1.5rem;
            border-left: 4px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
        ">
            <h1 style="margin-top: 0; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Dashboard Overview</h1>
            <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']}; margin-bottom: 0;">
                Summary of CPI data, key metrics, and visualization of won vs lost bids.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Introduction content with dark theme styling
        intro_content = f"""
        <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
            This dashboard provides comprehensive analysis of Cost Per Interview (CPI) for market research projects. 
            Compare won and lost bids to optimize your pricing strategies.
        </p>
        <ul style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
            <li><span style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Data Analysis:</span> Explore relationships between project parameters and CPI</li>
            <li><span style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Prediction:</span> Use ML models to predict optimal CPI for new projects</li>
            <li><span style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Insights:</span> Get strategic recommendations to improve win rates</li>
        </ul>
        """
        
        render_card(
            title="About This Dashboard",
            content=intro_content,
            icon='📊',
            accent_color=COLOR_SYSTEM['ACCENT']['BLUE']
        )
        
        # Calculate data summary
        data_summary = get_data_summary(combined_data)
        
        # Show key metrics
        st.markdown(f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid {COLOR_SYSTEM['ACCENT']['GREEN']};
        ">
            <h2 style="margin-top: 0; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Key Metrics</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics row with dark theme styling
        metrics = []
        
        # Won bids metrics
        if 'Won' in data_summary:
            metrics.append({
                "title": "Won Bids",
                "value": f"{data_summary['Won']['Count']}",
                "subtitle": f"Avg CPI: ${data_summary['Won']['Avg_CPI']:.2f}",
                "color": COLOR_SYSTEM['CHARTS']['WON'],
                "icon": "✓"
            })
        
        # Lost bids metrics
        if 'Lost' in data_summary:
            metrics.append({
                "title": "Lost Bids",
                "value": f"{data_summary['Lost']['Count']}",
                "subtitle": f"Avg CPI: ${data_summary['Lost']['Avg_CPI']:.2f}",
                "color": COLOR_SYSTEM['CHARTS']['LOST'],
                "icon": "✗"
            })
        
        # Price difference metric
        if 'Won' in data_summary and 'Lost' in data_summary:
            price_diff = data_summary['Lost']['Avg_CPI'] - data_summary['Won']['Avg_CPI']
            price_diff_pct = (price_diff / data_summary['Won']['Avg_CPI']) * 100
            
            metrics.append({
                "title": "Price Differential",
                "value": f"${price_diff:.2f}",
                "subtitle": f"{price_diff_pct:.1f}% higher for lost bids",
                "color": COLOR_SYSTEM['ACCENT']['PURPLE'],
                "icon": "↔"
            })
        
        # Average IR, LOI, and Completes
        metrics.append({
            "title": "Avg IR",
            "value": f"{data_summary['Combined']['Avg_IR']:.1f}%",
            "subtitle": "Average Incidence Rate",
            "color": COLOR_SYSTEM['ACCENT']['BLUE'],
            "icon": "📊"
        })
        
        metrics.append({
            "title": "Avg LOI",
            "value": f"{data_summary['Combined']['Avg_LOI']:.1f} min",
            "subtitle": "Average Length of Interview",
            "color": COLOR_SYSTEM['ACCENT']['GREEN'],
            "icon": "⏱"
        })
        
        # Display metrics using the metrics_row function
        metrics_row(metrics)
        
        # Data Distribution section
        st.markdown(f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1.5rem 0 1rem 0;
            border-left: 4px solid {COLOR_SYSTEM['ACCENT']['ORANGE']};
        ">
            <h2 style="margin-top: 0; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Data Distribution</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Create two columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Won vs Lost distribution
            fig_pie = create_type_distribution_chart(combined_data)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # CPI distribution boxplot
            fig_box = create_cpi_distribution_boxplot(won_data, lost_data)
            st.plotly_chart(fig_box, use_container_width=True)
        
        # CPI histogram comparison
        st.subheader("CPI Distribution Analysis")
        fig_hist = create_cpi_histogram_comparison(won_data, lost_data)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Create a container for the key findings from data
        findings_content = f"""
        <div style="background-color: {COLOR_SYSTEM['BACKGROUND']['DARK']}; padding: 1rem; border-radius: 0.5rem;">
            <h3 style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']}; margin-top: 0;">Key Findings from Data</h3>
            <ul style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                <li><span style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Average CPI Gap:</span> Lost bids are priced ${price_diff:.2f} higher than won bids on average</li>
                <li><span style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Price Range:</span> Won bids CPI ranges from ${data_summary['Won']['CPI_25th']:.2f} to ${data_summary['Won']['CPI_75th']:.2f} (middle 50%)</li>
                <li><span style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Efficiency:</span> The most cost-effective bids optimize the balance between IR, LOI, and sample size</li>
                <li><span style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Outliers:</span> Special attention should be paid to bids with very low IR (<5%) or very high LOI (>30 min)</li>
            </ul>
        </div>
        """
        
        render_card(
            title="Data Insights Summary",
            content=findings_content,
            icon='💡',
            accent_color=COLOR_SYSTEM['ACCENT']['PURPLE']
        )
        
        # Efficiency analysis section
        st.markdown(f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1.5rem 0 1rem 0;
            border-left: 4px solid {COLOR_SYSTEM['ACCENT']['GREEN']};
        ">
            <h2 style="margin-top: 0; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Efficiency Analysis</h2>
            <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']}; margin-bottom: 0;">
                Examining the relationship between multiple factors and CPI to identify optimal pricing patterns.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # CPI efficiency chart
        fig_efficiency = create_cpi_efficiency_chart(won_data, lost_data)
        st.plotly_chart(fig_efficiency, use_container_width=True)
        
        # Next steps guidance
        next_steps_content = f"""
        <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
            Based on this overview, we recommend the following next steps:
        </p>
        <ol style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
            <li>Explore the <span style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">CPI Analysis</span> section to understand detailed relationships between project parameters and pricing</li>
            <li>Use the <span style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">CPI Prediction</span> tool to generate pricing recommendations for new projects</li>
            <li>Review <span style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Insights & Recommendations</span> for strategic actions to improve win rates</li>
        </ol>
        """
        
        render_card(
            title="Next Steps",
            content=next_steps_content,
            icon='🚀',
            accent_color=COLOR_SYSTEM['ACCENT']['BLUE']
        )
        
    except Exception as e:
        logger.error(f"Error in show_overview: {e}", exc_info=True)
        st.error(f"An error occurred while displaying the overview: {str(e)}")
        
        error_content = f"""
        <p style="color: {COLOR_SYSTEM['ACCENT']['RED']};">
            An error occurred while rendering the overview component. This could be due to missing data or incompatible data format.
        </p>
        <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
            Error details: {str(e)}
        </p>
        <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
            Please check the console logs for more information.
        </p>
        """
        
        render_card(
            title="Error Loading Overview",
            content=error_content,
            icon='⚠️',
            accent_color=COLOR_SYSTEM['ACCENT']['RED']
        )
