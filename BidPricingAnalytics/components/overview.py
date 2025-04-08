"""
Overview component for the CPI Analysis & Prediction Dashboard.
Displays a high-level summary of the data and key metrics with enhanced styling.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any, Optional

# Import UI components
from ui_components import (
    render_card, 
    metrics_row, 
    grid_layout, 
    add_tooltip
)

# Import visualization utilities
from utils.visualization import (
    create_type_distribution_chart,
    create_cpi_distribution_boxplot,
    create_cpi_vs_ir_scatter,
    create_cpi_efficiency_chart
)

# Import color system
from config import COLOR_SYSTEM, TYPOGRAPHY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_overview(won_data: pd.DataFrame, lost_data: pd.DataFrame, combined_data: pd.DataFrame) -> None:
    """
    Display the overview dashboard showing key metrics and charts with enhanced styling.
    
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
            ">CPI Analysis Dashboard: Overview</h1>
            
            <p style="
                color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};
                font-size: 1rem;
                line-height: 1.6;
            ">
                This dashboard provides a comprehensive analysis of Cost Per Interview (CPI) 
                by examining key factors like Incidence Rate (IR), Length of Interview (LOI), 
                and Sample Size. Dive into the visualizations to understand pricing dynamics 
                and make data-driven decisions.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Key Metrics Section
        st.subheader("Key Performance Metrics")
        
        # Prepare metrics data
        metrics_data = [
            {
                "label": "Avg CPI (Won Bids)", 
                "value": f"${won_data['CPI'].mean():.2f}",
                "delta": f"{((won_data['CPI'].mean() - lost_data['CPI'].mean()) / lost_data['CPI'].mean() * 100):.1f}%"
            },
            {
                "label": "Avg IR (Won Bids)", 
                "value": f"{won_data['IR'].mean():.2f}%",
                "delta": f"{(won_data['IR'].mean() - lost_data['IR'].mean()):.2f}%"
            },
            {
                "label": "Avg LOI (Won Bids)", 
                "value": f"{won_data['LOI'].mean():.2f} min",
                "delta": f"{(won_data['LOI'].mean() - lost_data['LOI'].mean()):.2f} min"
            }
        ]
        
        # Display metrics with grid layout
        def create_metric_card(label, value, delta):
            st.markdown(f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                border-radius: 0.5rem;
                padding: 1rem;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                text-align: center;
                transition: transform 0.2s;
            " onmouseover="this.style.transform='translateY(-5px)'" onmouseout="this.style.transform='translateY(0)'">
                <h3 style="
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                    margin-bottom: 0.5rem;
                    font-size: 1rem;
                    font-weight: 600;
                ">{label}</h3>
                <div style="
                    font-size: 1.5rem;
                    font-weight: 700;
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                ">{value}</div>
                <div style="
                    font-size: 0.9rem;
                    color: {'#2ECC71' if float(delta.replace('%', '')) >= 0 else '#E74C3C'};
                    margin-top: 0.25rem;
                ">{delta}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Create grid of metric cards
        grid_layout(
            3, 
            [
                lambda: create_metric_card(**metrics_data[0]),
                lambda: create_metric_card(**metrics_data[1]),
                lambda: create_metric_card(**metrics_data[2])
            ]
        )

        # Key Visualizations Section
        st.markdown("""
        <h2 style="
            color: #2C3E50;
            margin-top: 2rem;
            margin-bottom: 1rem;
            font-size: 1.5rem;
            font-weight: 600;
        ">Key Data Visualizations</h2>
        """, unsafe_allow_html=True)

        # Create two columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Type Distribution Chart
            render_card(
                "Bid Type Distribution", 
                st.plotly_chart(
                    create_type_distribution_chart(combined_data), 
                    use_container_width=True
                )
            )
        
        with col2:
            # CPI Distribution Boxplot
            render_card(
                "CPI Distribution Comparison", 
                st.plotly_chart(
                    create_cpi_distribution_boxplot(won_data, lost_data), 
                    use_container_width=True
                )
            )

        # Full-width charts
        st.markdown("""
        <h3 style="
            color: #2C3E50;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            font-size: 1.25rem;
            font-weight: 600;
        ">Detailed Relationship Analysis</h3>
        """, unsafe_allow_html=True)

        # CPI vs IR Scatter Plot
        render_card(
            "CPI vs Incidence Rate (IR)", 
            st.plotly_chart(
                create_cpi_vs_ir_scatter(won_data, lost_data), 
                use_container_width=True
            )
        )

        # CPI Efficiency Chart
        render_card(
            "CPI Efficiency Metric", 
            st.plotly_chart(
                create_cpi_efficiency_chart(won_data, lost_data), 
                use_container_width=True
            )
        )

        # Insights Section
        st.markdown(f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin-top: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            border-left: 4px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
        ">
            <h3 style="
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-bottom: 1rem;
                font-size: 1.25rem;
                font-weight: 600;
            ">Key Insights</h3>
            
            <ul style="
                color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};
                font-size: 1rem;
                line-height: 1.6;
                padding-left: 1.5rem;
            ">
                <li>The average CPI for won bids is lower than lost bids, indicating pricing is a critical factor in winning projects.</li>
                <li>Incidence Rate (IR) shows a strong inverse relationship with CPI - lower IR typically means higher per-interview costs.</li>
                <li>Longer surveys (higher LOI) tend to have higher CPIs, reflecting the additional time and effort required.</li>
                <li>The efficiency metric reveals that successful bids balance IR, survey length, and sample size more effectively.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Action Call Section
        st.markdown(f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin-top: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            border-left: 4px solid {COLOR_SYSTEM['ACCENT']['GREEN']};
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
        ">
            <h3 style="
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-bottom: 1rem;
                font-size: 1.25rem;
                font-weight: 600;
            ">Next Steps</h3>
            
            <p style="
                color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};
                font-size: 1rem;
                line-height: 1.6;
                margin-bottom: 1rem;
            ">
                Explore our detailed analysis sections to gain deeper insights:
            </p>
            
            <div style="display: flex; gap: 1rem;">
                <a href="#cpi-analysis" style="
                    background-color: {COLOR_SYSTEM['ACCENT']['BLUE']};
                    color: white;
                    padding: 0.5rem 1rem;
                    border-radius: 0.3rem;
                    text-decoration: none;
                    font-weight: 600;
                    display: inline-block;
                ">Detailed CPI Analysis</a>
                
                <a href="#prediction" style="
                    background-color: {COLOR_SYSTEM['ACCENT']['GREEN']};
                    color: white;
                    padding: 0.5rem 1rem;
                    border-radius: 0.3rem;
                    text-decoration: none;
                    font-weight: 600;
                    display: inline-block;
                ">CPI Prediction Tool</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        logger.error(f"Error in overview component: {e}", exc_info=True)
        st.error(f"An error occurred while rendering the overview component: {str(e)}")