"""
Basic CPI Analysis component for the CPI Analysis & Prediction Dashboard.
Handles the main analysis structure and basic analyses (distribution and IR).
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional

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
        st.subheader("CPI Distribution Comparison")
        
        # Add description
        st.markdown("""
        This analysis shows the distribution of CPI values for won and lost bids.
        The histograms help visualize the range and frequency of different price points
        and identify potential pricing thresholds.
        """)
        
        # Create and display CPI histogram comparison
        fig = create_cpi_histogram_comparison(won_data, lost_data)
        st.plotly_chart(fig, use_container_width=True, key='analysis_cpi_histogram')
        
        # Add CPI statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Won Bids CPI Statistics")
            st.markdown(f"""
            - **Minimum:** ${won_data['CPI'].min():.2f}
            - **Maximum:** ${won_data['CPI'].max():.2f}
            - **Mean:** ${won_data['CPI'].mean():.2f}
            - **Median:** ${won_data['CPI'].median():.2f}
            - **Standard Deviation:** ${won_data['CPI'].std():.2f}
            """)
        
        with col2:
            st.subheader("Lost Bids CPI Statistics")
            st.markdown(f"""
            - **Minimum:** ${lost_data['CPI'].min():.2f}
            - **Maximum:** ${lost_data['CPI'].max():.2f}
            - **Mean:** ${lost_data['CPI'].mean():.2f}
            - **Median:** ${lost_data['CPI'].median():.2f}
            - **Standard Deviation:** ${lost_data['CPI'].std():.2f}
            """)
        
        # Add interpretation
        with st.expander("📊 Interpretation"):
            st.markdown("""
            ### What This Analysis Shows
            
            The histograms display the distribution of CPI values for won and lost bids, 
            revealing the price ranges that are most common in each category.
            
            ### Key Insights
            
            1. **Price Range Comparison**: Won bids typically have a lower CPI range compared to lost bids,
               indicating that competitive pricing is important for winning projects.
            
            2. **Distribution Shape**: The shape of the distribution provides insights into pricing patterns.
               A narrower distribution suggests more consistent pricing, while a wider distribution indicates
               more variable pricing based on project factors.
            
            3. **Overlap Areas**: Where the distributions overlap represents the competitive pricing zone
               where other factors besides price (such as reputation, capabilities, relationships) may
               determine bid success.
            
            4. **Pricing Thresholds**: The mean values (vertical dashed lines) can be used as reference
               points for setting competitive pricing thresholds.
            """)


def show_ir_analysis(won_data: pd.DataFrame, lost_data: pd.DataFrame, tab) -> None:
    """
    Display the Incidence Rate (IR) analysis in the given tab.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
        tab: Streamlit tab object
    """
    with tab:
        st.subheader("CPI Analysis by Incidence Rate (IR)")
        
        # Add description
        st.markdown("""
        This analysis explores how Incidence Rate (IR) - the percentage of people who qualify for a survey -
        affects CPI. Lower IR usually means it's harder to find qualified respondents, potentially
        justifying higher prices.
        """)
        
        # Create and display CPI vs IR scatter plot
        fig = create_cpi_vs_ir_scatter(won_data, lost_data)
        st.plotly_chart(fig, use_container_width=True, key='analysis_cpi_vs_ir_scatter')
        
        # Create and display CPI by IR Bin bar chart
        st.subheader("Average CPI by IR Bin")
        fig = create_bar_chart_by_bin(won_data, lost_data, 'IR_Bin', 'CPI',
                                    title='Average CPI by Incidence Rate Bin')
        st.plotly_chart(fig, use_container_width=True, key='analysis_cpi_by_ir_bin')
        
        # Add interpretation
        with st.expander("📊 Interpretation"):
            st.markdown("""
            ### Understanding the IR-CPI Relationship
            
            Incidence Rate (IR) is the percentage of people who qualify for a survey. It has a significant
            impact on CPI because it affects how difficult it is to find qualified respondents.
            
            ### Key Insights
            
            1. **Inverse Relationship**: As shown in both charts, there's generally an inverse relationship
               between IR and CPI - as IR increases, CPI tends to decrease. This is logical because higher
               incidence rates mean less screening effort is needed.
            
            2. **Price Gap by IR Level**: The bar chart reveals that the gap between won and lost CPIs
               varies across IR bins. This can help identify where competitive pricing sensitivity is highest.
            
            3. **Pricing Strategy**: For lower IR ranges (0-30%), pricing sensitivity appears higher,
               suggesting that competitive pricing is especially important for low-IR projects.
            
            4. **Diminishing Returns**: The benefit of higher IR on CPI appears to flatten above certain
               IR thresholds, suggesting that very high IR doesn't necessarily enable proportionally lower pricing.
            """)


def show_analysis(won_data: pd.DataFrame, lost_data: pd.DataFrame, combined_data: pd.DataFrame) -> None:
    """
    Display the CPI analysis dashboard showing detailed breakdown by different factors.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
        combined_data (pd.DataFrame): Combined DataFrame of won and lost bids
    """
    try:
        st.title("CPI Analysis: Won vs. Lost Bids")
        
        # Introduction
        st.markdown("""
        This section provides a detailed analysis of Cost Per Interview (CPI) by different
        factors that influence pricing. Use the tabs below to explore different analyses.
        """)
        
        # Create tabs for different analyses
        tabs = st.tabs([
            "CPI Distribution", 
            "By Incidence Rate (IR)", 
            "By Length of Interview (LOI)", 
            "By Sample Size",
            "Multi-Factor Analysis"
        ])
        
        # Display each analysis in its respective tab
        show_cpi_distribution(won_data, lost_data, tabs[0])
        show_ir_analysis(won_data, lost_data, tabs[1])
        
        # Call functions from analysis_advanced to handle the advanced analyses
        show_loi_analysis(won_data, lost_data, tabs[2])
        show_sample_size_analysis(won_data, lost_data, tabs[3])
        show_multi_factor_analysis(won_data, lost_data, tabs[4])
        
    except Exception as e:
        logger.error(f"Error in analysis component: {e}", exc_info=True)
        st.error(f"An error occurred while rendering the analysis component: {str(e)}")