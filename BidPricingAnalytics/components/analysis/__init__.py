"""
Analysis module for the CPI Analysis & Prediction Dashboard.
This package contains components for both basic and advanced analysis.
"""

from .analysis_basic import show_basic_analysis
from .analysis_advanced import show_advanced_analysis

# Combined analysis function that shows both basic and advanced components
def show_analysis(won_data, lost_data, combined_data):
    """
    Main entry point for the Analysis section of the dashboard.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids.
        lost_data (pd.DataFrame): DataFrame of Lost bids.
        combined_data (pd.DataFrame): Combined DataFrame of all bids.
    """
    import streamlit as st
    
    # Title and introduction
    st.markdown("""
    <div style="
        background-color: #1F1F1F;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border-left: 4px solid #00BFFF;
    ">
        <h1 style="color: #FFFFFF; margin-top: 0;">CPI Analysis</h1>
        <p style="color: #B0B0B0; margin-bottom: 0;">
            Comprehensive analysis of factors that influence Cost Per Interview (CPI) and their relationships.
            Use these insights to inform your pricing strategy and improve win rates.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for basic and advanced analysis
    tab1, tab2 = st.tabs(["Basic Analysis", "Advanced Analysis"])
    
    with tab1:
        show_basic_analysis(won_data, lost_data)
        
    with tab2:
        show_advanced_analysis(won_data, lost_data, combined_data)

__all__ = [
    'show_analysis',
    'show_basic_analysis',
    'show_advanced_analysis'
]
