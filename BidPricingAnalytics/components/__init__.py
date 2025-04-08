"""
Components module for the CPI Analysis & Prediction Dashboard.
This package contains UI components for different dashboard sections.
Provides a centralized import point for all dashboard component functions.
"""

# Import specific show functions from each component module
from .overview import show_overview
from .analysis import show_analysis
from .prediction import show_prediction
from .insights import show_insights

# Explicitly define what should be imported when using 'from components import *'
__all__ = [
    'show_overview',    # Overview dashboard component
    'show_analysis',    # Detailed CPI analysis component
    'show_prediction',  # CPI prediction and modeling component
    'show_insights'     # Insights and recommendations component
]