"""
Components module for the CPI Analysis & Prediction Dashboard.

This package provides all dashboard components for the various sections
of the application with proper organization and navigation.
"""

# Import main dashboard components
from .overview import show_overview
from .insights import show_insights
from .prediction import show_prediction

# Import analysis components 
# The analysis subfolder has its own __init__.py that exports show_analysis
from .analysis import show_analysis
from .analysis import show_basic_analysis, show_advanced_analysis

# Export all public functions
__all__ = [
    # Main dashboard components
    "show_overview",     # Overview dashboard component
    "show_analysis",     # Detailed CPI analysis component
    "show_prediction",   # CPI prediction and modeling component
    "show_insights",     # Insights and recommendations component
    
    # Individual analysis components (for direct access)
    "show_basic_analysis",   # Basic relationship analysis
    "show_advanced_analysis" # Advanced 3D and heatmap analysis
]
