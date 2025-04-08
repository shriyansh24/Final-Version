"""
Analysis module for the CPI Analysis & Prediction Dashboard.
This package contains the analysis components for different factors.
Provides a centralized import point for analysis-related functions.
"""

# Import the main analysis function from basic analysis module
from .analysis_basic import show_basic_analysis as show_analysis

# Import the advanced analysis function from analysis_advanced
from .analysis_advanced import show_advanced_analysis

# Explicitly define what should be imported when using 'from analysis import *'
__all__ = [
    'show_analysis',           # Main analysis entry point (from basic analysis)
    'show_advanced_analysis'   # Advanced analysis entry point
]