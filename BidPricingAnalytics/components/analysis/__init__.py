"""
Analysis module for the CPI Analysis & Prediction Dashboard.
This package contains the analysis components for different factors.
Provides a centralized import point for analysis-related functions.
"""

# Import the main analysis function from basic analysis module
from .analysis_basic import show_analysis

# Import advanced analysis functions (optional, but can be useful)
from .analysis_advanced import (
    show_loi_analysis,
    show_sample_size_analysis,
    show_multi_factor_analysis
)

# Explicitly define what should be imported when using 'from analysis import *'
__all__ = [
    'show_analysis',            # Main analysis entry point
    'show_loi_analysis',        # Length of Interview specific analysis
    'show_sample_size_analysis',# Sample size specific analysis
    'show_multi_factor_analysis'# Advanced multi-factor analysis
]