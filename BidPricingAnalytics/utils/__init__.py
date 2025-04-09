"""
Utilities module for the CPI Analysis & Prediction Dashboard.

This package centralizes data loading, processing, and visualization functions,
ensuring robust error handling and consistent styling across the dashboard.
"""

# Data loading and processing
from .data_loader import load_data

from .data_processor import (
    apply_all_bins,
    create_ir_bins,
    create_loi_bins,
    create_completes_bins,
    engineer_features,
    prepare_model_data,
    get_data_summary
)

# UI helper functions for visualizations
from ui_components import (
    apply_chart_styling,
    add_insights_annotation,
    add_data_point_annotation
)

# Configuration
from config import COLOR_SYSTEM, TYPOGRAPHY

# Basic Visualizations
from .visualization_basic import (
    WON_COLOR,
    LOST_COLOR,
    HEATMAP_COLORSCALE_WON,
    HEATMAP_COLORSCALE_LOST,
    COLORBLIND_PALETTE,
    create_type_distribution_chart,
    create_cpi_distribution_boxplot,
    create_cpi_histogram_comparison,
    create_cpi_efficiency_chart
)

# Analysis Visualizations
from .visualization_analysis import (
    create_cpi_vs_ir_scatter,
    create_bar_chart_by_bin,
    create_heatmap
)

# Prediction Visualizations
from .visualization_prediction import (
    create_feature_importance_chart,
    create_prediction_comparison_chart
)

# Robust Visualization API (with enhanced error handling)
from .visualization import (
    create_type_distribution_chart,
    create_cpi_distribution_boxplot,
    create_cpi_histogram_comparison,
    create_cpi_efficiency_chart,
    create_cpi_vs_ir_scatter,
    create_bar_chart_by_bin,
    create_heatmap,
    create_feature_importance_chart,
    create_prediction_comparison_chart
)

__all__ = [
    # Data loading and processing
    'load_data',
    'apply_all_bins',
    'create_ir_bins',
    'create_loi_bins',
    'create_completes_bins',
    'engineer_features',
    'prepare_model_data',
    'get_data_summary',
    
    # UI helpers for visualizations
    'apply_chart_styling',
    'add_insights_annotation',
    'add_data_point_annotation',
    
    # Constants and styling
    'WON_COLOR',
    'LOST_COLOR',
    'HEATMAP_COLORSCALE_WON',
    'HEATMAP_COLORSCALE_LOST',
    'COLORBLIND_PALETTE',
    'COLOR_SYSTEM',
    'TYPOGRAPHY',
    
    # Basic visualizations
    'create_type_distribution_chart',
    'create_cpi_distribution_boxplot',
    'create_cpi_histogram_comparison',
    'create_cpi_efficiency_chart',
    
    # Analysis visualizations
    'create_cpi_vs_ir_scatter',
    'create_bar_chart_by_bin',
    'create_heatmap',
    
    # Prediction visualizations
    'create_feature_importance_chart',
    'create_prediction_comparison_chart'
]
