"""
Utilities module for the CPI Analysis & Prediction Dashboard.
This package contains utility functions for data processing and visualization.
"""

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

# Import visualization functions explicitly from each module
# Basic visualizations
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

# Analysis visualizations
from .visualization_analysis import (
    create_cpi_vs_ir_scatter,
    create_bar_chart_by_bin,
    create_heatmap
)

# Prediction visualizations
from .visualization_prediction import (
    create_feature_importance_chart,
    create_prediction_comparison_chart
)

__all__ = [
    # Data loading
    'load_data',
    
    # Data processing
    'apply_all_bins',
    'create_ir_bins',
    'create_loi_bins',
    'create_completes_bins',
    'engineer_features',
    'prepare_model_data',
    'get_data_summary',
    
    # Constants
    'WON_COLOR',
    'LOST_COLOR',
    'HEATMAP_COLORSCALE_WON',
    'HEATMAP_COLORSCALE_LOST',
    'COLORBLIND_PALETTE',
    
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