"""
Models module for the CPI Analysis & Prediction Dashboard.

This package encapsulates all machine learning functionality including:

- Model training, evaluation, and hyperparameter tuning.
- Saving and loading of trained models.
- Generating CPI predictions and detailed pricing recommendations.
- Simulation of win probability and reporting of prediction metrics.

All functions include enhanced error handling, logging, and high-contrast styling 
for the modern dashboard theme.
"""

from .trainer import (
    build_models,
    save_models,
    load_models,
    cross_validate_models,
    get_model_results_html,
    get_feature_importance_html
)

from .predictor import (
    predict_cpi,
    get_recommendation,
    get_detailed_pricing_strategy,
    simulate_win_probability,
    get_prediction_metrics
)

__all__ = [
    "build_models",
    "save_models",
    "load_models",
    "cross_validate_models",
    "get_model_results_html",
    "get_feature_importance_html",
    "predict_cpi",
    "get_recommendation",
    "get_detailed_pricing_strategy",
    "simulate_win_probability",
    "get_prediction_metrics",
]
