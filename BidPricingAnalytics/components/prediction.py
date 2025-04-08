"""
CPI Prediction Component for Market Research Bid Pricing

This module provides an advanced machine learning-powered prediction tool 
that helps market researchers optimize their bid pricing strategy by 
analyzing key project parameters and historical data.

Key Features:
- Machine Learning Price Prediction
- Interactive Parameter Adjustment
- Win Probability Estimation
- Detailed Pricing Strategy Recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional

# Core Prediction Utilities
from models.trainer import build_models
from models.predictor import (
    predict_cpi, 
    get_recommendation, 
    get_detailed_pricing_strategy,
    simulate_win_probability
)

# Visualization and UI Components
from ui_components import (
    render_card, 
    metrics_row, 
    grid_layout
)

# Configuration and Styling
from config import COLOR_SYSTEM, TYPOGRAPHY

# Data Processing
from utils.data_processor import prepare_model_data

# Visualization Utilities
from utils.visualization import (
    create_feature_importance_chart,
    create_prediction_comparison_chart
)

# Configure logging for tracking and debugging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def show_prediction(
    combined_data_engineered: pd.DataFrame, 
    won_data: pd.DataFrame, 
    lost_data: pd.DataFrame
) -> None:
    """
    Advanced CPI Prediction Tool: A comprehensive pricing strategy generator
    
    This function creates an interactive Streamlit interface that allows users to:
    1. Explore predictive models' performance
    2. Input project parameters
    3. Generate CPI predictions
    4. Receive detailed pricing recommendations
    
    Args:
        combined_data_engineered (pd.DataFrame): Preprocessed historical bid data
        won_data (pd.DataFrame): Historical won bid data
        lost_data (pd.DataFrame): Historical lost bid data
    """
    try:
        # Page Introduction with Educational Context
        st.markdown(f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            border-left: 4px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
        ">
            <h1 style="
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-bottom: 1rem;
                font-size: 2rem;
                font-weight: 700;
            ">CPI Price Prediction Tool</h1>
            
            <p style="
                color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};
                font-size: 1rem;
                line-height: 1.6;
            ">
                💡 Machine Learning Meets Market Research Pricing
                
                This predictive tool helps you optimize your bid pricing by:
                • Analyzing historical bid data
                • Considering multiple project parameters
                • Providing data-driven pricing recommendations
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Data Validation and Model Preparation
        if len(combined_data_engineered) < 10:
            st.warning("""
            ### Insufficient Data for Prediction
            
            Our machine learning models require at least 10 historical data points 
            to generate reliable predictions. Please ensure you have:
            
            ✓ Comprehensive bid history
            ✓ Diverse project parameters
            ✓ Clean, structured data
            
            Recommendations:
            - Import more historical bid records
            - Verify data quality and completeness
            - Consult your data collection process
            """)
            return

        # Model Training with User Options
        with st.spinner("🔬 Training Predictive Models..."):
            try:
                # Prepare data for modeling
                X, y = prepare_model_data(combined_data_engineered)
                
                if len(X) == 0 or len(y) == 0:
                    st.error("Data preparation failed. Please review your dataset.")
                    return

                # Model training configuration
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    do_tuning = st.checkbox(
                        "Enable Advanced Model Tuning", 
                        help="More comprehensive but slower model optimization"
                    )
                
                with col2:
                    show_details = st.checkbox(
                        "Show Model Details", 
                        help="Display technical model performance metrics"
                    )

                # Build machine learning models
                trained_models, model_scores, feature_importance = build_models(X, y, do_tuning)
            
            except Exception as e:
                st.error(f"Model Training Error: {e}")
                logger.error(f"Model training failed: {e}", exc_info=True)
                return

        # Advanced Model Performance Section
        if show_details:
            st.header("🧠 Model Performance Insights")
            
            def render_model_performance_card(model_name, metrics):
                st.markdown(f"""
                <div style="
                    background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                    border-radius: 0.5rem;
                    padding: 1rem;
                    margin-bottom: 1rem;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                ">
                    <h3 style="
                        color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                        margin-bottom: 0.5rem;
                        font-size: 1.1rem;
                    ">{model_name} Performance</h3>
                    <div style="color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};">
                        {"".join(f"<p><strong>{metric}:</strong> {value:.4f}</p>" for metric, value in metrics.items())}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            grid_layout(
                len(model_scores), 
                [lambda m=model_name: render_model_performance_card(m, metrics) 
                 for model_name, metrics in model_scores.items()]
            )

            # Feature Importance Visualization
            if not feature_importance.empty:
                st.header("🔍 Feature Impact Analysis")
                render_card(
                    "Key Predictive Features", 
                    st.plotly_chart(
                        create_feature_importance_chart(feature_importance), 
                        use_container_width=True
                    )
                )

        # Interactive Prediction Input Section
        st.header("🎯 Predict Your CPI")
        
        # Dynamic range calculation for input sliders
        ir_config = {
            'min': max(1, int(combined_data_engineered['IR'].min())),
            'max': min(100, int(combined_data_engineered['IR'].max())),
            'default': int(combined_data_engineered['IR'].mean())
        }
        
        loi_config = {
            'min': max(1, int(combined_data_engineered['LOI'].min())),
            'max': min(60, int(combined_data_engineered['LOI'].max() * 1.2)),
            'default': int(combined_data_engineered['LOI'].mean())
        }
        
        completes_config = {
            'min': max(10, int(combined_data_engineered['Completes'].min())),
            'max': min(2000, int(combined_data_engineered['Completes'].max() * 1.2)),
            'default': int(combined_data_engineered['Completes'].mean())
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ir = st.slider(
                "🔢 Incidence Rate (%)", 
                min_value=ir_config['min'], 
                max_value=ir_config['max'], 
                value=ir_config['default'],
                help="Percentage of people who qualify for your survey"
            )
        
        with col2:
            loi = st.slider(
                "⏱ Length of Interview (min)", 
                min_value=loi_config['min'], 
                max_value=loi_config['max'], 
                value=loi_config['default'],
                help="Total duration of the survey"
            )
        
        with col3:
            completes = st.slider(
                "📊 Sample Size", 
                min_value=completes_config['min'], 
                max_value=completes_config['max'], 
                value=completes_config['default'],
                help="Number of completed survey responses"
            )
        
        # Prediction Trigger
        predict_button = st.button(
            "🚀 Generate CPI Prediction", 
            type="primary", 
            help="Calculate recommended Cost Per Interview based on project parameters"
        )
        
        # Prediction Results
        if predict_button:
            with st.spinner("🔮 Generating Intelligent Predictions..."):
                user_input = {
                    'IR': ir,
                    'LOI': loi,
                    'Completes': completes
                }
                
                # Core Prediction Logic
                predictions = predict_cpi(trained_models, user_input, X.columns)
                
                if not predictions:
                    st.error("Prediction generation failed. Please adjust input parameters.")
                    return
                
                # Calculate average prediction
                avg_prediction = sum(predictions.values()) / len(predictions)
                
                # Historical Comparisons
                won_avg = combined_data_engineered[
                    combined_data_engineered['Type'] == 'Won'
                ]['CPI'].mean()
                
                lost_avg = combined_data_engineered[
                    combined_data_engineered['Type'] == 'Lost'
                ]['CPI'].mean()
                
                # Prediction Visualization
                render_card(
                    "Prediction Comparison", 
                    st.plotly_chart(
                        create_prediction_comparison_chart(predictions, won_avg, lost_avg), 
                        use_container_width=True
                    )
                )
                
                # Individual and Average Predictions
                prediction_metrics = [
                    *[{
                        "label": f"{model} Prediction",
                        "value": f"${pred:.2f}"
                    } for model, pred in predictions.items()],
                    {
                        "label": "Average Prediction",
                        "value": f"${avg_prediction:.2f}",
                        "delta": f"{((avg_prediction - won_avg) / won_avg * 100):.1f}%"
                    }
                ]
                
                metrics_row(prediction_metrics)
                
                # Strategic Recommendation
                st.header("💡 Pricing Strategy Recommendation")
                
                recommendation = get_recommendation(avg_prediction, won_avg, lost_avg)
                render_card(
                    "Strategic Insight", 
                    f"""
                    <div style="
                        font-family: {TYPOGRAPHY['FONT_FAMILY']};
                        color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};
                    ">
                        <p><strong>Recommended Approach:</strong> {recommendation}</p>
                    </div>
                    """
                )
                
                # Win Probability Estimation
                win_prob = simulate_win_probability(
                    avg_prediction, 
                    user_input, 
                    won_data, 
                    lost_data
                )
                
                if win_prob:
                    st.metric(
                        "🏆 Estimated Win Probability", 
                        f"{win_prob['win_probability']:.1f}%",
                        help="Probability of winning based on historical bid patterns"
                    )
                
                # Detailed Strategic Guidance
                detailed_strategy = get_detailed_pricing_strategy(
                    avg_prediction, 
                    user_input, 
                    won_data, 
                    lost_data
                )
                
                with st.expander("📋 Comprehensive Pricing Strategy", expanded=False):
                    st.markdown(detailed_strategy)
        
    except Exception as e:
        logger.error(f"Unexpected error in prediction component: {e}", exc_info=True)
        st.error(f"An unexpected error occurred: {str(e)}")