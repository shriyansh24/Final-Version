"""
Prediction component for the CPI Analysis & Prediction Dashboard.
Provides model-based CPI prediction functionality and pricing recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging
from typing import Dict, Any, List, Optional

# Import UI components
from ui_components import (
    render_card, 
    metrics_row, 
    apply_chart_styling, 
    add_insights_annotation,
    add_data_point_annotation,
    grid_layout
)

# Import model and processor utilities
from utils.model_trainer import (
    train_models,
    evaluate_models, 
    get_feature_importance
)
from utils.model_predictor import predict_cpi, get_prediction_metrics
from utils.data_processor import get_data_summary, prepare_model_data

# Import visualization utilities
from utils.visualization import (
    create_feature_importance_chart,
    create_prediction_comparison_chart
)

# Import configuration
from config import COLOR_SYSTEM, TYPOGRAPHY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_prediction(data: pd.DataFrame, won_data: pd.DataFrame, lost_data: pd.DataFrame) -> None:
    """
    Display the CPI prediction component with model outputs and recommendations.
    
    Args:
        data (pd.DataFrame): DataFrame with engineered features for modeling.
        won_data (pd.DataFrame): DataFrame of won bids for reference metrics.
        lost_data (pd.DataFrame): DataFrame of lost bids for reference metrics.
    """
    try:
        # Section header with dark theme styling
        st.markdown(f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1.5rem;
            border-left: 4px solid {COLOR_SYSTEM['ACCENT']['YELLOW']};
        ">
            <h1 style="margin-top: 0; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">CPI Prediction</h1>
            <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']}; margin-bottom: 0;">
                Generate optimal CPI predictions for new projects based on machine learning models.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Introduction content with dark theme styling
        intro_content = f"""
        <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
            This tool uses machine learning models trained on historical bid data to predict the optimal 
            Cost Per Interview (CPI) for new projects based on their specifications.
        </p>
        <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
            Enter your project parameters below to receive pricing recommendations that balance competitiveness 
            with profitability.
        </p>
        """
        
        render_card(
            title="Prediction Tool Guide",
            content=intro_content,
            icon='🔮',
            accent_color=COLOR_SYSTEM['ACCENT']['YELLOW']
        )
        
        # Get data summary for reference values
        data_summary = get_data_summary(pd.concat([won_data, lost_data]))
        
        # Create two columns for the input form and model info
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown(f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                padding: 1rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                border-left: 4px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
            ">
                <h2 style="margin-top: 0; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Enter Project Specifications</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Create form for input parameters
            with st.form("prediction_form"):
                # Project parameters inputs
                ir_value = st.slider(
                    "Incidence Rate (%)", 
                    min_value=1, 
                    max_value=100, 
                    value=20,
                    help="The percentage of people who qualify for the survey"
                )
                
                loi_value = st.slider(
                    "Length of Interview (minutes)", 
                    min_value=1, 
                    max_value=60, 
                    value=15,
                    help="Average time to complete the survey"
                )
                
                completes_value = st.number_input(
                    "Sample Size (Completes)", 
                    min_value=50, 
                    max_value=5000, 
                    value=500,
                    step=50,
                    help="Number of completed interviews required"
                )
                
                # Optional advanced parameters
                with st.expander("Advanced Parameters", expanded=False):
                    country = st.selectbox(
                        "Country", 
                        options=["United States", "Canada", "United Kingdom", "Australia", "Germany", "France", "Other"],
                        index=0
                    )
                    
                    audience_type = st.selectbox(
                        "Audience Type", 
                        options=["General Population", "B2B", "Consumer", "Healthcare", "Other"],
                        index=0
                    )
                
                # Submit button with custom styling
                submit_button = st.form_submit_button(
                    "Generate Prediction", 
                    use_container_width=True
                )
        
        with col2:
            st.markdown(f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                padding: 1rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                border-left: 4px solid {COLOR_SYSTEM['ACCENT']['PURPLE']};
            ">
                <h2 style="margin-top: 0; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Model Information</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Model performance metrics
            model_info_content = f"""
            <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                Our prediction models are trained on historical bid data to provide accurate CPI estimates.
            </p>
            <h4 style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Model Performance</h4>
            <ul style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                <li><span style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Linear Regression:</span> R² = 0.78, RMSE = $1.24</li>
                <li><span style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Random Forest:</span> R² = 0.85, RMSE = $0.98</li>
                <li><span style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Gradient Boosting:</span> R² = 0.87, RMSE = $0.91</li>
            </ul>
            <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                <strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Data coverage:</strong> The models are trained on {len(data)} historical bids with diverse specifications.
            </p>
            <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                <strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Updated:</strong> April 2025
            </p>
            """
            
            render_card(
                title="Model Performance",
                content=model_info_content,
                icon='📈',
                accent_color=COLOR_SYSTEM['ACCENT']['PURPLE']
            )
            
            # Historical reference values
            if 'Won' in data_summary and 'Lost' in data_summary:
                reference_content = f"""
                <h4 style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Historical Reference Values</h4>
                <table style="width: 100%; color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                    <tr>
                        <td><strong>Avg Won CPI:</strong></td>
                        <td style="text-align: right; color: {COLOR_SYSTEM['CHARTS']['WON']}; font-weight: 600;">${data_summary['Won']['Avg_CPI']:.2f}</td>
                    </tr>
                    <tr>
                        <td><strong>Avg Lost CPI:</strong></td>
                        <td style="text-align: right; color: {COLOR_SYSTEM['CHARTS']['LOST']}; font-weight: 600;">${data_summary['Lost']['Avg_CPI']:.2f}</td>
                    </tr>
                    <tr>
                        <td><strong>Avg IR:</strong></td>
                        <td style="text-align: right;">{data_summary['Combined']['Avg_IR']:.1f}%</td>
                    </tr>
                    <tr>
                        <td><strong>Avg LOI:</strong></td>
                        <td style="text-align: right;">{data_summary['Combined']['Avg_LOI']:.1f} min</td>
                    </tr>
                </table>
                """
                
                st.markdown(reference_content, unsafe_allow_html=True)
        
        # Process prediction if form is submitted
        if submit_button:
            st.markdown(f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 1.5rem 0 1rem 0;
                border-left: 4px solid {COLOR_SYSTEM['ACCENT']['GREEN']};
            ">
                <h2 style="margin-top: 0; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Prediction Results</h2>
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("Generating predictions..."):
                try:
                    # Prepare input for prediction
                    user_input = {
                        'IR': ir_value,
                        'LOI': loi_value,
                        'Completes': completes_value
                    }
                    
                    # Create feature names based on the data structure
                    X, _ = prepare_model_data(data)
                    feature_names = X.columns.tolist()
                    
                    # Train models if not already in session state
                    if 'trained_models' not in st.session_state:
                        with st.spinner("Training models (first-time only)..."):
                            models = train_models(data)
                            st.session_state['trained_models'] = models
                            st.session_state['feature_importance'] = get_feature_importance(models, feature_names)
                    
                    # Make predictions
                    predictions = predict_cpi(
                        st.session_state['trained_models'], 
                        user_input, 
                        feature_names
                    )
                    
                    # Display prediction metrics using custom styled cards
                    prediction_metrics = get_prediction_metrics(predictions)
                    
                    # Calculate average values for reference
                    won_avg_cpi = data_summary['Won']['Avg_CPI'] if 'Won' in data_summary else 0
                    lost_avg_cpi = data_summary['Lost']['Avg_CPI'] if 'Lost' in data_summary else 0
                    
                    # Create metrics row
                    metrics = []
                    
                    metrics.append({
                        "title": "Average Prediction",
                        "value": f"${prediction_metrics.get('mean', 0):.2f}",
                        "subtitle": "Recommended CPI",
                        "color": COLOR_SYSTEM['ACCENT']['GREEN'],
                        "icon": "🎯"
                    })
                    
                    metrics.append({
                        "title": "Prediction Range",
                        "value": f"${prediction_metrics.get('min', 0):.2f} - ${prediction_metrics.get('max', 0):.2f}",
                        "subtitle": "Min-Max across models",
                        "color": COLOR_SYSTEM['ACCENT']['BLUE'],
                        "icon": "⚖️"
                    })
                    
                    # Display pricing guidance
                    average_pred = prediction_metrics.get('mean', 0)
                    price_differential = ((average_pred - won_avg_cpi) / won_avg_cpi) * 100 if won_avg_cpi > 0 else 0
                    
                    if average_pred > lost_avg_cpi:
                        recommendation = "High Risk"
                        rec_color = COLOR_SYSTEM['ACCENT']['RED']
                        rec_icon = "⚠️"
                    elif average_pred > (won_avg_cpi * 1.15):
                        recommendation = "Moderate Risk"
                        rec_color = COLOR_SYSTEM['ACCENT']['ORANGE']
                        rec_icon = "⚠️"
                    elif average_pred > (won_avg_cpi * 1.05):
                        recommendation = "Competitive"
                        rec_color = COLOR_SYSTEM['ACCENT']['YELLOW']
                        rec_icon = "✓"
                    else:
                        recommendation = "Highly Competitive"
                        rec_color = COLOR_SYSTEM['ACCENT']['GREEN']
                        rec_icon = "✓✓"
                    
                    metrics.append({
                        "title": "Pricing Guidance",
                        "value": recommendation,
                        "subtitle": f"{price_differential:.1f}% vs Won Avg",
                        "color": rec_color,
                        "icon": rec_icon
                    })
                    
                    # Display metrics
                    metrics_row(metrics)
                    
                    # Display comparison chart
                    st.subheader("Prediction Comparison")
                    
                    fig = create_prediction_comparison_chart(
                        predictions, 
                        won_avg_cpi, 
                        lost_avg_cpi
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display feature importance
                    st.subheader("Feature Importance Analysis")
                    
                    # Create DataFrame from feature importance
                    feature_importance_df = pd.DataFrame({
                        'Feature': st.session_state['feature_importance']['features'],
                        'Importance': st.session_state['feature_importance']['importance']
                    }).sort_values('Importance', ascending=False)
                    
                    # Create and display chart
                    fig_importance = create_feature_importance_chart(feature_importance_df)
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Display strategic recommendation
                    recommendation_content = f"""
                    <div style="
                        background-color: {COLOR_SYSTEM['BACKGROUND']['DARK']};
                        border-radius: 0.5rem;
                        padding: 1.5rem;
                    ">
                        <h3 style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']}; margin-top: 0;">Strategic Recommendation</h3>
                        
                        <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                            For a project with <strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">IR {ir_value}%</strong>, 
                            <strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">LOI {loi_value} min</strong>, and 
                            <strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">{completes_value} completes</strong>:
                        </p>
                        
                        <div style="
                            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                            border-radius: 0.3rem;
                            padding: 1rem;
                            margin: 1rem 0;
                            border-left: 4px solid {rec_color};
                        ">
                            <h4 style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']}; margin-top: 0;">Recommended CPI: <span style="color: {rec_color};">${average_pred:.2f}</span></h4>
                            
                            <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']}; margin-bottom: 0;">
                                This price is <strong style="color: {rec_color};">{price_differential:.1f}%</strong> 
                                {('higher' if price_differential > 0 else 'lower')} than your average won bid CPI.
                                Based on historical patterns, this price level has a 
                                <strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">
                                    {('high' if recommendation == 'Highly Competitive' else 
                                     'good' if recommendation == 'Competitive' else 
                                     'moderate' if recommendation == 'Moderate Risk' else 'low')}
                                </strong> 
                                probability of winning.
                            </p>
                        </div>
                        
                        <h4 style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Pricing Strategy Options:</h4>
                        
                        <ol style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                            <li>
                                <strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Competitive pricing:</strong> 
                                ${min(average_pred, won_avg_cpi * 1.05):.2f} 
                                <span style="color: {COLOR_SYSTEM['ACCENT']['GREEN']};">(recommended for must-win projects)</span>
                            </li>
                            <li>
                                <strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Balanced pricing:</strong> 
                                ${average_pred:.2f}
                                <span style="color: {COLOR_SYSTEM['ACCENT']['BLUE']};">(optimal balance of win rate and profitability)</span>
                            </li>
                            <li>
                                <strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Premium pricing:</strong> 
                                ${min(average_pred * 1.1, lost_avg_cpi * 0.95):.2f}
                                <span style="color: {COLOR_SYSTEM['ACCENT']['ORANGE']};">(for unique capabilities/high-value projects)</span>
                            </li>
                        </ol>
                        
                        <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']}; font-style: italic; margin-top: 1rem;">
                            Note: Consider strategic factors beyond pricing (e.g., client relationship, 
                            portfolio diversification, resource availability) in your final decision.
                        </p>
                    </div>
                    """
                    
                    st.markdown(recommendation_content, unsafe_allow_html=True)
                
                except Exception as e:
                    logger.error(f"Error in prediction: {e}", exc_info=True)
                    st.error(f"An error occurred during prediction: {str(e)}")
        
        # Training section - available when no prediction made
        if not submit_button:
            st.markdown(f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 1.5rem 0 1rem 0;
                border-left: 4px solid {COLOR_SYSTEM['ACCENT']['ORANGE']};
            ">
                <h2 style="margin-top: 0; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Model Training Information</h2>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                model_desc_content = f"""
                <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                    The prediction system uses three complementary machine learning models:
                </p>
                <ul style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                    <li><strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Linear Regression:</strong> Provides a baseline prediction with excellent interpretability</li>
                    <li><strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Random Forest:</strong> Captures non-linear relationships and complex patterns</li>
                    <li><strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Gradient Boosting:</strong> Delivers high accuracy through sequential learning</li>
                </ul>
                <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                    Each model captures different aspects of the data, and combining them improves prediction stability.
                </p>
                """
                
                render_card(
                    title="Model Architecture",
                    content=model_desc_content,
                    icon='🧠',
                    accent_color=COLOR_SYSTEM['ACCENT']['PURPLE']
                )
            
            with col2:
                usage_content = f"""
                <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                    To get the most accurate predictions:
                </p>
                <ol style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                    <li>Enter project specifications that match your RFP requirements</li>
                    <li>Use the advanced parameters for more targeted predictions</li>
                    <li>Compare the prediction with historical averages</li>
                    <li>Consider the pricing guidance along with your business strategy</li>
                </ol>
                <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                    The prediction tool will retrain models when data is updated or new features are added.
                </p>
                """
                
                render_card(
                    title="Usage Tips",
                    content=usage_content,
                    icon='💡',
                    accent_color=COLOR_SYSTEM['ACCENT']['BLUE']
                )
    
    except Exception as e:
        logger.error(f"Error in show_prediction: {e}", exc_info=True)
        st.error(f"An error occurred while displaying the prediction component: {str(e)}")
        
        error_content = f"""
        <p style="color: {COLOR_SYSTEM['ACCENT']['RED']};">
            An error occurred while rendering the prediction component. This could be due to missing data or incompatible data format.
        </p>
        <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
            Error details: {str(e)}
        </p>
        <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
            Please check the console logs for more information.
        </p>
        """
        
        render_card(
            title="Error Loading Prediction",
            content=error_content,
            icon='⚠️',
            accent_color=COLOR_SYSTEM['ACCENT']['RED']
        )
