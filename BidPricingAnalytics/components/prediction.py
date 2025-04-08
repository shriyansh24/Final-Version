"""
Prediction component for the CPI Analysis & Prediction Dashboard.
Handles model predictions and recommendation display.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import UI components
from ui_components import (
    render_card, metrics_row, apply_chart_styling,
    add_insights_annotation, grid_layout, render_icon_tabs
)

# Import visualization utilities
from utils.visualization import (
    create_feature_importance_chart,
    create_prediction_comparison_chart
)

# Import data utilities
from utils.data_processor import get_data_summary, prepare_model_data

# Import configuration
from config import (
    COLOR_SYSTEM, TYPOGRAPHY, RANDOM_STATE, TEST_SIZE, DEFAULT_MODELS
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(X, y, model_type='Random Forest'):
    """
    Train a model based on the specified type.
    
    Args:
        X (pd.DataFrame): Features dataframe
        y (pd.Series): Target variable
        model_type (str): Type of model to train
    
    Returns:
        tuple: (model, X_train, X_test, y_train, y_test)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Select and train model
    if model_type == 'Linear Regression':
        model = LinearRegression(**DEFAULT_MODELS['Linear Regression'])
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingRegressor(**DEFAULT_MODELS['Gradient Boosting'])
    else:  # Default to Random Forest
        model = RandomForestRegressor(**DEFAULT_MODELS['Random Forest'])
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target values
    
    Returns:
        dict: Evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'y_test': y_test,
        'y_pred': y_pred
    }

def get_feature_importance(model, feature_names, model_type):
    """
    Extract feature importance from the model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        model_type: Type of model
    
    Returns:
        pd.DataFrame: Feature importance dataframe
    """
    if model_type == 'Linear Regression':
        importance = np.abs(model.coef_)
    else:  # Random Forest or Gradient Boosting
        importance = model.feature_importances_
    
    # Create dataframe
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    return feature_importance

def predict_price(model, feature_values: Dict[str, float], feature_names: List[str]):
    """
    Make a prediction using the trained model.
    
    Args:
        model: Trained model
        feature_values (Dict[str, float]): Dictionary of feature values
        feature_names (List[str]): List of feature names
    
    Returns:
        float: Predicted CPI
    """
    # Create feature array
    features = np.array([feature_values.get(name, 0) for name in feature_names]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    return prediction

def show_prediction(data: pd.DataFrame, won_data: pd.DataFrame, lost_data: pd.DataFrame) -> None:
    """
    Display the prediction component with model training, evaluation, and prediction interface.
    
    Args:
        data (pd.DataFrame): Engineered data for modeling
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
    """
    try:
        # Add page header
        st.markdown(f"""
        <h2 style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            font-size: {TYPOGRAPHY['HEADING']['H2']['size']};
            font-weight: {TYPOGRAPHY['HEADING']['H2']['weight']};
            margin-bottom: 1rem;
        ">CPI Prediction & Price Optimization</h2>
        """, unsafe_allow_html=True)
        
        # Add introduction card
        intro_content = f"""
        <p style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
        ">
            This section lets you predict optimal CPI values based on project parameters.
            The model analyzes historical won and lost bids to recommend competitive pricing.
        </p>
        <p style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            margin-top: 0.5rem;
        ">
            <strong>How to use:</strong> Either train a new model or use the pre-trained model to 
            get CPI predictions based on project parameters.
        </p>
        """
        
        render_card(
            title="CPI Prediction Tools", 
            content=intro_content,
            icon=f'<span style="font-size: 1.5rem; color: {COLOR_SYSTEM["ACCENT"]["PURPLE"]};">🔮</span>'
        )
        
        # Prepare data for modeling
        X, y = prepare_model_data(data)
        
        if X.empty or len(y) == 0:
            st.error("Insufficient data for modeling. Please check your data.")
            return
        
        # Model training section
        st.markdown(f"""
        <h3 style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            font-size: {TYPOGRAPHY['HEADING']['H3']['size']};
            font-weight: {TYPOGRAPHY['HEADING']['H3']['weight']};
            margin: 1.5rem 0 1rem 0;
        ">Price Prediction & Recommendation</h3>
        """, unsafe_allow_html=True)
        
        # Create explanation card
        prediction_intro = f"""
        <p style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
        ">
            Enter the project parameters below to get a predicted CPI and pricing recommendation.
            The prediction is based on historical data patterns and the selected model.
        </p>
        <p style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            margin-top: 0.5rem;
        ">
            <strong>Note:</strong> The predicted CPI is a starting point. Consider additional
            factors such as client relationship, competition, and strategic importance.
        </p>
        """
        
        render_card(
            title="CPI Prediction Tool", 
            content=prediction_intro,
            icon=f'<span style="font-size: 1.5rem; color: {COLOR_SYSTEM["ACCENT"]["BLUE"]};">💰</span>'
        )
        
        # Check if a model is trained or load a default model
        if 'model' not in st.session_state:
            # Train a default model if not already trained
            with st.spinner("Preparing default prediction model..."):
                model, X_train, X_test, y_train, y_test = train_model(X, y, 'Random Forest')
                
                # Store model in session state
                st.session_state['model'] = model
                st.session_state['model_type'] = 'Random Forest'
                st.session_state['feature_names'] = X.columns.tolist()
                
                # Evaluate the model
                metrics = evaluate_model(model, X_test, y_test)
                st.session_state['model_metrics'] = metrics
                
                # Get feature importance
                feature_importance = get_feature_importance(model, X.columns, 'Random Forest')
                st.session_state['feature_importance'] = feature_importance
        
        # Create input form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ir_value = st.slider(
                "Incidence Rate (%)",
                min_value=float(data['IR'].min()),
                max_value=float(data['IR'].max()),
                value=float(data['IR'].median()),
                step=1.0,
                help="Percentage of people who qualify for the survey"
            )
        
        with col2:
            loi_value = st.slider(
                "Length of Interview (min)",
                min_value=float(data['LOI'].min()),
                max_value=float(data['LOI'].max()),
                value=float(data['LOI'].median()),
                step=1.0,
                help="Average survey duration in minutes"
            )
        
        with col3:
            completes_value = st.slider(
                "Sample Size (Completes)",
                min_value=int(data['Completes'].min()),
                max_value=int(data['Completes'].max()),
                value=int(data['Completes'].median()),
                step=50,
                help="Number of completed surveys required"
            )
        
        # Add additional parameters if available in the model
        feature_values = {
            'IR': ir_value,
            'LOI': loi_value,
            'Completes': completes_value
        }
        
        # Add Type_Won if it exists in the feature names
        if 'Type_Won' in st.session_state.get('feature_names', []):
            feature_values['Type_Won'] = 1  # Set to 1 to predict for won bids
        
        # Calculate derived features based on primary inputs
        feature_values['IR_LOI_Ratio'] = feature_values['IR'] / feature_values['LOI']
        feature_values['IR_Completes_Ratio'] = feature_values['IR'] / feature_values['Completes']
        feature_values['LOI_Completes_Ratio'] = feature_values['LOI'] / feature_values['Completes']
        
        # Add log transformations if they exist in the feature names
        if 'Log_Completes' in st.session_state.get('feature_names', []):
            feature_values['Log_Completes'] = np.log1p(feature_values['Completes'])
        if 'Log_IR' in st.session_state.get('feature_names', []):
            feature_values['Log_IR'] = np.log1p(feature_values['IR'])
        if 'Log_LOI' in st.session_state.get('feature_names', []):
            feature_values['Log_LOI'] = np.log1p(feature_values['LOI'])
        
        # Add interaction terms if they exist in the feature names
        if 'IR_LOI_Product' in st.session_state.get('feature_names', []):
            feature_values['IR_LOI_Product'] = feature_values['IR'] * feature_values['LOI']
        if 'IR_Completes_Product' in st.session_state.get('feature_names', []):
            feature_values['IR_Completes_Product'] = feature_values['IR'] * feature_values['Completes']
        
        # Add efficiency metric
        feature_values['CPI_Efficiency'] = (feature_values['IR'] / 100) * (1 / feature_values['LOI']) * feature_values['Completes']
        
        # Add normalized features if they exist
        if 'IR_Normalized' in st.session_state.get('feature_names', []):
            # Calculate using the formula from data_processor.py
            ir_mean = data['IR'].mean()
            ir_std = data['IR'].std()
            feature_values['IR_Normalized'] = (feature_values['IR'] - ir_mean) / ir_std
        
        if 'LOI_Normalized' in st.session_state.get('feature_names', []):
            loi_mean = data['LOI'].mean()
            loi_std = data['LOI'].std()
            feature_values['LOI_Normalized'] = (feature_values['LOI'] - loi_mean) / loi_std
            
        if 'Completes_Normalized' in st.session_state.get('feature_names', []):
            completes_mean = data['Completes'].mean()
            completes_std = data['Completes'].std()
            feature_values['Completes_Normalized'] = (feature_values['Completes'] - completes_mean) / completes_std
        
        # Get won and lost averages for comparison
        data_summary = get_data_summary(data)
        won_avg = data_summary.get('Won', {}).get('Avg_CPI', 0)
        lost_avg = data_summary.get('Lost', {}).get('Avg_CPI', 0)
        
        # Get similar projects for reference
        # Filter data for similar projects (within certain ranges)
        ir_range = 10  # IR +/- 10 percentage points
        loi_range = 5  # LOI +/- 5 minutes
        
        similar_won = won_data[
            (won_data['IR'] >= ir_value - ir_range) &
            (won_data['IR'] <= ir_value + ir_range) &
            (won_data['LOI'] >= loi_value - loi_range) &
            (won_data['LOI'] <= loi_value + loi_range)
        ]
        
        similar_lost = lost_data[
            (lost_data['IR'] >= ir_value - ir_range) &
            (lost_data['IR'] <= ir_value + ir_range) &
            (lost_data['LOI'] >= loi_value - loi_range) &
            (lost_data['LOI'] <= loi_value + loi_range)
        ]
        
        # Get averages of similar projects
        similar_won_avg = similar_won['CPI'].mean() if not similar_won.empty else won_avg
        similar_lost_avg = similar_lost['CPI'].mean() if not similar_lost.empty else lost_avg
        
        # Predict CPI button
        if st.button("Generate CPI Prediction", type="primary"):
            with st.spinner("Calculating optimal CPI..."):
                # Make prediction
                predicted_cpi = predict_price(
                    st.session_state['model'],
                    feature_values,
                    st.session_state['feature_names']
                )
                
                # Store prediction in session state
                st.session_state['predicted_cpi'] = predicted_cpi
                st.session_state['similar_won_avg'] = similar_won_avg
                st.session_state['similar_lost_avg'] = similar_lost_avg
                
                # Calculate additional predictions with different models
                all_predictions = {}
                
                # Add the current model prediction
                all_predictions[f"Current Model ({st.session_state['model_type']})"] = predicted_cpi
                
                # Train other models for comparison if not already in session state
                if 'all_model_predictions' not in st.session_state:
                    other_models = [m for m in ["Random Forest", "Linear Regression", "Gradient Boosting"] 
                                    if m != st.session_state['model_type']]
                    
                    for model_name in other_models:
                        other_model, _, _, _, _ = train_model(X, y, model_name)
                        other_pred = predict_price(other_model, feature_values, st.session_state['feature_names'])
                        all_predictions[model_name] = other_pred
                else:
                    # Use cached predictions
                    all_predictions.update(st.session_state['all_model_predictions'])
                
                # Store all predictions
                st.session_state['all_model_predictions'] = all_predictions
                
                # Success message
                st.success("CPI prediction generated successfully!")
        
        # Display prediction if available
        if 'predicted_cpi' in st.session_state:
            predicted_cpi = st.session_state['predicted_cpi']
            similar_won_avg = st.session_state['similar_won_avg']
            similar_lost_avg = st.session_state['similar_lost_avg']
            
            st.markdown("<hr>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <h4 style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                font-size: {TYPOGRAPHY['HEADING']['H4']['size']};
                font-weight: {TYPOGRAPHY['HEADING']['H4']['weight']};
                margin: 1rem 0 0.5rem 0;
            ">CPI Prediction Results</h4>
            """, unsafe_allow_html=True)
            
            # Determine if the predicted CPI is close to won or lost
            is_competitive = predicted_cpi <= (similar_won_avg * 1.1)  # Within 10% of won average
            is_high_risk = predicted_cpi >= (similar_lost_avg * 0.9)  # Within 90% of lost average
            
            pricing_category = "Competitive" if is_competitive else "High Risk" if is_high_risk else "Moderate"
            
            pricing_color = (
                COLOR_SYSTEM['ACCENT']['GREEN'] if pricing_category == "Competitive" else
                COLOR_SYSTEM['ACCENT']['RED'] if pricing_category == "High Risk" else
                COLOR_SYSTEM['ACCENT']['YELLOW']
            )
            
            # Create metrics for CPI values
            metrics_data = [
                {
                    "label": "Predicted CPI",
                    "value": f"${predicted_cpi:.2f}",
                    "delta": None
                },
                {
                    "label": "Similar Won CPI",
                    "value": f"${similar_won_avg:.2f}",
                    "delta": f"{((predicted_cpi / similar_won_avg) - 1) * 100:.1f}%" if similar_won_avg > 0 else None,
                    "delta_color": "normal"
                },
                {
                    "label": "Similar Lost CPI",
                    "value": f"${similar_lost_avg:.2f}",
                    "delta": f"{((predicted_cpi / similar_lost_avg) - 1) * 100:.1f}%" if similar_lost_avg > 0 else None,
                    "delta_color": "normal"
                }
            ]
            
            metrics_row(metrics_data)
            
            # Pricing recommendation card
            recommendation_content = f"""
            <div style="
                display: flex;
                align-items: center;
                margin-bottom: 1rem;
            ">
                <div style="
                    width: 20px;
                    height: 20px;
                    border-radius: 50%;
                    background-color: {pricing_color};
                    margin-right: 10px;
                "></div>
                <h4 style="
                    margin: 0;
                    color: {pricing_color};
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: 1.2rem;
                    font-weight: 600;
                ">
                    {pricing_category} Pricing
                </h4>
            </div>
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                <strong>Project parameters:</strong> IR {ir_value:.1f}%, LOI {loi_value:.1f} min, Sample size {completes_value}
            </p>
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-top: 0.5rem;
            ">
                <strong>Recommended price range:</strong>
            """
            
            # Different recommendation based on category
            if pricing_category == "Competitive":
                min_price = predicted_cpi * 0.95
                max_price = similar_won_avg * 1.1
                
                recommendation_content += f"""
                ${min_price:.2f} - ${max_price:.2f}
                </p>
                <p style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                    margin-top: 0.5rem;
                ">
                    This pricing is in line with historical won bids for similar projects.
                    You have flexibility to slightly increase price while remaining competitive.
                </p>
                """
            elif pricing_category == "High Risk":
                min_price = similar_won_avg * 0.9
                max_price = similar_won_avg * 1.05
                
                recommendation_content += f"""
                ${min_price:.2f} - ${max_price:.2f} <span style="color: {COLOR_SYSTEM['ACCENT']['RED']};">(Reduce from predicted)</span>
                </p>
                <p style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                    margin-top: 0.5rem;
                ">
                    <strong>Warning:</strong> The predicted price is close to or above historically lost bids.
                    Consider reducing the price to improve win probability.
                </p>
                """
            else:  # Moderate
                min_price = predicted_cpi * 0.95
                max_price = predicted_cpi * 1.05
                
                recommendation_content += f"""
                ${min_price:.2f} - ${max_price:.2f}
                </p>
                <p style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                    margin-top: 0.5rem;
                ">
                    This pricing falls between historical won and lost bids.
                    Consider factors beyond price such as client relationship, 
                    competition, and strategic importance of the project.
                </p>
                """
            
            render_card(
                title="Pricing Recommendation", 
                content=recommendation_content,
                accent_color=pricing_color
            )
            
            # Display model comparison
            if 'all_model_predictions' in st.session_state:
                # Create comparison chart
                fig = create_prediction_comparison_chart(
                    st.session_state['all_model_predictions'],
                    similar_won_avg,
                    similar_lost_avg
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Add project profitability calculator
            st.markdown(f"""
            <h4 style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                font-size: {TYPOGRAPHY['HEADING']['H4']['size']};
                font-weight: {TYPOGRAPHY['HEADING']['H4']['weight']};
                margin: 1.5rem 0 0.5rem 0;
            ">Project Profitability Calculator</h4>
            """, unsafe_allow_html=True)
            
            # Allow user to adjust pricing
            col1, col2 = st.columns(2)
            
            with col1:
                custom_cpi = st.number_input(
                    "Custom CPI ($)",
                    min_value=0.0,
                    max_value=float(data['CPI'].max() * 1.5),
                    value=float(predicted_cpi),
                    step=0.1,
                    format="%.2f",
                    help="Enter a custom CPI to calculate project profitability"
                )
            
            with col2:
                cost_per_complete = st.number_input(
                    "Cost Per Complete ($)",
                    min_value=0.0,
                    max_value=float(custom_cpi * 0.95),
                    value=float(custom_cpi * 0.75),  # Default to 75% of CPI as cost
                    step=0.1,
                    format="%.2f",
                    help="Estimated cost to your company per complete"
                )
            
            # Calculate profitability metrics
            total_revenue = custom_cpi * completes_value
            total_cost = cost_per_complete * completes_value
            profit = total_revenue - total_cost
            profit_margin = (profit / total_revenue) * 100 if total_revenue > 0 else 0
            
            # Display profitability metrics
            profitability_metrics = [
                {
                    "label": "Total Revenue",
                    "value": f"${total_revenue:.2f}",
                    "delta": None
                },
                {
                    "label": "Estimated Cost",
                    "value": f"${total_cost:.2f}",
                    "delta": None
                },
                {
                    "label": "Profit",
                    "value": f"${profit:.2f}",
                    "delta": f"{profit_margin:.1f}%",
                    "delta_color": "normal"
                }
            ]
            
            metrics_row(profitability_metrics)
            
            # Create a gauge chart for profit margin
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=profit_margin,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Profit Margin (%)", 'font': {'size': 24, 'family': TYPOGRAPHY['FONT_FAMILY']}},
                gauge={
                    'axis': {'range': [0, 50], 'tickwidth': 1, 'tickcolor': COLOR_SYSTEM['NEUTRAL']['DARKER']},
                    'bar': {'color': COLOR_SYSTEM['ACCENT']['GREEN']},
                    'bgcolor': COLOR_SYSTEM['NEUTRAL']['LIGHTEST'],
                    'borderwidth': 2,
                    'bordercolor': COLOR_SYSTEM['NEUTRAL']['LIGHT'],
                    'steps': [
                        {'range': [0, 15], 'color': COLOR_SYSTEM['ACCENT']['RED']},
                        {'range': [15, 25], 'color': COLOR_SYSTEM['ACCENT']['YELLOW']},
                        {'range': [25, 50], 'color': COLOR_SYSTEM['ACCENT']['GREEN']}
                    ],
                    'threshold': {
                        'line': {'color': COLOR_SYSTEM['NEUTRAL']['DARKEST'], 'width': 4},
                        'thickness': 0.75,
                        'value': profit_margin
                    }
                }
            ))
            
            # Apply styling
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
                font=dict(family=TYPOGRAPHY['FONT_FAMILY'])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Profitability assessment
            if profit_margin < 15:
                margin_category = "Low"
                margin_color = COLOR_SYSTEM['ACCENT']['RED']
                margin_message = "This profit margin is below the recommended minimum of 15%."
            elif profit_margin < 25:
                margin_category = "Moderate"
                margin_color = COLOR_SYSTEM['ACCENT']['YELLOW']
                margin_message = "This profit margin is acceptable but could be improved."
            else:
                margin_category = "Healthy"
                margin_color = COLOR_SYSTEM['ACCENT']['GREEN']
                margin_message = "This profit margin is healthy and above the target threshold."
            
            # Render profitability card
            profitability_content = f"""
            <div style="
                display: flex;
                align-items: center;
                margin-bottom: 1rem;
            ">
                <div style="
                    width: 20px;
                    height: 20px;
                    border-radius: 50%;
                    background-color: {margin_color};
                    margin-right: 10px;
                "></div>
                <h4 style="
                    margin: 0;
                    color: {margin_color};
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: 1.2rem;
                    font-weight: 600;
                ">
                    {margin_category} Profit Margin
                </h4>
            </div>
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                {margin_message}
            </p>
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-top: 0.5rem;
            ">
                <strong>Project summary:</strong>
            </p>
            <ul style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                <li><strong>Revenue:</strong> ${total_revenue:.2f} (${custom_cpi:.2f} × {completes_value} completes)</li>
                <li><strong>Cost:</strong> ${total_cost:.2f} (${cost_per_complete:.2f} × {completes_value} completes)</li>
                <li><strong>Profit:</strong> ${profit:.2f}</li>
                <li><strong>Margin:</strong> {profit_margin:.1f}%</li>
            </ul>
            """
            
            if profit_margin < 15:
                profitability_content += f"""
                <p style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                    color: {COLOR_SYSTEM['ACCENT']['RED']};
                    margin-top: 0.5rem;
                ">
                    <strong>Recommendation:</strong> Consider increasing the CPI to improve profitability
                    or seek ways to reduce costs.
                </p>
                """
            elif pricing_category == "High Risk" and profit_margin > 30:
                profitability_content += f"""
                <p style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                    color: {COLOR_SYSTEM['ACCENT']['YELLOW']};
                    margin-top: 0.5rem;
                ">
                    <strong>Strategic consideration:</strong> Your profit margin is high, but your price may
                    be uncompetitive. Consider reducing price to improve win probability while maintaining
                    acceptable profitability.
                </p>
                """
            
            render_card(
                title="Profitability Assessment", 
                content=profitability_content,
                accent_color=margin_color
            )
    
    except Exception as e:
        # Log error
        logger.error(f"Error in show_prediction: {e}", exc_info=True)
        
        # Display user-friendly error message
        st.error(f"An error occurred while generating predictions: {str(e)}")
        st.markdown(f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            border-radius: 0.5rem;
            padding: 1rem;
            margin-top: 1rem;
            border-left: 4px solid {COLOR_SYSTEM['ACCENT']['YELLOW']};
        ">
            <h4 style="margin-top: 0;">Troubleshooting</h4>
            <p>Please try the following:</p>
            <ul>
                <li>Refresh the page</li>
                <li>Try a different model type</li>
                <li>Ensure your input parameters are within reasonable ranges</li>
                <li>Check that your data has sufficient records for modeling</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)H3']['size']};
            font-weight: {TYPOGRAPHY['HEADING']['H3']['weight']};
            margin: 1rem 0;
        ">Model Training & Evaluation</h3>
        """, unsafe_allow_html=True)
        
        # Model selection
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            model_type = st.selectbox(
                "Select Model Type",
                ["Random Forest", "Linear Regression", "Gradient Boosting"],
                index=0
            )
        
        with col2:
            test_size = st.slider(
                "Test Data Percentage",
                min_value=10,
                max_value=40,
                value=int(TEST_SIZE * 100),
                step=5,
                help="Percentage of data to use for testing (remaining used for training)"
            ) / 100
        
        with col3:
            # Create a placeholder for train button
            train_button_col = st.empty()
        
        # Train model button
        if train_button_col.button("Train Model", type="primary"):
            with st.spinner(f"Training {model_type} model..."):
                # Update TEST_SIZE with the slider value
                global TEST_SIZE
                TEST_SIZE = test_size
                
                # Train the model
                model, X_train, X_test, y_train, y_test = train_model(X, y, model_type)
                
                # Store model in session state
                st.session_state['model'] = model
                st.session_state['model_type'] = model_type
                st.session_state['feature_names'] = X.columns.tolist()
                
                # Evaluate the model
                metrics = evaluate_model(model, X_test, y_test)
                st.session_state['model_metrics'] = metrics
                
                # Get feature importance
                feature_importance = get_feature_importance(model, X.columns, model_type)
                st.session_state['feature_importance'] = feature_importance
                
                # Display success message
                st.success(f"{model_type} model trained successfully!")
        
        # If model exists in session state, display evaluation metrics
        if 'model' in st.session_state and 'model_metrics' in st.session_state:
            metrics = st.session_state['model_metrics']
            
            # Display metrics
            st.markdown("<hr>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <h4 style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                font-size: {TYPOGRAPHY['HEADING']['H4']['size']};
                font-weight: {TYPOGRAPHY['HEADING']['H4']['weight']};
                margin: 1rem 0 0.5rem 0;
            ">Model Performance Metrics</h4>
            """, unsafe_allow_html=True)
            
            # Display metrics in a row
            metrics_data = [
                {
                    "label": "Mean Absolute Error",
                    "value": f"${metrics['MAE']:.2f}",
                    "delta": None
                },
                {
                    "label": "Root Mean Squared Error",
                    "value": f"${metrics['RMSE']:.2f}",
                    "delta": None
                },
                {
                    "label": "R² Score",
                    "value": f"{metrics['R²']:.3f}",
                    "delta": None
                }
            ]
            
            metrics_row(metrics_data)
            
            # Add explanation card
            metrics_explanation = f"""
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                <strong>MAE (Mean Absolute Error):</strong> Average absolute difference between predicted and actual CPI. 
                Lower is better, with ${metrics['MAE']:.2f} meaning predictions are off by about ${metrics['MAE']:.2f} on average.
            </p>
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-top: 0.5rem;
            ">
                <strong>RMSE (Root Mean Squared Error):</strong> Square root of the average squared difference between 
                predictions and actual values. More sensitive to large errors than MAE.
            </p>
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-top: 0.5rem;
            ">
                <strong>R² Score:</strong> Proportion of variance explained by the model. 
                Range from 0 to 1, where 1 is perfect prediction. 
                Current score of {metrics['R²']:.3f} means the model explains {metrics['R²'] * 100:.1f}% of CPI variability.
            </p>
            """
            
            render_card(
                title="Understanding Model Metrics", 
                content=metrics_explanation,
                accent_color=COLOR_SYSTEM['ACCENT']['BLUE']
            )
            
            # Actual vs Predicted plot
            st.markdown("<hr>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <h4 style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                font-size: {TYPOGRAPHY['HEADING']['H4']['size']};
                font-weight: {TYPOGRAPHY['HEADING']['H4']['weight']};
                margin: 1rem 0 0.5rem 0;
            ">Actual vs Predicted CPI</h4>
            """, unsafe_allow_html=True)
            
            # Create actual vs predicted scatter plot
            fig = go.Figure()
            
            # Add scatter plot
            fig.add_trace(go.Scatter(
                x=metrics['y_test'],
                y=metrics['y_pred'],
                mode='markers',
                marker=dict(
                    color=COLOR_SYSTEM['ACCENT']['PURPLE'],
                    size=8,
                    opacity=0.7,
                    line=dict(width=1, color=COLOR_SYSTEM['NEUTRAL']['WHITE'])
                ),
                name='Test Data Points',
                hovertemplate='Actual: $%{x:.2f}<br>Predicted: $%{y:.2f}<extra></extra>'
            ))
            
            # Add perfect prediction line
            min_val = min(metrics['y_test'].min(), metrics['y_pred'].min())
            max_val = max(metrics['y_test'].max(), metrics['y_pred'].max())
            
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(
                    color=COLOR_SYSTEM['NEUTRAL']['DARKER'],
                    width=2,
                    dash='dash'
                ),
                name='Perfect Prediction',
                hoverinfo='skip'
            ))
            
            # Apply consistent styling
            fig = apply_chart_styling(
                fig,
                title='Actual vs Predicted CPI Values',
                height=500
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title='Actual CPI ($)',
                yaxis_title='Predicted CPI ($)'
            )
            
            # Add insights annotations
            fig = add_insights_annotation(
                fig,
                "Points closer to the dashed line represent more accurate predictions. Points above the line are overestimated, while points below are underestimated.",
                0.01,
                0.95,
                width=220
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance chart
            st.markdown("<hr>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <h4 style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                font-size: {TYPOGRAPHY['HEADING']['H4']['size']};
                font-weight: {TYPOGRAPHY['HEADING']['H4']['weight']};
                margin: 1rem 0 0.5rem 0;
            ">Feature Importance</h4>
            """, unsafe_allow_html=True)
            
            # Create feature importance chart
            fig = create_feature_importance_chart(st.session_state['feature_importance'])
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add feature importance explanation
            importance_explanation = f"""
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                Feature importance shows which factors have the strongest influence on CPI prediction.
                Higher values indicate greater impact on the model's output.
            </p>
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-top: 0.5rem;
            ">
                <strong>Top 3 influential features:</strong>
            </p>
            <ol style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
            """
            
            # Add top 3 features
            top_features = st.session_state['feature_importance'].head(3)
            for _, row in top_features.iterrows():
                importance_explanation += f"<li><strong>{row['Feature']}</strong>: {row['Importance']:.4f}</li>"
            
            importance_explanation += f"""
            </ol>
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-top: 0.5rem;
            ">
                <strong>Pricing strategy:</strong> Focus on optimizing these key factors when setting CPI prices.
            </p>
            """
            
            render_card(
                title="Feature Importance Insights", 
                content=importance_explanation,
                accent_color=COLOR_SYSTEM['ACCENT']['GREEN']
            )
        
        # Price prediction section
        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <h3 style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            font-size: {TYPOGRAPHY['HEADING']['