"""
Model training functionality for the CPI Analysis & Prediction Dashboard.
Provides complete model building, training, evaluation, and feature importance analysis
with styling compatible with the high-contrast dark theme UI.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import color system if available for styled outputs
try:
    from config import COLOR_SYSTEM, RANDOM_STATE, TEST_SIZE
except ImportError:
    logger.warning("Could not import COLOR_SYSTEM from config. Using fallback colors.")
    # Fallback color system for console outputs
    COLOR_SYSTEM = {
        "ACCENT": {
            "BLUE": "#00BFFF",
            "ORANGE": "#FFB74D",
            "GREEN": "#66FF99",
            "RED": "#FF3333",
            "PURPLE": "#BB86FC"
        },
        "PRIMARY": {
            "MAIN": "#FFFFFF",
            "LIGHT": "#B0B0B0"
        },
        "BACKGROUND": {
            "CARD": "#1F1F1F",
            "MAIN": "#000000"
        },
        "NEUTRAL": {
            "LIGHT": "#3A3A3A"
        }
    }
    RANDOM_STATE = 42
    TEST_SIZE = 0.2

def train_models(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Train multiple models on the provided data and return them with evaluation metrics.
    
    Args:
        data (pd.DataFrame): Data with features and target.
        
    Returns:
        Dict[str, Any]: Dictionary of trained models.
    """
    logger.info("Starting model training process")
    
    try:
        # Prepare data
        if 'CPI' not in data.columns:
            logger.error("CPI target column not found in data")
            return {}
            
        # Define features and target
        feature_cols = [col for col in data.columns if col != 'CPI' and data[col].dtype.kind in 'fcib']
        X = data[feature_cols]
        y = data['CPI']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)
        }
        
        trained_models = {}
        model_scores = {}
        
        # Train and evaluate each model
        for name, model in models.items():
            logger.info(f"Training {name} model")
            model.fit(X_train, y_train)
            trained_models[name] = model
            
            # Evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            model_scores[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2
            }
            logger.info(f"{name} trained. R² score: {r2:.4f}")
        
        return trained_models
        
    except Exception as e:
        logger.error(f"Error in train_models: {e}", exc_info=True)
        return {}

def evaluate_models(models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    Evaluate trained models on test data.
    
    Args:
        models (Dict[str, Any]): Dictionary of trained models.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        
    Returns:
        Dict[str, Dict[str, float]]: Evaluation metrics for each model.
    """
    try:
        results = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2
            }
        return results
    except Exception as e:
        logger.error(f"Error in evaluate_models: {e}", exc_info=True)
        return {}

def cross_validate_models(X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, Dict[str, float]]:
    """
    Perform cross-validation on a set of models.
    
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        cv (int): Number of cross-validation folds.
        
    Returns:
        Dict[str, Dict[str, float]]: Cross-validation results.
    """
    try:
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)
        }
        
        results = {}
        for name, model in models.items():
            r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            mse_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
            mae_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
            
            results[name] = {
                'R²_mean': r2_scores.mean(),
                'R²_std': r2_scores.std(),
                'MSE_mean': mse_scores.mean(),
                'MSE_std': mse_scores.std(),
                'RMSE_mean': np.sqrt(mse_scores.mean()),
                'RMSE_std': np.sqrt(mse_scores.std()),
                'MAE_mean': mae_scores.mean(),
                'MAE_std': mae_scores.std()
            }
        
        return results
    except Exception as e:
        logger.error(f"Error in cross_validate_models: {e}", exc_info=True)
        return {}

def get_feature_importance(models: Dict[str, Any], feature_names: List[str]) -> Dict[str, Any]:
    """
    Extract feature importance from the trained models.
    
    Args:
        models (Dict[str, Any]): Dictionary of trained models.
        feature_names (List[str]): List of feature names.
        
    Returns:
        Dict[str, Any]: Feature importance data.
    """
    try:
        # Use Random Forest for feature importance if available
        if 'Random Forest' in models:
            model = models['Random Forest']
            importances = model.feature_importances_
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            sorted_features = [feature_names[i] for i in indices]
            sorted_importance = importances[indices]
            
            return {
                'features': sorted_features,
                'importance': sorted_importance,
                'model': 'Random Forest'
            }
        elif 'Gradient Boosting' in models:
            # Fallback to Gradient Boosting
            model = models['Gradient Boosting']
            importances = model.feature_importances_
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            sorted_features = [feature_names[i] for i in indices]
            sorted_importance = importances[indices]
            
            return {
                'features': sorted_features,
                'importance': sorted_importance,
                'model': 'Gradient Boosting'
            }
        else:
            logger.warning("No suitable model found for feature importance")
            return {
                'features': [],
                'importance': [],
                'model': 'None'
            }
    except Exception as e:
        logger.error(f"Error in get_feature_importance: {e}", exc_info=True)
        return {
            'features': [],
            'importance': [],
            'model': 'Error'
        }

def save_models(models: Dict[str, Any], directory: str = "models/saved_models") -> Dict[str, str]:
    """
    Save trained models to disk.
    
    Args:
        models (Dict[str, Any]): Dictionary of trained models.
        directory (str): Directory to save models.
        
    Returns:
        Dict[str, str]: Mapping from model names to saved paths.
    """
    try:
        import joblib
        
        # Ensure directory exists
        os.makedirs(directory, exist_ok=True)
        
        saved_paths = {}
        for name, model in models.items():
            # Create a safe filename
            safe_name = name.lower().replace(' ', '_')
            filename = f"cpi_model_{safe_name}.joblib"
            filepath = os.path.join(directory, filename)
            
            # Save the model
            joblib.dump(model, filepath)
            saved_paths[name] = filepath
            logger.info(f"Saved {name} model to {filepath}")
            
        return saved_paths
    except Exception as e:
        logger.error(f"Error in save_models: {e}", exc_info=True)
        return {}

def load_models(model_paths: Dict[str, str]) -> Dict[str, Any]:
    """
    Load trained models from disk.
    
    Args:
        model_paths (Dict[str, str]): Mapping from model names to file paths.
        
    Returns:
        Dict[str, Any]: Dictionary of loaded models.
    """
    try:
        import joblib
        
        loaded_models = {}
        for name, path in model_paths.items():
            if os.path.exists(path):
                model = joblib.load(path)
                loaded_models[name] = model
                logger.info(f"Loaded {name} model from {path}")
            else:
                logger.warning(f"Model file not found: {path}")
                
        return loaded_models
    except Exception as e:
        logger.error(f"Error in load_models: {e}", exc_info=True)
        return {}

def get_model_results_html(model_scores: Dict[str, Dict[str, float]]) -> str:
    """
    Generate HTML for model comparison results with high-contrast styling.
    
    Args:
        model_scores (Dict[str, Dict[str, float]]): Model evaluation metrics.
        
    Returns:
        str: HTML-formatted model comparison.
    """
    try:
        if not model_scores:
            return "<p>No model results available.</p>"
        
        # Find best model based on R² score
        best_model = max(model_scores.items(), key=lambda x: x[1].get('R²', 0))
        best_model_name = best_model[0]
        
        html = f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
        ">
            <h3 style="margin-top: 0; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Model Evaluation Results</h3>
            <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; margin-top: 1rem;">
                    <thead>
                        <tr style="background-color: #2E2E2E; border-bottom: 2px solid #3A3A3A;">
                            <th style="padding: 0.75rem; text-align: left; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Model</th>
                            <th style="padding: 0.75rem; text-align: center; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">R²</th>
                            <th style="padding: 0.75rem; text-align: center; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">RMSE</th>
                            <th style="padding: 0.75rem; text-align: center; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">MAE</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for name, scores in model_scores.items():
            is_best = name == best_model_name
            row_bg = "#2E2E2E" if is_best else "#1F1F1F"
            row_style = f"background-color: {row_bg}; border-bottom: 1px solid #3A3A3A;"
            highlight = f"color: {COLOR_SYSTEM['ACCENT']['GREEN']}; font-weight: 600;" if is_best else f"color: {COLOR_SYSTEM['PRIMARY']['MAIN']};"
            
            html += f"""
                        <tr style="{row_style}">
                            <td style="padding: 0.75rem; {highlight}">{name} {' (Best)' if is_best else ''}</td>
                            <td style="padding: 0.75rem; text-align: center; {highlight}">{scores.get('R²', 0):.4f}</td>
                            <td style="padding: 0.75rem; text-align: center; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">{scores.get('RMSE', 0):.4f}</td>
                            <td style="padding: 0.75rem; text-align: center; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">{scores.get('MAE', 0):.4f}</td>
                        </tr>
            """
        
        html += """
                    </tbody>
                </table>
            </div>
            <div style="margin-top: 1rem; font-size: 0.9rem; color: #B0B0B0;">
                <p>
                    <strong>R²</strong>: Coefficient of determination (higher is better)<br>
                    <strong>RMSE</strong>: Root Mean Squared Error (lower is better)<br>
                    <strong>MAE</strong>: Mean Absolute Error (lower is better)
                </p>
            </div>
        </div>
        """
        
        return html
    except Exception as e:
        logger.error(f"Error generating model results HTML: {e}", exc_info=True)
        return f"<p style='color: {COLOR_SYSTEM['ACCENT']['RED']};'>Error generating model results.</p>"

def get_feature_importance_html(feature_importance: Dict[str, Any], top_n: int = 10) -> str:
    """
    Generate HTML visualization of feature importance with high-contrast styling.
    
    Args:
        feature_importance (Dict[str, Any]): Feature importance data.
        top_n (int): Number of top features to display.
        
    Returns:
        str: HTML-formatted visualization.
    """
    try:
        if not feature_importance or 'features' not in feature_importance or not feature_importance['features']:
            return "<p>No feature importance data available.</p>"
        
        features = feature_importance['features'][:top_n]
        importances = feature_importance['importance'][:top_n]
        
        # Calculate the maximum importance for scaling the bars
        max_importance = max(importances) if len(importances) > 0 else 1
        
        html = f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid {COLOR_SYSTEM['ACCENT']['PURPLE']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
        ">
            <h3 style="margin-top: 0; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Feature Importance</h3>
            <div style="margin-top: 1rem;">
        """
        
        # Create bars for each feature
        for i, (feature, importance) in enumerate(zip(features, importances)):
            percent_width = (importance / max_importance) * 100
            
            html += f"""
                <div style="margin-bottom: 0.75rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <span style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">{feature}</span>
                        <span style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">{importance:.4f}</span>
                    </div>
                    <div style="
                        background-color: #2E2E2E;
                        height: 0.75rem;
                        border-radius: 0.25rem;
                        overflow: hidden;
                    ">
                        <div style="
                            width: {percent_width}%;
                            height: 100%;
                            background-color: {COLOR_SYSTEM['ACCENT']['PURPLE']};
                            border-radius: 0.25rem;
                        "></div>
                    </div>
                </div>
            """
        
        html += """
            </div>
            <div style="margin-top: 1rem; font-size: 0.9rem; color: #B0B0B0;">
                <p>Feature importance derived from Random Forest model coefficients.</p>
            </div>
        </div>
        """
        
        return html
    except Exception as e:
        logger.error(f"Error generating feature importance HTML: {e}", exc_info=True)
        return f"<p style='color: {COLOR_SYSTEM['ACCENT']['RED']};'>Error generating feature importance visualization.</p>"