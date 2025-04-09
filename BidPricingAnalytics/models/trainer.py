"""
ML model training and evaluation for the CPI Analysis & Prediction Dashboard.

Includes functions for building, training, hyperparameter tuning, evaluation, 
and saving/loading of prediction models with enhanced output styling.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, List, Tuple, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import color system for styled outputs if available
try:
    from ui_components import COLOR_SYSTEM
except ImportError:
    logger.warning("Could not import COLOR_SYSTEM from ui_components. Using fallback colors.")
    # Fallback color system for console outputs
    COLOR_SYSTEM = {
        "ACCENT": {
            "BLUE": "#00BFFF",
            "ORANGE": "#FFB74D",
            "GREEN": "#66FF99",
            "RED": "#FF3333",
            "PURPLE": "#BB86FC"
        }
    }

# Import configuration settings
try:
    import config
    MODEL_DIR = config.MODEL_DIR
    RANDOM_STATE = config.RANDOM_STATE
    TEST_SIZE = config.TEST_SIZE
except ImportError:
    logger.warning("Unable to import config. Using default settings.")
    import tempfile
    MODEL_DIR = os.path.join(tempfile.gettempdir(), "bidpricing_models")
    RANDOM_STATE = 42
    TEST_SIZE = 0.2

# Ensure model directory exists for saving models
os.makedirs(MODEL_DIR, exist_ok=True)

# Define model configurations including hyperparameter grids
MODEL_CONFIGS = {
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {
            'fit_intercept': [True, False],
            'positive': [True, False]
        }
    },
    'Ridge Regression': {
        'model': Ridge(random_state=RANDOM_STATE),
        'params': {
            'alpha': [0.01, 0.1, 1.0, 10.0],
            'fit_intercept': [True, False]
        }
    },
    'Lasso Regression': {
        'model': Lasso(random_state=RANDOM_STATE),
        'params': {
            'alpha': [0.01, 0.1, 1.0, 10.0],
            'fit_intercept': [True, False]
        }
    },
    'Random Forest': {
        'model': RandomForestRegressor(random_state=RANDOM_STATE),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(random_state=RANDOM_STATE),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    }
}

def build_models(X: pd.DataFrame, y: pd.Series, 
               do_hyperparameter_tuning: bool = False) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]], pd.DataFrame]:
    """
    Build and evaluate prediction models for CPI.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        do_hyperparameter_tuning (bool): Whether to perform hyperparameter tuning (GridSearchCV).

    Returns:
        Tuple containing:
        - Dictionary of trained models.
        - Dictionary of evaluation scores for each model.
        - DataFrame containing feature importance (from Random Forest) sorted in descending order.
    """
    try:
        logger.info("Starting model building process")
        
        # Use configured test size and random state
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        
        if do_hyperparameter_tuning:
            logger.info("Performing hyperparameter tuning")
            models = build_models_with_tuning(X_train, y_train, X_test, y_test)
        else:
            logger.info("Building models with default parameters")
            models = build_models_default(X_train, y_train, X_test, y_test)
        
        trained_models = models.get('trained_models', {})
        model_scores = models.get('model_scores', {})
        
        # Calculate feature importance using the Random Forest model if available
        if 'Random Forest' in trained_models:
            rf_model = trained_models['Random Forest']
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            logger.info("Feature importance calculated successfully")
        else:
            feature_importance = pd.DataFrame(columns=['Feature', 'Importance'])
            logger.warning("Random Forest model not found. Skipping feature importance calculation.")
        
        return trained_models, model_scores, feature_importance
    
    except Exception as e:
        logger.error(f"Error in build_models: {e}", exc_info=True)
        return {}, {}, pd.DataFrame(columns=['Feature', 'Importance'])

def build_models_default(X_train: pd.DataFrame, y_train: pd.Series, 
                       X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Build models with default parameters (without hyperparameter tuning).

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target variable.
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series): Test target variable.

    Returns:
        Dict[str, Any]: Dictionary containing:
        - 'trained_models': Dictionary of trained models.
        - 'model_scores': Dictionary of evaluation scores per model.
    """
    trained_models = {}
    model_scores = {}
    
    # Define basic models to train using default parameters
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=RANDOM_STATE),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)
    }
    
    for name, model in models.items():
        logger.info(f"Training {name} model")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict on test set and calculate evaluation metrics
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        model_scores[name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}
        logger.info(f"{name} model trained. R² score: {r2:.4f}")
    
    return {'trained_models': trained_models, 'model_scores': model_scores}

def build_models_with_tuning(X_train: pd.DataFrame, y_train: pd.Series, 
                           X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Build models with hyperparameter tuning using GridSearchCV.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target variable.
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series): Test target variable.

    Returns:
        Dict[str, Any]: Dictionary containing:
        - 'trained_models': Dictionary of tuned models.
        - 'model_scores': Dictionary of evaluation scores.
        - 'best_params': Dictionary of best parameters for each model.
    """
    trained_models = {}
    model_scores = {}
    best_params = {}
    
    for name, config_params in MODEL_CONFIGS.items():
        logger.info(f"Tuning {name} model")
        
        grid_search = GridSearchCV(
            estimator=config_params['model'],
            param_grid=config_params['params'],
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        trained_models[name] = grid_search.best_estimator_
        best_params[name] = grid_search.best_params_
        
        y_pred = grid_search.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        model_scores[name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}
        logger.info(f"{name} tuned. Best params: {grid_search.best_params_}. R²: {r2:.4f}")
    
    return {'trained_models': trained_models, 'model_scores': model_scores, 'best_params': best_params}

def save_models(models: Dict[str, Any], filename_prefix: str = 'cpi_model') -> Dict[str, str]:
    """
    Save trained models to disk.

    Args:
        models (Dict[str, Any]): Dictionary of trained models.
        filename_prefix (str): Filename prefix for saved models.

    Returns:
        Dict[str, str]: Mapping from model names to saved file paths.
    """
    try:
        saved_paths = {}
        
        for name, model in models.items():
            safe_name = name.lower().replace(' ', '_')
            filename = f"{filename_prefix}_{safe_name}.joblib"
            filepath = os.path.join(MODEL_DIR, filename)
            
            joblib.dump(model, filepath)
            saved_paths[name] = filepath
            logger.info(f"Saved {name} model to {filepath}")
        
        return saved_paths
    
    except Exception as e:
        logger.error(f"Error saving models: {e}", exc_info=True)
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
        logger.error(f"Error loading models: {e}", exc_info=True)
        return {}

def cross_validate_models(X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, Dict[str, float]]:
    """
    Perform cross-validation on a set of basic models.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        cv (int): Number of cross-validation folds.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary of cross-validation metrics for each model.
    """
    try:
        logger.info(f"Starting cross-validation with {cv} folds")
        
        cv_scores = {}
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=RANDOM_STATE),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)
        }
        
        for name, model in models.items():
            logger.info(f"Cross-validating {name} model")
            
            mse_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
            r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            mae_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
            
            cv_scores[name] = {
                'MSE_mean': mse_scores.mean(),
                'MSE_std': mse_scores.std(),
                'RMSE_mean': np.sqrt(mse_scores.mean()),
                'RMSE_std': np.sqrt(mse_scores.std()),
                'MAE_mean': mae_scores.mean(),
                'MAE_std': mae_scores.std(),
                'R²_mean': r2_scores.mean(),
                'R²_std': r2_scores.std()
            }
            
            logger.info(f"{name} CV complete. Mean R²: {r2_scores.mean():.4f}")
        
        return cv_scores
    
    except Exception as e:
        logger.error(f"Error in cross_validate_models: {e}", exc_info=True)
        return {}

def get_model_results_html(model_scores: Dict[str, Dict[str, float]]) -> str:
    """
    Convert model evaluation scores to HTML format with high-contrast styling.

    Args:
        model_scores (Dict[str, Dict[str, float]]): Dictionary of model evaluation scores.

    Returns:
        str: HTML-formatted table of model results.
    """
    try:
        if not model_scores:
            return "<p>No model results available.</p>"
        
        # Determine the best model based on R² score
        best_model = max(model_scores.items(), key=lambda x: x[1].get('R²', 0))
        best_model_name = best_model[0]
        
        html = f"""
        <div style="
            background-color: #1F1F1F;
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
            color: #FFFFFF;
        ">
            <h3 style="margin-top: 0; color: #FFFFFF;">Model Evaluation Results</h3>
            <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; margin-top: 1rem;">
                    <thead>
                        <tr style="background-color: #2E2E2E; border-bottom: 2px solid #3A3A3A;">
                            <th style="padding: 0.75rem; text-align: left; color: #FFFFFF;">Model</th>
                            <th style="padding: 0.75rem; text-align: center; color: #FFFFFF;">R²</th>
                            <th style="padding: 0.75rem; text-align: center; color: #FFFFFF;">RMSE</th>
                            <th style="padding: 0.75rem; text-align: center; color: #FFFFFF;">MAE</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for name, scores in model_scores.items():
            is_best = name == best_model_name
            row_bg = "#2E2E2E" if is_best else "#1F1F1F"
            row_style = f"background-color: {row_bg}; border-bottom: 1px solid #3A3A3A;"
            highlight = f"color: {COLOR_SYSTEM['ACCENT']['GREEN']}; font-weight: 600;" if is_best else "color: #FFFFFF;"
            
            html += f"""
                        <tr style="{row_style}">
                            <td style="padding: 0.75rem; {highlight}">{name} {' (Best)' if is_best else ''}</td>
                            <td style="padding: 0.75rem; text-align: center; {highlight}">{scores.get('R²', 0):.4f}</td>
                            <td style="padding: 0.75rem; text-align: center; color: #FFFFFF;">{scores.get('RMSE', 0):.4f}</td>
                            <td style="padding: 0.75rem; text-align: center; color: #FFFFFF;">{scores.get('MAE', 0):.4f}</td>
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

def get_feature_importance_html(feature_importance: pd.DataFrame, top_n: int = 10) -> str:
    """
    Convert feature importance DataFrame to HTML format with high-contrast styling.

    Args:
        feature_importance (pd.DataFrame): DataFrame with Feature and Importance columns.
        top_n (int): Number of top features to display.

    Returns:
        str: HTML-formatted visualization of feature importances.
    """
    try:
        if feature_importance.empty:
            return "<p>No feature importance data available.</p>"
        
        # Limit to top N features
        top_features = feature_importance.head(top_n)
        
        # Calculate the maximum importance for scaling the bars
        max_importance = top_features['Importance'].max()
        
        html = f"""
        <div style="
            background-color: #1F1F1F;
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid {COLOR_SYSTEM['ACCENT']['PURPLE']};
            color: #FFFFFF;
        ">
            <h3 style="margin-top: 0; color: #FFFFFF;">Feature Importance</h3>
            <div style="margin-top: 1rem;">
        """
        
        # Create bars for each feature
        for _, row in top_features.iterrows():
            feature = row['Feature']
            importance = row['Importance']
            percent_width = (importance / max_importance) * 100
            
            html += f"""
                <div style="margin-bottom: 0.75rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <span style="color: #FFFFFF;">{feature}</span>
                        <span style="color: #B0B0B0;">{importance:.4f}</span>
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
