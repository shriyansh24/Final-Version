"""
Data processing functionality for the CPI Analysis & Prediction Dashboard.
Includes functions for binning data, feature engineering, and advanced imputation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Union, Optional
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Import configuration for bin settings and feature engineering rules
from config import (
    IR_BINS, IR_BIN_LABELS,
    LOI_BINS, LOI_BIN_LABELS,
    COMPLETES_BINS, COMPLETES_BIN_LABELS,
    FEATURE_ENGINEERING_CONFIG,
    COLOR_SYSTEM
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_ir_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create IR (Incidence Rate) bins for analysis.

    Args:
        df (pd.DataFrame): Input dataframe with an 'IR' column.

    Returns:
        pd.DataFrame: Dataframe with an added 'IR_Bin' column.
    """
    try:
        df = df.copy()
        
        # Handle missing values before binning
        df['IR'] = pd.to_numeric(df['IR'], errors='coerce')
        
        # Only bin non-missing values
        mask = df['IR'].notna()
        df.loc[mask, 'IR_Bin'] = pd.cut(df.loc[mask, 'IR'], bins=IR_BINS, labels=IR_BIN_LABELS)
        
        return df
    except Exception as e:
        logger.error(f"Error in create_ir_bins: {e}", exc_info=True)
        return df

def create_loi_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create LOI (Length of Interview) bins for analysis.

    Args:
        df (pd.DataFrame): Input dataframe with a 'LOI' column.

    Returns:
        pd.DataFrame: Dataframe with an added 'LOI_Bin' column.
    """
    try:
        df = df.copy()
        
        # Handle missing values before binning
        df['LOI'] = pd.to_numeric(df['LOI'], errors='coerce')
        
        # Only bin non-missing values
        mask = df['LOI'].notna()
        df.loc[mask, 'LOI_Bin'] = pd.cut(df.loc[mask, 'LOI'], bins=LOI_BINS, labels=LOI_BIN_LABELS)
        
        return df
    except Exception as e:
        logger.error(f"Error in create_loi_bins: {e}", exc_info=True)
        return df

def create_completes_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Sample Size (Completes) bins for analysis.

    Args:
        df (pd.DataFrame): Input dataframe with a 'Completes' column.

    Returns:
        pd.DataFrame: Dataframe with an added 'Completes_Bin' column.
    """
    try:
        df = df.copy()
        
        # Handle missing values before binning
        df['Completes'] = pd.to_numeric(df['Completes'], errors='coerce')
        
        # Only bin non-missing values
        mask = df['Completes'].notna()
        df.loc[mask, 'Completes_Bin'] = pd.cut(
            df.loc[mask, 'Completes'], 
            bins=COMPLETES_BINS, 
            labels=COMPLETES_BIN_LABELS
        )
        
        return df
    except Exception as e:
        logger.error(f"Error in create_completes_bins: {e}", exc_info=True)
        return df

def apply_all_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply IR, LOI, and Completes binning functions to the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with added bin columns.
    """
    try:
        df = create_ir_bins(df)
        df = create_loi_bins(df)
        df = create_completes_bins(df)
        logger.info(f"Successfully applied all bins to dataframe with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error in apply_all_bins: {e}", exc_info=True)
        return df

def impute_missing_values(df: pd.DataFrame, method: str = 'knn', 
                          cols_to_impute: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Impute missing values using various advanced methods.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        method (str): Imputation method to use ('simple', 'knn', 'iterative')
        cols_to_impute (List[str], optional): List of columns to impute. If None, imputes all numeric columns.
        
    Returns:
        pd.DataFrame: Dataframe with imputed values.
    """
    try:
        df = df.copy()
        
        # If no columns specified, use all numeric columns
        if cols_to_impute is None:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            # Exclude target variable CPI if present for imputation
            if 'CPI' in numeric_cols:
                numeric_cols.remove('CPI')
            cols_to_impute = numeric_cols
        
        # Skip imputation if no numeric columns or dataframe is empty
        if not cols_to_impute or df.empty:
            logger.warning("No columns to impute or empty dataframe")
            return df
        
        # Check for columns that exist in the dataframe
        valid_cols = [col for col in cols_to_impute if col in df.columns]
        missing_cols = [col for col in cols_to_impute if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Columns not found in dataframe for imputation: {missing_cols}")
        
        if not valid_cols:
            logger.warning("No valid columns found for imputation")
            return df
        
        logger.info(f"Imputing {len(valid_cols)} columns using method: {method}")
        
        # Save non-numeric columns to add back later
        non_numeric_cols = df.columns.difference(valid_cols).tolist()
        non_numeric_data = df[non_numeric_cols].copy() if non_numeric_cols else None
        
        # Extract data for imputation
        data_to_impute = df[valid_cols].copy()
        
        # Log missing data summary
        missing_summary = data_to_impute.isna().sum()
        if missing_summary.sum() > 0:
            logger.info(f"Missing values before imputation:\n{missing_summary[missing_summary > 0]}")
        else:
            logger.info("No missing values to impute")
            return df
        
        # Apply specified imputation method
        if method == 'simple':
            # Use median for numeric columns
            imputer = SimpleImputer(strategy='median')
            imputed_data = imputer.fit_transform(data_to_impute)
            
        elif method == 'knn':
            # Use KNN imputation
            imputer = KNNImputer(n_neighbors=5, weights='distance')
            imputed_data = imputer.fit_transform(data_to_impute)
            
        elif method == 'iterative':
            # Use iterative imputation
            imputer = IterativeImputer(
                random_state=42,
                max_iter=10,
                verbose=0
            )
            imputed_data = imputer.fit_transform(data_to_impute)
        else:
            # Default to median if method not recognized
            logger.warning(f"Unrecognized imputation method: {method}. Using median imputation.")
            imputer = SimpleImputer(strategy='median')
            imputed_data = imputer.fit_transform(data_to_impute)
        
        # Convert back to DataFrame with original column names
        imputed_df = pd.DataFrame(imputed_data, columns=valid_cols, index=df.index)
        
        # Add back non-numeric columns
        if non_numeric_data is not None:
            imputed_df = pd.concat([imputed_df, non_numeric_data], axis=1)
        
        # Log missing data summary after imputation
        post_imputation_missing = imputed_df[valid_cols].isna().sum()
        if post_imputation_missing.sum() > 0:
            logger.warning(f"Still have missing values after imputation:\n{post_imputation_missing[post_imputation_missing > 0]}")
            # Fill any remaining NA values for numeric columns
            for col in valid_cols:
                if imputed_df[col].isna().any():
                    imputed_df[col] = imputed_df[col].fillna(data_to_impute[col].median())
        
        logger.info(f"Successfully imputed missing values using {method} method")
        return imputed_df
    
    except Exception as e:
        logger.error(f"Error in impute_missing_values: {e}", exc_info=True)
        logger.warning("Returning original dataframe due to imputation error")
        return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the dataframe with advanced imputation.
    
    This function ensures the key columns are numeric, uses imputation instead of dropping rows,
    creates ratio and product interaction features, log transformations, and efficiency metrics.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with additional engineered features and imputed values.
    """
    try:
        logger.info("Starting feature engineering")
        df = df.copy()
        
        # Record original row count
        original_rows = len(df)
        
        # Ensure key columns are numeric
        key_cols = ['IR', 'LOI', 'Completes', 'CPI']
        for col in key_cols:
            if col in df.columns:
                non_numeric = pd.to_numeric(df[col], errors='coerce').isna() & ~df[col].isna()
                if non_numeric.any():
                    logger.warning(f"Column {col} had {non_numeric.sum()} non-numeric values converted to NaN")
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle zeros in LOI and Completes which could cause issues in ratios
        # Replace zeros with a small positive value instead of dropping
        if 'LOI' in df.columns and (df['LOI'] == 0).any():
            zero_loi_count = (df['LOI'] == 0).sum()
            if zero_loi_count > 0:
                min_non_zero = df[df['LOI'] > 0]['LOI'].min() / 2 if (df['LOI'] > 0).any() else 0.5
                logger.info(f"Replacing {zero_loi_count} zero values in LOI with {min_non_zero}")
                df.loc[df['LOI'] == 0, 'LOI'] = min_non_zero
                
        if 'Completes' in df.columns and (df['Completes'] == 0).any():
            zero_completes_count = (df['Completes'] == 0).sum()
            if zero_completes_count > 0:
                min_non_zero = df[df['Completes'] > 0]['Completes'].min() / 2 if (df['Completes'] > 0).any() else 0.5
                logger.info(f"Replacing {zero_completes_count} zero values in Completes with {min_non_zero}")
                df.loc[df['Completes'] == 0, 'Completes'] = min_non_zero
        
        # Impute missing values in key columns using iterative imputation
        cols_to_impute = [col for col in key_cols if col in df.columns and col != 'CPI']
        if cols_to_impute:
            df = impute_missing_values(df, method='iterative', cols_to_impute=cols_to_impute)
        
        # Create basic ratio features - with safety for division
        if all(col in df.columns for col in ['IR', 'LOI']):
            df['IR_LOI_Ratio'] = df['IR'] / df['LOI'].replace(0, 0.5)
            
        if all(col in df.columns for col in ['IR', 'Completes']):
            df['IR_Completes_Ratio'] = df['IR'] / df['Completes'].replace(0, 0.5)
            
        if all(col in df.columns for col in ['LOI', 'Completes']):
            df['LOI_Completes_Ratio'] = df['LOI'] / df['Completes'].replace(0, 0.5)
        
        # Create interaction terms if enabled
        if FEATURE_ENGINEERING_CONFIG.get('create_interaction_terms', True):
            if all(col in df.columns for col in ['IR', 'LOI']):
                df['IR_LOI_Product'] = df['IR'] * df['LOI']
                
            if all(col in df.columns for col in ['CPI', 'LOI']):
                df['CPI_per_Minute'] = df['CPI'] / df['LOI'].replace(0, 0.5)
                
            if all(col in df.columns for col in ['IR', 'Completes']):
                df['IR_Completes_Product'] = df['IR'] * df['Completes']
                
            logger.info("Created interaction terms for feature engineering")
        
        # Create log transformations if enabled - safely handling zeros and negative values
        if FEATURE_ENGINEERING_CONFIG.get('create_log_transforms', True):
            # Function to safely create log transforms
            def safe_log(series):
                # Make sure values are positive before log transform
                min_val = series[series > 0].min() if (series > 0).any() else 1e-6
                adjusted = series.copy()
                adjusted[adjusted <= 0] = min_val / 2
                return np.log1p(adjusted)
            
            for col in ['Completes', 'IR', 'LOI', 'CPI']:
                if col in df.columns:
                    df[f'Log_{col}'] = safe_log(df[col])
            
            logger.info("Created log transformations for feature engineering")
        
        # Efficiency metric: (IR/100) × (1/LOI) × Completes
        if all(col in df.columns for col in ['IR', 'LOI', 'Completes']):
            df['CPI_Efficiency'] = (df['IR'] / 100) * (1 / df['LOI'].replace(0, 0.5)) * df['Completes']
        
        # Normalized versions of key variables
        for col in ['IR', 'LOI', 'Completes']:
            if col in df.columns:
                mean, std = df[col].mean(), df[col].std()
                if not pd.isna(std) and std > 0:
                    df[f'{col}_Normalized'] = (df[col] - mean) / std
        
        # Replace infinities with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Impute any newly created features with NaN values
        new_features = [col for col in df.columns if col not in key_cols and col != 'Type' 
                        and df[col].dtype.kind in 'fc']  # float or complex
        
        # For the derived features, use simple median imputation (faster and sufficient)
        if new_features:
            missing_in_new = df[new_features].isna().sum()
            if missing_in_new.sum() > 0:
                logger.info(f"Imputing {missing_in_new.sum()} missing values in derived features")
                df = impute_missing_values(df, method='simple', cols_to_impute=new_features)
        
        # Count final number of features created
        new_features_count = len(df.columns) - len(key_cols)
        logger.info(f"Feature engineering complete. Created {new_features_count} new features.")
        
        # Check for and report any remaining missing values
        remaining_missing = df.isna().sum()
        if remaining_missing.sum() > 0:
            logger.warning(f"Warning: Still have {remaining_missing.sum()} missing values after feature engineering")
            logger.debug(f"Columns with missing values:\n{remaining_missing[remaining_missing > 0]}")
        
        # Instead of dropping rows, use final pass of simple imputation for any remaining NaN
        if remaining_missing.sum() > 0:
            cols_with_missing = remaining_missing[remaining_missing > 0].index.tolist()
            df = impute_missing_values(df, method='simple', cols_to_impute=cols_with_missing)
        
        # Final check to ensure no missing values remain
        final_missing = df.isna().sum().sum()
        if final_missing > 0:
            logger.warning(f"Warning: {final_missing} missing values still remain. Final fallback to fillna")
            # Last resort - fill with median for each column
            for col in df.columns:
                if df[col].isna().any():
                    if df[col].dtype.kind in 'fc':  # float or complex
                        df[col] = df[col].fillna(df[col].median())
                    elif df[col].dtype.kind in 'iub':  # integer, unsigned, boolean
                        df[col] = df[col].fillna(df[col].median())
                    else:  # string or object - fill with most common value
                        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")
        
        # Compare row counts to make sure we didn't lose any data
        if len(df) != original_rows:
            logger.error(f"Error: Row count changed from {original_rows} to {len(df)}")
        else:
            logger.info(f"Successfully preserved all {original_rows} rows during feature engineering")
            
        return df
    
    except Exception as e:
        logger.error(f"Error in engineer_features: {e}", exc_info=True)
        return df

def prepare_model_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for modeling with advanced imputation instead of dropping rows.
    
    This function validates required columns, converts values to numeric,
    applies imputation for missing values, implements feature engineering,
    performs one-hot encoding for categorical variables, and ensures data quality.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features matrix (X) and target series (y).
    """
    try:
        logger.info("Starting data preparation for modeling")
        df = df.copy()
        
        required_cols = ['IR', 'LOI', 'Completes', 'CPI', 'Type']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {', '.join(missing_cols)}")
            return pd.DataFrame(), pd.Series()
        
        # Store the original target values before any processing
        original_target = df['CPI'].copy()
        
        # Ensure numeric conversion for key variables
        numeric_cols = ['IR', 'LOI', 'Completes', 'CPI']
        for col in numeric_cols:
            # Save info about non-numeric values
            non_numeric = pd.to_numeric(df[col], errors='coerce').isna() & ~df[col].isna()
            if non_numeric.any():
                logger.warning(f"Column {col}: {non_numeric.sum()} non-numeric values converted to NaN")
            
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Log information about missing values
        missing_before = df[numeric_cols].isna().sum()
        if missing_before.sum() > 0:
            logger.info(f"Missing values before imputation:\n{missing_before}")
        
        # Apply feature engineering with imputation
        df = engineer_features(df)
        
        # Make sure bin columns are created
        if 'IR_Bin' not in df.columns:
            df = apply_all_bins(df)
            logger.info("Applied binning to variables")
        
        # Decide which features to use - exclude the target variable
        feature_cols = [
            'IR', 'LOI', 'Completes', 'IR_LOI_Ratio',
            'IR_Completes_Ratio', 'Log_Completes', 'Type'
        ]
        
        additional_cols = [
            'IR_LOI_Product', 'CPI_per_Minute', 'Log_IR',
            'Log_LOI', 'IR_Normalized', 'LOI_Normalized',
            'Completes_Normalized', 'IR_Completes_Product',
            'IR_LOI_Norm_Interaction', 'IR_Squared', 'LOI_Squared',
            'CPI_Efficiency'
        ]
        
        # Add any additional columns that exist
        for col in additional_cols:
            if col in df.columns:
                feature_cols.append(col)
        
        # Get list of available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        
        # Handle categorical variable "Type"
        if 'Type' in available_cols:
            # Only include valid types
            valid_types = ['Won', 'Lost']
            invalid_types = df[~df['Type'].isin(valid_types)]['Type'].unique()
            
            if len(invalid_types) > 0:
                logger.warning(f"Found invalid Types: {invalid_types}. Will be converted to NaN")
                
            # Fill any missing or invalid Types with most common value
            most_common_type = df['Type'].value_counts().index[0]
            df['Type'] = df['Type'].apply(lambda x: x if x in valid_types else None)
            df['Type'] = df['Type'].fillna(most_common_type)
            
            # One-hot encode Type
            df = pd.get_dummies(df, columns=['Type'], drop_first=True)
            
            # Update available columns
            if 'Type_Won' in df.columns:
                available_cols = [col for col in available_cols if col != 'Type'] + ['Type_Won']
        
        # Add segment information if available
        if 'Segment' in df.columns and df['Segment'].notna().any():
            logger.info("Client segment data found, adding to model features")
            
            # Fill missing segments with "Unknown"
            df['Segment'] = df['Segment'].fillna('Unknown')
            
            # One-hot encode Segment
            df = pd.get_dummies(df, columns=['Segment'], drop_first=True)
            
            # Add segment columns to available features
            segment_cols = [col for col in df.columns if col.startswith('Segment_')]
            available_cols.extend(segment_cols)
        
        # Create X (features) and y (target) datasets
        X = df[available_cols].copy()
        y = original_target  # Use the original target values
        
        # Perform final imputation on X to ensure no missing values
        missing_in_X = X.isna().sum().sum()
        if missing_in_X > 0:
            logger.warning(f"Still have {missing_in_X} missing values in features. Performing final imputation.")
            # Use simple median imputation for any remaining missing values
            for col in X.columns:
                if X[col].isna().any():
                    X[col] = X[col].fillna(X[col].median() if X[col].dtype.kind in 'fc' else X[col].mode().iloc[0])
        
        # Check for missing values in target variable
        missing_in_y = y.isna().sum()
        if missing_in_y > 0:
            logger.warning(f"{missing_in_y} missing values in target variable (CPI)")
            
            # Impute missing target values using the median
            y = y.fillna(y.median())
        
        if len(X) < 10:
            logger.error(f"Too few samples for modeling: {len(X)}")
            return pd.DataFrame(), pd.Series()
        
        logger.info(f"Data preparation successful. X shape: {X.shape}, y shape: {y.shape}")
        logger.info(f"Features used: {', '.join(X.columns)}")
        
        return X, y
    
    except Exception as e:
        logger.error(f"Error in prepare_model_data: {e}", exc_info=True)
        return pd.DataFrame(), pd.Series()

def get_data_summary(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Generate summary statistics for key metrics by Type (Won/Lost).

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        Dict[str, Dict[str, float]]: Summary metrics grouped by Type.
    """
    summary = {}
    
    try:
        # Handle case where 'Type' column might be missing
        if 'Type' not in df.columns:
            logger.warning("Type column not found in dataframe for summary")
            return {'Combined': {'Count': len(df), 'Avg_CPI': df['CPI'].mean()}}
        
        # Ensure numeric conversion for calculation
        for col in ['CPI', 'IR', 'LOI', 'Completes']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Group by Type and calculate statistics
        grouped = df.groupby('Type')
        
        for name, group in grouped:
            summary[name] = {
                'Count': len(group),
                'Avg_CPI': group['CPI'].mean(),
                'Median_CPI': group['CPI'].median(),
                'Avg_IR': group['IR'].mean(),
                'Avg_LOI': group['LOI'].mean(),
                'Avg_Completes': group['Completes'].mean(),
                'CPI_25th': group['CPI'].quantile(0.25),
                'CPI_75th': group['CPI'].quantile(0.75),
                'CPI_95th': group['CPI'].quantile(0.95),
                'IR_LOI_Ratio_Avg': (group['IR'] / group['LOI'].replace(0, 0.5)).mean(),
                'Cost_per_Min': (group['CPI'] / group['LOI'].replace(0, 0.5)).mean(),
                'Cost_per_Complete': (group['CPI'] * group['Completes']).mean(),
                'CPI_StdDev': group['CPI'].std(),
                'IR_StdDev': group['IR'].std(),
                'LOI_StdDev': group['LOI'].std()
            }
            
            # Calculate efficiency metrics
            group_eff = (group['IR'] / 100) * (1 / group['LOI'].replace(0, 0.5)) * group['Completes']
            summary[name]['Efficiency_Metric_Avg'] = group_eff.mean()
            summary[name]['Efficiency_Metric_Median'] = group_eff.median()
        
        # Add a "Combined" group with stats for all data
        summary['Combined'] = {
            'Count': len(df),
            'Avg_CPI': df['CPI'].mean(),
            'Median_CPI': df['CPI'].median(),
            'Avg_IR': df['IR'].mean(),
            'Avg_LOI': df['LOI'].mean(),
            'Avg_Completes': df['Completes'].mean()
        }
        
    except Exception as e:
        logger.error(f"Error in get_data_summary: {e}", exc_info=True)
        
    return summary
