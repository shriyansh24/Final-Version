"""
Data processing functionality for the CPI Analysis & Prediction Dashboard.
Includes functions for binning data and feature engineering.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Union

# Import configuration for bin settings
from config import (IR_BINS, IR_BIN_LABELS, 
                    LOI_BINS, LOI_BIN_LABELS,
                    COMPLETES_BINS, COMPLETES_BIN_LABELS,
                    FEATURE_ENGINEERING_CONFIG)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_ir_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create IR (Incidence Rate) bins for analysis.
    
    Args:
        df (pd.DataFrame): Input dataframe with IR column
    
    Returns:
        pd.DataFrame: Dataframe with added IR_Bin column
    """
    try:
        df = df.copy()  # Create a copy to avoid modifying the original
        
        # Make sure IR is numeric
        df['IR'] = pd.to_numeric(df['IR'], errors='coerce')
        
        # Create bins
        df['IR_Bin'] = pd.cut(
            df['IR'],
            bins=IR_BINS,
            labels=IR_BIN_LABELS
        )
        
        return df
    
    except Exception as e:
        logger.error(f"Error in create_ir_bins: {e}", exc_info=True)
        # Return original dataframe if binning fails
        return df

def create_loi_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create LOI (Length of Interview) bins for analysis.
    
    Args:
        df (pd.DataFrame): Input dataframe with LOI column
    
    Returns:
        pd.DataFrame: Dataframe with added LOI_Bin column
    """
    try:
        df = df.copy()  # Create a copy to avoid modifying the original
        
        # Make sure LOI is numeric
        df['LOI'] = pd.to_numeric(df['LOI'], errors='coerce')
        
        # Create bins
        df['LOI_Bin'] = pd.cut(
            df['LOI'],
            bins=LOI_BINS,
            labels=LOI_BIN_LABELS
        )
        
        return df
    
    except Exception as e:
        logger.error(f"Error in create_loi_bins: {e}", exc_info=True)
        # Return original dataframe if binning fails
        return df

def create_completes_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Sample Size (Completes) bins for analysis.
    
    Args:
        df (pd.DataFrame): Input dataframe with Completes column
    
    Returns:
        pd.DataFrame: Dataframe with added Completes_Bin column
    """
    try:
        df = df.copy()  # Create a copy to avoid modifying the original
        
        # Make sure Completes is numeric
        df['Completes'] = pd.to_numeric(df['Completes'], errors='coerce')
        
        # Create bins
        df['Completes_Bin'] = pd.cut(
            df['Completes'],
            bins=COMPLETES_BINS,
            labels=COMPLETES_BIN_LABELS
        )
        
        return df
    
    except Exception as e:
        logger.error(f"Error in create_completes_bins: {e}", exc_info=True)
        # Return original dataframe if binning fails
        return df

def apply_all_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all binning functions to a dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Dataframe with all bin columns added
    """
    try:
        df = create_ir_bins(df)
        df = create_loi_bins(df)
        df = create_completes_bins(df)
        
        # Log success for monitoring
        logger.info(f"Successfully applied all bins to dataframe with {len(df)} rows")
        
        return df
        
    except Exception as e:
        logger.error(f"Error in apply_all_bins: {e}", exc_info=True)
        # Return original dataframe if binning fails
        return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Dataframe with additional engineered features
    """
    try:
        df = df.copy()  # Create a copy to avoid modifying the original
        
        # Make sure key columns are numeric
        for col in ['IR', 'LOI', 'Completes', 'CPI']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with missing values in key columns
        original_rows = len(df)
        df = df.dropna(subset=['IR', 'LOI', 'Completes', 'CPI'])
        dropped_rows = original_rows - len(df)
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows with missing values in key columns")
        
        # Basic features
        df['IR_LOI_Ratio'] = df['IR'] / df['LOI']
        df['IR_Completes_Ratio'] = df['IR'] / df['Completes']
        df['LOI_Completes_Ratio'] = df['LOI'] / df['Completes']
        
        # Check if we should create interaction terms
        if FEATURE_ENGINEERING_CONFIG.get('create_interaction_terms', True):
            df['IR_LOI_Product'] = df['IR'] * df['LOI']  # Interaction term
            df['CPI_per_Minute'] = df['CPI'] / df['LOI']  # Cost per minute
            
            # Add interaction between IR and Completes
            df['IR_Completes_Product'] = df['IR'] * df['Completes']
            
            logger.info("Created interaction terms for feature engineering")
        
        # Check if we should create log transforms
        if FEATURE_ENGINEERING_CONFIG.get('create_log_transforms', True):
            df['Log_Completes'] = np.log1p(df['Completes'])  # Log transformation for skewed distribution
            
            # Add log transforms for other important variables
            df['Log_IR'] = np.log1p(df['IR'])
            df['Log_LOI'] = np.log1p(df['LOI'])
            df['Log_CPI'] = np.log1p(df['CPI'])
            
            logger.info("Created log transformations for feature engineering")
        
        # Efficiency metric (always created)
        df['CPI_Efficiency'] = (df['IR'] / 100) * (1 / df['LOI']) * df['Completes']
        
        # Create normalized versions of key variables
        df['IR_Normalized'] = (df['IR'] - df['IR'].mean()) / df['IR'].std()
        df['LOI_Normalized'] = (df['LOI'] - df['LOI'].mean()) / df['LOI'].std()
        df['Completes_Normalized'] = (df['Completes'] - df['Completes'].mean()) / df['Completes'].std()
        
        # Replace any infinite values that might have been created during division
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill any NaN values created during feature engineering with appropriate values
        # Use median for ratio metrics
        for col in ['IR_LOI_Ratio', 'IR_Completes_Ratio', 'LOI_Completes_Ratio', 
                    'CPI_Efficiency', 'IR_Normalized', 'LOI_Normalized', 'Completes_Normalized']:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Log counts of features created
        logger.info(f"Feature engineering complete. Created {len(df.columns) - 4} new features.")
        
        return df
    
    except Exception as e:
        logger.error(f"Error in engineer_features: {e}", exc_info=True)
        # Return original dataframe if feature engineering fails
        return df

def prepare_model_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for modeling with improved error handling tailored to CPI data.
    
    Args:
        df (pd.DataFrame): Input dataframe with all required columns
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features dataframe (X) and target series (y)
    """
    try:
        logger.info("Starting data preparation for modeling")
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Validate required columns exist
        required_cols = ['IR', 'LOI', 'Completes', 'CPI', 'Type']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {', '.join(missing_cols)}")
            return pd.DataFrame(), pd.Series()
        
        # Handle missing values
        numeric_cols = ['IR', 'LOI', 'Completes', 'CPI']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            na_count = df[col].isna().sum()
            if na_count > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.warning(f"Column {col}: Replaced {na_count} NaN values with median {median_val:.2f}")
            if col in ['LOI', 'Completes'] and (df[col] == 0).any():
                min_non_zero = df[df[col] > 0][col].min()
                df.loc[df[col] == 0, col] = min_non_zero
                logger.warning(f"Column {col}: Replaced zero values with minimum non-zero value {min_non_zero}")
        
        # Create engineered features specific to CPI analysis
        try:
            # Apply standard feature engineering
            df = engineer_features(df)
            
            # Add bin columns if not present
            if 'IR_Bin' not in df.columns:
                df = apply_all_bins(df)
                
            logger.info("Successfully created engineered features")
        except Exception as e:
            logger.warning(f"Error in feature engineering: {e}")
        
        # Select base features for model
        feature_cols = [
            'IR', 'LOI', 'Completes', 'IR_LOI_Ratio', 
            'IR_Completes_Ratio', 'Log_Completes', 'Type'
        ]
        
        # Add additional engineered features if available
        additional_cols = [
            'IR_LOI_Product', 'CPI_per_Minute', 'Log_IR',
            'Log_LOI', 'IR_Normalized', 'LOI_Normalized',
            'Completes_Normalized', 'IR_Completes_Product'
        ]
        
        for col in additional_cols:
            if col in df.columns:
                feature_cols.append(col)
        
        # Filter to available columns only
        available_cols = [col for col in feature_cols if col in df.columns]
        
        # Handle categorical variables
        if 'Type' in available_cols:
            valid_types = ['Won', 'Lost']
            invalid_types = df[~df['Type'].isin(valid_types)]['Type'].unique()
            if len(invalid_types) > 0:
                logger.warning(f"Found invalid Type values: {invalid_types}. Removing these rows.")
                df = df[df['Type'].isin(valid_types)]
            
            # Create dummy variables
            df = pd.get_dummies(df, columns=['Type'], drop_first=True)
            
            # Update available columns
            if 'Type_Won' in df.columns:
                available_cols = [col for col in available_cols if col != 'Type'] + ['Type_Won']
        
        # Add segment information if available
        if 'Segment' in df.columns and df['Segment'].notna().any():
            logger.info("Client segment data found, adding to model features")
            
            # Fill missing Segment values
            df['Segment'] = df['Segment'].fillna('Unknown')
            
            # Create dummy variables
            df = pd.get_dummies(df, columns=['Segment'], drop_first=True)
            
            # Add segment columns to features
            segment_cols = [col for col in df.columns if col.startswith('Segment_')]
            available_cols.extend(segment_cols)
        
        # Filter features and target
        X = df[available_cols].copy()
        y = df['CPI']
        
        # Remove rows with missing values
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]
        
        # Check if we have enough data
        if len(X) < 10:
            logger.error(f"Too few samples after preprocessing: {len(X)}")
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
        df (pd.DataFrame): Input dataframe
    
    Returns:
        Dict[str, Dict[str, float]]: Dictionary of summary metrics by Type
    """
    summary = {}
    
    try:
        # Group by Type
        grouped = df.groupby('Type')
        
        # Calculate summary statistics for each group
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
                'CPI_95th': group['CPI'].quantile(0.95)
            }
            
            # Add additional useful metrics for dashboard
            summary[name]['IR_LOI_Ratio_Avg'] = (group['IR'] / group['LOI']).mean()
            summary[name]['Cost_per_Min'] = (group['CPI'] / group['LOI']).mean()
            summary[name]['Cost_per_Complete'] = (group['CPI'] * group['Completes']).mean()
            
            # Add standard deviations for key metrics
            summary[name]['CPI_StdDev'] = group['CPI'].std()
            summary[name]['IR_StdDev'] = group['IR'].std()
            summary[name]['LOI_StdDev'] = group['LOI'].std()
            
            # Calculate the Efficiency Metric
            group_eff = (group['IR'] / 100) * (1 / group['LOI']) * group['Completes']
            summary[name]['Efficiency_Metric_Avg'] = group_eff.mean()
            summary[name]['Efficiency_Metric_Median'] = group_eff.median()
    
    except Exception as e:
        logger.error(f"Error in get_data_summary: {e}", exc_info=True)
    
    return summary