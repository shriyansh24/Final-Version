"""
Data loading functionality for the CPI Analysis & Prediction Dashboard.
Handles loading data from Excel files with enhanced data validation and initial preprocessing.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, List, Optional
import logging
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from config import (
    INVOICED_JOBS_FILE, 
    LOST_DEALS_FILE, 
    ACCOUNT_SEGMENT_FILE, 
    FEATURE_ENGINEERING_CONFIG,
    COLOR_SYSTEM
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

COLUMN_MAPPING = {
    'Incidence Rate': 'IR',
    'IR%': 'IR',
    'Length of Interview': 'LOI',
    'Interview Length': 'LOI',
    'Number of Completes': 'Completes',
    'Customer Rate': 'CPI',
    'Cost Per Interview': 'CPI',
}

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns in a DataFrame based on known mappings to standard names.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Standardized DataFrame.
    """
    return df.rename(columns={k: v for k, v in COLUMN_MAPPING.items() if k in df.columns})

def validate_data_files() -> bool:
    """
    Validate that all required data files exist.
    
    Returns:
        bool: True if all required files exist, False otherwise.
    """
    files_to_check = [INVOICED_JOBS_FILE, LOST_DEALS_FILE]
    
    # Check optional files separately
    optional_files = [ACCOUNT_SEGMENT_FILE]
    
    missing_files = [str(f) for f in files_to_check if not os.path.exists(f)]
    
    if missing_files:
        logger.error(f"Missing required data files: {', '.join(missing_files)}")
        return False
        
    # Log warning for missing optional files
    missing_optional = [str(f) for f in optional_files if not os.path.exists(f)]
    if missing_optional:
        logger.warning(f"Missing optional data files: {', '.join(missing_optional)}")
    
    return True

def detect_data_anomalies(df: pd.DataFrame, dataset_name: str) -> List[str]:
    """
    Detect potential anomalies in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        dataset_name (str): Name of the dataset for logging.
        
    Returns:
        List[str]: List of anomaly messages.
    """
    anomalies = []
    
    # Check for duplicated rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        message = f"{dataset_name}: Found {duplicate_count} duplicate rows"
        logger.warning(message)
        anomalies.append(message)
    
    # Check key numeric columns for anomalies
    numeric_cols = ['IR', 'LOI', 'Completes', 'CPI']
    for col in numeric_cols:
        if col in df.columns:
            # Check for zeros that should likely be positive
            if col in ['LOI', 'CPI', 'Completes'] and (df[col] == 0).any():
                zeros = (df[col] == 0).sum()
                message = f"{dataset_name}: Column '{col}' has {zeros} zero values"
                logger.warning(message)
                anomalies.append(message)
                
            # Check for extreme outliers (beyond 5 standard deviations)
            if df[col].dtype.kind in 'ifc':  # integer, float, or complex
                mean, std = df[col].mean(), df[col].std()
                if not pd.isna(std) and std > 0:
                    outliers = df[df[col] > mean + 5*std].shape[0]
                    if outliers > 0:
                        message = f"{dataset_name}: Column '{col}' has {outliers} extreme high outliers"
                        logger.info(message)
                        anomalies.append(message)
    
    # Check for missing values
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        message = f"{dataset_name}: Missing values detected in columns: {', '.join(missing.index.tolist())}"
        logger.info(message)
        anomalies.append(message)
    
    return anomalies

@st.cache_data(ttl=3600)
def load_data() -> Dict[str, pd.DataFrame]:
    """
    Load and process data from Excel files with enhanced data validation and imputation.
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing:
        - 'won': Won deals dataframe.
        - 'won_filtered': Won deals with extreme values removed.
        - 'lost': Lost deals dataframe.
        - 'lost_filtered': Lost deals with extreme values removed.
        - 'combined': Combined won and lost deals.
        - 'combined_filtered': Combined dataset with outliers removed.
        - 'data_quality': Information about data quality issues.
    """
    data_quality_issues = []
    
    try:
        # Check if running in Streamlit and prepare progress indicators
        progress_placeholder = None
        in_streamlit = True
        try:
            _ = st.empty()
        except Exception:
            in_streamlit = False
            
        if in_streamlit:
            progress_placeholder = st.empty()
            # Use dark-themed styling for progress message
            progress_placeholder.markdown(f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
                margin-bottom: 1rem;
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                <h3 style="margin-top: 0; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Checking data files...</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Validate all required files exist
        if not validate_data_files():
            if in_streamlit:
                st.error("Required data files are missing. Please check the data directory.")
            raise FileNotFoundError("Required data files are missing. Check log for details.")
        
        if in_streamlit:
            progress_placeholder.markdown(f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
                margin-bottom: 1rem;
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                <h3 style="margin-top: 0; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Loading invoiced jobs data...</h3>
            </div>
            """, unsafe_allow_html=True)
            
        # Load won deals (invoiced jobs)
        won_df = pd.read_excel(INVOICED_JOBS_FILE)
        
        # Initial data validation for won_df
        if won_df.empty:
            st.error("The invoiced jobs file is empty.")
            raise ValueError("The invoiced jobs file is empty.")
            
        # Detect anomalies in won data
        won_anomalies = detect_data_anomalies(won_df, "Won deals")
        data_quality_issues.extend(won_anomalies)
        
        if in_streamlit:
            progress_placeholder.markdown(f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
                margin-bottom: 1rem;
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                <h3 style="margin-top: 0; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Processing invoiced jobs data...</h3>
            </div>
            """, unsafe_allow_html=True)
            
        # Basic preprocessing for won_df
        # Ensure 'Type' column exists and is set to 'Won'
        won_df['Type'] = 'Won'
        
        # Convert key columns to numeric with warnings for non-numeric values
        numeric_cols = ['IR', 'LOI', 'Completes', 'CPI', 'CPI_ORIGINAL']
        for col in numeric_cols:
            if col in won_df.columns:
                non_numeric = pd.to_numeric(won_df[col], errors='coerce').isna() & ~won_df[col].isna()
                if non_numeric.any():
                    message = f"Won deals: Found {non_numeric.sum()} non-numeric values in '{col}', converting to NaN for imputation"
                    logger.warning(message)
                    data_quality_issues.append(message)
                won_df[col] = pd.to_numeric(won_df[col], errors='coerce')
        
        if in_streamlit:
            progress_placeholder.markdown(f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
                margin-bottom: 1rem;
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                <h3 style="margin-top: 0; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Loading lost deals data...</h3>
            </div>
            """, unsafe_allow_html=True)
            
        # Load lost deals
        lost_df = pd.read_excel(LOST_DEALS_FILE)
        # Inject CPI if not present (from Customer Rate)
        if 'CPI' not in lost_df.columns and 'Customer Rate' in lost_df.columns:
            lost_df['CPI'] = lost_df['Customer Rate']
            logger.warning("'CPI' column missing in lost_df. Using 'Customer Rate' as CPI (per Data Dictionary).")

        # CPI from Customer Rate
        if 'CPI' not in lost_df.columns and 'Customer Rate' in lost_df.columns:
            lost_df['CPI'] = lost_df['Customer Rate']
            logger.warning("'CPI' column missing in lost_df. Using 'Customer Rate' as CPI (per Data Dictionary).")

        # IR from alternative naming
        if 'IR' not in lost_df.columns:
            possible_ir_cols = [col for col in lost_df.columns if col.strip().lower() == 'ir']
            if possible_ir_cols:
                lost_df['IR'] = lost_df[possible_ir_cols[0]]
                logger.warning(f"'IR' column inferred from '{possible_ir_cols[0]}'")
            else:
                lost_df['IR'] = np.nan
                logger.warning("'IR' column not found in lost_df. Filling with NaN.")

        # LOI
        if 'LOI' not in lost_df.columns:
            possible_loi_cols = [col for col in lost_df.columns if col.strip().lower() == 'loi']
            if possible_loi_cols:
                lost_df['LOI'] = lost_df[possible_loi_cols[0]]
                logger.warning(f"'LOI' column inferred from '{possible_loi_cols[0]}'")
            else:
                lost_df['LOI'] = np.nan
                logger.warning("'LOI' column not found in lost_df. Filling with NaN.")

        # Completes (does not exist in lost file — fill placeholder)
        if 'Completes' not in lost_df.columns:
            lost_df['Completes'] = np.nan
            logger.warning("'Completes' column not found in lost_df. Filling with NaN.")
            
        
        # Initial data validation for lost_df
        if lost_df.empty:
            st.error("The lost deals file is empty.")
            raise ValueError("The lost deals file is empty.")
            
        # Detect anomalies in lost data
        lost_anomalies = detect_data_anomalies(lost_df, "Lost deals")
        data_quality_issues.extend(lost_anomalies)
        
        if in_streamlit:
            progress_placeholder.markdown(f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
                margin-bottom: 1rem;
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                <h3 style="margin-top: 0; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Processing lost deals data...</h3>
            </div>
            """, unsafe_allow_html=True)
            
        # Basic preprocessing for lost_df
        # Ensure 'Type' column exists and is set to 'Lost'
        lost_df['Type'] = 'Lost'
        
        # Convert key columns to numeric with warnings for non-numeric values
        for col in numeric_cols:
            if col in lost_df.columns:
                non_numeric = pd.to_numeric(lost_df[col], errors='coerce').isna() & ~lost_df[col].isna()
                if non_numeric.any():
                    message = f"Lost deals: Found {non_numeric.sum()} non-numeric values in '{col}', converting to NaN for imputation"
                    logger.warning(message)
                    data_quality_issues.append(message)
                lost_df[col] = pd.to_numeric(lost_df[col], errors='coerce')
        
        if in_streamlit:
            progress_placeholder.markdown(f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
                margin-bottom: 1rem;
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                <h3 style="margin-top: 0; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Loading and processing client segment data...</h3>
            </div>
            """, unsafe_allow_html=True)
            
        # Load account segment data if available
        if os.path.exists(ACCOUNT_SEGMENT_FILE):
            try:
                account_segment_df = pd.read_csv(ACCOUNT_SEGMENT_FILE)
                
                if 'AccountID' in account_segment_df.columns and 'Segment' in account_segment_df.columns:
                    # Add segment to won_df and lost_df if they have AccountID columns
                    if 'AccountID' in won_df.columns:
                        won_df = won_df.merge(account_segment_df[['AccountID', 'Segment']], 
                                             on='AccountID', how='left')
                    
                    if 'AccountID' in lost_df.columns:
                        lost_df = lost_df.merge(account_segment_df[['AccountID', 'Segment']], 
                                               on='AccountID', how='left')
                    
                    logger.info("Successfully merged account segment data")
                else:
                    logger.warning("Account segment file doesn't have expected columns")
                    data_quality_issues.append("Account segment file doesn't have expected columns")
            except Exception as e:
                logger.error(f"Error loading account segment data: {str(e)}")
                data_quality_issues.append(f"Error loading account segment data: {str(e)}")
        
        if in_streamlit:
            progress_placeholder.markdown(f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
                margin-bottom: 1rem;
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                <h3 style="margin-top: 0; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Filtering and preprocessing data...</h3>
            </div>
            """, unsafe_allow_html=True)
            
        # Combine datasets for further processing
        combined_df = pd.concat([won_df, lost_df], ignore_index=True)
        
        # Apply basic filtering (handle extreme values)
        # Calculate 95th percentile for CPI to identify outliers
        cpi_95th = combined_df['CPI'].quantile(0.95)
        outlier_threshold = FEATURE_ENGINEERING_CONFIG.get('outlier_threshold', 0.95)
        
        # Create filtered datasets (with extreme values removed)
        won_df_filtered = won_df[won_df['CPI'] <= cpi_95th].copy()
        lost_df_filtered = lost_df[lost_df['CPI'] <= cpi_95th].copy()
        combined_df_filtered = combined_df[combined_df['CPI'] <= cpi_95th].copy()
        
        # Log information about filtered data
        won_filtered_pct = 100 * (1 - len(won_df_filtered) / len(won_df)) if len(won_df) > 0 else 0
        lost_filtered_pct = 100 * (1 - len(lost_df_filtered) / len(lost_df)) if len(lost_df) > 0 else 0
        
        if won_filtered_pct > 0:
            message = f"Filtered out {won_filtered_pct:.1f}% of won deals with CPI > {cpi_95th:.2f}"
            logger.info(message)
            data_quality_issues.append(message)
            
        if lost_filtered_pct > 0:
            message = f"Filtered out {lost_filtered_pct:.1f}% of lost deals with CPI > {cpi_95th:.2f}"
            logger.info(message)
            data_quality_issues.append(message)
            
        if in_streamlit:
            progress_placeholder.markdown(f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid {COLOR_SYSTEM['ACCENT']['GREEN']};
                margin-bottom: 1rem;
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                <h3 style="margin-top: 0; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Successfully loaded data</h3>
                <ul style="margin-bottom: 0; color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                    <li><b>Won bids</b>: {len(won_df)}</li>
                    <li><b>Lost bids</b>: {len(lost_df)}</li>
                    <li><b>Combined</b>: {len(combined_df)}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        # Return all datasets
        return {
            'won': won_df,
            'won_filtered': won_df_filtered,
            'lost': lost_df,
            'lost_filtered': lost_df_filtered,
            'combined': combined_df,
            'combined_filtered': combined_df_filtered,
            'data_quality_issues': data_quality_issues
        }
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}", exc_info=True)
        if in_streamlit and progress_placeholder:
            progress_placeholder.markdown(f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid {COLOR_SYSTEM['ACCENT']['RED']};
                margin-bottom: 1rem;
                color: {COLOR_SYSTEM['ACCENT']['RED']};
            ">
                <h3 style="margin-top: 0; color: {COLOR_SYSTEM['ACCENT']['RED']};">Error loading data</h3>
                <p style="margin-bottom: 0;">{str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
        raise
