"""
Data loading functionality for the CPI Analysis & Prediction Dashboard.
Handles loading data from Excel files and initial preprocessing.
"""

import os
import pandas as pd
import streamlit as st
from typing import Dict, Any
import logging
from config import (INVOICED_JOBS_FILE, LOST_DEALS_FILE, ACCOUNT_SEGMENT_FILE,
                   FEATURE_ENGINEERING_CONFIG, COLOR_SYSTEM)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_data() -> Dict[str, pd.DataFrame]:
    """
    Load and process the data from Excel files.
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing different dataframes:
            - 'won': Won deals dataframe
            - 'won_filtered': Won deals with extreme values filtered out
            - 'lost': Lost deals dataframe
            - 'lost_filtered': Lost deals with extreme values filtered out
            - 'combined': Combined dataframe of won and lost deals
            - 'combined_filtered': Combined dataframe with extreme values filtered out
    
    Raises:
        FileNotFoundError: If any of the required data files are missing
        ValueError: If data processing fails due to unexpected data structure
    """
    try:
        progress_placeholder = None
        progress_bar = None
        
        # Check if in Streamlit context for progress reporting
        in_streamlit = True
        try:
            _ = st.empty()
        except:
            in_streamlit = False
            
        if in_streamlit:
            progress_placeholder = st.empty()
            progress_placeholder.markdown(f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                border-radius: 0.5rem;
                padding: 1rem;
                margin-bottom: 1rem;
                border-left: 4px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
            ">
                <h4 style="margin-top: 0;">Data Loading: Initializing...</h4>
                <p>Checking data files...</p>
            </div>
            """, unsafe_allow_html=True)
            
            progress_bar = st.progress(0)
        
        # Check if files exist
        if not os.path.exists(INVOICED_JOBS_FILE):
            error_message = f"Could not find the invoiced jobs file: {INVOICED_JOBS_FILE}"
            logger.error(error_message)
            if in_streamlit:
                progress_placeholder.error(error_message)
            raise FileNotFoundError(error_message)
            
        if not os.path.exists(LOST_DEALS_FILE):
            error_message = f"Could not find the lost deals file: {LOST_DEALS_FILE}"
            logger.error(error_message)
            if in_streamlit:
                progress_placeholder.error(error_message)
            raise FileNotFoundError(error_message)
        
        # Update progress
        if in_streamlit:
            progress_bar.progress(10)
            progress_placeholder.markdown(f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                border-radius: 0.5rem;
                padding: 1rem;
                margin-bottom: 1rem;
                border-left: 4px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
            ">
                <h4 style="margin-top: 0;">Data Loading: 10%</h4>
                <p>Loading invoiced jobs data...</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Load invoiced jobs data (Won deals)
        logger.info(f"Loading invoiced jobs data from {INVOICED_JOBS_FILE}")
        invoiced_df = pd.read_excel(INVOICED_JOBS_FILE)
        
        # Log column names for debugging
        logger.debug(f"Columns in invoiced_df: {invoiced_df.columns.tolist()}")
        
        # Update progress
        if in_streamlit:
            progress_bar.progress(25)
            progress_placeholder.markdown(f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                border-radius: 0.5rem;
                padding: 1rem;
                margin-bottom: 1rem;
                border-left: 4px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
            ">
                <h4 style="margin-top: 0;">Data Loading: 25%</h4>
                <p>Processing invoiced jobs data...</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Rename columns to remove spaces
        invoiced_df = invoiced_df.rename(columns={
            ' CPI ': 'CPI',
            ' Actual Project Revenue ': 'Revenue',
            'Actual Project Revenue': 'Revenue',
            ' Revenue ': 'Revenue',
            'Revenue ': 'Revenue'
        })
        
        # Process Countries column
        invoiced_df['Countries'] = invoiced_df['Countries'].fillna('[]')
        invoiced_df['Country'] = invoiced_df['Countries'].apply(
            lambda x: x.replace('[', '').replace(']', '').replace('"', '')
        )
        invoiced_df['Country'] = invoiced_df['Country'].replace('', 'USA')
        
        # Create Won dataset
        won_df = invoiced_df[[
            'Project Code Parent', 'Client Name', 'CPI', 'Actual Ir', 'Actual Loi', 
            'Complete', 'Revenue', 'Invoiced Date', 'Country', 'Audience'
        ]].copy()
        
        # Rename columns for consistency
        won_df = won_df.rename(columns={
            'Project Code Parent': 'ProjectId',
            'Client Name': 'Client',
            'Actual Ir': 'IR',
            'Actual Loi': 'LOI',
            'Complete': 'Completes',
            'Invoiced Date': 'Date'
        })
        
        # Add type column
        won_df['Type'] = 'Won'
        
        # Update progress
        if in_streamlit:
            progress_bar.progress(40)
            progress_placeholder.markdown(f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                border-radius: 0.5rem;
                padding: 1rem;
                margin-bottom: 1rem;
                border-left: 4px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
            ">
                <h4 style="margin-top: 0;">Data Loading: 40%</h4>
                <p>Loading lost deals data...</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Load lost deals data
        logger.info(f"Loading lost deals data from {LOST_DEALS_FILE}")
        lost_df_raw = pd.read_excel(LOST_DEALS_FILE)
        
        # Update progress
        if in_streamlit:
            progress_bar.progress(55)
            progress_placeholder.markdown(f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                border-radius: 0.5rem;
                padding: 1rem;
                margin-bottom: 1rem;
                border-left: 4px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
            ">
                <h4 style="margin-top: 0;">Data Loading: 55%</h4>
                <p>Processing lost deals data...</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Filter for Sample items only
        lost_df = lost_df_raw[lost_df_raw['Item'] == 'Sample'].copy()
        
        # Create Lost dataset
        lost_df = lost_df[[
            'Record Id', 'Account Name', 'Customer Rate', 'IR', 'LOI', 
            'Qty', 'Item Amount', 'Description (Items)', 'Deal Name'
        ]].copy()
        
        # Rename columns for consistency
        lost_df = lost_df.rename(columns={
            'Record Id': 'DealId',
            'Account Name': 'Client',
            'Customer Rate': 'CPI',
            'Qty': 'Completes',
            'Item Amount': 'Revenue',
            'Description (Items)': 'Country',
            'Deal Name': 'ProjectName'
        })
        
        # Add type column
        lost_df['Type'] = 'Lost'
        
        # Update progress
        if in_streamlit:
            progress_bar.progress(70)
            progress_placeholder.markdown(f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                border-radius: 0.5rem;
                padding: 1rem;
                margin-bottom: 1rem;
                border-left: 4px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
            ">
                <h4 style="margin-top: 0;">Data Loading: 70%</h4>
                <p>Loading and processing client segment data...</p>
            </div>
            """, unsafe_allow_html=True)
            
        # Try to load account segment data
        if os.path.exists(ACCOUNT_SEGMENT_FILE):
            logger.info(f"Loading account segment data from {ACCOUNT_SEGMENT_FILE}")
            try:
                client_segments = pd.read_csv(ACCOUNT_SEGMENT_FILE)
                # Clean up column names
                client_segments = client_segments.rename(columns={
                    'Account Name': 'Client',
                    'Client Segment Type': 'Segment'
                })
                
                # Log Segment distribution before merging
                logger.info(f"Segment distribution before merging: {client_segments['Segment'].value_counts().to_dict()}")

                # Clean up Client names for better matching
                client_segments['Client'] = client_segments['Client'].str.strip().str.upper()
                won_df['Client'] = won_df['Client'].str.strip().str.upper()
                lost_df['Client'] = lost_df['Client'].str.strip().str.upper()

                # Merge segments with won and lost dataframes
                won_df = pd.merge(
                    won_df, 
                    client_segments[['Client', 'Segment']], 
                    on='Client', 
                    how='left'
                )
                lost_df = pd.merge(
                    lost_df, 
                    client_segments[['Client', 'Segment']], 
                    on='Client', 
                    how='left'
                )
                
                # Log merge success rate
                won_match = won_df['Segment'].notna().mean() * 100
                lost_match = lost_df['Segment'].notna().mean() * 100
                logger.info(f"Won deals segment match rate: {won_match:.2f}%")
                logger.info(f"Lost deals segment match rate: {lost_match:.2f}%")
                logger.info("Successfully merged client segment data")
            except Exception as e:
                logger.warning(f"Failed to load or merge segment data: {e}")
        
        # Update progress
        if in_streamlit:
            progress_bar.progress(85)
            progress_placeholder.markdown(f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                border-radius: 0.5rem;
                padding: 1rem;
                margin-bottom: 1rem;
                border-left: 4px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
            ">
                <h4 style="margin-top: 0;">Data Loading: 85%</h4>
                <p>Filtering and preprocessing data...</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Convert CPI columns to numeric before filtering
        won_df['CPI'] = pd.to_numeric(won_df['CPI'], errors='coerce')
        lost_df['CPI'] = pd.to_numeric(lost_df['CPI'], errors='coerce')
        
        # Make sure numeric columns are numeric
        for col in ['IR', 'LOI', 'Completes', 'Revenue']:
            won_df[col] = pd.to_numeric(won_df[col], errors='coerce')
            lost_df[col] = pd.to_numeric(lost_df[col], errors='coerce')
        
        # Filter out invalid CPI values
        won_df = won_df[won_df['CPI'].notna() & (won_df['CPI'] > 0)]
        lost_df = lost_df[lost_df['CPI'].notna() & (lost_df['CPI'] > 0)]
        
        # Log data shapes
        logger.info(f"Won deals count: {len(won_df)}")
        logger.info(f"Lost deals count: {len(lost_df)}")
        
        # Filter out extreme values based on config
        outlier_threshold = FEATURE_ENGINEERING_CONFIG.get('outlier_threshold', 0.95)
        
        won_percentile = won_df['CPI'].quantile(outlier_threshold)
        lost_percentile = lost_df['CPI'].quantile(outlier_threshold)
        
        won_df_filtered = won_df[won_df['CPI'] <= won_percentile]
        lost_df_filtered = lost_df[lost_df['CPI'] <= lost_percentile]
        
        # Determine common columns for combined dataset
        common_columns = ['Client', 'CPI', 'IR', 'LOI', 'Completes', 'Revenue', 'Country', 'Type']
        
        # Add Segment if it exists
        if 'Segment' in won_df.columns and 'Segment' in lost_df.columns:
            common_columns.append('Segment')
        
        # Create a single combined dataframe with only the common columns
        combined_df = pd.concat(
            [won_df[common_columns], lost_df[common_columns]],
            ignore_index=True
        )
        
        # Create a filtered combined dataset
        combined_df_filtered = pd.concat(
            [won_df_filtered[common_columns], lost_df_filtered[common_columns]],
            ignore_index=True
        )
        
        # Update progress
        if in_streamlit:
            progress_bar.progress(95)
            progress_placeholder.markdown(f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                border-radius: 0.5rem;
                padding: 1rem;
                margin-bottom: 1rem;
                border-left: 4px solid {COLOR_SYSTEM['ACCENT']['GREEN']};
            ">
                <h4 style="margin-top: 0;">Data Loading: 95%</h4>
                <p>Finalizing data preparation...</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Log data shapes after filtering
        logger.info(f"Won deals filtered count: {len(won_df_filtered)}")
        logger.info(f"Lost deals filtered count: {len(lost_df_filtered)}")
        logger.info(f"Combined filtered count: {len(combined_df_filtered)}")
        
        # Calculate filtering stats for logging
        won_filter_pct = (len(won_df) - len(won_df_filtered)) / len(won_df) * 100 if len(won_df) > 0 else 0
        lost_filter_pct = (len(lost_df) - len(lost_df_filtered)) / len(lost_df) * 100 if len(lost_df) > 0 else 0
        
        logger.info(f"Won deals filtered out: {won_filter_pct:.1f}% (CPI > {won_percentile:.2f})")
        logger.info(f"Lost deals filtered out: {lost_filter_pct:.1f}% (CPI > {lost_percentile:.2f})")
        
        # Complete progress
        if in_streamlit:
            progress_bar.progress(100)
            progress_placeholder.markdown(f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                border-radius: 0.5rem;
                padding: 1rem;
                margin-bottom: 1rem;
                border-left: 4px solid {COLOR_SYSTEM['ACCENT']['GREEN']};
            ">
                <h4 style="margin-top: 0;">Data Loading: Complete</h4>
                <p>Successfully loaded {len(won_df)} won bids and {len(lost_df)} lost bids.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Clear progress bar after 1 second
            import time
            time.sleep(1)
            progress_bar.empty()
        
        # Return all datasets
        return {
            'won': won_df,
            'won_filtered': won_df_filtered,
            'lost': lost_df,
            'lost_filtered': lost_df_filtered,
            'combined': combined_df,
            'combined_filtered': combined_df_filtered
        }
    
    except Exception as e:
        logger.error(f"Error in load_data: {e}", exc_info=True)
        
        # Show error message in Streamlit if available
        if 'progress_placeholder' in locals() and progress_placeholder is not None:
            progress_placeholder.error(f"Error loading data: {str(e)}")
            
            # Provide additional help based on error type
            if isinstance(e, FileNotFoundError):
                progress_placeholder.markdown(f"""
                <div style="
                    background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                    border-radius: 0.5rem;
                    padding: 1rem;
                    margin-top: 1rem;
                    border-left: 4px solid {COLOR_SYSTEM['ACCENT']['YELLOW']};
                ">
                    <h4 style="margin-top: 0;">Troubleshooting Help</h4>
                    <p>Make sure the following data files exist in the correct location:</p>
                    <ul>
                        <li><code>{os.path.basename(INVOICED_JOBS_FILE)}</code></li>
                        <li><code>{os.path.basename(LOST_DEALS_FILE)}</code></li>
                    </ul>
                    <p>Check that the file paths in <code>config.py</code> are correct.</p>
                </div>
                """, unsafe_allow_html=True)
        
        raise

if __name__ == "__main__":
    # Test the data loading function
    try:
        data = load_data()
        print("Data loaded successfully.")
        for key, df in data.items():
            print(f"{key}: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")