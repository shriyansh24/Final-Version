"""
Configuration settings for the CPI Analysis & Prediction Dashboard.
Centralizes all configuration parameters for easy maintenance.
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Data files
INVOICED_JOBS_FILE = DATA_DIR / "invoiced_jobs_this_year_20240912T18_36_36.439126Z.xlsx"
LOST_DEALS_FILE = DATA_DIR / "DealItemReportLOST.xlsx"
ACCOUNT_SEGMENT_FILE = DATA_DIR / "Account+List+with+Segment.csv"

# App configuration
APP_TITLE = "CPI Analysis & Prediction Dashboard"
APP_ICON = "📊"
APP_LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

# Comprehensive color system
COLOR_SYSTEM = {
    # Primary palette
    'PRIMARY': {
        'MAIN': '#2C3E50',       # Dark blue-slate - headers, primary elements
        'LIGHT': '#34495E',      # Lighter slate - secondary elements
        'DARK': '#1A252F',       # Darker slate - footer, emphasis
        'CONTRAST': '#ECF0F1'    # Off-white - text on dark backgrounds
    },
    
    # Accent colors
    'ACCENT': {
        'BLUE': '#3498DB',       # Bright blue - won bids, primary accent
        'ORANGE': '#E67E22',     # Orange - lost bids
        'GREEN': '#2ECC71',      # Green - positive indicators
        'RED': '#E74C3C',        # Red - negative indicators
        'PURPLE': '#9B59B6',     # Purple - predictions
        'YELLOW': '#F1C40F'      # Yellow - warnings, highlights
    },
    
    # Neutral tones
    'NEUTRAL': {
        'WHITE': '#FFFFFF',
        'LIGHTEST': '#F8F9FA',
        'LIGHTER': '#E9ECEF',
        'LIGHT': '#DEE2E6',
        'MEDIUM': '#CED4DA',
        'DARK': '#ADB5BD',
        'DARKER': '#6C757D',
        'DARKEST': '#343A40',
        'BLACK': '#212529'
    },
    
    # Semantic colors (for specific meanings)
    'SEMANTIC': {
        'SUCCESS': '#27AE60',
        'WARNING': '#F39C12',
        'ERROR': '#C0392B',
        'INFO': '#2980B9'
    },
    
    # Chart-specific colors 
    'CHARTS': {
        'WON': '#3498DB',        # Blue
        'LOST': '#E67E22',       # Orange
        'WON_TRANS': 'rgba(52, 152, 219, 0.7)',
        'LOST_TRANS': 'rgba(230, 126, 34, 0.7)',
        'GRADIENT_1': '#2980B9',
        'GRADIENT_2': '#6DD5FA',
        'GRADIENT_3': '#FFFFFF'
    },
    
    # Background colors
    'BACKGROUND': {
        'MAIN': '#F8F9FA',       # Main background
        'CARD': '#FFFFFF',       # Card background
        'DARK': '#2C3E50',       # Dark sections
        'ALT': '#EBF5FB'         # Alternate background
    }
}

# Typography
TYPOGRAPHY = {
    'FONT_FAMILY': '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    'HEADING': {
        'H1': {'size': '2rem', 'weight': '700', 'height': '1.2'},
        'H2': {'size': '1.5rem', 'weight': '600', 'height': '1.3'},
        'H3': {'size': '1.17rem', 'weight': '600', 'height': '1.4'},
        'H4': {'size': '1rem', 'weight': '600', 'height': '1.5'}
    },
    'BODY': {
        'LARGE': {'size': '1.1rem', 'weight': '400', 'height': '1.5'},
        'NORMAL': {'size': '1rem', 'weight': '400', 'height': '1.5'},
        'SMALL': {'size': '0.875rem', 'weight': '400', 'height': '1.4'}
    }
}

# Data binning configurations
IR_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
IR_BIN_LABELS = ['0-10', '10-20', '20-30', '30-40', '40-50', 
                '50-60', '60-70', '70-80', '80-90', '90-100']

LOI_BINS = [0, 5, 10, 15, 20, float('inf')]
LOI_BIN_LABELS = ['Very Short (1-5 min)', 'Short (6-10 min)', 
                 'Medium (11-15 min)', 'Long (16-20 min)', 'Very Long (20+ min)']

COMPLETES_BINS = [0, 100, 500, 1000, float('inf')]
COMPLETES_BIN_LABELS = ['Small (1-100)', 'Medium (101-500)', 
                       'Large (501-1000)', 'Very Large (1000+)']

# Visualization settings
# For backward compatibility
WON_COLOR = COLOR_SYSTEM['CHARTS']['WON']
LOST_COLOR = COLOR_SYSTEM['CHARTS']['LOST']

# Chart styling defaults
CHART_HEIGHT = 500
CHART_MARGIN = dict(l=10, r=10, t=50, b=10)
CHART_BACKGROUND = COLOR_SYSTEM['NEUTRAL']['WHITE']
CHART_GRID_COLOR = COLOR_SYSTEM['NEUTRAL']['LIGHT']

# Heatmap color scales (color-blind friendly)
HEATMAP_COLORSCALE_WON = 'Viridis'  # Good color-blind friendly option for sequential data
HEATMAP_COLORSCALE_LOST = 'Plasma'  # Another good color-blind friendly option

# Model configurations
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
HYPERPARAMETER_TUNING = False  # Whether to perform hyperparameter tuning by default

# Cache settings
CACHE_TTL = 3600  # Cache time-to-live in seconds (1 hour)

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# File paths for saving models
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Default model configurations
DEFAULT_MODELS = {
    'Linear Regression': {
        'fit_intercept': True,
        'positive': False
    },
    'Random Forest': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'random_state': RANDOM_STATE
    },
    'Gradient Boosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'random_state': RANDOM_STATE
    }
}

# Feature engineering settings
FEATURE_ENGINEERING_CONFIG = {
    'create_interaction_terms': True,
    'create_log_transforms': True,
    'handle_outliers': True,
    'outlier_threshold': 0.95  # 95th percentile for outlier detection
}

# Dashboard section settings
SHOW_ADVANCED_OPTIONS = False  # Whether to show advanced options by default
DATA_SAMPLE_SIZE = 1000  # Maximum number of rows to display in data tables