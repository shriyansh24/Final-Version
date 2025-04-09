"""
Configuration settings for the CPI Analysis & Prediction Dashboard.
Centralizes all configuration parameters for easy maintenance.
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
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

# High Contrast & Color Blind Friendly Color System
COLOR_SYSTEM = {
    "PRIMARY": {
        "MAIN": "#FFFFFF",  # White text on dark backgrounds
        "LIGHT": "#F2F2F2",  # Lighter elements, e.g. secondary text or borders
        "CONTRAST": "#000000"  # Black for contrast on white elements
    },
    "ACCENT": {
        "BLUE": "#00BFFF",  # Bright blue for primary accents and active elements
        "ORANGE": "#FFB74D",  # Vivid orange for lost bids and warnings
        "GREEN": "#66FF99",  # Neon green for successes
        "RED": "#FF3333",  # Bright red for errors/alerts
        "PURPLE": "#BB86FC",  # VS Code–inspired purple for predictions
        "YELLOW": "#FFFF00"  # Bright yellow for highlights
    },
    "NEUTRAL": {
        "WHITE": "#FFFFFF",
        "LIGHTEST": "#1F1F1F",  # Background for cards
        "LIGHTER": "#2E2E2E",
        "LIGHT": "#3A3A3A",
        "MEDIUM": "#4B4B4B",
        "DARK": "#707070",
        "DARKEST": "#000000"  # Use pure black for high–contrast text if needed
    },
    "SEMANTIC": {
        "SUCCESS": "#81FF7F",
        "WARNING": "#FFBF00",
        "ERROR": "#FF564B",
        "INFO": "#00FFFF"
    },
    "CHARTS": {
        "WON": "#00CFFF",  # Vivid blue for won bids
        "LOST": "#FFB74D",  # Vivid orange for lost bids
        "WON_TRANS": "rgba(0,207,255,0.7)",
        "LOST_TRANS": "rgba(255,183,77,0.7)",
        "GRADIENT_1": "#FF0080",
        "GRADIENT_2": "#FFD500",
        "GRADIENT_3": "#FFFFFF"
    },
    "BACKGROUND": {
        "MAIN": "#000000",  # Full black background for high contrast
        "CARD": "#1F1F1F",  # Dark gray for cards or panels
        "DARK": "#000000",
        "ALT": "#121212"  # Alternate dark background (very subtle differences)
    }
}

# Typography settings
TYPOGRAPHY = {
    "FONT_FAMILY": "\"Inter\", -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif",
    "HEADING": {
        "H1": {"size": "2rem", "weight": "700", "height": "1.2"},
        "H2": {"size": "1.5rem", "weight": "600", "height": "1.3"},
        "H3": {"size": "1.2rem", "weight": "600", "height": "1.4"},
        "H4": {"size": "1rem", "weight": "600", "height": "1.5"}
    },
    "BODY": {
        "LARGE": {"size": "1.1rem", "weight": "400", "height": "1.5"},
        "NORMAL": {"size": "1rem", "weight": "400", "height": "1.5"},
        "SMALL": {"size": "0.875rem", "weight": "400", "height": "1.4"}
    }
}

# Data binning configurations
IR_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
IR_BIN_LABELS = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
LOI_BINS = [0, 5, 10, 15, 20, float('inf')]
LOI_BIN_LABELS = [
    'Very Short (1-5 min)',
    'Short (6-10 min)',
    'Medium (11-15 min)',
    'Long (16-20 min)',
    'Very Long (20+ min)'
]

COMPLETES_BINS = [0, 100, 500, 1000, float('inf')]
COMPLETES_BIN_LABELS = [
    'Small (1-100)',
    'Medium (101-500)',
    'Large (501-1000)',
    'Very Large (1000+)'
]

# Visualization defaults
WON_COLOR = COLOR_SYSTEM['CHARTS']['WON']
LOST_COLOR = COLOR_SYSTEM['CHARTS']['LOST']
CHART_HEIGHT = 500
CHART_MARGIN = dict(l=10, r=10, t=50, b=10)
CHART_BACKGROUND = COLOR_SYSTEM['BACKGROUND']['CARD']  # Using dark background for charts
CHART_GRID_COLOR = COLOR_SYSTEM['NEUTRAL']['LIGHT']

# Heatmap color scales (using colorblind friendly options)
HEATMAP_COLORSCALE_WON = 'Viridis'
HEATMAP_COLORSCALE_LOST = 'Plasma'

# Model configurations
RANDOM_STATE = 42
TEST_SIZE = 0.2

# File paths for saving models and other artifacts
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

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

FEATURE_ENGINEERING_CONFIG = {
    'create_interaction_terms': True,
    'create_log_transforms': True,
    'handle_outliers': True,
    'outlier_threshold': 0.95  # 95th percentile to filter extreme values
}

SHOW_ADVANCED_OPTIONS = False
DATA_SAMPLE_SIZE = 1000
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
