"""
UI Components for the CPI Analysis & Prediction Dashboard.
This module provides reusable UI components and styling functions
for a modern, high-contrast dashboard.
"""

import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Callable, Optional, Union, Any

# Import color system and typography from config
from config import COLOR_SYSTEM, TYPOGRAPHY

def load_custom_css():
    """Load custom CSS to override Streamlit defaults."""
    # Check if assets directory exists
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    css_path = os.path.join(assets_dir, "style.css")
    if os.path.exists(css_path):
        with open(css_path, 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        # Fallback inline CSS if file not found
        st.markdown("""
        <style>
        /* Base styling */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #000000 !important;
            font-family: 'Inter', sans-serif;
            color: #FFFFFF;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #1F1F1F;
            border-right: 1px solid #3A3A3A;
        }
        
        /* Metric Styling */
        [data-testid="stMetric"] {
            background-color: #1F1F1F;
            border-radius: 0.5rem;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            border-left: 4px solid #00BFFF;
        }
        
        [data-testid="stMetricLabel"] {
            font-weight: 600 !important;
            color: #FFFFFF !important;
        }
        
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            color: #00BFFF !important;
        }
        
        /* Button Styling */
        .stButton > button {
            background-color: #00BFFF;
            color: #000000;
            border: none;
            border-radius: 0.3rem;
            padding: 0.6rem 1rem;
            font-weight: 600;
        }
        
        .stButton > button:hover {
            background-color: #00A3DB;
        }
        </style>
        """, unsafe_allow_html=True)

def render_logo():
    """Render the BidPricing Analytics logo."""
    logo_svg = """
    <svg width="200" height="48" viewBox="0 0 200 48" fill="none" xmlns="http://www.w3.org/2000/svg">
        <!-- Logo Background -->
        <rect width="48" height="48" rx="8" fill="#1F1F1F"/>
        
        <!-- Logo Graphic -->
        <path d="M14 24C14 18.4772 18.4772 14 24 14C29.5228 14 34 18.4772 34 24C34 29.5228 29.5228 34 24 34" stroke="#00BFFF" stroke-width="3" stroke-linecap="round"/>
        <path d="M24 34C18.4772 34 14 29.5228 14 24" stroke="#FFB74D" stroke-width="3" stroke-linecap="round"/>
        
        <!-- Chart Bars -->
        <rect x="20" y="20" width="2" height="8" rx="1" fill="#FFFFFF"/>
        <rect x="24" y="18" width="2" height="10" rx="1" fill="#FFFFFF"/>
        <rect x="28" y="22" width="2" height="6" rx="1" fill="#FFFFFF"/>
        
        <!-- Company Name -->
        <text x="56" y="28" font-family="Inter, sans-serif" font-weight="700" font-size="16" fill="#FFFFFF">BidPricing</text>
        <text x="56" y="38" font-family="Inter, sans-serif" font-weight="400" font-size="12" fill="#707070">Analytics</text>
    </svg>
    """
    
    return st.markdown(logo_svg, unsafe_allow_html=True)

def render_header(current_page="Dashboard"):
    """Render a branded header with navigation."""
    header_html = f"""
    <header style="
        background-color: #1F1F1F;
        padding: 1rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    ">
        <div style="display: flex; align-items: center;">
            <svg width="40" height="40" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                <!-- Logo Background -->
                <rect width="48" height="48" rx="8" fill="#121212"/>
                
                <!-- Logo Graphic -->
                <path d="M14 24C14 18.4772 18.4772 14 24 14C29.5228 14 34 18.4772 34 24C34 29.5228 29.5228 34 24 34" stroke="#00BFFF" stroke-width="3" stroke-linecap="round"/>
                <path d="M24 34C18.4772 34 14 29.5228 14 24" stroke="#FFB74D" stroke-width="3" stroke-linecap="round"/>
                
                <!-- Chart Bars -->
                <rect x="20" y="20" width="2" height="8" rx="1" fill="#FFFFFF"/>
                <rect x="24" y="18" width="2" height="10" rx="1" fill="#FFFFFF"/>
                <rect x="28" y="22" width="2" height="6" rx="1" fill="#FFFFFF"/>
            </svg>
            
            <div style="margin-left: 1rem;">
                <h1 style="
                    margin: 0;
                    font-size: 1.5rem;
                    font-weight: 700;
                    color: #FFFFFF;
                ">BidPricing Analytics</h1>
                <p style="
                    margin: 0;
                    font-size: 0.9rem;
                    color: #707070;
                ">CPI Analysis & Prediction Dashboard</p>
            </div>
        </div>
        
        <div>
            <p style="
                margin: 0;
                font-size: 1rem;
                font-weight: 500;
                color: #707070;
                text-align: right;
            ">
                <span style="color: #00BFFF;">●</span> {current_page}
            </p>
        </div>
    </header>
    """
    
    return st.markdown(header_html, unsafe_allow_html=True)

def render_card(title, content, icon=None, accent_color=None):
    """Render a custom card component with title and content."""
    if accent_color is None:
        accent_color = COLOR_SYSTEM['ACCENT']['BLUE']
    
    card_html = f"""
    <div style="
        background-color: #1F1F1F;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        border-left: 4px solid {accent_color};
    ">
        <div style="
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
        ">
            {f'<div style="margin-right: 0.75rem; font-size: 1.5rem; color: {accent_color};">{icon}</div>' if icon else ''}
            <h3 style="
                margin: 0;
                color: #FFFFFF;
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['HEADING']['H3']['size']};
                font-weight: {TYPOGRAPHY['HEADING']['H3']['weight']};
            ">{title}</h3>
        </div>
        <div style="color: #B0B0B0;">
            {content}
        </div>
    </div>
    """
    return st.markdown(card_html, unsafe_allow_html=True)

def apply_chart_styling(fig, title=None, height=500, show_legend=True):
    """Apply consistent styling to all charts."""
    fig.update_layout(
        title=dict(
            text=title if title else fig.layout.title.text,
            font=dict(
                family=TYPOGRAPHY['FONT_FAMILY'],
                size=20, 
                color="#FFFFFF"
            ),
            x=0.01,
            xanchor='left',
            y=0.95,
            yanchor='top'
        ),
        legend=dict(
            font=dict(
                family=TYPOGRAPHY['FONT_FAMILY'],
                size=12, 
                color="#FFFFFF"
            ),
            bgcolor="rgba(31, 31, 31, 0.8)",
            bordercolor="#3A3A3A",
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ) if show_legend else dict(visible=False),
        font=dict(
            family=TYPOGRAPHY['FONT_FAMILY'],
            size=12,
            color="#FFFFFF"
        ),
        xaxis=dict(
            title_font=dict(
                family=TYPOGRAPHY['FONT_FAMILY'],
                size=14,
                color="#FFFFFF"
            ),
            tickfont=dict(
                family=TYPOGRAPHY['FONT_FAMILY'],
                size=12,
                color="#707070"
            ),
            gridcolor="#3A3A3A",
            zerolinecolor="#4B4B4B"
        ),
        yaxis=dict(
            title_font=dict(
                family=TYPOGRAPHY['FONT_FAMILY'],
                size=14,
                color="#FFFFFF"
            ),
            tickfont=dict(
                family=TYPOGRAPHY['FONT_FAMILY'],
                size=12,
                color="#707070"
            ),
            gridcolor="#3A3A3A",
            zerolinecolor="#4B4B4B"
        ),
        plot_bgcolor="#121212",
        paper_bgcolor="#121212",
        margin=dict(l=10, r=10, t=50, b=10),
        height=height,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#2E2E2E",
            font_size=12,
            font_family=TYPOGRAPHY['FONT_FAMILY'],
            font_color="#FFFFFF"
        ),
    )
    
    return fig

def add_insights_annotation(fig, text, x_pos, y_pos, width=200):
    """Add an insight annotation to a chart."""
    fig.add_annotation(
        x=x_pos,
        y=y_pos,
        xref="paper",
        yref="paper",
        text=text,
        showarrow=False,
        font=dict(
            family=TYPOGRAPHY['FONT_FAMILY'],
            size=11,
            color="#FFFFFF"
        ),
        align="left",
        bgcolor="#2E2E2E",
        bordercolor="#3A3A3A",
        borderwidth=1,
        borderpad=4,
        width=width
    )
    return fig

def metrics_row(metrics_data):
    """
    Display a row of metrics with enhanced styling.
    
    Args:
        metrics_data: List of dicts with keys 'label', 'value', 'delta', 'delta_color'
    """
    cols = st.columns(len(metrics_data))
    
    for i, metric in enumerate(metrics_data):
        with cols[i]:
            delta_color = "normal"
            if "delta_color" in metric:
                delta_color = metric["delta_color"]
                
            st.metric(
                label=metric["label"],
                value=metric["value"],
                delta=metric.get("delta", None),
                delta_color=delta_color
            )

def grid_layout(num_columns, elements, widths=None, heights=None, gap="1rem"):
    """Create a responsive grid layout with custom column widths."""
    if widths is None:
        widths = [f"{100/num_columns}%" for _ in range(num_columns)]
    
    # Create CSS grid
    st.markdown(f"""
    <style>
    .grid-container {{
        display: grid;
        grid-template-columns: {" ".join(widths)};
        {f"grid-template-rows: {' '.join(heights)};" if heights else ""}
        gap: {gap};
        padding: 0.5rem 0;
    }}
    .grid-item {{
        background-color: #1F1F1F;
        border-radius: 0.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        padding: 1rem;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Opening grid container
    st.markdown('<div class="grid-container">', unsafe_allow_html=True)
    
    # Add each element in a grid item
    for element_func in elements:
        st.markdown('<div class="grid-item">', unsafe_allow_html=True)
        element_func()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Closing grid container
    st.markdown('</div>', unsafe_allow_html=True)

def setup_ui():
    """Initialize the UI components and styles."""
    # Set page config
    st.set_page_config(
        page_title="BidPricing Analytics",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Render logo in sidebar
    with st.sidebar:
        render_logo()

def add_data_point_annotation(fig, x, y, text, direction="up", color=None):
    """Add an annotation to highlight a specific data point."""
    if color is None:
        color = COLOR_SYSTEM['ACCENT']['BLUE']
        
    arrow_settings = {
        "up": dict(ax=0, ay=-40),
        "down": dict(ax=0, ay=40),
        "left": dict(ax=-40, ay=0),
        "right": dict(ax=40, ay=0)
    }
    
    fig.add_annotation(
        x=x,
        y=y,
        text=text,
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowcolor=color,
        arrowwidth=2,
        font=dict(
            family=TYPOGRAPHY['FONT_FAMILY'],
            size=11,
            color=COLOR_SYSTEM['PRIMARY']['MAIN']
        ),
        bgcolor=COLOR_SYSTEM['BACKGROUND']['CARD'],
        bordercolor=COLOR_SYSTEM['NEUTRAL']['LIGHT'],
        borderwidth=1,
        borderpad=4,
        **arrow_settings[direction]
    )
    return fig
