"""
UI Components for the CPI Analysis & Prediction Dashboard.
This module provides reusable UI components and styling functions.
"""

import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Callable, Optional, Union, Any

# Import color system and typography from config
from config import COLOR_SYSTEM, TYPOGRAPHY

# UI Components

def load_custom_css():
    """Load custom CSS to override Streamlit defaults."""
    # Check if assets directory exists
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    css_path = os.path.join(assets_dir, "style.css")
    
    if os.path.exists(css_path):
        with open(css_path, 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        # If CSS file doesn't exist yet, use inline CSS
        st.markdown(f"""
        <style>
        /* Base Styles */
        html, body, [data-testid="stAppViewContainer"] {{
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
        }}

        /* Header Styling */
        .stApp header {{
            background-color: {COLOR_SYSTEM['PRIMARY']['MAIN']} !important;
            border-bottom: 1px solid {COLOR_SYSTEM['PRIMARY']['LIGHT']};
        }}

        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background-color: {COLOR_SYSTEM['BACKGROUND']['MAIN']};
            border-right: 1px solid {COLOR_SYSTEM['NEUTRAL']['LIGHT']};
        }}

        [data-testid="stSidebar"] .css-1d391kg {{
            padding-top: 2rem;
        }}

        /* Metric Styling */
        [data-testid="stMetric"] {{
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            border-radius: 0.5rem;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        [data-testid="stMetric"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }}

        [data-testid="stMetricLabel"] {{
            font-weight: 600 !important;
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']} !important;
        }}

        [data-testid="stMetricValue"] {{
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']} !important;
        }}

        [data-testid="stMetricDelta"] {{
            font-size: 0.9rem !important;
            font-weight: 500 !important;
        }}

        /* Button Styling */
        .stButton > button {{
            background-color: {COLOR_SYSTEM['ACCENT']['BLUE']};
            color: white;
            border: none;
            border-radius: 0.3rem;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: background-color 0.2s;
        }}

        .stButton > button:hover {{
            background-color: {COLOR_SYSTEM['SEMANTIC']['INFO']};
        }}

        /* Card-like containers */
        .css-card {{
            background-color: white;
            border-radius: 0.5rem;
            padding: 1rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }}

        /* DataFrames and tables */
        [data-testid="stTable"] {{
            border-radius: 0.5rem;
            overflow: hidden;
            border: 1px solid {COLOR_SYSTEM['NEUTRAL']['LIGHT']};
        }}

        .dataframe {{
            border-collapse: separate;
            border-spacing: 0;
        }}

        .dataframe th {{
            background-color: {COLOR_SYSTEM['BACKGROUND']['MAIN']};
            padding: 0.75rem !important;
            text-align: left !important;
            font-weight: 600 !important;
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']} !important;
            border-top: 1px solid {COLOR_SYSTEM['NEUTRAL']['LIGHT']} !important;
            border-bottom: 2px solid {COLOR_SYSTEM['NEUTRAL']['LIGHT']} !important;
        }}

        .dataframe td {{
            padding: 0.75rem !important;
            border-bottom: 1px solid {COLOR_SYSTEM['NEUTRAL']['LIGHT']} !important;
        }}

        .dataframe tr:nth-child(even) {{
            background-color: {COLOR_SYSTEM['BACKGROUND']['MAIN']};
        }}

        .dataframe tr:hover {{
            background-color: {COLOR_SYSTEM['NEUTRAL']['LIGHTER']};
        }}
        </style>
        """, unsafe_allow_html=True)

def render_logo():
    """Render the BidPricing Analytics logo."""
    logo_svg = f"""
    <svg width="200" height="48" viewBox="0 0 200 48" fill="none" xmlns="http://www.w3.org/2000/svg">
        <!-- Logo Background -->
        <rect width="48" height="48" rx="8" fill="{COLOR_SYSTEM['PRIMARY']['MAIN']}"/>
        
        <!-- Logo Graphic -->
        <path d="M14 24C14 18.4772 18.4772 14 24 14C29.5228 14 34 18.4772 34 24C34 29.5228 29.5228 34 24 34" stroke="{COLOR_SYSTEM['ACCENT']['BLUE']}" stroke-width="3" stroke-linecap="round"/>
        <path d="M24 34C18.4772 34 14 29.5228 14 24" stroke="{COLOR_SYSTEM['ACCENT']['ORANGE']}" stroke-width="3" stroke-linecap="round"/>
        
        <!-- Chart Bars -->
        <rect x="20" y="20" width="2" height="8" rx="1" fill="{COLOR_SYSTEM['PRIMARY']['CONTRAST']}"/>
        <rect x="24" y="18" width="2" height="10" rx="1" fill="{COLOR_SYSTEM['PRIMARY']['CONTRAST']}"/>
        <rect x="28" y="22" width="2" height="6" rx="1" fill="{COLOR_SYSTEM['PRIMARY']['CONTRAST']}"/>
        
        <!-- Company Name -->
        <text x="56" y="28" font-family="Inter, sans-serif" font-weight="700" font-size="16" fill="{COLOR_SYSTEM['PRIMARY']['MAIN']}">BidPricing</text>
        <text x="56" y="38" font-family="Inter, sans-serif" font-weight="400" font-size="12" fill="{COLOR_SYSTEM['NEUTRAL']['DARKER']}">Analytics</text>
    </svg>
    """
    
    return st.markdown(logo_svg, unsafe_allow_html=True)

def render_header(current_page="Dashboard"):
    """Render a branded header with navigation."""
    header_html = f"""
    <header style="
        background-color: white;
        padding: 1rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    ">
        <div style="display: flex; align-items: center;">
            <svg width="40" height="40" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                <!-- Logo Background -->
                <rect width="48" height="48" rx="8" fill="{COLOR_SYSTEM['PRIMARY']['MAIN']}"/>
                
                <!-- Logo Graphic -->
                <path d="M14 24C14 18.4772 18.4772 14 24 14C29.5228 14 34 18.4772 34 24C34 29.5228 29.5228 34 24 34" stroke="{COLOR_SYSTEM['ACCENT']['BLUE']}" stroke-width="3" stroke-linecap="round"/>
                <path d="M24 34C18.4772 34 14 29.5228 14 24" stroke="{COLOR_SYSTEM['ACCENT']['ORANGE']}" stroke-width="3" stroke-linecap="round"/>
                
                <!-- Chart Bars -->
                <rect x="20" y="20" width="2" height="8" rx="1" fill="{COLOR_SYSTEM['PRIMARY']['CONTRAST']}"/>
                <rect x="24" y="18" width="2" height="10" rx="1" fill="{COLOR_SYSTEM['PRIMARY']['CONTRAST']}"/>
                <rect x="28" y="22" width="2" height="6" rx="1" fill="{COLOR_SYSTEM['PRIMARY']['CONTRAST']}"/>
            </svg>
            
            <div style="margin-left: 1rem;">
                <h1 style="
                    margin: 0;
                    font-size: 1.5rem;
                    font-weight: 700;
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                ">BidPricing Analytics</h1>
                <p style="
                    margin: 0;
                    font-size: 0.9rem;
                    color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                ">CPI Analysis & Prediction Dashboard</p>
            </div>
        </div>
        
        <div>
            <p style="
                margin: 0;
                font-size: 1rem;
                font-weight: 500;
                color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
            ">
                <span style="color: {COLOR_SYSTEM['ACCENT']['BLUE']};">●</span> {current_page}
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
        background-color: white;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        border-top: 4px solid {accent_color};
        font-family: {TYPOGRAPHY['FONT_FAMILY']};
    ">
        <div style="
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
        ">
            {f'<div style="margin-right: 0.75rem;">{icon}</div>' if icon else ''}
            <h3 style="
                margin: 0;
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['HEADING']['H3']['size']};
                font-weight: {TYPOGRAPHY['HEADING']['H3']['weight']};
            ">{title}</h3>
        </div>
        <div>
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
                color=COLOR_SYSTEM['PRIMARY']['MAIN']
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
                color=COLOR_SYSTEM['PRIMARY']['MAIN']
            ),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor=COLOR_SYSTEM['NEUTRAL']['LIGHT'],
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
            color=COLOR_SYSTEM['PRIMARY']['MAIN']
        ),
        xaxis=dict(
            title_font=dict(
                family=TYPOGRAPHY['FONT_FAMILY'],
                size=14,
                color=COLOR_SYSTEM['PRIMARY']['MAIN']
            ),
            tickfont=dict(
                family=TYPOGRAPHY['FONT_FAMILY'],
                size=12,
                color=COLOR_SYSTEM['NEUTRAL']['DARKER']
            ),
            gridcolor=COLOR_SYSTEM['NEUTRAL']['LIGHT'],
            zerolinecolor=COLOR_SYSTEM['NEUTRAL']['MEDIUM']
        ),
        yaxis=dict(
            title_font=dict(
                family=TYPOGRAPHY['FONT_FAMILY'],
                size=14,
                color=COLOR_SYSTEM['PRIMARY']['MAIN']
            ),
            tickfont=dict(
                family=TYPOGRAPHY['FONT_FAMILY'],
                size=12,
                color=COLOR_SYSTEM['NEUTRAL']['DARKER']
            ),
            gridcolor=COLOR_SYSTEM['NEUTRAL']['LIGHT'],
            zerolinecolor=COLOR_SYSTEM['NEUTRAL']['MEDIUM']
        ),
        plot_bgcolor=COLOR_SYSTEM['NEUTRAL']['WHITE'],
        paper_bgcolor=COLOR_SYSTEM['NEUTRAL']['WHITE'],
        margin=dict(l=10, r=10, t=50, b=10),
        height=height,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor=COLOR_SYSTEM['NEUTRAL']['WHITE'],
            font_size=12,
            font_family=TYPOGRAPHY['FONT_FAMILY']
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
            color=COLOR_SYSTEM['PRIMARY']['MAIN']
        ),
        align="left",
        bgcolor=COLOR_SYSTEM['NEUTRAL']['LIGHTEST'],
        bordercolor=COLOR_SYSTEM['NEUTRAL']['LIGHT'],
        borderwidth=1,
        borderpad=4,
        width=width
    )
    return fig

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
        bgcolor=COLOR_SYSTEM['NEUTRAL']['WHITE'],
        bordercolor=color,
        borderwidth=1,
        borderpad=4,
        **arrow_settings[direction]
    )
    return fig

def metrics_row(metrics_data):
    """
    Display a row of metrics with enhanced styling.
    
    Args:
        metrics_data: List of dicts with keys 'label', 'value', 'delta', 'color'
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
    """Create a responsive grid layout with custom column widths.
    
    Args:
        num_columns: Number of columns in the grid
        elements: List of functions that render Streamlit elements
        widths: List of custom widths for each column (CSS values)
        heights: List of custom heights for each row (CSS values)
        gap: Gap between grid items (CSS value)
    """
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
        background-color: white;
        border-radius: 0.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
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

def render_icon_tabs(tabs_config):
    """
    Render custom tabs with icons.
    
    Args:
        tabs_config: List of dicts with keys 'icon', 'label', and 'content_func'
    """
    # Create tab headers
    tabs_html = "<div class='icon-tabs'><div class='tab-header'>"
    
    for i, tab in enumerate(tabs_config):
        active_class = "active" if i == 0 else ""
        tabs_html += f"""
        <button class="tab-button {active_class}" onclick="switchTab({i})">
            <span class="tab-icon">{tab['icon']}</span>
            <span class="tab-label">{tab['label']}</span>
        </button>
        """
    
    tabs_html += "</div><div class='tab-content'>"
    
    # Create tab content containers
    for i, tab in enumerate(tabs_config):
        display_style = "block" if i == 0 else "none"
        tabs_html += f'<div id="tab-{i}" class="tab-pane" style="display: {display_style};">'
        tabs_html += f'<div id="tab-content-{i}"></div>'
        tabs_html += '</div>'
    
    tabs_html += "</div></div>"
    
    # Add JS for tab switching
    tabs_js = """
    <script>
    function switchTab(tabIndex) {
        // Hide all tabs
        const tabPanes = document.querySelectorAll('.tab-pane');
        tabPanes.forEach(pane => pane.style.display = 'none');
        
        // Remove active class from all buttons
        const tabButtons = document.querySelectorAll('.tab-button');
        tabButtons.forEach(button => button.classList.remove('active'));
        
        // Show selected tab
        document.getElementById('tab-' + tabIndex).style.display = 'block';
        
        // Add active class to selected button
        tabButtons[tabIndex].classList.add('active');
    }
    </script>
    """
    
    # Add CSS for tabs
    tabs_css = """
    <style>
    .icon-tabs {
        margin: 1.5rem 0;
    }
    
    .tab-header {
        display: flex;
        border-bottom: 1px solid #DEE2E6;
        margin-bottom: 1rem;
    }
    
    .tab-button {
        display: flex;
        align-items: center;
        padding: 0.75rem 1.25rem;
        background: none;
        border: none;
        border-bottom: 3px solid transparent;
        cursor: pointer;
        transition: all 0.2s;
        color: #6C757D;
        font-weight: 500;
    }
    
    .tab-button:hover {
        color: #3498DB;
        background-color: #F8F9FA;
    }
    
    .tab-button.active {
        color: #3498DB;
        border-bottom: 3px solid #3498DB;
    }
    
    .tab-icon {
        margin-right: 0.5rem;
        font-size: 1.2rem;
    }
    
    .tab-content {
        padding: 0.5rem;
    }
    </style>
    """
    
    # Render the tabs structure
    st.markdown(tabs_css + tabs_html + tabs_js, unsafe_allow_html=True)
    
    # Render content in each tab (hidden initially)
    for i, tab in enumerate(tabs_config):
        with st.container():
            tab_content = tab['content_func']()
            st.markdown(f"""
            <script>
                document.getElementById('tab-content-{i}').innerHTML = `{tab_content}`;
            </script>
            """, unsafe_allow_html=True)

def add_tooltip(element_id, tooltip_text):
    """Add an interactive tooltip to an element."""
    tooltip_js = f"""
    <script>
        // Create tooltip element
        const tooltip = document.createElement('div');
        tooltip.className = 'custom-tooltip';
        tooltip.innerHTML = '{tooltip_text}';
        document.body.appendChild(tooltip);
        
        // Add event listeners to target element
        const element = document.getElementById('{element_id}');
        if (element) {{
            element.addEventListener('mouseover', (e) => {{
                tooltip.style.display = 'block';
                tooltip.style.left = e.pageX + 10 + 'px';
                tooltip.style.top = e.pageY + 10 + 'px';
            }});
            
            element.addEventListener('mousemove', (e) => {{
                tooltip.style.left = e.pageX + 10 + 'px';
                tooltip.style.top = e.pageY + 10 + 'px';
            }});
            
            element.addEventListener('mouseout', () => {{
                tooltip.style.display = 'none';
            }});
        }}
    </script>
    
    <style>
        .custom-tooltip {{
            display: none;
            position: absolute;
            background-color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            color: white;
            padding: 0.5rem 0.75rem;
            border-radius: 0.3rem;
            font-size: 0.8rem;
            max-width: 250px;
            z-index: 1000;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }}
    </style>
    """
    
    return st.markdown(tooltip_js, unsafe_allow_html=True)

def initialize_ui():
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
    
    # Apply broader app styling
    st.markdown(
        f"""
        <style>
            /* Page background */
            .stApp {{
                background-color: {COLOR_SYSTEM['BACKGROUND']['MAIN']};
            }}
            
            /* Headings */
            h1, h2, h3, h4, h5, h6 {{
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            }}
            
            /* Body text */
            p, li, div {{
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            }}
            
            /* Cards */
            div[data-testid="stVerticalBlock"] > div {{
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                border-radius: 0.5rem;
                padding: 0.5rem;
                margin: 0 0 1rem 0;
            }}
            
            /* Metric cards */
            div[data-testid="metric-container"] {{
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                border-radius: 0.5rem;
                padding: 1rem;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Main function to use in app entry point
def setup_ui():
    """Setup UI components for the app."""
    initialize_ui()
    render_header()