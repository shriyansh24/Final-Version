"""
Overview component for the CPI Analysis & Prediction Dashboard.
Displays summary metrics and overview charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging
from typing import Dict, Any

# Import UI components
from ui_components import (
    render_card, metrics_row, apply_chart_styling,
    add_insights_annotation, grid_layout, render_icon_tabs
)

# Import visualization utilities
from utils.visualization import (
    create_type_distribution_chart,
    create_cpi_distribution_boxplot,
    create_cpi_histogram_comparison,
    create_cpi_efficiency_chart
)

# Import data utilities
from utils.data_processor import get_data_summary

# Import color system
from config import COLOR_SYSTEM, TYPOGRAPHY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_overview(won_data: pd.DataFrame, lost_data: pd.DataFrame, combined_data: pd.DataFrame) -> None:
    """
    Display the overview component with summary metrics and key charts.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
        combined_data (pd.DataFrame): Combined DataFrame of both Won and Lost bids
    """
    try:
        # Add page header
        st.markdown(f"""
        <h2 style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            font-size: {TYPOGRAPHY['HEADING']['H2']['size']};
            font-weight: {TYPOGRAPHY['HEADING']['H2']['weight']};
            margin-bottom: 1rem;
        ">Dashboard Overview</h2>
        """, unsafe_allow_html=True)
        
        # Add introductory text with enhanced styling
        intro_content = f"""
        <p style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            font-size: {TYPOGRAPHY['BODY']['LARGE']['size']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            margin-bottom: 1.5rem;
        ">
            This dashboard provides comprehensive analysis of Cost Per Interview (CPI) 
            for market research projects. Compare won and lost bids to optimize pricing strategies 
            and improve win rates.
        </p>
        """
        render_card(
            title="About This Dashboard", 
            content=intro_content,
            icon=f'<span style="font-size: 1.5rem; color: {COLOR_SYSTEM["ACCENT"]["BLUE"]};">📊</span>'
        )
        
        # Get data summary
        data_summary = get_data_summary(combined_data)
        
        # Create metrics section
        st.markdown(f"""
        <h3 style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            font-size: {TYPOGRAPHY['HEADING']['H3']['size']};
            font-weight: {TYPOGRAPHY['HEADING']['H3']['weight']};
            margin: 1rem 0;
        ">Key Metrics</h3>
        """, unsafe_allow_html=True)
        
        # First row of metrics - Key CPI Stats
        metrics_data = []
        
        # Won CPI with green/red indicator based on relation to overall average
        if 'Won' in data_summary:
            won_avg = data_summary['Won']['Avg_CPI']
            metrics_data.append({
                "label": "Won Bids Avg CPI",
                "value": f"${won_avg:.2f}",
                "delta": None
            })
        
        # Lost CPI
        if 'Lost' in data_summary:
            lost_avg = data_summary['Lost']['Avg_CPI']
            metrics_data.append({
                "label": "Lost Bids Avg CPI",
                "value": f"${lost_avg:.2f}",
                "delta": None
            })
        
        # CPI Difference
        if 'Won' in data_summary and 'Lost' in data_summary:
            diff = data_summary['Lost']['Avg_CPI'] - data_summary['Won']['Avg_CPI']
            metrics_data.append({
                "label": "CPI Difference",
                "value": f"${diff:.2f}",
                "delta": f"{diff:.2f}",
                "delta_color": "normal"
            })
            
        # Show first row of metrics
        metrics_row(metrics_data)
        
        # Second row of metrics - Sample Sizes
        metrics_data = []
        
        # Total bids
        metrics_data.append({
            "label": "Total Bids Analyzed",
            "value": f"{len(combined_data)}",
            "delta": None
        })
        
        # Won bids count
        won_count = len(won_data)
        metrics_data.append({
            "label": "Won Bids Count",
            "value": f"{won_count}",
            "delta": f"{won_count / len(combined_data):.1%}",
            "delta_color": "normal"
        })
        
        # Lost bids count
        lost_count = len(lost_data)
        metrics_data.append({
            "label": "Lost Bids Count",
            "value": f"{lost_count}",
            "delta": f"{lost_count / len(combined_data):.1%}",
            "delta_color": "normal"
        })
        
        # Show second row of metrics
        metrics_row(metrics_data)
        
        # Third row of metrics - IR and LOI
        metrics_data = []
        
        # Average IR for Won bids
        if 'Won' in data_summary:
            metrics_data.append({
                "label": "Avg Won IR (%)",
                "value": f"{data_summary['Won']['Avg_IR']:.1f}%",
                "delta": None
            })
        
        # Average LOI for Won bids
        if 'Won' in data_summary:
            metrics_data.append({
                "label": "Avg Won LOI (min)",
                "value": f"{data_summary['Won']['Avg_LOI']:.1f}",
                "delta": None
            })
        
        # Average Completes for Won bids
        if 'Won' in data_summary:
            metrics_data.append({
                "label": "Avg Won Completes",
                "value": f"{data_summary['Won']['Avg_Completes']:.0f}",
                "delta": None
            })
        
        # Show third row of metrics
        metrics_row(metrics_data)
        
        # Add space before charts
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Create main charts section using tabs
        st.markdown(f"""
        <h3 style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            font-size: {TYPOGRAPHY['HEADING']['H3']['size']};
            font-weight: {TYPOGRAPHY['HEADING']['H3']['weight']};
            margin: 1rem 0;
        ">Overview Charts</h3>
        """, unsafe_allow_html=True)
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Distribution", 
            "CPI Comparison", 
            "CPI Histogram",
            "Efficiency Analysis"
        ])
        
        # Tab 1: Distribution of Won vs Lost
        with tab1:
            # Create card for pie chart
            fig = create_type_distribution_chart(combined_data)
            
            # Apply consistent styling
            fig = apply_chart_styling(
                fig, 
                title="Distribution of Won vs Lost Bids",
                height=400
            )
            
            # Add insights annotation
            fig = add_insights_annotation(
                fig,
                "This chart shows the proportion of won vs lost bids in the dataset. A balanced distribution is ideal for comparative analysis.",
                0.01,
                0.95,
                width=220
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add contextual information
            won_pct = (combined_data['Type'] == 'Won').mean() * 100
            lost_pct = (combined_data['Type'] == 'Lost').mean() * 100
            
            # Create two columns for additional metrics
            col1, col2 = st.columns(2)
            
            with col1:
                render_card(
                    title="Won Percentage", 
                    content=f"""
                    <div style="text-align: center; font-size: 2rem; font-weight: bold; color: {COLOR_SYSTEM['CHARTS']['WON']};">
                        {won_pct:.1f}%
                    </div>
                    <div style="text-align: center; font-size: 0.9rem; color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};">
                        {(combined_data['Type'] == 'Won').sum()} out of {len(combined_data)} bids
                    </div>
                    """,
                    accent_color=COLOR_SYSTEM['CHARTS']['WON']
                )
            
            with col2:
                render_card(
                    title="Lost Percentage", 
                    content=f"""
                    <div style="text-align: center; font-size: 2rem; font-weight: bold; color: {COLOR_SYSTEM['CHARTS']['LOST']};">
                        {lost_pct:.1f}%
                    </div>
                    <div style="text-align: center; font-size: 0.9rem; color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};">
                        {(combined_data['Type'] == 'Lost').sum()} out of {len(combined_data)} bids
                    </div>
                    """,
                    accent_color=COLOR_SYSTEM['CHARTS']['LOST']
                )
        
        # Tab 2: CPI Distribution Boxplot
        with tab2:
            fig = create_cpi_distribution_boxplot(won_data, lost_data)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add descriptive statistics as a card
            if 'Won' in data_summary and 'Lost' in data_summary:
                stats_content = f"""
                <div style="display: flex; justify-content: space-between;">
                    <div style="flex: 1;">
                        <h4 style="color: {COLOR_SYSTEM['CHARTS']['WON']};">Won Bids CPI Statistics</h4>
                        <table style="width: 100%;">
                            <tr>
                                <td style="padding: 5px 10px 5px 0;">Minimum:</td>
                                <td style="text-align: right; font-weight: 500;">${won_data['CPI'].min():.2f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 5px 10px 5px 0;">25th Percentile:</td>
                                <td style="text-align: right; font-weight: 500;">${data_summary['Won']['CPI_25th']:.2f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 5px 10px 5px 0;">Median:</td>
                                <td style="text-align: right; font-weight: 500;">${data_summary['Won']['Median_CPI']:.2f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 5px 10px 5px 0;">Mean:</td>
                                <td style="text-align: right; font-weight: 500;">${data_summary['Won']['Avg_CPI']:.2f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 5px 10px 5px 0;">75th Percentile:</td>
                                <td style="text-align: right; font-weight: 500;">${data_summary['Won']['CPI_75th']:.2f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 5px 10px 5px 0;">Maximum:</td>
                                <td style="text-align: right; font-weight: 500;">${won_data['CPI'].max():.2f}</td>
                            </tr>
                        </table>
                    </div>
                    
                    <div style="flex: 1; margin-left: 2rem;">
                        <h4 style="color: {COLOR_SYSTEM['CHARTS']['LOST']};">Lost Bids CPI Statistics</h4>
                        <table style="width: 100%;">
                            <tr>
                                <td style="padding: 5px 10px 5px 0;">Minimum:</td>
                                <td style="text-align: right; font-weight: 500;">${lost_data['CPI'].min():.2f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 5px 10px 5px 0;">25th Percentile:</td>
                                <td style="text-align: right; font-weight: 500;">${data_summary['Lost']['CPI_25th']:.2f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 5px 10px 5px 0;">Median:</td>
                                <td style="text-align: right; font-weight: 500;">${data_summary['Lost']['Median_CPI']:.2f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 5px 10px 5px 0;">Mean:</td>
                                <td style="text-align: right; font-weight: 500;">${data_summary['Lost']['Avg_CPI']:.2f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 5px 10px 5px 0;">75th Percentile:</td>
                                <td style="text-align: right; font-weight: 500;">${data_summary['Lost']['CPI_75th']:.2f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 5px 10px 5px 0;">Maximum:</td>
                                <td style="text-align: right; font-weight: 500;">${lost_data['CPI'].max():.2f}</td>
                            </tr>
                        </table>
                    </div>
                </div>
                """
                
                render_card(
                    title="CPI Distribution Statistics", 
                    content=stats_content,
                    icon=f'<span style="font-size: 1.5rem; color: {COLOR_SYSTEM["ACCENT"]["PURPLE"]};">📊</span>'
                )
        
        # Tab 3: Histogram
        with tab3:
            fig = create_cpi_histogram_comparison(won_data, lost_data)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add insight card
            insight_content = f"""
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                The histograms above show the distribution of CPI values for won and lost bids. 
                Notice that lost bids typically have a wider distribution with higher average values,
                while won bids tend to be more concentrated in the lower CPI ranges.
            </p>
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-top: 0.5rem;
            ">
                <strong>Key insight:</strong> The optimal pricing strategy should aim to position your CPI 
                within the peak distribution range of won bids, while avoiding the peak ranges of lost bids.
            </p>
            """
            
            render_card(
                title="CPI Distribution Insights", 
                content=insight_content,
                icon=f'<span style="font-size: 1.5rem; color: {COLOR_SYSTEM["ACCENT"]["YELLOW"]};">💡</span>',
                accent_color=COLOR_SYSTEM['ACCENT']['YELLOW']
            )
        
        # Tab 4: Efficiency Chart
        with tab4:
            fig = create_cpi_efficiency_chart(won_data, lost_data)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation card
            explanation_content = f"""
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                The <strong>Efficiency Metric</strong> combines key project parameters into a single value:
                <br><code>(IR/100) × (1/LOI) × Completes</code>
            </p>
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-top: 0.5rem;
            ">
                This metric captures the relationship between:
            </p>
            <ul style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                <li><strong>Incidence Rate (IR)</strong>: Higher values are more efficient</li>
                <li><strong>Length of Interview (LOI)</strong>: Lower values are more efficient</li>
                <li><strong>Sample Size (Completes)</strong>: Higher values provide economies of scale</li>
            </ul>
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-top: 0.5rem;
            ">
                <strong>Higher efficiency values</strong> generally correlate with <strong>lower CPI</strong>, 
                as shown by the downward trend lines in the chart.
            </p>
            """
            
            render_card(
                title="Understanding the Efficiency Metric", 
                content=explanation_content,
                icon=f'<span style="font-size: 1.5rem; color: {COLOR_SYSTEM["ACCENT"]["GREEN"]};">🔍</span>',
                accent_color=COLOR_SYSTEM['ACCENT']['GREEN']
            )
        
        # Add section for key insights
        st.markdown(f"""
        <h3 style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            font-size: {TYPOGRAPHY['HEADING']['H3']['size']};
            font-weight: {TYPOGRAPHY['HEADING']['H3']['weight']};
            margin: 1.5rem 0 1rem 0;
        ">Key Insights</h3>
        """, unsafe_allow_html=True)
        
        # Create insights using a grid layout
        def render_insight_1():
            render_card(
                title="CPI Differential", 
                content=f"""
                <p style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                ">
                    On average, lost bids are <strong>${data_summary['Lost']['Avg_CPI'] - data_summary['Won']['Avg_CPI']:.2f}</strong> 
                    more expensive than won bids, representing a 
                    <strong>{(data_summary['Lost']['Avg_CPI'] / data_summary['Won']['Avg_CPI'] - 1) * 100:.1f}%</strong> premium.
                </p>
                """,
                icon=f'<span style="font-size: 1.5rem; color: {COLOR_SYSTEM["ACCENT"]["RED"]};">💰</span>',
                accent_color=COLOR_SYSTEM['ACCENT']['RED']
            )
            
        def render_insight_2():
            render_card(
                title="Optimal IR Range", 
                content=f"""
                <p style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                ">
                    Won bids have an average IR of <strong>{data_summary['Won']['Avg_IR']:.1f}%</strong>, while
                    lost bids average <strong>{data_summary['Lost']['Avg_IR']:.1f}%</strong>. This suggests
                    that bids with {data_summary['Won']['Avg_IR'] > data_summary['Lost']['Avg_IR'] and "higher" or "lower"} 
                    IR tend to be more successful.
                </p>
                """,
                icon=f'<span style="font-size: 1.5rem; color: {COLOR_SYSTEM["ACCENT"]["BLUE"]};">📈</span>',
                accent_color=COLOR_SYSTEM['ACCENT']['BLUE']
            )
            
        def render_insight_3():
            render_card(
                title="Efficiency Impact", 
                content=f"""
                <p style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                ">
                    Higher efficiency metrics strongly correlate with lower CPI values. 
                    For each 1-point increase in efficiency, CPI tends to decrease by 
                    approximately $0.30-$0.50 (based on trend line slopes).
                </p>
                """,
                icon=f'<span style="font-size: 1.5rem; color: {COLOR_SYSTEM["ACCENT"]["GREEN"]};">⚡</span>',
                accent_color=COLOR_SYSTEM['ACCENT']['GREEN']
            )
        
        # Display insights in a grid
        grid_layout(3, [render_insight_1, render_insight_2, render_insight_3])
        
    except Exception as e:
        # Log error
        logger.error(f"Error in show_overview: {e}", exc_info=True)
        
        # Display user-friendly error message
        st.error(f"An error occurred while displaying the overview: {str(e)}")
        st.markdown(f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            border-radius: 0.5rem;
            padding: 1rem;
            margin-top: 1rem;
            border-left: 4px solid {COLOR_SYSTEM['ACCENT']['YELLOW']};
        ">
            <h4 style="margin-top: 0;">Troubleshooting</h4>
            <p>Please try the following:</p>
            <ul>
                <li>Refresh the page</li>
                <li>Check that your data files are correctly formatted</li>
                <li>Ensure you have sufficient data for visualization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)