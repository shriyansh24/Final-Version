"""
Advanced CPI Analysis component for the CPI Analysis & Prediction Dashboard.
Handles advanced analyses (LOI, sample size, and multi-factor).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging
from typing import Dict, List, Tuple, Any, Optional

# Import visualization utilities
from utils.visualization import (
    create_bar_chart_by_bin,
    create_heatmap
)

# Import constants
from utils.visualization import (
    HEATMAP_COLORSCALE_WON,
    HEATMAP_COLORSCALE_LOST
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_loi_analysis(won_data: pd.DataFrame, lost_data: pd.DataFrame, tab) -> None:
    """
    Display the Length of Interview (LOI) analysis in the given tab.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
        tab: Streamlit tab object
    """
    with tab:
        st.subheader("CPI Analysis by Length of Interview (LOI)")
        
        # Add description
        st.markdown("""
        This analysis explores how Length of Interview (LOI) - the duration of the survey in minutes -
        affects CPI. Longer surveys typically command higher prices to compensate respondents for
        their additional time.
        """)
        
        # Create CPI vs LOI scatter plot
        fig = go.Figure()
        
        # Add Won data
        fig.add_trace(go.Scatter(
            x=won_data['LOI'], 
            y=won_data['CPI'], 
            mode='markers',
            marker=dict(color='#3288bd', size=8, opacity=0.6),
            name="Won",
            hovertemplate='<b>Won Bid</b><br>LOI: %{x:.1f} min<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<extra></extra>',
            customdata=won_data[['IR']]
        ))
        
        # Add Lost data
        fig.add_trace(go.Scatter(
            x=lost_data['LOI'], 
            y=lost_data['CPI'], 
            mode='markers',
            marker=dict(color='#f58518', size=8, opacity=0.6),
            name="Lost",
            hovertemplate='<b>Lost Bid</b><br>LOI: %{x:.1f} min<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<extra></extra>',
            customdata=lost_data[['IR']]
        ))
        
        # Add trend lines
        # Won trend line
        won_trend_x = np.linspace(won_data['LOI'].min(), won_data['LOI'].max(), 100)
        won_coeffs = np.polyfit(won_data['LOI'], won_data['CPI'], 1)
        won_trend_y = np.polyval(won_coeffs, won_trend_x)
        
        fig.add_trace(go.Scatter(
            x=won_trend_x,
            y=won_trend_y,
            mode='lines',
            line=dict(color='#3288bd', width=2),
            name='Won Trend',
            hoverinfo='skip'
        ))
        
        # Lost trend line
        lost_trend_x = np.linspace(lost_data['LOI'].min(), lost_data['LOI'].max(), 100)
        lost_coeffs = np.polyfit(lost_data['LOI'], lost_data['CPI'], 1)
        lost_trend_y = np.polyval(lost_coeffs, lost_trend_x)
        
        fig.add_trace(go.Scatter(
            x=lost_trend_x,
            y=lost_trend_y,
            mode='lines',
            line=dict(color='#f58518', width=2),
            name='Lost Trend',
            hoverinfo='skip'
        ))
        
        # Update layout
        fig.update_layout(
            title_text="CPI vs Length of Interview (LOI) Relationship",
            height=500,
            xaxis_title="Length of Interview (minutes)",
            yaxis_title="CPI ($)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
            yaxis=dict(gridcolor='rgba(0,0,0,0.1)', tickprefix='$')
        )
        
        st.plotly_chart(fig, use_container_width=True, key='analysis_cpi_vs_loi_scatter')
        
        # Create and display CPI by LOI Bin bar chart
        st.subheader("Average CPI by LOI Bin")
        fig = create_bar_chart_by_bin(won_data, lost_data, 'LOI_Bin', 'CPI',
                                    title='Average CPI by Length of Interview Bin')
        st.plotly_chart(fig, use_container_width=True, key='analysis_cpi_by_loi_bin')
        
        # Add interpretation
        with st.expander("📊 Interpretation"):
            st.markdown("""
            ### Understanding the LOI-CPI Relationship
            
            Length of Interview (LOI) is the duration of the survey in minutes. It directly affects
            respondent compensation and therefore influences the overall CPI.
            
            ### Key Insights
            
            1. **Positive Correlation**: Both charts show a clear positive correlation between LOI and CPI.
               As surveys get longer, prices increase to compensate respondents for their time.
            
            2. **Pricing Slope**: The trend lines in the scatter plot show how CPI typically increases with
               each additional minute of survey length. This can be used as a guideline for pricing adjustment.
            
            3. **Won vs. Lost Comparison**: The gap between won and lost bids widens as LOI increases,
               suggesting that competitive pricing becomes even more critical for longer surveys.
            
            4. **Pricing Thresholds**: The bar chart reveals clear pricing thresholds for different LOI bins,
               which can serve as benchmarks when pricing new projects.
            """)


def show_sample_size_analysis(won_data: pd.DataFrame, lost_data: pd.DataFrame, tab) -> None:
    """
    Display the Sample Size analysis in the given tab.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
        tab: Streamlit tab object
    """
    with tab:
        st.subheader("CPI Analysis by Sample Size (Completes)")
        
        # Add description
        st.markdown("""
        This analysis explores how Sample Size (the number of completed interviews) affects CPI.
        Larger samples typically benefit from volume discounts, resulting in lower per-unit costs.
        """)
        
        # Create CPI vs Completes scatter plot
        fig = go.Figure()
        
        # Add Won data
        fig.add_trace(go.Scatter(
            x=won_data['Completes'], 
            y=won_data['CPI'], 
            mode='markers',
            marker=dict(color='#3288bd', size=8, opacity=0.6),
            name="Won",
            hovertemplate='<b>Won Bid</b><br>Completes: %{x}<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<br>LOI: %{customdata[1]:.1f} min<extra></extra>',
            customdata=won_data[['IR', 'LOI']]
        ))
        
        # Add Lost data
        fig.add_trace(go.Scatter(
            x=lost_data['Completes'], 
            y=lost_data['CPI'], 
            mode='markers',
            marker=dict(color='#f58518', size=8, opacity=0.6),
            name="Lost",
            hovertemplate='<b>Lost Bid</b><br>Completes: %{x}<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<br>LOI: %{customdata[1]:.1f} min<extra></extra>',
            customdata=lost_data[['IR', 'LOI']]
        ))
        
        # Add trend lines using logarithmic fit (better for sample size relationships)
        # Won trend
        won_x = won_data['Completes']
        won_y = won_data['CPI']
        won_log_x = np.log(won_x)
        won_coeffs = np.polyfit(won_log_x, won_y, 1)
        
        won_trend_x = np.linspace(won_x.min(), won_x.max(), 100)
        won_trend_y = won_coeffs[0] * np.log(won_trend_x) + won_coeffs[1]
        
        fig.add_trace(go.Scatter(
            x=won_trend_x,
            y=won_trend_y,
            mode='lines',
            line=dict(color='#3288bd', width=2),
            name='Won Trend',
            hoverinfo='skip'
        ))
        
        # Lost trend
        lost_x = lost_data['Completes']
        lost_y = lost_data['CPI']
        lost_log_x = np.log(lost_x)
        lost_coeffs = np.polyfit(lost_log_x, lost_y, 1)
        
        lost_trend_x = np.linspace(lost_x.min(), lost_x.max(), 100)
        lost_trend_y = lost_coeffs[0] * np.log(lost_trend_x) + lost_coeffs[1]
        
        fig.add_trace(go.Scatter(
            x=lost_trend_x,
            y=lost_trend_y,
            mode='lines',
            line=dict(color='#f58518', width=2),
            name='Lost Trend',
            hoverinfo='skip'
        ))
        
        # Update layout
        fig.update_layout(
            title_text="CPI vs Sample Size (Completes) Relationship",
            height=500,
            xaxis_title="Sample Size (Number of Completes)",
            yaxis_title="CPI ($)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            xaxis=dict(gridcolor='rgba(0,0,0,0.1)', type='log'),  # Log scale for better visualization
            yaxis=dict(gridcolor='rgba(0,0,0,0.1)', tickprefix='$')
        )
        
        st.plotly_chart(fig, use_container_width=True, key='analysis_cpi_vs_sample_scatter')
        
        # Create and display CPI by Completes Bin bar chart
        st.subheader("Average CPI by Sample Size Bin")
        fig = create_bar_chart_by_bin(won_data, lost_data, 'Completes_Bin', 'CPI',
                                    title='Average CPI by Sample Size Bin')
        st.plotly_chart(fig, use_container_width=True, key='analysis_cpi_by_sample_bin')
        
        # Add interpretation
        with st.expander("📊 Interpretation"):
            st.markdown("""
            ### Understanding the Sample Size-CPI Relationship
            
            Sample size (number of completes) affects CPI through economies of scale - larger projects
            typically have lower per-unit costs due to volume efficiencies.
            
            ### Key Insights
            
            1. **Negative Correlation**: Both charts show that as sample size increases, CPI tends to decrease,
               following a logarithmic curve (diminishing returns). This reflects standard volume discount practices.
            
            2. **Volume Discount Thresholds**: The bar chart reveals clear pricing thresholds at different
               sample size bins, providing guidance on appropriate volume discount levels.
            
            3. **Won vs. Lost Comparison**: The gap between won and lost bids changes across sample size bins,
               suggesting different pricing sensitivities at different volumes.
            
            4. **Large Sample Competitiveness**: The competitive gap appears larger for very large samples,
               indicating that pricing competitiveness may be especially important for high-volume projects.
            """)


def show_multi_factor_analysis(won_data: pd.DataFrame, lost_data: pd.DataFrame, tab) -> None:
    """
    Display the Multi-Factor Analysis in the given tab.
    """
    with tab:
        st.subheader("Multi-Factor Analysis")

        st.markdown("""
        This analysis examines how CPI is influenced by multiple factors simultaneously,
        particularly focusing on the combined effect of Incidence Rate (IR) and Length of Interview (LOI).
        """)

        # Early exit if not enough data
        if len(won_data) < 5 or len(lost_data) < 5:
            st.warning("Not enough data for multi-factor analysis. Need at least 5 data points in each category.")
            return

        # --- Heatmaps ---
        st.subheader("IR and LOI Combined Influence on CPI")

        try:
            col1, col2 = st.columns(2)

            with col1:
                if 'IR_Bin' in won_data.columns and 'LOI_Bin' in won_data.columns:
                    won_grouped = won_data.groupby(['IR_Bin', 'LOI_Bin'])['CPI'].agg(['mean', 'count']).reset_index()
                    won_grouped = won_grouped[won_grouped['count'] >= 2]
                    if len(won_grouped) >= 4:
                        won_pivot = won_grouped.pivot(index='IR_Bin', columns='LOI_Bin', values='mean')
                        fig = create_heatmap(won_pivot, "Won Deals: Average CPI by IR and LOI", HEATMAP_COLORSCALE_WON)
                        st.plotly_chart(fig, use_container_width=True, key='analysis_heatmap_won')
                    else:
                        st.info("Not enough Won data combinations to create a meaningful heatmap.")
                else:
                    st.info("Missing required binned columns in Won data.")

            with col2:
                if 'IR_Bin' in lost_data.columns and 'LOI_Bin' in lost_data.columns:
                    lost_grouped = lost_data.groupby(['IR_Bin', 'LOI_Bin'])['CPI'].agg(['mean', 'count']).reset_index()
                    lost_grouped = lost_grouped[lost_grouped['count'] >= 2]
                    if len(lost_grouped) >= 4:
                        lost_pivot = lost_grouped.pivot(index='IR_Bin', columns='LOI_Bin', values='mean')
                        fig = create_heatmap(lost_pivot, "Lost Deals: Average CPI by IR and LOI", HEATMAP_COLORSCALE_LOST)
                        st.plotly_chart(fig, use_container_width=True, key='analysis_heatmap_lost')
                    else:
                        st.info("Not enough Lost data combinations to create a meaningful heatmap.")
                else:
                    st.info("Missing required binned columns in Lost data.")
        except Exception as e:
            logger.error(f"Error creating individual heatmaps: {e}", exc_info=True)
            st.error(f"Could not generate heatmaps: {str(e)}")

        # --- Differential Heatmap ---
        st.subheader("CPI Differential: Lost vs. Won")

        try:
            won_data_marked = won_data.copy()
            lost_data_marked = lost_data.copy()
            won_data_marked['Type'] = 'Won'
            lost_data_marked['Type'] = 'Lost'
            combined_data = pd.concat([won_data_marked, lost_data_marked])

            grouped = combined_data.groupby(['IR_Bin', 'LOI_Bin', 'Type'])['CPI'].agg(['mean', 'count']).reset_index()
            grouped = grouped[grouped['count'] >= 2]

            pivot_df = grouped.pivot_table(index='IR_Bin', columns=['LOI_Bin', 'Type'], values='mean').reset_index()
            if pivot_df.shape[0] < 3 or pivot_df.shape[1] < 3:
                st.info("Not enough data for differential visualization.")
            else:
                diff_data = []
                for ir_bin in grouped['IR_Bin'].unique():
                    for loi_bin in grouped['LOI_Bin'].unique():
                        won_val = grouped[(grouped['IR_Bin'] == ir_bin) & (grouped['LOI_Bin'] == loi_bin) & (grouped['Type'] == 'Won')]['mean'].values
                        lost_val = grouped[(grouped['IR_Bin'] == ir_bin) & (grouped['LOI_Bin'] == loi_bin) & (grouped['Type'] == 'Lost')]['mean'].values
                        if len(won_val) > 0 and len(lost_val) > 0:
                            diff_data.append({'IR_Bin': ir_bin, 'LOI_Bin': loi_bin, 'Difference': lost_val[0] - won_val[0]})

                if diff_data:
                    diff_df = pd.DataFrame(diff_data)
                    diff_pivot = diff_df.pivot(index='IR_Bin', columns='LOI_Bin', values='Difference')
                    fig = px.imshow(
                        diff_pivot,
                        labels=dict(x="LOI Bin", y="IR Bin", color="CPI Difference ($)"),
                        x=diff_pivot.columns,
                        y=diff_pivot.index,
                        title="CPI Differential: Lost Minus Won",
                        color_continuous_scale="RdBu_r",
                        aspect="auto",
                        text_auto='.2f'
                    )
                    fig.update_layout(
                        height=600,
                        coloraxis_colorbar=dict(title="CPI Diff ($)", tickprefix="$", len=0.75),
                        plot_bgcolor='rgba(255,255,255,1)',
                        paper_bgcolor='rgba(255,255,255,1)',
                        font=dict(family="Arial, sans-serif", size=12, color="black")
                    )
                    fig.update_xaxes(tickangle=45, title_font=dict(size=14), title_standoff=25)
                    fig.update_yaxes(title_font=dict(size=14), title_standoff=25)
                    st.plotly_chart(fig, use_container_width=True, key='analysis_heatmap_differential')
                else:
                    st.info("Not enough matching combinations to create a differential heatmap.")
        except Exception as e:
            logger.error(f"Error creating differential heatmap: {e}", exc_info=True)
            st.error(f"Could not generate differential visualization: {str(e)}")

        # --- 3D Plot ---
        st.subheader("Combined Factor Impact on Total Project Cost")

        try:
            won_data_copy = won_data.copy()
            lost_data_copy = lost_data.copy()
            won_data_copy['Total_Cost'] = won_data_copy['CPI'] * won_data_copy['Completes']
            lost_data_copy['Total_Cost'] = lost_data_copy['CPI'] * lost_data_copy['Completes']

            fig = go.Figure()

            fig.add_trace(go.Scatter3d(
                x=won_data_copy['IR'],
                y=won_data_copy['LOI'],
                z=won_data_copy['Total_Cost'],
                mode='markers',
                marker=dict(size=won_data_copy['Completes'] / 50, color='#3288bd', opacity=0.7, symbol='circle'),
                name='Won',
                hovertemplate='<b>Won Bid</b><br>IR: %{x:.1f}%<br>LOI: %{y:.1f} min<br>Total Cost: $%{z:.2f}<br>Completes: %{customdata[0]}<br>CPI: $%{customdata[1]:.2f}<extra></extra>',
                customdata=won_data_copy[['Completes', 'CPI']]
            ))

            fig.add_trace(go.Scatter3d(
                x=lost_data_copy['IR'],
                y=lost_data_copy['LOI'],
                z=lost_data_copy['Total_Cost'],
                mode='markers',
                marker=dict(size=lost_data_copy['Completes'] / 50, color='#f58518', opacity=0.7, symbol='circle'),
                name='Lost',
                hovertemplate='<b>Lost Bid</b><br>IR: %{x:.1f}%<br>LOI: %{y:.1f} min<br>Total Cost: $%{z:.2f}<br>Completes: %{customdata[0]}<br>CPI: $%{customdata[1]:.2f}<extra></extra>',
                customdata=lost_data_copy[['Completes', 'CPI']]
            ))

            fig.update_layout(
                title='3D Visualization: IR, LOI, and Total Project Cost',
                scene=dict(
                    xaxis_title='Incidence Rate (%)',
                    yaxis_title='Length of Interview (min)',
                    zaxis_title='Total Project Cost ($)',
                    xaxis=dict(backgroundcolor='rgb(255, 255, 255)'),
                    yaxis=dict(backgroundcolor='rgb(255, 255, 255)'),
                    zaxis=dict(backgroundcolor='rgb(255, 255, 255)')
                ),
                height=700,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig, use_container_width=True, key='analysis_3d_visualization')

            with st.expander("📊 Interpretation"):
                st.markdown("""
                ### Understanding 3D Relationships
                
                The 3D scatter plot visualizes the combined effect of IR, LOI, and sample size (represented by
                marker size) on total project cost, providing a holistic view of how these factors interact.
                
                ### Key Insights
                
                1. **Complex Relationships**: The 3D plot reveals that project costs are influenced by the
                   interaction of multiple factors, not just individual variables in isolation.
                
                2. **Scale Effect**: Marker sizes represent sample sizes, showing how larger projects (bigger
                   markers) relate to other factors - revealing that high-cost projects often have lower IRs
                   and higher LOIs.
                
                3. **Won vs. Lost Clustering**: The spatial distribution of won vs. lost bids in the 3D space
                   reveals pricing patterns and competitive thresholds across different combinations of factors.
                
                4. **Total Cost Perspective**: While CPI is important, this view brings focus to total project
                   cost, which is ultimately what matters for budget decisions and overall competitiveness.
                """)
        except Exception as e:
            logger.error(f"Error in 3D plot: {e}", exc_info=True)
            st.error(f"Could not generate 3D visualization: {str(e)}")
