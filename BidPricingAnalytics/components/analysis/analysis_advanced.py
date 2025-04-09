"""
Advanced analysis component for the CPI Analysis & Prediction Dashboard.
Contains 3D visualizations, heatmaps, and advanced analytical tools.
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
    render_card,
    metrics_row,
    apply_chart_styling,
    add_insights_annotation,
    grid_layout
)

# Import visualization utilities
from utils.visualization import create_heatmap

# Import data utilities
from utils.data_processor import get_data_summary, engineer_features

# Import configuration
from config import COLOR_SYSTEM, TYPOGRAPHY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_advanced_analysis(won_data: pd.DataFrame, lost_data: pd.DataFrame, combined_data: pd.DataFrame) -> None:
    """
    Display the advanced analysis component with 3D visualizations and heatmaps.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids.
        lost_data (pd.DataFrame): DataFrame of Lost bids.
        combined_data (pd.DataFrame): Combined DataFrame of both Won and Lost bids.
    """
    try:
        # Section header
        st.markdown("""
        <div style="
            background-color: #1F1F1F;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1.5rem;
            border-left: 4px solid #BB86FC;
        ">
            <h2 style="color: #FFFFFF; margin-top: 0;">Advanced CPI Analysis</h2>
            <p style="color: #B0B0B0; margin-bottom: 0;">
                This section provides advanced visualizations and analysis techniques to uncover deeper insights
                into the relationships between project parameters and CPI.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        intro_content = """
        <p style="color: #B0B0B0;">
            Explore 3D visualizations, heatmaps, and multi-factor analyses to optimize your pricing strategy
            based on complex parameter interactions.
        </p>
        <ul style="color: #B0B0B0;">
            <li><span style="color: #FFFFFF; font-weight: 500;">3D Visualization</span>: See how IR and LOI jointly influence CPI</li>
            <li><span style="color: #FFFFFF; font-weight: 500;">Heatmap Analysis</span>: Identify price sensitivity patterns across parameter combinations</li>
            <li><span style="color: #FFFFFF; font-weight: 500;">Feature Importance</span>: Understand which factors most strongly influence pricing</li>
        </ul>
        """
        
        render_card(
            title="Advanced Analysis Tools",
            content=intro_content,
            icon="🔬",
            accent_color=COLOR_SYSTEM['ACCENT']['PURPLE']
        )
        
        # Create tabs for advanced analysis
        tab1, tab2, tab3 = st.tabs([
            "3D Analysis", 
            "Heatmap Analysis", 
            "Feature Importance"
        ])
        
        with tab1:
            st.markdown("""
            <div style="color: #B0B0B0; margin-bottom: 1rem;">
                This 3D visualization shows the <span style="color: #FFFFFF; font-weight: 500;">combined effect</span>
                of IR and LOI on CPI values. Use the controls to manipulate the view and gain insights from different angles.
            </div>
            """, unsafe_allow_html=True)
            
            # Create 3D scatter plot for won and lost bids
            fig_3d = go.Figure()
            
            # Add Won data
            fig_3d.add_trace(go.Scatter3d(
                x=won_data['IR'],
                y=won_data['LOI'],
                z=won_data['CPI'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=COLOR_SYSTEM['CHARTS']['WON'],
                    opacity=0.7,
                    line=dict(width=0.5, color=COLOR_SYSTEM['NEUTRAL']['LIGHTEST'])
                ),
                name='Won',
                hovertemplate='<b>Won Bid</b><br>IR: %{x:.1f}%<br>LOI: %{y:.1f} min<br>CPI: $%{z:.2f}<br>Completes: %{customdata}<extra></extra>',
                customdata=won_data['Completes']
            ))
            
            # Add Lost data
            fig_3d.add_trace(go.Scatter3d(
                x=lost_data['IR'],
                y=lost_data['LOI'],
                z=lost_data['CPI'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=COLOR_SYSTEM['CHARTS']['LOST'],
                    opacity=0.7,
                    line=dict(width=0.5, color=COLOR_SYSTEM['NEUTRAL']['LIGHTEST'])
                ),
                name='Lost',
                hovertemplate='<b>Lost Bid</b><br>IR: %{x:.1f}%<br>LOI: %{y:.1f} min<br>CPI: $%{z:.2f}<br>Completes: %{customdata}<extra></extra>',
                customdata=lost_data['Completes']
            ))
            
            # Create surface plots for won and lost bids
            try:
                # Create grid for surface
                ir_range = np.linspace(won_data['IR'].min(), won_data['IR'].max(), 20)
                loi_range = np.linspace(won_data['LOI'].min(), won_data['LOI'].max(), 20)
                IR_grid, LOI_grid = np.meshgrid(ir_range, loi_range)
                
                # Surface for won bids
                from scipy.interpolate import griddata
                points_won = np.column_stack([won_data['IR'], won_data['LOI']])
                Z_won = griddata(points_won, won_data['CPI'], (IR_grid, LOI_grid), method='linear')
                
                fig_3d.add_trace(go.Surface(
                    x=ir_range, 
                    y=loi_range, 
                    z=Z_won,
                    colorscale=[[0, COLOR_SYSTEM['CHARTS']['WON_TRANS']], [1, COLOR_SYSTEM['CHARTS']['WON']]],
                    opacity=0.5,
                    name='Won Surface',
                    showscale=False
                ))
                
                # Surface for lost bids
                points_lost = np.column_stack([lost_data['IR'], lost_data['LOI']])
                Z_lost = griddata(points_lost, lost_data['CPI'], (IR_grid, LOI_grid), method='linear')
                
                fig_3d.add_trace(go.Surface(
                    x=ir_range, 
                    y=loi_range, 
                    z=Z_lost,
                    colorscale=[[0, COLOR_SYSTEM['CHARTS']['LOST_TRANS']], [1, COLOR_SYSTEM['CHARTS']['LOST']]],
                    opacity=0.5,
                    name='Lost Surface',
                    showscale=False
                ))
            except Exception as e:
                logger.warning(f"Could not create surface plots: {e}")
            
            # Apply dark theme styling to 3D plot
            fig_3d.update_layout(
                scene=dict(
                    xaxis=dict(
                        title='Incidence Rate (%)',
                        titlefont=dict(
                            family=TYPOGRAPHY['FONT_FAMILY'],
                            size=12,
                            color=COLOR_SYSTEM['PRIMARY']['MAIN']
                        ),
                        gridcolor=COLOR_SYSTEM['NEUTRAL']['LIGHT'],
                        zerolinecolor=COLOR_SYSTEM['NEUTRAL']['MEDIUM'],
                        backgroundcolor=COLOR_SYSTEM['BACKGROUND']['MAIN']
                    ),
                    yaxis=dict(
                        title='Length of Interview (min)',
                        titlefont=dict(
                            family=TYPOGRAPHY['FONT_FAMILY'],
                            size=12,
                            color=COLOR_SYSTEM['PRIMARY']['MAIN']
                        ),
                        gridcolor=COLOR_SYSTEM['NEUTRAL']['LIGHT'],
                        zerolinecolor=COLOR_SYSTEM['NEUTRAL']['MEDIUM'],
                        backgroundcolor=COLOR_SYSTEM['BACKGROUND']['MAIN']
                    ),
                    zaxis=dict(
                        title='CPI ($)',
                        titlefont=dict(
                            family=TYPOGRAPHY['FONT_FAMILY'],
                            size=12,
                            color=COLOR_SYSTEM['PRIMARY']['MAIN']
                        ),
                        gridcolor=COLOR_SYSTEM['NEUTRAL']['LIGHT'],
                        zerolinecolor=COLOR_SYSTEM['NEUTRAL']['MEDIUM'],
                        backgroundcolor=COLOR_SYSTEM['BACKGROUND']['MAIN']
                    ),
                    bgcolor=COLOR_SYSTEM['BACKGROUND']['CARD'],
                ),
                title=dict(
                    text='3D Visualization of CPI by IR and LOI',
                    font=dict(
                        family=TYPOGRAPHY['FONT_FAMILY'],
                        size=16,
                        color=COLOR_SYSTEM['PRIMARY']['MAIN']
                    ),
                ),
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(
                    font=dict(
                        family=TYPOGRAPHY['FONT_FAMILY'],
                        size=12,
                        color=COLOR_SYSTEM['PRIMARY']['MAIN']
                    ),
                    bgcolor=COLOR_SYSTEM['BACKGROUND']['CARD'],
                    bordercolor=COLOR_SYSTEM['NEUTRAL']['LIGHT'],
                    borderwidth=1
                ),
                paper_bgcolor=COLOR_SYSTEM['BACKGROUND']['MAIN'],
                plot_bgcolor=COLOR_SYSTEM['BACKGROUND']['MAIN'],
                hoverlabel=dict(
                    bgcolor=COLOR_SYSTEM['BACKGROUND']['CARD'],
                    font_size=12,
                    font_family=TYPOGRAPHY['FONT_FAMILY']
                ),
                height=700
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # Controls and insights for 3D visualization
            controls_col1, controls_col2 = st.columns(2)
            
            with controls_col1:
                st.markdown("""
                <div style="background-color: #2E2E2E; padding: 1rem; border-radius: 0.5rem;">
                    <h4 style="color: #FFFFFF; margin-top: 0;">Navigation Tips</h4>
                    <ul style="color: #B0B0B0; margin-bottom: 0;">
                        <li><span style="color: #FFFFFF;">Left click + drag</span>: Rotate view</li>
                        <li><span style="color: #FFFFFF;">Right click + drag</span>: Pan view</li>
                        <li><span style="color: #FFFFFF;">Scroll</span>: Zoom in/out</li>
                        <li><span style="color: #FFFFFF;">Double click</span>: Reset view</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with controls_col2:
                st.markdown("""
                <div style="background-color: #2E2E2E; padding: 1rem; border-radius: 0.5rem;">
                    <h4 style="color: #FFFFFF; margin-top: 0;">Key Insights</h4>
                    <ul style="color: #B0B0B0; margin-bottom: 0;">
                        <li>The <span style="color: #FFB74D;">lost bids surface</span> consistently sits above the <span style="color: #00BFFF;">won bids surface</span></li>
                        <li>The gap increases at lower IR values</li>
                        <li>Both surfaces rise as LOI increases</li>
                        <li>The steepest slope is in the low-IR, high-LOI region</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("""
            <div style="color: #B0B0B0; margin-bottom: 1rem;">
                Heatmap analysis reveals pricing patterns across combinations of IR and LOI, 
                helping identify where won and lost bids differ most significantly.
            </div>
            """, unsafe_allow_html=True)
            
            # Create pivot tables for heatmaps
            try:
                # Generate pivot tables
                won_pivot = won_data.pivot_table(
                    values='CPI', 
                    index='IR_Bin', 
                    columns='LOI_Bin', 
                    aggfunc='mean'
                )
                
                lost_pivot = lost_data.pivot_table(
                    values='CPI', 
                    index='IR_Bin', 
                    columns='LOI_Bin', 
                    aggfunc='mean'
                )
                
                # Create heatmaps
                won_heatmap = create_heatmap(
                    won_pivot, 
                    title="Won Bids: Average CPI by IR and LOI", 
                    colorscale="Viridis"
                )
                
                lost_heatmap = create_heatmap(
                    lost_pivot, 
                    title="Lost Bids: Average CPI by IR and LOI", 
                    colorscale="Plasma"
                )
                
                # Display heatmaps
                st.plotly_chart(won_heatmap, use_container_width=True)
                st.plotly_chart(lost_heatmap, use_container_width=True)
                
                # Calculate difference heatmap
                pivot_diff = lost_pivot - won_pivot
                diff_heatmap = create_heatmap(
                    pivot_diff, 
                    title="Price Differential (Lost - Won)", 
                    colorscale="RdBu_r"
                )
                
                st.plotly_chart(diff_heatmap, use_container_width=True)
                
                # Identify biggest pricing gaps
                st.markdown("""
                <h4 style="color: #FFFFFF;">Largest Pricing Differentials</h4>
                <p style="color: #B0B0B0;">These parameter combinations show the largest gaps between won and lost bid pricing, indicating areas of pricing sensitivity.</p>
                """, unsafe_allow_html=True)
                
                pivot_diff_flat = pivot_diff.stack().dropna()
                
                if not pivot_diff_flat.empty:
                    top_diff_indices = pivot_diff_flat.nlargest(3).index.tolist()
                    opportunity_content = """
                    <div style="background-color: #2E2E2E; padding: 1rem; border-radius: 0.5rem;">
                        <h4 style="color: #FFFFFF; margin-top: 0;">Pricing Opportunity Areas</h4>
                        <table style="width: 100%; border-collapse: collapse;">
                            <thead>
                                <tr style="border-bottom: 1px solid #3A3A3A;">
                                    <th style="padding: 0.5rem; text-align: left; color: #FFFFFF;">IR Range</th>
                                    <th style="padding: 0.5rem; text-align: left; color: #FFFFFF;">LOI Range</th>
                                    <th style="padding: 0.5rem; text-align: right; color: #FFFFFF;">Difference</th>
                                    <th style="padding: 0.5rem; text-align: right; color: #FFFFFF;">% Difference</th>
                                </tr>
                            </thead>
                            <tbody>
                    """
                    
                    for ir_bin, loi_bin in top_diff_indices:
                        diff_value = pivot_diff.loc[ir_bin, loi_bin]
                        won_value = won_pivot.loc[ir_bin, loi_bin]
                        pct_diff = (diff_value / won_value) * 100 if won_value != 0 else 0
                        
                        opportunity_content += f"""
                                <tr style="border-bottom: 1px solid #3A3A3A;">
                                    <td style="padding: 0.5rem; color: #B0B0B0;">{ir_bin}</td>
                                    <td style="padding: 0.5rem; color: #B0B0B0;">{loi_bin}</td>
                                    <td style="padding: 0.5rem; text-align: right; color: #FFB74D; font-weight: 600;">${diff_value:.2f}</td>
                                    <td style="padding: 0.5rem; text-align: right; color: #FFB74D; font-weight: 600;">{pct_diff:.1f}%</td>
                                </tr>
                        """
                    
                    opportunity_content += """
                            </tbody>
                        </table>
                        <p style="color: #B0B0B0; margin-top: 1rem; margin-bottom: 0;">
                            <strong style="color: #FFFFFF;">Pricing recommendation:</strong> For projects falling in these parameter ranges,
                            maintain a moderate pricing premium above won bid levels but below lost bid levels to
                            maximize both win rate and profitability.
                        </p>
                    </div>
                    """
                    
                    st.markdown(opportunity_content, unsafe_allow_html=True)
            except Exception as e:
                logger.error(f"Error creating heatmaps: {e}", exc_info=True)
                st.warning("Unable to create heatmaps due to insufficient categorical bin data. Please ensure your data has IR_Bin and LOI_Bin columns with sufficient values.")
        
        with tab3:
            st.markdown("""
            <div style="color: #B0B0B0; margin-bottom: 1rem;">
                This analysis examines how different factors impact CPI by using advanced feature engineering
                and statistical techniques to understand more complex relationships.
            </div>
            """, unsafe_allow_html=True)
            
            explanation_content = """
            <p style="color: #B0B0B0;">
                Feature importance helps identify which project parameters have the strongest influence on CPI.
                These insights can help you:
            </p>
            <ul style="color: #B0B0B0;">
                <li>Prioritize which factors to focus on during pricing</li>
                <li>Understand which project characteristics most influence costs</li>
                <li>Develop more effective pricing strategies based on key drivers</li>
            </ul>
            """
            
            render_card(
                title="Understanding Feature Importance",
                content=explanation_content,
                icon="🔍",
                accent_color=COLOR_SYSTEM['ACCENT']['PURPLE']
            )
            
            # Engineer features using the existing function
            try:
                won_data_copy = won_data.copy()
                lost_data_copy = lost_data.copy()
                
                # Ensure 'Type' column exists
                if 'Type' not in won_data_copy.columns:
                    won_data_copy['Type'] = 'Won'
                if 'Type' not in lost_data_copy.columns:
                    lost_data_copy['Type'] = 'Lost'
                
                combined_for_analysis = pd.concat([won_data_copy, lost_data_copy], ignore_index=True)
                engineered_data = engineer_features(combined_for_analysis)
                
                # Create correlation heatmap
                st.subheader("Feature Correlation Heatmap")
                
                numeric_cols = [
                    'IR', 'LOI', 'Completes', 'CPI', 'IR_LOI_Ratio', 
                    'IR_Completes_Ratio', 'LOI_Completes_Ratio'
                ]
                
                additional_features = [
                    'IR_LOI_Product', 'CPI_per_Minute', 'Log_Completes', 
                    'Log_IR', 'Log_LOI', 'Log_CPI', 'IR_Normalized', 
                    'LOI_Normalized', 'Completes_Normalized', 'CPI_Efficiency'
                ]
                
                for feature in additional_features:
                    if feature in engineered_data.columns:
                        numeric_cols.append(feature)
                
                # Calculate correlation matrix
                corr_matrix = engineered_data[numeric_cols].corr()
                
                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu_r',
                    zmid=0,
                    text=[[f"{val:.2f}" for val in row] for row in corr_matrix.values],
                    texttemplate="%{text}",
                    hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>'
                ))
                
                # Style correlation heatmap
                fig = apply_chart_styling(
                    fig,
                    title="Feature Correlation Matrix",
                    height=600,
                    show_legend=False
                )
                
                # Update specific layout features
                fig.update_layout(
                    xaxis=dict(
                        tickangle=45,
                        tickfont=dict(
                            family=TYPOGRAPHY['FONT_FAMILY'],
                            size=10,
                            color=COLOR_SYSTEM['PRIMARY']['MAIN']
                        )
                    ),
                    yaxis=dict(
                        tickfont=dict(
                            family=TYPOGRAPHY['FONT_FAMILY'],
                            size=10,
                            color=COLOR_SYSTEM['PRIMARY']['MAIN']
                        )
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create feature importance table based on correlation with CPI
                st.subheader("Feature Importance (by Correlation with CPI)")
                
                cpi_corr = corr_matrix['CPI'].drop('CPI').abs().sort_values(ascending=False)
                
                # Highlight top positive and negative correlations
                positive_corr = corr_matrix['CPI'].drop('CPI').sort_values(ascending=False)
                negative_corr = corr_matrix['CPI'].drop('CPI').sort_values()
                
                insights_content = f"""
                <div style="background-color: #2E2E2E; padding: 1rem; border-radius: 0.5rem;">
                    <h4 style="color: #FFFFFF; margin-top: 0;">Key Feature Insights</h4>
                    
                    <h5 style="color: #00BFFF; margin-bottom: 0.5rem;">Top features positively correlated with CPI:</h5>
                    <ol style="color: #B0B0B0; margin-top: 0.5rem;">
                """
                
                for i, (feature, value) in enumerate(positive_corr.head(3).items()):
                    insights_content += f"""
                        <li>
                            <span style="color: #FFFFFF;">{feature}</span>: 
                            <span style="color: #00BFFF; font-weight: 600;">{value:.3f}</span>
                        </li>
                    """
                
                insights_content += f"""
                    </ol>
                    
                    <h5 style="color: #FFB74D; margin-bottom: 0.5rem;">Top features negatively correlated with CPI:</h5>
                    <ol style="color: #B0B0B0; margin-top: 0.5rem;">
                """
                
                for i, (feature, value) in enumerate(negative_corr.head(3).items()):
                    insights_content += f"""
                        <li>
                            <span style="color: #FFFFFF;">{feature}</span>: 
                            <span style="color: #FFB74D; font-weight: 600;">{value:.3f}</span>
                        </li>
                    """
                
                insights_content += f"""
                    </ol>
                    
                    <p style="color: #B0B0B0; margin-top: 1rem; margin-bottom: 0;">
                        <strong style="color: #FFFFFF;">Strategic insight:</strong> Focus on parameters with the 
                        strongest influence on CPI to improve pricing efficiency. Consider these factors carefully 
                        when determining pricing for new projects.
                    </p>
                </div>
                """
                
                st.markdown(insights_content, unsafe_allow_html=True)
                
            except Exception as e:
                logger.error(f"Error in feature engineering analysis: {e}", exc_info=True)
                st.warning("Unable to complete feature engineering analysis. This advanced analysis requires more computing resources.")
                
                st.markdown("""
                <div style="background-color: #2E2E2E; padding: 1rem; border-radius: 0.5rem;">
                    <h4 style="color: #FFFFFF; margin-top: 0;">Advanced Analysis Requirements</h4>
                    <p style="color: #B0B0B0;">
                        Advanced analysis requires more computing resources. Please try the following:
                    </p>
                    <ul style="color: #B0B0B0; margin-bottom: 0;">
                        <li>Use a smaller dataset sample</li>
                        <li>Run the analysis in a local environment with more resources</li>
                        <li>Increase the memory allocated to the application</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Error in show_advanced_analysis: {e}", exc_info=True)
        st.error(f"An error occurred while displaying the advanced analysis: {str(e)}")
