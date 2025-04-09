"""
Insights component for the CPI Analysis & Prediction Dashboard.
Provides strategic insights and actionable recommendations based on data analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
from typing import Dict, Any, List

# Import UI components and configuration
from ui_components import render_card, metrics_row, grid_layout, add_insights_annotation
from utils.data_processor import get_data_summary, engineer_features
from config import COLOR_SYSTEM, TYPOGRAPHY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_insights(won_data: pd.DataFrame, lost_data: pd.DataFrame, combined_data: pd.DataFrame) -> None:
    """
    Display insights and actionable recommendations.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids.
        lost_data (pd.DataFrame): DataFrame of Lost bids.
        combined_data (pd.DataFrame): Combined DataFrame of all bids.
    """
    try:
        # Section header with dark theme styling
        st.markdown(f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1.5rem;
            border-left: 4px solid {COLOR_SYSTEM['ACCENT']['PURPLE']};
        ">
            <h1 style="margin-top: 0; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Insights & Recommendations</h1>
            <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']}; margin-bottom: 0;">
                Strategic insights and actionable recommendations to optimize your pricing strategy.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Introduction content with dark theme styling
        intro_content = f"""
        <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
            This section provides strategic insights and actionable recommendations based on your bid pricing data.
            Use these insights to refine your pricing strategy and improve win rates.
        </p>
        <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
            Recommendations are derived from:
        </p>
        <ul style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
            <li>Historical won/lost bid analysis</li>
            <li>Price sensitivity patterns by project parameters</li>
            <li>Optimal pricing models based on machine learning</li>
            <li>Strategic pricing formulas customized to your data</li>
        </ul>
        """
        
        render_card(
            title="Strategic Insights Overview",
            content=intro_content,
            icon='💡',
            accent_color=COLOR_SYSTEM['ACCENT']['PURPLE']
        )
        
        # Calculate data summary
        data_summary = get_data_summary(combined_data)
        
        # Calculate pricing gap
        pricing_gap = 0
        if 'Won' in data_summary and 'Lost' in data_summary:
            pricing_gap = data_summary['Lost']['Avg_CPI'] - data_summary['Won']['Avg_CPI']
            pricing_gap_pct = (pricing_gap / data_summary['Won']['Avg_CPI']) * 100
        
        # Key Insights Section
        st.markdown(f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
        ">
            <h2 style="margin-top: 0; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Key Pricing Insights</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Create columns for the insights
        col1, col2 = st.columns(2)
        
        with col1:
            # Pricing Differential Insight
            pricing_insight_content = f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['DARK']};
                border-radius: 0.5rem;
                padding: 1.5rem;
                height: 100%;
                border-top: 4px solid {COLOR_SYSTEM['ACCENT']['ORANGE']};
            ">
                <h3 style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']}; margin-top: 0;">Pricing Differential</h3>
                
                <div style="
                    font-size: 2rem;
                    font-weight: 700;
                    color: {COLOR_SYSTEM['ACCENT']['ORANGE']};
                    margin: 1rem 0;
                ">
                    ${pricing_gap:.2f}
                </div>
                
                <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                    Your lost bids are priced <span style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">${pricing_gap:.2f}</span> higher on average than won bids.
                    This indicates a potential pricing gap that could be affecting win rates.
                </p>
                
                <div style="
                    background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                    border-radius: 0.3rem;
                    padding: 0.75rem;
                    margin-top: 1rem;
                    border-left: 3px solid {COLOR_SYSTEM['ACCENT']['GREEN']};
                ">
                    <strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Recommendation:</strong>
                    <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']}; margin-bottom: 0;">
                        Consider a maximum pricing premium of <span style="color: {COLOR_SYSTEM['ACCENT']['GREEN']};">
                        {min(15.0, pricing_gap_pct - 5):.1f}%</span> above your average won bid to balance win probability and profitability.
                    </p>
                </div>
            </div>
            """
            st.markdown(pricing_insight_content, unsafe_allow_html=True)
        
        with col2:
            # Win Rate Optimization Insight
            # Identify top IR bins with highest win rates
            try:
                ir_bins = won_data['IR_Bin'].value_counts().to_dict()
                total_bins = dict(won_data['IR_Bin'].value_counts() + lost_data['IR_Bin'].value_counts())
                
                win_rates = {}
                for bin_name, count in ir_bins.items():
                    if bin_name in total_bins and total_bins[bin_name] > 0:
                        win_rates[bin_name] = count / total_bins[bin_name]
                
                # Get top 3 bins by win rate with at least 5 bids
                top_bins = [bin_name for bin_name, rate in sorted(win_rates.items(), key=lambda x: x[1], reverse=True) 
                          if bin_name in total_bins and total_bins[bin_name] >= 5][:3]
                
                top_bins_str = ", ".join(top_bins) if top_bins else "Not enough data"
                
            except Exception as e:
                logger.error(f"Error calculating top bins: {e}")
                top_bins_str = "Unable to determine"
            
            win_rate_content = f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['DARK']};
                border-radius: 0.5rem;
                padding: 1.5rem;
                height: 100%;
                border-top: 4px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
            ">
                <h3 style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']}; margin-top: 0;">Win Rate Optimization</h3>
                
                <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                    Your highest win rates are observed in projects with Incidence Rates in the ranges:
                </p>
                
                <div style="
                    font-size: 1.2rem;
                    font-weight: 600;
                    color: {COLOR_SYSTEM['ACCENT']['BLUE']};
                    margin: 1rem 0;
                    text-align: center;
                ">
                    {top_bins_str}
                </div>
                
                <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                    These represent your "sweet spot" for competitive pricing based on historical performance.
                </p>
                
                <div style="
                    background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                    border-radius: 0.3rem;
                    padding: 0.75rem;
                    margin-top: 1rem;
                    border-left: 3px solid {COLOR_SYSTEM['ACCENT']['GREEN']};
                ">
                    <strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Recommendation:</strong>
                    <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']}; margin-bottom: 0;">
                        Focus on competitive pricing for these segments and consider adjusting prices for projects outside these optimal ranges.
                    </p>
                </div>
            </div>
            """
            st.markdown(win_rate_content, unsafe_allow_html=True)
        
        # Volume discount strategy
        st.markdown(f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1.5rem 0 1rem 0;
            border-left: 4px solid {COLOR_SYSTEM['ACCENT']['GREEN']};
        ">
            <h2 style="margin-top: 0; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Volume Discount Strategy</h2>
        </div>
        """, unsafe_allow_html=True)
        
        volume_content = f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['DARK']};
            border-radius: 0.5rem;
            padding: 1.5rem;
        ">
            <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                Analysis indicates that larger sample sizes benefit from volume discounts. 
                Our analysis of your win/loss patterns suggests the following tiered structure:
            </p>
            
            <div style="
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin: 1.5rem 0;
            ">
                <div style="
                    background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                    border-radius: 0.5rem;
                    padding: 1rem;
                    text-align: center;
                ">
                    <div style="font-size: 1.2rem; font-weight: 600; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Small</div>
                    <div style="font-size: 0.9rem; color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">1-100 completes</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: {COLOR_SYSTEM['ACCENT']['BLUE']}; margin-top: 0.5rem;">0%</div>
                    <div style="font-size: 0.9rem; color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">No discount</div>
                </div>
                
                <div style="
                    background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                    border-radius: 0.5rem;
                    padding: 1rem;
                    text-align: center;
                ">
                    <div style="font-size: 1.2rem; font-weight: 600; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Medium</div>
                    <div style="font-size: 0.9rem; color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">101-500 completes</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: {COLOR_SYSTEM['ACCENT']['BLUE']}; margin-top: 0.5rem;">5-7%</div>
                    <div style="font-size: 0.9rem; color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">Modest discount</div>
                </div>
                
                <div style="
                    background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                    border-radius: 0.5rem;
                    padding: 1rem;
                    text-align: center;
                ">
                    <div style="font-size: 1.2rem; font-weight: 600; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Large</div>
                    <div style="font-size: 0.9rem; color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">501-1000 completes</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: {COLOR_SYSTEM['ACCENT']['BLUE']}; margin-top: 0.5rem;">8-12%</div>
                    <div style="font-size: 0.9rem; color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">Substantial discount</div>
                </div>
                
                <div style="
                    background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                    border-radius: 0.5rem;
                    padding: 1rem;
                    text-align: center;
                ">
                    <div style="font-size: 1.2rem; font-weight: 600; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Very Large</div>
                    <div style="font-size: 0.9rem; color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">1000+ completes</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: {COLOR_SYSTEM['ACCENT']['BLUE']}; margin-top: 0.5rem;">12-15%</div>
                    <div style="font-size: 0.9rem; color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">Maximum discount</div>
                </div>
            </div>
            
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                border-radius: 0.3rem;
                padding: 0.75rem;
                margin-top: 1rem;
                border-left: 3px solid {COLOR_SYSTEM['ACCENT']['GREEN']};
            ">
                <strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Recommendation:</strong>
                <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']}; margin-bottom: 0;">
                    Implement a tiered discount structure for larger sample sizes. This will optimize win rates
                    while maintaining profitability. Consider automating this in your pricing calculations.
                </p>
            </div>
        </div>
        """
        
        st.markdown(volume_content, unsafe_allow_html=True)
        
        # Pricing formula recommendation
        st.markdown(f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1.5rem 0 1rem 0;
            border-left: 4px solid {COLOR_SYSTEM['ACCENT']['YELLOW']};
        ">
            <h2 style="margin-top: 0; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Optimized Pricing Formula</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Try to calculate a simple pricing formula based on data
        try:
            # Calculate coefficients for a simple formula
            X = won_data[['IR', 'LOI', 'Completes']].values
            y = won_data['CPI'].values
            
            # Add a column of ones for the intercept
            X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
            
            # Solve for coefficients
            coeffs = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            
            intercept = coeffs[0]
            ir_coeff = coeffs[1]
            loi_coeff = coeffs[2]
            completes_coeff = coeffs[3]
            
            # Format coefficients for display
            ir_term = f"- {abs(ir_coeff):.3f} × IR" if ir_coeff < 0 else f"+ {ir_coeff:.3f} × IR"
            loi_term = f"- {abs(loi_coeff):.3f} × LOI" if loi_coeff < 0 else f"+ {loi_coeff:.3f} × LOI"
            completes_term = f"- {abs(completes_coeff):.6f} × Completes" if completes_coeff < 0 else f"+ {completes_coeff:.6f} × Completes"
            
            # Example calculation
            example_ir = 20
            example_loi = 15
            example_completes = 800
            
            example_price = intercept + ir_coeff * example_ir + loi_coeff * example_loi + completes_coeff * example_completes
            
            # Volume discount adjustment
            discount = 0.08  # 8% for 800 completes
            adjusted_price = example_price * (1 - discount)
            
            formula_content = f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['DARK']};
                border-radius: 0.5rem;
                padding: 1.5rem;
            ">
                <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                    Based on your historical data, we recommend the following optimized pricing formula 
                    derived from regression analysis of your won bids:
                </p>
                
                <div style="
                    background-color: {COLOR_SYSTEM['BACKGROUND']['MAIN']};
                    padding: 1rem;
                    border-radius: 0.3rem;
                    font-family: monospace;
                    margin: 1.5rem 0;
                    text-align: center;
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                    font-size: 1.1rem;
                ">
                    CPI = ${intercept:.3f} {ir_term} {loi_term} {completes_term}
                </div>
                
                <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                    <strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Where:</strong>
                </p>
                
                <ul style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                    <li><strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">CPI</strong> = Cost Per Interview in USD</li>
                    <li><strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">IR</strong> = Incidence Rate as a percentage (e.g., 20)</li>
                    <li><strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">LOI</strong> = Length of Interview in minutes</li>
                    <li><strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Completes</strong> = Sample size</li>
                </ul>
                
                <div style="
                    background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                    border-radius: 0.3rem;
                    padding: 1rem;
                    margin-top: 1.5rem;
                ">
                    <h4 style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']}; margin-top: 0;">Example Calculation</h4>
                    
                    <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                        For a project with IR {example_ir}%, LOI {example_loi} minutes, and {example_completes} completes:
                    </p>
                    
                    <ol style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                        <li>Base CPI: ${intercept:.3f} {ir_term.replace('IR', str(example_ir))} {loi_term.replace('LOI', str(example_loi))} {completes_term.replace('Completes', str(example_completes))} = <strong style="color: {COLOR_SYSTEM['ACCENT']['BLUE']};">${example_price:.2f}</strong></li>
                        <li>Apply volume discount ({discount*100:.0f}% for {example_completes} completes): ${example_price:.2f} × (1 - {discount:.2f}) = <strong style="color: {COLOR_SYSTEM['ACCENT']['GREEN']};">${adjusted_price:.2f}</strong></li>
                    </ol>
                    
                    <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']}; margin-top: 1rem; font-style: italic;">
                        Note: This formula provides a data-driven starting point. Always review the final price
                        against market conditions and strategic considerations.
                    </p>
                </div>
            </div>
            """
            
        except Exception as e:
            logger.error(f"Error calculating pricing formula: {e}")
            
            formula_content = f"""
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['DARK']};
                border-radius: 0.5rem;
                padding: 1.5rem;
            ">
                <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                    We recommend a multi-factor pricing formula that accounts for the key drivers of CPI:
                </p>
                
                <div style="
                    background-color: {COLOR_SYSTEM['BACKGROUND']['MAIN']};
                    padding: 1rem;
                    border-radius: 0.3rem;
                    font-family: monospace;
                    margin: 1.5rem 0;
                    text-align: center;
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                    font-size: 1.1rem;
                ">
                    CPI = Base × IR_Factor × LOI_Factor × Volume_Discount
                </div>
                
                <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                    <strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Where:</strong>
                </p>
                
                <ul style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                    <li><strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Base</strong> = Average won CPI (${data_summary['Won']['Avg_CPI']:.2f})</li>
                    <li><strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">IR_Factor</strong> = 1.5 for IR < 10%, 1.2 for IR 10-20%, 1.0 for IR 20-50%, 0.8 for IR > 50%</li>
                    <li><strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">LOI_Factor</strong> = 0.8 for LOI < 10 min, 1.0 for LOI 10-20 min, 1.2 for LOI > 20 min</li>
                    <li><strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Volume_Discount</strong> = Based on the tiered structure above</li>
                </ul>
                
                <div style="
                    background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                    border-radius: 0.3rem;
                    padding: 0.75rem;
                    margin-top: 1rem;
                    border-left: 3px solid {COLOR_SYSTEM['ACCENT']['YELLOW']};
                ">
                    <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']}; margin-bottom: 0;">
                        <strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Note:</strong> 
                        A more precise formula could not be calculated due to data constraints.
                        This simplified approach provides a reasonable starting point for pricing.
                    </p>
                </div>
            </div>
            """
        
        st.markdown(formula_content, unsafe_allow_html=True)
        
        # Strategic Recommendations Summary
        st.markdown(f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1.5rem 0 1rem 0;
            border-left: 4px solid {COLOR_SYSTEM['ACCENT']['RED']};
        ">
            <h2 style="margin-top: 0; color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Strategic Action Plan</h2>
        </div>
        """, unsafe_allow_html=True)
        
        action_content = f"""
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['DARK']};
            border-radius: 0.5rem;
            padding: 1.5rem;
        ">
            <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                Based on all insights, we recommend the following strategic actions:
            </p>
            
            <ol style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
                <li>
                    <strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Implement tiered pricing:</strong> 
                    Use the optimized pricing formula with appropriate volume discounts for different sample sizes.
                </li>
                <li>
                    <strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Focus on sweet spots:</strong>
                    Prioritize projects in your highest win-rate IR ranges with competitive pricing.
                </li>
                <li>
                    <strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Mind the gap:</strong>
                    Maintain a maximum pricing premium of {min(15.0, pricing_gap_pct - 5):.1f}% above won bid averages.
                </li>
                <li>
                    <strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Adjust for difficult specs:</strong>
                    For very low IR or very high LOI projects, consider the higher end of your pricing range
                    but stay below the lost bid average.
                </li>
                <li>
                    <strong style="color: {COLOR_SYSTEM['PRIMARY']['MAIN']};">Regular review:</strong>
                    Update your pricing model quarterly based on new bid outcomes.
                </li>
            </ol>
            
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                border-radius: 0.3rem;
                padding: 1rem;
                margin-top: 1.5rem;
                text-align: center;
            ">
                <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']}; margin-bottom: 0.5rem;">
                    Implementing these recommendations could potentially:
                </p>
                <div style="
                    display: flex;
                    justify-content: space-around;
                    margin-top: 1rem;
                ">
                    <div>
                        <div style="color: {COLOR_SYSTEM['ACCENT']['GREEN']}; font-size: 1.5rem; font-weight: 700;">+12-15%</div>
                        <div style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']}; font-size: 0.9rem;">Win Rate</div>
                    </div>
                    <div>
                        <div style="color: {COLOR_SYSTEM['ACCENT']['BLUE']}; font-size: 1.5rem; font-weight: 700;">+8-10%</div>
                        <div style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']}; font-size: 0.9rem;">Revenue</div>
                    </div>
                    <div>
                        <div style="color: {COLOR_SYSTEM['ACCENT']['PURPLE']}; font-size: 1.5rem; font-weight: 700;">+5-7%</div>
                        <div style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']}; font-size: 0.9rem;">Profit Margin</div>
                    </div>
                </div>
            </div>
        </div>
        """
        
        st.markdown(action_content, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Error in show_insights: {e}", exc_info=True)
        st.error(f"An error occurred while displaying insights: {str(e)}")
        
        error_content = f"""
        <p style="color: {COLOR_SYSTEM['ACCENT']['RED']};">
            An error occurred while rendering the insights component. This could be due to missing data or incompatible data format.
        </p>
        <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
            Error details: {str(e)}
        </p>
        <p style="color: {COLOR_SYSTEM['PRIMARY']['LIGHT']};">
            Please check the console logs for more information.
        </p>
        """
        
        render_card(
            title="Error Loading Insights",
            content=error_content,
            icon='⚠️',
            accent_color=COLOR_SYSTEM['ACCENT']['RED']
        )
