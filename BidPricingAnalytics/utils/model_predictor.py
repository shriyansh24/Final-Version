"""
Model prediction functionality for the CPI Analysis & Prediction Dashboard.
Provides robust prediction and recommendation functions with high-contrast UI styling.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import color system for styled outputs
try:
    from config import COLOR_SYSTEM
except ImportError:
    logger.warning("Could not import COLOR_SYSTEM from config. Using fallback colors.")
    # Fallback color system for console outputs
    COLOR_SYSTEM = {
        "ACCENT": {
            "BLUE": "#00BFFF",
            "ORANGE": "#FFB74D",
            "GREEN": "#66FF99",
            "RED": "#FF3333",
            "PURPLE": "#BB86FC"
        },
        "PRIMARY": {
            "MAIN": "#FFFFFF",
            "LIGHT": "#B0B0B0"
        },
        "BACKGROUND": {
            "CARD": "#1F1F1F",
            "MAIN": "#000000"
        },
        "NEUTRAL": {
            "LIGHT": "#3A3A3A"
        },
        "CHARTS": {
            "WON": "#00CFFF",
            "LOST": "#FFB74D"
        }
    }

def predict_cpi(models: Dict[str, Any], user_input: Dict[str, float], feature_names: List[str]) -> Dict[str, float]:
    """
    Predict CPI based on user input using trained models.
    
    Args:
        models (Dict[str, Any]): Dictionary of trained models.
        user_input (Dict[str, float]): Dictionary with user input parameters.
        feature_names (List[str]): List of feature names expected by the models.
        
    Returns:
        Dict[str, float]: Dictionary of model predictions keyed by model name.
    """
    try:
        logger.info(f"Making predictions with user input: {user_input}")
        
        # Create a DataFrame using user_input ensuring the correct column order
        input_df = pd.DataFrame([user_input], columns=[col for col in feature_names if col in user_input])
        
        # Fill missing features with default value 0
        for col in feature_names:
            if col not in input_df.columns or input_df[col].isna().any():
                input_df[col] = 0
                logger.info(f"Feature '{col}' missing or NaN in input. Defaulting to 0.")
        
        # Add derived features if they are not present
        if 'IR' in input_df and 'LOI' in input_df and 'IR_LOI_Ratio' not in input_df:
            input_df['IR_LOI_Ratio'] = input_df['IR'] / input_df['LOI'].replace(0, 1)  # Avoid division by zero
        
        if 'IR' in input_df and 'Completes' in input_df and 'IR_Completes_Ratio' not in input_df:
            input_df['IR_Completes_Ratio'] = input_df['IR'] / input_df['Completes'].replace(0, 1)
        
        if 'LOI' in input_df and 'Completes' in input_df and 'LOI_Completes_Ratio' not in input_df:
            input_df['LOI_Completes_Ratio'] = input_df['LOI'] / input_df['Completes'].replace(0, 1)
        
        if 'IR' in input_df and 'LOI' in input_df and 'IR_LOI_Product' not in input_df:
            input_df['IR_LOI_Product'] = input_df['IR'] * input_df['LOI']
        
        if 'Completes' in input_df and 'Log_Completes' not in input_df:
            input_df['Log_Completes'] = np.log1p(input_df['Completes'])
        
        # Add the Type column if needed (assume we want won bid prediction by default)
        if 'Type_Won' in feature_names and 'Type_Won' not in input_df:
            input_df['Type_Won'] = 1
        
        # Ensure final DataFrame has exactly the same columns (in order) as feature_names
        final_input = input_df.reindex(columns=feature_names, fill_value=0)
        
        predictions = {}
        
        # Run predictions for each model in the provided dictionary
        for model_name, model in models.items():
            try:
                pred_value = model.predict(final_input)[0]
                predictions[model_name] = pred_value
                logger.info(f"{model_name} prediction: ${pred_value:.2f}")
            except Exception as e:
                logger.error(f"Error making prediction with {model_name} model: {e}", exc_info=True)
                predictions[model_name] = None
        
        # Remove any models that returned None
        predictions = {k: v for k, v in predictions.items() if v is not None}
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error in predict_cpi: {e}", exc_info=True)
        return {}

def get_prediction_metrics(predictions: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate summary statistics for a set of model predictions.
    
    Args:
        predictions (Dict[str, float]): Dictionary of model predictions.
        
    Returns:
        Dict[str, float]: Dictionary with statistical metrics.
    """
    try:
        if not predictions:
            return {}
        
        values = list(predictions.values())
        
        metrics = {
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'median': sorted(values)[len(values) // 2],
            'range': max(values) - min(values),
            'std': np.std(values)
        }
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error in get_prediction_metrics: {e}", exc_info=True)
        return {}

def get_recommendation(predicted_cpi: float, won_avg: float, lost_avg: float) -> Dict[str, str]:
    """
    Generate a pricing recommendation based on predicted CPI and historical averages.
    
    Args:
        predicted_cpi (float): Predicted CPI.
        won_avg (float): Average CPI for won bids.
        lost_avg (float): Average CPI for lost bids.
        
    Returns:
        Dict[str, str]: Dictionary with recommendation text and styled HTML.
    """
    try:
        midpoint = (won_avg + lost_avg) / 2
        diff_percentage = ((predicted_cpi - won_avg) / won_avg) * 100
        
        if predicted_cpi <= won_avg * 0.9:
            recommendation = (
                "The predicted CPI is significantly lower than the average won bid. "
                "This may boost win rates but risks margin erosion. Consider a moderate increase "
                f"toward the won average of ${won_avg:.2f}."
            )
            status = "low"
            status_color = COLOR_SYSTEM["ACCENT"]["ORANGE"]
            
        elif predicted_cpi <= won_avg:
            recommendation = (
                "The predicted CPI is slightly below the average for won bids, indicating a competitive price."
            )
            status = "competitive"
            status_color = COLOR_SYSTEM["ACCENT"]["GREEN"]
            
        elif predicted_cpi <= midpoint:
            recommendation = (
                "The CPI is moderately above the won average but still below the midpoint, balancing "
                "competition and profitability."
            )
            status = "balanced"
            status_color = COLOR_SYSTEM["ACCENT"]["BLUE"]
            
        elif predicted_cpi <= lost_avg:
            recommendation = (
                "The CPI is nearing the lost bid range. This could improve margins but may risk win rates. "
                "Evaluate other strategic factors."
            )
            status = "cautious"
            status_color = COLOR_SYSTEM["ACCENT"]["PURPLE"]
            
        else:
            recommendation = (
                "The predicted CPI is above the average for lost bids. This is likely too high for competitive "
                "bidding. Consider lowering the price."
            )
            status = "high"
            status_color = COLOR_SYSTEM["ACCENT"]["RED"]
        
        # Add percentage difference
        diff_note = f" (Predicted CPI is {diff_percentage:+.1f}% compared to won average.)"
        recommendation += diff_note
        
        # Create HTML version with styling
        html_recommendation = f"""
        <div style="padding: 0.5rem 0;">
            <div style="
                display: inline-block;
                padding: 0.25rem 0.5rem;
                background-color: {status_color};
                color: #000000;
                border-radius: 0.25rem;
                font-weight: 600;
                font-size: 0.8rem;
                margin-bottom: 0.5rem;
                text-transform: uppercase;
            ">{status}</div>
            <div style="margin-top: 0.3rem;">{recommendation}</div>
        </div>
        """
        
        return {
            "text": recommendation,
            "html": html_recommendation,
            "status": status,
            "status_color": status_color
        }
    
    except Exception as e:
        logger.error(f"Error in get_recommendation: {e}", exc_info=True)
        error_msg = "Unable to generate recommendation due to an error."
        return {
            "text": error_msg,
            "html": f"<div style='color: {COLOR_SYSTEM['ACCENT']['RED']};'>{error_msg}</div>",
            "status": "error",
            "status_color": COLOR_SYSTEM["ACCENT"]["RED"]
        }

def get_detailed_pricing_strategy(predicted_cpi: float, user_input: Dict[str, float],
                                won_data: pd.DataFrame, lost_data: pd.DataFrame) -> Dict[str, str]:
    """
    Generate a detailed pricing strategy based on predicted CPI and user inputs.
    
    Args:
        predicted_cpi (float): Predicted CPI value.
        user_input (Dict[str, float]): Dictionary of user inputs.
        won_data (pd.DataFrame): DataFrame of won bids.
        lost_data (pd.DataFrame): DataFrame of lost bids.
        
    Returns:
        Dict[str, str]: Dictionary with 'markdown' and 'html' formatted strategy.
    """
    try:
        ir = user_input.get('IR', 0)
        loi = user_input.get('LOI', 0)
        completes = user_input.get('Completes', 0)
        
        # Define filter ranges for similar projects
        ir_lower, ir_upper = max(0, ir - 10), min(100, ir + 10)
        loi_lower, loi_upper = max(0, loi - 5), loi + 5
        completes_lower, completes_upper = max(0, completes * 0.5), completes * 1.5
        
        similar_won = won_data[
            (won_data['IR'] >= ir_lower) & (won_data['IR'] <= ir_upper) &
            (won_data['LOI'] >= loi_lower) & (won_data['LOI'] <= loi_upper) &
            (won_data['Completes'] >= completes_lower) & (won_data['Completes'] <= completes_upper)
        ]
        
        similar_lost = lost_data[
            (lost_data['IR'] >= ir_lower) & (lost_data['IR'] <= ir_upper) &
            (lost_data['LOI'] >= loi_lower) & (lost_data['LOI'] <= loi_upper) &
            (lost_data['Completes'] >= completes_lower) & (lost_data['Completes'] <= completes_upper)
        ]
        
        # Build strategy in Markdown format
        md_lines = []
        md_lines.append(f"### Detailed Pricing Strategy for IR={ir}%, LOI={loi} min, Completes={completes}")
        md_lines.append("")
        
        if not similar_won.empty:
            md_lines.append(f"**Similar Won Projects (n={len(similar_won)}):**")
            md_lines.append(f"- Average CPI: ${similar_won['CPI'].mean():.2f}")
            md_lines.append(f"- Range: ${similar_won['CPI'].min():.2f} - ${similar_won['CPI'].max():.2f}")
            md_lines.append("")
        
        if not similar_lost.empty:
            md_lines.append(f"**Similar Lost Projects (n={len(similar_lost)}):**")
            md_lines.append(f"- Average CPI: ${similar_lost['CPI'].mean():.2f}")
            md_lines.append(f"- Range: ${similar_lost['CPI'].min():.2f} - ${similar_lost['CPI'].max():.2f}")
            md_lines.append("")
        
        md_lines.append("**Recommended Pricing Adjustments:**")
        
        # Recommend adjustments based on IR
        if ir < 20:
            md_lines.append("- **Low IR:** Consider a 15–20% premium due to lower respondent qualification rates.")
            ir_adjustment = 17.5  # Midpoint of 15-20%
        elif ir < 50:
            md_lines.append("- **Medium IR:** Add a 5–10% premium.")
            ir_adjustment = 7.5   # Midpoint of 5-10%
        else:
            md_lines.append("- **High IR:** No additional premium needed.")
            ir_adjustment = 0
        
        # Adjustments based on LOI
        if loi > 20:
            md_lines.append("- **Long LOI:** Increase price by 15–20% for the extended survey duration.")
            loi_adjustment = 17.5  # Midpoint of 15-20%
        elif loi > 10:
            md_lines.append("- **Medium LOI:** Increase by 5–10%.")
            loi_adjustment = 7.5   # Midpoint of 5-10%
        else:
            md_lines.append("- **Short LOI:** Minimal adjustment needed.")
            loi_adjustment = 0
        
        # Discount based on sample size
        if completes > 1000:
            md_lines.append("- **Large Sample:** Apply a 15–20% discount due to economies of scale.")
            completes_adjustment = -17.5  # Negative for discount
        elif completes > 500:
            md_lines.append("- **Medium Sample:** Apply a 10–15% discount.")
            completes_adjustment = -12.5
        elif completes > 100:
            md_lines.append("- **Small Sample:** Apply a 5–10% discount.")
            completes_adjustment = -7.5
        else:
            md_lines.append("- **No Sample Discount:** Sample size is too low.")
            completes_adjustment = 0
        
        total_adjustment = ir_adjustment + loi_adjustment + completes_adjustment
        adjusted_cpi = predicted_cpi * (1 + total_adjustment / 100)
        
        md_lines.append("")
        md_lines.append(f"**Net Adjustment:** {total_adjustment:+.1f}%")
        md_lines.append(f"**Base CPI:** ${predicted_cpi:.2f}")
        md_lines.append(f"**Recommended CPI:** ${adjusted_cpi:.2f}")
        
        # Compare to similar projects
        if not similar_won.empty and not similar_lost.empty:
            won_avg = similar_won['CPI'].mean()
            lost_avg = similar_lost['CPI'].mean()
            
            if adjusted_cpi <= won_avg:
                md_lines.append(f"- The recommended CPI (${adjusted_cpi:.2f}) is below the average won CPI (${won_avg:.2f}), implying a highly competitive price.")
            elif adjusted_cpi <= (won_avg + lost_avg) / 2:
                md_lines.append(f"- The recommended CPI (${adjusted_cpi:.2f}) is moderate between the averages of won (${won_avg:.2f}) and lost (${lost_avg:.2f}) bids.")
            else:
                md_lines.append(f"- The recommended CPI (${adjusted_cpi:.2f}) is near the lost bids average (${lost_avg:.2f}), which may risk lower win rates.")
        
        # Create HTML version with high-contrast styling
        html_strategy = f"""
        <div style="
            background-color: #1F1F1F;
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid {COLOR_SYSTEM['ACCENT']['PURPLE']};
            color: #FFFFFF;
        ">
            <h3 style="color: #FFFFFF; margin-top: 0;">Detailed Pricing Strategy</h3>
            <p style="color: #B0B0B0; margin-bottom: 1.5rem;">
                IR: <span style="color: #FFFFFF; font-weight: 600;">{ir}%</span> | 
                LOI: <span style="color: #FFFFFF; font-weight: 600;">{loi} min</span> | 
                Completes: <span style="color: #FFFFFF; font-weight: 600;">{completes}</span>
            </p>
            
            <div style="display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 1.5rem;">
        """
        
        # Similar projects in HTML
        if not similar_won.empty:
            html_strategy += f"""
                <div style="flex: 1; min-width: 200px; background-color: #2E2E2E; padding: 1rem; border-radius: 0.3rem; border-left: 3px solid {COLOR_SYSTEM['ACCENT']['BLUE']};">
                    <h4 style="margin-top: 0; color: #FFFFFF;">Similar Won Projects</h4>
                    <p style="color: #B0B0B0; margin-bottom: 0.5rem;">n = {len(similar_won)}</p>
                    <div style="font-size: 1.4rem; color: {COLOR_SYSTEM['ACCENT']['BLUE']}; font-weight: 600; margin-bottom: 0.5rem;">
                        ${similar_won['CPI'].mean():.2f}
                    </div>
                    <p style="color: #B0B0B0; margin: 0;">Range: ${similar_won['CPI'].min():.2f} - ${similar_won['CPI'].max():.2f}</p>
                </div>
            """
        
        if not similar_lost.empty:
            html_strategy += f"""
                <div style="flex: 1; min-width: 200px; background-color: #2E2E2E; padding: 1rem; border-radius: 0.3rem; border-left: 3px solid {COLOR_SYSTEM['ACCENT']['ORANGE']};">
                    <h4 style="margin-top: 0; color: #FFFFFF;">Similar Lost Projects</h4>
                    <p style="color: #B0B0B0; margin-bottom: 0.5rem;">n = {len(similar_lost)}</p>
                    <div style="font-size: 1.4rem; color: {COLOR_SYSTEM['ACCENT']['ORANGE']}; font-weight: 600; margin-bottom: 0.5rem;">
                        ${similar_lost['CPI'].mean():.2f}
                    </div>
                    <p style="color: #B0B0B0; margin: 0;">Range: ${similar_lost['CPI'].min():.2f} - ${similar_lost['CPI'].max():.2f}</p>
                </div>
            """
        
        html_strategy += """
            </div>
            
            <h4 style="color: #FFFFFF; margin-bottom: 1rem;">Recommended Pricing Adjustments</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;">
        """
        
        # IR adjustment in HTML
        adj_color = COLOR_SYSTEM['ACCENT']['RED'] if ir_adjustment < 0 else COLOR_SYSTEM['ACCENT']['GREEN']
        html_strategy += f"""
                <div style="background-color: #2E2E2E; padding: 1rem; border-radius: 0.3rem;">
                    <h5 style="margin-top: 0; color: #FFFFFF;">IR Adjustment</h5>
                    <div style="font-size: 1.2rem; color: {adj_color}; font-weight: 600;">
                        {ir_adjustment:+.1f}%
                    </div>
                    <p style="color: #B0B0B0; margin: 0; font-size: 0.9rem;">
                        {'Low' if ir < 20 else 'Medium' if ir < 50 else 'High'} IR: {ir}%
                    </p>
                </div>
        """
        
        # LOI adjustment in HTML
        adj_color = COLOR_SYSTEM['ACCENT']['RED'] if loi_adjustment < 0 else COLOR_SYSTEM['ACCENT']['GREEN']
        html_strategy += f"""
                <div style="background-color: #2E2E2E; padding: 1rem; border-radius: 0.3rem;">
                    <h5 style="margin-top: 0; color: #FFFFFF;">LOI Adjustment</h5>
                    <div style="font-size: 1.2rem; color: {adj_color}; font-weight: 600;">
                        {loi_adjustment:+.1f}%
                    </div>
                    <p style="color: #B0B0B0; margin: 0; font-size: 0.9rem;">
                        {'Long' if loi > 20 else 'Medium' if loi > 10 else 'Short'} LOI: {loi} min
                    </p>
                </div>
        """
        
        # Completes adjustment in HTML
        adj_color = COLOR_SYSTEM['ACCENT']['RED'] if completes_adjustment < 0 else COLOR_SYSTEM['ACCENT']['GREEN']
        html_strategy += f"""
                <div style="background-color: #2E2E2E; padding: 1rem; border-radius: 0.3rem;">
                    <h5 style="margin-top: 0; color: #FFFFFF;">Sample Adjustment</h5>
                    <div style="font-size: 1.2rem; color: {adj_color}; font-weight: 600;">
                        {completes_adjustment:+.1f}%
                    </div>
                    <p style="color: #B0B0B0; margin: 0; font-size: 0.9rem;">
                        {'Large' if completes > 1000 else 'Medium' if completes > 500 else 'Small' if completes > 100 else 'Minimal'} sample: {completes}
                    </p>
                </div>
        """
        
        # Final pricing in HTML
        adj_color = COLOR_SYSTEM['ACCENT']['RED'] if total_adjustment < 0 else COLOR_SYSTEM['ACCENT']['GREEN']
        html_strategy += f"""
            </div>
            
            <div style="
                background-color: #2E2E2E;
                padding: 1.5rem;
                border-radius: 0.3rem;
                margin-top: 1rem;
                display: flex;
                align-items: center;
                justify-content: space-between;
            ">
                <div>
                    <div style="display: flex; align-items: baseline; gap: 0.5rem; margin-bottom: 0.5rem;">
                        <h4 style="margin: 0; color: #FFFFFF;">Base CPI:</h4>
                        <span style="font-size: 1.1rem; color: #FFFFFF;">${predicted_cpi:.2f}</span>
                    </div>
                    <div style="display: flex; align-items: baseline; gap: 0.5rem; margin-bottom: 0.5rem;">
                        <h4 style="margin: 0; color: #FFFFFF;">Net Adjustment:</h4>
                        <span style="font-size: 1.1rem; color: {adj_color};">{total_adjustment:+.1f}%</span>
                    </div>
                </div>
                <div style="
                    background-color: #1F1F1F;
                    padding: 1rem 1.5rem;
                    border-radius: 0.3rem;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                ">
                    <span style="font-size: 0.9rem; color: #B0B0B0; margin-bottom: 0.3rem;">Recommended CPI</span>
                    <span style="font-size: 1.8rem; font-weight: 700; color: {COLOR_SYSTEM['ACCENT']['BLUE']};">
                        ${adjusted_cpi:.2f}
                    </span>
                </div>
            </div>
        </div>
        """
        
        return {
            "markdown": "\n".join(md_lines),
            "html": html_strategy,
            "adjusted_cpi": adjusted_cpi,
            "base_cpi": predicted_cpi,
            "total_adjustment": total_adjustment,
            "adjustments": {
                "ir": ir_adjustment,
                "loi": loi_adjustment,
                "completes": completes_adjustment
            }
        }
    
    except Exception as e:
        logger.error(f"Error in get_detailed_pricing_strategy: {e}", exc_info=True)
        error_md = "Unable to generate detailed pricing strategy due to an error."
        error_html = f"<div style='color: {COLOR_SYSTEM['ACCENT']['RED']};'>{error_md}</div>"
        return {"markdown": error_md, "html": error_html}

def simulate_win_probability(predicted_cpi: float, user_input: Dict[str, float],
                          won_data: pd.DataFrame, lost_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Simulate the win probability based on the predicted CPI by comparing it within the CPI distribution.
    
    Args:
        predicted_cpi (float): Predicted CPI value.
        user_input (Dict[str, float]): Dictionary of user inputs.
        won_data (pd.DataFrame): DataFrame of won bids.
        lost_data (pd.DataFrame): DataFrame of lost bids.
        
    Returns:
        Dict[str, Any]: Dictionary containing the CPI percentile, win probability, and visualization data.
    """
    try:
        # Combine won and lost data, assigning a "Won" flag
        combined_data = pd.concat([
            won_data.assign(Won=1),
            lost_data.assign(Won=0)
        ], ignore_index=True)
        
        percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        cpi_percentiles = {p: combined_data['CPI'].quantile(p / 100) for p in percentiles}
        
        # Find which percentile bracket the predicted CPI falls into
        cpi_percentile = 0
        for p in sorted(percentiles):
            if predicted_cpi <= cpi_percentiles[p]:
                cpi_percentile = p
                break
        
        if cpi_percentile == 0:
            cpi_percentile = 100  # If above all percentiles
        
        # Calculate win rates by percentile bracket
        wins_by_percentile = {}
        for p in percentiles:
            subset = combined_data[combined_data['CPI'] <= combined_data['CPI'].quantile(p / 100)]
            wins_by_percentile[p] = subset['Won'].mean() * 100 if len(subset) > 0 else 0
        
        # Get win probability corresponding to the predicted CPI's percentile
        win_probability = wins_by_percentile.get(cpi_percentile, min(wins_by_percentile.values()))
        
        # Create visualization-friendly data with high-contrast colors
        viz_data = {
            'percentiles': percentiles,
            'cpi_values': [cpi_percentiles[p] for p in percentiles],
            'win_rates': [wins_by_percentile[p] for p in percentiles],
            'colors': {
                'line': COLOR_SYSTEM["ACCENT"]["BLUE"],
                'point': COLOR_SYSTEM["ACCENT"]["GREEN"],
                'highlight': COLOR_SYSTEM["ACCENT"]["PURPLE"]
            }
        }
        
        # Enhanced output for high-contrast dashboard
        result = {
            'cpi_percentile': cpi_percentile,
            'win_probability': win_probability,
            'percentile_data': viz_data,
            # HTML component for direct rendering
            'html': f"""
            <div style="
                background-color: #1F1F1F;
                border-radius: 0.5rem;
                padding: 1.5rem;
                margin-bottom: 1rem;
                border-left: 4px solid {COLOR_SYSTEM['ACCENT']['GREEN']};
                color: #FFFFFF;
                text-align: center;
            ">
                <h3 style="margin-top: 0; margin-bottom: 1rem; color: #FFFFFF;">Win Probability Estimate</h3>
                <div style="font-size: 2.5rem; font-weight: 700; color: {COLOR_SYSTEM['ACCENT']['GREEN']};">
                    {win_probability:.1f}%
                </div>
                <p style="color: #B0B0B0; margin: 0.5rem 0;">
                    Your bid is in the <strong>{cpi_percentile}th</strong> percentile of all bids.
                </p>
                <div style="
                    background-color: #2E2E2E;
                    height: 0.5rem;
                    border-radius: 0.25rem;
                    margin: 1rem 0;
                    position: relative;
                ">
                    <div style="
                        position: absolute;
                        height: 0.5rem;
                        width: {cpi_percentile}%;
                        background-color: {COLOR_SYSTEM['ACCENT']['BLUE']};
                        border-radius: 0.25rem;
                    "></div>
                </div>
                <p style="color: #B0B0B0; margin: 0; font-size: 0.9rem;">
                    CPI: ${predicted_cpi:.2f}
                </p>
            </div>
            """
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error in simulate_win_probability: {e}", exc_info=True)
        return {
            'error': str(e),
            'html': f"""
            <div style="color: {COLOR_SYSTEM['ACCENT']['RED']}; padding: 1rem; text-align: center;">
                Unable to simulate win probability due to an error.
            </div>
            """
        }