def main():
    """Main application function with improved component isolation and error handling."""
    
    st.sidebar.title("CPI Analysis & Prediction Dashboard")
    
    # Navigation
    app_mode = st.sidebar.radio(
        "Choose a mode",
        ["Overview", "CPI Analysis", "CPI Prediction", "Insights & Recommendations"]
    )
    
    # Error container for displaying component errors
    error_container = st.container()
    
    # Loading state display
    with st.spinner("Loading data..."):
        try:
            # Use improved data loading function
            data_result = load_data()
            
            if data_result['status'] == 'error':
                for error in data_result['errors']:
                    error_container.error(f"Data Loading Error: {error}")
                st.stop()
                
            # Display warnings if any
            if 'warnings' in data_result and data_result['warnings']:
                with st.expander("Data Loading Warnings", expanded=False):
                    for warning in data_result['warnings']:
                        st.warning(warning)
            
            # Extract datasets from result
            won_df = data_result['won']
            won_df_filtered = data_result['won_filtered']
            lost_df = data_result['lost']
            lost_df_filtered = data_result['lost_filtered']
            combined_df = data_result['combined']
            combined_df_filtered = data_result['combined_filtered']
            
            logger.info(f"Data loaded: Won deals: {len(won_df)}, Lost deals: {len(lost_df)}")
            
        except Exception as e:
            error_container.error(f"Failed to load data: {str(e)}")
            st.error("Please check that all required data files exist and are in the correct format.")
            st.stop()
    
    # Apply binning to all dataframes with error handling
    try:
        with st.spinner("Processing data..."):
            won_df = apply_all_bins(won_df)
            won_df_filtered = apply_all_bins(won_df_filtered)
            lost_df = apply_all_bins(lost_df)
            lost_df_filtered = apply_all_bins(lost_df_filtered)
            combined_df = apply_all_bins(combined_df)
            combined_df_filtered = apply_all_bins(combined_df_filtered)
            logger.info("Successfully applied binning to all datasets")
    except Exception as e:
        error_container.error(f"Error processing data: {str(e)}")
        logger.error(f"Error applying binning: {str(e)}")
        # Continue with unmodified dataframes instead of stopping
    
    # Sidebar filtering options
    st.sidebar.markdown("---")
    st.sidebar.title("Filtering Options")
    
    show_filtered = st.sidebar.checkbox(
        "Filter out extreme values (>95th percentile)",
        value=True,
        help="Remove outliers with very high CPI values to focus on typical cases"
    )
    
    # Choose datasets based on filtering option
    if show_filtered:
        won_data = won_df_filtered
        lost_data = lost_df_filtered
        combined_data = combined_df_filtered
    else:
        won_data = won_df
        lost_data = lost_df
        combined_data = combined_df
    
    # Component-specific session state to track errors
    if 'component_errors' not in st.session_state:
        st.session_state.component_errors = {
            'overview': [],
            'analysis': [],
            'prediction': [],
            'insights': []
        }
    
    # Overview mode
    if app_mode == "Overview":
        st.title("CPI Analysis Dashboard: Overview")
        
        try:
            # Component description
            st.markdown("""
            This dashboard analyzes the Cost Per Impression (CPI) between won and lost bids 
            to identify meaningful differences. The three main factors that influence CPI are:
            - **IR (Incidence Rate)**: The percentage of people who qualify for a survey
            - **LOI (Length of Interview)**: How long the survey takes to complete
            - **Sample Size (Completes)**: The number of completed surveys
            
            Use the navigation menu on the left to explore different analyses and tools.
            """)
            
            # Key metrics with error handling
            st.header("Key Metrics")
            try:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Safely get average CPI with fallback
                    try:
                        won_avg_cpi = won_data['CPI'].mean()
                        lost_avg_cpi = lost_data['CPI'].mean()
                        cpi_diff = lost_avg_cpi - won_avg_cpi
                        
                        st.metric("Average CPI - Won", f"${won_avg_cpi:.2f}")
                        st.metric("Average CPI - Lost", f"${lost_avg_cpi:.2f}")
                        st.metric("CPI Difference", f"${cpi_diff:.2f}")
                    except Exception as e:
                        st.error(f"Unable to calculate CPI metrics: {str(e)}")
                
                with col2:
                    # Safely get average IR with fallback
                    try:
                        won_avg_ir = won_data['IR'].mean()
                        lost_avg_ir = lost_data['IR'].mean()
                        ir_diff = lost_avg_ir - won_avg_ir
                        
                        st.metric("Average IR - Won", f"{won_avg_ir:.2f}%")
                        st.metric("Average IR - Lost", f"{lost_avg_ir:.2f}%")
                        st.metric("IR Difference", f"{ir_diff:.2f}%")
                    except Exception as e:
                        st.error(f"Unable to calculate IR metrics: {str(e)}")
                
                with col3:
                    # Safely get average LOI with fallback
                    try:
                        won_avg_loi = won_data['LOI'].mean()
                        lost_avg_loi = lost_data['LOI'].mean()
                        loi_diff = lost_avg_loi - won_avg_loi
                        
                        st.metric("Average LOI - Won", f"{won_avg_loi:.2f} min")
                        st.metric("Average LOI - Lost", f"{lost_avg_loi:.2f} min")
                        st.metric("LOI Difference", f"{loi_diff:.2f} min")
                    except Exception as e:
                        st.error(f"Unable to calculate LOI metrics: {str(e)}")
            except Exception as e:
                st.error(f"Error displaying metrics: {str(e)}")
            
            # Overview charts with error handling
            st.header("CPI Distribution Comparison")
            try:
                fig = safe_create_cpi_distribution_boxplot(won_data, lost_data)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating distribution boxplot: {str(e)}")
                logger.error(f"Error in boxplot visualization: {str(e)}")
            
            try:
                fig = safe_create_cpi_histogram_comparison(won_data, lost_data)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating histogram comparison: {str(e)}")
                logger.error(f"Error in histogram visualization: {str(e)}")
                
        except Exception as e:
            st.error(f"Error in Overview component: {str(e)}")
            st.session_state.component_errors['overview'].append(str(e))
            logger.error(f"Overview component error: {str(e)}")
    
    # CPI Analysis mode
    elif app_mode == "CPI Analysis":
        st.title("CPI Analysis: Won vs. Lost Bids")
        
        try:
            # Component description
            st.markdown("""
            This section examines relationships between CPI and key project parameters:
            - Incidence Rate (IR)
            - Length of Interview (LOI)
            - Sample Size (Completes)
            
            These visualizations help identify patterns that affect pricing strategies.
            """)
            
            # Create tabs for different analyses
            tab1, tab2, tab3 = st.tabs(["IR Analysis", "LOI Analysis", "Sample Size Analysis"])
            
            with tab1:
                st.header("CPI Analysis by Incidence Rate (IR)")
                
                try:
                    # CPI vs IR bar chart by bin
                    fig = safe_create_bar_chart_by_bin(
                        won_data, 
                        lost_data, 
                        bin_column='IR_Bin', 
                        title="Average CPI by Incidence Rate (IR) Range"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating IR analysis chart: {str(e)}")
                    logger.error(f"Error in IR bin visualization: {str(e)}")
                
                # IR Analysis insight
                st.markdown("""
                #### IR Impact Insight
                
                Low Incidence Rate (IR) typically results in higher CPI due to increased difficulty
                in finding qualified respondents. The gap between Won and Lost bids tends to be
                largest in the lower IR ranges, suggesting pricing sensitivity is highest there.
                """)
            
            with tab2:
                st.header("CPI Analysis by Length of Interview (LOI)")
                
                try:
                    # CPI vs LOI bar chart by bin
                    fig = safe_create_bar_chart_by_bin(
                        won_data, 
                        lost_data, 
                        bin_column='LOI_Bin', 
                        title="Average CPI by Length of Interview (LOI) Range"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating LOI analysis chart: {str(e)}")
                    logger.error(f"Error in LOI bin visualization: {str(e)}")
                
                # LOI Analysis insight
                st.markdown("""
                #### LOI Impact Insight
                
                Longer surveys (higher LOI) generally command higher prices as respondents must be 
                compensated for their time. The difference between Won and Lost bids increases with 
                LOI, indicating that competitive pricing becomes more critical for longer surveys.
                """)
            
            with tab3:
                st.header("CPI Analysis by Sample Size (Completes)")
                
                try:
                    # CPI vs Completes bar chart by bin
                    fig = safe_create_bar_chart_by_bin(
                        won_data, 
                        lost_data, 
                        bin_column='Completes_Bin', 
                        title="Average CPI by Sample Size Range"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating Sample Size analysis chart: {str(e)}")
                    logger.error(f"Error in Completes bin visualization: {str(e)}")
                
                # Sample Size Analysis insight
                st.markdown("""
                #### Sample Size Impact Insight
                
                Larger sample sizes tend to benefit from economies of scale, resulting in lower 
                per-unit costs. Implementing a tiered discount structure for larger projects can 
                improve competitiveness while maintaining profitability.
                """)
            
        except Exception as e:
            st.error(f"Error in Analysis component: {str(e)}")
            st.session_state.component_errors['analysis'].append(str(e))
            logger.error(f"Analysis component error: {str(e)}")
    
    # CPI Prediction mode
    elif app_mode == "CPI Prediction":
        st.title("CPI Prediction Model")
        
        try:
            # Component description
            st.markdown("""
            This tool uses machine learning to predict the optimal CPI (Cost Per Interview) based on:
            - **IR (Incidence Rate)**: The percentage of people who qualify for a survey
            - **LOI (Length of Interview)**: How long the survey takes in minutes
            - **Sample Size**: The number of completed interviews
            
            Enter your project parameters to receive a CPI prediction and pricing recommendation.
            """)
            
            # Build models
            with st.spinner("Training prediction models..."):
                try:
                    # Apply feature engineering to the data
                    combined_data_engineered = engineer_features(combined_data)
                    
                    # Build models with improved error handling
                    model_result = build_models(combined_data_engineered)
                    
                    if model_result['status'] == 'error':
                        for error in model_result['errors']:
                            st.error(f"Model Training Error: {error}")
                        st.warning("Prediction functionality is not available due to model training errors.")
                        has_models = False
                    else:
                        has_models = True
                        
                        # Display any warnings
                        if model_result['warnings']:
                            with st.expander("Model Training Warnings", expanded=False):
                                for warning in model_result['warnings']:
                                    st.warning(warning)
                        
                        # Extract models and feature importance
                        models = model_result['trained_models']
                        model_scores = model_result['model_scores']
                        feature_importance = model_result['feature_importance']
                        
                        # Get feature names for prediction
                        X, _ = prepare_model_data(combined_data_engineered)
                        feature_names = X.columns.tolist()
                except Exception as e:
                    st.error(f"Failed to train prediction models: {str(e)}")
                    logger.error(f"Model training error: {str(e)}")
                    has_models = False
            
            # User input form
            st.header("Enter Project Specifications")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                ir = st.slider("Incidence Rate (%)", 1, 100, 50)
            
            with col2:
                loi = st.slider("Length of Interview (min)", 1, 60, 15)
            
            with col3:
                completes = st.slider("Sample Size (Completes)", 10, 2000, 500)
            
            user_input = {
                'IR': ir,
                'LOI': loi,
                'Completes': completes
            }
            
            # Prediction button
            predict_button = st.button("Predict CPI")
            
            if predict_button and has_models:
                with st.spinner("Generating prediction..."):
                    try:
                        # Make prediction with improved error handling
                        prediction_result = predict_cpi(models, user_input, feature_names)
                        
                        if prediction_result['status'] == 'error':
                            for error in prediction_result['errors']:
                                st.error(f"Prediction Error: {error}")
                        else:
                            # Display any warnings
                            if prediction_result['warnings']:
                                with st.expander("Prediction Warnings", expanded=False):
                                    for warning in prediction_result['warnings']:
                                        st.warning(warning)
                            
                            # Display predictions
                            st.subheader("CPI Predictions")
                            
                            if prediction_result['predictions']:
                                col1, col2, col3 = st.columns(3)
                                
                                model_names = list(prediction_result['predictions'].keys())
                                if 'Average' in model_names:
                                    model_names.remove('Average')
                                    
                                # Display individual model predictions
                                for i, name in enumerate(model_names[:3]):  # Show first 3 models
                                    with [col1, col2, col3][i % 3]:
                                        st.metric(name, f"${prediction_result['predictions'][name]:.2f}")
                                
                                # Display average prediction
                                if 'Average' in prediction_result['predictions']:
                                    avg_prediction = prediction_result['predictions']['Average']
                                    st.metric("Average Prediction", f"${avg_prediction:.2f}")
                                    
                                    # Get average CPIs for reference
                                    won_avg = won_data['CPI'].mean()
                                    lost_avg = lost_data['CPI'].mean()
                                    
                                    # Generate recommendation
                                    recommendation = get_recommendation(avg_prediction, won_avg, lost_avg)
                                    
                                    st.markdown(f"""
                                    **Comparison:**
                                    - Average CPI for Won bids: ${won_avg:.2f}
                                    - Average CPI for Lost bids: ${lost_avg:.2f}
                                    - Your predicted CPI: ${avg_prediction:.2f}
                                    
                                    **Recommendation:**
                                    {recommendation['recommendation']}
                                    """)
                                    
                                    # Display feature importance if available
                                    if feature_importance is not None:
                                        st.subheader("Feature Importance Analysis")
                                        try:
                                            fig = create_feature_importance_chart(feature_importance)
                                            st.plotly_chart(fig, use_container_width=True)
                                        except Exception as e:
                                            st.error(f"Error displaying feature importance: {str(e)}")
                            else:
                                st.warning("No valid predictions were generated.")
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
                        logger.error(f"Prediction error: {str(e)}")
            elif predict_button and not has_models:
                st.error("Cannot make predictions because models could not be trained. Please check data quality.")
            
        except Exception as e:
            st.error(f"Error in Prediction component: {str(e)}")
            st.session_state.component_errors['prediction'].append(str(e))
            logger.error(f"Prediction component error: {str(e)}")
    
    # Insights & Recommendations mode
    elif app_mode == "Insights & Recommendations":
        st.title("Insights & Recommendations")
        
        try:
            # Component description
            st.markdown("""
            Based on the analysis of CPI data between won and lost bids, this section provides
            strategic insights and actionable recommendations to optimize your pricing strategy.
            """)
            
            # Key insights 
            st.header("Key Findings")
            
            try:
                # Calculate summary statistics safely
                won_avg_cpi = won_data['CPI'].mean() if 'CPI' in won_data.columns else None
                lost_avg_cpi = lost_data['CPI'].mean() if 'CPI' in lost_data.columns else None
                
                if won_avg_cpi is not None and lost_avg_cpi is not None:
                    st.markdown(f"""
                    1. **Overall CPI Difference**: There is a gap of ${lost_avg_cpi - won_avg_cpi:.2f} between 
                       the average CPI for won bids (${won_avg_cpi:.2f}) and lost bids (${lost_avg_cpi:.2f}).
                       This suggests that pricing is a critical factor in bid success.
                       
                    2. **IR Impact**: Lower IR values generally correlate with higher CPIs, as it becomes 
                       more difficult to find qualified respondents. Lost bids tend to have higher CPIs at all IR levels,
                       but the difference is most pronounced at lower IR levels.
                       
                    3. **LOI Impact**: As LOI increases, CPI tends to increase for both won and lost bids.
                       However, lost bids show a steeper increase in CPI as LOI gets longer, suggesting that
                       pricing for longer surveys may be a key differentiator.
                       
                    4. **Sample Size Effect**: Larger sample sizes tend to have lower per-unit CPIs
                       due to economies of scale. Lost bids often don't sufficiently account for this scaling effect.
                    """)
                else:
                    st.warning("Unable to calculate complete insights due to missing data.")
                    
            except Exception as e:
                st.error(f"Error calculating insights: {str(e)}")
                logger.error(f"Error in insights calculation: {str(e)}")
            
            # Recommendations
            st.header("Recommendations for Pricing Strategy")
            
            try:
                # Calculate pricing tiers safely
                low_ir_price = None
                med_ir_price = None
                high_ir_price = None
                
                try:
                    low_ir_lost = lost_data[lost_data['IR'] <= 20]['CPI'].quantile(0.25)
                    med_ir_lost = lost_data[(lost_data['IR'] > 20) & (lost_data['IR'] <= 50)]['CPI'].quantile(0.25)
                    high_ir_lost = lost_data[lost_data['IR'] > 50]['CPI'].quantile(0.25)
                    
                    # Only set values if calculations succeed
                    if not pd.isna(low_ir_lost):
                        low_ir_price = low_ir_lost
                    if not pd.isna(med_ir_lost):
                        med_ir_price = med_ir_lost
                    if not pd.isna(high_ir_lost):
                        high_ir_price = high_ir_lost
                except Exception as e:
                    logger.error(f"Error calculating price tiers: {str(e)}")
                
                st.markdown("""
                Based on our analysis, we recommend the following pricing strategies to improve bid success rates:
                
                1. **IR-Based Pricing Tiers**: Implement a clear pricing structure based on IR ranges, with higher prices
                   for lower IR projects.
                """)
                
                if low_ir_price is not None and med_ir_price is not None and high_ir_price is not None:
                    st.markdown(f"""
                   Our analysis suggests the following price adjustments for different IR ranges:
                   - Low IR (0-20%): Keep CPIs below ${low_ir_price:.2f} 
                   - Medium IR (21-50%): Keep CPIs below ${med_ir_price:.2f}
                   - High IR (51-100%): Keep CPIs below ${high_ir_price:.2f}
                    """)
                
                st.markdown("""
                2. **LOI Multipliers**: Apply multipliers to the base CPI based on LOI:
                   - Short LOI (1-10 min): Base CPI
                   - Medium LOI (11-20 min): Base CPI × 1.3
                   - Long LOI (21+ min): Base CPI × 1.5
                   
                3. **Sample Size Discounts**: Implement volume discounts for larger projects:
                   - Small (1-100 completes): Standard CPI
                   - Medium (101-500 completes): 5% discount
                   - Large (501-1000 completes): 10% discount
                   - Very Large (1000+ completes): 15% discount
                   
                4. **Combined Factor Pricing Model**: Use a prediction model to optimize pricing for different
                   combinations of IR, LOI, and sample size. This approach can help provide competitive yet
                   profitable pricing.
                """)
            
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
                logger.error(f"Error in recommendations: {str(e)}")
            
        except Exception as e:
            st.error(f"Error in Insights component: {str(e)}")
            st.session_state.component_errors['insights'].append(str(e))
            logger.error(f"Insights component error: {str(e)}")
    
    # Display any accumulated errors in the session state
    if any(errors for errors in st.session_state.component_errors.values()):
        with st.expander("View Component Errors", expanded=False):
            for component, errors in st.session_state.component_errors.items():
                if errors:
                    st.subheader(f"{component.title()} Component Errors")
                    for i, error in enumerate(errors):
                        st.error(f"Error {i+1}: {error}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred in the application: {str(e)}")
        logger.error(f"Unexpected error in main application: {e}", exc_info=True)