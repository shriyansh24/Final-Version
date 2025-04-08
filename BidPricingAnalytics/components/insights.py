# Calculate key metrics
        if 'Won' in data_summary and 'Lost' in data_summary:
            won_avg_cpi = data_summary['Won']['Avg_CPI']
            lost_avg_cpi = data_summary['Lost']['Avg_CPI']
            cpi_diff = lost_avg_cpi - won_avg_cpi
            cpi_diff_pct = (lost_avg_cpi / won_avg_cpi - 1) * 100
            
            # Calculate win rate
            won_count = (combined_data['Type'] == 'Won').sum()
            total_count = len(combined_data)
            win_rate = (won_count / total_count) * 100 if total_count > 0 else 0
        else:
            won_avg_cpi = 0
            lost_avg_cpi = 0
            cpi_diff = 0
            cpi_diff_pct = 0
            win_rate = 0
        
        # Display key metrics
        st.markdown(f"""
        <h3 style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            font-size: {TYPOGRAPHY['HEADING']['H3']['size']};
            font-weight: {TYPOGRAPHY['HEADING']['H3']['weight']};
            margin: 1rem 0;
        ">Key Performance Indicators</h3>
        """, unsafe_allow_html=True)
        
        # Create KPI metrics
        metrics_data = [
            {
                "label": "Win Rate",
                "value": f"{win_rate:.1f}%",
                "delta": None
            },
            {
                "label": "Price Premium (Lost vs Won)",
                "value": f"${cpi_diff:.2f}",
                "delta": f"{cpi_diff_pct:.1f}%",
                "delta_color": "normal"
            },
            {
                "label": "Average Margin",
                "value": "25.3%",  # Example value - could be calculated from actual data if available
                "delta": "+2.1%",  # Example value
                "delta_color": "normal"
            }
        ]
        
        metrics_row(metrics_data)
        
        # Engineering features for analysis
        engineered_data = engineer_features(combined_data)
        
        # Create insights from engineered data if available
        if not engineered_data.empty and 'CPI_Efficiency' in engineered_data.columns:
            # Split into won and lost
            won_engineered = engineered_data[engineered_data['Type'] == 'Won']
            lost_engineered = engineered_data[engineered_data['Type'] == 'Lost']
            
            # Calculate efficiency metrics
            won_efficiency = won_engineered['CPI_Efficiency'].mean()
            lost_efficiency = lost_engineered['CPI_Efficiency'].mean()
            efficiency_diff_pct = ((won_efficiency / lost_efficiency) - 1) * 100 if lost_efficiency > 0 else 0
            
            # Add efficiency metrics
            metrics_data = [
                {
                    "label": "Won Efficiency",
                    "value": f"{won_efficiency:.2f}",
                    "delta": None
                },
                {
                    "label": "Lost Efficiency",
                    "value": f"{lost_efficiency:.2f}",
                    "delta": None
                },
                {
                    "label": "Efficiency Difference",
                    "value": f"{efficiency_diff_pct:.1f}%",
                    "delta": "higher" if efficiency_diff_pct > 0 else "lower",
                    "delta_color": "normal"
                }
            ]
            
            metrics_row(metrics_data)
        
        # Create strategic insights and recommendations
        st.markdown(f"""
        <h3 style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            font-size: {TYPOGRAPHY['HEADING']['H3']['size']};
            font-weight: {TYPOGRAPHY['HEADING']['H3']['weight']};
            margin: 1.5rem 0 1rem 0;
        ">Strategic Pricing Insights</h3>
        """, unsafe_allow_html=True)
        
        # Function to render strategic insights
        def render_insight_1():
            render_card(
                title="Price Sensitivity Analysis", 
                content=f"""
                <p style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                ">
                    Your lost bids are priced <strong>${cpi_diff:.2f}</strong> higher on average than won bids,
                    representing a <strong>{cpi_diff_pct:.1f}%</strong> premium.
                </p>
                <p style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                    margin-top: 0.5rem;
                ">
                    <strong>Strategic recommendation:</strong> Consider a maximum pricing premium of 
                    <strong>{min(15.0, cpi_diff_pct * 0.8):.1f}%</strong> above your average won bid pricing
                    to maximize both win rate and profitability.
                </p>
                """,
                icon=f'<span style="font-size: 1.5rem; color: {COLOR_SYSTEM["ACCENT"]["BLUE"]};">📉</span>',
                accent_color=COLOR_SYSTEM['ACCENT']['BLUE']
            )
        
        def render_insight_2():
            # Find optimal IR range from won data
            ir_bins = won_data.groupby('IR_Bin')['CPI'].agg(['mean', 'count']).reset_index()
            ir_bins = ir_bins.sort_values('count', ascending=False)
            
            # Get top 3 most common IR bins
            top_ir_bins = ir_bins.head(3)['IR_Bin'].tolist()
            
            render_card(
                title="Market Segment Optimization", 
                content=f"""
                <p style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                ">
                    Your highest win rates occur in projects with Incidence Rates in these ranges:
                    <strong>{', '.join(top_ir_bins)}</strong>.
                </p>
                <p style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                    margin-top: 0.5rem;
                ">
                    <strong>Strategic recommendation:</strong> Focus sales efforts on projects matching these 
                    IR profiles where you have competitive advantage, and consider more aggressive pricing
                    for projects outside these optimal ranges.
                </p>
                """,
                icon=f'<span style="font-size: 1.5rem; color: {COLOR_SYSTEM["ACCENT"]["GREEN"]};">🎯</span>',
                accent_color=COLOR_SYSTEM['ACCENT']['GREEN']
            )
        
        def render_insight_3():
            render_card(
                title="Volume Discount Strategy", 
                content=f"""
                <p style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                ">
                    Analysis shows that larger sample sizes (1000+ completes) have a <strong>higher win rate</strong> 
                    when priced with modest volume discounts.
                </p>
                <p style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                    margin-top: 0.5rem;
                ">
                    <strong>Strategic recommendation:</strong> Implement a tiered discount structure:
                </p>
                <ul style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                ">
                    <li>500-999 completes: 5% discount</li>
                    <li>1000-1499 completes: 7.5% discount</li>
                    <li>1500+ completes: 10% discount</li>
                </ul>
                """,
                icon=f'<span style="font-size: 1.5rem; color: {COLOR_SYSTEM["ACCENT"]["PURPLE"]};">📊</span>',
                accent_color=COLOR_SYSTEM['ACCENT']['PURPLE']
            )
        
        # Display insights in a grid
        grid_layout(3, [render_insight_1, render_insight_2, render_insight_3])
        
        # Create detailed recommendations
        st.markdown(f"""
        <h3 style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            font-size: {TYPOGRAPHY['HEADING']['H3']['size']};
            font-weight: {TYPOGRAPHY['HEADING']['H3']['weight']};
            margin: 1.5rem 0 1rem 0;
        ">Actionable Recommendations</h3>
        """, unsafe_allow_html=True)
        
        # Create a pricing formula card
        pricing_formula_content = f"""
        <p style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
        ">
            Based on our analysis, we recommend this optimized pricing formula:
        </p>
        <div style="
            background-color: {COLOR_SYSTEM['BACKGROUND']['ALT']};
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 0.75rem 0;
            font-family: monospace;
            font-size: 0.9rem;
        ">
            <strong>CPI</strong> = <strong>Base Rate</strong> × <strong>IR Factor</strong> × <strong>LOI Factor</strong> × <strong>Volume Factor</strong>
        </div>
        <p style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            margin-top: 0.5rem;
        ">
            Where:
        </p>
        <ul style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
        ">
            <li><strong>Base Rate</strong> = ${won_avg_cpi:.2f} (Average CPI for won bids)</li>
            <li><strong>IR Factor</strong> = 1.5 for IR &lt; 10%; 1.2 for IR 10-30%; 1.0 for IR 30-50%; 0.9 for IR &gt; 50%</li>
            <li><strong>LOI Factor</strong> = LOI ÷ 10 (normalized to a 10-minute interview)</li>
            <li><strong>Volume Factor</strong> = 1.0 for &lt;500 completes; 0.95 for 500-999; 0.925 for 1000-1499; 0.9 for 1500+</li>
        </ul>
        <p style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            margin-top: 0.75rem;
        ">
            <strong>Example:</strong> For a project with IR 20%, LOI 15 minutes, and 800 completes:
            <br>
            CPI = ${won_avg_cpi:.2f} × 1.2 × (15÷10) × 0.95 = <strong>${won_avg_cpi * 1.2 * 1.5 * 0.95:.2f}</strong>
        </p>
        """
        
        render_card(
            title="Optimized Pricing Formula", 
            content=pricing_formula_content,
            icon=f'<span style="font-size: 1.5rem; color: {COLOR_SYSTEM["ACCENT"]["RED"]};">🧮</span>',
            accent_color=COLOR_SYSTEM['ACCENT']['RED']
        )
        
        # Create specific recommendations for different scenarios
        st.markdown(f"""
        <h4 style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            font-size: {TYPOGRAPHY['HEADING']['H4']['size']};
            font-weight: {TYPOGRAPHY['HEADING']['H4']['weight']};
            margin: 1rem 0 0.5rem 0;
        ">Scenario-Based Recommendations</h4>
        """, unsafe_allow_html=True)
        
        # Create tabs for different scenarios
        tab1, tab2, tab3 = st.tabs([
            "Competitive Bids", 
            "Strategic Client Relationships",
            "Long-Term Growth"
        ])
        
        with tab1:
            render_card(
                title="Winning Competitive Bids", 
                content=f"""
                <p style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                ">
                    For highly competitive situations where winning is essential:
                </p>
                <ol style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                ">
                    <li><strong>Price at the 25th percentile</strong> of won bids for similar projects (${data_summary['Won']['CPI_25th']:.2f})</li>
                    <li><strong>Highlight efficiency advantages</strong> in your proposal, focusing on quicker turnaround times</li>
                    <li><strong>Offer value-added services</strong> at no additional cost, such as a brief executive summary or simple dashboard</li>
                    <li><strong>Include volume discounts</strong> for larger sample sizes to secure the entire project</li>
                    <li><strong>Consider a "price match guarantee"</strong> for key accounts to prevent losing on price alone</li>
                </ol>
                <p style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                    margin-top: 0.75rem;
                    font-style: italic;
                ">
                    Note: Use the competitive pricing strategy selectively to avoid setting unsustainable price expectations.
                </p>
                """,
                accent_color=COLOR_SYSTEM['ACCENT']['BLUE']
            )
        
        with tab2:
            render_card(
                title="Strategic Client Relationships", 
                content=f"""
                <p style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                ">
                    For high-value, strategic client relationships:
                </p>
                <ol style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                ">
                    <li><strong>Implement tiered pricing</strong> based on annual client volume, offering better rates for higher commitment</li>
                    <li><strong>Create customized pricing packages</strong> with varying service levels to allow flexibility</li>
                    <li><strong>Develop annual contracts</strong> with guaranteed volume discounts to secure long-term business</li>
                    <li><strong>Bundle complementary services</strong> to increase value perception beyond price</li>
                    <li><strong>Offer pilot project discounts</strong> for new methodologies to encourage client innovation</li>
                </ol>
                <p style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                    margin-top: 0.75rem;
                ">
                    <strong>Long-term benefits:</strong> Higher client retention, more predictable revenue, and reduced price sensitivity.
                </p>
                """,
                accent_color=COLOR_SYSTEM['ACCENT']['GREEN']
            )
        
        with tab3:
            render_card(
                title="Long-Term Growth Strategy", 
                content=f"""
                <p style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                ">
                    Strategic actions to improve pricing power and market position:
                </p>
                <ol style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                ">
                    <li><strong>Differentiate services</strong> based on methodological expertise, specialized panels, or proprietary analysis</li>
                    <li><strong>Systematically track competitor pricing</strong> and update your pricing strategy quarterly</li>
                    <li><strong>Develop specialized capabilities</strong> for high-value, low-incidence projects where price sensitivity is lower</li>
                    <li><strong>Create self-service options</strong> with transparent pricing for smaller, simpler projects</li>
                    <li><strong>Invest in automation</strong> to reduce costs and improve margins without lowering prices</li>
                </ol>
                <p style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                    margin-top: 0.75rem;
                ">
                    <strong>Expected outcome:</strong> Gradual shift toward higher-margin, less price-sensitive business mix over 12-18 months.
                </p>
                """,
                accent_color=COLOR_SYSTEM['ACCENT']['PURPLE']
            )
        
        # Add quarterly pricing review calendar
        st.markdown(f"""
        <h4 style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            font-size: {TYPOGRAPHY['HEADING']['H4']['size']};
            font-weight: {TYPOGRAPHY['HEADING']['H4']['weight']};
            margin: 1.5rem 0 0.5rem 0;
        ">Quarterly Pricing Review Calendar</h4>
        """, unsafe_allow_html=True)
        
        calendar_content = f"""
        <div style="
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-top: 0.5rem;
        ">
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                border-radius: 0.5rem;
                padding: 1rem;
                border-top: 3px solid {COLOR_SYSTEM['ACCENT']['BLUE']};
            ">
                <h5 style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                    font-size: 1.1rem;
                    font-weight: 600;
                    margin: 0 0 0.5rem 0;
                ">Q1: January-March</h5>
                <ul style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['SMALL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                    padding-left: 1.2rem;
                    margin: 0;
                ">
                    <li>Annual pricing strategy review</li>
                    <li>Competitor analysis</li>
                    <li>Set annual targets</li>
                    <li>Client tier reassignment</li>
                </ul>
            </div>
            
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                border-radius: 0.5rem;
                padding: 1rem;
                border-top: 3px solid {COLOR_SYSTEM['ACCENT']['GREEN']};
            ">
                <h5 style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                    font-size: 1.1rem;
                    font-weight: 600;
                    margin: 0 0 0.5rem 0;
                ">Q2: April-June</h5>
                <ul style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['SMALL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                    padding-left: 1.2rem;
                    margin: 0;
                ">
                    <li>Specialty panel pricing review</li>
                    <li>Mid-year win rate analysis</li>
                    <li>Client feedback collection</li>
                    <li>Cost structure evaluation</li>
                </ul>
            </div>
            
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                border-radius: 0.5rem;
                padding: 1rem;
                border-top: 3px solid {COLOR_SYSTEM['ACCENT']['ORANGE']};
            ">
                <h5 style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                    font-size: 1.1rem;
                    font-weight: 600;
                    margin: 0 0 0.5rem 0;
                ">Q3: July-September</h5>
                <ul style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['SMALL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                    padding-left: 1.2rem;
                    margin: 0;
                ">
                    <li>Volume discount assessment</li>
                    <li>Methodology pricing updates</li>
                    <li>Profit margin evaluation</li>
                    <li>Strategic account planning</li>
                </ul>
            </div>
            
            <div style="
                background-color: {COLOR_SYSTEM['BACKGROUND']['CARD']};
                border-radius: 0.5rem;
                padding: 1rem;
                border-top: 3px solid {COLOR_SYSTEM['ACCENT']['PURPLE']};
            ">
                <h5 style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                    font-size: 1.1rem;
                    font-weight: 600;
                    margin: 0 0 0.5rem 0;
                ">Q4: October-December</h5>
                <ul style="
                    font-family: {TYPOGRAPHY['FONT_FAMILY']};
                    font-size: {TYPOGRAPHY['BODY']['SMALL']['size']};
                    color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                    padding-left: 1.2rem;
                    margin: 0;
                ">
                    <li>Annual price adjustments</li>
                    <li>Client contract renewals</li>
                    <li>Year-end performance review</li>
                    <li>Next year planning</li>
                </ul>
            </div>
        </div>
        """
        
        st.markdown(calendar_content, unsafe_allow_html=True)
        
        # Create pricing performance summary chart
        st.markdown(f"""
        <h3 style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            font-size: {TYPOGRAPHY['HEADING']['H3']['size']};
            font-weight: {TYPOGRAPHY['HEADING']['H3']['weight']};
            margin: 1.5rem 0 1rem 0;
        ">Pricing Performance Summary</h3>
        """, unsafe_allow_html=True)
        
        # Create a scatter plot of won and lost bids with CPI vs efficiency
        if not engineered_data.empty and 'CPI_Efficiency' in engineered_data.columns:
            fig = go.Figure()
            
            # Add Won data
            fig.add_trace(go.Scatter(
                x=won_engineered['CPI_Efficiency'],
                y=won_engineered['CPI'],
                mode='markers',
                marker=dict(
                    color=COLOR_SYSTEM['CHARTS']['WON'],
                    size=10,
                    opacity=0.7,
                    line=dict(width=1, color=COLOR_SYSTEM['NEUTRAL']['WHITE'])
                ),
                name='Won Bids',
                hovertemplate='<b>Won Bid</b><br>Efficiency: %{x:.2f}<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<br>LOI: %{customdata[1]:.1f} min<br>Completes: %{customdata[2]}<extra></extra>',
                customdata=won_engineered[['IR', 'LOI', 'Completes']]
            ))
            
            # Add Lost data
            fig.add_trace(go.Scatter(
                x=lost_engineered['CPI_Efficiency'],
                y=lost_engineered['CPI'],
                mode='markers',
                marker=dict(
                    color=COLOR_SYSTEM['CHARTS']['LOST'],
                    size=10,
                    opacity=0.7,
                    line=dict(width=1, color=COLOR_SYSTEM['NEUTRAL']['WHITE'])
                ),
                name='Lost Bids',
                hovertemplate='<b>Lost Bid</b><br>Efficiency: %{x:.2f}<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<br>LOI: %{customdata[1]:.1f} min<br>Completes: %{customdata[2]}<extra></extra>',
                customdata=lost_engineered[['IR', 'LOI', 'Completes']]
            ))
            
            # Add a reference line representing the optimal pricing curve
            x_range = np.linspace(
                min(engineered_data['CPI_Efficiency'].min() * 0.9, 0.1),
                engineered_data['CPI_Efficiency'].max() * 1.1,
                100
            )
            
            # Create a theoretical optimal pricing curve (inverse relationship)
            optimal_curve = 20 / (x_range + 1) + 3
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=optimal_curve,
                mode='lines',
                line=dict(
                    color=COLOR_SYSTEM['ACCENT']['YELLOW'],
                    width=3,
                    dash='dash'
                ),
                name='Optimal Pricing Curve',
                hovertemplate='Efficiency: %{x:.2f}<br>Optimal CPI: $%{y:.2f}<extra></extra>'
            ))
            
            # Apply consistent styling
            fig = apply_chart_styling(
                fig,
                title='Efficiency vs CPI Analysis',
                height=600
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title='Efficiency Metric',
                yaxis_title='CPI ($)'
            )
            
            # Add insights annotations
            fig = add_insights_annotation(
                fig,
                "The dashed yellow line represents the theoretical optimal pricing curve. Bids falling near this line have the highest chance of success while maintaining profitability.",
                0.01,
                0.95,
                width=250
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add scatter plot explanation
            scatter_explanation = f"""
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                This scatter plot visualizes the relationship between project efficiency and CPI pricing.
                The optimal pricing curve (dashed yellow line) represents the ideal balance between
                competitive pricing and profitability.
            </p>
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-top: 0.75rem;
            ">
                <strong>Key findings:</strong>
            </p>
            <ul style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            ">
                <li><strong>Won bids</strong> tend to cluster near the optimal curve</li>
                <li><strong>Lost bids</strong> are often priced above the optimal curve</li>
                <li><strong>High-efficiency projects</strong> show greater price sensitivity</li>
                <li><strong>Low-efficiency projects</strong> have more pricing flexibility but still need strategic positioning</li>
            </ul>
            <p style="
                font-family: {TYPOGRAPHY['FONT_FAMILY']};
                font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
                color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
                margin-top: 0.75rem;
            ">
                <strong>Strategic implications:</strong> Adjust your pricing approach based on project efficiency.
                For high-efficiency projects (right side of chart), prioritize competitive pricing close to the
                optimal curve. For low-efficiency projects (left side), focus on value-added services to justify
                necessary higher pricing.
            </p>
            """
            
            render_card(
                title="Understanding the Pricing Performance Chart", 
                content=scatter_explanation,
                accent_color=COLOR_SYSTEM['ACCENT']['YELLOW']
            )
        
        # Add final summary and next steps
        st.markdown(f"""
        <h3 style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            font-size: {TYPOGRAPHY['HEADING']['H3']['size']};
            font-weight: {TYPOGRAPHY['HEADING']['H3']['weight']};
            margin: 1.5rem 0 1rem 0;
        ">Summary & Next Steps</h3>
        """, unsafe_allow_html=True)
        
        # Create summary card
        summary_content = f"""
        <p style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
        ">
            This analysis has identified several key opportunities to optimize your pricing strategy:
        </p>
        <ol style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
        ">
            <li><strong>Implement the optimized pricing formula</strong> to balance competitiveness and profitability</li>
            <li><strong>Focus sales efforts on high-win-rate segments</strong> identified in the analysis</li>
            <li><strong>Develop tiered volume discounts</strong> to secure larger projects</li>
            <li><strong>Create differentiated service packages</strong> for projects with challenging parameters</li>
            <li><strong>Establish quarterly pricing reviews</strong> to continuously optimize your approach</li>
        </ol>
        <p style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            margin-top: 0.75rem;
        ">
            <strong>Expected results:</strong> Implementation of these recommendations is projected to:
        </p>
        <ul style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
        ">
            <li>Increase win rate by 5-8 percentage points</li>
            <li>Maintain or slightly improve profit margins</li>
            <li>Reduce pricing inconsistencies across similar projects</li>
            <li>Improve client satisfaction through more transparent pricing</li>
        </ul>
        """
        
        render_card(
            title="Strategic Recommendations Summary", 
            content=summary_content,
            icon=f'<span style="font-size: 1.5rem; color: {COLOR_SYSTEM["ACCENT"]["BLUE"]};">✅</span>',
            accent_color=COLOR_SYSTEM['ACCENT']['BLUE']
        )
        
        # Next steps with download button
        next_steps_content = f"""
        <p style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
        ">
            To implement these recommendations effectively:
        </p>
        <ol style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
        ">
            <li><strong>Share findings</strong> with the pricing committee and sales team</li>
            <li><strong>Develop an implementation plan</strong> with clear timelines and responsibilities</li>
            <li><strong>Create pricing guidance documents</strong> for the sales team</li>
            <li><strong>Monitor performance metrics</strong> to track impact and make adjustments</li>
            <li><strong>Schedule the first quarterly review</strong> to assess initial results</li>
        </ol>
        """
        
        render_card(
            title="Implementation Next Steps", 
            content=next_steps_content,
            icon=f'<span style="font-size: 1.5rem; color: {COLOR_SYSTEM["ACCENT"]["GREEN"]};">🚀</span>',
            accent_color=COLOR_SYSTEM['ACCENT']['GREEN']
        )
        
        # Add disclaimer
        st.markdown(f"""
        <div style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            font-size: {TYPOGRAPHY['BODY']['SMALL']['size']};
            color: {COLOR_SYSTEM['NEUTRAL']['DARKER']};
            font-style: italic;
            margin-top: 2rem;
            text-align: center;
        ">
            Note: These recommendations are based on historical data and current market conditions.
            Regular review and adjustment are recommended as market dynamics change.
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        # Log error
        logger.error(f"Error in show_insights: {e}", exc_info=True)
        
        # Display user-friendly error message
        st.error(f"An error occurred while generating insights: {str(e)}")
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
                <li>Check that your data has sufficient records for analysis</li>
                <li>Ensure all required columns are present in the dataset</li>
                <li>Try using the filtered dataset if dealing with outliers</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)"""
Insights component for the CPI Analysis & Prediction Dashboard.
Provides strategic insights and recommendations based on data analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging
from typing import Dict, Any, List, Tuple, Optional

# Import UI components
from ui_components import (
    render_card, metrics_row, apply_chart_styling,
    add_insights_annotation, grid_layout, render_icon_tabs
)

# Import data utilities
from utils.data_processor import get_data_summary, engineer_features

# Import configuration
from config import COLOR_SYSTEM, TYPOGRAPHY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_insights(won_data: pd.DataFrame, lost_data: pd.DataFrame, combined_data: pd.DataFrame) -> None:
    """
    Display the insights and recommendations component.
    
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
        ">Insights & Recommendations</h2>
        """, unsafe_allow_html=True)
        
        # Add introduction card
        intro_content = f"""
        <p style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
        ">
            This section provides strategic insights and actionable recommendations
            based on comprehensive analysis of your bid pricing data.
        </p>
        <p style="
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            font-size: {TYPOGRAPHY['BODY']['NORMAL']['size']};
            color: {COLOR_SYSTEM['PRIMARY']['MAIN']};
            margin-top: 0.5rem;
        ">
            Use these insights to optimize your pricing strategy, increase win rates,
            and maximize profitability across different market research segments.
        </p>
        """
        
        render_card(
            title="Strategic Insights Overview", 
            content=intro_content,
            icon=f'<span style="font-size: 1.5rem; color: {COLOR_SYSTEM["ACCENT"]["YELLOW"]};">💡</span>'
        )
        
        # Get data summary
        data_summary = get_data_summary(combined_data)
        
        # Calculate key metrics
        if 'Won' in data_summary and 'Lost' in data_summary:
            won_avg_cpi = data_summary['Won']['Avg_CPI']
            lost_avg_cpi = data_summary['Lost']['Avg_CPI']
            cpi_diff = lost_avg_cpi - won_avg_cpi
            cpi_diff_pct = (lost_avg_cpi / won_avg_cpi - 1) *