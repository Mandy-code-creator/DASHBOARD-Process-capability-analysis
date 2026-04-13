import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Process Capability (SPC)", layout="wide", page_icon="📊")

st.title("📊 Process Capability Analysis (SPC)")
st.markdown("Upload your production data to automatically calculate Ca, Cp, Cpk and visualize the process distribution by Coil.")

# 1. Upload Data
uploaded_file = st.file_uploader("Upload Data File (CSV or Excel)", type=['csv', 'xlsx'])

if uploaded_file:
    try:
        # Read file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Clean column names (remove hidden spaces)
        df.columns = df.columns.str.strip()

        st.markdown("---")
        st.markdown("### 🎛️ Global Data Filters")
        
        # EXACT COLUMN MATCHING based on your uploaded images
        req_filters = ['LINE', '鋼種', '訂單寬度']
        missing_filters = [c for c in req_filters if c not in df.columns]
        
        if missing_filters:
            st.error(f"Missing required filter columns: {', '.join(missing_filters)}. Please ensure your Excel file contains 'LINE', '鋼種', and '訂單寬度'.")
            st.stop()

        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            lines = st.multiselect("Factory Line (LINE)", options=df['LINE'].dropna().unique(), default=df['LINE'].dropna().unique())
        with col_f2:
            grades = st.multiselect("Steel Grade (鋼種)", options=df['鋼種'].dropna().unique(), default=df['鋼種'].dropna().unique())
        with col_f3:
            # Force Order Width to numeric for the dropdown
            df['訂單寬度'] = pd.to_numeric(df['訂單寬度'], errors='coerce')
            unique_widths = sorted(df['訂單寬度'].dropna().unique())
            selected_widths = st.multiselect("Order Width (訂單寬度)", options=unique_widths, default=unique_widths)

        # Apply Filters
        filtered_df = df[
            (df['LINE'].isin(lines)) & 
            (df['鋼種'].isin(grades)) &
            (df['訂單寬度'].isin(selected_widths))
        ].copy()

        st.markdown("---")
        st.markdown("### 🎯 Capability Parameters")
        
        # Identify Target Columns
        potential_targets = ['YS', 'TS', 'EL', 'TENSILE_YIELD', 'TENSILE_TENSILE', 'TENSILE_ELONG', 'skp+t/l']
        available_targets = [c for c in potential_targets if c in df.columns]
        
        # Fallback if specific names aren't found
        if not available_targets:
            available_targets = filtered_df.select_dtypes(include=np.number).columns.tolist()

        if not available_targets:
            st.error("No numeric target columns found for analysis.")
            st.stop()
            
        # --- IDENTIFY COIL NUMBER COLUMN ---
        coil_col = None
        potential_coil_names = ['COIL_NO', 'COIL NO', 'Coil_No', 'CoilNo', '製造批號', 'Batch']
        for col in potential_coil_names:
            if col in df.columns:
                coil_col = col
                break
        
        # Fallback if no coil column
        if not coil_col:
            filtered_df['COIL_INDEX'] = range(1, len(filtered_df) + 1)
            coil_col = 'COIL_INDEX'

        col_prop, col_target, col_lsl, col_usl = st.columns(4)
        
        with col_prop:
            target_col = st.selectbox("Select Parameter to Analyze", options=available_targets)
        
        # Ensure coil col is string
        if coil_col != 'COIL_INDEX':
            filtered_df[coil_col] = filtered_df[coil_col].astype(str)

        # --- CRITICAL FIX: SORTING CHRONOLOGICALLY ---
        # Sort data by production date/time to prevent spaghetti charts
        time_cols = [c for c in ['生產日期', '開始時間', 'Time', 'Date', '生產年'] if c in filtered_df.columns]
        if time_cols:
            filtered_df = filtered_df.sort_values(by=time_cols)
        else:
            # Fallback to original Excel row order
            filtered_df = filtered_df.sort_index()

        analysis_df = filtered_df.dropna(subset=[target_col]).copy()
        analysis_df[target_col] = pd.to_numeric(analysis_df[target_col], errors='coerce')
        analysis_df = analysis_df.dropna(subset=[target_col])
        
        # Remove duplicate coil testing (keep the latest test) to prevent vertical lines in the chart
        if coil_col != 'COIL_INDEX':
            analysis_df = analysis_df.drop_duplicates(subset=[coil_col], keep='last')

        data_series = analysis_df[target_col]
        
        if len(data_series) < 2:
            st.warning("Not enough data points after filtering. Please adjust your filters.")
            st.stop()

        mean = data_series.mean()
        std = data_series.std()

        with col_target:
            target_val = st.number_input("Target Value", value=float(mean), format="%.3f")
        with col_lsl:
            lsl = st.number_input("Lower Spec Limit (LSL)", value=float(mean - 3*std), format="%.3f")
        with col_usl:
            usl = st.number_input("Upper Spec Limit (USL)", value=float(mean + 3*std), format="%.3f")

        # SPC Calculations
        ca = (mean - target_val) / ((usl - lsl) / 2) if usl != lsl else 0
        cp = (usl - lsl) / (6 * std) if std > 0 else 0
        cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std)) if std > 0 else 0

        # Metrics Display
        st.markdown("### 🏆 SPC Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Process Mean (μ)", f"{mean:.3f}")
        m2.metric("Accuracy (Ca)", f"{ca:.3f}", delta="Ideal closer to 0", delta_color="off")
        m3.metric("Precision (Cp)", f"{cp:.3f}", delta="≥ 1.33 is Good")
        
        is_capable = cpk >= 1.33
        m4.metric("Capability (Cpk)", f"{cpk:.3f}", delta="Capable" if is_capable else "Incapable", delta_color="normal" if is_capable else "inverse")

        st.markdown("---")
        
        # Interactive Charts & Data Table
        tab1, tab2, tab3 = st.tabs(["📊 Distribution Chart", "📈 Trend by Coil", "📋 Detail Data"])

        with tab1:
            fig_dist = go.Figure()

            # Histogram
            fig_dist.add_trace(go.Histogram(
                x=data_series,
                histnorm='probability density',
                name='Actual Data',
                marker_color='#4dabf7',
                opacity=0.75,
                xbins=dict(size=(data_series.max() - data_series.min()) / 30 if data_series.max() > data_series.min() else 1)
            ))

            # Normal Curve
            if std > 0:
                x_curve = np.linspace(data_series.min() - 1*std, data_series.max() + 1*std, 500)
                y_curve = norm.pdf(x_curve, mean, std)
                fig_dist.add_trace(go.Scatter(
                    x=x_curve,
                    y=y_curve,
                    mode='lines',
                    name='Normal Curve',
                    line=dict(color='#343a40', width=3)
                ))

            # Limits
            fig_dist.add_vline(x=lsl, line_dash="dash", line_color="red", annotation_text="LSL", annotation_position="top left")
            fig_dist.add_vline(x=usl, line_dash="dash", line_color="red", annotation_text="USL", annotation_position="top right")
            fig_dist.add_vline(x=target_val, line_color="green", line_width=2, annotation_text="Target", annotation_position="top right")

            fig_dist.update_layout(
                title=f"Process Distribution with Normal Curve: {target_col}",
                xaxis_title=target_col,
                yaxis_title="Probability Density",
                bargap=0.05,
                hovermode="x unified",
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with tab2:
            # Trend Chart explicitly using COIL_NO as the X-axis
            fig_trend = px.line(
                analysis_df, 
                x=coil_col,
                y=target_col, 
                title=f"Process Trend per Coil: {target_col}", 
                markers=True,
                color_discrete_sequence=['#ffc000']
            )
            
            # Highlight Out of Spec (OOC) points in Red on the trend chart
            ooc_df = analysis_df[(analysis_df[target_col] < lsl) | (analysis_df[target_col] > usl)]
            if not ooc_df.empty:
                fig_trend.add_trace(go.Scatter(
                    x=ooc_df[coil_col],
                    y=ooc_df[target_col],
                    mode='markers',
                    marker=dict(color='red', size=10, symbol='x'),
                    name='Out of Spec (OOC)'
                ))

            fig_trend.add_hline(y=mean, line_color="blue", annotation_text="Mean (μ)")
            fig_trend.add_hline(y=lsl, line_dash="dash", line_color="red", annotation_text="LSL")
            fig_trend.add_hline(y=usl, line_dash="dash", line_color="red", annotation_text="USL")
            
            # --- CRITICAL FIX: FORCE X-AXIS ARRAY ORDER ---
            fig_trend.update_layout(
                xaxis_title="Coil Number" if coil_col != 'COIL_INDEX' else "Production Sequence",
                xaxis=dict(
                    type='category',
                    categoryorder='array',
                    categoryarray=analysis_df[coil_col].tolist() # STRICTLY ENFORCE ORDER
                )
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with tab3:
            st.markdown(f"### 📋 Detailed Coil Data for {target_col}")
            
            # Create a clean display dataframe
            display_df = analysis_df[[coil_col, target_col]].copy()
            
            # Determine Status
            display_df['Status'] = 'OK'
            display_df.loc[display_df[target_col] < lsl, 'Status'] = 'Below LSL'
            display_df.loc[display_df[target_col] > usl, 'Status'] = 'Above USL'
            
            # Function to highlight OOC rows
            def highlight_ooc(row):
                if row['Status'] != 'OK':
                    return ['background-color: #ffcccc'] * len(row)
                return [''] * len(row)
                
            st.dataframe(
                display_df.style.apply(highlight_ooc, axis=1).format({target_col: "{:.3f}"}),
                use_container_width=True,
                hide_index=True
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a file to begin analysis.")
