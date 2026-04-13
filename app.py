import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Process Capability (SPC)", layout="wide", page_icon="📊")

st.title("📊 Process Capability Analysis (SPC)")
st.markdown("Upload your production data to automatically calculate Ca, Cp, Cpk and visualize the process distribution.")

# 1. Upload Data
uploaded_file = st.file_uploader("Upload Data File (CSV or Excel)", type=['csv', 'xlsx'])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Clean column names (strip whitespace to prevent hidden KeyErrors)
        df.columns = df.columns.str.strip()

        st.markdown("---")
        st.markdown("### 🎛️ Global Data Filters")
        
        # Safely identify columns (Fallback to index if specific names are missing)
        line_col = 'LINE' if 'LINE' in df.columns else (df.select_dtypes(include='object').columns[0] if len(df.select_dtypes(include='object').columns) > 0 else df.columns[0])
        grade_col = '鋼種' if '鋼種' in df.columns else (df.select_dtypes(include='object').columns[1] if len(df.select_dtypes(include='object').columns) > 1 else df.columns[1])
        width_col = '訂單寬度' if '訂單寬度' in df.columns else (df.select_dtypes(include='number').columns[0] if len(df.select_dtypes(include='number').columns) > 0 else df.columns[2])

        # Modern horizontal filter layout
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            lines = st.multiselect(f"Factory Line ({line_col})", options=df[line_col].dropna().unique(), default=df[line_col].dropna().unique())
        with col_f2:
            grades = st.multiselect(f"Steel Grade ({grade_col})", options=df[grade_col].dropna().unique(), default=df[grade_col].dropna().unique())
        with col_f3:
            if pd.api.types.is_numeric_dtype(df[width_col]):
                min_w, max_w = float(df[width_col].min()), float(df[width_col].max())
                if min_w < max_w:
                    width_range = st.slider(f"Order Width ({width_col})", min_w, max_w, (min_w, max_w))
                else:
                    width_range = (min_w, max_w)
                    st.info(f"Fixed Width: {min_w}")
            else:
                width_range = None
                st.info("Width column is not numeric.")

        # Apply Filters
        if width_range:
            filtered_df = df[
                (df[line_col].isin(lines)) & 
                (df[grade_col].isin(grades)) & 
                (df[width_col].between(width_range[0], width_range[1]))
            ].copy()
        else:
            filtered_df = df[
                (df[line_col].isin(lines)) & 
                (df[grade_col].isin(grades))
            ].copy()

        st.markdown("---")
        st.markdown("### 🎯 Capability Parameters")
        
        # FIX KEY ERROR: Dynamically extract ONLY numeric columns for analysis
        numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
        
        if not numeric_cols:
            st.error("No numeric columns found in the dataset for analysis.")
            st.stop()

        col_prop, col_target, col_lsl, col_usl = st.columns(4)
        
        with col_prop:
            target_col = st.selectbox("Select Parameter to Analyze", options=numeric_cols)
        
        data_series = filtered_df[target_col].dropna()
        
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
        cp = (usl - lsl) / (6 * std) if std != 0 else 0
        cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std)) if std != 0 else 0

        # Metrics Display
        st.markdown("### 🏆 SPC Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Process Mean (μ)", f"{mean:.3f}")
        m2.metric("Accuracy (Ca)", f"{ca:.3f}", delta="Ideal closer to 0", delta_color="off")
        m3.metric("Precision (Cp)", f"{cp:.3f}", delta="≥ 1.33 is Good")
        
        is_capable = cpk >= 1.33
        m4.metric("Capability (Cpk)", f"{cpk:.3f}", delta="Capable" if is_capable else "Incapable", delta_color="normal" if is_capable else "inverse")

        st.markdown("---")
        
        # Interactive Charts
        tab1, tab2 = st.tabs(["📊 Distribution Chart", "📈 Trend Chart"])

        with tab1:
            fig_hist = px.histogram(
                filtered_df, 
                x=target_col, 
                nbins=30, 
                marginal="box", 
                title=f"Distribution of {target_col}", 
                color_discrete_sequence=['#4dabf7'],
                opacity=0.8
            )
            fig_hist.add_vline(x=lsl, line_dash="dash", line_color="red", annotation_text="LSL")
            fig_hist.add_vline(x=usl, line_dash="dash", line_color="red", annotation_text="USL")
            fig_hist.add_vline(x=target_val, line_color="green", line_width=2, annotation_text="Target")
            fig_hist.update_layout(bargap=0.1)
            st.plotly_chart(fig_hist, use_container_width=True)

        with tab2:
            fig_trend = px.line(
                filtered_df.reset_index(), 
                y=target_col, 
                title=f"Process Trend: {target_col}", 
                markers=True,
                color_discrete_sequence=['#ffc000']
            )
            fig_trend.add_hline(y=mean, line_color="blue", annotation_text="Mean (μ)")
            fig_trend.add_hline(y=lsl, line_dash="dash", line_color="red", annotation_text="LSL")
            fig_trend.add_hline(y=usl, line_dash="dash", line_color="red", annotation_text="USL")
            st.plotly_chart(fig_trend, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a file to begin analysis.")
