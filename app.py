import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Process Capability (SPC)", layout="wide", page_icon="📊")

st.title("📊 Mechanical Property Comprehensive Analysis")
st.markdown("Upload your production data to automatically calculate Ca, Cp, Cpk and visualize the process distribution and trends.")

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

        # --- INVISIBLE DATA CORRECTIONS ---
        # Group GE00 and GE01 together based on management rules
        if '鋼種' in df.columns:
            df['鋼種'] = df['鋼種'].replace(['GE00', 'GE01'], 'GE00/GE01')
        
        # Remove GF from analysis to prevent data errors
        if 'Metallic_Type' in df.columns:
            df = df[df['Metallic_Type'].astype(str).str.strip().str.upper() != 'GF']

        st.markdown("---")
        st.markdown("### 🎛️ Global Data Filters")
        
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
            df['訂單寬度'] = pd.to_numeric(df['訂單寬度'], errors='coerce')
            unique_widths = sorted(df['訂單寬度'].dropna().unique())
            selected_widths = st.multiselect("Order Width (訂單寬度)", options=unique_widths, default=unique_widths)

        # Apply Filters
        filtered_df = df[
            (df['LINE'].isin(lines)) & 
            (df['鋼種'].isin(grades)) &
            (df['訂單寬度'].isin(selected_widths))
        ].copy()

        # --- CRITICAL FIX: IDENTIFY COIL COLUMN & SORT STRICTLY ---
        coil_col = None
        potential_coil_names = ['COIL_NO', 'COIL NO', 'Coil_No', 'CoilNo', '製造批號', 'Batch']
        for col in potential_coil_names:
            if col in filtered_df.columns:
                coil_col = col
                break
        
        time_cols = [c for c in ['生產日期', '開始時間', 'Time', 'Date'] if c in filtered_df.columns]
        sort_cols = time_cols + ([coil_col] if coil_col else [])
        
        if sort_cols:
            filtered_df = filtered_df.sort_values(by=sort_cols).reset_index(drop=True)
        else:
            filtered_df = filtered_df.sort_index().reset_index(drop=True)

        st.markdown("---")
        st.markdown("### 🏆 Comprehensive Process Capability Overview")
        
        # Identify Target Columns automatically
        potential_targets = ['YS', 'TS', 'EL', 'TENSILE_YIELD', 'TENSILE_TENSILE', 'TENSILE_ELONG', 'skp+t/l']
        available_targets = [c for c in potential_targets if c in df.columns]
        
        if not available_targets:
            available_targets = filtered_df.select_dtypes(include=np.number).columns.tolist()

        if not available_targets:
            st.error("No numeric target columns found for analysis.")
            st.stop()

        # Create settings expander for Specs (so users can tweak LSL/USL for each chart)
        with st.expander("⚙️ Specification Limits Settings (LSL / USL / Target)", expanded=False):
            st.markdown("Adjust the specifications for each parameter. By default, limits are estimated using $\pm 3\sigma$.")
            specs = {}
            for target in available_targets:
                st.markdown(f"**{target}**")
                s_mean = filtered_df[target].mean()
                s_std = filtered_df[target].std()
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    t_val = st.number_input(f"Target ({target})", value=float(s_mean), key=f"t_{target}")
                with sc2:
                    l_val = st.number_input(f"LSL ({target})", value=float(s_mean - 3*s_std), key=f"l_{target}")
                with sc3:
                    u_val = st.number_input(f"USL ({target})", value=float(s_mean + 3*s_std), key=f"u_{target}")
                specs[target] = {'tgt': t_val, 'lsl': l_val, 'usl': u_val}
                st.write("")

        # --- DRAW PARALLEL CHARTS ---
        st.markdown("---")
        
        for i in range(0, len(available_targets), 2):
            grid_cols = st.columns(2)
            
            for j in range(2):
                if i + j < len(available_targets):
                    target_col = available_targets[i + j]
                    col = grid_cols[j]
                    
                    with col:
                        # Clean data for specific target
                        analysis_df = filtered_df.dropna(subset=[target_col]).copy()
                        analysis_df[target_col] = pd.to_numeric(analysis_df[target_col], errors='coerce')
                        analysis_df = analysis_df.dropna(subset=[target_col])
                        
                        # Remove duplicate coils to prevent vertical line glitches
                        if coil_col:
                            analysis_df = analysis_df.drop_duplicates(subset=[coil_col], keep='last')

                        data_series = analysis_df[target_col]
                        
                        if len(data_series) < 2:
                            st.warning(f"Not enough data for {target_col}.")
                            continue

                        mean = data_series.mean()
                        std = data_series.std()
                        count = len(data_series)

                        target_val = specs[target_col]['tgt']
                        lsl = specs[target_col]['lsl']
                        usl = specs[target_col]['usl']

                        ca = ((mean - target_val) / ((usl - lsl) / 2)) * 100 if usl != lsl else 0
                        cp = (usl - lsl) / (6 * std) if std > 0 else 0
                        cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std)) if std > 0 else 0

                        # ==========================================
                        # 1. DISTRIBUTION CHART (Professional Style)
                        # ==========================================
                        fig_dist = go.Figure()
                        
                        # Histogram with clean borders
                        fig_dist.add_trace(go.Histogram(
                            x=data_series, 
                            histnorm='probability density', 
                            name='Actual Data', 
                            marker=dict(color='#5A9BD4', line=dict(color='white', width=1)), 
                            opacity=0.85
                        ))
                        
                        # Smooth Normal Curve
                        if std > 0:
                            x_curve = np.linspace(data_series.min() - 1*std, data_series.max() + 1*std, 500)
                            fig_dist.add_trace(go.Scatter(
                                x=x_curve, y=norm.pdf(x_curve, mean, std), 
                                mode='lines', name='Normal Curve', 
                                line=dict(color='#FF9D00', width=3)
                            ))
                        
                        # Spec Lines
                        fig_dist.add_vline(x=lsl, line_dash="dash", line_color="#E03A3C", line_width=2, annotation_text=f"LSL: {lsl:.1f}", annotation_position="top left")
                        fig_dist.add_vline(x=usl, line_dash="dash", line_color="#E03A3C", line_width=2, annotation_text=f"USL: {usl:.1f}", annotation_position="top right")
                        fig_dist.add_vline(x=target_val, line_dash="dot", line_color="#4CAF50", line_width=2, annotation_text="TGT", annotation_position="top right")
                        fig_dist.add_vline(x=mean, line_dash="solid", line_color="#333333", line_width=1.5, annotation_text=f"Mean: {mean:.1f}", annotation_position="top left")
                        
                        fig_dist.update_layout(
                            title=dict(text=f"<b>Distribution: {target_col}</b>", font=dict(size=16)),
                            height=320, 
                            margin=dict(l=20, r=20, t=50, b=20), 
                            showlegend=False, 
                            bargap=0.05,
                            template='plotly_white',
                            yaxis=dict(showticklabels=False, showgrid=False)
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)

                        # ==========================================
                        # 2. HTML METRICS CARD
                        # ==========================================
                        is_capable = cpk >= 1.33
                        icon = "✅" if is_capable else "❌"
                        status_text = "Capable" if is_capable else "Not Capable"
                        border_color = "#28a745" if is_capable else "#dc3545"
                        
                        card_html = f"""
                        <div style="border: 1px solid #e0e0e0; border-left: 6px solid {border_color}; border-radius: 8px; background-color: #ffffff; padding: 12px; margin-bottom: 10px; box-shadow: 0px 2px 4px rgba(0,0,0,0.05);">
                            <div style="display: flex; align-items: center; border-bottom: 1px solid #eee; padding-bottom: 6px; margin-bottom: 8px; flex-wrap: wrap; gap: 10px;">
                                <span style="font-size: 15px; font-weight: bold; color: {border_color}; min-width: 120px;">{icon} {status_text}</span>
                                <span style="color: #ccc;">|</span>
                                <span style="font-family: monospace; font-size: 13px; color: #444;"><b>LSL:</b> {lsl:.1f} &nbsp;&nbsp; <b>USL:</b> {usl:.1f}</span>
                                <span style="color: #ccc;">|</span>
                                <span style="font-family: monospace; font-size: 13px; color: #444;"><b>n:</b> {count}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; font-family: monospace; font-size: 14px; margin-left: 5px; padding-top: 4px;">
                                <span><b style="color:#555;">Mean:</b> {mean:.2f}</span>
                                <span><b style="color:#555;">Std:</b> {std:.3f}</span>
                                <span style="color: #d9534f; font-weight: bold;">Cpk = {cpk:.3f}</span>
                                <span style="font-weight: bold; color: #333;">Cp = {cp:.3f}</span>
                            </div>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)

                        # ==========================================
                        # 3. TREND CHART (Professional Style)
                        # ==========================================
                        x_data = analysis_df[coil_col].astype(str) if coil_col else analysis_df.index.astype(str)
                        
                        fig_trend = go.Figure()
                        
                        # Trend line with clear markers
                        fig_trend.add_trace(go.Scatter(
                            x=x_data,
                            y=analysis_df[target_col],
                            mode='lines+markers',
                            name='Process Data',
                            line=dict(color='#5A9BD4', width=2),
                            marker=dict(size=7, color='#5A9BD4', line=dict(color='white', width=1))
                        ))

                        # Highlight OOC points with prominent red circles
                        ooc_df = analysis_df[(analysis_df[target_col] < lsl) | (analysis_df[target_col] > usl)]
                        if not ooc_df.empty:
                            ooc_x = ooc_df[coil_col].astype(str) if coil_col else ooc_df.index.astype(str)
                            fig_trend.add_trace(go.Scatter(
                                x=ooc_x,
                                y=ooc_df[target_col],
                                mode='markers',
                                marker=dict(color='#E03A3C', size=10, symbol='circle', line=dict(color='white', width=1)),
                                name='OOC'
                            ))

                        # Horizontal reference lines
                        fig_trend.add_hline(y=mean, line_color="#333333", line_width=1.5, line_dash="solid", annotation_text="Mean", annotation_font_size=10)
                        fig_trend.add_hline(y=lsl, line_dash="dash", line_color="#E03A3C", line_width=2, annotation_text="LSL", annotation_font_size=10)
                        fig_trend.add_hline(y=usl, line_dash="dash", line_color="#E03A3C", line_width=2, annotation_text="USL", annotation_font_size=10)
                        
                        fig_trend.update_layout(
                            title=dict(text=f"<b>Trend: {target_col}</b>", font=dict(size=14)),
                            height=280, 
                            margin=dict(l=20, r=20, t=40, b=20),
                            xaxis_title="",
                            showlegend=False,
                            template='plotly_white',
                            xaxis=dict(
                                type='category',
                                categoryorder='array',
                                categoryarray=x_data.tolist(),
                                showgrid=False,
                                zeroline=False,
                                tickangle=45
                            ),
                            yaxis=dict(showgrid=True, gridcolor='#eeeeee', zeroline=False)
                        )
                        st.plotly_chart(fig_trend, use_container_width=True)
                        st.markdown("<br><br>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a file to begin analysis.")
