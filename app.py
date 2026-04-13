import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Mechanical Property SPC Dashboard", layout="wide", page_icon="📊")

st.title("📊 Mechanical Property Comprehensive Analysis")
st.markdown("Automated SPC calculation focusing on Process Control Limits (UCL/LCL).")

# 1. Upload Data
uploaded_file = st.file_uploader("Upload Excel or CSV file", type=['csv', 'xlsx'])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        df.columns = df.columns.str.strip()

        # --- DATA CORRECTIONS ---
        if '鋼種' in df.columns:
            df['鋼種'] = df['鋼種'].replace(['GE00', 'GE01'], 'GE00/GE01')
        if 'Metallic_Type' in df.columns:
            df = df[df['Metallic_Type'].astype(str).str.strip().str.upper() != 'GF']

        st.markdown("### 🎛️ Global Data Filters")
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            lines = st.multiselect("Factory Line (LINE)", options=df['LINE'].dropna().unique(), default=df['LINE'].dropna().unique())
        with col_f2:
            grades = st.multiselect("Steel Grade (鋼種)", options=df['鋼種'].dropna().unique(), default=df['鋼種'].dropna().unique())
        with col_f3:
            df['訂單寬度'] = pd.to_numeric(df['訂單寬度'], errors='coerce')
            unique_widths = sorted(df['訂單寬度'].dropna().unique())
            selected_widths = st.multiselect("Order Width (訂單寬度)", options=unique_widths, default=unique_widths)

        filtered_df = df[(df['LINE'].isin(lines)) & (df['鋼種'].isin(grades)) & (df['訂單寬度'].isin(selected_widths))].copy()

        # --- IDENTIFY COIL COLUMN & SORT ---
        coil_col = next((c for c in ['COIL_NO', 'COIL NO', 'Coil_No', '製造批號', 'Batch'] if c in filtered_df.columns), None)
        time_cols = [c for c in ['生產日期', '開始時間', 'Time', 'Date'] if c in filtered_df.columns]
        sort_cols = time_cols + ([coil_col] if coil_col else [])
        if sort_cols:
            filtered_df = filtered_df.sort_values(by=sort_cols).reset_index(drop=True)

        # Identify Targets
        potential_targets = ['YS', 'TS', 'EL', 'TENSILE_YIELD', 'TENSILE_TENSILE', 'TENSILE_ELONG', 'skp+t/l']
        available_targets = [c for c in potential_targets if c in filtered_df.columns]

        # Specifications settings
        with st.expander("⚙️ Customer Specification Settings (Optional)", expanded=False):
            st.info("If LSL/USL are left as 0, Ca/Cp/Cpk will show as 'N/A'.")
            specs = {}
            for target in available_targets:
                st.markdown(f"**{target}**")
                sc1, sc2, sc3 = st.columns(3)
                with sc1: t_val = st.number_input(f"Target ({target})", value=0.0, key=f"t_{target}")
                with sc2: l_val = st.number_input(f"LSL ({target})", value=0.0, key=f"l_{target}")
                with sc3: u_val = st.number_input(f"USL ({target})", value=0.0, key=f"u_{target}")
                specs[target] = {'tgt': t_val, 'lsl': l_val, 'usl': u_val}

        # --- DRAW GRID VIEW ---
        st.markdown("---")
        for i in range(0, len(available_targets), 2):
            grid_cols = st.columns(2)
            for j in range(2):
                if i + j < len(available_targets):
                    target_col = available_targets[i + j]
                    with grid_cols[j]:
                        analysis_df = filtered_df.dropna(subset=[target_col]).copy()
                        analysis_df[target_col] = pd.to_numeric(analysis_df[target_col], errors='coerce')
                        analysis_df = analysis_df.dropna(subset=[target_col])
                        if coil_col: analysis_df = analysis_df.drop_duplicates(subset=[coil_col], keep='last')
                        
                        data_series = analysis_df[target_col]
                        if len(data_series) < 2: continue
                        
                        mean, std, count = data_series.mean(), data_series.std(), len(data_series)
                        d_max, d_min = data_series.max(), data_series.min()
                        ucl, lcl = mean + 3*std, mean - 3*std
                        
                        # Fetch Specs
                        lsl, usl, tgt = specs[target_col]['lsl'], specs[target_col]['usl'], specs[target_col]['tgt']
                        
                        # --- SPC LOGIC: Check for missing Specs ---
                        if lsl == 0 and usl == 0:
                            ca_display, cp_display, cpk_display = "N/A (No Specs)", "N/A", "N/A"
                            spec_active = False
                        else:
                            ca = ((mean - tgt) / ((usl - lsl) / 2)) * 100 if usl != lsl else 0
                            cp = (usl - lsl) / (6 * std) if std > 0 else 0
                            cpk = min((usl - mean)/(3*std), (mean - lsl)/(3*std)) if std > 0 else 0
                            ca_display, cp_display, cpk_display = f"{ca:.1f}%", f"{cp:.3f}", f"{cpk:.3f}"
                            spec_active = True

                        # 1. DISTRIBUTION CHART
                        fig_dist = px.histogram(analysis_df, x=target_col, color='鋼種', nbins=20, barmode='stack', color_discrete_sequence=['#1F497D', '#4F81BD', '#8DB4E2'])
                        fig_dist.update_traces(marker_line_color='white', marker_line_width=1, opacity=0.9)
                        
                        if std > 0:
                            x_curve = np.linspace(d_min - std, d_max + std, 200)
                            bin_w = (d_max - d_min)/20 if d_max > d_min else 1
                            y_curve = norm.pdf(x_curve, mean, std) * count * bin_w
                            fig_dist.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', line=dict(color='#FFB300', width=2), showlegend=False))

                        if spec_active:
                            fig_dist.add_vline(x=lsl, line_color="red", line_width=2)
                            fig_dist.add_annotation(x=lsl, y=0.95, yref='paper', text=f"LSL: {lsl:.1f}", showarrow=False, font=dict(color="red", size=10), xanchor="right", xshift=-5)
                            fig_dist.add_vline(x=usl, line_color="red", line_width=2)
                            fig_dist.add_annotation(x=usl, y=0.95, yref='paper', text=f"USL: {usl:.1f}", showarrow=False, font=dict(color="red", size=10), xanchor="left", xshift=5)
                        
                        fig_dist.add_vline(x=mean, line_color="#333", line_dash="dash", line_width=1.5)
                        fig_dist.add_annotation(x=mean, y=1.05, yref='paper', text=f"Mean: {mean:.1f}", showarrow=False, font=dict(color="#333", size=10))

                        fig_dist.update_layout(
                            title=dict(text=f"<b>{target_col} Distribution</b>", x=0.5, xanchor='center'),
                            height=380, plot_bgcolor='white', margin=dict(l=40, r=40, b=80, t=100),
                            legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                            xaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True, gridcolor='#F0F0F0'),
                            yaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True, gridcolor='#F0F0F0')
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)

                        # 2. METRICS CARD (with N/A handling)
                        st.markdown(f"""
                        <div style="padding:12px; border-radius:8px; border-left: 6px solid #4F81BD; background-color:#f8f9fa; margin-bottom:15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                            <span style="font-size:16px;"><b>Cpk: {cpk_display}</b> | <b>Cp: {cp_display}</b> | <b>Ca: {ca_display}</b></span><br>
                            <span style="color:#666; font-size:13px;">Mean: {mean:.2f} | Std: {std:.3f} | n: {count}</span>
                        </div>
                        """, unsafe_allow_html=True)

                        # 3. TRENDING CHART (UCL & LCL)
                        x_axis = analysis_df[coil_col].astype(str) if coil_col else analysis_df.index.astype(str)
                        fig_trend = go.Figure()
                        fig_trend.add_trace(go.Scatter(x=x_axis, y=data_series, mode='lines+markers', line=dict(color='#4F81BD', width=2), marker=dict(size=6, color='white', line=dict(color='#4F81BD', width=2))))
                        
                        fig_trend.add_hline(y=mean, line_color="#333", line_width=1.5, annotation_text=f"Mean: {mean:.1f}")
                        fig_trend.add_hline(y=ucl, line_color="#FF8C00", line_width=1.5, line_dash="dash", annotation_text=f"UCL: {ucl:.1f}")
                        fig_trend.add_hline(y=lcl, line_color="#FF8C00", line_width=1.5, line_dash="dash", annotation_text=f"LCL: {lcl:.1f}")
                        
                        fig_trend.update_layout(
                            title=dict(text=f"<b>{target_col} Trend (Control Limits)</b>", x=0.5, xanchor='center'),
                            height=320, plot_bgcolor='#F9F9F9', margin=dict(l=40, r=100, t=30, b=40),
                            xaxis=dict(type='category', showgrid=False, linecolor='black'),
                            yaxis=dict(showgrid=True, gridcolor='#E0E0E0', showline=True, linecolor='black')
                        )
                        st.plotly_chart(fig_trend, use_container_width=True)
                        st.markdown("<br>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload a file to begin.")
