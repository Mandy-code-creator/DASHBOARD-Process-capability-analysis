import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Mechanical Property SPC Dashboard", layout="wide", page_icon="📊")

st.title("📊 Mechanical Property Comprehensive Analysis")
st.markdown("Hệ thống tự động tính toán Ca, Cp, Cpk và trực quan hóa biểu đồ phân phối & xu hướng tối giản.")

# 1. Upload Data
uploaded_file = st.file_uploader("Tải lên file Excel hoặc CSV", type=['csv', 'xlsx'])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        df.columns = df.columns.str.strip()

        # --- DATA CORRECTIONS (Management Rules) ---
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

        # Identify Target Columns
        potential_targets = ['YS', 'TS', 'EL', 'TENSILE_YIELD', 'TENSILE_TENSILE', 'TENSILE_ELONG', 'skp+t/l']
        available_targets = [c for c in potential_targets if c in filtered_df.columns]

        with st.expander("⚙️ Specification Limits Settings (LSL / USL)", expanded=False):
            specs = {}
            for target in available_targets:
                s_mean, s_std = filtered_df[target].mean(), filtered_df[target].std()
                sc1, sc2, sc3 = st.columns(3)
                with sc1: t_val = st.number_input(f"Target ({target})", value=float(s_mean or 0), key=f"t_{target}")
                with sc2: l_val = st.number_input(f"LSL ({target})", value=float((s_mean or 0) - 3*(s_std or 0)), key=f"l_{target}")
                with sc3: u_val = st.number_input(f"USL ({target})", value=float((s_mean or 0) + 3*(s_std or 0)), key=f"u_{target}")
                specs[target] = {'tgt': t_val, 'lsl': l_val, 'usl': u_val}

        # --- DRAW GRID VIEW ---
        st.markdown("---")
        for i in range(0, len(available_targets), 2):
            grid_cols = st.columns(2)
            for j in range(2):
                if i + j < len(available_targets):
                    target_col = available_targets[i + j]
                    with grid_cols[j]:
                        # Clean data for target
                        analysis_df = filtered_df.dropna(subset=[target_col]).copy()
                        analysis_df[target_col] = pd.to_numeric(analysis_df[target_col], errors='coerce')
                        analysis_df = analysis_df.dropna(subset=[target_col])
                        if coil_col: analysis_df = analysis_df.drop_duplicates(subset=[coil_col], keep='last')
                        
                        data_series = analysis_df[target_col]
                        if len(data_series) < 2: continue
                        
                        mean, std, count = data_series.mean(), data_series.std(), len(data_series)
                        lsl, usl = specs[target_col]['lsl'], specs[target_col]['usl']
                        cpk = min((usl - mean)/(3*std), (mean - lsl)/(3*std)) if std > 0 else 0

                        # 1. DISTRIBUTION CHART (Standard Stacked)
                        fig_dist = px.histogram(analysis_df, x=target_col, color='鋼種', nbins=20, barmode='stack', color_discrete_sequence=['#1F497D', '#4F81BD', '#8DB4E2'])
                        fig_dist.update_traces(marker_line_color='white', marker_line_width=1, opacity=0.9)
                        
                        # Add Normal Curve
                        if std > 0:
                            x_curve = np.linspace(data_series.min() - std, data_series.max() + std, 200)
                            bin_w = (data_series.max() - data_series.min())/20 if data_series.max() > data_series.min() else 1
                            y_curve = norm.pdf(x_curve, mean, std) * count * bin_w
                            fig_dist.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', line=dict(color='#FFB300', width=2), showlegend=False))

                        fig_dist.add_vline(x=lsl, line_color="red", line_width=2)
                        fig_dist.add_vline(x=usl, line_color="red", line_width=2)
                        
                        fig_dist.update_layout(
                            title=dict(text=f"<b>{target_col} Distribution</b>", x=0.5, xanchor='center'),
                            height=350, plot_bgcolor='white', margin=dict(b=60),
                            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
                            xaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True),
                            yaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True)
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)

                        # 2. METRICS CARD (Minimal)
                        st.markdown(f"**Cpk: {cpk:.3f}** | Mean: {mean:.2f} | Std: {std:.3f} | n: {count}")

                        # 3. TRENDING CHART (MINIMAL: MAX, MEAN, MIN ONLY)
                        x_axis = analysis_df[coil_col].astype(str) if coil_col else analysis_df.index.astype(str)
                        d_max, d_min = data_series.max(), data_series.min()
                        
                        fig_trend = go.Figure()
                        
                        # Process Line
                        fig_trend.add_trace(go.Scatter(
                            x=x_axis, y=data_series, mode='lines+markers', 
                            line=dict(color='#4F81BD', width=2), 
                            marker=dict(size=6, color='white', line=dict(color='#4F81BD', width=2)),
                            name='Data'
                        ))
                        
                        # --- Horizontal Mean Line ---
                        fig_trend.add_hline(
                            y=mean, line_color="#333333", line_width=1.5, line_dash="dash",
                            annotation_text=f"<b>MEAN: {mean:.1f}</b>", annotation_position="right"
                        )
                        
                        # --- MAX Annotation ---
                        fig_trend.add_annotation(
                            x=analysis_df.loc[data_series.idxmax(), coil_col] if coil_col else str(data_series.idxmax()),
                            y=d_max, text=f"<b>MAX: {d_max:.1f}</b>",
                            showarrow=True, arrowhead=1, ax=0, ay=-35, font=dict(color="#1A5276", size=11),
                            bgcolor="white", bordercolor="#1A5276", borderwidth=1
                        )
                        
                        # --- MIN Annotation ---
                        fig_trend.add_annotation(
                            x=analysis_df.loc[data_series.idxmin(), coil_col] if coil_col else str(data_series.idxmin()),
                            y=d_min, text=f"<b>MIN: {d_min:.1f}</b>",
                            showarrow=True, arrowhead=1, ax=0, ay=35, font=dict(color="#922B21", size=11),
                            bgcolor="white", bordercolor="#922B21", borderwidth=1
                        )

                        fig_trend.update_layout(
                            title=dict(text=f"<b>{target_col} Trending</b>", x=0.5, xanchor='center'),
                            height=300, plot_bgcolor='#FFFFFF', margin=dict(l=20, r=100, t=30, b=20),
                            xaxis=dict(type='category', categoryorder='array', categoryarray=x_axis.tolist(), showgrid=False, showline=True, linecolor='black'),
                            yaxis=dict(showgrid=True, gridcolor='#F0F0F0', showline=True, linecolor='black')
                        )
                        st.plotly_chart(fig_trend, use_container_width=True)
                        st.markdown("<br><br>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Đã xảy ra lỗi: {e}")
else:
    st.info("Vui lòng tải file dữ liệu (Excel/CSV) lên để bắt đầu phân tích.")
