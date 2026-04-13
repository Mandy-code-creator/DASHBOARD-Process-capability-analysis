import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Mechanical Property Grid View", layout="wide")

st.title("📊 Mechanical Property Comprehensive Analysis")

uploaded_file = st.file_uploader("Upload Data File (CSV or Excel)", type=['csv', 'xlsx'])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        df.columns = df.columns.str.strip()

        # Áp dụng logic quản lý: Gộp GE00/GE01 và loại bỏ GF
        if '鋼種' in df.columns:
            df['鋼種'] = df['鋼種'].replace(['GE00', 'GE01'], 'GE00/GE01')
        if 'Metallic_Type' in df.columns:
            df = df[df['Metallic_Type'].astype(str).str.strip().str.upper() != 'GF']

        # Bộ lọc hàng ngang
        st.markdown("### 🎛️ Filters")
        f1, f2, f3 = st.columns(3)
        with f1:
            lines = st.multiselect("LINE", options=df['LINE'].unique(), default=df['LINE'].unique())
        with f2:
            grades = st.multiselect("Steel Grade", options=df['鋼種'].unique(), default=df['鋼種'].unique())
        with f3:
            df['訂單寬度'] = pd.to_numeric(df['訂單寬度'], errors='coerce')
            u_widths = sorted(df['訂單寬度'].dropna().unique())
            sel_widths = st.multiselect("Order Width", options=u_widths, default=u_widths)

        filtered_df = df[(df['LINE'].isin(lines)) & (df['鋼種'].isin(grades)) & (df['訂單寬度'].isin(sel_widths))].copy()

        # Xác định các cột mục tiêu
        targets = ['TENSILE_YIELD', 'TENSILE_TENSILE', 'TENSILE_ELONG', 'skp+t/l']
        available_targets = [t for t in targets if t in filtered_df.columns]

        # Sắp xếp theo thời gian để trending line chính xác
        time_cols = [c for c in ['生產日期', '開始時間', 'Time'] if c in filtered_df.columns]
        if time_cols:
            filtered_df = filtered_df.sort_values(by=time_cols)

        # Tạo lưới hiển thị: 2 cột
        st.markdown("---")
        for i in range(0, len(available_targets), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(available_targets):
                    target = available_targets[i + j]
                    with cols[j]:
                        st.subheader(f"Analysis: {target}")
                        
                        data = filtered_df[target].dropna()
                        if len(data) < 2:
                            st.warning(f"Insufficient data for {target}")
                            continue
                        
                        mean, std = data.mean(), data.std()
                        lsl, usl = mean - 3*std, mean + 3*std # Mặc định 3-sigma
                        
                        # 1. Biểu đồ Phân bố (Distribution)
                        fig_dist = go.Figure()
                        fig_dist.add_trace(go.Histogram(x=data, histnorm='probability density', name='Data', marker_color='#4dabf7', opacity=0.7))
                        x_range = np.linspace(data.min(), data.max(), 100)
                        fig_dist.add_trace(go.Scatter(x=x_range, y=norm.pdf(x_range, mean, std), mode='lines', name='Normal', line=dict(color='#343a40', width=2)))
                        fig_dist.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
                        st.plotly_chart(fig_dist, use_container_width=True)

                        # 2. Thẻ chỉ số SPC (Metrics)
                        cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))
                        st.markdown(f"""
                        <div style="padding:10px; border-radius:5px; background-color:#f1f3f5; border-left: 5px solid #228be6; margin-bottom:10px">
                            <small>Mean: <b>{mean:.2f}</b> | Std: <b>{std:.3f}</b> | n: <b>{len(data)}</b></small><br>
                            <span style="color:#d9480f"><b>Cpk: {cpk:.3f}</b></span>
                        </div>
                        """, unsafe_allow_html=True)

                        # 3. Biểu đồ Trending Line
                        fig_trend = px.line(filtered_df, y=target, markers=True, color_discrete_sequence=['#2c3e50'])
                        fig_trend.add_hline(y=mean, line_color="#5bc0de", line_dash="dash")
                        fig_trend.update_layout(height=250, margin=dict(l=20, r=20, t=10, b=20), xaxis_title="Coil Sequence")
                        st.plotly_chart(fig_trend, use_container_width=True)
                        st.markdown("<br>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload production data to start.")
