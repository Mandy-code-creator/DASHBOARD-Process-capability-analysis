import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from scipy import stats

st.set_page_config(page_title="Steel Mechanical Property SPC", layout="wide")

st.title("📊 Quản lý Năng lực Quy trình Cơ tính Thép")

# 1. Tải dữ liệu
uploaded_file = st.file_uploader("Tải lên file dữ liệu (Excel hoặc CSV)", type=['csv', 'xlsx'])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Làm sạch dữ liệu theo quy tắc ledger của bạn (ví dụ loại bỏ GF)
    if 'Metallic_Type' in df.columns:
        df = df[df['Metallic_Type'].str.strip().upper() != 'GF']

    # 2. Bộ lọc (Sidebar)
    st.sidebar.header("Bộ lọc dữ liệu")
    lines = st.sidebar.multiselect("LINE", options=df['LINE'].unique(), default=df['LINE'].unique())
    grades = st.sidebar.multiselect("鋼種 (Grade)", options=df['鋼種'].unique(), default=df['鋼種'].unique())
    
    # Lọc độ rộng theo khoảng
    min_w, max_w = float(df['訂單寬度'].min()), float(df['訂單寬度'].max())
    width_range = st.sidebar.slider("訂單寬度 (Width)", min_w, max_w, (min_w, max_w))

    filtered_df = df[
        (df['LINE'].isin(lines)) & 
        (df['鋼種'].isin(grades)) & 
        (df['訂單寬度'].between(width_range[0], width_range[1]))
    ]

    # 3. Cấu hình SPC
    st.subheader("⚙️ Cấu hình chỉ số cơ tính")
    col_prop, col_target, col_lsl, col_usl = st.columns(4)
    
    with col_prop:
        target_col = st.selectbox("Chọn chỉ số phân tích", options=['Hardness_LINE', 'YS', 'TS', 'EL'])
    
    data_series = filtered_df[target_col].dropna()
    mean = data_series.mean()
    std = data_series.std()

    with col_target:
        target_val = st.number_input("Target Value", value=float(mean))
    with col_lsl:
        lsl = st.number_input("LSL", value=float(mean - 3*std))
    with col_usl:
        usl = st.number_input("USL", value=float(mean + 3*std))

    # 4. Tính toán
    if len(data_series) > 1:
        ca = (mean - target_val) / ((usl - lsl) / 2)
        cp = (usl - lsl) / (6 * std)
        cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))

        # Hiển thị Metric
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean (μ)", f"{mean:.2f}")
        m2.metric("Ca", f"{ca:.3f}")
        m3.metric("Cp", f"{cp:.3f}")
        m4.metric("Cpk", f"{cpk:.3f}", delta="Đạt" if cpk > 1.33 else "Kém", delta_color="normal")

        # 5. Biểu đồ
        tab1, tab2 = st.tabs(["Biểu đồ phân bố (Distribution)", "Biểu đồ xu hướng (Trend)"])

        with tab1:
            # Biểu đồ Histogram + Phân bố chuẩn
            fig_hist = px.histogram(filtered_df, x=target_col, nbins=30, marginal="box", 
                                  title=f"Phân bố của {target_col}", opacity=0.7)
            fig_hist.add_vline(x=lsl, line_dash="dash", line_color="red", annotation_text="LSL")
            fig_hist.add_vline(x=usl, line_dash="dash", line_color="red", annotation_text="USL")
            fig_hist.add_vline(x=target_val, line_color="green", annotation_text="Target")
            st.plotly_chart(fig_hist, use_container_width=True)

        with tab2:
            # Biểu đồ Trending Line
            fig_trend = px.line(filtered_df, y=target_col, title=f"Xu hướng biến động {target_col}", markers=True)
            fig_trend.add_hline(y=mean, line_color="blue", annotation_text="Average")
            st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("Không đủ dữ liệu để tính toán.")
