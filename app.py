import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Process Capability (SPC)", layout="wide", page_icon="📊")

st.title("📊 Mechanical Property Capability Analysis")
st.markdown("Upload your production data to calculate Ca, Cp, Cpk and visualize distribution and trends.")

# Upload file
uploaded_file = st.file_uploader("Upload Data File (CSV or Excel)", type=['csv', 'xlsx'])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df.columns = df.columns.str.strip()

    # Bộ lọc
    req_filters = ['LINE', '鋼種', '訂單寬度']
    missing_filters = [c for c in req_filters if c not in df.columns]
    if missing_filters:
        st.error(f"Missing required filter columns: {', '.join(missing_filters)}")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        lines = st.multiselect("LINE", df['LINE'].unique(), default=df['LINE'].unique())
    with col2:
        grades = st.multiselect("鋼種", df['鋼種'].unique(), default=df['鋼種'].unique())
    with col3:
        df['訂單寬度'] = pd.to_numeric(df['訂單寬度'], errors='coerce')
        widths = sorted(df['訂單寬度'].dropna().unique())
        selected_widths = st.multiselect("訂單寬度", widths, default=widths)

    filtered_df = df[(df['LINE'].isin(lines)) & (df['鋼種'].isin(grades)) & (df['訂單寬度'].isin(selected_widths))]

    # Chọn chỉ số cơ tính
    numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found.")
        st.stop()

    column = st.selectbox("Chọn chỉ số cơ tính", numeric_cols)

    target = st.number_input("Target", value=float(filtered_df[column].mean()))
    lsl = st.number_input("LSL", value=float(filtered_df[column].mean() - 3*filtered_df[column].std()))
    usl = st.number_input("USL", value=float(filtered_df[column].mean() + 3*filtered_df[column].std()))

    data_series = filtered_df[column].dropna()
    mean = data_series.mean()
    std = data_series.std()

    ca = abs(mean - target) / ((usl - lsl)/2) if usl != lsl else 0
    cp = (usl - lsl) / (6*std) if std > 0 else 0
    cpk = min((usl - mean)/(3*std), (mean - lsl)/(3*std)) if std > 0 else 0

    st.markdown(f"**Ca = {ca:.3f}, Cp = {cp:.3f}, Cpk = {cpk:.3f}**")

    # Biểu đồ phân bố
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=data_series, histnorm='probability density', name='Data', marker_color='#4dabf7'))
    x_curve = np.linspace(data_series.min(), data_series.max(), 500)
    fig_dist.add_trace(go.Scatter(x=x_curve, y=norm.pdf(x_curve, mean, std), mode='lines', name='Normal Curve', line=dict(color='black')))
    fig_dist.add_vline(x=lsl, line_dash="dash", line_color="red")
    fig_dist.add_annotation(x=lsl, y=0.02, text="LSL", showarrow=False, font=dict(color="red"))
    fig_dist.add_vline(x=usl, line_dash="dash", line_color="red")
    fig_dist.add_annotation(x=usl, y=0.02, text="USL", showarrow=False, font=dict(color="red"))
    fig_dist.add_vline(x=target, line_color="green")
    fig_dist.add_annotation(x=target, y=0.02, text="Target", showarrow=False, font=dict(color="green"))
    fig_dist.update_layout(title="Distribution with Spec Limits", bargap=0.05)
    st.plotly_chart(fig_dist, use_container_width=True)

    # Biểu đồ trending line
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(y=data_series, mode='lines+markers', name='Process Data', line=dict(color='blue')))
    fig_trend.add_hline(y=lsl, line_color="red", line_dash="dash")
    fig_trend.add_annotation(x=0, y=lsl, text="LSL", showarrow=False, font=dict(color="red"))
    fig_trend.add_hline(y=usl, line_color="red", line_dash="dash")
    fig_trend.add_annotation(x=0, y=usl, text="USL", showarrow=False, font=dict(color="red"))
    fig_trend.add_hline(y=target, line_color="green", line_dash="dot")
    fig_trend.add_annotation(x=0, y=target, text="Target", showarrow=False, font=dict(color="green"))
    fig_trend.update_layout(title="Trending Line with Spec Limits", xaxis_title="Sample Index", yaxis_title=column)
    st.plotly_chart(fig_trend, use_container_width=True)
