import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")

st.title("📊 QC Capability Analysis Dashboard")

# Upload file
uploaded_file = st.file_uploader("Upload Excel/CSV", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Raw Data", df.head())

    # Sidebar filter
    st.sidebar.header("Filter")

    line = st.sidebar.multiselect("LINE", df["LINE"].unique())
    steel = st.sidebar.multiselect("鋼種", df["鋼種"].unique())
    width = st.sidebar.multiselect("訂單寬度", df["訂單寬度"].unique())

    filtered_df = df.copy()

    if line:
        filtered_df = filtered_df[filtered_df["LINE"].isin(line)]
    if steel:
        filtered_df = filtered_df[filtered_df["鋼種"].isin(steel)]
    if width:
        filtered_df = filtered_df[filtered_df["訂單寬度"].isin(width)]

    st.write("### Filtered Data", filtered_df)

    # Select column
    numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
    selected_col = st.selectbox("Select Mechanical Property", numeric_cols)

    # Input spec
    col1, col2, col3 = st.columns(3)
    with col1:
        USL = st.number_input("USL")
    with col2:
        LSL = st.number_input("LSL")
    with col3:
        Target = st.number_input("Target")

    if selected_col:
        data = filtered_df[selected_col].dropna()

        mean = data.mean()
        std = data.std()

        Cp = (USL - LSL) / (6 * std) if std != 0 else 0
        Cpu = (USL - mean) / (3 * std) if std != 0 else 0
        Cpl = (mean - LSL) / (3 * std) if std != 0 else 0
        Cpk = min(Cpu, Cpl)
        Ca = (mean - Target) / ((USL - LSL) / 2) if (USL - LSL) != 0 else 0

        st.metric("Mean", round(mean, 2))
        st.metric("Std Dev", round(std, 2))
        st.metric("Cp", round(Cp, 2))
        st.metric("Cpk", round(Cpk, 2))
        st.metric("Ca", round(Ca, 2))

        # Histogram
        fig = px.histogram(data, nbins=30, title="Distribution")
        fig.add_vline(x=USL, line_dash="dash", line_color="red")
        fig.add_vline(x=LSL, line_dash="dash", line_color="red")
        fig.add_vline(x=mean, line_dash="dash", line_color="green")
        st.plotly_chart(fig, use_container_width=True)

        # Trend line
        if "日期" in filtered_df.columns:
            trend_df = filtered_df.sort_values("日期")

            fig2 = px.line(trend_df, x="日期", y=selected_col, title="Trend Line")
            fig2.add_hline(y=USL, line_dash="dash", line_color="red")
            fig2.add_hline(y=LSL, line_dash="dash", line_color="red")
            st.plotly_chart(fig2, use_container_width=True)

        # Box plot
        fig3 = px.box(filtered_df, y=selected_col, title="Box Plot")
        st.plotly_chart(fig3, use_container_width=True)
