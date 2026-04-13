import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Process Capability Analysis (Ca, Cp, Cpk)")

# Upload file
uploaded_file = st.file_uploader("Upload dữ liệu CSV/XLSX", type=["csv", "xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Bộ lọc
    line = st.selectbox("Chọn LINE", df["LINE"].unique())
    grade = st.selectbox("Chọn 鋼種", df["鋼種"].unique())
    width = st.selectbox("Chọn 訂單寬度", df["訂單寬度"].unique())

    filtered_df = df[(df["LINE"] == line) & 
                     (df["鋼種"] == grade) & 
                     (df["訂單寬度"] == width)]

    st.write("Dữ liệu sau lọc:", filtered_df.head())

    # Chọn chỉ số cơ tính
    column = st.selectbox("Chọn chỉ số cơ tính", 
                          ["屈服强度", "抗拉强度", "伸长率", "Y/S", "N.VALUE", "HARDNESS"])

    usl = st.number_input("Nhập USL")
    lsl = st.number_input("Nhập LSL")
    target = st.number_input("Nhập Target")

    def calculate_indices(data, usl, lsl, target):
        mu = np.mean(data)
        sigma = np.std(data, ddof=1)
        ca = abs(mu - target) / ((usl - lsl)/2)
        cp = (usl - lsl) / (6 * sigma)
        cpk = min((usl - mu)/(3*sigma), (mu - lsl)/(3*sigma))
        return ca, cp, cpk

    def plot_distribution(data, usl, lsl):
        fig, ax = plt.subplots()
        sns.histplot(data, kde=True, ax=ax)
        ax.axvline(usl, color='r', linestyle='--', label="USL")
        ax.axvline(lsl, color='b', linestyle='--', label="LSL")
        ax.set_title("Biểu đồ phân bố")
        ax.legend()
        return fig

    def plot_trending(data):
        fig, ax = plt.subplots()
        ax.plot(data.values, marker='o')
        ax.set_title("Trending Line")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Value")
        return fig

    if st.button("Tính chỉ số"):
        ca, cp, cpk = calculate_indices(filtered_df[column], usl, lsl, target)
        st.success(f"Ca = {ca:.3f}, Cp = {cp:.3f}, Cpk = {cpk:.3f}")

        st.pyplot(plot_distribution(filtered_df[column], usl, lsl))
        st.pyplot(plot_trending(filtered_df[column]))
