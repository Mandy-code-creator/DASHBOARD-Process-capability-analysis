import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm

# Page Config
st.set_page_config(page_title="Mechanical Property SPC Dashboard", layout="wide", page_icon="📊")

st.title("📊 Mechanical Property Comprehensive Analysis")
st.markdown("Hệ thống tự động tính toán SPC và kiểm soát năng lực quy trình theo chuẩn giới hạn khách hàng.")

# 1. Upload Data
uploaded_file = st.file_uploader("Tải lên file Excel hoặc CSV", type=['csv', 'xlsx'])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        df.columns = df.columns.str.strip()

        # --- DATA CORRECTIONS (Based on factory rules) ---
        if '鋼種' in df.columns:
            df['鋼種'] = df['鋼種'].replace(['GE00', 'GE01'], 'GE00/GE01')
        if 'Metallic_Type' in df.columns:
            df = df[df['Metallic_Type'].astype(str).str.strip().str.upper() != 'GF']

        # --- EXTRACT YEAR (年度) FROM DATE ---
        time_cols_for_year = [c for c in ['生產日期', '開始時間', 'Time', 'Date'] if c in df.columns]
        if time_cols_for_year:
            main_time_col = time_cols_for_year[0]
            df['年度'] = pd.to_datetime(df[main_time_col], errors='coerce').dt.year
            valid_years = df['年度'].dropna().astype(int).astype(str).unique().tolist()
            year_options = sorted(valid_years, reverse=True) 
            df['年度'] = df['年度'].fillna(-1).astype(int).astype(str) 
        else:
            df['年度'] = 'N/A'
            year_options = ['N/A']

        # --- 🎛️ GLOBAL FILTERS ---
        st.markdown("### 🎛️ Global Data Filters")
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        
        with col_f1:
            selected_years = st.multiselect("Year (年度)", options=year_options, default=year_options)
        with col_f2:
            lines = st.multiselect("Factory Line (LINE)", options=df['LINE'].dropna().unique(), default=df['LINE'].dropna().unique())
        with col_f3:
            grades = st.multiselect("Steel Grade (鋼種)", options=df['鋼種'].dropna().unique(), default=df['鋼種'].dropna().unique())
        with col_f4:
            df['訂單寬度'] = pd.to_numeric(df['訂單寬度'], errors='coerce')
            unique_widths = sorted(df['訂單寬度'].dropna().unique())
            selected_widths = st.multiselect("Order Width (訂單寬度)", options=unique_widths, default=unique_widths)

        # APPLY FILTERS
        filtered_df = df[
            (df['年度'].isin(selected_years)) &
            (df['LINE'].isin(lines)) & 
            (df['鋼種'].isin(grades)) & 
            (df['訂單寬度'].isin(selected_widths))
        ].copy()

        # --- IDENTIFY COIL COLUMN & SORT ---
        coil_col = next((c for c in ['COIL_NO', 'COIL NO', 'Coil_No', '製造批號', 'Batch'] if c in filtered_df.columns), None)
        time_cols = [c for c in ['生產日期', '開始時間', 'Time', 'Date'] if c in filtered_df.columns]
        sort_cols = time_cols + ([coil_col] if coil_col else [])
        if sort_cols:
            filtered_df = filtered_df.sort_values(by=sort_cols).reset_index(drop=True)

        # Identify Target Columns
        potential_targets = ['YS', 'TS', 'EL', 'TENSILE_YIELD', 'TENSILE_TENSILE', 'TENSILE_ELONG', 'skp+t/l', 'HARDNESS', 'HRB', 'HRC', 'HV']
        available_targets = [c for c in potential_targets if c in filtered_df.columns]

        # 2. Specification Limits Setting
        with st.expander("⚙️ Customer Specification Settings (LSL / USL)", expanded=False):
            st.info("Các giới hạn đã được tự động điền. Bạn có thể chỉnh sửa nếu cần.")
            specs = {}
            for target in available_targets:
                def_lsl, def_usl, def_tgt = 0.0, 0.0, 0.0
                t_upper = target.upper()
                
                if 'YS' in t_upper or 'YIELD' in t_upper:
                    def_lsl, def_usl = 230.0, 460.0
                    def_tgt = (230.0 + 460.0) / 2
                elif 'TS' in t_upper or 'TENSILE_TENSILE' in t_upper:
                    def_lsl, def_usl = 310.0, 550.0
                    def_tgt = (310.0 + 550.0) / 2
                elif 'EL' in t_upper or 'ELONG' in t_upper:
                    def_lsl, def_usl = 20.0, 0.0 
                    def_tgt = 0.0 
                elif 'HARDNESS' in t_upper or 'HRB' in t_upper or 'HRC' in t_upper or 'HV' in t_upper:
                    def_lsl, def_usl = 0.0, 78.0 
                    def_tgt = 0.0 

                st.markdown(f"**Settings for {target}**")
                sc1, sc2, sc3 = st.columns(3)
                with sc1: t_val = st.number_input(f"Target ({target})", value=float(def_tgt), key=f"t_{target}")
                with sc2: l_val = st.number_input(f"LSL ({target})", value=float(def_lsl), key=f"l_{target}")
                with sc3: u_val = st.number_input(f"USL ({target})", value=float(def_usl), key=f"u_{target}")
                specs[target] = {'tgt': t_val, 'lsl': l_val, 'usl': u_val}

        # --- DRAW GRID VIEW ---
        st.markdown("---")
        for i in range(0, len(available_targets), 2):
            grid_cols = st.columns(2)
            for j in range(2):
                if i + j < len(available_targets):
                    target_col = available_targets[i + j]
                    with grid_cols[j]:
                        # Data Cleaning
                        analysis_df = filtered_df.dropna(subset=[target_col]).copy()
                        analysis_df[target_col] = pd.to_numeric(analysis_df[target_col], errors='coerce')
                        analysis_df = analysis_df.dropna(subset=[target_col])
                        if coil_col:
                            analysis_df = analysis_df.drop_duplicates(subset=[coil_col], keep='last')
                        
                        data_series = analysis_df[target_col]
                        if len(data_series) < 2:
                            st.warning(f"Không đủ dữ liệu cho {target_col}")
                            continue
                        
                        # Basic Stats
                        mean, std, count = data_series.mean(), data_series.std(), len(data_series)
                        d_max, d_min = data_series.max(), data_series.min()
                        ucl, lcl = mean + 3*std, mean - 3*std
                        
                        # Fetch Specs
                        lsl_in, usl_in, tgt_in = specs[target_col]['lsl'], specs[target_col]['usl'], specs[target_col]['tgt']
                        
                        # --- SPC LOGIC (TỔNG THỂ) ---
                        if lsl_in == 0 and usl_in == 0:
                            ca_dis, cp_dis, cpk_dis = "N/A", "N/A", "N/A"
                            status_color = "#6c757d"
                            spec_active = False
                        elif usl_in == 0 and lsl_in > 0: # Min Only
                            cpk = (mean - lsl_in) / (3 * std) if std > 0 else 0
                            ca_dis, cp_dis, cpk_dis = "N/A", "N/A", f"{cpk:.3f}"
                            status_color = "#28a745" if cpk >= 1.33 else "#dc3545"
                            spec_active = True
                        elif lsl_in == 0 and usl_in > 0: # Max Only
                            cpk = (usl_in - mean) / (3 * std) if std > 0 else 0
                            ca_dis, cp_dis, cpk_dis = "N/A", "N/A", f"{cpk:.3f}"
                            status_color = "#28a745" if cpk >= 1.33 else "#dc3545"
                            spec_active = True
                        else: # Hai phía
                            ca = ((mean - tgt_in) / ((usl_in - lsl_in) / 2)) * 100 if usl_in != lsl_in else 0
                            cp = (usl_in - lsl_in) / (6 * std) if std > 0 else 0
                            cpk = min((usl_in - mean)/(3*std), (mean - lsl_in)/(3*std)) if std > 0 else 0
                            ca_dis, cp_dis, cpk_dis = f"{ca:.1f}%", f"{cp:.3f}", f"{cpk:.3f}"
                            status_color = "#28a745" if cpk >= 1.33 else "#dc3545"
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
                            if lsl_in > 0:
                                fig_dist.add_vline(x=lsl_in, line_color="red", line_width=2)
                                fig_dist.add_annotation(x=lsl_in, y=0.95, yref='paper', text=f"LSL: {lsl_in:.1f}", showarrow=False, font=dict(color="red", size=10), xanchor="right", xshift=-5)
                            if usl_in > 0:
                                fig_dist.add_vline(x=usl_in, line_color="red", line_width=2)
                                fig_dist.add_annotation(x=usl_in, y=0.95, yref='paper', text=f"USL: {usl_in:.1f}", showarrow=False, font=dict(color="red", size=10), xanchor="left", xshift=5)
                        
                        fig_dist.add_vline(x=mean, line_color="#333", line_dash="dash", line_width=1.5)
                        fig_dist.add_annotation(x=mean, y=1.05, yref='paper', text=f"Mean: {mean:.1f}", showarrow=False, font=dict(color="#333", size=10))
                        
                        fig_dist.add_vline(x=ucl, line_color="#FF8C00", line_dash="dash", line_width=1.5)
                        fig_dist.add_annotation(x=ucl, y=0.85, yref='paper', text=f"UCL: {ucl:.1f}", showarrow=False, font=dict(color="#FF8C00", size=10))
                        fig_dist.add_vline(x=lcl, line_color="#FF8C00", line_dash="dash", line_width=1.5)
                        fig_dist.add_annotation(x=lcl, y=0.85, yref='paper', text=f"LCL: {lcl:.1f}", showarrow=False, font=dict(color="#FF8C00", size=10))

                        fig_dist.update_layout(
                            title=dict(text=f"<b>{target_col} Distribution</b>", x=0.5, xanchor='center'),
                            height=380, plot_bgcolor='white', margin=dict(l=40, r=40, b=100, t=100),
                            legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                            xaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True, gridcolor='#F0F0F0'),
                            yaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True, gridcolor='#F0F0F0')
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)

                        # 2. METRICS CARD (TỔNG THỂ)
                        st.markdown(f"""
                        <div style="padding:10px; border-radius:8px; border-left: 6px solid {status_color}; background-color:#f8f9fa; margin-bottom:10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                            <span style="font-size:15px;"><b>Overall Cpk: {cpk_dis}</b> | <b>Cp: {cp_dis}</b> | <b>Ca: {ca_dis}</b></span><br>
                            <span style="color:#666; font-size:12px;">Mean: {mean:.2f} | Std: {std:.3f} | n: {count}</span>
                        </div>
                        """, unsafe_allow_html=True)

                        # 3. TRENDING CHART
                        x_axis = analysis_df[coil_col].astype(str) if coil_col else analysis_df.index.astype(str)
                        fig_trend = go.Figure()
                        fig_trend.add_trace(go.Scatter(x=x_axis, y=data_series, mode='lines+markers', line=dict(color='#4F81BD', width=2), marker=dict(size=6, color='white', line=dict(color='#4F81BD', width=2))))
                        
                        fig_trend.add_hline(y=mean, line_color="#333", line_width=1.5, annotation_text=f"Mean: {mean:.1f}", annotation_position="right")
                        fig_trend.add_hline(y=ucl, line_color="#FF8C00", line_width=1.5, line_dash="dash", annotation_text=f"UCL: {ucl:.1f}", annotation_position="right")
                        fig_trend.add_hline(y=lcl, line_color="#FF8C00", line_width=1.5, line_dash="dash", annotation_text=f"LCL: {lcl:.1f}", annotation_position="right")
                        
                        fig_trend.update_layout(
                            title=dict(text=f"<b>{target_col} Trend</b>", x=0.5, xanchor='center'),
                            height=300, plot_bgcolor='#F9F9F9', margin=dict(l=40, r=120, t=30, b=40),
                            xaxis=dict(type='category', showgrid=False, linecolor='black', tickangle=45),
                            yaxis=dict(showgrid=True, gridcolor='#E0E0E0', showline=True, linecolor='black')
                        )
                        st.plotly_chart(fig_trend, use_container_width=True)

                        # 4. BẢNG SO SÁNH THEO NĂM (YEAR-OVER-YEAR TABLE)
                        if '年度' in analysis_df.columns:
                            yoy_data = []
                            # Loại bỏ các dòng không có năm hợp lệ
                            valid_year_df = analysis_df[analysis_df['年度'] != '-1']
                            years = sorted(valid_year_df['年度'].unique(), reverse=True)
                            
                            for y in years:
                                ydf = valid_year_df[valid_year_df['年度'] == y]
                                y_mean, y_std, y_count = ydf[target_col].mean(), ydf[target_col].std(), len(ydf)
                                
                                # Tính SPC riêng cho từng năm
                                if lsl_in == 0 and usl_in == 0:
                                    y_ca_dis, y_cp_dis, y_cpk_dis = "N/A", "N/A", "N/A"
                                elif usl_in == 0 and lsl_in > 0: 
                                    y_cpk = (y_mean - lsl_in) / (3 * y_std) if y_std > 0 else 0
                                    y_ca_dis, y_cp_dis, y_cpk_dis = "N/A", "N/A", f"{y_cpk:.3f}"
                                elif lsl_in == 0 and usl_in > 0: 
                                    y_cpk = (usl_in - y_mean) / (3 * y_std) if y_std > 0 else 0
                                    y_ca_dis, y_cp_dis, y_cpk_dis = "N/A", "N/A", f"{y_cpk:.3f}"
                                else: 
                                    y_ca = ((y_mean - tgt_in) / ((usl_in - lsl_in) / 2)) * 100 if usl_in != lsl_in else 0
                                    y_cp = (usl_in - lsl_in) / (6 * y_std) if y_std > 0 else 0
                                    y_cpk = min((usl_in - y_mean)/(3*y_std), (y_mean - lsl_in)/(3*y_std)) if y_std > 0 else 0
                                    y_ca_dis, y_cp_dis, y_cpk_dis = f"{y_ca:.1f}%", f"{y_cp:.3f}", f"{y_cpk:.3f}"
                                    
                                yoy_data.append({
                                    "Năm (年度)": y,
                                    "n (Mẫu)": y_count,
                                    "Cpk": y_cpk_dis,
                                    "Cp": y_cp_dis,
                                    "Ca": y_ca_dis,
                                    "Mean": f"{y_mean:.2f}",
                                    "Std": f"{y_std:.3f}"
                                })
                                
                            if yoy_data:
                                yoy_df = pd.DataFrame(yoy_data)
                                st.markdown(f"<p style='font-size:14px; font-weight:bold; color:#1F497D; margin-bottom:5px;'>📊 Bảng So sánh Năng lực theo Năm</p>", unsafe_allow_html=True)
                                st.dataframe(yoy_df, use_container_width=True, hide_index=True)
                                
                        st.markdown("<br><br>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Critical Error: {e}")
else:
    st.info("Vui lòng tải file dữ liệu lên để bắt đầu.")
