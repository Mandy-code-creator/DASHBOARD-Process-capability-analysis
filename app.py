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
        
        # Clean column names
        df.columns = df.columns.str.strip()

        # --- INVISIBLE DATA CORRECTIONS ---
        if '鋼種' in df.columns:
            df['鋼種'] = df['鋼種'].replace(['GE00', 'GE01'], 'GE00/GE01')
        
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

        # --- IDENTIFY COIL COLUMN & SORT STRICTLY ---
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
        
        potential_targets = ['YS', 'TS', 'EL', 'TENSILE_YIELD', 'TENSILE_TENSILE', 'TENSILE_ELONG', 'skp+t/l']
        available_targets = [c for c in potential_targets if c in df.columns]
        
        if not available_targets:
            available_targets = filtered_df.select_dtypes(include=np.number).columns.tolist()

        if not available_targets:
            st.error("No numeric target columns found for analysis.")
            st.stop()

        with st.expander("⚙️ Specification Limits Settings (LSL / USL / Target)", expanded=False):
            st.markdown("Adjust the specifications for each parameter. By default, limits are estimated using $\pm 3\sigma$.")
            specs = {}
            for target in available_targets:
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

        # --- DRAW PARALLEL CHARTS ---
        st.markdown("---")
        
        for i in range(0, len(available_targets), 2):
            grid_cols = st.columns(2)
            
            for j in range(2):
                if i + j < len(available_targets):
                    target_col = available_targets[i + j]
                    col = grid_cols[j]
                    
                    with col:
                        # Clean data
                        analysis_df = filtered_df.dropna(subset=[target_col]).copy()
                        analysis_df[target_col] = pd.to_numeric(analysis_df[target_col], errors='coerce')
                        analysis_df = analysis_df.dropna(subset=[target_col])
                        
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

                        st.subheader(f"Analysis: {target_col}")

                        # =====================================================================
                        # 1. DISTRIBUTION CHART (Stacked, Boxed, Dark Blue + Normal Curve)
                        # =====================================================================
                        has_grade = '鋼種' in analysis_df.columns and analysis_df['鋼種'].nunique() > 0
                        dark_blue_palette = ['#4F81BD', '#1F497D', '#8DB4E2', '#B8CCE4', '#003366']

                        if has_grade:
                            fig_dist = px.histogram(
                                analysis_df, x=target_col, color='鋼種', nbins=20, 
                                barmode='stack', color_discrete_sequence=dark_blue_palette
                            )
                        else:
                            fig_dist = px.histogram(
                                analysis_df, x=target_col, nbins=20, color_discrete_sequence=['#4F81BD']
                            )
                            
                        fig_dist.update_traces(marker_line_color='white', marker_line_width=1.5, opacity=0.9)

                        # Add Normal Curve
                        if std > 0:
                            x_curve = np.linspace(data_series.min() - 1*std, data_series.max() + 1*std, 500)
                            bin_width = (data_series.max() - data_series.min()) / 20 if data_series.max() > data_series.min() else 1
                            y_curve = norm.pdf(x_curve, mean, std) * count * bin_width
                            
                            fig_dist.add_trace(go.Scatter(
                                x=x_curve, y=y_curve, mode='lines', name='Normal Curve', 
                                line=dict(color='#FFB300', width=2.5), showlegend=False
                            ))

                        # LSL & USL (Solid Red Lines) - NO TGT LINE
                        fig_dist.add_vline(x=lsl, line_width=2, line_dash="solid", line_color="red")
                        fig_dist.add_annotation(x=lsl, y=0.98, yref='paper', text=f"<b>LSL<br>{lsl:.0f}</b>", showarrow=False, font=dict(color="red", size=11), xanchor="right", xshift=-5)
                        
                        fig_dist.add_vline(x=usl, line_width=2, line_dash="solid", line_color="red")
                        fig_dist.add_annotation(x=usl, y=0.98, yref='paper', text=f"<b>USL<br>{usl:.0f}</b>", showarrow=False, font=dict(color="red", size=11), xanchor="left", xshift=5)

                        # Mean Boxes
                        if has_grade:
                            for idx_trace, trace in enumerate(fig_dist.data):
                                if trace.name == 'Normal Curve': continue 
                                grade_mean = analysis_df[analysis_df['鋼種'] == trace.name][target_col].mean()
                                if pd.notna(grade_mean):
                                    fig_dist.add_vline(x=grade_mean, line_width=1.5, line_dash="dash", line_color="#333333")
                                    y_pos = 0.80 - (idx_trace % 5) * 0.12 
                                    fig_dist.add_annotation(
                                        x=grade_mean, y=y_pos, yref='paper', text=f"<b>{grade_mean:.1f}</b>",
                                        showarrow=False, font=dict(color="white", size=10),
                                        bgcolor=trace.marker.color, bordercolor="black", borderwidth=1, borderpad=3, xanchor="center"
                                    )
                        else:
                            fig_dist.add_vline(x=mean, line_width=1.5, line_dash="dash", line_color="#333333")
                            fig_dist.add_annotation(
                                x=mean, y=0.80, yref='paper', text=f"<b>{mean:.1f}</b>", showarrow=False, font=dict(color="white", size=10),
                                bgcolor="#4F81BD", bordercolor="black", borderwidth=1, borderpad=3, xanchor="center"
                            )

                        fig_dist.update_layout(
                            title=dict(text=f"<b>{target_col} (Overall)</b>", font=dict(size=14, color='black'), x=0.5, xanchor='center'),
                            height=380, margin=dict(l=20, r=20, t=40, b=60), showlegend=True,
                            legend=dict(title="", font=dict(size=10), orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
                            plot_bgcolor='white', xaxis_title="", yaxis_title="",
                            xaxis=dict(showgrid=False, zeroline=False, showline=True, linewidth=1, linecolor='black', mirror=True),
                            yaxis=dict(showgrid=False, zeroline=False, showline=True, linewidth=1, linecolor='black', mirror=True)
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)

                        # =====================================================================
                        # 2. HTML METRICS CARD
                        # =====================================================================
                        is_capable = cpk >= 1.33
                        icon = "✅" if is_capable else "❌"
                        status_text = "Capable" if is_capable else "Not Capable"
                        border_color = "#28a745" if is_capable else "#dc3545"
                        
                        card_html = f"""
                        <div style="border: 1px solid #e0e0e0; border-left: 6px solid {border_color}; border-radius: 8px; background-color: #fafafa; padding: 15px; margin-bottom: 10px; box-shadow: 2px 2px 8px rgba(0,0,0,0.05);">
                            <div style="display: flex; align-items: center; border-bottom: 1px solid #ddd; padding-bottom: 8px; margin-bottom: 10px; flex-wrap: wrap; gap: 10px;">
                                <span style="font-size: 16px; font-weight: 700; color: {border_color}; min-width: 130px;">{icon} {status_text}</span>
                                <span style="color: #aaa;">|</span>
                                <span style="font-family: monospace; font-size: 13px; color: #333;"><b>LSL:</b> {lsl:.0f} &nbsp;&nbsp; <b>USL:</b> {usl:.0f}</span>
                                <span style="color: #aaa;">|</span>
                                <span style="font-family: monospace; font-size: 13px; color: #333;"><b>n:</b> {count} &nbsp;&nbsp; <b>Mean:</b></span>
                            </div>
                            <div style="font-family: monospace; font-size: 13px; color: #333; margin-bottom: 12px; margin-left: 5px;">
                                {mean:.2f} &nbsp;&nbsp; <b>Std:</b> {std:.3f}
                            </div>
                            <div style="display: flex; gap: 20px; font-family: monospace; font-size: 14px; margin-left: 5px;">
                                <span style="color: #d9534f; font-weight: bold;">Cpk = {cpk:.3f}</span>
                                <span style="font-weight: bold; color: #333;">Cp = {cp:.3f}</span>
                                <span style="font-weight: bold; color: #333;">Ca = {ca:.1f}%</span>
                            </div>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)

                        # =====================================================================
                        # 3. TREND CHART (With Shading, Regression, and MAX/MIN/MEAN)
                        # =====================================================================
                        x_data = analysis_df[coil_col].astype(str) if coil_col else analysis_df.index.astype(str)
                        y_data = analysis_df[target_col].values

                        # Data Extremes
                        d_max = data_series.max()
                        d_min = data_series.min()
                        idx_max = data_series.idxmax()
                        idx_min = data_series.idxmin()
                        x_max = analysis_df.loc[idx_max, coil_col] if coil_col else str(idx_max)
                        x_min = analysis_df.loc[idx_min, coil_col] if coil_col else str(idx_min)

                        # Regression trend line
                        xi = np.arange(len(y_data))
                        z = np.polyfit(xi, y_data, 1)
                        trend_line = np.poly1d(z)(xi)

                        oos_mask = (y_data < lsl) | (y_data > usl)

                        fig_trend = go.Figure()

                        # ±1σ / ±2σ / ±3σ background shading
                        if std > 0:
                            for x0, x1, fill in [
                                (mean - 3*std, mean - 2*std, 'rgba(239,68,68,0.07)'),
                                (mean + 2*std, mean + 3*std, 'rgba(239,68,68,0.07)'),
                                (mean - 2*std, mean - 1*std, 'rgba(245,158,11,0.07)'),
                                (mean + 1*std, mean + 2*std, 'rgba(245,158,11,0.07)'),
                                (mean - 1*std, mean + 1*std, 'rgba(34,197,94,0.07)'),
                            ]:
                                fig_trend.add_hrect(y0=x0, y1=x1, fillcolor=fill, layer='below', line_width=0)

                        # Main process line
                        fig_trend.add_trace(go.Scatter(
                            x=x_data, y=y_data, mode='lines+markers', name='Process Data',
                            line=dict(color='#94a3b8', width=1.5),
                            marker=dict(size=6, color=['#ef4444' if m else '#3b82f6' for m in oos_mask], line=dict(width=0))
                        ))

                        # OOC cross markers
                        if oos_mask.any():
                            fig_trend.add_trace(go.Scatter(
                                x=x_data[oos_mask], y=y_data[oos_mask], mode='markers',
                                marker=dict(color='#ef4444', size=10, symbol='x', line=dict(width=2, color='#ef4444')), name='OOC'
                            ))

                        # Regression trend line
                        fig_trend.add_trace(go.Scatter(
                            x=x_data, y=trend_line, mode='lines', name='Trend', line=dict(color='#8b5cf6', width=2, dash='dash')
                        ))

                        # Annotate Data Max / Min
                        fig_trend.add_annotation(
                            x=x_max, y=d_max, text=f"<b>MAX: {d_max:.1f}</b>",
                            showarrow=True, arrowhead=1, ax=0, ay=-30, font=dict(color="green", size=10)
                        )
                        fig_trend.add_annotation(
                            x=x_min, y=d_min, text=f"<b>MIN: {d_min:.1f}</b>",
                            showarrow=True, arrowhead=1, ax=0, ay=30, font=dict(color="red", size=10)
                        )

                        # Reference lines: Mean, LSL, USL, ±3σ (Control Limits)
                        ref_lines = [
                            (mean, f'MEAN = {mean:.1f}', '#3b82f6', 'solid'),
                            (lsl, f'LSL = {lsl:.1f}', '#ef4444', 'dash'),
                            (usl, f'USL = {usl:.1f}', '#ef4444', 'dash'),
                        ]
                        if std > 0:
                            ref_lines += [
                                (mean + 3*std, f'UCL (+3σ) = {mean+3*std:.1f}', '#f59e0b', 'dot'),
                                (mean - 3*std, f'LCL (-3σ) = {mean-3*std:.1f}', '#f59e0b', 'dot'),
                            ]

                        for y_val, label, color, dash in ref_lines:
                            fig_trend.add_hline(
                                y=y_val, line=dict(color=color, width=1.5, dash=dash),
                                annotation=dict(
                                    text=f"<b>{label}</b>", font=dict(color=color, size=10, family='Arial'),
                                    bgcolor='rgba(255,255,255,0.85)', xanchor='right'
                                ),
                                annotation_position='right',
                            )

                        fig_trend.update_layout(
                            height=320, margin=dict(l=20, r=110, t=10, b=20),
                            showlegend=False, paper_bgcolor='white', plot_bgcolor='white',
                            xaxis=dict(
                                type='category', categoryorder='array', categoryarray=x_data.tolist(),
                                showgrid=False, linecolor='#e2e8f0', tickfont=dict(size=9, color='#94a3b8'),
                                title=dict(text='Coil Sequence', font=dict(size=11, color='#94a3b8')), nticks=15
                            ),
                            yaxis=dict(
                                showgrid=True, gridcolor='#f1f5f9', linecolor='#e2e8f0',
                                title=dict(text=target_col, font=dict(size=11, color='#64748b'))
                            )
                        )
                        st.plotly_chart(fig_trend, use_container_width=True)
                        st.markdown("<br>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a file to begin analysis.")
