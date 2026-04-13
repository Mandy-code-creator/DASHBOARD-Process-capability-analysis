import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Process Capability (SPC)", layout="wide", page_icon="📊")

st.title("📊 Process Capability Analysis (SPC)")
st.markdown("Upload your production data to automatically calculate Ca, Cp, Cpk and visualize the process distribution.")

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
        # Create grid columns for parallel side-by-side display
        grid_cols = st.columns(2)
        
        for idx, target_col in enumerate(available_targets):
            # Select column (Left or Right)
            col = grid_cols[idx % 2]
            
            with col:
                analysis_df = filtered_df.dropna(subset=[target_col]).copy()
                analysis_df[target_col] = pd.to_numeric(analysis_df[target_col], errors='coerce')
                analysis_df = analysis_df.dropna(subset=[target_col])
                
                data_series = analysis_df[target_col]
                
                if len(data_series) < 2:
                    st.warning(f"Not enough data for {target_col}.")
                    continue

                mean = data_series.mean()
                std = data_series.std()
                count = len(data_series)

                # Fetch specs
                target_val = specs[target_col]['tgt']
                lsl = specs[target_col]['lsl']
                usl = specs[target_col]['usl']

                # Calculations
                ca = ((mean - target_val) / ((usl - lsl) / 2)) * 100 if usl != lsl else 0
                cp = (usl - lsl) / (6 * std) if std > 0 else 0
                cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std)) if std > 0 else 0

                # Plot Histogram colored by Grade
                fig = px.histogram(
                    analysis_df, 
                    x=target_col, 
                    color='鋼種',
                    nbins=20, 
                    title=f"<b>{target_col} (Overall)</b>",
                    barmode='stack',
                    opacity=0.8,
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                
                # Add Vertical Lines (LSL, USL, Mean, Target)
                fig.add_vline(x=lsl, line_width=2.5, line_dash="solid", line_color="#d9534f")
                fig.add_annotation(x=lsl, y=1, yref='paper', text=f"<b>LSL<br>{lsl:.0f}</b>", showarrow=False, font=dict(color="#d9534f", size=12), xanchor="right", xshift=-5)

                fig.add_vline(x=usl, line_width=2.5, line_dash="solid", line_color="#d9534f")
                fig.add_annotation(x=usl, y=1, yref='paper', text=f"<b>USL<br>{usl:.0f}</b>", showarrow=False, font=dict(color="#d9534f", size=12), xanchor="left", xshift=5)

                fig.add_vline(x=mean, line_width=2, line_dash="dash", line_color="#0275d8")
                fig.add_annotation(x=mean, y=0.9, yref='paper', text=f"Mean<br>{mean:.1f}", showarrow=False, font=dict(color="white", size=11), bgcolor="#0275d8", borderpad=2, yshift=10)
                
                fig.add_vline(x=target_val, line_width=2, line_dash="dot", line_color="#5cb85c")
                fig.add_annotation(x=target_val, y=0.8, yref='paper', text=f"TGT<br>{target_val:.1f}", showarrow=False, font=dict(color="white", size=11), bgcolor="#5cb85c", borderpad=2, yshift=10)

                fig.update_layout(
                    xaxis_title="",
                    yaxis_title="",
                    showlegend=True,
                    legend_title_text="",
                    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.8)", bordercolor="Black", borderwidth=1),
                    margin=dict(l=20, r=20, t=50, b=20)
                )

                st.plotly_chart(fig, use_container_width=True)

                # --- HTML METRICS CARD ---
                is_capable = cpk >= 1.33
                icon = "✅" if is_capable else "❌"
                status_text = "Capable" if is_capable else "Not Capable"
                border_color = "#28a745" if is_capable else "#dc3545"
                
                card_html = f"""
                <div style="border: 1px solid #e0e0e0; border-left: 6px solid {border_color}; border-radius: 8px; background-color: #fafafa; padding: 15px; margin-bottom: 30px; box-shadow: 2px 2px 8px rgba(0,0,0,0.05);">
                    <div style="display: flex; align-items: center; border-bottom: 1px solid #ddd; padding-bottom: 8px; margin-bottom: 10px; flex-wrap: wrap; gap: 10px;">
                        <span style="font-size: 18px; font-weight: 700; color: {border_color}; min-width: 140px;">{icon} {status_text}</span>
                        <span style="color: #aaa;">|</span>
                        <span style="font-family: monospace; font-size: 14px; color: #333;"><b>LSL:</b> {lsl:.0f} &nbsp;&nbsp; <b>USL:</b> {usl:.0f}</span>
                        <span style="color: #aaa;">|</span>
                        <span style="font-family: monospace; font-size: 14px; color: #333;"><b>n:</b> {count} &nbsp;&nbsp; <b>Mean:</b></span>
                    </div>
                    <div style="font-family: monospace; font-size: 14px; color: #333; margin-bottom: 12px; margin-left: 5px;">
                        {mean:.2f} &nbsp;&nbsp; <b>Std:</b> {std:.3f}
                    </div>
                    <div style="display: flex; gap: 30px; font-family: monospace; font-size: 15px; margin-left: 5px;">
                        <span style="color: #d9534f; font-weight: bold;">Cpk = {cpk:.3f}</span>
                        <span style="font-weight: bold; color: #333;">Cp = {cp:.3f}</span>
                        <span style="font-weight: bold; color: #333;">Ca = {ca:.1f}%</span>
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a file to begin analysis.")
