import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import io

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SPC Analyzer — Cơ Tính",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #f0f4f8; }

[data-testid="stSidebar"] { background: #1a2332 !important; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] { background: #2d4a6b !important; }
[data-testid="stSidebar"] label { font-size: 0.75rem !important; font-weight: 600 !important;
    text-transform: uppercase; letter-spacing: 1px; color: #64748b !important; }
[data-testid="stSidebar"] h3 { color: #e2e8f0 !important; font-size: 0.68rem !important;
    font-weight: 700; text-transform: uppercase; letter-spacing: 2px;
    border-bottom: 1px solid #2d3748; padding-bottom: 8px; margin: 20px 0 12px; }

.app-header {
    background: linear-gradient(135deg, #1a2332 0%, #243447 100%);
    border-radius: 12px; padding: 28px 36px; margin-bottom: 20px;
    border-left: 4px solid #3b82f6; overflow: hidden;
}
.app-header h1 { color: #f1f5f9; font-size: 1.7rem; font-weight: 700; margin: 0 0 6px; letter-spacing: -0.3px; }
.app-header p  { color: #64748b; margin: 0; font-size: 0.88rem; }

.kpi-row { display: flex; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 80px; background: white; border-radius: 10px;
    padding: 12px 14px; border: 1px solid #e2e8f0; border-top: 3px solid;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.kpi-card .kpi-label { font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.2px; color: #94a3b8; margin-bottom: 5px; }
.kpi-card .kpi-val   { font-family: 'IBM Plex Mono', monospace; font-size: 1.45rem; font-weight: 600; line-height: 1; }
.kpi-card .kpi-sub   { font-size: 0.68rem; color: #94a3b8; margin-top: 4px; }
.kpi-green  { border-top-color: #22c55e; } .kpi-green  .kpi-val { color: #16a34a; }
.kpi-yellow { border-top-color: #f59e0b; } .kpi-yellow .kpi-val { color: #d97706; }
.kpi-red    { border-top-color: #ef4444; } .kpi-red    .kpi-val { color: #dc2626; }
.kpi-blue   { border-top-color: #3b82f6; } .kpi-blue   .kpi-val { color: #2563eb; }
.kpi-slate  { border-top-color: #64748b; } .kpi-slate  .kpi-val { color: #475569; }

.sec-hdr { font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.5px; color: #3b82f6; padding: 6px 0; margin: 8px 0 14px;
    border-bottom: 1px solid #e2e8f0; }

.cap-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px;
    border-radius: 20px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.3px; }
.cap-ex  { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
.cap-ok  { background: #dbeafe; color: #1d4ed8; border: 1px solid #93c5fd; }
.cap-mg  { background: #fef9c3; color: #a16207; border: 1px solid #fde047; }
.cap-no  { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }

.analysis-card {
    background: white; border-radius: 12px; border: 1px solid #e2e8f0;
    padding: 18px 20px; margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.analysis-card h3 { font-size: 0.95rem; font-weight: 700; color: #1e293b;
    margin: 0 0 14px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }

.stTabs [data-baseweb="tab-list"] { background: white; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 4px; gap: 2px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; color: #64748b;
    font-size: 0.82rem; font-weight: 600; padding: 8px 18px; }
.stTabs [aria-selected="true"] { background: #1a2332 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)


# ─── SPC Math ─────────────────────────────────────────────────────────────────
def calc_ca(mean, usl, lsl):
    T = usl - lsl
    if T == 0: return np.nan
    return (mean - (usl + lsl) / 2) / (T / 2)

def calc_cp(std, usl, lsl):
    if std <= 0: return np.nan
    return (usl - lsl) / (6 * std)

def calc_cpk(mean, std, usl, lsl):
    if std <= 0: return np.nan
    return min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))

def cap_class(cpk):
    if   cpk >= 1.67: return "Xuất sắc ≥1.67", "cap-ex"
    elif cpk >= 1.33: return "Tốt ≥1.33",       "cap-ok"
    elif cpk >= 1.00: return "Đạt ≥1.00",        "cap-mg"
    else:             return "Không đạt <1.00",   "cap-no"

def kpi_color(val, lo=1.0, hi=1.33):
    if np.isnan(val): return "kpi-slate"
    return "kpi-green" if val >= hi else "kpi-yellow" if val >= lo else "kpi-red"

def ca_color(ca):
    a = abs(ca)
    return "kpi-green" if a <= 0.125 else "kpi-yellow" if a <= 0.25 else "kpi-red"


# ─── Charts ───────────────────────────────────────────────────────────────────
BASE = dict(
    paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="IBM Plex Sans", size=11, color="#475569"),
    margin=dict(l=44, r=44, t=40, b=36),
)

def plot_distribution(series, lsl, usl, title):
    mean, std = series.mean(), series.std()
    fig = go.Figure()

    if std > 0:
        for x0, x1, col in [
            (mean-3*std, mean-2*std, "#fee2e2"), (mean+2*std, mean+3*std, "#fee2e2"),
            (mean-2*std, mean-std,   "#fef9c3"), (mean+std,   mean+2*std, "#fef9c3"),
            (mean-std,   mean+std,   "#dcfce7"),
        ]:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=col, opacity=0.45, layer="below", line_width=0)

    fig.add_trace(go.Histogram(
        x=series, histnorm="probability density",
        marker_color="#3b82f6", opacity=0.65, name="Data",
        nbinsx=min(40, max(10, len(series)//5)),
    ))

    if std > 0:
        x_r = np.linspace(min(series.min(), lsl-std), max(series.max(), usl+std), 400)
        fig.add_trace(go.Scatter(
            x=x_r, y=norm.pdf(x_r, mean, std),
            mode="lines", line=dict(color="#0f172a", width=2.2), name="Normal",
        ))

    for v, lbl, clr in [(lsl,"LSL","#ef4444"),(usl,"USL","#ef4444"),(mean,"Mean","#3b82f6")]:
        dash = "dash" if lbl != "Mean" else "dot"
        fig.add_vline(x=v, line=dict(color=clr, width=1.5, dash=dash))
        fig.add_annotation(x=v, yref="paper", y=1.04, text=lbl,
                           font=dict(color=clr, size=9), showarrow=False)

    fig.update_layout(
        **BASE, title=dict(text=title, font=dict(size=12, color="#1e293b"), x=0),
        height=300, showlegend=False,
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
    )
    return fig

def plot_trend(sub, col, lsl, usl, coil_col=None):
    mean = sub[col].mean()
    std  = sub[col].std()
    y_vals = sub[col].values
    x_vals = sub[coil_col].astype(str) if coil_col and coil_col in sub.columns else sub.index.astype(str)

    xi = np.arange(len(y_vals))
    trend_y = np.poly1d(np.polyfit(xi, y_vals, 1))(xi)
    oos_mask = (y_vals < lsl) | (y_vals > usl)

    fig = go.Figure()

    if std > 0:
        for mult, col_shade in [(3,"#fee2e230"),(2,"#fef9c330")]:
            fig.add_hrect(y0=mean-mult*std, y1=mean+mult*std,
                          fillcolor=col_shade, layer="below", line_width=0)

    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode="lines+markers", name="Data",
        line=dict(color="#334155", width=1.4),
        marker=dict(size=4.5,
                    color=["#ef4444" if m else "#3b82f6" for m in oos_mask]),
    ))
    if oos_mask.any():
        fig.add_trace(go.Scatter(
            x=x_vals[oos_mask], y=y_vals[oos_mask], mode="markers",
            marker=dict(color="#ef4444", size=9, symbol="x-thin",
                        line=dict(width=2.5, color="#ef4444")),
            name="OOS",
        ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=trend_y, mode="lines", name="Trend",
        line=dict(color="#8b5cf6", width=1.8, dash="dash"),
    ))

    for v, lbl, clr, dsh in [
        (usl, "USL", "#ef4444", "dot"),
        (lsl, "LSL", "#ef4444", "dot"),
        (mean, f"X̄={mean:.2f}", "#3b82f6", "dash"),
        (mean+3*std, "+3σ", "#f59e0b", "longdash") if std > 0 else (None,)*4,
        (mean-3*std, "−3σ", "#f59e0b", "longdash") if std > 0 else (None,)*4,
    ]:
        if v is None: continue
        fig.add_hline(y=v, line=dict(color=clr, width=1, dash=dsh))
        fig.add_annotation(xref="paper", x=1.01, y=v, text=lbl,
                           font=dict(color=clr, size=8), showarrow=False, xanchor="left")

    fig.update_layout(
        **BASE, height=240, showlegend=False,
        xaxis=dict(type="category", categoryorder="array",
                   categoryarray=x_vals.tolist(),
                   showgrid=False, tickfont=dict(size=8)),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
    )
    return fig

def plot_cpk_bar(rows):
    cols = [r["Parameter"] for r in rows]
    cpks = [r["Cpk"] for r in rows]
    clrs = ["#22c55e" if v >= 1.33 else "#f59e0b" if v >= 1.0 else "#ef4444" for v in cpks]

    fig = go.Figure(go.Bar(
        x=cols, y=cpks, marker_color=clrs,
        text=[f"{v:.3f}" for v in cpks], textposition="outside",
        textfont=dict(size=11, color="#1e293b"), width=0.5,
    ))
    for thresh, clr, lbl in [(1.67,"#15803d","1.67"),(1.33,"#22c55e","1.33"),(1.0,"#f59e0b","1.00")]:
        fig.add_hline(y=thresh, line=dict(color=clr, dash="dash", width=1.3),
                      annotation_text=f"Cpk={lbl}", annotation_font=dict(color=clr, size=9))
    fig.update_layout(
        **BASE, height=310,
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", range=[0, max(cpks)*1.25+0.3]),
        xaxis=dict(showgrid=False),
        title=dict(text="Cpk Comparison — All Parameters", font=dict(size=13, color="#1e293b"), x=0),
    )
    return fig


# ─── Sample data ──────────────────────────────────────────────────────────────
@st.cache_data
def make_sample():
    np.random.seed(7)
    n = 250
    return pd.DataFrame({
        "COIL_NO":  [f"C{str(i).zfill(4)}" for i in range(n)],
        "LINE":     np.random.choice(["LINE-1","LINE-2","LINE-3"], n),
        "鋼種":     np.random.choice(["SS400","SM490","S355","GE00/GE01"], n),
        "訂單寬度": np.random.choice([1200,1500,1800,2000], n),
        "YS":       np.random.normal(355, 18, n),
        "TS":       np.random.normal(490, 15, n),
        "EL":       np.random.normal(22,  2.5, n),
    })

DEFAULT_SPECS = {
    "YS": (310, 420),
    "TS": (450, 550),
    "EL": (18,  28),
}

COIL_CANDIDATES = ["COIL_NO","COIL NO","Coil_No","CoilNo","製造批號","Batch"]
TIME_CANDIDATES = ["生產日期","開始時間","Time","Date"]
POTENTIAL_TARGETS = ["YS","TS","EL","TENSILE_YIELD","TENSILE_TENSILE","TENSILE_ELONG","skp+t/l"]


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    st.markdown("""
    <div class="app-header">
        <h1>⚙️ Mechanical Property SPC Analyzer</h1>
        <p>Ca · Cp · Cpk &nbsp;·&nbsp; Distribution · Trending &nbsp;·&nbsp; Lọc LINE / 鋼種 / 訂單寬度</p>
    </div>""", unsafe_allow_html=True)

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📁 Data")
        uploaded = st.file_uploader("Upload Excel / CSV", type=["xlsx","xls","csv"],
                                    label_visibility="collapsed")

        if uploaded:
            try:
                df_raw = (pd.read_csv(uploaded) if uploaded.name.endswith(".csv")
                          else pd.read_excel(uploaded))
                df_raw.columns = df_raw.columns.str.strip()
                if "鋼種" in df_raw.columns:
                    df_raw["鋼種"] = df_raw["鋼種"].replace(["GE00","GE01"], "GE00/GE01")
                if "Metallic_Type" in df_raw.columns:
                    df_raw = df_raw[df_raw["Metallic_Type"].astype(str).str.strip().str.upper() != "GF"]
                st.success(f"✅ {len(df_raw):,} rows loaded")
            except Exception as e:
                st.error(f"Read error: {e}")
                df_raw = make_sample()
        else:
            st.info("Using sample data")
            df_raw = make_sample()

        st.markdown("### 🔍 Filters")
        FILTER_COLS = ["LINE","鋼種","訂單寬度"]
        miss = [c for c in FILTER_COLS if c not in df_raw.columns]
        if miss:
            st.error(f"Missing: {', '.join(miss)}")
            return

        df_raw["訂單寬度"] = pd.to_numeric(df_raw["訂單寬度"], errors="coerce")

        sel_line  = st.multiselect("LINE",  sorted(df_raw["LINE"].dropna().unique()),
                                   default=sorted(df_raw["LINE"].dropna().unique()))
        sel_grade = st.multiselect("鋼種",  sorted(df_raw["鋼種"].dropna().unique()),
                                   default=sorted(df_raw["鋼種"].dropna().unique()))
        sel_width = st.multiselect("訂單寬度", sorted(df_raw["訂單寬度"].dropna().unique()),
                                   default=sorted(df_raw["訂單寬度"].dropna().unique()))

        df = df_raw[
            df_raw["LINE"].isin(sel_line) &
            df_raw["鋼種"].isin(sel_grade) &
            df_raw["訂單寬度"].isin(sel_width)
        ].copy()

        st.markdown(
            f"<div style='font-size:0.78rem;color:#94a3b8'>"
            f"📋 <b style='color:#e2e8f0'>{len(df):,}</b> rows after filter</div>",
            unsafe_allow_html=True,
        )

        # Sort
        coil_col = next((c for c in COIL_CANDIDATES if c in df.columns), None)
        time_cols = [c for c in TIME_CANDIDATES if c in df.columns]
        sort_by = time_cols + ([coil_col] if coil_col else [])
        if sort_by:
            df = df.sort_values(by=sort_by).reset_index(drop=True)

        st.markdown("### 📐 Parameters")
        numeric_cols = [c for c in df.select_dtypes(include=np.number).columns
                        if c not in ["LINE","訂單寬度"]]
        preferred = [c for c in POTENTIAL_TARGETS if c in numeric_cols]
        targets = st.multiselect("Analysis parameters", numeric_cols,
                                 default=preferred if preferred else numeric_cols[:3])

        if not targets:
            st.warning("Select at least one parameter.")
            return

        st.markdown("### ⚙️ Spec Limits")
        specs = {}
        for t in targets:
            with st.expander(t, expanded=False):
                s = df[t].dropna()
                mean0, std0 = float(s.mean()), float(s.std())
                def_lsl, def_usl = DEFAULT_SPECS.get(t, (mean0 - 3*std0, mean0 + 3*std0))
                lsl_v = st.number_input("LSL", value=float(def_lsl), key=f"lsl_{t}", format="%.3f")
                usl_v = st.number_input("USL", value=float(def_usl), key=f"usl_{t}", format="%.3f")
                tgt_v = st.number_input("Target", value=float((lsl_v+usl_v)/2), key=f"tgt_{t}", format="%.3f")
                specs[t] = {"lsl": lsl_v, "usl": usl_v, "tgt": tgt_v}

    if len(df) < 3:
        st.warning("⚠️ Not enough data after filtering.")
        return

    # ── BUILD SUMMARY ──────────────────────────────────────────────────────────
    summary_rows = []
    for t in targets:
        s = df[t].dropna()
        if len(s) < 3: continue
        lsl, usl = specs[t]["lsl"], specs[t]["usl"]
        mean, std = s.mean(), s.std()
        ca  = calc_ca(mean, usl, lsl)
        cp  = calc_cp(std, usl, lsl)
        cpk = calc_cpk(mean, std, usl, lsl)
        oos = int(((s < lsl) | (s > usl)).sum())
        grade, _ = cap_class(cpk)
        summary_rows.append({
            "Parameter": t, "N": len(s),
            "Mean": round(mean,3), "Std": round(std,4),
            "LSL": lsl, "USL": usl,
            "Ca": round(float(ca),4), "Cp": round(float(cp),4), "Cpk": round(float(cpk),4),
            "OOS": oos, "Grade": grade,
        })
    summary_df = pd.DataFrame(summary_rows)

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab_ov, tab_det, tab_data, tab_dl = st.tabs([
        "📊  Overview", "🔬  Detail Analysis", "🗂️  Data Table", "📤  Export",
    ])

    # ══════ TAB 1: OVERVIEW ══════════════════════════════════════════════════
    with tab_ov:
        if len(summary_rows):
            st.plotly_chart(plot_cpk_bar(summary_rows), use_container_width=True)

            st.markdown('<div class="sec-hdr">Summary Table</div>', unsafe_allow_html=True)

            def row_style(row):
                try:
                    cpk = float(row["Cpk"])
                    bg = "#dcfce7" if cpk >= 1.33 else "#fef9c3" if cpk >= 1.0 else "#fee2e2"
                    return [f"background-color:{bg}"] * len(row)
                except:
                    return [""] * len(row)

            st.dataframe(
                summary_df.style.apply(row_style, axis=1)
                .format({"Mean":"{:.3f}","Std":"{:.4f}","Ca":"{:.4f}","Cp":"{:.4f}","Cpk":"{:.4f}"}),
                use_container_width=True, height=min(42*len(summary_df)+64, 420),
            )

    # ══════ TAB 2: DETAIL ════════════════════════════════════════════════════
    with tab_det:
        for i in range(0, len(targets), 2):
            pair = targets[i:i+2]
            cols_ui = st.columns(len(pair))

            for col_ui, t in zip(cols_ui, pair):
                lsl, usl = specs[t]["lsl"], specs[t]["usl"]
                s = df[t].dropna()
                if len(s) < 3:
                    col_ui.warning(f"Not enough data: {t}")
                    continue

                sub = df.dropna(subset=[t])
                if coil_col and coil_col in sub.columns:
                    sub = sub.drop_duplicates(subset=[coil_col], keep="last")
                sub = sub.reset_index(drop=True)

                mean, std = float(s.mean()), float(s.std())
                ca  = float(calc_ca(mean, usl, lsl))
                cp  = float(calc_cp(std, usl, lsl))
                cpk = float(calc_cpk(mean, std, usl, lsl))
                oos = int(((s < lsl) | (s > usl)).sum())
                grade, badge_cls = cap_class(cpk)

                with col_ui:
                    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)

                    st.markdown(f"""
                    <h3>📈 {t}
                        <span class="cap-badge {badge_cls}">{grade}</span>
                    </h3>""", unsafe_allow_html=True)

                    ca_cls  = ca_color(ca)
                    cpk_cls = kpi_color(cpk)
                    cp_cls  = kpi_color(cp)
                    oos_cls = "kpi-red" if oos > 0 else "kpi-green"

                    st.markdown(f"""
                    <div class="kpi-row">
                        <div class="kpi-card {cpk_cls}">
                            <div class="kpi-label">Cpk</div>
                            <div class="kpi-val">{cpk:.3f}</div>
                            <div class="kpi-sub">≥1.33 = Good</div>
                        </div>
                        <div class="kpi-card {cp_cls}">
                            <div class="kpi-label">Cp</div>
                            <div class="kpi-val">{cp:.3f}</div>
                            <div class="kpi-sub">Precision</div>
                        </div>
                        <div class="kpi-card {ca_cls}">
                            <div class="kpi-label">Ca</div>
                            <div class="kpi-val">{ca:.3f}</div>
                            <div class="kpi-sub">|Ca|≤0.125=A</div>
                        </div>
                        <div class="kpi-card kpi-blue">
                            <div class="kpi-label">n / Mean</div>
                            <div class="kpi-val">{len(s)}</div>
                            <div class="kpi-sub">{mean:.2f} ± {std:.2f}</div>
                        </div>
                        <div class="kpi-card {oos_cls}">
                            <div class="kpi-label">OOS</div>
                            <div class="kpi-val">{oos}</div>
                            <div class="kpi-sub">Out of spec</div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                    st.plotly_chart(plot_distribution(s, lsl, usl, f"Distribution — {t}"),
                                    use_container_width=True)
                    st.plotly_chart(plot_trend(sub, t, lsl, usl, coil_col),
                                    use_container_width=True)

                    st.markdown("</div>", unsafe_allow_html=True)

    # ══════ TAB 3: DATA TABLE ════════════════════════════════════════════════
    with tab_data:
        c1, c2, c3 = st.columns(3)
        for grp, col_ui, lbl in [("LINE",c1,"By LINE"),("鋼種",c2,"By 鋼種"),("訂單寬度",c3,"By 訂單寬度")]:
            if grp in df.columns:
                col_ui.markdown(f"**{lbl}**")
                col_ui.dataframe(
                    df.groupby(grp).size().reset_index(name="Count")
                      .sort_values("Count", ascending=False),
                    use_container_width=True, height=220,
                )
        st.markdown('<div class="sec-hdr">Filtered Data</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, height=420)

    # ══════ TAB 4: EXPORT ════════════════════════════════════════════════════
    with tab_dl:
        st.markdown('<div class="sec-hdr">Download Results</div>', unsafe_allow_html=True)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            summary_df.to_excel(w, sheet_name="SPC Summary", index=False)
            df.to_excel(w, sheet_name="Filtered Data", index=False)

        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Download Excel (.xlsx)", data=buf.getvalue(),
                           file_name="spc_report.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
        c2.download_button("⬇️ Download CSV", data=summary_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="spc_summary.csv", mime="text/csv",
                           use_container_width=True)

        st.markdown('<div class="sec-hdr" style="margin-top:24px">Preview</div>', unsafe_allow_html=True)
        st.dataframe(summary_df, use_container_width=True)

        st.markdown("""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
                    padding:18px 22px;margin-top:20px;font-size:0.83rem;color:#475569;line-height:1.9">
        <b>📖 Hướng dẫn đọc chỉ số</b><br>
        <b>Ca</b> = (X̄ − M) / (T/2) — |Ca| ≤ 0.125 = A · ≤ 0.25 = B · ≤ 0.50 = C · > 0.50 = D<br>
        <b>Cp</b> = T / 6σ — ≥ 1.67 = A+ · ≥ 1.33 = A · ≥ 1.00 = B · < 1.00 = C<br>
        <b>Cpk</b> = min(Cpu, Cpl) — ≥ 1.67 Xuất sắc · ≥ 1.33 Tốt · ≥ 1.00 Đạt · < 1.00 Không đạt
        </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
