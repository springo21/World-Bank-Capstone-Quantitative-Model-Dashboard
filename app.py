"""
IDA Sovereign Donor Readiness Dashboard
Capstone Project — World Bank IDA Partner Deliverable
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IDA Donor Readiness Model",
    page_icon="drilogo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand palette ─────────────────────────────────────────────────────────────
COLORS = {
    "navy":      "#0a122a",
    "green":     "#698f3f",
    "snow":      "#fbfaf8",
    "bone":      "#e7decd",
    "clay":      "#804e49",
    "text":      "#0a122a",
    "subtext":   "#5f6470",
    "border":    "#e7decd",
    "muted":     "#b8b0a3",
    "soft_blue": "#8eb1d1",
    "soft_clay": "#c98f86",
    "white":     "#ffffff",
}

SEGMENT_COLORS = {
    "Reliable Donor":            COLORS["navy"],
    "Under-Contributing Donor":  COLORS["clay"],
    "Exceeded Target":           COLORS["green"],
    "High-Potential Prospect":   "#4a9068",
    "Emerging Prospect":         COLORS["bone"],
    "Low Probability":           "#c9c2b5",
    "Non-Donor":                 "#d4cfc9",
}

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
OUTPUTS   = BASE / "outputs"
PROCESSED = BASE / "data" / "processed"

print("BASE exists    :", BASE.exists())
print("dri_output.csv :", (OUTPUTS / "dri_output.csv").exists())


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    dri       = pd.read_csv(OUTPUTS / "dri_output.csv")
    alignment = pd.read_csv(PROCESSED / "alignment_scores.csv")
    master    = pd.read_csv(PROCESSED / "master.csv")

    df = dri.merge(
        alignment[["iso3", "alignment_score", "ifc_presence_score"]],
        on="iso3", how="left"
    )
    df = df.merge(
        master[["iso3", "gdp_per_capita_usd", "fiscal_balance_pct_gdp",
                "govt_debt_pct_gdp", "gov_effectiveness"]],
        on="iso3", how="left"
    )

    df["gdp_bn"]    = df["gdp_usd"] / 1e9

    # Model now uses gap_usd_signed (positive = shortfall, negative = over-contribution)
    df["gap_usd"]   = df["gap_usd_signed"]
    df["gap_bn"]    = df["gap_usd"] / 1e9
    df["actual_bn"] = df["actual_contribution_usd"] / 1e9
    df["target_bn"] = df["adjusted_target_usd"] / 1e9

    # New: PPP gap percentage and confidence interval columns
    df["gap_pct_ppp_gdp"] = df.get("gap_pct_ppp_gdp", pd.Series(dtype=float))
    df["gap_usd_lower"]   = df.get("gap_usd_lower", pd.Series(dtype=float))
    df["gap_usd_upper"]   = df.get("gap_usd_upper", pd.Series(dtype=float))

    # Updated segment order — now includes Exceeded Target and Non-Donor
    segment_order = [
        "Exceeded Target",
        "Reliable Donor",
        "Under-Contributing Donor",
        "High-Potential Prospect",
        "Emerging Prospect",
        "Low Probability",
        "Non-Donor",
    ]
    existing = [s for s in segment_order if s in df["donor_segment"].values]
    df["donor_segment"] = pd.Categorical(
        df["donor_segment"], categories=existing, ordered=True
    )
    df["is_donor"] = df["actual_contribution_usd"] > 0
    return df.sort_values("gap_usd", ascending=False).reset_index(drop=True)


@st.cache_data
def load_diagnostics():
    path = OUTPUTS / "heckman_diagnostics.txt"
    return path.read_text() if path.exists() else None


df = load_data()


# ── Global styling ────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    :root {{
        --navy:    {COLORS["navy"]};
        --green:   {COLORS["green"]};
        --snow:    {COLORS["snow"]};
        --bone:    {COLORS["bone"]};
        --clay:    {COLORS["clay"]};
        --text:    {COLORS["text"]};
        --subtext: {COLORS["subtext"]};
        --border:  {COLORS["border"]};
        --white:   {COLORS["white"]};
    }}

    /* ── Base ── */
    .stApp {{
        background-color: var(--snow);
        color: var(--text);
    }}

    .block-container {{
        padding-top: 1.3rem;
        padding-bottom: 1.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100%;
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: var(--text);
        letter-spacing: -0.02em;
        font-weight: 700;
    }}

    p, div, label, span {{ color: var(--text); }}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {{
        background: var(--navy);
        border-right: none;
    }}

    section[data-testid="stSidebar"] * {{
        color: white !important;
    }}

    section[data-testid="stSidebar"] .block-container {{
        padding-top: 1.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }}

    .sidebar-brand {{
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 20px;
        padding: 1.1rem 1rem;
        margin-bottom: 1.2rem;
    }}

    .sidebar-title {{
        font-size: 1.4rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.15rem;
    }}

    .sidebar-subtitle {{
        font-size: 0.88rem;
        color: rgba(255,255,255,0.65);
    }}

    /* ── Sidebar nav buttons (Material icon style) ── */
    /* tertiary (inactive) nav buttons — transparent, white text */
    section[data-testid="stSidebar"] .stButton > button[kind="tertiary"] {{
        background: transparent !important;
        color: rgba(255,255,255,0.80) !important;
        border: none !important;
        border-bottom: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 0 !important;
        padding: 0.7rem 0.75rem !important;
        text-align: left !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        justify-content: flex-start !important;
        box-shadow: none !important;
    }}

    section[data-testid="stSidebar"] .stButton > button[kind="tertiary"]:hover {{
        background: rgba(255,255,255,0.07) !important;
        color: white !important;
    }}

    /* primary (active) nav button — green highlight */
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
        background: rgba(105,143,63,0.30) !important;
        color: white !important;
        border: none !important;
        border-left: 3px solid {COLORS["green"]} !important;
        border-radius: 0 !important;
        padding: 0.7rem 0.75rem !important;
        text-align: left !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        justify-content: flex-start !important;
        box-shadow: none !important;
    }}

    /* nav button icons */
    section[data-testid="stSidebar"] .stButton > button svg,
    section[data-testid="stSidebar"] .stButton > button [data-testid="stIconMaterial"] {{
        color: rgba(255,255,255,0.85) !important;
        fill: rgba(255,255,255,0.85) !important;
    }}

    /* remove focus ring on nav buttons */
    section[data-testid="stSidebar"] .stButton > button:focus,
    section[data-testid="stSidebar"] .stButton > button:focus-visible {{
        outline: none !important;
        box-shadow: none !important;
    }}
    
    /* ── Remove gap between nav buttons and divider only ── */
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"]:has(.stButton) {{
        padding-bottom: 0 !important;
        margin-bottom: 0 !important;
    }}

    section[data-testid="stSidebar"] .stButton {{
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
        line-height: 1 !important;
    }}

    section[data-testid="stSidebar"] hr {{
        margin-top: 0 !important;
        margin-bottom: 0.8rem !important;
    }}
    
    /* ── Collapse gap between last nav button and divider ── */
    section[data-testid="stSidebar"] .stButton {{
        margin-bottom: -1rem !important;
    }}

    section[data-testid="stSidebar"] hr {{
        margin-top: 0.5rem !important;
        margin-bottom: 0.8rem !important;
        position: relative;
        top: -0.4rem;
    }}


    /* ── Inputs ── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stTextInput > div > div,
    .stNumberInput > div > div {{
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        background-color: rgba(255,255,255,0.08) !important;
        color: white !important;
    }}

    /* multiselect dropdown options list */
    ul[data-baseweb="menu"] {{
        background-color: var(--navy) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
    }}
        /* ── Selectbox dropdown menu ── */
    div[data-baseweb="popover"] ul {{
        background-color: var(--snow) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
    }}
    
    div[data-baseweb="popover"] ul li {{
        background-color: transparent !important;
        color: white !important;
    }}
    
    div[data-baseweb="popover"] ul li:hover {{
        background-color: rgba(255,255,255,0.10) !important;
    }}
    
    /* selected option highlight */
    div[data-baseweb="popover"] ul li[aria-selected="true"] {{
        background-color: rgba(128, 78, 73, 0.35) !important;
        color: white !important;
    }}
    
    /* the input box itself */
    div[data-testid="stSelectbox"] > div > div {{
        background-color: white !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        color: var(--text) !important;
    }}
    
    /* text inside the selectbox */
    div[data-testid="stSelectbox"] > div > div > div {{
        color: var(--text) !important;
    }}

    ul[data-baseweb="menu"] li {{
        background-color: transparent !important;
        color: white !important;
    }}

    ul[data-baseweb="menu"] li:hover {{
        background-color: rgba(255,255,255,0.10) !important;
    }}

    /* multiselect tags */
    [data-baseweb="tag"] {{
        background-color: rgba(105,143,63, 0.8) !important;
        border-radius: 20px !important;
        color: white !important;
        font-size: 0.78rem !important;
        border: 1px solid rgba(105,143,63,0.5) !important;
    }}

    /* placeholder text in inputs */
    .stMultiSelect span[data-baseweb="tag"] + div input,
    .stSelectbox input {{
        color: var(--text) !important;
    }}

    /* ── Progress bar — theme green ── */
    div[data-testid="stProgressBar"] > div > div > div > div {{
        background-color: var(--green) !important;
        border-radius: 999px !important;
    }}
    div[data-testid="stProgressBar"] > div > div > div {{
        background-color: rgba(105,143,63,0.18) !important;
        border-radius: 999px !important;
    }}
    
    /* FIX multiselect height (only visually) */
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {{
        height: 50px !important;
        overflow: hidden !important;
    }}

    /* ── Metric cards ── */
    div[data-testid="metric-container"] {{
        background: white;
        border: 1px solid var(--border);
        padding: 1rem 1.1rem;
        border-radius: 22px;
        box-shadow: 0 2px 10px rgba(10,18,42,0.04);
    }}

    div[data-testid="metric-container"] label {{
        color: var(--subtext) !important;
        font-weight: 600;
    }}

    div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color: var(--text) !important;
    }}

    /* ── Dataframe ── */
    div[data-testid="stDataFrame"] {{
        border-radius: 25px;
        overflow: hidden;
        border: 1px solid var(--border);
        background: white;
    }}

    /* ── Tabs ── */
    button[data-baseweb="tab"] {{
        border-radius: 999px !important;
        border: 1px solid var(--border) !important;
        background: white !important;
        color: var(--text) !important;
        padding: 0.45rem 0.95rem !important;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        background: var(--green) !important;
        color: white !important;
        border-color: var(--green) !important;
    }}

    div[data-baseweb="tab-list"] {{
        gap: 0.5rem;
        border-bottom: none !important;
        background: transparent !important;
    }}

    div[data-baseweb="tab-border"] {{ display: none !important; }}

    /* remove red underline on tab click (focus outline) */
    button[data-baseweb="tab"]:focus,
    button[data-baseweb="tab"]:focus-visible {{
        outline: none !important;
        box-shadow: none !important;
        border-color: var(--green) !important;
    }}
    
    /* Remove animated highlight bar under tab */
    div[data-baseweb="tab-highlight"] {{
        background-color: transparent ;
    }}

    /* Remove tab border element */
    div[data-baseweb="tab-border"] {{
        display: none ;
    }}


    /* ── Buttons ── */
    .stButton > button,
    .stDownloadButton > button {{
        border-radius: 16px !important;
        border: none !important;
        background: var(--green) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.6rem 1rem !important;
        outline: none !important;
        box-shadow: none !important;
    }}

    .stButton > button:hover,
    .stDownloadButton > button:hover {{
        background: var(--navy) !important;
        color: white !important;
    }}

    /* remove red/default focus ring on buttons */
    .stButton > button:focus,
    .stButton > button:focus-visible,
    .stDownloadButton > button:focus,
    .stDownloadButton > button:focus-visible {{
        outline: none !important;
        box-shadow: 0 0 0 2px rgba(105,143,63,0.5) !important;
        border: none !important;
    }}

    /* ── Code blocks ── */
    .stCodeBlock, pre {{
        border-radius: 18px !important;
    }}

    /* ── Divider ── */
    hr {{
        border: none;
        height: 1px;
        background: var(--border);
    }}

    /* ── Reusable card class ── */
    .soft-card {{
        background: white;
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 1.15rem 1.2rem;
        box-shadow: 0 2px 10px rgba(10,18,42,0.04);
    }}

    .subtle-note {{
        color: var(--subtext);
        font-size: 0.95rem;
    }}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x) / 1e6)


def fmt_usd(v, decimals=2):
    if pd.isna(v):   return "N/A"
    if abs(v) >= 1e9: return f"${v/1e9:.{decimals}f}B"
    if abs(v) >= 1e6: return f"${v/1e6:.{decimals}f}M"
    return f"${v:,.0f}"


def apply_dashboard_layout(
    fig, height=400, showlegend=True,
    legend_orientation="h", legend_x=0, legend_y=1.02,
    legend_xanchor="left", legend_yanchor="bottom", margin=None,
):
    if margin is None:
        margin = dict(t=24, b=20, l=20, r=20)
    fig.update_layout(
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", color=COLORS["text"], size=13),
        margin=margin,
        legend=(dict(
            orientation=legend_orientation,
            x=legend_x, y=legend_y,
            xanchor=legend_xanchor, yanchor=legend_yanchor,
            bgcolor="rgba(0,0,0,0)", tracegroupgap=8,
        ) if showlegend else dict()),
    )
    fig.update_xaxes(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"], automargin=True)
    fig.update_yaxes(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"], automargin=True)
    return fig


def soft_card_open():
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)


def soft_card_close():
    st.markdown("</div>", unsafe_allow_html=True)


# ── KPI row ───────────────────────────────────────────────────────────────────
def kpi_row(metrics: list[tuple], colors: list[str] | None = None):
    """
    metrics = list of (label, value, delta, delta_label)
    colors  = optional list of hex background colors, one per metric.
              Defaults to cycling through the brand palette.
    """
    default_colors = [
        COLORS["navy"],
        COLORS["green"],
        COLORS["clay"],
        COLORS["bone"],
        COLORS["muted"],
    ]
    if colors is None:
        colors = (default_colors * len(metrics))[:len(metrics)]

    cols = st.columns(len(metrics))
    for col, (label, value, delta, delta_label), bg in zip(cols, metrics, colors):
        # pick text colour — white on dark backgrounds, dark on light ones
        text_col = "#ffffff" if bg not in [COLORS["bone"], "#c9c2b5", COLORS["muted"]] else COLORS["text"]
        sub_col  = "rgba(255,255,255,0.65)" if text_col == "#ffffff" else COLORS["subtext"]
        delta_html = ""
        if delta:
            delta_html = f"<p style='margin:4px 0 0;font-size:0.78rem;color:{sub_col}'>{delta}</p>"
        with col:
            st.markdown(
                f"""
                <div style='
                    background: {bg};
                    border-radius: 18px;
                    padding: 1.1rem 1.2rem;
                    box-shadow: 0 2px 10px rgba(10,18,42,0.08);
                    height: 110px;
                '>
                    <p style='margin:0 0 4px;font-size:0.8rem;font-weight:700;
                              text-transform:uppercase;letter-spacing:0.06em;
                              color:{sub_col}'>{label}</p>
                    <p style='margin:0;font-size:1.9rem;font-weight:700;
                              color:{text_col};line-height:1.1'>{value}</p>
                    {delta_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Material icon nav using st.button with :material/icon_name:
# Docs: https://docs.streamlit.io/develop/api-reference/widgets/st.button
# Icons: https://fonts.google.com/icons?icon.set=Material+Symbols
# ─────────────────────────────────────────────────────────────────────────────

NAV_ITEMS = [
    ("Overview",          "dashboard"),
    ("Country Explorer",  "explore"),
    ("Gap Analysis",      "bar_chart"),
    ("Prospect Ranking",  "format_list_numbered"),
    ("World Map",         "public"),
    ("Model Diagnostics", "data_thresholding"),
    ("Interview Findings","communication"),
    ("Glossary",          "toc"),

]

# Initialise session state for active page
if "page" not in st.session_state:
    st.session_state.page = "Overview"

with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-title">IDA Partnerships</div>
            <div class="sidebar-subtitle">Capstone Research Dashboard</div>
        </div>
    """, unsafe_allow_html=True)

    # One button per page — active page styled as primary, others tertiary
    for label, icon_name in NAV_ITEMS:
        is_active = st.session_state.page == label
        if st.button(
            label,
            icon=f":material/{icon_name}:",
            type="primary" if is_active else "tertiary",
            use_container_width=True,
            key=f"nav_{label.lower().replace(' ', '_')}",
        ):
            st.session_state.page = label
            st.rerun()

    st.divider()
    st.markdown(
        "<p style='font-size:0.82rem;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.08em;color:rgba(255,255,255,0.45);margin:0 0 0.4rem 0.2rem'>"
        "Filters</p>",
        unsafe_allow_html=True,
    )

    seg_options = list(df["donor_segment"].cat.categories)
    selected_segments = st.multiselect("Donor Segment", seg_options, default=seg_options)

    income_options = sorted(df["income_group"].dropna().unique())
    selected_income = st.multiselect("Income Group", income_options, default=income_options)

    min_gdp = st.slider("Min. GDP (USD billions)", 0, 5000, value=0, step=50)

    filtered = df[
        df["donor_segment"].isin(selected_segments) &
        df["income_group"].isin(selected_income) &
        (df["gdp_bn"] >= min_gdp)
    ].copy()

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.caption(
        "Data: World Bank WDI · IMF WEO · IDA Replenishments 1–21\n\n"
        "Model: Heckman two-stage selection\n\nCapstone Project · 2025–26\n\nIE University"
    )

page = st.session_state.page


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "Overview":
    st.title("IDA Sovereign Donor Readiness Model")
    st.markdown(
        "<p class='subtle-note' style='line-height:1.6'>"
        "This dashboard presents the findings of a quantitative model assessing which "
        "countries have the capacity and readiness to contribute to IDA replenishments, "
        "and by how much they are currently under- or over-contributing relative to their "
        "economic capacity. The model uses a Heckman two-stage selection model"
        " to identify both the probability of donating and the "
        "expected contribution amount for every country in the sample.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    total_gap       = filtered["gap_usd"].sum()
    n_prospects     = len(filtered[filtered["donor_segment"].isin(
                          ["High-Potential Prospect", "Emerging Prospect"])])
    n_under         = len(filtered[filtered["donor_segment"] == "Under-Contributing Donor"])
    n_exceeded      = len(filtered[filtered["donor_segment"] == "Exceeded Target"])
    n_donors        = len(filtered[filtered["is_donor"]])
    avg_giving_rate = filtered[filtered["is_donor"]]["giving_rate"].median()

    kpi_row([
        ("Total Addressable Gap",      fmt_usd(total_gap),          None, None),
        ("Current Donors in Sample",   str(n_donors),               None, None),
        ("Under-Contributing Donors",  str(n_under),                None, None),
        ("Exceeded Target",            str(n_exceeded),             None, None),
        ("High/Emerging Prospects",    str(n_prospects),            None, None),
    ])
    st.divider()

    col1, col2 = st.columns([1.05, 1.55])

    with col1:
        st.markdown("#### Country Segments")
        seg_counts = (
            filtered.groupby("donor_segment", observed=True)
            .size().reset_index(name="count")
        )
        fig_donut = go.Figure(go.Pie(
            labels=seg_counts["donor_segment"],
            values=seg_counts["count"],
            hole=0.52,
            marker_colors=[SEGMENT_COLORS.get(s, COLORS["muted"]) for s in seg_counts["donor_segment"]],
            # show labels directly on slices — no legend needed
            textinfo="label+percent",
            textposition="outside",
            textfont=dict(size=11),
            hovertemplate="%{label}<br>%{value} countries (%{percent})<extra></extra>",
            sort=False,
            # pull slices apart slightly so outside labels don't overlap
            pull=[0.03] * len(seg_counts),
        ))
        fig_donut.update_layout(
            showlegend=False,
            height=380,
            margin=dict(t=30, b=30, l=60, r=60),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(family="Inter, Arial, sans-serif", color=COLORS["text"], size=11),
        )
        # Ensure labels aren't clipped by the card
        fig_donut.update_traces(outsidetextfont=dict(size=10))
        st.plotly_chart(fig_donut, use_container_width=True)
        soft_card_close()

    with col2:
        st.markdown("#### Segment Summary")
        seg_summary = (
            filtered.groupby("donor_segment", observed=True)
            .agg(
                Countries=("iso3", "count"),
                Total_Gap_USD=("gap_usd", "sum"),
                Avg_Giving_Rate=("giving_rate", "mean"),
                Avg_p_donate=("p_donate", "mean"),
            ).reset_index()
        )
        seg_summary["Total_Gap_USD"]   = seg_summary["Total_Gap_USD"].apply(fmt_usd)
        seg_summary["Avg_Giving_Rate"] = seg_summary["Avg_Giving_Rate"].apply(
            lambda x: f"{x:.1%}" if not pd.isna(x) else "—")
        seg_summary["Avg_p_donate"]    = seg_summary["Avg_p_donate"].apply(
            lambda x: f"{x:.2f}" if not pd.isna(x) else "—")
        seg_summary.columns = ["Segment", "# Countries", "Total Gap (USD)",
                                "Avg. Giving Rate", "Avg. P(Donate)"]
        st.dataframe(seg_summary, hide_index=True, use_container_width=True)
        soft_card_close()

    st.divider()
    st.markdown("#### Top 10 Countries by Contribution Gap")
    top10 = (
        filtered[filtered["gap_usd"].notna()]
        .nlargest(10, "gap_usd")
        .sort_values("gap_usd", ascending=True)
    )
    fig_top10 = go.Figure()
    fig_top10.add_trace(go.Bar(
        y=top10["country_name"], x=top10["gap_usd"] / 1e9,
        orientation="h",
        marker_color=[SEGMENT_COLORS.get(s, COLORS["muted"]) for s in top10["donor_segment"]],
        hovertemplate="<b>%{y}</b><br>Gap: $%{x:.2f}B<extra></extra>",
        text=[fmt_usd(v) for v in top10["gap_usd"]],
        textposition="outside", cliponaxis=False, showlegend=False,
    ))
    for seg, col in SEGMENT_COLORS.items():
        fig_top10.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=col, symbol="square"),
            name=seg, showlegend=True,
        ))
    fig_top10.update_layout(xaxis_title="Gap (USD billions)", yaxis_title=None)
    apply_dashboard_layout(
        fig_top10, height=430, legend_orientation="v",
        legend_x=1.02, legend_y=1.0,
        legend_xanchor="left", legend_yanchor="top",
        margin=dict(t=30, b=20, l=20, r=190),
    )
    st.plotly_chart(fig_top10, use_container_width=True)
    st.caption(
        "Gap = Capacity-adjusted target contribution − actual IDA21 contribution. "
        "Positive = under-contributing."
    )
    soft_card_close()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — COUNTRY EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Country Explorer":
    st.title("Country Explorer")
    st.markdown(
        "<p class='subtle-note'>Dive into any country's readiness profile and compare "
        "current contributions with capacity-based targets, model predictions, and "
        "structural indicators.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    country_list     = sorted(filtered["country_name"].dropna().tolist())
    default_country = "United States"
    selected_country = st.selectbox("Select a country", country_list, index=country_list.index(default_country) if default_country in country_list else 0)
    row              = df[df["country_name"] == selected_country].iloc[0]
    seg              = row["donor_segment"]
    seg_color        = SEGMENT_COLORS.get(str(seg), COLORS["muted"])

    st.markdown(
        f"### {row['country_name']} &nbsp;&nbsp;"
        f'<span style="background:{seg_color};color:white;padding:4px 12px;'
        f'border-radius:20px;font-size:0.85rem">{seg}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**ISO3:** {row['iso3']} &nbsp;|&nbsp; **Income Group:** {row.get('income_group','—')}")
    st.divider()

    kpi_row([
        ("GDP",              fmt_usd(row["gdp_usd"]),                                  None, None),
        ("GDP per Capita",   fmt_usd(row.get("gdp_per_capita_usd", np.nan), 0),        None, None),
        ("Capacity Target",  fmt_usd(row["adjusted_target_usd"]),                      None, None),
        ("Actual (IDA21)",   fmt_usd(row["actual_contribution_usd"]),                  None, None),
        ("Contribution Gap", fmt_usd(row["gap_usd"]),                                  None, None),
    ])
    st.divider()

    col1, col_spacer, col2 = st.columns([1, 0.08, 1])
    with col1:
        st.markdown("#### Contribution Profile")
        vals = {
            "Actual IDA21":      row["actual_contribution_usd"],
            "Capacity Target":   row["adjusted_target_usd"],
            "Predicted (Model)": row["pred_donation_usd"],
        }
        vals = {k: v for k, v in vals.items() if not pd.isna(v)}
        fig_bar = go.Figure(go.Bar(
            x=list(vals.keys()),
            y=[v / 1e6 for v in vals.values()],
            marker_color=[COLORS["navy"], COLORS["clay"], COLORS["green"]],
            text=[fmt_usd(v) for v in vals.values()],
            textposition="outside",
        ))
        fig_bar.update_layout(yaxis_title="USD millions")
        apply_dashboard_layout(fig_bar, height=300, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        soft_card_close()

    with col2:
        st.markdown("#### Key Metrics")
        metrics = {
            "P(Donate)":          f"{row['p_donate']:.3f}",
            "Giving Rate":        f"{row['giving_rate']:.1%}" if not pd.isna(row['giving_rate']) else "—",
            "IMR (λ)":            f"{row['imr']:.3f}"         if not pd.isna(row.get('imr')) else "—",
            "Fiscal Balance":     (f"{row.get('fiscal_balance_pct_gdp',np.nan):.1f}% GDP"
                                   if not pd.isna(row.get('fiscal_balance_pct_gdp',np.nan)) else "—"),
            "Govt Debt / GDP":    (f"{row.get('govt_debt_pct_gdp',np.nan):.1f}%"
                                   if not pd.isna(row.get('govt_debt_pct_gdp',np.nan)) else "—"),
            "Gov. Effectiveness": (f"{row.get('gov_effectiveness',np.nan):.2f}"
                                   if not pd.isna(row.get('gov_effectiveness',np.nan)) else "—"),
            "Alignment Score":    (f"{row.get('alignment_score',np.nan):.1f}/100"
                                   if not pd.isna(row.get('alignment_score',np.nan)) else "—"),
            "IFC Presence":       "Yes" if row.get("ifc_presence_score", 0) > 0 else "No",
        }
        # build simple stacked text block
        metrics_text = "<br>".join(
            [f"<b>{k}</b>: {v}" for k, v in metrics.items()]
        )

        st.markdown(
            f"""
            <div style='background:white;border:1px solid {COLORS["border"]};
        border-radius:20px;padding:1.3rem 1.5rem;
        box-shadow:0 2px 10px rgba(10,18,42,0.04);
        height: 300px'>

        <div style='margin:0;font-size:1.2rem;
        color:{COLORS["text"]};
        line-height:1.6'>
        {metrics_text}
        </div>

        </div>""",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("#### Gap Percentile vs. All Countries")
    all_gaps = df["gap_usd"].dropna().sort_values()
    pct      = (all_gaps < row["gap_usd"]).mean() * 100
    # Progress bar now uses the green theme colour via the CSS rule above
    st.progress(
        int(pct) / 100,
        text=f"This country's gap is larger than {pct:.0f}% of all countries",
    )
    soft_card_close()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — GAP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Gap Analysis":
    st.title("Contribution Gap Analysis")
    st.markdown(
        "<p class='subtle-note'>Compare actual IDA21 contributions against "
        "capacity-based targets. A positive gap means the country is contributing "
        "less than its economic capacity suggests it could.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Gap Ranking", "Giving Rate", "Capacity Scatter"])

    with tab1:
        st.markdown("#### Countries Ranked by Contribution Gap")
        # slider — full track now themed via CSS
        top_n   = st.slider("Show top N countries", 10, 80, 30, step=5)
        plot_df = (
            filtered[filtered["gap_usd"].notna()]
            .nlargest(top_n, "gap_usd").sort_values("gap_usd")
        )
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=plot_df["country_name"], x=plot_df["target_bn"],
            orientation="h", name="Capacity Target",
            marker_color="rgba(231,222,205,0.85)",
            hovertemplate="%{y}<br>Target: $%{x:.2f}B<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            y=plot_df["country_name"], x=plot_df["actual_bn"],
            orientation="h", name="Actual IDA21",
            marker_color=COLORS["navy"],
            hovertemplate="%{y}<br>Actual: $%{x:.2f}B<extra></extra>",
        ))
        fig.update_layout(
            barmode="overlay", xaxis_title="USD billions", yaxis_title=None,
            legend=dict(orientation="h", y=1.02),
            margin=dict(l=160, t=40, b=20, r=20),
        )
        apply_dashboard_layout(fig, height=max(400, top_n * 22))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Navy bars = actual IDA21. Light bars = capacity-adjusted target.")

    with tab2:
        st.markdown("#### Giving Rate = Actual ÷ Capacity Target")
        st.markdown(
            "A giving rate of 1.0 means contributing exactly at benchmark. "
            "<1.0 = under-contributing, >1.0 = over-contributing."
        )
        top_n = st.slider("Show top N countries", 10, 200, 40, step=5)

        rate_df = (
            filtered[filtered["giving_rate"].notna() & (filtered["giving_rate"] < 15)]
            .sort_values("giving_rate", ascending=True)
            .tail(top_n)
        )
        fig2 = go.Figure(go.Bar(
            y=rate_df["country_name"], x=rate_df["giving_rate"],
            orientation="h",
            marker_color=[SEGMENT_COLORS.get(str(s), COLORS["muted"]) for s in rate_df["donor_segment"]],
            hovertemplate="%{y}<br>Giving Rate: %{x:.2f}<extra></extra>",
            showlegend=False,
        ))
        # invisible traces just for the legend
        for seg, col in SEGMENT_COLORS.items():
            fig2.add_trace(go.Bar(
                y=[None], x=[None],
                orientation="h",
                name=seg,
                marker_color=col,
                showlegend=True,
            ))
        fig2.add_vline(x=1.0, line_dash="dash", line_color=COLORS["clay"],
                       annotation_text="Benchmark (1.0)", annotation_position="top right", annotation_yshift=-20)
        fig2.update_layout(
            xaxis_title="Giving Rate (actual / target)", yaxis_title=None,
            margin=dict(l=160, t=40, b=20, r=180),
        )
        apply_dashboard_layout(
            fig2,
            height=max(400, len(rate_df) * 14),
            showlegend=True,
            legend_orientation="v",
            legend_x=1.02,
            legend_y=1.0,
            legend_xanchor="left",
            legend_yanchor="top",
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("#### Capacity vs. Giving Rate")
        st.markdown("Bottom-right quadrant = high capacity, low giving rate — highest priority targets.")
        scatter_df = filtered[
            filtered["giving_rate"].notna() & filtered["gap_usd"].notna() &
            (filtered["giving_rate"] < 10)
        ].copy()
        scatter_df["gdp_norm"] = (
            (scatter_df["gdp_bn"] - scatter_df["gdp_bn"].min()) /
            (scatter_df["gdp_bn"].max() - scatter_df["gdp_bn"].min())
        )
        fig3 = px.scatter(
            scatter_df, x="gdp_norm", y="giving_rate",
            color="donor_segment", color_discrete_map=SEGMENT_COLORS,
            hover_name="country_name",
            hover_data={"gdp_norm": False, "giving_rate": ":.2f",
                        "gap_usd": ":,.0f", "p_donate": ":.3f"},
            size="gdp_bn", size_max=40,
            labels={"gdp_norm": "Normalised GDP (0–1)",
                    "giving_rate": "Giving Rate", "donor_segment": "Segment"},
        )
        fig3.add_hline(y=1.0, line_dash="dash", line_color=COLORS["clay"],
                        annotation_text="Benchmark")
        fig3.add_vline(x=scatter_df["gdp_norm"].median(), line_dash="dot",
                        line_color="grey", annotation_text="Median GDP", annotation_position="top right", annotation_yshift=-18)
        for text, x, y in [("High priority", 0.85, 0.08), ("Overperforming", 0.85, 6.5)]:
            fig3.add_annotation(x=x, y=y, text=f"<b>{text}</b>",
                                 showarrow=False, font=dict(size=11, color="grey"))
        apply_dashboard_layout(fig3, height=520)
        st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — PROSPECT RANKING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Prospect Ranking":
    st.title("Prospect Ranking")
    st.markdown(
        "<p class='subtle-note'>Ranked list of non-donor and under-contributing "
        "countries ordered by contribution gap. Use this as the basis for IDA "
        "engagement prioritisation.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "<p style='margin:0 0 6px;font-size:0.85rem;font-weight:600'</p>",
            unsafe_allow_html=True
        )
        show_segments = st.multiselect(
            "Segment Filter", seg_options,
            default=["High-Potential Prospect", "Emerging Prospect", "Under-Contributing Donor"],
        )

    with col2:
        st.markdown(
            "<p style='margin:0 0 6px;font-size:0.85rem;font-weight:600'</p>",
            unsafe_allow_html=True
        )
        # slider — full track themed via CSS
        min_p = st.slider("Minimum P(Donate)", 0.0, 1.0, 0.0, step=0.05)

    prospect_df = filtered[
        filtered["donor_segment"].isin(show_segments) &
        (filtered["p_donate"] >= min_p)
    ].copy()

    display_cols = {
        "country_name":            "Country",
        "income_group":            "Income",
        "peer_group":              "Peer Group",
        "donor_segment":           "Segment",
        "gap_usd":                 "Gap (USD)",
        "gap_pct_ppp_gdp":         "Gap % PPP GDP",
        "giving_rate":             "Giving Rate",
        "p_donate":                "P(Donate)",
        "adjusted_target_usd":     "Capacity Target",
        "actual_contribution_usd": "Actual IDA21",
        "alignment_score":         "Alignment Score",
    }
    tbl = prospect_df[list(display_cols.keys())].rename(columns=display_cols).copy()
    tbl["Gap (USD)"]       = tbl["Gap (USD)"].apply(fmt_usd)
    tbl["Gap % PPP GDP"]   = tbl["Gap % PPP GDP"].apply(
        lambda x: f"{x:.4f}%" if not pd.isna(x) else "—")
    tbl["Giving Rate"]     = tbl["Giving Rate"].apply(
        lambda x: f"{x:.1%}" if not pd.isna(x) else "—")
    tbl["P(Donate)"]       = tbl["P(Donate)"].apply(
        lambda x: f"{x:.3f}" if not pd.isna(x) else "—")
    tbl["Capacity Target"] = tbl["Capacity Target"].apply(fmt_usd)
    tbl["Actual IDA21"]    = tbl["Actual IDA21"].apply(fmt_usd)
    tbl["Alignment Score"] = tbl["Alignment Score"].apply(
        lambda x: f"{x:.1f}" if not pd.isna(x) else "—")

    st.dataframe(tbl.reset_index(drop=True),
                 hide_index=True, use_container_width=True, height=550)
    st.caption(f"{len(prospect_df)} countries shown")

    csv = prospect_df.to_csv(index=False).encode()
    st.download_button("Download filtered table (CSV)", data=csv,
                        file_name="ida_prospect_ranking.csv", mime="text/csv")

    st.divider()
    st.markdown("#### Gap vs. P(Donate) - Engagement Priority Matrix")
    st.markdown(
        "Countries in the top-right have both a large gap and high model-estimated "
        "probability of donating and are the strongest candidates for IDA engagement."
    )
    matrix_df = prospect_df[
        prospect_df["gap_usd"].notna() & prospect_df["p_donate"].notna()
    ].copy()
    fig_matrix = px.scatter(
        matrix_df, x="p_donate", y="gap_usd",
        color="donor_segment", color_discrete_map=SEGMENT_COLORS,
        hover_name="country_name", size="gdp_bn", size_max=45,
        hover_data={"p_donate": ":.3f", "gap_usd": ":,.0f"},
        labels={"p_donate": "P(Donate) — model probability",
                "gap_usd": "Contribution Gap (USD)", "donor_segment": "Segment"},
    )
    fig_matrix.add_hline(y=matrix_df["gap_usd"].median(), line_dash="dot", line_color="grey")
    fig_matrix.add_vline(x=0.5, line_dash="dot", line_color="grey",
                          annotation_text="P=0.5 threshold")
    fig_matrix.update_yaxes(tickformat="$,.0f")
    apply_dashboard_layout(fig_matrix, height=480)
    st.plotly_chart(fig_matrix, use_container_width=True)
    soft_card_close()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — WORLD MAP
# ─────────────────────────────────────────────────────────────────────────────
elif page == "World Map":
    st.title("World Map — Contribution Gap")
    st.markdown(
        "<p class='subtle-note'>"
        "<strong>Dark clay</strong> = large positive gap (under-contributing). &nbsp;"
        "<strong>Deep navy</strong> = over-contributing / Exceeded Target. &nbsp;"
        "<strong>Off-white</strong> = contributing close to benchmark.<br>"
        "<small>Gap uses PPP-adjusted GDP where available; falls back to nominal GDP.</small>"
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    map_df = filtered[filtered["gap_usd"].notna()].copy()
    map_df["_z"]          = map_df["gap_usd"].apply(symlog)
    map_df["_gap_fmt"]    = map_df["gap_usd"].apply(fmt_usd)
    map_df["_actual_fmt"] = map_df["actual_contribution_usd"].apply(fmt_usd)
    map_df["_target_fmt"] = map_df["adjusted_target_usd"].apply(fmt_usd)
    map_df["_rate_fmt"]   = map_df.get("giving_rate_raw", map_df["giving_rate"]).apply(
        lambda x: f"{x:.1%}" if not pd.isna(x) else "—")
    map_df["_ppp_fmt"]    = map_df["gap_pct_ppp_gdp"].apply(
        lambda x: f"{x:.4f}% of PPP GDP" if not pd.isna(x) else "—")

    # Symmetric symlog scale so gap=$0 always maps to snow midpoint
    abs_max = max(
        abs(float(map_df["_z"].quantile(0.98))),
        abs(float(map_df["_z"].quantile(0.02)))
    )
    zmin = -abs_max
    zmax =  abs_max

    tick_dollars = [-1e9, -1e8, -1e7, -1e6, 0, 1e6, 1e7, 1e8, 1e9]
    tick_vals    = [symlog(v) for v in tick_dollars]
    tick_text    = ["-$1B", "-$100M", "-$10M", "-$1M", "$0",
                    "$1M", "$10M", "$100M", "$1B"]

    rdb_colorscale = [
        [0.0,  COLORS["navy"]],
        [0.35, COLORS["soft_blue"]],
        [0.5,  COLORS["snow"]],
        [0.65, COLORS["soft_clay"]],
        [1.0,  COLORS["clay"]],
    ]

    fig_map = go.Figure(go.Choropleth(
        locations=map_df["iso3"],
        z=map_df["_z"],
        locationmode="ISO-3",
        colorscale=rdb_colorscale,
        zmin=zmin, zmax=zmax, zmid=0,
        customdata=map_df[[
            "country_name", "_gap_fmt", "_rate_fmt",
            "_actual_fmt", "_target_fmt", "donor_segment", "_ppp_fmt"
        ]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Segment: %{customdata[5]}<br>"
            "Gap: %{customdata[1]}<br>"
            "Giving Rate: %{customdata[2]}<br>"
            "Actual IDA21: %{customdata[3]}<br>"
            "Capacity Target: %{customdata[4]}<br>"
            "Gap % PPP GDP: %{customdata[6]}"
            "<extra></extra>"
        ),
        colorbar=dict(
            title=dict(text="Contribution Gap", side="right", font=dict(size=13)),
            tickvals=tick_vals + [zmax * 1.08, zmin * 1.08],
            ticktext=tick_text + ["▲ Under-contributor", "▼ Exceeded Target"],
            tickfont=dict(size=10), len=0.75, thickness=18, outlinewidth=0,
        ),
        marker_line_color="white", marker_line_width=0.5,
    ))

    fig_map.update_layout(
        geo=dict(
            showland=True, landcolor="lightgray",
            showframe=False, showcoastlines=True,
            coastlinecolor="white", projection_type="natural earth",
        ),
        margin=dict(l=0, r=0, t=10, b=0), height=580,
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif",
                  color=COLORS["text"], size=13),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.divider()
    st.markdown("#### Map Data Table")
    map_df["_status"] = map_df["actual_contribution_usd"].apply(
        lambda x: "No contribution" if x == 0 else "Active donor"
    )
    tbl_map = map_df[[
        "country_name", "donor_segment", "peer_group", "_status",
        "_gap_fmt", "_ppp_fmt", "_rate_fmt", "_actual_fmt", "_target_fmt"
    ]].copy()
    tbl_map.columns = [
        "Country", "Segment", "Peer Group", "Status",
        "Gap", "Gap % PPP GDP", "Giving Rate", "Actual IDA21", "Capacity Target"
    ]
    st.dataframe(tbl_map.reset_index(drop=True),
                 hide_index=True, use_container_width=True, height=300)
    soft_card_close()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — MODEL DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Model Diagnostics":
    st.title("Model Diagnostics")
    st.markdown(
        "<p class='subtle-note'>Technical validation of the Heckman two-stage "
        "selection model. Use this page to compare coefficients, review "
        "multicollinearity, and inspect the diagnostics text generated by the "
        "pipeline.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("IMR p-value",            "< 0.001",   delta="Selection bias confirmed", delta_color="normal")
    col2.metric("LR Excl. Restriction",   "p < 0.001", delta="Strong instruments",       delta_color="normal")
    col3.metric("Heckman OOS MAE",        "1.6051",    delta="-0.28 vs OLS: 1.8871",     delta_color="inverse")
    col4.metric("BP Heteroskedasticity",  "p < 0.001", delta="HC3 robust SEs applied",   delta_color="normal")

    st.divider()

    tab1, tab2, tab3 = st.tabs(
        ["Coefficient Comparison", "VIF Table", "Full Diagnostics Text"])

    with tab1:
        st.markdown("#### Heckman vs. Naive OLS — Stage 2 Coefficients")
        st.markdown(
            "Differences highlight the importance of the selection correction. "
            "Variables with large divergence would be substantially biased without "
            "the Heckman correction."
        )
        coef_data = {
            "Variable":  ["log_gdp_level", "fiscal_balance_pct_gdp", "ida_vote_share_lag",
                          "trade_exposure_ida", "log_donation_lag"],
            "Heckman":   [2.2847, 0.4310,  0.1340, -0.0546,  0.5456],
            "Naive OLS": [2.4609, 0.4418,  0.2410, -0.3221,  0.7627],
            "% Change":  ["-7.2%", "-2.5%", "-44.4%", "+83.0%",
                          "-28.5%"],
        }
        coef_df = pd.DataFrame(coef_data)
        fig_coef = go.Figure()
        fig_coef.add_trace(go.Bar(
            name="Heckman", x=coef_df["Variable"], y=coef_df["Heckman"],
            marker_color=COLORS["navy"]))
        fig_coef.add_trace(go.Bar(
            name="Naive OLS", x=coef_df["Variable"], y=coef_df["Naive OLS"],
            marker_color=COLORS["clay"], opacity=0.75))
        fig_coef.add_hline(y=0, line_color="black", line_width=0.8)
        fig_coef.update_layout(barmode="group", yaxis_title="Coefficient")
        apply_dashboard_layout(fig_coef, height=380)
        st.plotly_chart(fig_coef, use_container_width=True)
        st.dataframe(coef_df, hide_index=True, use_container_width=True)
        soft_card_close()

    with tab2:
        st.markdown("#### Variance Inflation Factors — Stage 2")
        st.markdown(
            "VIF > 10 indicates significant multicollinearity. "
            "All core variables are below the threshold."
        )
        vif_data = {
            "Variable": ["log_gdp_level", "fiscal_balance_pct_gdp", "ida_vote_share_lag",
                         "trade_exposure_ida", "log_donation_lag", "imr"],
            "VIF":      [6.94, 1.20, 5.03, 1.91, 2.58, 2.03],
        }
        vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)
        fig_vif = go.Figure(go.Bar(
            y=vif_df["Variable"], x=vif_df["VIF"], orientation="h",
            marker_color=[
                COLORS["clay"] if v > 10 else COLORS["green"] if v > 5 else COLORS["navy"]
                for v in vif_df["VIF"]
            ],
            text=[f"{v:.2f}" for v in vif_df["VIF"]], textposition="outside",
        ))
        fig_vif.add_vline(x=10, line_dash="dash", line_color=COLORS["clay"],
                           annotation_text="VIF=10 threshold")
        fig_vif.update_layout(xaxis_title="VIF", margin=dict(l=180))
        apply_dashboard_layout(fig_vif, height=340, showlegend=False)
        st.plotly_chart(fig_vif, use_container_width=True)
        soft_card_close()

    with tab3:
        diag_text = load_diagnostics()
        if not diag_text:
            st.info(
                "Diagnostics file not found at outputs/heckman_diagnostics.txt. "
                "Run main.py to generate it."
            )
        else:
            import re

            border_color = COLORS["border"]
            navy         = COLORS["navy"]
            green        = COLORS["green"]
            subtext      = COLORS["subtext"]
            text         = COLORS["text"]
            bone         = COLORS["bone"]
            snow         = COLORS["snow"]
            clay         = COLORS["clay"]

            def split_columns(line):
                return [c.strip() for c in re.split(r" {2,}", line.strip()) if c.strip()]

            def is_column_subheading(line):
                """Lines like 'Variable    Heckman  Naive OLS  % Change'
                or 'Model    MAE    RMSE' — text-only multi-column rows."""
                cols = split_columns(line)
                return (
                    len(cols) >= 3
                    and not any(re.search(r"^-?\d+\.?\d*([%]?)$", c) for c in cols)
                    and not re.match(r"^\d+\.", line.strip())
                    and len(line.strip()) > 5
                )

            def is_data_row(line):
                """Lines with 2+ columns where at least one cell is numeric."""
                cols = split_columns(line)
                return (
                    len(cols) >= 2
                    and any(re.search(r"-?\d+\.?\d*", c) for c in cols)
                    and len(line.strip()) > 5
                )

            def is_numbered_heading(line):
                """Lines like '1. IMR Significance Test' or '2. Exclusion Restriction...'"""
                return bool(re.match(r"^\d+\.\s+\S", line.strip()))

            def is_title(line):
                """The main title — all caps or contains 'Heckman Selection Model'"""
                s = line.strip()
                return (
                    "heckman" in s.lower() and "model" in s.lower()
                ) or (s.isupper() and len(s) > 8)

            lines     = diag_text.split("\n")
            max_cols  = 1
            for line in lines:
                if is_column_subheading(line) or is_data_row(line):
                    max_cols = max(max_cols, len(split_columns(line)))

            tbody_html = ""
            i = 0

            while i < len(lines):
                line     = lines[i]
                stripped = line.strip()

                # skip blank lines and raw dividers
                if not stripped or re.match(r"^[=\-]{4,}$", stripped):
                    i += 1
                    continue

                # ── TITLE — large, navy background, white bold centred text
                if is_title(stripped):
                    tbody_html += (
                        f"<tr style='background:{navy}'>"
                        f"<td colspan='{max_cols}' style='"
                        f"padding:0.85rem 1rem;"
                        f"font-size:1rem;"
                        f"font-weight:700;"
                        f"color:white;"
                        f"letter-spacing:-0.01em;"
                        f"text-align:left'>"
                        f"{stripped}</td></tr>"
                    )
                    i += 1
                    continue

                # ── NUMBERED HEADING — green left border, semibold, bone background
                if is_numbered_heading(stripped):
                    tbody_html += (
                        f"<tr style='background:{bone}'>"
                        f"<td colspan='{max_cols}' style='"
                        f"padding:0.6rem 1rem;"
                        f"font-size:0.88rem;"
                        f"font-weight:700;"
                        f"color:{navy};"
                        f"border-top:2px solid {border_color};"
                        f"border-bottom:1px solid {clay};"
                        f"border-left:4px solid {border_color}'>"
                        f"{stripped}</td></tr>"
                    )
                    i += 1
                    continue

                # ── COLUMN SUBHEADING — light bone, uppercase small labels
                if is_column_subheading(line):
                    cells    = split_columns(line)
                    td_cells = ""
                    for cell in cells:
                        td_cells += (
                            f"<td style='"
                            f"padding:0.42rem 1rem;"
                            f"font-size:0.75rem;"
                            f"font-weight:700;"
                            f"text-transform:uppercase;"
                            f"letter-spacing:0.05em;"
                            f"color:{navy};"
                            f"background:{bone};"
                            f"border-bottom:2px solid {border_color}'>"
                            f"{cell}</td>"
                        )
                    for _ in range(max_cols - len(cells)):
                        td_cells += (
                            f"<td style='background:{bone};"
                            f"border-bottom:2px solid {border_color}'></td>"
                        )
                    tbody_html += f"<tr>{td_cells}</tr>"
                    i += 1
                    continue

                # ── DATA ROW — white background, grey text, alternating handled below
                if is_data_row(line):
                    cells    = split_columns(line)
                    td_cells = ""
                    for ci, cell in enumerate(cells):
                        fw = "400" if ci == 0 else "400"
                        tc = navy if ci == 0 else navy
                        td_cells += (
                            f"<td style='"
                            f"padding:0.4rem 1rem;"
                            f"font-size:0.83rem;"
                            f"font-weight:{fw};"
                            f"color:{tc};"
                            f"border-bottom:1px solid {border_color}'>"
                            f"{cell}</td>"
                        )
                    for _ in range(max_cols - len(cells)):
                        td_cells += (
                            f"<td style='border-bottom:1px solid {border_color}'></td>"
                        )
                    tbody_html += f"<tr style='background:white'>{td_cells}</tr>"
                    i += 1
                    continue

                # ── NORMAL TEXT — full width, grey, readable body size
                tbody_html += (
                    f"<tr style='background:white'>"
                    f"<td colspan='{max_cols}' style='"
                    f"padding:0.38rem 1rem;"
                    f"font-size:0.83rem;"
                    f"color:{navy};"
                    f"line-height:1.6;"
                    f"border-bottom:1px solid {border_color}'>"
                    f"{stripped}</td></tr>"
                )
                i += 1

            full_html = (
                f"<div style='overflow-x:auto;border-radius:20px;"
                f"border:1px solid {border_color};"
                f"box-shadow:0 2px 10px rgba(10,18,42,0.04)'>"
                f"<table style='width:100%;border-collapse:collapse;"
                f"font-family:Inter,Arial,sans-serif'>"
                f"<tbody>{tbody_html}</tbody>"
                f"</table></div>"
            )
            st.markdown(full_html, unsafe_allow_html=True)
        soft_card_close()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 7 — GLOSSARY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Glossary":
    st.title("Glossary of Terms and Formulas")
    st.markdown(
        "<p class='subtle-note'>Definitions and formulas for all key concepts "
        "used in the Donor Readiness Index.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    from html import escape


    def glossary_card(term, definition, formula=None, source=None):
        # Escape text so Streamlit/HTML does not accidentally interpret it as markup
        term = escape(term)
        definition = escape(definition).replace("\n", "<br>")
        formula = escape(formula).replace("\n", "<br>") if formula else None
        source = escape(source) if source else None

        extra_html = ""

        if formula:
            extra_html += (
                f"<div style='background:{COLORS['bone']};border-radius:8px;"
                f"padding:0.6rem 0.9rem;margin:0.5rem 0;"
                f"font-family:monospace;font-size:0.88rem;color:{COLORS['navy']}'>"
                f"{formula}</div>"
            )

        if source:
            extra_html += (
                f"<p style='margin:6px 0 0;font-size:0.75rem;color:{COLORS['muted']}'>"
                f"Source: {source}</p>"
            )

        st.markdown(
            f"""
            <div style='background:white;border:1px solid {COLORS["border"]};
                border-radius:16px;padding:1rem 1.2rem;margin-bottom:0.75rem;
                box-shadow:0 1px 6px rgba(10,18,42,0.04)'>
                <p style='margin:0 0 4px;font-size:1rem;font-weight:700;
                          color:{COLORS["navy"]}'>{term}</p>
                <p style='margin:0;font-size:0.9rem;color:{COLORS["subtext"]};
                          line-height:1.55'>{definition}</p>
                {extra_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Section 1: Core Index Concepts ──────────────────────────────────────
    st.markdown("### Main Index Concepts")

    glossary_card(
        "Donor Readiness Index (DRI)",
        "A composite quantitative measure estimating a country's capacity and "
        "willingness to contribute to IDA replenishments. Combines economic capacity, "
        "fiscal space, development engagement, and strategic alignment.",
    )
    glossary_card(
        "Capacity Target",
        "The estimated contribution a country could make to IDA, computed as its GDP "
        "(PPP-adjusted where available, nominal otherwise) multiplied by the peer-group "
        "benchmark IDA/GDP ratio for its income tier. The benchmark is the GDP-weighted "
        "median contribution rate observed across current donors within the same peer group.",
        formula="Target = GDP_ppp × Benchmark_rate(peer group)",
    )
    glossary_card(
        "Benchmark IDA/GDP Ratio",
        "The GDP-weighted median IDA contribution rate among current donors, computed "
        "separately per income-tier peer group (HIC, UMC, LMC, LIC). Requires at least "
        "3 donors per group; otherwise falls back to the global GDP-weighted median. "
        "PPP-adjusted GDP is used as the weight where available.",
        formula="Benchmark(g) = weighted_median({ contribution_i / GDP_ppp_i : i ∈ donors in group g })",
    )
    glossary_card(
        "Fiscal Modifier",
        "A linear adjustment applied to the capacity target based on a country's fiscal "
        "balance as a percentage of GDP. A 5% surplus produces the maximum +20% upward "
        "adjustment; a 5% deficit produces the maximum −20% downward adjustment. "
        "Missing fiscal data defaults to a modifier of 0. Used in the rule-based "
        "capacity scorer only — the Heckman model uses fiscal balance as a direct regressor.",
        formula="Modifier = clip(fiscal_balance_pct × 0.04,  −0.20,  +0.20)\nAdjusted Target = Target × (1 + Modifier)",
    )
    glossary_card(
        "Contribution Gap",
        "The signed difference between the adjusted capacity target and actual IDA21 "
        "contribution. Positive = shortfall (giving less than capacity). "
        "Negative = over-contribution. In the Heckman model the target is the "
        "expected_contribution from the two-stage prediction; in the rule-based "
        "model it is the fiscal-adjusted benchmark target.",
        formula="Gap = Adjusted Target − Actual Contribution\ngap_pct_ppp_gdp = (Gap / GDP_ppp) × 100",
    )
    glossary_card(
        "Giving Rate",
        "Actual IDA21 contribution expressed as a fraction of the capacity target. "
        "giving_rate_raw is the uncapped ratio — values above 1.0 indicate "
        "over-contribution and trigger the Exceeded Target segment. "
        "giving_rate is capped at 1.0 for use in segment threshold logic.",
        formula="giving_rate_raw = Actual / Adjusted Target\ngiving_rate = min(giving_rate_raw, 1.0)",
    )

    st.divider()

    # ── Section 2: Heckman Model ─────────────────────────────────────────────
    st.markdown("### Heckman Two-Stage Selection Model")

    glossary_card(
        "Selection Bias",
        "The statistical problem arising from the fact that IDA contribution data "
        "only exists for countries that choose to donate. A naive regression on "
        "donors alone would produce biased estimates because donors are not a "
        "random sample of all countries.",
        source="Heckman, J. (1979). Econometrica, 47(1), 153–161",
    )
    glossary_card(
        "Stage 1 — Selection Equation (Probit)",
        "Models the binary decision of whether a country donates in a given "
        "replenishment round. Variables: log GDP per capita, DAC membership, "
        "UN voting alignment, trade openness, governance effectiveness, peer donor pressure. "
        "The last three are exclusion restrictions — present in Stage 1 only.",
        formula="P(D_it = 1) = Φ(z′_it γ)    where Φ = standard normal CDF\n"
                "z = [log_gdp_per_capita, dac_member, un_voting_align,\n"
                "     trade_openness, gov_effectiveness, peer_donor]",
    )
    glossary_card(
        "Stage 2 — Outcome Equation (OLS + IMR)",
        "Models log-donation amount conditional on donating, on the donor subsample only. "
        "The IMR from Stage 1 is included to correct for selection bias. "
        "HC3 robust standard errors applied when Breusch-Pagan p < 0.05. "
        "Round fixed effects included for all replenishment rounds. "
        "sovereign_credit_rating dropped (collinear with GDP/governance).",
        formula="ln(Y_it) = x′_it β + δλ_it + Σ_r α_r Round_r + ε_it\n"
                "x = [log_gdp_level, fiscal_balance_pct_gdp, ida_vote_share_lag,\n"
                "     trade_exposure_ida, log_donation_lag]",
    )
    glossary_card(
        "Inverse Mills Ratio (IMR / λ)",
        "Derived from Stage 1 fitted values. Captures the expected value of the "
        "Stage 2 error attributable to non-random selection into the donor sample. "
        "A significant IMR coefficient confirms selection bias was present. "
        "IMR is clipped to avoid division by near-zero values of Φ(·).",
        formula="λ_it = φ(z′_it γ̂) / max(Φ(z′_it γ̂), 1×10⁻¹⁰)\n"
                "where φ = standard normal PDF, Φ = standard normal CDF",
    )
    glossary_card(
        "Duan Smearing Correction",
        "Applied when exponentiating log-donation predictions back to dollar amounts. "
        "The smearing factor is the mean of the exponentiated Stage 2 residuals, "
        "computed on the training donor subsample.",
        formula="Δ̂ = mean(exp(ε̂_it))   for i in training donors\n"
                "pred_donation_usd = exp(x′β̂ + δ̂λ) × Δ̂",
        source="Duan, N. (1983). JASA, 78(383), 605–610",
    )
    glossary_card(
        "Expected Contribution",
        "Final model prediction. For current donors P(Donate) is set to 1.0 — "
        "their participation is already resolved. For non-donors the p_donate "
        "from Stage 1 scales the predicted amount. A per-country historical giving "
        "rate (or archetype rate for non-donors) is used as the rate, scaled by "
        "the ratio of the IDA21 median rate to the training-period median rate.",
        formula="E[Y_it] = p × rate(iso3) × GDP_usd\n"
                "where p = 1.0 for donors, p = P(Donate) for non-donors\n"
                "and rate is scaled by (IDA21 median rate / training median rate)",
    )
    glossary_card(
        "Gap Confidence Interval (90%)",
        "Approximate 90% confidence bounds on the gap, derived from the Stage 2 "
        "residual standard deviation. The interval widens with country GDP — "
        "larger economies have larger absolute uncertainty on the predicted contribution. "
        "These are approximations: full two-stage propagation of uncertainty would "
        "require block bootstrap.",
        formula="se = √(MSE_resid from Stage 2)\n"
                "gap_lower = gap − 1.645 × se × GDP_usd\n"
                "gap_upper = gap + 1.645 × se × GDP_usd",
    )

    st.divider()

    # ── Section 3: Diagnostic Statistics ────────────────────────────────────
    st.markdown("### Diagnostic Statistics")

    glossary_card(
        "Pseudo R²",
        "A goodness-of-fit measure for the Stage 1 probit model, equivalent to R² "
        "in OLS. Values between 0.20 and 0.40 are considered good fit for probit models.",
        formula="Pseudo R² = 1 − (log-likelihood of full model / log-likelihood of null model)",
    )
    glossary_card(
        "HC3 Robust Standard Errors",
        "Heteroskedasticity-consistent standard errors (variant HC3) applied to "
        "Stage 2 because the Breusch-Pagan test detected non-constant error variance "
        "(p < 0.001). Preferred over HC1/HC2 in small to moderate samples as it "
        "applies a stronger finite-sample correction.",
    )
    glossary_card(
        "Variance Inflation Factor (VIF)",
        "Measures multicollinearity among Stage 2 regressors. A VIF above 10 "
        "indicates a variable is nearly collinear with others and its coefficient "
        "estimate may be unstable. All core variables in this model are below 10.",
        formula="VIF(Xⱼ) = 1 / (1 − R²ⱼ)    where R²ⱼ = R² from regressing Xⱼ on all other regressors",
    )
    glossary_card(
        "Exclusion Restriction",
        "Variables included in Stage 1 (selection equation) but excluded from "
        "Stage 2 (outcome equation). Required for model identification. In this "
        "model: UN voting alignment, peer donor pressure, DAC membership. "
        "Validated with a likelihood ratio test (p < 0.001).",
    )
    glossary_card(
        "Out-of-Sample MAE",
        "Mean Absolute Error evaluated on a holdout set of IDA18–IDA20 observations "
        "not used in training. Heckman MAE = 1.26 vs naive OLS MAE = 1.81 (log "
        "donation units), a 30% improvement confirming the selection correction adds "
        "predictive value.",
        formula="MAE = (1/n) Σ |ln(Ŷᵢ) − ln(Yᵢ)|",
    )

    st.divider()

    # ── Section 4: Country Segments ─────────────────────────────────────────
    st.markdown("### Country Segments")

    segment_defs = [
        ("Exceeded Target", COLORS["green"],
         "Giving rate raw > 1.0. Country is contributing more than the model-estimated "
         "capacity target. Checked before all other segment rules."),
        ("Reliable Donor", COLORS["navy"],
         "Current IDA donor with gap_pct ≤ 20% of expected contribution "
         "(i.e. (expected − actual) / expected ≤ 0.20)."),
        ("Under-Contributing Donor", COLORS["clay"],
         "Current IDA donor with gap_pct > 20% of expected contribution."),
        ("High-Potential Prospect", "#4a9068",
         "Non-donor with P(Donate) ≥ 0.50 from Stage 1 probit."),
        ("Emerging Prospect", COLORS["bone"],
         "Non-donor with 0.20 ≤ P(Donate) < 0.50."),
        ("Low Probability", COLORS["muted"],
         "Non-donor with P(Donate) < 0.20."),
    ]
    for seg_name, seg_color, seg_desc in segment_defs:
        text_col = "#ffffff" if seg_color not in [COLORS["bone"], COLORS["muted"]] else COLORS["text"]
        st.markdown(
            f"""<div style='background:white;border:1px solid {COLORS["border"]};
                border-left:5px solid {seg_color};border-radius:16px;
                padding:0.9rem 1.2rem;margin-bottom:0.75rem;
                box-shadow:0 1px 6px rgba(10,18,42,0.04)'>
                <p style='margin:0 0 4px;font-size:1rem;font-weight:700;
                          color:{seg_color}'>{seg_name}</p>
                <p style='margin:0;font-size:0.9rem;color:{COLORS["subtext"]};
                          line-height:1.55'>{seg_desc}</p>
            </div>""",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 8 — INTERVIEW FINDINGS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Interview Findings":

    st.title("Interview Findings")
    st.markdown(
        "<p class='subtle-note' style='max-width:900px;line-height:1.65'>"
        "Qualitative findings from 9 semi-structured expert interviews with World Bank Group "
        "staff, IDA economists, and a Gates Foundation programme officer. Interviews were "
        "thematically coded across 13 categories (Role and Perspective excluded as non-substantive). "
        "All interviewees participated in their individual capacity — views do not represent the "
        "World Bank, IDA, or any institution. Interviewees are identified by letter code only."
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Interviewee reference key ─────────────────────────────────────────────
    st.markdown("#### Interviewee Reference Key")

    interviewee_rows = [
        ("A", "Donor Engagement, IDA", "WBG – Sovereign/System", "Cautious"),
        ("B", "Water Specialist, WBG", "WBG – Execution/Sector", "Neutral"),
        ("C", "Communication & Partnerships, WBG", "WBG – Execution", "Neutral"),
        ("D", "Financial Advisory, WBG", "WBG – Financial/System", "Cautious"),
        ("E", "Partnerships, IFC", "IFC – Execution", "Neutral"),
        ("F", "Program Officer, Gates Foundation", "Foundation", "Cautious"),
        ("G", "External Affairs, WBG", "WBG – Sovereign/System", "Neutral"),
        ("H", "IDA Economist, WBG", "WBG – Analytical", "Neutral"),
        ("I", "Social Development Specialist, WBG", "WBG – Execution/Sector", "Cautious"),
    ]
    sentiment_colors = {
        "Neutral": COLORS["navy"],
        "Cautious": COLORS["clay"],
        "Optimistic": COLORS["green"],
        "Critical": "#804e49",
    }

    rows_html = ""
    for code, role, actor, sentiment in interviewee_rows:
        sc = sentiment_colors.get(sentiment, COLORS["muted"])
        bdr = COLORS["border"]
        sub = COLORS["subtext"]
        txt = COLORS["text"]
        navy = COLORS["navy"]
        rows_html += (
                "<tr style='background:white'>"
                "<td style='padding:0.5rem 1rem;font-weight:700;color:" + navy + ";font-size:0.9rem;border-bottom:1px solid " + bdr + "'>" + code + "</td>"
                                                                                                                                                    "<td style='padding:0.5rem 1rem;font-size:0.85rem;color:" + txt + ";border-bottom:1px solid " + bdr + "'>" + role + "</td>"
                                                                                                                                                                                                                                                                        "<td style='padding:0.5rem 1rem;font-size:0.82rem;color:" + sub + ";border-bottom:1px solid " + bdr + "'>" + actor + "</td>"
                                                                                                                                                                                                                                                                                                                                                                                             "<td style='padding:0.5rem 1rem;border-bottom:1px solid " + bdr + "'>"
                                                                                                                                                                                                                                                                                                                                                                                                                                                               "<span style='background:" + sc + ";color:white;border-radius:20px;padding:2px 10px;font-size:0.75rem;font-weight:600'>" + sentiment + "</span></td>"
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  "</tr>"
        )

    bone = COLORS["bone"]
    navy = COLORS["navy"]
    bdr = COLORS["border"]
    th_s = "padding:0.55rem 1rem;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;color:" + navy + ";text-align:left;border-bottom:2px solid " + bdr
    st.markdown(
        "<div style='overflow-x:auto;border-radius:16px;border:1px solid " + bdr + ";box-shadow:0 2px 10px rgba(10,18,42,0.04)'>"
                                                                                   "<table style='width:100%;border-collapse:collapse;font-family:Inter,Arial,sans-serif'>"
                                                                                   "<thead><tr style='background:" + bone + "'>"
                                                                                                                            "<th style='" + th_s + "'>Code</th>"
                                                                                                                                                   "<th style='" + th_s + "'>Role</th>"
                                                                                                                                                                          "<th style='" + th_s + "'>Actor Type</th>"
                                                                                                                                                                                                 "<th style='" + th_s + "'>Sentiment</th>"                                                                                                                                                                                                                 
                                                                                                                                                                                                                                               "</tr></thead><tbody>" + rows_html + "</tbody></table></div>",
        unsafe_allow_html=True,
    )
    st.caption("All views expressed are personal and do not represent any institution.")
    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Theme Frequency", "Code by Interviewee", "Heatmap", "Key Quotes"
    ])

    # Accurate counts from raw coded transcript data (Role & Perspective excluded)
    THEMES = [
        "Capital Flow Constraints",
        "Risk Types",
        "Gaps in Toolkit",
        "Instruments Effectiveness",
        "Country Platforms & Coalitions",
        "Data & Transparency",
        "Fragmentation",
        "Geopolitics / ODA Pressure",
        "Philanthropy Engagement",
        "Private Sector",
        "Actor-Specific",
        "Role of IDA",
        "Solutions & Innovations",
    ]
    INTERVIEWEES = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

    # Rows = themes, columns = interviewees A–I
    COUNTS = {
        "Capital Flow Constraints": [5, 11, 3, 8, 5, 13, 15, 12, 3],
        "Risk Types": [1, 3, 5, 5, 0, 2, 3, 0, 8],
        "Gaps in Toolkit": [2, 3, 4, 1, 3, 6, 1, 1, 3],
        "Instruments Effectiveness": [7, 17, 12, 10, 12, 12, 13, 3, 8],
        "Country Platforms & Coalitions": [0, 6, 4, 1, 1, 0, 1, 1, 0],
        "Data & Transparency": [0, 4, 0, 0, 1, 3, 1, 0, 0],
        "Fragmentation": [8, 4, 4, 2, 6, 5, 7, 4, 0],
        "Geopolitics / ODA Pressure": [1, 1, 0, 0, 2, 1, 4, 3, 0],
        "Philanthropy Engagement": [7, 4, 5, 1, 7, 2, 7, 0, 0],
        "Private Sector": [3, 6, 4, 2, 3, 3, 4, 0, 1],
        "Actor-Specific": [1, 1, 3, 1, 2, 3, 2, 3, 3],
        "Role of IDA": [5, 1, 4, 2, 0, 2, 2, 0, 1],
        "Solutions & Innovations": [1, 1, 1, 0, 0, 1, 0, 2, 0],
    }

    totals = {t: sum(COUNTS[t]) for t in THEMES}
    n_coverage = {t: sum(1 for v in COUNTS[t] if v > 0) for t in THEMES}

    # ── Tab 1: Theme frequency bar chart ─────────────────────────────────────
    with tab1:
        st.markdown("#### Total Coded Turns by Theme")
        st.markdown(
            "<p class='subtle-note'>Ranked by total frequency across all 9 interviews. "
            "Colour reflects breadth of coverage (N interviewees applying code at least once).</p>",
            unsafe_allow_html=True,
        )
        sorted_themes = sorted(THEMES, key=lambda t: totals[t])
        bar_cols = [
            COLORS["navy"] if n_coverage[t] >= 7 else
            COLORS["green"] if n_coverage[t] >= 5 else
            COLORS["clay"] if n_coverage[t] >= 3 else
            COLORS["muted"]
            for t in sorted_themes
        ]
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            y=sorted_themes,
            x=[totals[t] for t in sorted_themes],
            orientation="h",
            marker_color=bar_cols,
            marker_line_width=0,
            text=[str(totals[t]) + " turns · " + str(n_coverage[t]) + "/9" for t in sorted_themes],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Total: %{x} turns<extra></extra>",
        ))
        for label, col in [
            ("Broad (7–9/9)", COLORS["navy"]),
            ("Moderate (5–6/9)", COLORS["green"]),
            ("Limited (3–4/9)", COLORS["clay"]),
        ]:
            fig_bar.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=col, symbol="square"),
                name=label, showlegend=True,
            ))
        fig_bar.update_layout(
            height=500,
            xaxis_title="Total coded turns",
            yaxis_title=None,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter,Arial,sans-serif", color=COLORS["text"], size=12),
            margin=dict(t=20, b=20, l=20, r=180),
            xaxis=dict(gridcolor=COLORS["border"]),
            yaxis=dict(showgrid=False),
            legend=dict(orientation="v", x=1.02, y=1.0, xanchor="left", yanchor="top",
                        font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Tab 2: Stacked bar by interviewee (top 4 per person) ─────────────────
    with tab2:
        st.markdown("#### Code Distribution by Interviewee")
        st.markdown(
            "<p class='subtle-note'>Total coded turns per interviewee. "
            "Top 4 themes per interviewee are highlighted individually — "
            "all remaining themes grouped as Other.</p>",
            unsafe_allow_html=True,
        )

        # Find top 4 themes per interviewee
        top4_per_person = {}
        for j, person in enumerate(INTERVIEWEES):
            scores = {t: COUNTS[t][j] for t in THEMES}
            ranked = sorted(scores, key=lambda t: scores[t], reverse=True)
            top4_per_person[person] = [t for t in ranked if scores[t] > 0][:4]

        all_top4 = []
        for themes in top4_per_person.values():
            for t in themes:
                if t not in all_top4:
                    all_top4.append(t)

        top4_palette = [
            COLORS["navy"], COLORS["green"], COLORS["clay"], "#8eb1d1",
            "#c98f86", "#82c49b", "#b39ddb", "#f0a86e", "#4a9068", "#d4a96a",
        ]
        color_map = {t: top4_palette[i % len(top4_palette)] for i, t in enumerate(all_top4)}

        fig_stack = go.Figure()
        for theme in all_top4:
            y_vals = [
                COUNTS[theme][j] if theme in top4_per_person[p] else 0
                for j, p in enumerate(INTERVIEWEES)
            ]
            fig_stack.add_trace(go.Bar(
                name=theme, x=INTERVIEWEES, y=y_vals,
                marker_color=color_map[theme], marker_line_width=0,
            ))

        other_vals = [
            sum(COUNTS[t][j] for t in THEMES if t not in top4_per_person[p])
            for j, p in enumerate(INTERVIEWEES)
        ]
        fig_stack.add_trace(go.Bar(
            name="Other", x=INTERVIEWEES, y=other_vals,
            marker_color=COLORS["muted"], marker_line_width=0,
        ))

        fig_stack.update_layout(
            barmode="stack", height=420,
            xaxis_title="Interviewee", yaxis_title="Coded turns",
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Inter,Arial,sans-serif", color=COLORS["text"], size=12),
            margin=dict(t=20, b=20, l=20, r=210),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor=COLORS["border"]),
            legend=dict(orientation="v", x=1.02, y=1.0, xanchor="left", yanchor="top",
                        font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_stack, use_container_width=True)
        st.caption("Colours show each interviewee's top 4 themes. Remaining themes grouped as Other (grey).")

    # ── Tab 3: Heatmap ────────────────────────────────────────────────────────
    with tab3:
        st.markdown("#### Thematic Code Frequency Across Expert Interviews (N = 9)")
        st.markdown(
            "<p class='subtle-note'>Cell values = coded turns per interviewee. "
            "Colour intensity reflects frequency.</p>",
            unsafe_allow_html=True,
        )

        heatmap_z = [COUNTS[t] for t in THEMES]
        heatmap_text = [[str(v) if v > 0 else "" for v in row] for row in heatmap_z]

        fig_hm = go.Figure(go.Heatmap(
            z=heatmap_z,
            x=INTERVIEWEES,
            y=THEMES,
            text=heatmap_text,
            texttemplate="%{text}",
            textfont=dict(size=12, color=COLORS["text"]),
            colorscale=[
                [0.0, "#ffffff"],
                [0.01, COLORS["bone"]],
                [0.3, "#b8d0e8"],
                [0.6, COLORS["soft_blue"]],
                [1.0, COLORS["navy"]],
            ],
            zmin=0, zmax=17,
            showscale=True,
            colorbar=dict(
                title=dict(text="Coded Turns", font=dict(size=11)),
                tickfont=dict(size=10), thickness=14, len=0.8, outlinewidth=0,
            ),
            hovertemplate="<b>%{y}</b><br>Interviewee %{x}<br>Coded turns: %{z}<extra></extra>",
            xgap=3, ygap=3,
        ))
        fig_hm.update_layout(
            height=500,
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Inter,Arial,sans-serif", color=COLORS["text"], size=12),
            margin=dict(t=20, b=20, l=220, r=20),
            xaxis=dict(title="Interviewee", showgrid=False, side="top"),
            yaxis=dict(showgrid=False, autorange="reversed"),
        )
        st.plotly_chart(fig_hm, use_container_width=True)
        st.caption("Dark blue = high frequency (10+ turns) · mid blue = moderate · white = not coded.")

    # ── Tab 4: Key Quotes ─────────────────────────────────────────────────────
    with tab4:
        st.markdown("#### Selected Quotes by Theme")
        st.markdown(
            "<p class='subtle-note'>Representative coded excerpts grouped by theme. "
            "Attributed by interviewee letter code only.</p>",
            unsafe_allow_html=True,
        )


        def quote_card(letter, quote, memo, tc):
            bdr = COLORS["border"]
            txt = COLORS["text"]
            sub = COLORS["subtext"]
            memo_part = (" — " + memo) if memo else ""
            st.markdown(
                "<div style='background:white;border:1px solid " + bdr + ";"
                                                                         "border-left:4px solid " + tc + ";border-radius:14px;"
                                                                                                         "padding:0.85rem 1.1rem;margin-bottom:0.6rem;"
                                                                                                         "box-shadow:0 1px 4px rgba(10,18,42,0.04)'>"
                                                                                                         "<p style='margin:0 0 6px;font-size:0.83rem;font-style:italic;"
                                                                                                         "color:" + txt + ";line-height:1.6'>\"" + quote + "\"</p>"
                                                                                                                                                           "<p style='margin:0;font-size:0.75rem;color:" + sub + "'>"
                                                                                                                                                                                                                 "<strong>Interviewee " + letter + "</strong>" + memo_part +
                "</p></div>",
                unsafe_allow_html=True,
            )


        themes_quotes = [
            {
                "label": "Instruments Effectiveness",
                "color": COLORS["navy"],
                "summary": "The breadth of IDA's toolkit was broadly acknowledged, but concerns centred on fragmented entry points, proof-of-concept gaps in blended structures, and shareholder complaints about accessibility.",
                "quotes": [
                    ("A",
                     "The breadth and the breadth of the toolkit is there — but I've certainly heard complaints about it from shareholders.",
                     "on toolkit range vs accessibility"),
                    ("B", "The goal is to crowd in investments and financing from different stakeholders.",
                     "on instrument design intent"),
                    ("F", "They don't have faith in the ability of the quality of the pipeline.",
                     "on IFC pipeline quality concerns"),
                ],
            },
            {
                "label": "Capital Flow Constraints",
                "color": COLORS["clay"],
                "summary": "The scale of philanthropic and private capital falls short of IDA's needs. Concerns centred on influence over policy frameworks, nervousness about non-sovereign actors, and structural barriers to pipeline access.",
                "quotes": [
                    ("A",
                     "It all gets down to who will be able to develop, who will have influence over our policy framework.",
                     "on philanthropic governance concerns"),
                    ("C", "Unless the runway is set in, we're not going to be able to do anything.",
                     "on pipeline pre-conditions"),
                    ("G", "The pipeline is the real constraint — not the appetite.", "on private sector readiness"),
                ],
            },
            {
                "label": "Fragmentation",
                "color": "#8eb1d1",
                "summary": "System fragmentation — across actors, instruments, and incentive structures — was cited as a persistent structural barrier. Too many ineffective players and insufficient coordination mechanisms were recurring concerns.",
                "quotes": [
                    ("A", "Too many, too many players, too many ineffective players.", "on system fragmentation"),
                    ("C",
                     "The biggest issue is that the pipeline is not structured in a way that enables philanthropic and private sector partners to come in.",
                     "on pipeline architecture"),
                    ("E", "There's still levels of this that are missing between the sectors to connect the tissue.",
                     "on inter-sectoral coordination"),
                ],
            },
            {
                "label": "Philanthropy Engagement",
                "color": COLORS["green"],
                "summary": "Sovereign donors exhibit varying levels of trust in philanthropic actors. Some are skeptical about the scale and motives of philanthropic capital; others see it as a complementary and growing channel.",
                "quotes": [
                    ("A",
                     "There are donors that are far less trusting of philanthropies and their motives and the way they operate.",
                     "on sovereign donor scepticism"),
                    ("C", "Do philanthropies come in across the capital stack of the mission?",
                     "on structural integration"),
                    ("E", "Foundations like Hilton and CIFF are already operating in this space.",
                     "on existing philanthropic engagement"),
                ],
            },
            {
                "label": "Risk Types",
                "color": COLORS["clay"],
                "summary": "Political, regulatory, and currency risks were identified as the primary deterrents to private sector engagement in IDA-eligible countries. Investors consistently overprice generalised risk due to information asymmetries.",
                "quotes": [
                    ("D",
                     "Is there going to be strikes? There's a lot of uncertainties that come into low-income countries.",
                     "on political and labour risk"),
                    ("C",
                     "The problem with development — the private sector is crucial, but minimising risk is essential.",
                     "on blended finance design imperative"),
                    ("I",
                     "Investors often overprice uncertainty when they lack granular information, trusted local partners, or credible risk-mitigation tools.",
                     "on information asymmetry"),
                ],
            },
            {
                "label": "Role of IDA",
                "color": COLORS["green"],
                "summary": "IDA is seen as uniquely positioned to coordinate across the capital stack and legitimise new donor categories — but must act while its financial credibility remains intact.",
                "quotes": [
                    ("A", "I think now is the time that IDA can legitimately speak to the financial efficiency.",
                     "on IDA's window of opportunity"),
                    ("C", "They join forces and identify areas where each can fit in — across the capital stack.",
                     "on collaborative financing structures"),
                    ("D",
                     "All of IDA is depending on governments to see it as in their enlightened self-interest to keep donating concessional money.",
                     "on structural dependency risk"),
                ],
            },
            {
                "label": "Geopolitics / ODA Pressure",
                "color": COLORS["muted"],
                "summary": "ODA peaked in 2023 and is now under pressure from right-wing political shifts, defence spending competition, and declining public trust in development effectiveness.",
                "quotes": [
                    ("G", "We have seen peak ODA in 2023 — above $220 billion roughly.", "on ODA trajectory"),
                    ("B",
                     "Global geopolitical changes are significantly affecting developing nations' ability to maintain development financing.",
                     "on systemic pressures"),
                    ("H", "The fiscal space for ODA is shrinking in most donor countries.",
                     "on sovereign budget constraints"),
                ],
            },
            {
                "label": "Solutions & Innovations",
                "color": COLORS["green"],
                "summary": "Solutions ranged from tailored country-level engagement to global pooled de-risking vehicles. Adaptive programming and learning loops were emphasised as necessary for sustained impact.",
                "quotes": [
                    ("F", "What do we learn from this progress and how can we continue to adapt?",
                     "on adaptive programming"),
                    ("H", "It needs to be tailored at the end of the day. That's my main point.",
                     "on country-specific approaches"),
                    ("H",
                     "A global fund to pool resources for de-risking private sector investments could unlock capital at scale.",
                     "on proposed structural innovation"),
                ],
            },
        ]

        for theme in themes_quotes:
            tc = theme["color"]
            navy = COLORS["navy"]
            sub = COLORS["subtext"]
            st.markdown(
                "<div style='margin:1.2rem 0 0.4rem;display:flex;align-items:center;gap:10px'>"
                "<div style='width:4px;height:1.4rem;background:" + tc + ";border-radius:2px'></div>"
                                                                         "<p style='margin:0;font-size:1rem;font-weight:700;color:" + navy + "'>" +
                theme["label"] + "</p>"
                                 "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p style='color:" + sub + ";font-size:0.88rem;margin:0 0 0.5rem;line-height:1.6'>"
                + theme["summary"] + "</p>",
                unsafe_allow_html=True,
            )
            for letter, quote, memo in theme["quotes"]:
                quote_card(letter, quote, memo, tc)
