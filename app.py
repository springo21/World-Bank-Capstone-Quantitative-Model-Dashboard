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
    page_title="IDA Donor Readiness Index",
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
    "High-Potential Prospect":   COLORS["green"],
    "Emerging Prospect":         COLORS["bone"],
    "Low Probability":           "#c9c2b5",
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
    df["gap_bn"]    = df["gap_usd"] / 1e9
    df["actual_bn"] = df["actual_contribution_usd"] / 1e9
    df["target_bn"] = df["adjusted_target_usd"] / 1e9

    segment_order = [
        "Reliable Donor", "Under-Contributing Donor",
        "High-Potential Prospect", "Emerging Prospect", "Low Probability",
    ]
    df["donor_segment"] = pd.Categorical(
        df["donor_segment"], categories=segment_order, ordered=True
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
        color: rgba(255,255,255,0.7) !important;
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
]

# Initialise session state for active page
if "page" not in st.session_state:
    st.session_state.page = "Overview"

with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-title">IDA Readiness</div>
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
        "<p style='font-size:0.72rem;font-weight:700;text-transform:uppercase;"
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
    st.title("IDA Sovereign Donor Readiness Index")
    st.markdown(
        "<p class='subtle-note' style='max-width:900px;line-height:1.6'>"
        "This dashboard presents the findings of a quantitative model assessing which "
        "countries have the capacity and readiness to contribute to IDA replenishments, "
        "and by how much they are currently under- or over-contributing relative to their "
        "economic capacity. The model combines a Heckman two-stage selection model with a "
        "rule-based capacity scorer to identify both the probability of donating and the "
        "expected contribution amount for every country in the sample.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    total_gap       = filtered["gap_usd"].sum()
    n_prospects     = len(filtered[filtered["donor_segment"].isin(
                          ["High-Potential Prospect", "Emerging Prospect"])])
    n_under         = len(filtered[filtered["donor_segment"] == "Under-Contributing Donor"])
    n_donors        = len(filtered[filtered["is_donor"]])
    avg_giving_rate = filtered[filtered["is_donor"]]["giving_rate"].median()

    kpi_row([
        ("Total Addressable Gap",      fmt_usd(total_gap),          None, None),
        ("Current Donors in Sample",   str(n_donors),               None, None),
        ("Under-Contributing Donors",  str(n_under),                None, None),
        ("High/Emerging Prospects",    str(n_prospects),            None, None),
        ("Median Giving Rate (donors)",f"{avg_giving_rate:.0%}",    None, None),
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
        "<p class='subtle-note'>Drill into any country's readiness profile and compare "
        "current contributions with capacity-based targets, model predictions, and "
        "structural indicators.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    country_list     = sorted(filtered["country_name"].dropna().tolist())
    selected_country = st.selectbox("Select a country", country_list)
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

    col1, col2 = st.columns(2)
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
        for k, v in metrics.items():
            a, b = st.columns([1.4, 1])
            a.markdown(f"**{k}**")
            b.markdown(v)
        soft_card_close()

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
        st.markdown("#### Giving Rate: Actual ÷ Capacity Target")
        st.markdown(
            "A giving rate of 1.0 means contributing exactly at benchmark. "
            "<1.0 = under-contributing, >1.0 = over-contributing."
        )
        rate_df = (
            filtered[filtered["giving_rate"].notna() & (filtered["giving_rate"] < 15)]
            .sort_values("giving_rate", ascending=True)
        )
        fig2 = go.Figure(go.Bar(
            y=rate_df["country_name"], x=rate_df["giving_rate"],
            orientation="h",
            marker_color=[SEGMENT_COLORS.get(str(s), COLORS["muted"]) for s in rate_df["donor_segment"]],
            hovertemplate="%{y}<br>Giving Rate: %{x:.2f}<extra></extra>",
        ))
        fig2.add_vline(x=1.0, line_dash="dash", line_color=COLORS["clay"],
                        annotation_text="Benchmark (1.0)", annotation_position="top right")
        fig2.update_layout(
            xaxis_title="Giving Rate (actual / target)", yaxis_title=None,
            margin=dict(l=160, t=40, b=20, r=20),
        )
        apply_dashboard_layout(fig2, height=max(400, len(rate_df) * 14))
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
                        line_color="grey", annotation_text="Median GDP")
        for text, x, y in [("High priority", 0.85, 0.08), ("Overperforming", 0.85, 2.5)]:
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
        show_segments = st.multiselect(
            "Filter by segment", seg_options,
            default=["High-Potential Prospect", "Emerging Prospect", "Under-Contributing Donor"],
        )
    with col2:
        # slider — full track themed via CSS
        min_p = st.slider("Minimum P(Donate)", 0.0, 1.0, 0.0, step=0.05)

    prospect_df = filtered[
        filtered["donor_segment"].isin(show_segments) &
        (filtered["p_donate"] >= min_p)
    ].copy()

    display_cols = {
        "country_name":            "Country",
        "income_group":            "Income",
        "donor_segment":           "Segment",
        "gap_usd":                 "Gap (USD)",
        "giving_rate":             "Giving Rate",
        "p_donate":                "P(Donate)",
        "adjusted_target_usd":     "Capacity Target",
        "actual_contribution_usd": "Actual IDA21",
        "alignment_score":         "Alignment Score",
    }
    tbl = prospect_df[list(display_cols.keys())].rename(columns=display_cols).copy()
    tbl["Gap (USD)"]       = tbl["Gap (USD)"].apply(fmt_usd)
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
    st.markdown("#### Gap vs. P(Donate) — Engagement Priority Matrix")
    st.markdown(
        "Countries in the top-right have both a large gap and high model-estimated "
        "probability of donating — strongest candidates for IDA engagement."
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
        "<strong>Deep navy</strong> = over-contributing relative to capacity. &nbsp;"
        "<strong>White</strong> = contributing at benchmark.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    map_df            = filtered[filtered["gap_usd"].notna()].copy()
    map_df["_z"]      = map_df["gap_usd"].apply(symlog)
    zmax              = float(map_df["_z"].quantile(0.98))
    zmin              = float(map_df["_z"].quantile(0.02))
    map_df["_gap_fmt"]    = map_df["gap_usd"].apply(fmt_usd)
    map_df["_actual_fmt"] = map_df["actual_contribution_usd"].apply(fmt_usd)
    map_df["_target_fmt"] = map_df["adjusted_target_usd"].apply(fmt_usd)
    map_df["_rate_fmt"]   = map_df["giving_rate"].apply(
        lambda x: f"{x:.1%}" if not pd.isna(x) else "—")

    tick_dollars = [-1e9, -1e8, -1e7, -1e6, 0, 1e6, 1e7, 1e8, 1e9]
    tick_vals    = [symlog(v) for v in tick_dollars]
    tick_text    = ["-$1B", "-$100M", "-$10M", "-$1M", "$0",
                    "$1M", "$10M", "$100M", "$1B"]

    rdb_colorscale = [
        [0.0,  COLORS["navy"]],
        [0.35, COLORS["soft_blue"]],
        [0.5,  COLORS["snow"]],
        [0.7,  COLORS["soft_clay"]],
        [1.0,  COLORS["clay"]],
    ]

    fig_map = go.Figure(go.Choropleth(
        locations=map_df["iso3"], z=map_df["_z"], locationmode="ISO-3",
        colorscale=rdb_colorscale, zmin=zmin, zmax=zmax, zmid=0,
        customdata=map_df[["country_name", "_gap_fmt", "_rate_fmt",
                            "_actual_fmt", "_target_fmt", "donor_segment"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>Segment: %{customdata[5]}<br>"
            "Gap: %{customdata[1]}<br>Giving Rate: %{customdata[2]}<br>"
            "Actual IDA21: %{customdata[3]}<br>Capacity Target: %{customdata[4]}"
            "<extra></extra>"),
        colorbar=dict(
            title=dict(text="Contribution Gap", side="right", font=dict(size=13)),
            tickvals=tick_vals, ticktext=tick_text,
            tickfont=dict(size=10), len=0.75, thickness=18, outlinewidth=0,
        ),
        marker_line_color="white", marker_line_width=0.5,
    ))
    fig_map.update_layout(
        geo=dict(showland=True, landcolor="lightgray", showframe=False,
                 showcoastlines=True, coastlinecolor="white",
                 projection_type="natural earth"),
        margin=dict(l=0, r=0, t=10, b=0), height=580,
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", color=COLORS["text"], size=13),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.divider()
    st.markdown("#### Map Data Table")
    tbl_map = map_df[["country_name", "donor_segment", "_gap_fmt",
                       "_rate_fmt", "_actual_fmt", "_target_fmt"]].copy()
    tbl_map.columns = ["Country", "Segment", "Gap",
                        "Giving Rate", "Actual IDA21", "Capacity Target"]
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
    col3.metric("Heckman OOS MAE",        "1.2573",    delta="vs OLS: 1.8145",           delta_color="inverse")
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
                          "trade_exposure_ida", "log_donation_lag", "us_eu_ally",
                          "sovereign_credit_rating"],
            "Heckman":   [2.2294, 0.1375,  0.1355, -0.1641,  0.3654, -1.7159, 0.2207],
            "Naive OLS": [2.6406, 0.1459,  0.1606, -0.2940,  0.6283, -0.8609, 0.3890],
            "% Change":  ["-15.6%", "-5.7%", "-15.7%", "+44.2%",
                          "-41.8%", "-99.3%", "-43.3%"],
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
        soft_card_open()
        st.markdown("#### Variance Inflation Factors — Stage 2")
        st.markdown(
            "VIF > 10 indicates serious multicollinearity. "
            "All core variables are below the threshold."
        )
        vif_data = {
            "Variable": ["log_gdp_level", "fiscal_balance_pct_gdp", "ida_vote_share_lag",
                         "trade_exposure_ida", "log_donation_lag", "us_eu_ally",
                         "sovereign_credit_rating", "imr"],
            "VIF":      [7.42, 1.57, 5.40, 2.41, 2.72, 1.76, 1.93, 2.38],
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
        soft_card_open()
        diag_text = load_diagnostics()
        if diag_text:
            st.code(diag_text, language=None)
        else:
            st.info(
                "Diagnostics file not found at `outputs/heckman_diagnostics.txt`. "
                "Run `main.py` to generate it."
            )
        soft_card_close()