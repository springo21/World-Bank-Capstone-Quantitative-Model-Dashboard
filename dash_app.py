from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html, dash_table

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
OUTPUTS = BASE / "outputs"
PROCESSED = BASE / "data" / "processed"

# ── Palette ───────────────────────────────────────────────────────────────────
COLORS = {
    "navy": "#0a122a",
    "green": "#698f3f",
    "snow": "#fbfaf8",
    "bone": "#e7decd",
    "clay": "#804e49",
    "text": "#0a122a",
    "subtext": "#6e6b65",
    "border": "#e7decd",
    "muted": "#bfb7ab",
    "white": "#ffffff",
    "panel": "#f7f4ef",
    "line": "#d9d1c3",
}

SEGMENT_COLORS = {
    "Reliable Donor": COLORS["navy"],
    "Under-Contributing Donor": COLORS["clay"],
    "High-Potential Prospect": COLORS["green"],
    "Emerging Prospect": COLORS["bone"],
    "Low Probability": "#c9c2b5",
}

INCOME_COLORS = {
    "HIC": COLORS["navy"],
    "UMC": COLORS["green"],
    "LMC": COLORS["bone"],
    "LIC": COLORS["clay"],
}

SEGMENT_ORDER = [
    "Reliable Donor",
    "Under-Contributing Donor",
    "High-Potential Prospect",
    "Emerging Prospect",
    "Low Probability",
]

PAGES = [
    "Overview",
    "Country Explorer",
    "Gap Analysis",
    "Prospect Ranking",
    "World Map",
    "Model Diagnostics",
]

# ── Data loading ──────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    dri = pd.read_csv(OUTPUTS / "dri_output.csv")
    alignment = pd.read_csv(PROCESSED / "alignment_scores.csv")
    master = pd.read_csv(PROCESSED / "master.csv")

    df = dri.merge(
        alignment[["iso3", "alignment_score", "ifc_presence_score"]],
        on="iso3", how="left"
    )
    df = df.merge(
        master[[
            "iso3", "gdp_per_capita_usd", "fiscal_balance_pct_gdp",
            "govt_debt_pct_gdp", "gov_effectiveness"
        ]],
        on="iso3", how="left"
    )

    df["gdp_bn"] = df["gdp_usd"] / 1e9
    df["gap_bn"] = df["gap_usd"] / 1e9
    df["actual_bn"] = df["actual_contribution_usd"] / 1e9
    df["target_bn"] = df["adjusted_target_usd"] / 1e9
    df["donor_segment"] = pd.Categorical(
        df["donor_segment"], categories=SEGMENT_ORDER, ordered=True
    )
    df["is_donor"] = df["actual_contribution_usd"] > 0
    return df.sort_values("gap_usd", ascending=False).reset_index(drop=True)


def load_diagnostics() -> str | None:
    path = OUTPUTS / "heckman_diagnostics.txt"
    return path.read_text() if path.exists() else None


DF = load_data()
DIAGNOSTICS_TEXT = load_diagnostics()
SEG_OPTIONS = [s for s in SEGMENT_ORDER if s in DF["donor_segment"].astype(str).unique()]
INCOME_OPTIONS = sorted([x for x in DF["income_group"].dropna().unique().tolist()])
COUNTRY_OPTIONS = sorted(DF["country_name"].dropna().unique().tolist())


# ── Helpers ───────────────────────────────────────────────────────────────────
def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x) / 1e6)


def fmt_usd(v, decimals=2):
    if pd.isna(v):
        return "N/A"
    if abs(v) >= 1e9:
        return f"${v / 1e9:.{decimals}f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.{decimals}f}M"
    return f"${v:,.0f}"


def fmt_pct(v, digits=1):
    return "—" if pd.isna(v) else f"{v:.{digits}%}"


def filter_df(df: pd.DataFrame, segments, incomes, min_gdp):
    out = df.copy()
    if segments:
        out = out[out["donor_segment"].astype(str).isin(segments)]
    if incomes:
        out = out[out["income_group"].isin(incomes)]
    out = out[out["gdp_bn"] >= min_gdp]
    return out.copy()


def base_figure(fig, height=360, show_legend=True):
    fig.update_layout(
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", color=COLORS["text"], size=12),
        margin=dict(t=40, r=20, b=30, l=20),
        legend=(
            dict(orientation="h", y=1.05, x=0, bgcolor="rgba(0,0,0,0)")
            if show_legend else dict()
        ),
    )
    fig.update_xaxes(gridcolor=COLORS["line"], zerolinecolor=COLORS["line"])
    fig.update_yaxes(gridcolor=COLORS["line"], zerolinecolor=COLORS["line"])
    return fig


def card(children, class_name="dash-card", style=None):
    st = {
        "background": COLORS["white"],
        "borderRadius": "22px",
        "padding": "18px 18px 16px 18px",
        "boxShadow": "0 10px 28px rgba(10,18,42,0.05)",
        "border": f"1px solid {COLORS['border']}",
    }
    if style:
        st.update(style)
    return html.Div(children, className=class_name, style=st)


def stat_card(title, value, bg=None, fg=None):
    bg = bg or COLORS["white"]
    fg = fg or COLORS["text"]
    return html.Div(
        [
            html.Div(title, style={"fontSize": "14px", "fontWeight": 700, "opacity": 0.9, "marginBottom": "14px"}),
            html.Div(value, style={"fontSize": "34px", "fontWeight": 700, "lineHeight": 1.1}),
        ],
        style={
            "background": bg,
            "color": fg,
            "borderRadius": "22px",
            "padding": "18px 20px",
            "border": "none" if bg != COLORS["white"] else f"1px solid {COLORS['border']}",
            "boxShadow": "0 10px 28px rgba(10,18,42,0.05)",
            "minHeight": "118px",
        },
    )


def make_table(df, page_size=8):
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        page_size=page_size,
        style_as_list_view=True,
        style_table={"overflowX": "auto", "borderRadius": "18px", "overflow": "hidden"},
        style_header={
            "backgroundColor": COLORS["snow"],
            "fontWeight": 700,
            "color": COLORS["subtext"],
            "border": f"1px solid {COLORS['border']}",
        },
        style_cell={
            "padding": "12px",
            "backgroundColor": "white",
            "border": f"1px solid {COLORS['border']}",
            "color": COLORS["text"],
            "textAlign": "left",
            "fontFamily": "Inter, Arial, sans-serif",
            "fontSize": 14,
            "whiteSpace": "normal",
            "height": "auto",
        },
    )


def donut_figure(seg_counts: pd.DataFrame):
    fig = go.Figure(go.Pie(
        labels=seg_counts["donor_segment"],
        values=seg_counts["count"],
        hole=0.58,
        marker_colors=[SEGMENT_COLORS.get(str(s), COLORS["muted"]) for s in seg_counts["donor_segment"]],
        textinfo="percent",
        hovertemplate="%{label}<br>%{value} countries<extra></extra>",
    ))
    fig.update_layout(showlegend=True, margin=dict(t=10, b=10, l=10, r=10), height=330, paper_bgcolor="white")
    return fig


def top_gap_figure(top10: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top10["country_name"],
        y=top10["gap_usd"] / 1e9,
        marker_color=[SEGMENT_COLORS.get(str(s), COLORS["muted"]) for s in top10["donor_segment"]],
        text=[fmt_usd(v) for v in top10["gap_usd"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Gap: $%{y:.2f}B<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(yaxis_title="Gap (USD billions)", xaxis_title=None)
    return base_figure(fig, height=360, show_legend=False)


def overview_layout(filtered: pd.DataFrame):
    total_gap = filtered["gap_usd"].sum()
    n_prospects = len(filtered[filtered["donor_segment"].astype(str).isin(["High-Potential Prospect", "Emerging Prospect"])])
    n_under = len(filtered[filtered["donor_segment"].astype(str) == "Under-Contributing Donor"])
    n_donors = len(filtered[filtered["is_donor"]])
    donor_rates = filtered[filtered["is_donor"]]["giving_rate"].dropna()
    med_giving = donor_rates.median() if not donor_rates.empty else np.nan

    seg_counts = filtered.groupby("donor_segment", observed=True).size().reset_index(name="count")
    seg_summary = (
        filtered.groupby("donor_segment", observed=True)
        .agg(
            Countries=("iso3", "count"),
            Total_Gap_USD=("gap_usd", "sum"),
            Avg_Giving_Rate=("giving_rate", "mean"),
            Avg_p_donate=("p_donate", "mean"),
        )
        .reset_index()
    )
    seg_summary = seg_summary.rename(columns={"donor_segment": "Segment", "Countries": "# Countries"})
    seg_summary["Total Gap (USD)"] = seg_summary["Total_Gap_USD"].apply(fmt_usd)
    seg_summary["Avg. Giving Rate"] = seg_summary["Avg_Giving_Rate"].apply(fmt_pct)
    seg_summary["Avg. P(Donate)"] = seg_summary["Avg_p_donate"].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "—")
    seg_summary = seg_summary[["Segment", "# Countries", "Total Gap (USD)", "Avg. Giving Rate", "Avg. P(Donate)"]]

    top10 = filtered[filtered["gap_usd"].notna()].nlargest(10, "gap_usd")

    return html.Div([
        html.Div([
            html.Div([
                html.H1("Dashboard", style={"margin": 0, "fontSize": "28px", "fontWeight": 700}),
            ]),
            html.Div([
                dcc.Input(
                    placeholder="Search countries, segments, metrics",
                    type="text",
                    style={
                        "width": "330px",
                        "height": "44px",
                        "borderRadius": "999px",
                        "border": "none",
                        "padding": "0 18px",
                        "background": "#111111",
                        "color": "white",
                        "outline": "none",
                    },
                ),
            ]),
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "18px"}),

        html.Div([
            html.Div(stat_card("Total Addressable Gap", fmt_usd(total_gap, 2), bg=COLORS["green"], fg="white"), style={"flex": 1}),
            html.Div(stat_card("Current Donors", str(n_donors)), style={"flex": 1}),
            html.Div(stat_card("High / Emerging Prospects", str(n_prospects), bg=COLORS["bone"]), style={"flex": 1}),
            html.Div(stat_card("Under-Contributing Donors", str(n_under), bg=COLORS["clay"], fg="white"), style={"flex": 1}),
            html.Div(stat_card("Median Giving Rate", f"{med_giving:.0%}" if not pd.isna(med_giving) else "—"), style={"flex": 1}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "18px"}),

        html.Div([
            html.Div(card([
                html.Div([
                    html.H3("Country Segments", style={"margin": 0}),
                ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "8px"}),
                dcc.Graph(figure=donut_figure(seg_counts), config={"displayModeBar": False})
            ]), style={"flex": "1.1"}),
            html.Div(card([
                html.H3("Segment Summary", style={"marginTop": 0}),
                make_table(seg_summary, page_size=6)
            ]), style={"flex": "1.4"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "18px"}),

        html.Div(card([
            html.H3("Top Countries by Contribution Gap", style={"marginTop": 0}),
            dcc.Graph(figure=top_gap_figure(top10), config={"displayModeBar": False}),
        ])),
    ])


def country_explorer_layout(selected_country: str | None):
    country_name = selected_country or COUNTRY_OPTIONS[0]
    row = DF[DF["country_name"] == country_name].iloc[0]
    seg = row["donor_segment"]
    seg_color = SEGMENT_COLORS.get(str(seg), COLORS["muted"])

    vals = {
        "Actual IDA21": row["actual_contribution_usd"],
        "Capacity Target": row["adjusted_target_usd"],
        "Predicted": row["pred_donation_usd"],
    }
    vals = {k: v for k, v in vals.items() if not pd.isna(v)}
    fig = go.Figure(go.Bar(
        x=list(vals.keys()),
        y=[v / 1e6 for v in vals.values()],
        marker_color=[COLORS["navy"], COLORS["green"], COLORS["clay"]][:len(vals)],
        text=[fmt_usd(v) for v in vals.values()],
        textposition="outside",
        showlegend=False,
    ))
    fig.update_layout(yaxis_title="USD millions")
    fig = base_figure(fig, height=320, show_legend=False)

    details = [
        ("P(Donate)", f"{row['p_donate']:.3f}"),
        ("Giving Rate", fmt_pct(row["giving_rate"])),
        ("IMR (λ)", f"{row['imr']:.3f}" if not pd.isna(row.get("imr")) else "—"),
        ("Fiscal Balance", f"{row.get('fiscal_balance_pct_gdp', np.nan):.1f}% GDP" if not pd.isna(row.get("fiscal_balance_pct_gdp", np.nan)) else "—"),
        ("Govt Debt / GDP", f"{row.get('govt_debt_pct_gdp', np.nan):.1f}%" if not pd.isna(row.get("govt_debt_pct_gdp", np.nan)) else "—"),
        ("Gov. Effectiveness", f"{row.get('gov_effectiveness', np.nan):.2f}" if not pd.isna(row.get("gov_effectiveness", np.nan)) else "—"),
        ("Alignment Score", f"{row.get('alignment_score', np.nan):.1f}/100" if not pd.isna(row.get("alignment_score", np.nan)) else "—"),
        ("IFC Presence", "Yes" if row.get("ifc_presence_score", 0) > 0 else "No"),
    ]

    return html.Div([
        html.Div([
            html.H1(country_name, style={"margin": 0, "fontSize": "32px"}),
            html.Span(str(seg), style={"background": seg_color, "color": "white", "padding": "6px 14px", "borderRadius": "999px", "fontSize": "13px", "fontWeight": 700}),
        ], style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "6px"}),
        html.Div(f"ISO3: {row['iso3']} · Income Group: {row.get('income_group', '—')}", style={"color": COLORS["subtext"], "marginBottom": "18px"}),
        html.Div([
            html.Div(stat_card("GDP", fmt_usd(row["gdp_usd"])), style={"flex": 1}),
            html.Div(stat_card("GDP per Capita", fmt_usd(row.get("gdp_per_capita_usd", np.nan), 0)), style={"flex": 1}),
            html.Div(stat_card("Capacity Target", fmt_usd(row["adjusted_target_usd"]), bg=COLORS["green"], fg="white"), style={"flex": 1}),
            html.Div(stat_card("Actual IDA21", fmt_usd(row["actual_contribution_usd"]), bg=COLORS["bone"]), style={"flex": 1}),
            html.Div(stat_card("Contribution Gap", fmt_usd(row["gap_usd"]), bg=COLORS["clay"], fg="white"), style={"flex": 1}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "18px"}),
        html.Div([
            html.Div(card([html.H3("Contribution Profile", style={"marginTop": 0}), dcc.Graph(figure=fig, config={"displayModeBar": False})]), style={"flex": 1}),
            html.Div(card([
                html.H3("Key Metrics", style={"marginTop": 0}),
                html.Div([
                    html.Div([
                        html.Div(k, style={"fontWeight": 700}),
                        html.Div(v, style={"color": COLORS["subtext"]}),
                    ], style={"display": "flex", "justifyContent": "space-between", "padding": "10px 0", "borderBottom": f"1px solid {COLORS['border']}"})
                    for k, v in details
                ])
            ]), style={"flex": 1}),
        ], style={"display": "flex", "gap": "16px"}),
    ])


def gap_analysis_layout(filtered: pd.DataFrame, top_n: int):
    plot_df = filtered[filtered["gap_usd"].notna()].nlargest(top_n, "gap_usd").sort_values("gap_usd")
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(y=plot_df["country_name"], x=plot_df["target_bn"], orientation="h", name="Capacity Target", marker_color=COLORS["bone"]))
    fig1.add_trace(go.Bar(y=plot_df["country_name"], x=plot_df["actual_bn"], orientation="h", name="Actual IDA21", marker_color=COLORS["navy"]))
    fig1.update_layout(barmode="overlay", xaxis_title="USD billions", yaxis_title=None, margin=dict(l=160, t=30, b=20, r=20))
    fig1 = base_figure(fig1, height=max(420, top_n * 22))

    rate_df = filtered[filtered["giving_rate"].notna() & (filtered["giving_rate"] < 15)].sort_values("giving_rate", ascending=True)
    fig2 = go.Figure(go.Bar(
        y=rate_df["country_name"], x=rate_df["giving_rate"], orientation="h",
        marker_color=[SEGMENT_COLORS.get(str(s), COLORS["muted"]) for s in rate_df["donor_segment"]],
        showlegend=False,
    ))
    fig2.add_vline(x=1.0, line_dash="dash", line_color=COLORS["clay"], annotation_text="Benchmark")
    fig2.update_layout(xaxis_title="Giving Rate", yaxis_title=None, margin=dict(l=160, t=30, b=20, r=20))
    fig2 = base_figure(fig2, height=max(420, len(rate_df) * 14), show_legend=False)

    scatter_df = filtered[filtered["giving_rate"].notna() & filtered["gap_usd"].notna() & (filtered["giving_rate"] < 10)].copy()
    scatter_df["gdp_norm"] = (scatter_df["gdp_bn"] - scatter_df["gdp_bn"].min()) / (scatter_df["gdp_bn"].max() - scatter_df["gdp_bn"].min())
    fig3 = px.scatter(
        scatter_df,
        x="gdp_norm",
        y="giving_rate",
        color="donor_segment",
        color_discrete_map=SEGMENT_COLORS,
        hover_name="country_name",
        hover_data={"gdp_norm": False, "giving_rate": ":.2f", "gap_usd": ":,.0f", "p_donate": ":.3f"},
        size="gdp_bn",
        size_max=42,
        labels={"gdp_norm": "Normalised GDP (0–1)", "giving_rate": "Giving Rate"},
    )
    fig3.add_hline(y=1.0, line_dash="dash", line_color=COLORS["clay"], annotation_text="Benchmark")
    fig3.add_vline(x=scatter_df["gdp_norm"].median(), line_dash="dot", line_color=COLORS["subtext"], annotation_text="Median GDP")
    fig3 = base_figure(fig3, height=460)

    return html.Div([
        html.H1("Gap Analysis", style={"marginTop": 0}),
        html.Div(style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "16px", "marginBottom": "16px"}, children=[
            card([html.H3("Gap Ranking", style={"marginTop": 0}), dcc.Graph(figure=fig1, config={"displayModeBar": False})]),
            card([html.H3("Giving Rate", style={"marginTop": 0}), dcc.Graph(figure=fig2, config={"displayModeBar": False})]),
        ]),
        card([html.H3("Capacity Scatter", style={"marginTop": 0}), dcc.Graph(figure=fig3, config={"displayModeBar": False})]),
    ])


def prospect_ranking_layout(filtered: pd.DataFrame, min_p: float):
    chosen = [s for s in ["High-Potential Prospect", "Emerging Prospect", "Under-Contributing Donor"] if s in SEG_OPTIONS]
    prospect_df = filtered[
        filtered["donor_segment"].astype(str).isin(chosen) &
        (filtered["p_donate"] >= min_p)
    ].copy()

    tbl = prospect_df[[
        "country_name", "income_group", "donor_segment", "gap_usd", "giving_rate",
        "p_donate", "adjusted_target_usd", "actual_contribution_usd", "alignment_score"
    ]].rename(columns={
        "country_name": "Country",
        "income_group": "Income",
        "donor_segment": "Segment",
        "gap_usd": "Gap (USD)",
        "giving_rate": "Giving Rate",
        "p_donate": "P(Donate)",
        "adjusted_target_usd": "Capacity Target",
        "actual_contribution_usd": "Actual IDA21",
        "alignment_score": "Alignment Score",
    }).copy()
    tbl["Gap (USD)"] = tbl["Gap (USD)"].apply(fmt_usd)
    tbl["Giving Rate"] = tbl["Giving Rate"].apply(fmt_pct)
    tbl["P(Donate)"] = tbl["P(Donate)"].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "—")
    tbl["Capacity Target"] = tbl["Capacity Target"].apply(fmt_usd)
    tbl["Actual IDA21"] = tbl["Actual IDA21"].apply(fmt_usd)
    tbl["Alignment Score"] = tbl["Alignment Score"].apply(lambda x: f"{x:.1f}" if not pd.isna(x) else "—")

    matrix_df = prospect_df[prospect_df["gap_usd"].notna() & prospect_df["p_donate"].notna()].copy()
    fig = px.scatter(
        matrix_df,
        x="p_donate",
        y="gap_usd",
        color="donor_segment",
        color_discrete_map=SEGMENT_COLORS,
        hover_name="country_name",
        size="gdp_bn",
        size_max=45,
        labels={"p_donate": "P(Donate)", "gap_usd": "Contribution Gap (USD)"},
    )
    fig.add_hline(y=matrix_df["gap_usd"].median(), line_dash="dot", line_color=COLORS["subtext"])
    fig.add_vline(x=0.5, line_dash="dot", line_color=COLORS["subtext"], annotation_text="P=0.5")
    fig = base_figure(fig, height=470)

    return html.Div([
        html.H1("Prospect Ranking", style={"marginTop": 0}),
        card([html.H3("Prospect Table", style={"marginTop": 0}), html.Div(f"{len(prospect_df)} countries shown", style={"color": COLORS['subtext'], "marginBottom": "10px"}), make_table(tbl, page_size=10)]),
        html.Div(style={"height": "16px"}),
        card([html.H3("Engagement Priority Matrix", style={"marginTop": 0}), dcc.Graph(figure=fig, config={"displayModeBar": False})]),
    ])


def world_map_layout(filtered: pd.DataFrame):
    map_df = filtered[filtered["gap_usd"].notna()].copy()
    map_df["_z"] = map_df["gap_usd"].apply(symlog)
    zmax = float(map_df["_z"].quantile(0.98))
    zmin = float(map_df["_z"].quantile(0.02))
    map_df["_gap_fmt"] = map_df["gap_usd"].apply(fmt_usd)
    map_df["_actual_fmt"] = map_df["actual_contribution_usd"].apply(fmt_usd)
    map_df["_target_fmt"] = map_df["adjusted_target_usd"].apply(fmt_usd)
    map_df["_rate_fmt"] = map_df["giving_rate"].apply(fmt_pct)

    tick_dollars = [-1e9, -1e8, -1e7, -1e6, 0, 1e6, 1e7, 1e8, 1e9]
    tick_vals = [symlog(v) for v in tick_dollars]
    tick_text = ["-$1B", "-$100M", "-$10M", "-$1M", "$0", "$1M", "$10M", "$100M", "$1B"]

    fig = go.Figure(go.Choropleth(
        locations=map_df["iso3"],
        z=map_df["_z"],
        locationmode="ISO-3",
        colorscale=[
            [0.0, COLORS["navy"]],
            [0.35, "#89a7c6"],
            [0.50, COLORS["snow"]],
            [0.70, "#c7938c"],
            [1.0, COLORS["clay"]],
        ],
        zmin=zmin, zmax=zmax, zmid=0,
        customdata=map_df[["country_name", "_gap_fmt", "_rate_fmt", "_actual_fmt", "_target_fmt", "donor_segment"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Segment: %{customdata[5]}<br>"
            "Gap: %{customdata[1]}<br>"
            "Giving Rate: %{customdata[2]}<br>"
            "Actual IDA21: %{customdata[3]}<br>"
            "Capacity Target: %{customdata[4]}<extra></extra>"
        ),
        colorbar=dict(title="Contribution Gap", tickvals=tick_vals, ticktext=tick_text, len=0.72, thickness=16),
        marker_line_color="white",
        marker_line_width=0.5,
    ))
    fig.update_layout(
        geo=dict(showland=True, landcolor="lightgray", showframe=False, showcoastlines=True, coastlinecolor="white", projection_type="natural earth"),
        height=560, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="white", font=dict(color=COLORS["text"]),
    )

    tbl_map = map_df[["country_name", "donor_segment", "_gap_fmt", "_rate_fmt", "_actual_fmt", "_target_fmt"]].copy()
    tbl_map.columns = ["Country", "Segment", "Gap", "Giving Rate", "Actual IDA21", "Capacity Target"]

    return html.Div([
        html.H1("World Map", style={"marginTop": 0}),
        card([dcc.Graph(figure=fig, config={"displayModeBar": False})]),
        html.Div(style={"height": "16px"}),
        card([html.H3("Map Data Table", style={"marginTop": 0}), make_table(tbl_map, page_size=10)]),
    ])


def diagnostics_layout():
    coef_df = pd.DataFrame({
        "Variable": ["log_gdp_level", "fiscal_balance_pct_gdp", "ida_vote_share_lag", "trade_exposure_ida", "log_donation_lag", "us_eu_ally", "sovereign_credit_rating"],
        "Heckman": [2.2294, 0.1375, 0.1355, -0.1641, 0.3654, -1.7159, 0.2207],
        "Naive OLS": [2.6406, 0.1459, 0.1606, -0.2940, 0.6283, -0.8609, 0.3890],
        "% Change": ["-15.6%", "-5.7%", "-15.7%", "+44.2%", "-41.8%", "-99.3%", "-43.3%"],
    })
    fig_coef = go.Figure()
    fig_coef.add_trace(go.Bar(name="Heckman", x=coef_df["Variable"], y=coef_df["Heckman"], marker_color=COLORS["navy"]))
    fig_coef.add_trace(go.Bar(name="Naive OLS", x=coef_df["Variable"], y=coef_df["Naive OLS"], marker_color=COLORS["clay"], opacity=0.8))
    fig_coef.add_hline(y=0, line_color="black", line_width=0.8)
    fig_coef = base_figure(fig_coef, height=360)

    vif_df = pd.DataFrame({
        "Variable": ["log_gdp_level", "fiscal_balance_pct_gdp", "ida_vote_share_lag", "trade_exposure_ida", "log_donation_lag", "us_eu_ally", "sovereign_credit_rating", "imr"],
        "VIF": [7.42, 1.57, 5.40, 2.41, 2.72, 1.76, 1.93, 2.38],
    }).sort_values("VIF", ascending=False)
    fig_vif = go.Figure(go.Bar(
        y=vif_df["Variable"], x=vif_df["VIF"], orientation="h",
        marker_color=[COLORS["clay"] if v > 10 else COLORS["green"] if v > 5 else COLORS["navy"] for v in vif_df["VIF"]],
        text=[f"{v:.2f}" for v in vif_df["VIF"]], textposition="outside", showlegend=False,
    ))
    fig_vif.add_vline(x=10, line_dash="dash", line_color=COLORS["clay"], annotation_text="VIF=10")
    fig_vif.update_layout(margin=dict(l=180, t=30, b=20, r=20))
    fig_vif = base_figure(fig_vif, height=340, show_legend=False)

    return html.Div([
        html.H1("Model Diagnostics", style={"marginTop": 0}),
        html.Div([
            html.Div(stat_card("IMR p-value", "< 0.001", bg=COLORS["green"], fg="white"), style={"flex": 1}),
            html.Div(stat_card("LR Exclusion Test", "p < 0.001"), style={"flex": 1}),
            html.Div(stat_card("Heckman OOS MAE", "1.2573"), style={"flex": 1}),
            html.Div(stat_card("BP Heteroskedasticity", "p < 0.001", bg=COLORS["clay"], fg="white"), style={"flex": 1}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "18px"}),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1.2fr 0.9fr", "gap": "16px", "marginBottom": "16px"}, children=[
            card([html.H3("Coefficient Comparison", style={"marginTop": 0}), dcc.Graph(figure=fig_coef, config={"displayModeBar": False}), make_table(coef_df, page_size=8)]),
            card([html.H3("VIF Table", style={"marginTop": 0}), dcc.Graph(figure=fig_vif, config={"displayModeBar": False})]),
        ]),
        card([
            html.H3("Full Diagnostics Text", style={"marginTop": 0}),
            html.Pre(DIAGNOSTICS_TEXT or "Diagnostics file not found.", style={"whiteSpace": "pre-wrap", "fontSize": "13px", "margin": 0, "lineHeight": 1.55}),
        ]),
    ])


def render_page(page, filtered, selected_country, top_n, min_p):
    if page == "Overview":
        return overview_layout(filtered)
    if page == "Country Explorer":
        return country_explorer_layout(selected_country)
    if page == "Gap Analysis":
        return gap_analysis_layout(filtered, top_n)
    if page == "Prospect Ranking":
        return prospect_ranking_layout(filtered, min_p)
    if page == "World Map":
        return world_map_layout(filtered)
    return diagnostics_layout()


app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("IDA", style={"fontSize": "44px", "fontWeight": 800, "letterSpacing": "-0.04em", "lineHeight": 1.0}),
                                html.Div("Readiness", style={"fontSize": "20px", "fontWeight": 700, "marginTop": "6px"}),
                            ],
                            style={"marginBottom": "28px"},
                        ),
                        dcc.RadioItems(
                            id="page-radio",
                            options=[{"label": p, "value": p} for p in PAGES],
                            value="Overview",
                            labelStyle={
                                "display": "block",
                                "padding": "12px 14px",
                                "borderRadius": "14px",
                                "marginBottom": "8px",
                                "cursor": "pointer",
                                "color": "rgba(255,255,255,0.92)",
                                "fontWeight": 500,
                            },
                            inputStyle={"marginRight": "10px"},
                            style={"marginBottom": "28px"},
                        ),
                        html.Div(
                            [
                                html.Div("Filters", style={"fontWeight": 700, "fontSize": "18px", "marginBottom": "12px"}),
                                html.Div("Segment", style={"fontSize": "13px", "marginBottom": "6px", "color": "rgba(255,255,255,0.75)"}),
                                dcc.Dropdown(
                                    id="segment-filter",
                                    options=[{"label": s, "value": s} for s in SEG_OPTIONS],
                                    value=SEG_OPTIONS,
                                    multi=True,
                                    style={"color": COLORS["text"], "marginBottom": "12px"},
                                ),
                                html.Div("Income", style={"fontSize": "13px", "marginBottom": "6px", "color": "rgba(255,255,255,0.75)"}),
                                dcc.Dropdown(
                                    id="income-filter",
                                    options=[{"label": s, "value": s} for s in INCOME_OPTIONS],
                                    value=INCOME_OPTIONS,
                                    multi=True,
                                    style={"color": COLORS["text"], "marginBottom": "12px"},
                                ),
                                html.Div("Min GDP (USD bn)", style={"fontSize": "13px", "marginBottom": "10px", "color": "rgba(255,255,255,0.75)"}),
                                dcc.Slider(id="min-gdp-filter", min=0, max=5000, step=50, value=0, tooltip={"placement": "bottom"}),
                            ]
                        ),
                        html.Div(style={"flex": 1}),
                        html.Div(
                            "World Bank IDA dashboard\nHeckman selection model\nCapstone project",
                            style={"fontSize": "12px", "lineHeight": 1.7, "whiteSpace": "pre-wrap", "color": "rgba(255,255,255,0.55)"},
                        ),
                    ],
                    style={
                        "width": "280px",
                        "background": COLORS["navy"],
                        "color": "white",
                        "padding": "28px 20px",
                        "borderRadius": "28px 0 0 28px",
                        "display": "flex",
                        "flexDirection": "column",
                        "minHeight": "860px",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(id="top-title", style={"fontSize": "20px", "fontWeight": 700}),
                                html.Div(
                                    [
                                        dcc.Input(
                                            placeholder="Search countries, segments, metrics",
                                            type="text",
                                            style={
                                                "width": "290px",
                                                "height": "42px",
                                                "borderRadius": "999px",
                                                "border": "none",
                                                "padding": "0 16px",
                                                "background": "#111111",
                                                "color": "white",
                                                "outline": "none",
                                            },
                                        ),
                                    ],
                                    style={"display": "flex", "alignItems": "center", "gap": "12px"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "padding": "18px 22px",
                                "borderBottom": f"1px solid {COLORS['border']}",
                            },
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div("Country Explorer", style={"fontWeight": 700, "fontSize": "13px", "marginBottom": "6px"}),
                                        dcc.Dropdown(
                                            id="country-select",
                                            options=[{"label": c, "value": c} for c in COUNTRY_OPTIONS],
                                            value=COUNTRY_OPTIONS[0] if COUNTRY_OPTIONS else None,
                                            style={"color": COLORS["text"]},
                                        ),
                                    ],
                                    style={"flex": 1},
                                ),
                                html.Div(
                                    [
                                        html.Div("Top N", style={"fontWeight": 700, "fontSize": "13px", "marginBottom": "6px"}),
                                        dcc.Slider(id="top-n-slider", min=10, max=80, step=5, value=30, marks=None, tooltip={"placement": "bottom"}),
                                    ],
                                    style={"flex": 1},
                                ),
                                html.Div(
                                    [
                                        html.Div("Min P(Donate)", style={"fontWeight": 700, "fontSize": "13px", "marginBottom": "6px"}),
                                        dcc.Slider(id="min-p-slider", min=0, max=1, step=0.05, value=0, marks=None, tooltip={"placement": "bottom"}),
                                    ],
                                    style={"flex": 1},
                                ),
                            ],
                            style={"display": "flex", "gap": "20px", "padding": "16px 22px 8px 22px"},
                        ),
                        html.Div(id="page-content", style={"padding": "14px 22px 22px 22px"}),
                    ],
                    style={
                        "flex": 1,
                        "background": COLORS["panel"],
                        "borderRadius": "0 28px 28px 0",
                        "minHeight": "860px",
                    },
                ),
            ],
            style={
                "display": "flex",
                "width": "1300px",
                "margin": "30px auto",
                "background": "transparent",
            },
        ),
    ],
    style={
        "background": "#efefef",
        "minHeight": "100vh",
        "fontFamily": "Inter, Arial, sans-serif",
        "color": COLORS["text"],
        "padding": "20px 0",
    },
)


@app.callback(
    Output("page-content", "children"),
    Output("top-title", "children"),
    Input("page-radio", "value"),
    Input("segment-filter", "value"),
    Input("income-filter", "value"),
    Input("min-gdp-filter", "value"),
    Input("country-select", "value"),
    Input("top-n-slider", "value"),
    Input("min-p-slider", "value"),
)
def update_page(page, segments, incomes, min_gdp, selected_country, top_n, min_p):
    filtered = filter_df(DF, segments, incomes, min_gdp or 0)
    title = page
    return render_page(page, filtered, selected_country, top_n or 30, min_p or 0), title


if __name__ == "__main__":
    app.run(debug=True)
