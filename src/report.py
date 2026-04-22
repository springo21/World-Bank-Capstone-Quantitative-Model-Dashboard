"""
Reporting and chart generation for the Donor Readiness Index.

Ranks countries by capacity gap and produces:
  - outputs/dri_output.csv            — ranked per-country summary
  - outputs/charts/chart1_gap_ranking.png
  - outputs/charts/chart2_giving_rate.png
  - outputs/charts/chart3_capacity_vs_giving_rate.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
CHARTS = OUTPUTS / "charts"

CAPACITY_SCORES_PATH = DATA_PROCESSED / "capacity_scores.csv"
DRI_OUTPUT_PATH = OUTPUTS / "dri_output.csv"

DPI = 150
INCOME_COLORS = {
    "HIC": "#2196F3",   # blue
    "UMC": "#FF9800",   # orange
    "LMC": "#4CAF50",   # green
    "LIC": "#9C27B0",   # purple
}
DEFAULT_COLOR = "#607D8B"


def _ensure_output_dirs():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Merge and rank
# ---------------------------------------------------------------------------

def build_dri_output(capacity: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Sort capacity scores by gap_usd descending.
    Writes outputs/dri_output.csv and returns the DataFrame.
    """
    _ensure_output_dirs()

    if capacity is None:
        capacity = pd.read_csv(CAPACITY_SCORES_PATH)

    # Sort by gap descending (largest gap = most underperforming)
    merged = capacity.sort_values("gap_usd", ascending=False, na_position="last")
    merged = merged.reset_index(drop=True)

    merged.to_csv(DRI_OUTPUT_PATH, index=False)
    logger.info("DRI output written to %s (%d countries)", DRI_OUTPUT_PATH, len(merged))
    return merged


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _income_color(income_group: str | None) -> str:
    return INCOME_COLORS.get(str(income_group), DEFAULT_COLOR)


def _millions(x, _):
    return f"${x/1e6:.0f}M"


def _billions(x, _):
    if abs(x) >= 1e9:
        return f"${x/1e9:.1f}B"
    return f"${x/1e6:.0f}M"


# ---------------------------------------------------------------------------
# Chart 1: Gap ranking bar chart
# ---------------------------------------------------------------------------

def chart1_gap_ranking(dri: pd.DataFrame, top_n: int = 30) -> None:
    """Horizontal bar chart — top N countries by gap_usd."""
    valid = dri[dri["gap_usd"].notna()].copy()
    if len(valid) < top_n:
        logger.info("Chart 1: only %d countries with valid gap (requested %d)", len(valid), top_n)
    plot_df = valid.head(top_n).sort_values("gap_usd", ascending=True)

    colors = [_income_color(g) for g in plot_df["income_group"]]
    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.35)))
    bars = ax.barh(plot_df["iso3"], plot_df["gap_usd"], color=colors, edgecolor="white", linewidth=0.5)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_billions))
    ax.set_xlabel("Contribution Gap (Target − Actual)", fontsize=11)
    ax.set_title(f"Donor Readiness Index — Top {len(plot_df)} Countries by IDA Contribution Gap", fontsize=13, pad=12)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.tick_params(axis="y", labelsize=9)

    # Income group legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=label)
        for label, color in INCOME_COLORS.items()
        if label in plot_df["income_group"].values
    ]
    if handles:
        ax.legend(handles=handles, title="Income Group", loc="lower right", fontsize=8)

    plt.tight_layout()
    path = CHARTS / "chart1_gap_ranking.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart 1 saved to %s", path)


# ---------------------------------------------------------------------------
# Chart 2: Giving rate bar chart
# ---------------------------------------------------------------------------

def chart2_giving_rate(dri: pd.DataFrame) -> None:
    """Horizontal bar chart of giving rate for all countries, sorted ascending."""
    valid = dri[dri["giving_rate"].notna()].sort_values("giving_rate", ascending=True).copy()

    colors = ["#4CAF50" if r >= 1.0 else "#F44336" for r in valid["giving_rate"]]
    fig, ax = plt.subplots(figsize=(10, max(6, len(valid) * 0.28)))
    ax.barh(valid["iso3"], valid["giving_rate"], color=colors, edgecolor="white", linewidth=0.3)
    ax.axvline(1.0, color="black", linewidth=1.2, linestyle="--", label="Benchmark (1.0)")
    ax.set_xlabel("Giving Rate (Actual / Adjusted Target)", fontsize=11)
    ax.set_title("Donor Readiness Index — Giving Rate by Country", fontsize=13, pad=12)
    ax.tick_params(axis="y", labelsize=8)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", label="At or above benchmark"),
        Patch(facecolor="#F44336", label="Below benchmark"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    path = CHARTS / "chart2_giving_rate.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart 2 saved to %s", path)


# ---------------------------------------------------------------------------
# Chart 3: Capacity vs. giving rate scatter
# ---------------------------------------------------------------------------

def chart3_capacity_vs_giving_rate(dri: pd.DataFrame) -> None:
    """Scatter: adjusted_target_usd (x) vs. giving_rate (y), ISO3 labels."""
    valid = dri[dri["adjusted_target_usd"].notna() & dri["giving_rate"].notna()].copy()

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = [_income_color(g) for g in valid["income_group"]]
    ax.scatter(valid["adjusted_target_usd"], valid["giving_rate"], c=colors, s=60, alpha=0.75, zorder=3)

    for _, row in valid.iterrows():
        ax.annotate(
            row["iso3"],
            (row["adjusted_target_usd"], row["giving_rate"]),
            fontsize=7, alpha=0.8, xytext=(3, 3), textcoords="offset points",
        )

    # Reference lines
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", label="Giving rate = 1.0")
    median_cap = valid["adjusted_target_usd"].median()
    ax.axvline(median_cap, color="gray", linewidth=0.8, linestyle=":", label="Median capacity target")

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_billions))
    ax.set_xlabel("Adjusted Capacity Target (USD)", fontsize=11)
    ax.set_ylabel("Giving Rate (Actual / Target)", fontsize=11)
    ax.set_title("Capacity vs. Giving Rate", fontsize=13, pad=12)
    ax.legend(fontsize=9)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=label)
        for label, color in INCOME_COLORS.items()
        if label in valid["income_group"].values
    ]
    if handles:
        ax.legend(handles=handles, title="Income Group", loc="upper right", fontsize=8)

    plt.tight_layout()
    path = CHARTS / "chart3_capacity_vs_giving_rate.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart 3 saved to %s", path)


# ---------------------------------------------------------------------------
# Chart 5: All countries by gap (full ranked list)
# ---------------------------------------------------------------------------

def chart5_all_countries_gap(dri: pd.DataFrame) -> None:
    """Horizontal bar chart — all countries ranked by gap_usd."""
    valid = dri[dri["gap_usd"].notna()].sort_values("gap_usd", ascending=True).copy()

    colors = ["#E53935" if g < 0 else _income_color(ig)
              for g, ig in zip(valid["gap_usd"], valid["income_group"])]

    fig, ax = plt.subplots(figsize=(12, max(8, len(valid) * 0.22)))
    ax.barh(valid["country_name"], valid["gap_usd"], color=colors, edgecolor="none", height=0.8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_billions))
    ax.set_xlabel("Contribution Gap (Target − Actual)", fontsize=11)
    ax.set_title("Donor Readiness Index — All Countries by IDA Contribution Gap", fontsize=13, pad=12)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.tick_params(axis="y", labelsize=7)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=color, label=label)
        for label, color in INCOME_COLORS.items()
        if label in valid["income_group"].values
    ] + [Patch(facecolor="#E53935", label="Over-contributor")]
    ax.legend(handles=handles, title="Income Group", loc="lower right", fontsize=8)

    plt.tight_layout()
    path = CHARTS / "chart5_all_countries_gap.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart 5 saved to %s", path)


# ---------------------------------------------------------------------------
# World map: interactive choropleth
# ---------------------------------------------------------------------------

def _load_country_interior_points() -> dict[str, tuple[float, float]]:
    """
    Return {iso3: (lat, lon)} using shapely representative_point() — guaranteed inside polygon.

    Downloads Natural Earth 110m countries on first call and caches to data/cache/.
    """
    import json
    import geopandas as gpd

    cache_path = ROOT / "data" / "cache" / "country_interior_points.json"
    if cache_path.exists():
        with open(cache_path) as f:
            raw = json.load(f)
        return {k: tuple(v) for k, v in raw.items()}

    logger.info("Downloading Natural Earth 110m countries for label placement (cached after this run)...")
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)

    points: dict[str, tuple[float, float]] = {}
    for _, row in world.iterrows():
        iso3 = row.get("ADM0_A3") or row.get("ISO_A3")
        if iso3 and iso3 != "-99":
            pt = row.geometry.representative_point()
            points[iso3] = (pt.y, pt.x)  # (lat, lon)

    with open(cache_path, "w") as f:
        json.dump(points, f)
    logger.info("Interior points cached to %s (%d countries)", cache_path, len(points))
    return points


def generate_world_map(dri: pd.DataFrame) -> None:
    """Interactive choropleth HTML map — gap_usd shaded dark red (under-contributor) to dark blue (over-contributor).

    Uses a symlog colour scale (sign(x) * log1p(|x| / 1e6)) so that the gradient
    is spread across orders of magnitude rather than compressed by a handful of
    large-gap outliers. Country labels are shown only for the top 30 countries
    by absolute gap to avoid overlap on small nations.

    Colour convention:
        Dark red   = large positive gap = giving well below capacity (under-contributor)
        White/cream = close to benchmark
        Dark blue  = negative gap = giving above capacity (over-contributor)
    """
    import plotly.graph_objects as go

    valid = dri[dri["gap_usd"].notna()].copy()

    def _fmt_usd(v):
        if pd.isna(v):
            return "N/A"
        if abs(v) >= 1e9:
            return f"${v/1e9:.2f}B"
        return f"${v/1e6:.2f}M"

    def _fmt_pct(v):
        return "N/A" if pd.isna(v) else f"{v:.1%}"

    valid["_gap_fmt"] = valid["gap_usd"].apply(_fmt_usd)
    valid["_rate_fmt"] = valid["giving_rate"].apply(_fmt_pct)
    valid["_target_fmt"] = valid["adjusted_target_usd"].apply(_fmt_usd)

    # Symlog transform: spread gradient across orders of magnitude.
    # Units: millions of USD so that log1p(1) ≈ $1M and log1p(1000) ≈ $1B.
    # Positive gap (under-contributor) → higher _z value → red end of scale
    # Negative gap (over-contributor)  → lower  _z value → blue end of scale
    def _symlog(x):
        return np.sign(x) * np.log1p(np.abs(x) / 1e6)

    valid["_z"] = valid["gap_usd"].apply(_symlog)

    # Use symmetric bounds around zero so the diverging scale is visually balanced
    z_abs = float(np.nanquantile(np.abs(valid["_z"]), 0.98))
    if not np.isfinite(z_abs) or z_abs == 0:
        z_abs = float(np.nanmax(np.abs(valid["_z"]))) if len(valid) else 1.0
    if not np.isfinite(z_abs) or z_abs == 0:
        z_abs = 1.0

    zmin = -z_abs
    zmax = z_abs

    # Colorbar tick positions (symlog scale) with human-readable dollar labels
    tick_dollars = [-1e9, -1e8, -1e7, -1e6, 0, 1e6, 1e7, 1e8, 1e9]
    tick_vals = [_symlog(v) for v in tick_dollars]
    tick_text = ["-$1B", "-$100M", "-$10M", "-$1M", "$0", "$1M", "$10M", "$100M", "$1B"]

    # Keep only ticks that fall within the plotted scale range
    tick_pairs = [(v, t) for v, t in zip(tick_vals, tick_text) if zmin <= v <= zmax]
    if not tick_pairs:
        tick_pairs = [(0, "$0")]
    tick_vals, tick_text = zip(*tick_pairs)
    tick_vals = list(tick_vals)
    tick_text = list(tick_text)

    # Custom diverging colorscale: dark blue (over-contributor, negative gap)
    # through white/cream at zero, to dark red (under-contributor, positive gap).
    rdb_colorscale = [
        [0.0,  "#1a3a6b"],   # dark blue  — over-contributor (negative gap)
        [0.15, "#2e6db4"],   # mid blue
        [0.35, "#9ec8e0"],   # light blue
        [0.5,  "#f7f4f0"],   # off-white  — at benchmark (gap ≈ 0)
        [0.65, "#e8967a"],   # light red
        [0.85, "#c0392b"],   # mid red
        [1.0,  "#8b1a1a"],   # dark red   — under-contributor (positive gap)
    ]

    fig = go.Figure(go.Choropleth(
        locations=valid["iso3"],
        z=valid["_z"],
        locationmode="ISO-3",
        colorscale=rdb_colorscale,
        zmin=zmin,
        zmax=zmax,
        zmid=0,
        customdata=valid[["country_name", "_gap_fmt", "_rate_fmt", "_target_fmt"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Gap: %{customdata[1]}<br>"
            "Giving Rate: %{customdata[2]}<br>"
            "Capacity Target: %{customdata[3]}"
            "<extra></extra>"
        ),
        colorbar=dict(
            title=dict(
                text="Contribution Gap",
                side="right",
                font=dict(size=13),
            ),
            tickvals=tick_vals,
            ticktext=tick_text,
            tickfont=dict(size=10),
            len=0.75,
            thickness=18,
            outlinewidth=0,
            x=1.02,
            y=0.50,
            yanchor="middle",
        ),
        marker_line_color="white",
        marker_line_width=0.5,
    ))

    # Labels: top countries by absolute gap to avoid overlap on small nations
    interior = _load_country_interior_points()
    label_rows = (
        valid.assign(_abs_gap=valid["gap_usd"].abs())
        .nlargest(30, "_abs_gap")
        .copy()
    )
    label_rows = label_rows[label_rows["iso3"].isin(interior)].copy()
    label_rows["_lat"] = label_rows["iso3"].map(lambda c: interior[c][0])
    label_rows["_lon"] = label_rows["iso3"].map(lambda c: interior[c][1])

    fig.add_trace(go.Scattergeo(
        lat=label_rows["_lat"],
        lon=label_rows["_lon"],
        text=label_rows["country_name"],
        mode="text",
        textfont=dict(size=8, color="black"),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(
            text="Donor Readiness Index — IDA Contribution Gap by Country",
            font=dict(size=16)
        ),
        geo=dict(
            showland=True,
            landcolor="lightgray",
            showframe=False,
            showcoastlines=True,
            coastlinecolor="white",
            projection_type="natural earth",
        ),
        margin=dict(l=0, r=110, t=50, b=0),
        annotations=[
            dict(
                x=1.10,  # ⬅️ move further right (more padding)
                y=0.875,  # ⬅️ align with top of colorbar (len=0.75 → spans ~0.125–0.875)
                xref="paper",
                yref="paper",
                text="▲ Under-contributor",
                showarrow=False,
                font=dict(size=11, color="#8b1a1a"),
                align="left",
                yanchor="middle"  # ⬅️ center text vertically at this point
            ),
            dict(
                x=1.10,
                y=0.125,  # ⬅️ align with bottom of colorbar
                xref="paper",
                yref="paper",
                text="▼ Over-contributor",
                showarrow=False,
                font=dict(size=11, color="#1a3a6b"),
                align="left",
                yanchor="middle"
            ),
        ],
    )

    # Warn on unmatched ISO-3 codes
    for iso3 in valid["iso3"]:
        if iso3 not in interior:
            logger.warning("ISO-3 code '%s' not found in Natural Earth geometry — label omitted", iso3)

    path = CHARTS / "chart5_world_map.html"
    fig.write_html(str(path), include_plotlyjs=True)
    logger.info("World map saved to %s", path)


# ---------------------------------------------------------------------------
# Main report function
# ---------------------------------------------------------------------------

def generate_report(
    capacity: pd.DataFrame | None = None,
    top_n: int = 30,
    stage1_result=None,
    stage2_result=None,
    train=None,
    round_cols=None,
) -> pd.DataFrame:
    """
    Build the DRI output CSV and generate all charts.

    Optional stage1_result, stage2_result, train, round_cols: if provided
    (Heckman path), the regression table is regenerated here too. If the
    Heckman scorer already called generate_regression_table(), this is a no-op
    because the files already exist — but passing them in is harmless.

    Returns the DRI DataFrame.
    """
    _ensure_output_dirs()
    dri = build_dri_output(capacity)
    chart1_gap_ranking(dri, top_n=top_n)
    chart2_giving_rate(dri)
    chart3_capacity_vs_giving_rate(dri)
    chart5_all_countries_gap(dri)
    generate_world_map(dri)

    # Regression table — generated by heckman.score_capacity() when the
    # Heckman path is used. If stage results are passed in explicitly here
    # (e.g. from a custom notebook), generate them now.
    if all(x is not None for x in [stage1_result, stage2_result, train, round_cols]):
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent))
            from heckman import generate_regression_table
            generate_regression_table(
                stage1_result=stage1_result,
                stage2_result=stage2_result,
                train=train,
                round_cols=round_cols,
            )
        except Exception as exc:
            logger.warning("Could not generate regression table: %s", exc)

    logger.info("All charts generated in %s", CHARTS)
    return dri