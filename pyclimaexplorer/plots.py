
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


DARK_BG  = "#ffffff"
PAPER_BG = "#1a1d27"
GRID_CLR = "#2e3347"
TEXT_CLR = "#e0e4f0"

_LAYOUT_BASE = dict(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=DARK_BG,
    font=dict(color=TEXT_CLR, family="Inter, sans-serif", size=13),
    margin=dict(l=10, r=10, t=40, b=10),
)


def heatmap_figure(
    lats: np.ndarray,
    lons: np.ndarray,
    data: np.ndarray,
    title: str,
    colorscale: str,
    unit: str,
    anomaly: bool = False,
) -> go.Figure:
    

   
    step = max(1, len(lats) // 60)
    lats_d = lats[::step]
    lons_d = lons[::step]
    data_d = data[::step, :]
    step2  = max(1, len(lons) // 120)
    lons_d = lons_d[::step2]
    data_d = data_d[:, ::step2]

    LAT, LON = np.meshgrid(lats_d, lons_d, indexing="ij")
    flat_lat  = LAT.ravel()
    flat_lon  = LON.ravel()
    flat_data = data_d.ravel()

   
    if anomaly:
        vmax = float(np.nanpercentile(np.abs(flat_data), 97))
        vmin, vmax = -vmax, vmax
    else:
        vmin = float(np.nanpercentile(flat_data, 2))
        vmax = float(np.nanpercentile(flat_data, 98))

    fig = go.Figure(go.Scattergeo(
        lat=flat_lat,
        lon=flat_lon,
        mode="markers",
        marker=dict(
            size=4,
            color=flat_data,
            colorscale=colorscale,
            cmin=vmin,
            cmax=vmax,
            colorbar=dict(
                title=dict(text=unit, side="right"),
                thickness=14,
                len=0.75,
                x=1.01,
            ),
            showscale=True,
            opacity=0.92,
        ),
        text=[f"Lat {la:.1f}° Lon {lo:.1f}°<br>{v:.2f} {unit}"
              for la, lo, v in zip(flat_lat, flat_lon, flat_data)],
        hoverinfo="text",
    ))

    fig.update_geos(
        showland=True,   landcolor="#1e2333",
        showocean=True,  oceancolor="#151a2b",
        showcoastlines=True, coastlinecolor="#3d4a6b", coastlinewidth=0.6,
        showframe=False,
        projection_type="natural earth",
        bgcolor=DARK_BG,
    )
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, x=0.5, font=dict(size=15)),
        height=430,
        geo=dict(bgcolor=DARK_BG),
    )
    return fig


def time_series_figure(
    ts: pd.Series,
    title: str,
    unit: str,
    add_trend: bool = True,
    highlight_year: int | None = None,
) -> go.Figure:
    

    fig = go.Figure()

    # Main line
    fig.add_trace(go.Scatter(
        x=ts.index,
        y=ts.values,
        mode="lines",
        name=ts.name,
        line=dict(color="#4f8ef7", width=1.5),
        hovertemplate="%{x|%b %Y}<br>%{y:.2f} " + unit + "<extra></extra>",
    ))

    # Rolling 12-month mean
    if len(ts) >= 12:
        rolling = ts.rolling(12, center=True).mean()
        fig.add_trace(go.Scatter(
            x=rolling.index,
            y=rolling.values,
            mode="lines",
            name="12-month mean",
            line=dict(color="#f7a44f", width=2.5),
            hovertemplate="%{x|%b %Y}<br>Avg: %{y:.2f} " + unit + "<extra></extra>",
        ))

    # Linear trend
    if add_trend and len(ts) >= 24:
        x_num = np.arange(len(ts))
        mask  = ~np.isnan(ts.values)
        if mask.sum() > 2:
            coef = np.polyfit(x_num[mask], ts.values[mask], 1)
            trend = np.polyval(coef, x_num)
            trend_per_decade = coef[0] * 120   # monthly steps → /decade
            fig.add_trace(go.Scatter(
                x=ts.index,
                y=trend,
                mode="lines",
                name=f"Trend ({trend_per_decade:+.2f} {unit}/decade)",
                line=dict(color="#e85757", width=2, dash="dash"),
                hoverinfo="skip",
            ))

    # Vertical highlight for selected year
    if highlight_year:
        yr_ts = ts[ts.index.year == highlight_year]
        if not yr_ts.empty:
            fig.add_vrect(
                x0=yr_ts.index[0],
                x1=yr_ts.index[-1],
                fillcolor="#4f8ef7",
                opacity=0.08,
                layer="below",
                line_width=0,
            )

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, x=0.5, font=dict(size=15)),
        xaxis=dict(
            title="Date",
            showgrid=True, gridcolor=GRID_CLR, gridwidth=0.5,
            zeroline=False,
        ),
        yaxis=dict(
            title=unit,
            showgrid=True, gridcolor=GRID_CLR, gridwidth=0.5,
            zeroline=False,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor=GRID_CLR, borderwidth=1,
            x=0.01, y=0.99, xanchor="left", yanchor="top",
        ),
        height=340,
    )
    return fig


def comparison_figure(
    ts_a: pd.Series,
    ts_b: pd.Series,
    label_a: str,
    label_b: str,
    unit: str,
) -> go.Figure:
    

    fig = go.Figure()

    for ts, color, name in [
        (ts_a, "#4f8ef7", label_a),
        (ts_b, "#f74f8e", label_b),
    ]:
        fig.add_trace(go.Scatter(
            x=ts.index,
            y=ts.values,
            mode="lines",
            name=name,
            line=dict(color=color, width=2),
            hovertemplate=f"{name}<br>%{{x|%b %Y}}: %{{y:.2f}} {unit}<extra></extra>",
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=f"Comparison: {label_a} vs {label_b}", x=0.5, font=dict(size=15)),
        xaxis=dict(showgrid=True, gridcolor=GRID_CLR),
        yaxis=dict(title=unit, showgrid=True, gridcolor=GRID_CLR),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor=GRID_CLR, borderwidth=1),
        height=340,
    )
    return fig


def side_by_side_heatmaps(
    lats: np.ndarray, lons: np.ndarray,
    data_a: np.ndarray, data_b: np.ndarray,
    title_a: str, title_b: str,
    colorscale: str, unit: str,
) -> go.Figure:
    

    step  = max(1, len(lats) // 50)
    step2 = max(1, len(lons) // 100)
    lats_d = lats[::step]
    lons_d = lons[::step2]
    da     = data_a[::step, ::step2]
    db     = data_b[::step, ::step2]

    LAT, LON = np.meshgrid(lats_d, lons_d, indexing="ij")
    flat_lat = LAT.ravel()
    flat_lon = LON.ravel()

    all_vals = np.concatenate([da.ravel(), db.ravel()])
    vmin = float(np.nanpercentile(all_vals, 2))
    vmax = float(np.nanpercentile(all_vals, 98))

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scattergeo"}, {"type": "scattergeo"}]],
        subplot_titles=[title_a, title_b],
        horizontal_spacing=0.02,
    )

    for col, flat_data, show_bar in [(1, da.ravel(), False), (2, db.ravel(), True)]:
        fig.add_trace(go.Scattergeo(
            lat=flat_lat,
            lon=flat_lon,
            mode="markers",
            marker=dict(
                size=3,
                color=flat_data,
                colorscale=colorscale,
                cmin=vmin, cmax=vmax,
                colorbar=dict(title=unit, thickness=12, x=1.02) if show_bar else None,
                showscale=show_bar,
                opacity=0.9,
            ),
            hovertemplate=f"Lat %{{lat:.1f}} Lon %{{lon:.1f}}<br>%{{marker.color:.2f}} {unit}<extra></extra>",
        ), row=1, col=col)

    for geo in ["geo", "geo2"]:
        fig.update_layout(**{
            geo: dict(
                showland=True,   landcolor="#1e2333",
                showocean=True,  oceancolor="#151a2b",
                showcoastlines=True, coastlinecolor="#3d4a6b", coastlinewidth=0.6,
                showframe=False,
                projection_type="natural earth",
                bgcolor=DARK_BG,
            )
        })

    fig.update_layout(
        **_LAYOUT_BASE,
        height=380,
        title=dict(text="Side-by-Side Comparison", x=0.5, font=dict(size=15)),
    )
    return fig


def anomaly_distribution_figure(ts: pd.Series, unit: str, title: str) -> go.Figure:
    
    vals = ts.dropna().values

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=vals,
        nbinsx=40,
        name="Distribution",
        marker_color="#4f8ef7",
        opacity=0.75,
        hovertemplate="%{x:.2f} " + unit + "<br>Count: %{y}<extra></extra>",
    ))

    # Zero line for anomaly context
    if ts.name and "anomaly" in str(ts.name).lower():
        fig.add_vline(x=0, line=dict(color="#e85757", width=1.5, dash="dash"))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, x=0.5, font=dict(size=15)),
        xaxis=dict(title=unit, showgrid=True, gridcolor=GRID_CLR),
        yaxis=dict(title="Count", showgrid=True, gridcolor=GRID_CLR),
        height=280,
    )
    return fig
