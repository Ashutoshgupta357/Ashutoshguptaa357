
import os, time
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="PyClimaExplorer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark theme overrides */
.stApp { background: #0e1117; color: #e0e4f0; }
.stSidebar { background: #13161f !important; }
.stSidebar .css-1d391kg { padding: 1rem 0.75rem; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: #1a1d27;
    border: 1px solid #2e3347;
    border-radius: 10px;
    padding: 14px 18px;
}

/* Section headers */
.section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #7b9ef7;
    border-bottom: 1px solid #2e3347;
    padding-bottom: 6px;
    margin-bottom: 12px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Story card */
.story-card {
    background: #1a1d27;
    border-left: 3px solid #4f8ef7;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 0.9rem;
    color: #b0b8d0;
    margin-bottom: 16px;
    line-height: 1.6;
}

/* Stat badge */
.stat-badge {
    display: inline-block;
    background: #1e3058;
    color: #7eb3f7;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.82rem;
    font-weight: 500;
    margin: 2px;
}

/* Upload zone */
.upload-hint {
    color: #6b7280;
    font-size: 0.85rem;
    text-align: center;
    padding: 8px;
}

/* Hide Streamlit menu & footer */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

from climate_data import (
    load_dataset, get_variables, get_var_meta, get_time_index,
    get_spatial_slice, get_time_series, get_global_mean_series,
    VARIABLE_META, STORY_PRESETS,
)
from plots import (
    heatmap_figure, time_series_figure, comparison_figure,
    side_by_side_heatmaps, anomaly_distribution_figure,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌍 PyClimaExplorer")
    st.markdown("<div style='color:#6b7280;font-size:0.8rem'>Technex '26 · Hack-it-out</div>",
                unsafe_allow_html=True)
    st.divider()

    # ── Dataset selection ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Dataset</div>', unsafe_allow_html=True)

    demo_path = os.path.join(os.path.dirname(__file__), "data", "climate_demo.nc")
    use_demo  = os.path.exists(demo_path)

    data_source = st.radio(
        "Source",
        options=["Demo dataset", "Upload your own .nc file"],
        index=0,
        label_visibility="collapsed",
    )

    dataset_path = None
    if data_source == "Demo dataset":
        if use_demo:
            dataset_path = demo_path
            st.success("✓ ERA5-style demo loaded", icon="✅")
        else:
            st.warning("Demo file not found.\nRun: `python generate_data.py`")
    else:
        uploaded = st.file_uploader(
            "Upload NetCDF file",
            type=["nc", "nc4", "netcdf"],
            label_visibility="collapsed",
        )
        if uploaded:
            tmp = f"/tmp/{uploaded.name}"
            with open(tmp, "wb") as f:
                f.write(uploaded.read())
            dataset_path = tmp
            st.success(f"✓ Loaded: {uploaded.name}", icon="✅")
        else:
            st.markdown('<div class="upload-hint">↑ Drop a .nc file here</div>',
                        unsafe_allow_html=True)

    st.divider()

    # ── Mode ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Mode</div>', unsafe_allow_html=True)
    app_mode = st.selectbox(
        "Mode",
        ["🔭 Explorer", "📖 Story Mode", "⚖️ Comparison Mode"],
        label_visibility="collapsed",
    )

    st.divider()

    # ── Controls (shown only when data loaded) ────────────────────────────────
    if dataset_path:
        ds    = load_dataset(dataset_path)
        times = get_time_index(ds)
        years = sorted(set(times.year))
        months_map = {
            1:"January", 2:"February", 3:"March", 4:"April",
            5:"May", 6:"June", 7:"July", 8:"August",
            9:"September", 10:"October", 11:"November", 12:"December",
        }

        avail_vars = get_variables(ds)
        # Prefer known variables first
        known_order = list(VARIABLE_META.keys())
        avail_sorted = sorted(avail_vars, key=lambda v: (known_order.index(v) if v in known_order else 99))

        var_options = {get_var_meta(v)["icon"] + " " + get_var_meta(v)["label"]: v for v in avail_sorted}

        st.markdown('<div class="section-header">Variable</div>', unsafe_allow_html=True)
        var_display = st.selectbox("Variable", list(var_options.keys()), label_visibility="collapsed")
        selected_var = var_options[var_display]
        meta = get_var_meta(selected_var)

        # Story mode preset picker
        if app_mode == "📖 Story Mode":
            st.divider()
            st.markdown('<div class="section-header">Story</div>', unsafe_allow_html=True)
            story_key = st.selectbox("Choose story", list(STORY_PRESETS.keys()), label_visibility="collapsed")
            story = STORY_PRESETS[story_key]
            # Override var, year, month from story
            selected_var = story["variable"] if story["variable"] in avail_sorted else selected_var
            meta         = get_var_meta(selected_var)
            sel_year  = story["year"] if story["year"] in years else years[-1]
            sel_month = story["month"]
            sel_lat   = story["lat"]
            sel_lon   = story["lon"]
            anomaly   = story["anomaly"]
        else:
            st.markdown('<div class="section-header">Time Slice</div>', unsafe_allow_html=True)
            sel_year  = st.select_slider("Year",  options=years, value=years[-1])
            sel_month = st.select_slider("Month", options=list(months_map.keys()),
                                         format_func=lambda m: months_map[m], value=7)
            st.divider()
            st.markdown('<div class="section-header">Location</div>', unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                sel_lat = st.number_input("Latitude °N",  min_value=float(ds.latitude.min()),
                                          max_value=float(ds.latitude.max()), value=28.6, step=2.5)
            with col_b:
                sel_lon = st.number_input("Longitude °E", min_value=float(ds.longitude.min()),
                                          max_value=float(ds.longitude.max()), value=77.2, step=2.5)
            st.divider()
            st.markdown('<div class="section-header">Options</div>', unsafe_allow_html=True)
            anomaly = st.checkbox("Show anomaly (vs climatology)", value=False)

        # Comparison mode extras
        if app_mode == "⚖️ Comparison Mode":
            st.divider()
            st.markdown('<div class="section-header">Compare With</div>', unsafe_allow_html=True)
            cmp_year_a = st.select_slider("Period A", options=years, value=years[0])
            cmp_year_b = st.select_slider("Period B", options=years, value=years[-1])
            cmp_lat_b  = st.number_input("Lat B °N", min_value=float(ds.latitude.min()),
                                          max_value=float(ds.latitude.max()),
                                          value=sel_lat + 10.0, step=2.5)
            cmp_lon_b  = st.number_input("Lon B °E", min_value=float(ds.longitude.min()),
                                          max_value=float(ds.longitude.max()),
                                          value=sel_lon + 0.0, step=2.5)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════
if not dataset_path:
    # ── Landing screen ────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center;padding:80px 20px 40px;'>
        <div style='font-size:4rem'>🌍</div>
        <h1 style='font-size:2.4rem;font-weight:700;color:#e0e4f0;margin:12px 0 6px'>
            PyClimaExplorer
        </h1>
        <p style='color:#6b7280;font-size:1.1rem;max-width:520px;margin:0 auto 32px'>
            Interactive climate data visualiser for NetCDF datasets (ERA5, CESM, and more).
            Load the demo dataset or upload your own <code>.nc</code> file to explore.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "🗺️", "Global Heatmap",   "Spatial distribution of any variable at any time step"),
        (c2, "📈", "Time Series",       "Track how climate changes at a specific location over time"),
        (c3, "⚖️", "Comparison Mode",   "Side-by-side comparison of two years or two locations"),
    ]:
        with col:
            st.markdown(f"""
            <div style='background:#1a1d27;border:1px solid #2e3347;border-radius:12px;padding:20px;text-align:center'>
                <div style='font-size:2rem'>{icon}</div>
                <div style='font-weight:600;margin:8px 0 4px;color:#e0e4f0'>{title}</div>
                <div style='color:#6b7280;font-size:0.85rem;line-height:1.5'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("👈  Select **Demo dataset** in the sidebar to get started instantly, "
            "or run `python generate_data.py` to generate the demo file first.")

else:
    # ── Header ────────────────────────────────────────────────────────────────
    title_anom = " (Anomaly)" if anomaly else ""
    month_name = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                  7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

    st.markdown(
        f"<h2 style='margin-bottom:4px;color:#e0e4f0'>"
        f"{meta['icon']} {meta['label']}{title_anom} — "
        f"{month_name[sel_month]} {sel_year}</h2>",
        unsafe_allow_html=True,
    )

    # ── KPI metrics ───────────────────────────────────────────────────────────
    # Find time index
    time_mask = [(t.year == sel_year and t.month == sel_month) for t in times]
    time_idx  = next((i for i, m in enumerate(time_mask) if m), 0)

    lats_arr, lons_arr, spatial = get_spatial_slice(ds, selected_var, time_idx, anomaly=anomaly)
    ts_local = get_time_series(ds, selected_var, sel_lat, sel_lon, anomaly=anomaly)

    gmin  = float(np.nanmin(spatial))
    gmax  = float(np.nanmax(spatial))
    gmean = float(np.nanmean(spatial))
    local_val = ts_local.iloc[time_idx] if time_idx < len(ts_local) else float("nan")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Global Min",  f"{gmin:.1f} {meta['unit']}")
    m2.metric("Global Mean", f"{gmean:.1f} {meta['unit']}")
    m3.metric("Global Max",  f"{gmax:.1f} {meta['unit']}")
    m4.metric(f"Local ({sel_lat:.0f}°N, {sel_lon:.0f}°E)", f"{local_val:.1f} {meta['unit']}")

    st.markdown("")

    # ══════════════════════════════════════════════════════════════════════════
    if app_mode == "📖 Story Mode":
        st.markdown(f'<div class="story-card">📖 <strong>{story_key}</strong><br>{story["desc"]}</div>',
                    unsafe_allow_html=True)

    # ── Heatmap ───────────────────────────────────────────────────────────────
    heatmap_title = (
        f"{meta['label']} {'Anomaly ' if anomaly else ''}— "
        f"{month_name[sel_month]} {sel_year}"
    )
    col_map, col_ts = st.columns([3, 2])

    with col_map:
        with st.spinner("Rendering heatmap…"):
            fig_map = heatmap_figure(
                lats_arr, lons_arr, spatial,
                title=heatmap_title,
                colorscale=meta["colorscale"],
                unit=meta["unit"],
                anomaly=anomaly,
            )
        st.plotly_chart(fig_map, use_container_width=True)

        # Location marker overlay
        st.markdown(
            f"<div style='text-align:center;color:#6b7280;font-size:0.82rem;margin-top:-8px'>"
            f"📍 Selected location: <strong>{sel_lat:.1f}°N, {sel_lon:.1f}°E</strong>"
            f"</div>", unsafe_allow_html=True
        )

    # ── Time series ───────────────────────────────────────────────────────────
    with col_ts:
        ts_title = (
            f"{meta['label']} {'Anomaly ' if anomaly else ''}at "
            f"{sel_lat:.1f}°N, {sel_lon:.1f}°E"
        )
        with st.spinner("Rendering time series…"):
            fig_ts = time_series_figure(
                ts_local, title=ts_title,
                unit=meta["unit"],
                add_trend=True,
                highlight_year=sel_year,
            )
        st.plotly_chart(fig_ts, use_container_width=True)

        # Distribution histogram
        fig_dist = anomaly_distribution_figure(
            ts_local,
            unit=meta["unit"],
            title=f"Value distribution at selected point",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    if app_mode == "⚖️ Comparison Mode":
        st.markdown('<div class="section-header">⚖️ Comparison Mode</div>',
                    unsafe_allow_html=True)
        sub1, sub2 = st.columns(2)

        # Side-by-side heatmaps
        mask_a = [(t.year == cmp_year_a and t.month == sel_month) for t in times]
        mask_b = [(t.year == cmp_year_b and t.month == sel_month) for t in times]
        idx_a  = next((i for i, m in enumerate(mask_a) if m), 0)
        idx_b  = next((i for i, m in enumerate(mask_b) if m), 0)

        _, _, spatial_a = get_spatial_slice(ds, selected_var, idx_a, anomaly=anomaly)
        _, _, spatial_b = get_spatial_slice(ds, selected_var, idx_b, anomaly=anomaly)

        with st.spinner("Rendering comparison heatmaps…"):
            fig_cmp_map = side_by_side_heatmaps(
                lats_arr, lons_arr,
                spatial_a, spatial_b,
                title_a=f"{month_name[sel_month]} {cmp_year_a}",
                title_b=f"{month_name[sel_month]} {cmp_year_b}",
                colorscale=meta["colorscale"],
                unit=meta["unit"],
            )
        st.plotly_chart(fig_cmp_map, use_container_width=True)

        # Time-series comparison: location A vs location B
        ts_a = get_time_series(ds, selected_var, sel_lat,   sel_lon,   anomaly=anomaly)
        ts_b = get_time_series(ds, selected_var, cmp_lat_b, cmp_lon_b, anomaly=anomaly)

        fig_cmp_ts = comparison_figure(
            ts_a, ts_b,
            label_a=f"{sel_lat:.1f}°N, {sel_lon:.1f}°E",
            label_b=f"{cmp_lat_b:.1f}°N, {cmp_lon_b:.1f}°E",
            unit=meta["unit"],
        )
        st.plotly_chart(fig_cmp_ts, use_container_width=True)

        # Difference heatmap
        diff = spatial_b - spatial_a
        with st.spinner("Rendering difference map…"):
            fig_diff = heatmap_figure(
                lats_arr, lons_arr, diff,
                title=f"Difference: {cmp_year_b} − {cmp_year_a}  ({month_name[sel_month]})",
                colorscale="RdBu_r",
                unit=meta["unit"],
                anomaly=True,
            )
        st.plotly_chart(fig_diff, use_container_width=True)

    # ── Global mean trend ─────────────────────────────────────────────────────
    if app_mode != "⚖️ Comparison Mode":
        with st.expander("📈 Global mean time series", expanded=False):
            with st.spinner("Computing global mean…"):
                gm_ts  = get_global_mean_series(ds, selected_var)
                fig_gm = time_series_figure(
                    gm_ts,
                    title=f"Latitude-weighted global mean {meta['label']}",
                    unit=meta["unit"],
                    add_trend=True,
                    highlight_year=sel_year,
                )
            st.plotly_chart(fig_gm, use_container_width=True)
            gm_trend_text = ""
            vals = gm_ts.values
            x = np.arange(len(vals))
            mask = ~np.isnan(vals)
            if mask.sum() > 2:
                coef = np.polyfit(x[mask], vals[mask], 1)
                rate = coef[0] * 120
                gm_trend_text = (
                    f"Linear trend: **{rate:+.3f} {meta['unit']} per decade**  "
                    f"over {len(years)} years"
                )
            if gm_trend_text:
                st.markdown(gm_trend_text)

    # ── Dataset info ──────────────────────────────────────────────────────────
    with st.expander("ℹ️ Dataset info", expanded=False):
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.markdown("**Dimensions**")
            for dim, size in ds.dims.items():
                st.markdown(
                    f'<span class="stat-badge">{dim}: {size}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown("")
            st.markdown("**Variables**")
            for v in avail_vars:
                vm = get_var_meta(v)
                st.markdown(
                    f'<span class="stat-badge">{vm["icon"]} {v} ({vm["unit_raw"]})</span>',
                    unsafe_allow_html=True,
                )
        with info_col2:
            st.markdown("**Time range**")
            st.markdown(f"From `{times[0].strftime('%Y-%m')}` to `{times[-1].strftime('%Y-%m')}`  "
                        f"({len(times)} steps)")
            st.markdown("**Spatial coverage**")
            st.markdown(
                f"Lat: `{float(ds.latitude.min()):.1f}°` to `{float(ds.latitude.max()):.1f}°`  \n"
                f"Lon: `{float(ds.longitude.min()):.1f}°` to `{float(ds.longitude.max()):.1f}°`"
            )
            if hasattr(ds, "title"):
                st.markdown(f"**Title:** {ds.title}")
            if hasattr(ds, "source"):
                st.markdown(f"**Source:** {ds.source}")
