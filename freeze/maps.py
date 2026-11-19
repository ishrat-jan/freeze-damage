"""PNG maps and charts for the report.

Large rasters are plotted with matplotlib imshow after downsampling (taking
every n-th pixel), never with xarray's .plot (far too slow at 10 m).
Axes show UTM 17N coordinates in km.
"""
import math

import matplotlib

matplotlib.use("Agg")                     # no screen needed (works on Kaggle/servers)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from freeze.damage import DAMAGE_NAMES, NODATA  # noqa: E402


# ---------- helpers ----------
def _step(shape, cfg):
    """Downsampling step so the longer side is at most max_pixels."""
    return max(1, math.ceil(max(shape) / cfg["maps"]["max_pixels"]))


def _extent_km(da):
    """imshow extent (left, right, bottom, top) in km from the pixel centres."""
    x, y = da.x.values, da.y.values
    dx, dy = abs(x[1] - x[0]) / 2, abs(y[1] - y[0]) / 2
    return [(x.min() - dx) / 1000, (x.max() + dx) / 1000, (y.min() - dy) / 1000, (y.max() + dy) / 1000]


def _figure(da, title, cfg):
    c = cfg["maps"]["colors"]
    fig, ax = plt.subplots(figsize=(7, 9), facecolor=c["surface"])
    ax.set_facecolor(c["surface"])
    ax.set_title(title, color=c["text"], fontsize=11, loc="left")
    ax.set_xlabel("Easting (km, UTM 17N)", color=c["text_muted"], fontsize=8)
    ax.set_ylabel("Northing (km)", color=c["text_muted"], fontsize=8)
    ax.tick_params(colors=c["text_muted"], labelsize=7)
    for s in ax.spines.values():
        s.set_color(c["background_land"])
    return fig, ax


def _save(fig, name, cfg):
    path = cfg["run"]["out"] / "maps" / f"{name}.png"
    fig.savefig(path, dpi=cfg["maps"]["dpi"], bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _legend(ax, labels, colors, cfg, title=None):
    handles = [Patch(facecolor=col, edgecolor="none", label=lab) for lab, col in zip(labels, colors)]
    leg = ax.legend(handles=handles, loc="lower left", fontsize=7, title=title, title_fontsize=7,
                    frameon=True, facecolor=cfg["maps"]["colors"]["surface"], edgecolor="none")
    for t in leg.get_texts():
        t.set_color(cfg["maps"]["colors"]["text"])


# ---------- maps ----------
def rgb_image(refl, day, cfg):
    """True-colour composite (downsampled, contrast-stretched) as an RGB array."""
    s = _step(refl.red.shape[-2:], cfg)
    sel = refl.sel(time=np.datetime64(day))
    rgb = np.dstack([sel[b].values[::s, ::s] for b in ["red", "green", "blue"]])
    lo, hi = np.nanpercentile(rgb, cfg["maps"]["rgb_stretch_pct"])
    return np.nan_to_num(np.clip((rgb - lo) / (hi - lo), 0, 1), nan=1.0)


def rgb_map(refl, day, cfg):
    fig, ax = _figure(refl.red, f"True colour, {day}", cfg)
    ax.imshow(rgb_image(refl, day, cfg), extent=_extent_km(refl.red), interpolation="nearest")
    return _save(fig, f"rgb_{day}", cfg)


def continuous_map(da, name, title, label, cfg, cmap="Greens", vmin=0, vmax=1):
    """Continuous raster (e.g. NDVI, NDVI change) with a colour bar."""
    s = _step(da.shape, cfg)
    fig, ax = _figure(da, title, cfg)
    im = ax.imshow(da.values[::s, ::s], extent=_extent_km(da), cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label(label, fontsize=8, color=cfg["maps"]["colors"]["text_muted"])
    cb.ax.tick_params(labelsize=7)
    return _save(fig, name, cfg)


def change_cmap(cfg):
    """Diverging map: red = NDVI loss, grey = no change, blue = gain."""
    c = cfg["maps"]["colors"]
    return LinearSegmentedColormap.from_list("change", [c["change_loss"], c["change_mid"], c["change_gain"]])


def class_map(classes, labels, colors, name, title, cfg, land=None, legend_title=None, zero=None):
    """Categorical raster. Class k (1..n) gets colors[k-1]; nodata is not coloured.

    land: optional boolean mask drawn in light grey underneath (for context).
    zero: optional (label, colour) to also draw class 0 (e.g. undamaged citrus);
          otherwise class 0 is left transparent.
    """
    c = cfg["maps"]["colors"]
    s = _step(classes.shape, cfg)
    fig, ax = _figure(classes, title, cfg)
    ext = _extent_km(classes)
    if land is not None:
        bg = np.where(land.values[::s, ::s], 1, np.nan)
        ax.imshow(bg, extent=ext, cmap=ListedColormap([c["background_land"]]), interpolation="nearest")
    v = classes.values[::s, ::s].astype(float)
    v[v == NODATA] = np.nan
    if zero is not None:
        labels, colors = [zero[0]] + list(labels), [zero[1]] + list(colors)
        v = v + 1                      # shift so class 0 uses the first colour
    v[v == 0] = np.nan
    ax.imshow(v, extent=ext, cmap=ListedColormap(colors), vmin=0.5, vmax=len(colors) + 0.5,
              interpolation="nearest")
    _legend(ax, labels, colors, cfg, legend_title)
    return _save(fig, name, cfg)


def kill_zone_map(refl, gdf, cfg):
    """Kill-zone outlines on the true-colour image, largest zones labelled."""
    m, c = cfg["maps"], cfg["maps"]["colors"]
    fig, ax = _figure(refl.red, f"Kill zones ({len(gdf)}) on true colour, {m['rgb_date']}", cfg)
    ax.imshow(rgb_image(refl, m["rgb_date"], cfg), extent=_extent_km(refl.red), interpolation="nearest")
    if len(gdf):
        g = gdf.copy()
        g["geometry"] = g.geometry.scale(1 / 1000, 1 / 1000, origin=(0, 0))   # m -> km
        g.boundary.plot(ax=ax, color=c["kill_zone"], linewidth=1.2)
        # Zones are small at AOI scale: a ring marker shows where each one is
        pts = g.geometry.centroid
        ax.scatter(pts.x, pts.y, s=60, facecolors="none", edgecolors=c["kill_zone"], linewidths=1.2)
        for _, r in g.head(m["label_top_zones"]).iterrows():
            p = r.geometry.centroid
            ax.annotate(f"{r.zone_id}", (p.x, p.y), xytext=(5, 5), textcoords="offset points",
                        fontsize=7, color=c["text"],
                        bbox={"boxstyle": "round,pad=0.15", "fc": c["surface"], "ec": "none", "alpha": 0.8})
    _legend(ax, ["kill zone (ID = largest first)"], [c["kill_zone"]], cfg)
    return _save(fig, "kill_zones", cfg)


# ---------- charts ----------
def _chart_axes(title, cfg, figsize=(7, 3.6)):
    c = cfg["maps"]["colors"]
    fig, ax = plt.subplots(figsize=figsize, facecolor=c["surface"])
    ax.set_facecolor(c["surface"])
    ax.set_title(title, color=c["text"], fontsize=10, loc="left")
    ax.tick_params(colors=c["text_muted"], labelsize=8, length=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(c["background_land"])
    ax.yaxis.grid(True, color=c["background_land"], linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def rate_chart(labels, rates, areas, name, title, xlabel, cfg):
    """Bar chart of citrus damage rate per group (e.g. low-lying index bins)."""
    c = cfg["maps"]["colors"]
    fig, ax = _chart_axes(title, cfg)
    bars = ax.bar(labels, 100 * np.asarray(rates), color=c["series"][0], width=0.6)
    weights = np.asarray(areas)
    mean = 100 * (np.asarray(rates) * weights).sum() / max(weights.sum(), 1e-9)
    ax.axhline(mean, color=c["text_muted"], linewidth=1, linestyle="--")
    ax.text(len(labels) - 0.5, mean, f" mean {mean:.1f}%", va="bottom", ha="right", fontsize=7,
            color=c["text_muted"])
    for bar, ha in zip(bars, areas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{ha:.0f} ha", ha="center",
                va="bottom", fontsize=7, color=c["text_muted"])
    ax.set_ylabel("% of citrus damaged", fontsize=8, color=c["text_muted"])
    ax.set_xlabel(xlabel, fontsize=8, color=c["text_muted"])
    return _save(fig, name, cfg)


def lowlying_chart(by_low, cfg):
    """MAIN frost result: damage rate per low-lying index bin."""
    return rate_chart(list(by_low.lowlying_index), by_low.damage_rate, by_low.citrus_ha,
                      "damage_by_lowlying", "Citrus damage rate by low-lying index",
                      "elevation relative to surroundings (negative = hollow)", cfg)


def elevation_chart(by_band, cfg):
    """SUPPLEMENTARY: damage rate per elevation band (bands with enough pixels only)."""
    b = by_band[by_band.citrus_pixels >= cfg["terrain"]["min_band_pixels"]]
    labels = [f"{int(f)}-{int(t)}" for f, t in zip(b.band_from_m, b.band_to_m)]
    return rate_chart(labels, b.damage_rate, b.citrus_ha, "damage_by_elevation",
                      "Supplementary: citrus damage rate by elevation band", "elevation (m)", cfg)


def prior_year_chart(df, cfg):
    """Median NDVI change, 2026 vs prior year, one panel per period, bars per group."""
    c = cfg["maps"]["colors"]
    y = cfg["prior_year"]["label"]
    periods = list(dict.fromkeys(df.period))
    groups = list(dict.fromkeys(df.group))
    fig, axes = plt.subplots(1, len(periods), figsize=(8, 3.6), sharey=True, facecolor=c["surface"])
    years = [(y, f"median_change_{y}", c["series"][1]), ("2026", "median_change_2026", c["series"][0])]
    w = 0.38
    for ax, period in zip(np.atleast_1d(axes), periods):
        sub = df[df.period == period].set_index("group").loc[groups]
        ax.set_facecolor(c["surface"])
        for i, (label, col, color) in enumerate(years):
            x = np.arange(len(groups)) + (i - 0.5) * w
            ax.bar(x, sub[col], width=w * 0.92, color=color, label=label)
            for xi, v in zip(x, sub[col]):
                ax.text(xi, v, f"{v:.3f}", ha="center", va="top" if v < 0 else "bottom", fontsize=6.5,
                        color=c["text_muted"])
        d26 = sub.date_2026.iloc[0][5:]
        dprior = sub[f"date_{y}"].iloc[0][5:]
        ax.set_title(f"{period}: {dprior} ({y}) vs {d26} (2026)", fontsize=9, color=c["text"], loc="left")
        ax.set_xticks(range(len(groups)), groups, fontsize=7)
        ax.axhline(0, color=c["text_muted"], linewidth=0.8)
        ax.axhline(cfg["thresholds"]["mild"], color=c["change_loss"], linewidth=0.8, linestyle="--")
        ax.tick_params(colors=c["text_muted"], labelsize=7, length=0)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.yaxis.grid(True, color=c["background_land"], linewidth=0.6)
        ax.set_axisbelow(True)
    first = np.atleast_1d(axes)[0]
    first.set_ylabel("median NDVI change vs baseline", fontsize=8, color=c["text_muted"])
    first.legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("Normal seasonal change (prior year) vs 2026; dashed = damage threshold",
                 fontsize=10, color=c["text"], x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, "prior_year_comparison", cfg)


def seasonal_chart(df, cfg):
    """Median NDVI change per group and date: is citrus different from controls?"""
    c = cfg["maps"]["colors"]
    groups = list(dict.fromkeys(df.group))
    dates = list(dict.fromkeys(df.date))
    fig, ax = _chart_axes("Median NDVI change vs baseline: citrus vs control vegetation", cfg)
    w = 0.8 / len(groups)
    for i, g in enumerate(groups):
        vals = [df[(df.group == g) & (df.date == d)].median_change.iloc[0] for d in dates]
        x = np.arange(len(dates)) + (i - (len(groups) - 1) / 2) * w
        ax.bar(x, vals, width=w * 0.92, color=c["series"][i % len(c["series"])], label=g)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.3f}", ha="center", va="top", fontsize=7, color=c["text_muted"])
    ax.axhline(0, color=c["text_muted"], linewidth=0.8)
    ax.axhline(cfg["thresholds"]["mild"], color=c["change_loss"], linewidth=0.8, linestyle="--")
    ax.text(len(dates) - 0.5, cfg["thresholds"]["mild"], " damage threshold", fontsize=7,
            color=c["text_muted"], va="bottom", ha="right")
    ax.set_xticks(range(len(dates)), dates)
    ax.set_ylabel("median NDVI change", fontsize=8, color=c["text_muted"])
    ax.legend(fontsize=7, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.1),
              ncol=len(groups), borderaxespad=0)
    ax.set_title(ax.get_title(loc="left"), pad=22, loc="left", fontsize=10, color=c["text"])
    return _save(fig, "seasonal_check", cfg)


def kill_zone_zoom(dmg, citrus, gdf, land, cfg):
    """Close-up around the largest kill zone: citrus damage classes + zone outlines."""
    if not len(gdf):
        return None
    m, c = cfg["maps"], cfg["maps"]["colors"]
    half = m["zoom_window_km"] * 1000 / 2
    p = gdf.iloc[0].geometry.centroid                  # gdf is sorted largest first
    win = {"x": slice(p.x - half, p.x + half), "y": slice(p.y + half, p.y - half)}   # y runs north->south
    sub = dmg.where(citrus, NODATA).sel(**win)
    labels = [DAMAGE_NAMES[k] for k in (1, 2, 3)]
    fig, ax = _figure(sub, f"Zoom on kill zone {gdf.iloc[0].zone_id}: citrus damage {dmg.attrs.get("date", "")}", cfg)
    ext = _extent_km(sub)
    ax.imshow(np.where(land.sel(**win).values, 1, np.nan), extent=ext,
              cmap=ListedColormap([c["background_land"]]), interpolation="nearest")
    v = sub.values.astype(float)
    v[v == NODATA] = np.nan
    ax.imshow(v + 1, extent=ext, cmap=ListedColormap([c["citrus_ok"]] + list(c["damage"])),
              vmin=0.5, vmax=4.5, interpolation="nearest")
    g = gdf.copy()
    g["geometry"] = g.geometry.scale(1 / 1000, 1 / 1000, origin=(0, 0))
    g.boundary.plot(ax=ax, color=c["text"], linewidth=1.2)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    _legend(ax, ["none"] + labels + ["kill zone outline"],
            [c["citrus_ok"]] + list(c["damage"]) + [c["text"]], cfg, "citrus damage")
    return _save(fig, "kill_zone_zoom", cfg)
