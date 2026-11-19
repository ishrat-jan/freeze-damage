"""Write the Markdown report (report.md) and export it to PDF (report.pdf).

The text is filled in from the result tables, so a new run (e.g. the full 10 m
run on Kaggle) produces a report with its own numbers. The report uses plain
ASCII so the PDF can use the standard built-in fonts.
"""
import markdown
import pandas as pd
from fpdf import FPDF


def _md_table(df, cols=None):
    """DataFrame -> Markdown table."""
    df = df if cols is None else df[cols]
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def _img(name, caption):
    return f"![{caption}](maps/{name}.png)\n\n*{caption}*\n"


def _get(df, **where):
    """Single value lookup helper: _get(df, date=..., group=..., col='x')."""
    col = where.pop("col")
    m = df
    for k, v in where.items():
        m = m[m[k] == v]
    return m[col].iloc[0]


def build_markdown(r, cfg):
    """r: dict of results from run_pipeline.main()."""
    d, t, rep = cfg["dates"], cfg["thresholds"], cfg["report"]
    py, ter = cfg["prior_year"], cfg["terrain"]
    y = py["label"]
    early, late = d["damage_early"], d["damage_late"]
    res = cfg["run"]["res"]
    dmg, rec, area, cover = r["dmg_tab"], r["rec_tab"], r["area_tab"], r["cover_tab"]
    season, prior, kz, noise = r["season"], r["prior"], r["kz"], r["noise"].iloc[0]
    by_band, by_low = r["by_band"], r["by_low"]

    def dpct(day, cls="damaged (mild or worse)", group="citrus", col="pct"):
        return _get(dmg, date=day, group=group, **{"class": cls}, col=col)

    def area_of(prefix):
        return area[area.category.str.startswith(prefix)].area_ha.iloc[0]

    cdl_total, assessed = area_of("CDL citrus"), area_of("assessed")
    no_base, low_ndvi = area_of("excluded: no clear baseline"), area_of("excluded: baseline NDVI")
    no_dmg_data = area_of("excluded: not clear")
    kz_area = kz.area_ha.sum() if len(kz) else 0
    new_dmg = _get(rec, group="citrus", **{"class": "new damage"}, col="pct")

    # Baseline sensitivity: compact table + one sentence per damage date
    sens = r["sens"]
    ref = sens.variant.iloc[0]
    sens_tab = sens.assign(date=sens.date.str[5:])[
        ["variant", "date", "assessed_ha", "damaged_pct", "shared_ha",
         f"shared_damaged_pct_{ref}", "shared_damaged_pct"]
    ].rename(columns={"assessed_ha": "own ha", "damaged_pct": "own %", "shared_ha": f"shared with {ref} ha",
                      f"shared_damaged_pct_{ref}": f"{ref} % (shared)", "shared_damaged_pct": "this % (shared)"})
    sens_versions = "\n".join(
        f"- {x.variant}: {x.baseline} ({', '.join(pd.Timestamp(v).strftime('%d %b') for v in cfg['baseline_sensitivity']['variants'][x.variant]['dates'])})"
        for _, x in sens[sens.date == early].iterrows())
    # +/- uncertainty from the baseline choice (variants that cover enough of the reference citrus)
    min_share = cfg["baseline_sensitivity"]["uncertainty_min_share"]
    ref_ha = sens[sens.variant == ref].set_index("date").assessed_ha
    big = sens[(sens.variant != ref) & (sens.assessed_ha >= min_share * sens.date.map(ref_ha))]
    unc = (big.shared_damaged_pct - big[f"shared_damaged_pct_{ref}"]).abs().max() if len(big) else float("nan")
    unc_txt = f"about +/-{unc:.0f} percentage points" if unc == unc else "not estimated"
    sens_lines = []
    for day in [early, late]:
        for _, x in sens[(sens.date == day) & (sens.variant != ref)].iterrows():
            diff = x.shared_damaged_pct - x[f"shared_damaged_pct_{ref}"]
            sens_lines.append(f"- {day[5:]}, {x.variant} vs {ref} on {x.shared_ha} ha shared citrus: "
                              f"{x.shared_damaged_pct}% vs {x[f'shared_damaged_pct_{ref}']}% ({diff:+.1f} points).")
    sens_text = "\n".join(sens_lines)
    sv = sens[sens.date == early].set_index("variant")
    sl = sens[sens.date == late].set_index("variant")
    few = sv.n_dates.idxmin()                       # variant with the fewest baseline dates
    few_share = 100 * sv.loc[few, "assessed_ha"] / sv.loc[ref, "assessed_ha"]
    lighter = all(x.loc[few, f"shared_damaged_pct_{ref}"] < x.loc[ref, "damaged_pct"] for x in (sv, sl))
    same_pattern = all(sl.loc[k, "damaged_pct"] > sv.loc[k, "damaged_pct"] for k in sv.index)
    sens_text += "\n\n**How to read this.**\n\n- Baseline difference to " + ref + " on shared citrus (median NDVI): " + \
        ", ".join(f"{k} {sv.loc[k, 'shared_baseline_minus_ref']:+.3f}" for k in sv.index if k != ref) + \
        """. A higher baseline gives more damage, a lower one less. Each baseline is a median of only a few
  clear dates, so adding or removing a date moves individual pixels either way, and the effect need not be
  the same on both damage dates."""
    sens_text += (f"\n- Version {few} ({sv.loc[few, 'baseline']}) can assess only {sv.loc[few, 'assessed_ha']} ha "
                  f"({few_share:.0f}% of the citrus {ref} assesses), because its dates are often cloudy, so its "
                  f"\"own\" % is not comparable with {ref}'s.")
    if lighter:
        sens_text += (f" Its pixels are also a lightly damaged part of the area ({ref} gives them less damage "
                      f"than the citrus as a whole).")
    sens_text += ("\n- Removing a date shows how much that date matters: if a freeze had already lowered NDVI "
                  "on a baseline date, removing it would raise the baseline and the damage %.")
    sens_text += (f"\n- Versions covering at least {min_share:.0%} of {ref}'s citrus change the damage % by up to "
                  f"{unc:.1f} points, so the damage figures in this report should be read as {unc_txt}. "
                  f"{ref} is kept as the main result because it uses the most clear observations.")
    if same_pattern:
        sens_text += (f"\n- The exact damage % depends on the baseline, but the main pattern holds under every "
                      f"version: less damage on {early[5:]}, much more on {late[5:]}.")

    # Prior-year comparison (citrus)
    pc = prior[prior.group == "citrus"].set_index("period")
    def prior_line(period):
        p = pc.loc[period]
        return (f"{period} ({p.date_2026[5:]}): normal-year ({y}) citrus median change "
                f"{p[f'median_change_{y}']:+.3f} with {p[f'damaged_pct_{y}']}% of citrus below the threshold; "
                f"2026: {p.median_change_2026:+.3f} and {p.damaged_pct_2026}%. "
                f"Difference: {p.change_2026_minus_prior:+.3f} NDVI, {p.damaged_pct_2026_minus_prior:+.1f} percentage points.")

    # Frost: citrus damage rate per frost-prone class (same classes as the map)
    t_hollow = ter["hollow_m"]
    hollow_def = "lower than" if t_hollow == 0 else f"more than {abs(t_hollow)} m below"
    fr = r["by_frost"].set_index("frost_class")
    hollows = fr.drop(index="not low-lying", errors="ignore")
    hollow_rate = (hollows.damage_rate * hollows.citrus_pixels).sum() / max(hollows.citrus_pixels.sum(), 1)
    other_rate = fr.damage_rate.get("not low-lying", float("nan"))
    hollow_ha = hollows.citrus_ha.sum()
    frost_date = early if ter["damage_date"] == "early" else late

    top = kz.head(rep["top_zones_in_report"])
    kz_cols = ["zone_id", "area_ha", "mean_change_early", "mean_change_late", "severe_share", "lon", "lat"]
    base_days = ", ".join(d["baseline"])
    n_base = len(d["baseline"])
    cit_cover = cover.set_index("clear_baseline_dates")
    one_or_less = cit_cover.loc[[0, 1], "citrus_pct"].sum()

    md = f"""# {rep['title']}

Resolution: **{res} m**{' (TEST MODE - coarse pixels, numbers are approximate)' if cfg['run']['test'] else ''}.
Sentinel-2 L2A, tile 17RMM, {cfg['sentinel2']['start_date']} to {cfg['sentinel2']['end_date']}.

## 1. Summary

- **Citrus area:** USDA CDL {cfg['citrus']['cdl_year']} maps **{cdl_total} ha** of citrus in the AOI; **{assessed} ha** could be assessed (section 3.3 explains the difference).
- **Damaged citrus** (NDVI drop of {abs(t['mild']):.2f} or more vs the January baseline):
  **{dpct(early)}%** on {early} and **{dpct(late)}%** on {late}. Both dates are reported; neither alone is "the" answer (section 4).
  Depending on the choice of baseline dates, these figures are uncertain by {unc_txt} (section 4).
- Damage is mostly mild. Severe citrus damage: {dpct(early, 'severe', col='area_ha')} ha on {early}, {dpct(late, 'severe', col='area_ha')} ha on {late}.
- **Delayed symptoms:** {new_dmg}% of citrus is damaged on {late} but was not on {early}. Freeze-injured leaves often stay green for days to weeks before they desiccate and drop, so damage is expected to grow after the first post-freeze image.
- **Normal seasonal change ({y}):** {prior_line('early')} {prior_line('late')}
- **Kill zones** (moderate+ on {late} and already damaged on {early}, >= {cfg['kill_zones']['min_area_ha']} ha): **{len(kz)} zones, {kz_area:.2f} ha**.
- **Frost-prone areas:** citrus in hollows ({hollow_def} the mean elevation within {ter['lowlying_radius_m']} m; {hollow_ha:.1f} ha of assessed citrus) was damaged at a rate of **{100 * hollow_rate:.1f}%** vs **{100 * other_rate:.1f}%** elsewhere ({frost_date}).

## 2. Method

1. Sentinel-2 L2A from Earth Search (STAC), loaded with odc-stac on a {res} m UTM 17N grid. Reflectance = DN / {cfg['sentinel2']['scale']}. The BOA offset ({cfg['sentinel2']['boa_offset']}) is detected per scene from the data, because Earth Search's offset flag is wrong for several {y} scenes; no scene used here contains the offset.
2. Cloud mask: keep only SCL classes {cfg['sentinel2']['clear_scl_classes']} and pixels with nir + red > 0.
3. NDVI per date. Baseline = per-pixel median of the clear observations on {base_days}. Vegetation = baseline NDVI >= {t['vegetation_ndvi']}.
4. Change = NDVI(date) - baseline for {early} (early) and {late} (later).
5. Damage classes: none >= {t['mild']}, mild {t['mild']} to {t['moderate']}, moderate {t['moderate']} to {t['severe']}, severe < {t['severe']}.
6. Recovery ({late} vs {early}) with a tolerance of {t['recovery_tolerance']} NDVI.
7. Citrus = USDA CDL {cfg['citrus']['cdl_year']} class {cfg['citrus']['cdl_class']}, reprojected with nearest neighbour.
8. Kill zones: connected patches ({cfg['kill_zones']['connectivity']}-neighbour), small patches removed, exported as polygons.
9. Seasonal checks: (a) citrus vs control vegetation in 2026 (not subtracted - the controls also froze); (b) the same NDVI change computed for {y} (baseline {', '.join(py['baseline'])}; {py['damage_early']} and {py['damage_late']}), same AOI, masks, thresholds and citrus mask.
10. Frost-prone areas: Copernicus DEM GLO-30, low-lying index = elevation minus the local mean within {ter['lowlying_radius_m']} m.

## 3. Data quality

### 3.1 Scenes

{_md_table(r['inventory'])}

### 3.2 Clear pixels per date (% of AOI)

{_md_table(r['clear_tab'], ['date', 'clear_pct', 'cloud_pct', 'cirrus_pct', 'water_pct'])}

The January dates are affected by cloud (4, 7 Jan) and by Sen2Cor's cirrus flag (17, 22, 27 Jan).
All five dates are used in the baseline median, each masked first, so a pixel gets a baseline from
whichever dates were clear there.

### 3.3 Baseline coverage

{_md_table(cover)}

{one_or_less:.1f}% of citrus has at most one clear baseline date; for these pixels the baseline is a
single observation rather than a median, so their change values are less reliable.

{_img('baseline_count', 'Number of clear baseline dates per pixel')}

### 3.4 Citrus area: CDL total vs assessed

{_md_table(area)}

The CDL maps {cdl_total} ha of citrus, but only {assessed} ha can be assessed:
- {no_base} ha has no clear baseline date (cloud or cirrus on every baseline date), so no change can be computed.
- {low_ndvi} ha has a baseline NDVI below {t['vegetation_ndvi']}. These are sparse canopies - young or reset
  plantings, abandoned or cleared groves, or CDL mislabels - where an NDVI drop does not reliably indicate freeze damage.
- {no_dmg_data} ha was not clear on at least one damage date.

Damage percentages below refer to assessed citrus (vegetated, clear on that date).

### 3.5 Noise check (no-freeze NDVI difference, vegetated pixels)

{_md_table(r['noise'])}

Only {noise['pct_below_mild_threshold']}% of stable vegetated pixels fall below the {t['mild']} threshold
between two pre-freeze dates, so the "mild" class is well above normal date-to-date noise.

## 4. Damage on {early} and {late}

{_md_table(dmg[dmg.group == 'citrus'], ['date', 'class', 'area_ha', 'pct'])}

**How to read the two dates.** {early} is about 4-5 days after the freeze peak (1-2 Feb). Freeze-injured
citrus leaves often stay on the tree and remain fairly green for days to weeks before they wilt, desiccate
and drop, so {early} is likely to *underestimate* the damage. {late} captures more of the delayed leaf
drop, but is also further from the baseline, so normal seasonal change has more time to add up
(section 6). The true freeze damage is most likely between the two numbers, and closer to {late} where
the prior-year change is small.

{_img(f'damage_citrus_{early}', f'Citrus damage classes, {early}')}
{_img(f'damage_citrus_{late}', f'Citrus damage classes, {late}')}
{_img(f'ndvi_change_{late}', f'NDVI change {late} vs baseline, all vegetation')}

### Baseline sensitivity

The same classification with three sets of baseline dates (citrus, % damaged = mild or worse).
"own" = all citrus that version can assess (fewer baseline dates -> fewer assessable pixels, so
"own" mixes a different baseline with different pixels). "shared" = only citrus assessed by both
that version and {ref}, so the baselines are compared on the same pixels.

{sens_versions}

{_md_table(sens_tab)}

{sens_text}

## 5. Recovery ({late} vs {early})

{_md_table(rec[rec.group == 'citrus'], ['class', 'area_ha', 'pct'])}

"New damage" ({new_dmg}% of citrus) is consistent with delayed leaf drop after the freeze. "Same or worse"
means *persistent damage as of {late}*, not confirmed permanent loss.

{_img('recovery_citrus', 'Citrus recovery classes')}

## 6. Seasonal checks

### 6.1 Prior year ({y}): how much change is normal?

{y} scenes used (clear % of AOI; the offset column shows the BOA offset applied, decided from the data -
Earth Search's flag is wrong for some {y} scenes):

{_md_table(r['clear_prior'], ['date', 'clear_pct', 'cloud_pct', 'cirrus_pct', 'boa_offset_used', 'es_offset_flag'])}

{_md_table(prior)}

- {prior_line('early')}
- {prior_line('late')}

The {y} values show how much NDVI normally changes between the January baseline and these February dates
without this freeze, measured the same way. The part of the 2026 change that is beyond the {y} change is the
best available estimate of the freeze effect. This is a single prior year, so it shows one example of
normal variation, not an average.

{_img('prior_year_comparison', f'Median NDVI change, {y} vs 2026')}

### 6.2 Citrus vs control vegetation (2026)

{_md_table(season, ['period', 'date', 'group', 'n_pixels', 'median_change', 'damaged_pct', 'citrus_minus_group'])}

The control vegetation also experienced the freeze (pasture, native vegetation and even pine can be
damaged), so it is not subtracted from citrus. It shows whether citrus reacted more strongly than other
vegetation in the same images.

{_img('seasonal_check', 'Median NDVI change: citrus vs control vegetation, 2026')}

## 7. Kill zones

{len(kz)} kill zones, {kz_area:.2f} ha in total. Largest zones (full list: tables/kill_zones.csv,
polygons: vectors/kill_zones.gpkg):

{_md_table(top, kz_cols) if len(kz) else 'No kill zones found.'}

{_img('kill_zones', 'Kill-zone locations') if len(kz) else ''}
{_img('kill_zone_zoom', 'Close-up of the largest kill zone') if len(kz) else ''}

## 8. Frost-prone areas

Cold air drains downhill on calm, clear freezing nights and pools in low ground. The low-lying index
compares each pixel's elevation with the mean elevation within {ter['lowlying_radius_m']} m
(negative = lower than the surroundings). Damage date used: {frost_date}.

Damage rate by frost-prone class (the classes shown on the map):

{_md_table(r['by_frost'])}

Damage rate by low-lying index bin:

{_md_table(by_low)}

{_img('damage_by_lowlying', 'Citrus damage rate by low-lying index')}
{_img('frost_prone', 'Frost-prone areas: hollows and deep hollows')}

**Supplementary note - absolute elevation bands.** Damage per {ter['elevation_band_m']} m elevation band is
in tables/damage_by_elevation_band.csv and maps/damage_by_elevation.png. Absolute elevation mostly
reflects where the groves happen to be (most citrus is in one or two bands), so it is not used for the
frost-prone map.

## 9. Limitations

- The damage dates are about 1 and 3 weeks after the freeze. "Same or worse" means persistent damage as of {late}, not confirmed tree loss; a spring image (new flush) is needed to separate dead from defoliated trees.
- The prior-year comparison uses one year only. Central Florida also had a cold outbreak around 21-22 January {y} (the Gulf Coast winter storm); check local weather records - if groves were affected, the {y} change is not a fully "normal" season and may itself include some cold effect.
- The baseline may already include damage from storm Ezra (30 Dec - 1 Jan); 4 Jan is the first baseline date after it.
- Cloud and cirrus masking leave many pixels with only one or two baseline observations (section 3.3).
- CDL is a modelled crop map; some citrus pixels may be misclassified, and recently abandoned groves may still be labelled citrus.
- Copernicus DEM is a 30 m surface model with about 2 m vertical accuracy; small hollows are not resolved, and lakes moderate local temperatures, which the index does not model.
- The frost-prone map relates past damage to terrain; it is a planning guide, not a temperature model.
{'- TEST MODE: 60 m pixels blur small groves and reduce the damaged share; use the 10 m run for reporting.' if cfg['run']['test'] else ''}

## 10. Output files

- rasters/: ndvi_baseline, baseline_clear_count, ndvi_change_*, damage_class_*, recovery_class, kill_zone_id, citrus_mask, dem, lowlying_index, frost_prone_class (GeoTIFF, UTM 17N)
- vectors/kill_zones.gpkg
- tables/: all CSV tables shown above (incl. prior_year_comparison.csv, baseline_coverage.csv, baseline_sensitivity.csv)
- maps/: PNG maps and charts
"""
    return md


def write_pdf(md_text, out_dir, cfg):
    """Markdown -> HTML -> PDF with fpdf2 (pure Python, no system libraries)."""
    html = markdown.markdown(md_text, extensions=["tables"])
    # fpdf needs absolute image paths and an explicit image width
    width = cfg["report"]["pdf_image_width_mm"]
    html = html.replace('src="maps/', f'width="{width * 3}" src="{out_dir}/maps/')
    # Tables: thin borders, small font
    html = html.replace("<table>", '<table border="1" width="100%">').replace("<th>", '<th align="left">')
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=9)
    pdf.write_html(html, table_line_separators=True)
    path = out_dir / "report.pdf"
    pdf.output(str(path))
    return path


def write_report(r, cfg):
    out = cfg["run"]["out"]
    md = build_markdown(r, cfg)
    (out / "report.md").write_text(md)
    pdf = write_pdf(md, out, cfg)
    return out / "report.md", pdf
