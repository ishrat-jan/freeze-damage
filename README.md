# Freeze damage assessment – Lake County (Clermont), Sentinel-2

An open-source Python pipeline that maps freeze damage to citrus after the
January–February 2026 Florida freeze, using Sentinel-2 NDVI change detection
(4 Jan – 23 Feb 2026, tile 17RMM).

It supports three goals from the brief:

| Goal | Output |
|---|---|
| Insurance claims: kill zones within orchards | `vectors/kill_zones.gpkg`, `tables/kill_zones.csv`, `maps/kill_zones.png`, `maps/kill_zone_zoom.png` |
| Recovery monitoring: regrowth vs persistent damage | `rasters/recovery_class.tif`, `tables/recovery_summary.csv`, `maps/recovery_citrus.png` |
| Future planning: frost-prone areas | `rasters/frost_prone_class.tif`, `rasters/lowlying_index.tif`, `tables/damage_by_frost_class.csv`, `tables/damage_by_lowlying.csv`, `maps/frost_prone.png` |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_pipeline.py --test   # 60 m quick run (~65 MB download incl. 2025, ~2-3 min)
python run_pipeline.py          # full 10 m run (~1-2 GB download; use Kaggle)
```

Results go to `outputs/<resolution>m/`, so a test run never overwrites the
10 m results. Downloads are cached in `data/cache/`, so a second run skips them.

### On Kaggle

1. Create a notebook with Internet enabled and upload/clone this repository.
2. `!pip install -r requirements.txt`
3. `!python run_pipeline.py`
4. Download `outputs/10m/` (or save it as a Kaggle dataset).

## Configuration

Every setting (AOI, dates, thresholds, URLs, colours, paths) is in
`config.yaml`. The code has no hard-coded values. Edit the config, not the code.

## How it works

| Step | Module | What it does |
|---|---|---|
| 1 | `freeze/config.py` | Load `config.yaml`, apply test mode, create folders |
| 2 | `freeze/sentinel.py` | STAC search (Earth Search), scene inventory, load bands with odc-stac (cached), BOA offset detected per scene from the data |
| 3 | `freeze/masks.py` | Clear-pixel mask (SCL 4/5, nir+red > 0), water mask, vegetation mask, clear-fraction table |
| 4 | `freeze/ndvi.py` | NDVI, baseline median (4/7/17/22/27 Jan, each cloud-masked), noise check, change maps |
| 5 | `freeze/damage.py` | Damage classes (none/mild/moderate/severe) and recovery classes |
| 6 | `freeze/citrus.py` | USDA CDL from CropScape (retries, expiring link), reprojected to the NDVI grid |
| 7 | `freeze/zones.py` | Kill zones: connected patches, size filter, polygons with statistics |
| 8 | `freeze/seasonal.py` | Citrus vs control vegetation (2026), and 2026 vs prior-year change |
| 8b | `freeze/sensitivity.py` | Baseline sensitivity: citrus damage % with baseline versions A/B/C (`baseline_sensitivity` in config) |
| 8c | `freeze/prior_year.py` | Same NDVI change for 2025 (same AOI, masks, thresholds): how much change is normal |
| 9 | `freeze/terrain.py` | Copernicus DEM, low-lying index (main), frost-prone map; elevation bands (supplementary) |
| 10 | `freeze/tables.py` | Summary CSV tables |
| 11 | `freeze/maps.py` | PNG maps and charts (imshow with downsampling) |
| 12 | `freeze/report.py` | `report.md` and `report.pdf` filled in from the results |
| – | `freeze/io_utils.py` | GeoTIFF writer |
| – | `run_pipeline.py` | Runs everything in order |

### Definitions

- **Damage** = NDVI(date) − baseline NDVI, on vegetated pixels (baseline NDVI ≥ 0.3):
  none ≥ −0.10 > mild ≥ −0.20 > moderate ≥ −0.30 > severe.
  The −0.10 threshold comes from the noise check (27 Jan − 17 Jan, before the main freeze).
- **Recovery** (21 Feb vs 6 Feb): recovered / partly recovered / same or worse / new damage,
  with a ±0.05 NDVI tolerance.
- **Kill zone** = citrus that is moderate or worse on 21 Feb **and** already damaged on 6 Feb;
  connected patches of at least 0.1 ha.
- **Frost-prone** = low-lying index (elevation minus the mean within 300 m): hollow −4 to 0 m (any ground below its surroundings),
  deep hollow < −4 m. Absolute elevation bands are a supplementary table only.
- **BOA offset**: Earth Search's `earthsearch:boa_offset_applied` flag is wrong for several 2025
  scenes, so the offset is detected from the data (an offset is present only if almost no
  clear-land red DN is below 1000). Disagreements are printed as warnings.

## Limitations

See section 9 of the generated report. The most important ones:
- 21 Feb is only about 3 weeks after the freeze. "Same or worse" means persistent damage, not confirmed tree loss.
- Freeze symptoms (leaf desiccation and drop) are delayed by days to weeks, so 6 Feb underestimates
  damage; both dates are reported. The 2025 comparison shows how much change is normal for the season
  (one prior year only; late January 2025 also had a cold outbreak).
- The damage % moves by a few percentage points with the choice of baseline dates
  (`tables/baseline_sensitivity.csv`, report section 4).
- Cloud and cirrus leave many pixels with only 1–2 baseline observations (`maps/baseline_count.png`).
