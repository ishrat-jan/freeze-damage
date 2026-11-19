"""Baseline sensitivity test: does the citrus damage % depend on the baseline dates?

For each baseline variant in config.yaml (baseline_sensitivity.variants) we redo
the baseline median, vegetation mask and damage classes exactly as in the main
run, then report citrus damage on both damage dates.

Two percentages per variant and date:
  damaged_pct        : on the citrus that variant can assess (its own pixels).
                       Fewer baseline dates -> fewer assessable pixels, so this
                       mixes "different baseline" with "different pixels".
  shared_baseline_minus_ref: median baseline difference to A on shared pixels.
  shared_* columns   : only on citrus assessed by BOTH this variant and the
                       reference (first variant, A) on that date, so the two are
                       compared on the same pixels.
"""
import numpy as np
import pandas as pd

from freeze import damage, masks, ndvi
from freeze.damage import NODATA


def baseline_sensitivity(nd, cit, cfg):
    variants = cfg["baseline_sensitivity"]["variants"]
    d = cfg["dates"]
    days = [d["damage_early"], d["damage_late"]]
    ha = cfg["run"]["pixel_ha"]
    c = cit.values

    # Change and "assessed" mask per variant and date
    change, assessed, damaged, bases = {}, {}, {}, {}
    for key, v in variants.items():
        base, _ = ndvi.baseline(nd, v["dates"])
        bases[key] = base.values
        veg = masks.vegetation_mask(base, cfg)
        for day in days:
            ch = ndvi.change(nd, base, day)
            dmg = damage.classify_damage(ch, veg, cfg).values
            assessed[key, day] = c & (dmg != NODATA)
            damaged[key, day] = (dmg >= 1) & (dmg != NODATA)
            change[key, day] = ch.values

    ref = next(iter(variants))          # the first variant (A) is the reference
    rows = []
    for day in days:
        for key, v in variants.items():
            a, dm = assessed[key, day], damaged[key, day]
            shared = a & assessed[ref, day]
            pct = lambda m, x: round(100 * float(x[m].mean()), 1) if m.any() else np.nan
            rows.append({
                "variant": key,
                "baseline": v["label"],
                "n_dates": len(v["dates"]),
                "date": day,
                "assessed_ha": round(a.sum() * ha, 1),
                "damaged_pct": pct(a, dm),
                "shared_ha": round(shared.sum() * ha, 1),
                f"shared_damaged_pct_{ref}": pct(shared, damaged[ref, day]),
                "shared_damaged_pct": pct(shared, dm),
                "shared_median_change": round(float(np.median(change[key, day][shared])), 3) if shared.any() else np.nan,
                # median (this baseline - reference baseline): > 0 means this baseline is greener
                "shared_baseline_minus_ref": round(float(np.median(bases[key][shared] - bases[ref][shared])), 3)
                if shared.any() else np.nan,
            })
    df = pd.DataFrame(rows)
    df.to_csv(cfg["run"]["out"] / "tables" / "baseline_sensitivity.csv", index=False)
    return df
