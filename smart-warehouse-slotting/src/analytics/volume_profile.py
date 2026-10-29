"""Phase-3 volume & capacity-class profiling.

Maps each SKU to the bin capacity classes it can physically fit (per-dimension and volume),
and reports footprint density, required storage volume/weight, and any mismatches.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def volume_profiles(products: pd.DataFrame, locations: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    """Per-SKU profile.

    - unit_volume_cm3 (already on products)
    - total_volume_cm3 = quantity_on_hand * unit_volume_cm3
    - total_weight_kg = quantity_on_hand * unit_weight_kg
    - fits_median_bin: fits the representative bin's per-dimension and volume
    - compatible_bins: number of location capacity classes the SKU fits
    """
    bins_ = locations[["usable_volume_cm3", "width_cm", "height_cm", "depth_cm"]].drop_duplicates()
    med_bin = bins_.median()

    # Canonical inventory is one row per (sku, bin); aggregate to on-hand per SKU.
    on_hand = inventory.groupby("sku_id", as_index=False)["quantity_on_hand"].sum()
    df = products.merge(on_hand, on="sku_id", how="left")
    df["quantity_on_hand"] = df["quantity_on_hand"].fillna(0).astype(int)
    df["total_volume_cm3"] = df["quantity_on_hand"] * df["unit_volume_cm3"]
    df["total_weight_kg"] = df["quantity_on_hand"] * df["unit_weight_kg"]

    # Count compatible bins (per-dimension fit) and median-bin fit.
    def _fits(row, b):
        return (
            row["width_cm"] <= b["width_cm"]
            and row["height_cm"] <= b["height_cm"]
            and row["depth_cm"] <= b["depth_cm"]
            and row["unit_volume_cm3"] <= b["usable_volume_cm3"]
        )

    n_compatible = []
    fits_med = []
    for _, row in df.iterrows():
        c = 0
        for _, b in bins_.iterrows():
            if _fits(row, b):
                c += 1
        n_compatible.append(c)
        fits_med.append(bool(_fits(row, med_bin)))

    df["compatible_bin_classes"] = n_compatible
    df["fits_median_bin"] = fits_med
    df["vol_share_of_median_bin"] = df["unit_volume_cm3"] / med_bin["usable_volume_cm3"] if med_bin["usable_volume_cm3"] else float("nan")
    cols = [
        "sku_id",
        "category_id",
        "unit_volume_cm3",
        "unit_weight_kg",
        "quantity_on_hand",
        "total_volume_cm3",
        "total_weight_kg",
        "compatible_bin_classes",
        "fits_median_bin",
        "vol_share_of_median_bin",
    ]
    return df[cols]


def build_volume_profiles(processed: Path, products: pd.DataFrame, out: Path | None = None) -> pd.DataFrame:
    locations = pd.read_csv(processed / "locations.csv")
    inventory = pd.read_csv(processed / "inventory.csv")
    df = volume_profiles(products, locations, inventory)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    return df