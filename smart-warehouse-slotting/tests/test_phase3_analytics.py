"""Phase-3 tests: demand, ABC, affinity, volume-profile modules."""

import pandas as pd
import pytest

from src.analytics import abc, affinity, demand, volume_profile


def _demand_fixture():
    return pd.DataFrame(
        {
            "order_id": ["O1", "O1", "O2", "O3", "O3", "O4"],
            "sku_id": ["P", "Q", "P", "P", "R", "Q"],
            "quantity": [1, 1, 2, 1, 1, 1],
        }
    )


def test_demand_summary():
    items = _demand_fixture()
    all_skus = ["P", "Q", "R", "Z"]  # Z has zero demand
    df = demand.demand_summary(items, all_skus, assign_zero=True)
    assert set(df["sku_id"]) == {"P", "Q", "R", "Z"}
    assert df.loc[df["sku_id"] == "P", "demand_frequency"].iloc[0] == 3  # P in O1,O2,O3
    assert df.loc[df["sku_id"] == "P", "total_units_sold"].iloc[0] == 4  # 1+2+1
    assert df.loc[df["sku_id"] == "Z", "demand_frequency"].iloc[0] == 0


def test_abc_classify_covers_all():
    summary = demand.demand_summary(_demand_fixture(), ["P", "Q", "R", "Z"], assign_zero=True)
    abc_df = abc.abc_classify_summary(summary, a_share=0.8, b_share=0.95)
    assert set(abc_df["sku_id"]) == {"P", "Q", "R", "Z"}
    assert abc_df["abc_class"].isin(["A", "B", "C"]).all()
    # zero-demand SKU must be C
    assert abc_df.loc[abc_df["sku_id"] == "Z", "abc_class"].iloc[0] == "C"


def test_abc_boundaries():
    # 5 SKUs: A should be the smallest set reaching 80% of cumulative demand.
    summary = pd.DataFrame(
        {"sku_id": ["a", "b", "c", "d", "e"], "velocity": [100, 60, 40, 30, 20]}
    )
    abc_df = abc.abc_classify_summary(summary, a_share=0.80, b_share=0.95)
    # total demand = 250; A boundary at 200 (a+b); B at 237.5 (a+b+c)
    assert abc_df.loc[abc_df["sku_id"] == "a", "abc_class"].iloc[0] == "A"
    assert abc_df.loc[abc_df["sku_id"] == "b", "abc_class"].iloc[0] in ("A", "B")
    a_max = abc_df[abc_df["abc_class"] == "A"]["cumulative_share"].max()
    assert a_max <= 0.80 + 1e-9


def test_affinity_pair_statistics():
    items = pd.DataFrame(
        {
            "order_id": ["O1", "O1", "O2", "O2", "O3", "O3"],
            "sku_id": ["P", "Q", "P", "Q", "P", "R"],
            "quantity": [1, 1, 1, 1, 1, 1],
        }
    )
    pairs = affinity.pair_statistics(items)
    row = pairs[(pairs["sku_a"] == "P") & (pairs["sku_b"] == "Q")].iloc[0]
    # P,Q co-occur in O1 and O2 => 2; P in all 3, Q in 2, total orders 3.
    assert row["co_count"] == 2
    assert row["support"] == pytest.approx(2 / 3)
    assert row["confidence_ab"] == pytest.approx(2 / 3)
    assert row["lift"] == pytest.approx((2 / 3) / (2 / 3))


def test_affinity_clusters_disjoint():
    pairs = pd.DataFrame(
        {
            "sku_a": ["a", "b", "c"],
            "sku_b": ["b", "c", "d"],
            "co_count": [10, 10, 10],
            "support": [0.5] * 3,
            "confidence_ab": [0.9] * 3,
            "confidence_ba": [0.9] * 3,
            "lift": [5.0, 5.0, 5.0],
        }
    )
    clusters = affinity.affinity_clusters(pairs, lift_threshold=5.0, min_co_count=1)
    members = {s for cl in clusters for s in cl}
    assert members == {"a", "b", "c", "d"}
    # connected path a-b-c-d collapsed to one component
    assert len(clusters) == 1 and len(clusters[0]) == 4


def test_volume_profiles():
    products = pd.DataFrame(
        {
            "sku_id": ["P", "Q"],
            "category_id": ["CAT001", "CAT001"],
            "width_cm": [10, 10],
            "height_cm": [10, 10],
            "depth_cm": [10, 10],
            "unit_weight_kg": [1.0, 1.0],
            "unit_volume_cm3": [1000, 1000],
        }
    )
    locations = pd.DataFrame(
        {
            "location_id": ["L1", "L2"],
            "usable_volume_cm3": [20000, 20000],
            "width_cm": [40, 40],
            "height_cm": [40, 40],
            "depth_cm": [40, 40],
        }
    )
    inventory = pd.DataFrame({"sku_id": ["P", "Q"], "quantity_on_hand": [3, 5]})
    vp = volume_profile.volume_profiles(products, locations, inventory)
    assert vp.loc[vp["sku_id"] == "P", "total_volume_cm3"].iloc[0] == 3000
    assert vp.loc[vp["sku_id"] == "Q", "total_weight_kg"].iloc[0] == 5.0
    assert vp["fits_median_bin"].all()