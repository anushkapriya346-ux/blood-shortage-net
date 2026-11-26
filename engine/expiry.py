"""FEFO inventory ordering, 72-hour expiry flagging, wastage projection, and redistribution.

Provides unit-level First-Expiry-First-Out (FEFO) queues, flags units expiring
within 72 hours, projects wastage against forecast demand, and recommends
transfers of expiring stock to nearby higher-demand facilities.
"""

from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path
from typing import Any, Optional

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.db import load_table
from engine.forecast import forecast_demand
from engine.risk import _resolve_reference_date, compute_expiry_aware_usable_units

COMPATIBLE_RECIPIENTS_RBC: dict[str, list[str]] = {
    "O-": ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"],
    "O+": ["O+", "A+", "B+", "AB+"],
    "A-": ["A-", "A+", "AB-", "AB+"],
    "A+": ["A+", "AB+"],
    "B-": ["B-", "B+", "AB-", "AB+"],
    "B+": ["B+", "AB+"],
    "AB-": ["AB-", "AB+"],
    "AB+": ["AB+"],
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))


def get_fefo_inventory(
    inventory_df: Optional[pd.DataFrame] = None,
    facilities_df: Optional[pd.DataFrame] = None,
    consumption_df: Optional[pd.DataFrame] = None,
    reference_date: Optional[date] = None,
    db_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Return inventory ordered by First-Expiry-First-Out (FEFO) with 72h flags."""
    if inventory_df is None:
        inventory_df = load_table("inventory_units", db_path=db_path)
    if facilities_df is None:
        facilities_df = load_table("facilities", db_path=db_path)
    if consumption_df is None:
        consumption_df = load_table("daily_consumption", db_path=db_path)

    ref_date = _resolve_reference_date(consumption_df, inventory_df, reference_date)

    if inventory_df is None or inventory_df.empty:
        return pd.DataFrame(
            columns=[
                "id",
                "facility_id",
                "facility_name",
                "blood_group",
                "component",
                "collected_date",
                "expiry_date",
                "expires_in_days",
                "expiring_72h",
                "fefo_rank",
            ]
        )

    inv = inventory_df.copy()
    inv["expires_in_days"] = (pd.to_datetime(inv["expiry_date"]).dt.date - ref_date).apply(
        lambda d: d.days
    )
    inv = inv[inv["expires_in_days"] >= 1].copy()

    if not facilities_df.empty:
        fac_names = facilities_df.set_index("id")["name"].to_dict()
        inv["facility_name"] = inv["facility_id"].map(fac_names).fillna(inv["facility_id"])
    else:
        inv["facility_name"] = inv["facility_id"]

    inv["expiring_72h"] = inv["expires_in_days"] <= 3
    inv = inv.sort_values(
        by=["facility_id", "blood_group", "component", "expires_in_days", "collected_date", "id"]
    ).reset_index(drop=True)
    inv["fefo_rank"] = (
        inv.groupby(["facility_id", "blood_group", "component"]).cumcount() + 1
    )
    return inv


def flag_expiring_units(
    fefo_df: Optional[pd.DataFrame] = None,
    hours_threshold: int = 72,
    db_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Return units expiring within `hours_threshold` hours sorted by urgency (FEFO)."""
    if fefo_df is None:
        fefo_df = get_fefo_inventory(db_path=db_path)
    if fefo_df.empty:
        return fefo_df
    max_days = max(1, hours_threshold // 24)
    exp = fefo_df[fefo_df["expires_in_days"] <= max_days].copy()
    return exp.sort_values(
        by=["expires_in_days", "component", "facility_id", "blood_group"]
    ).reset_index(drop=True)


def project_facility_wastage(
    facilities_df: Optional[pd.DataFrame] = None,
    inventory_df: Optional[pd.DataFrame] = None,
    forecast_df: Optional[pd.DataFrame] = None,
    consumption_df: Optional[pd.DataFrame] = None,
    reference_date: Optional[date] = None,
    db_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Project unused unit wastage per facility given FEFO expiry and forecast demand."""
    if facilities_df is None:
        facilities_df = load_table("facilities", db_path=db_path)
    if inventory_df is None:
        inventory_df = load_table("inventory_units", db_path=db_path)
    if consumption_df is None:
        consumption_df = load_table("daily_consumption", db_path=db_path)
    if forecast_df is None:
        forecast_df = forecast_demand(
            consumption_df=consumption_df, facilities_df=facilities_df, db_path=db_path
        )

    fefo = get_fefo_inventory(
        inventory_df=inventory_df,
        facilities_df=facilities_df,
        consumption_df=consumption_df,
        reference_date=reference_date,
    )

    rows: list[dict[str, Any]] = []
    for _, fac in facilities_df.iterrows():
        fac_id = fac["id"]
        fac_inv = fefo[fefo["facility_id"] == fac_id]
        total_units = int(len(fac_inv))
        exp_72h = int(fac_inv["expiring_72h"].sum()) if not fac_inv.empty else 0

        total_wastage = 0.0
        total_usable = 0.0
        for bg in fac_inv["blood_group"].unique() if not fac_inv.empty else []:
            bg_inv = fac_inv[fac_inv["blood_group"] == bg]
            fc_sub = forecast_df[
                (forecast_df["facility_id"] == fac_id) & (forecast_df["blood_group"] == bg)
            ]
            daily_dem = float(fc_sub["forecast"].mean()) if not fc_sub.empty else 0.0
            usable, wastage, _ = compute_expiry_aware_usable_units(
                bg_inv["expires_in_days"].tolist(), daily_dem
            )
            total_usable += usable
            total_wastage += wastage

        wastage_pct = round((total_wastage / total_units * 100.0), 1) if total_units > 0 else 0.0
        rows.append(
            {
                "facility_id": fac_id,
                "facility_name": fac["name"],
                "facility_type": fac["type"],
                "region": fac["region"],
                "total_units": total_units,
                "expiring_72h_units": exp_72h,
                "projected_usable_units": round(total_usable, 1),
                "projected_wastage_units": round(total_wastage, 1),
                "wastage_pct": wastage_pct,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            by=["projected_wastage_units", "expiring_72h_units"], ascending=False
        ).reset_index(drop=True)
    return df


def suggest_expiring_redistributions(
    facilities_df: Optional[pd.DataFrame] = None,
    inventory_df: Optional[pd.DataFrame] = None,
    forecast_df: Optional[pd.DataFrame] = None,
    consumption_df: Optional[pd.DataFrame] = None,
    max_radius_km: float = 150.0,
    reference_date: Optional[date] = None,
    db_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Suggest reallocating units expiring within 72h to higher-demand facilities."""
    if facilities_df is None:
        facilities_df = load_table("facilities", db_path=db_path)
    if inventory_df is None:
        inventory_df = load_table("inventory_units", db_path=db_path)
    if consumption_df is None:
        consumption_df = load_table("daily_consumption", db_path=db_path)
    if forecast_df is None:
        forecast_df = forecast_demand(
            consumption_df=consumption_df, facilities_df=facilities_df, db_path=db_path
        )

    fefo = get_fefo_inventory(
        inventory_df=inventory_df,
        facilities_df=facilities_df,
        consumption_df=consumption_df,
        reference_date=reference_date,
    )
    expiring = flag_expiring_units(fefo, hours_threshold=72)
    if expiring.empty or facilities_df.empty:
        return pd.DataFrame(
            columns=[
                "from_id",
                "from_name",
                "to_id",
                "to_name",
                "blood_group",
                "recipient_group",
                "component",
                "units",
                "min_expires_in_days",
                "distance_km",
                "unit_ids",
                "reason",
            ]
        )

    fac_map = facilities_df.set_index("id").to_dict(orient="index")

    # Compute each facility x blood_group daily demand & current non-expiring stock
    demand_map: dict[tuple[str, str], float] = {}
    stock_map: dict[tuple[str, str], int] = {}
    for fac_id in facilities_df["id"]:
        for bg in ["O+", "B+", "A+", "AB+", "O-", "B-", "A-", "AB-"]:
            fc_sub = forecast_df[
                (forecast_df["facility_id"] == fac_id) & (forecast_df["blood_group"] == bg)
            ]
            demand_map[(fac_id, bg)] = float(fc_sub["forecast"].mean()) if not fc_sub.empty else 0.0
            stock_map[(fac_id, bg)] = int(
                len(fefo[(fefo["facility_id"] == fac_id) & (fefo["blood_group"] == bg)])
            )

    # Group expiring units by (facility_id, blood_group, component)
    suggestions: list[dict[str, Any]] = []
    # Track allocated demand capacity at receivers so we don't overload one site
    received_load: dict[tuple[str, str], float] = {}

    grouped = expiring.groupby(["facility_id", "blood_group", "component"], sort=False)
    for (src_id, bg, comp), grp_df in grouped:
        src_demand = demand_map.get((src_id, bg), 0.0)
        # How many units can the source realistically consume within 3 days?
        src_3day_capacity = int(math.floor(src_demand * 3.0))
        excess_units_df = grp_df.iloc[src_3day_capacity:] if len(grp_df) > src_3day_capacity else pd.DataFrame()
        if excess_units_df.empty:
            continue

        # Find best candidate receivers within max_radius_km with high demand / low cover
        src_info = fac_map[src_id]
        candidates = []
        allowed_recips = (
            COMPATIBLE_RECIPIENTS_RBC.get(bg, [bg]) if comp == "RBC" else [bg]
        )
        for dst_id, dst_info in fac_map.items():
            if dst_id == src_id:
                continue
            dist = _haversine_km(src_info["lat"], src_info["lon"], dst_info["lat"], dst_info["lon"])
            if dist > max_radius_km:
                continue
            for rec_bg in allowed_recips:
                dst_dem = demand_map.get((dst_id, rec_bg), 0.0)
                if dst_dem <= 0.05:
                    continue
                dst_stock = stock_map.get((dst_id, rec_bg), 0) + received_load.get((dst_id, rec_bg), 0)
                cover = dst_stock / dst_dem
                exact_bonus = 0 if rec_bg == bg else 1
                # Receiver can absorb units within 3 days if 3-day demand > 0.8
                absorbable = max(0, int(round(dst_dem * 3.5)) - int(received_load.get((dst_id, rec_bg), 0)))
                if absorbable <= 0 and cover >= 7.0:
                    continue
                candidates.append((exact_bonus, cover, -dst_dem, dist, dst_id, rec_bg, max(1, absorbable), dst_dem))

        candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        remaining_df = excess_units_df.copy()

        for _, cover, _, dist, dst_id, rec_bg, absorbable, dst_dem in candidates:
            if remaining_df.empty:
                break
            take_n = min(len(remaining_df), absorbable)
            if take_n <= 0:
                continue
            batch = remaining_df.iloc[:take_n]
            remaining_df = remaining_df.iloc[take_n:]
            received_load[(dst_id, rec_bg)] = received_load.get((dst_id, rec_bg), 0) + take_n

            min_exp = int(batch["expires_in_days"].min())
            dst_name = fac_map[dst_id]["name"]
            src_name = src_info["name"]
            suggestions.append(
                {
                    "from_id": src_id,
                    "from_name": src_name,
                    "to_id": dst_id,
                    "to_name": dst_name,
                    "blood_group": bg,
                    "recipient_group": rec_bg,
                    "component": comp,
                    "units": int(take_n),
                    "min_expires_in_days": min_exp,
                    "distance_km": round(dist, 1),
                    "unit_ids": batch["id"].tolist(),
                    "reason": (
                        f"Expiring in {min_exp}d at {src_name} (excess over 72h local demand); "
                        f"{dst_name} uses {dst_dem:.1f} {rec_bg}/day ({cover:.1f}d cover)"
                    ),
                }
            )

    res_df = pd.DataFrame(suggestions)
    if not res_df.empty:
        res_df = res_df.sort_values(
            by=["min_expires_in_days", "units", "distance_km"], ascending=[True, False, True]
        ).reset_index(drop=True)
    return res_df


if __name__ == "__main__":
    fefo_df = get_fefo_inventory()
    exp_72 = flag_expiring_units(fefo_df)
    wastage_df = project_facility_wastage(inventory_df=fefo_df)
    redist_df = suggest_expiring_redistributions(inventory_df=fefo_df)
    print(f"expiry.py self-test: active_units={len(fefo_df)}, expiring_72h={len(exp_72)}, suggestions={len(redist_df)}")
    print("\nProjected wastage by facility:")
    print(wastage_df[["facility_name", "total_units", "expiring_72h_units", "projected_wastage_units", "wastage_pct"]].to_string(index=False))
    if not redist_df.empty:
        print("\nTop 5 expiring-unit redistributions:")
        print(redist_df[["from_name", "to_name", "blood_group", "component", "units", "min_expires_in_days", "distance_km"]].head(5).to_string(index=False))
