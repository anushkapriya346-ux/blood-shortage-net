"""Expiry-aware risk scoring and shortage alert engine.

Computes usable units (excluding stock projected to expire before consumption),
days of cover, projected stock-out dates, composite 0-100 risk scores, risk
levels (Critical/High/Watch/Safe), and human-readable alerts.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.db import load_table
from engine.forecast import forecast_demand

BLOOD_GROUPS = ["O+", "B+", "A+", "AB+", "O-", "B-", "A-", "AB-"]

RARITY_WEIGHTS: dict[str, float] = {
    "O+": 15.0,
    "B+": 20.0,
    "A+": 30.0,
    "AB+": 55.0,
    "O-": 88.0,
    "B-": 92.0,
    "A-": 92.0,
    "AB-": 100.0,
}

RISK_ORDER = {"Critical": 0, "High": 1, "Watch": 2, "Safe": 3}


def _resolve_reference_date(
    consumption_df: Optional[pd.DataFrame],
    inventory_df: Optional[pd.DataFrame],
    reference_date: Optional[date] = None,
) -> date:
    if reference_date is not None:
        return reference_date
    if consumption_df is not None and not consumption_df.empty:
        return pd.to_datetime(consumption_df["date"]).max().date()
    if inventory_df is not None and not inventory_df.empty:
        return pd.to_datetime(inventory_df["collected_date"]).max().date()
    return date.today()


def compute_expiry_aware_usable_units(
    days_left_list: list[int],
    daily_demand: float,
) -> tuple[float, float, int]:
    """Simulate continuous FEFO consumption against forecast demand.

    Returns (usable_units, projected_wastage_units, active_units_count).
    """
    active = sorted(d for d in days_left_list if d >= 1)
    n_active = len(active)
    if n_active == 0:
        return 0.0, 0.0, 0

    if daily_demand <= 1e-6:
        # Zero forecast demand: stock will sit unused until expiry
        return float(n_active), float(n_active), n_active

    time_cursor = 0.0
    usable = 0.0
    for d_exp in active:
        if time_cursor >= d_exp:
            continue
        remaining_window = d_exp - time_cursor
        consumable = min(1.0, remaining_window * daily_demand)
        usable += consumable
        time_cursor += consumable / daily_demand

    wastage = max(0.0, float(n_active) - usable)
    return round(usable, 2), round(wastage, 2), n_active


def _days_cover_score(days_of_cover: float) -> float:
    """Map days of cover to a 0-100 severity score."""
    if days_of_cover <= 0.0:
        return 100.0
    if days_of_cover < 2.0:
        return float(100.0 - 10.0 * days_of_cover)  # 80..100
    if days_of_cover < 4.0:
        return float(80.0 - 12.5 * (days_of_cover - 2.0))  # 55..80
    if days_of_cover <= 7.0:
        return float(55.0 - (25.0 / 3.0) * (days_of_cover - 4.0))  # 30..55
    return float(max(0.0, 30.0 * np.exp(-0.22 * (days_of_cover - 7.0))))


def classify_risk_level(days_of_cover: float) -> str:
    """Classify risk level based on days of cover."""
    if days_of_cover < 2.0:
        return "Critical"
    if days_of_cover < 4.0:
        return "High"
    if days_of_cover <= 7.0:
        return "Watch"
    return "Safe"


def assess_risk(
    facilities_df: Optional[pd.DataFrame] = None,
    inventory_df: Optional[pd.DataFrame] = None,
    forecast_df: Optional[pd.DataFrame] = None,
    consumption_df: Optional[pd.DataFrame] = None,
    reference_date: Optional[date] = None,
    db_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Compute expiry-aware risk metrics and alert messages per facility x blood_group."""
    if facilities_df is None:
        facilities_df = load_table("facilities", db_path=db_path)
    if inventory_df is None:
        inventory_df = load_table("inventory_units", db_path=db_path)
    if forecast_df is None:
        if consumption_df is None:
            consumption_df = load_table("daily_consumption", db_path=db_path)
        forecast_df = forecast_demand(
            consumption_df=consumption_df, facilities_df=facilities_df, db_path=db_path
        )

    ref_date = _resolve_reference_date(consumption_df, inventory_df, reference_date)

    if facilities_df.empty:
        return pd.DataFrame()

    inv = inventory_df.copy() if inventory_df is not None else pd.DataFrame()
    if not inv.empty:
        inv["days_left"] = (pd.to_datetime(inv["expiry_date"]).dt.date - ref_date).apply(
            lambda d: d.days
        )
    else:
        inv = pd.DataFrame(columns=["facility_id", "blood_group", "days_left"])

    rows: list[dict[str, Any]] = []
    for _, fac in facilities_df.iterrows():
        fac_id = fac["id"]
        fac_name = fac["name"]

        for bg in BLOOD_GROUPS:
            fc_sub = forecast_df[
                (forecast_df["facility_id"] == fac_id)
                & (forecast_df["blood_group"] == bg)
            ]
            if fc_sub.empty:
                daily_demand = 0.0
                band_width = 0.0
            else:
                daily_demand = float(fc_sub["forecast"].mean())
                band_width = float((fc_sub["upper_80"] - fc_sub["lower_80"]).mean())

            inv_sub = inv[(inv["facility_id"] == fac_id) & (inv["blood_group"] == bg)]
            days_left_list = inv_sub["days_left"].tolist() if not inv_sub.empty else []

            usable_units, projected_wastage, active_units = compute_expiry_aware_usable_units(
                days_left_list, daily_demand
            )
            expiring_72h = int(sum(1 for d in days_left_list if 1 <= d <= 3))

            if daily_demand <= 1e-6:
                days_of_cover = 30.0
                stockout_date = None
                stockout_str = "No demand"
            else:
                days_of_cover = round(usable_units / daily_demand, 2)
                days_of_cover_capped = min(days_of_cover, 60.0)
                stockout_dt = ref_date + timedelta(days=int(np.floor(days_of_cover_capped)))
                stockout_date = stockout_dt.isoformat()
                stockout_str = stockout_dt.strftime("%b %d")

            level = classify_risk_level(days_of_cover)

            doc_component = _days_cover_score(days_of_cover)
            rarity_component = RARITY_WEIGHTS.get(bg, 50.0)
            expiry_loss_ratio = (
                min(100.0, (projected_wastage / active_units) * 100.0)
                if active_units > 0
                else 0.0
            )
            volatility_component = (
                min(100.0, (band_width / max(daily_demand, 0.1)) * 35.0)
                if daily_demand > 1e-6
                else 0.0
            )

            risk_score = round(
                float(
                    np.clip(
                        0.5 * doc_component
                        + 0.2 * rarity_component
                        + 0.2 * expiry_loss_ratio
                        + 0.1 * volatility_component,
                        0.0,
                        100.0,
                    )
                ),
                1,
            )

            if level in ("Critical", "High", "Watch"):
                alert_msg = (
                    f"{level.upper()}: {bg} at {fac_name}, "
                    f"{days_of_cover:.1f} days of cover, stock-out in ~{stockout_str}"
                )
            else:
                alert_msg = ""

            rows.append(
                {
                    "facility_id": fac_id,
                    "facility_name": fac_name,
                    "facility_type": fac["type"],
                    "region": fac["region"],
                    "blood_group": bg,
                    "total_units": active_units,
                    "usable_units": usable_units,
                    "expiring_72h": expiring_72h,
                    "projected_wastage": projected_wastage,
                    "forecast_daily_demand": round(daily_demand, 2),
                    "days_of_cover": round(min(days_of_cover, 99.9), 2),
                    "projected_stockout_date": stockout_date,
                    "stockout_display": stockout_str,
                    "risk_score": risk_score,
                    "level": level,
                    "alert": alert_msg,
                }
            )

    risk_df = pd.DataFrame(rows)
    risk_df["_lvl_rank"] = risk_df["level"].map(RISK_ORDER).fillna(9)
    risk_df = (
        risk_df.sort_values(by=["_lvl_rank", "risk_score"], ascending=[True, False])
        .drop(columns=["_lvl_rank"])
        .reset_index(drop=True)
    )
    return risk_df


def get_alerts(risk_df: pd.DataFrame) -> pd.DataFrame:
    """Return active alerts sorted by severity and risk score."""
    if risk_df is None or risk_df.empty:
        return pd.DataFrame()
    alerts = risk_df[risk_df["level"].isin(["Critical", "High", "Watch"])].copy()
    alerts["_lvl_rank"] = alerts["level"].map(RISK_ORDER).fillna(9)
    alerts = (
        alerts.sort_values(by=["_lvl_rank", "risk_score"], ascending=[True, False])
        .drop(columns=["_lvl_rank"])
        .reset_index(drop=True)
    )
    return alerts


if __name__ == "__main__":
    rdf = assess_risk()
    alerts_df = get_alerts(rdf)
    print(f"risk.py self-test: {len(rdf)} facility-group pairs evaluated.")
    print("Risk level breakdown:")
    print(rdf["level"].value_counts().to_string())
    print("\nTop 8 alerts:")
    for msg in alerts_df["alert"].head(8):
        print(" -", msg)
