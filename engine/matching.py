"""Compatibility-aware greedy blood transfer matching and allocation engine.

Implements ABO/Rh compatibility for RBC, ABO-preference matching for Platelets,
Haversine distance filtering, and greedy surplus-to-deficit allocation that
never pushes donor facilities below their safe reserve threshold.
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
from engine.expiry import get_fefo_inventory
from engine.forecast import forecast_demand
from engine.risk import RISK_ORDER, _resolve_reference_date, assess_risk

# Donor -> allowed Recipients for RBC (O- universal donor, AB+ universal recipient)
RBC_COMPAT_MAP: dict[str, list[str]] = {
    "O-": ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"],
    "O+": ["O+", "A+", "B+", "AB+"],
    "A-": ["A-", "A+", "AB-", "AB+"],
    "A+": ["A+", "AB+"],
    "B-": ["B-", "B+", "AB-", "AB+"],
    "B+": ["B+", "AB+"],
    "AB-": ["AB-", "AB+"],
    "AB+": ["AB+"],
}

# Recipient <- ordered list of compatible Donor groups for RBC (exact match first,
# then least-universal donors before O- so universal O- is preserved!)
RBC_DONOR_PREFERENCE: dict[str, list[str]] = {
    "O-": ["O-"],
    "O+": ["O+", "O-"],
    "A-": ["A-", "O-"],
    "A+": ["A+", "A-", "O+", "O-"],
    "B-": ["B-", "O-"],
    "B+": ["B+", "B-", "O+", "O-"],
    "AB-": ["AB-", "A-", "B-", "O-"],
    "AB+": ["AB+", "AB-", "A+", "A-", "B+", "B-", "O+", "O-"],
}

# Platelets match by ABO preference only (Rh-agnostic within ABO group, exact first)
PLATELET_DONOR_PREFERENCE: dict[str, list[str]] = {
    "O-": ["O-", "O+"],
    "O+": ["O+", "O-"],
    "A-": ["A-", "A+"],
    "A+": ["A+", "A-"],
    "B-": ["B-", "B+"],
    "B+": ["B+", "B-"],
    "AB-": ["AB-", "AB+"],
    "AB+": ["AB+", "AB-"],
}

SAFE_COVER_DAYS = 7.0
TARGET_COVER_DAYS = 5.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two coordinates in kilometers."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return round(2.0 * r * math.asin(min(1.0, math.sqrt(a))), 2)


def is_compatible(donor_group: str, recipient_group: str, component: str = "RBC") -> bool:
    """Check clinical compatibility for a given component."""
    if component == "Platelets":
        return donor_group in PLATELET_DONOR_PREFERENCE.get(recipient_group, [])
    return recipient_group in RBC_COMPAT_MAP.get(donor_group, [])


def recommend_transfers(
    facilities_df: Optional[pd.DataFrame] = None,
    inventory_df: Optional[pd.DataFrame] = None,
    forecast_df: Optional[pd.DataFrame] = None,
    consumption_df: Optional[pd.DataFrame] = None,
    risk_df: Optional[pd.DataFrame] = None,
    max_radius_km: float = 150.0,
    safe_cover_days: float = SAFE_COVER_DAYS,
    reference_date: Optional[date] = None,
    db_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Compute ranked greedy transfer recommendations across the hospital network.

    Priority order:
      1. Critical deficits first (<2d cover), then High (<4d), then Watch (<5d target)
      2. Exact blood-group matches before compatible substitutions (never waste O-
         when exact stock exists)
      3. Near-expiry donor surplus first
      4. Nearest Haversine distance within `max_radius_km`
    Never draws a donor below `safe_cover_days` of its own demand.
    """
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
    if risk_df is None:
        risk_df = assess_risk(
            facilities_df=facilities_df,
            inventory_df=inventory_df,
            forecast_df=forecast_df,
            consumption_df=consumption_df,
            reference_date=reference_date,
        )

    fefo = get_fefo_inventory(
        inventory_df=inventory_df,
        facilities_df=facilities_df,
        consumption_df=consumption_df,
        reference_date=reference_date,
    )
    if facilities_df.empty or fefo.empty or risk_df.empty:
        return pd.DataFrame(
            columns=[
                "from",
                "to",
                "from_id",
                "to_id",
                "group",
                "recipient_group",
                "component",
                "units",
                "distance_km",
                "expires_in_days",
                "reason",
                "sentence",
                "priority_score",
            ]
        )

    fac_info = facilities_df.set_index("id").to_dict(orient="index")

    # Split facility x blood_group daily demand into RBC / Platelets share
    comp_share: dict[tuple[str, str, str], float] = {}
    if not consumption_df.empty:
        grp_comp = (
            consumption_df.groupby(["facility_id", "blood_group", "component"])["units_used"]
            .sum()
            .to_dict()
        )
        for fac_id in facilities_df["id"]:
            for bg in ["O+", "B+", "A+", "AB+", "O-", "B-", "A-", "AB-"]:
                rbc_c = grp_comp.get((fac_id, bg, "RBC"), 0)
                plt_c = grp_comp.get((fac_id, bg, "Platelets"), 0)
                tot = rbc_c + plt_c
                if tot > 0:
                    comp_share[(fac_id, bg, "RBC")] = rbc_c / tot
                    comp_share[(fac_id, bg, "Platelets")] = plt_c / tot
                else:
                    comp_share[(fac_id, bg, "RBC")] = 0.76
                    comp_share[(fac_id, bg, "Platelets")] = 0.24

    # Track remaining donor unit lists (FEFO-sorted) per (facility_id, blood_group, component)
    donor_queues: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for (fac_id, bg, comp), grp_df in fefo.groupby(["facility_id", "blood_group", "component"]):
        donor_queues[(fac_id, bg, comp)] = grp_df.to_dict(orient="records")

    # Compute donor safe reserve units per (facility_id, blood_group, component)
    daily_demand_map = {
        (r["facility_id"], r["blood_group"]): float(r["forecast_daily_demand"])
        for _, r in risk_df.iterrows()
    }

    def _donor_surplus_count(fac_id: str, bg: str, comp: str) -> int:
        queue = donor_queues.get((fac_id, bg, comp), [])
        if not queue:
            return 0
        dem_total = daily_demand_map.get((fac_id, bg), 0.0)
        share = comp_share.get((fac_id, bg, comp), 0.76 if comp == "RBC" else 0.24)
        dem_comp = dem_total * share
        # Reserve required to keep donor >= safe_cover_days (minimum 1 unit if active demand)
        reserve = int(math.ceil(dem_comp * safe_cover_days)) if dem_comp > 0.02 else 1
        if bg == "O-" and comp == "RBC":
            reserve = max(reserve, 2)
        return max(0, len(queue) - reserve)

    # Build deficit list prioritized by Critical -> High -> Watch (< TARGET_COVER_DAYS)
    deficits: list[dict[str, Any]] = []
    for _, row in risk_df.iterrows():
        if row["level"] not in ("Critical", "High", "Watch"):
            continue
        fac_id = row["facility_id"]
        bg = row["blood_group"]
        dem_total = float(row["forecast_daily_demand"])
        if dem_total <= 1e-6:
            continue

        # Target cover: bring Critical/High/Watch up to TARGET_COVER_DAYS (5 days)
        target_units = max(1, int(math.ceil(dem_total * TARGET_COVER_DAYS)))
        needed_total = max(0, target_units - int(round(float(row["usable_units"]))))
        if needed_total <= 0 and row["level"] in ("Critical", "High"):
            needed_total = max(1, int(math.ceil(dem_total * 3.0)))
        if needed_total <= 0:
            continue

        # Allocate RBC deficit primarily (and Platelets if low)
        for comp in ("RBC", "Platelets"):
            share = comp_share.get((fac_id, bg, comp), 0.76 if comp == "RBC" else 0.24)
            comp_need = max(1 if (comp == "RBC" and row["level"] == "Critical") else 0, int(round(needed_total * share)))
            if comp_need <= 0:
                continue
            deficits.append(
                {
                    "to_id": fac_id,
                    "to_name": row["facility_name"],
                    "recipient_group": bg,
                    "component": comp,
                    "need": comp_need,
                    "level": row["level"],
                    "days_of_cover": float(row["days_of_cover"]),
                    "risk_score": float(row["risk_score"]),
                }
            )

    deficits.sort(
        key=lambda d: (
            RISK_ORDER.get(d["level"], 9),
            -d["risk_score"],
            d["days_of_cover"],
            0 if d["component"] == "RBC" else 1,
        )
    )

    recommendations: list[dict[str, Any]] = []

    for deficit in deficits:
        to_id = deficit["to_id"]
        rec_bg = deficit["recipient_group"]
        comp = deficit["component"]
        remaining_need = deficit["need"]
        to_lat = float(fac_info[to_id]["lat"])
        to_lon = float(fac_info[to_id]["lon"])

        pref_donors = (
            PLATELET_DONOR_PREFERENCE.get(rec_bg, [rec_bg])
            if comp == "Platelets"
            else RBC_DONOR_PREFERENCE.get(rec_bg, [rec_bg])
        )

        # Two-pass donor search: pass 0 = exact blood group only across all donors;
        # pass 1 = compatible substitutions (never waste O- if exact stock exists!)
        for match_pass in (0, 1):
            if remaining_need <= 0:
                break
            candidate_groups = [rec_bg] if match_pass == 0 else [g for g in pref_donors if g != rec_bg]

            for donor_bg in candidate_groups:
                if remaining_need <= 0:
                    break
                # Rank eligible donors by: near-expiry surplus first, then nearest distance
                eligible_donors = []
                for src_id, src_meta in fac_info.items():
                    if src_id == to_id:
                        continue
                    avail = _donor_surplus_count(src_id, donor_bg, comp)
                    if avail <= 0:
                        continue
                    dist = haversine_km(src_meta["lat"], src_meta["lon"], to_lat, to_lon)
                    if dist > max_radius_km:
                        continue
                    queue = donor_queues[(src_id, donor_bg, comp)]
                    min_exp = int(queue[0]["expires_in_days"])
                    near_exp_flag = 0 if min_exp <= 3 else 1
                    eligible_donors.append((near_exp_flag, min_exp, dist, -avail, src_id))

                eligible_donors.sort()

                for _, min_exp, dist, _, src_id in eligible_donors:
                    if remaining_need <= 0:
                        break
                    avail = _donor_surplus_count(src_id, donor_bg, comp)
                    if avail <= 0:
                        continue
                    transfer_n = min(remaining_need, avail)
                    queue = donor_queues[(src_id, donor_bg, comp)]
                    # Pop FEFO earliest-expiring units
                    transferred_units = queue[:transfer_n]
                    donor_queues[(src_id, donor_bg, comp)] = queue[transfer_n:]
                    remaining_need -= transfer_n

                    earliest_exp = int(transferred_units[0]["expires_in_days"])
                    src_name = fac_info[src_id]["name"]
                    to_name = deficit["to_name"]
                    match_note = (
                        "exact match"
                        if donor_bg == rec_bg
                        else f"compatible match for {rec_bg}"
                    )
                    expiry_note = (
                        f"near-expiry surplus ({earliest_exp}d left)"
                        if earliest_exp <= 3
                        else f"expires in {earliest_exp}d"
                    )
                    reason = (
                        f"{deficit['level']} deficit ({deficit['days_of_cover']:.1f}d cover) — "
                        f"{match_note}, {expiry_note}"
                    )
                    sentence = (
                        f"Transfer {transfer_n} unit{'s' if transfer_n != 1 else ''} "
                        f"{donor_bg} {comp} from {src_name} to {to_name} "
                        f"({dist:.1f} km, expires in {earliest_exp} day{'s' if earliest_exp != 1 else ''})."
                    )

                    prio = round(
                        (100 - RISK_ORDER.get(deficit["level"], 3) * 25)
                        + deficit["risk_score"] * 0.3
                        + (15.0 if earliest_exp <= 3 else 0.0)
                        - dist * 0.1,
                        1,
                    )

                    recommendations.append(
                        {
                            "from": src_name,
                            "to": to_name,
                            "from_id": src_id,
                            "to_id": to_id,
                            "group": donor_bg,
                            "recipient_group": rec_bg,
                            "component": comp,
                            "units": int(transfer_n),
                            "distance_km": round(dist, 1),
                            "expires_in_days": earliest_exp,
                            "deficit_level": deficit["level"],
                            "reason": reason,
                            "sentence": sentence,
                            "priority_score": prio,
                        }
                    )

    rec_df = pd.DataFrame(recommendations)
    if not rec_df.empty:
        rec_df = rec_df.sort_values(by="priority_score", ascending=False).reset_index(drop=True)
    return rec_df


def suggest_donation_drives(risk_df: pd.DataFrame, top_n: int = 4) -> pd.DataFrame:
    """Recommend targeted blood donation drives for the highest-risk groups and regions."""
    if risk_df is None or risk_df.empty:
        return pd.DataFrame()
    at_risk = risk_df[risk_df["level"].isin(["Critical", "High", "Watch"])].copy()
    if at_risk.empty:
        return pd.DataFrame()

    grouped = (
        at_risk.groupby(["region", "blood_group"], as_index=False)
        .agg(
            facilities_affected=("facility_id", "nunique"),
            avg_risk_score=("risk_score", "mean"),
            min_cover_days=("days_of_cover", "min"),
            daily_network_demand=("forecast_daily_demand", "sum"),
        )
        .sort_values(by=["avg_risk_score", "facilities_affected"], ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    grouped["recommended_target_units"] = (
        grouped["daily_network_demand"] * 7.0
    ).apply(lambda x: max(10, int(math.ceil(x / 5.0) * 5)))
    grouped["avg_risk_score"] = grouped["avg_risk_score"].round(1)
    grouped["min_cover_days"] = grouped["min_cover_days"].round(1)
    grouped["campaign_focus"] = grouped.apply(
        lambda r: (
            f"Mobile donor camp in {r['region']} targeting {r['blood_group']} donors "
            f"(+{int(r['recommended_target_units'])} units)"
        ),
        axis=1,
    )
    return grouped


if __name__ == "__main__":
    recs = recommend_transfers()
    drives = suggest_donation_drives(assess_risk())
    print(f"matching.py self-test: {len(recs)} transfer recommendations generated.")
    if not recs.empty:
        for s in recs["sentence"].head(8):
            print(" -", s)
    if not drives.empty:
        print("\nTop donation drive suggestions:")
        for c in drives["campaign_focus"]:
            print(" -", c)
