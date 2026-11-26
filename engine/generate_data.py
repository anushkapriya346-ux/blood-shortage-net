"""Synthetic data generator for the Patna/Bihar blood network prototype.

Generates 8 facilities, 12 months of daily consumption with weekly/yearly
seasonality and 15-20 emergency events, and unit-level inventory with deliberate
shortage, overstock, and near-expiry scenarios for demonstration.

Runnable directly as:
    python engine/generate_data.py
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.db import init_db, save_table

SEED = 42

BLOOD_GROUP_WEIGHTS: dict[str, float] = {
    "O+": 0.370,
    "B+": 0.320,
    "A+": 0.220,
    "AB+": 0.070,
    "O-": 0.010,
    "B-": 0.008,
    "A-": 0.008,
    "AB-": 0.004,
}

BLOOD_GROUPS = list(BLOOD_GROUP_WEIGHTS.keys())
COMPONENTS = ["RBC", "Platelets"]
COMPONENT_WEIGHTS = {"RBC": 0.76, "Platelets": 0.24}
SHELF_LIFE_DAYS = {"RBC": 35, "Platelets": 5}

FACILITIES_DATA = [
    {
        "id": "FAC01",
        "name": "PMCH Patna (Patna Medical College)",
        "type": "hospital",
        "lat": 25.6207,
        "lon": 85.1610,
        "region": "Central Patna",
        "capacity_factor": 1.80,
    },
    {
        "id": "FAC02",
        "name": "AIIMS Patna",
        "type": "hospital",
        "lat": 25.5621,
        "lon": 85.0419,
        "region": "Phulwari Sharif",
        "capacity_factor": 1.70,
    },
    {
        "id": "FAC03",
        "name": "IGIMS Sheikhpura",
        "type": "hospital",
        "lat": 25.6114,
        "lon": 85.0872,
        "region": "West Patna",
        "capacity_factor": 1.40,
    },
    {
        "id": "FAC04",
        "name": "NMCH Agamkuan",
        "type": "hospital",
        "lat": 25.6002,
        "lon": 85.1972,
        "region": "East Patna",
        "capacity_factor": 1.30,
    },
    {
        "id": "FAC05",
        "name": "Patna City Blood Bank",
        "type": "blood_bank",
        "lat": 25.5964,
        "lon": 85.2215,
        "region": "Patna City",
        "capacity_factor": 1.15,
    },
    {
        "id": "FAC06",
        "name": "Danapur District Hospital",
        "type": "hospital",
        "lat": 25.6368,
        "lon": 85.0462,
        "region": "Danapur",
        "capacity_factor": 0.85,
    },
    {
        "id": "FAC07",
        "name": "Bihar Red Cross Blood Bank",
        "type": "blood_bank",
        "lat": 25.6152,
        "lon": 85.1376,
        "region": "Gandhi Maidan",
        "capacity_factor": 1.25,
    },
    {
        "id": "FAC08",
        "name": "Hajipur Sadar Hospital",
        "type": "hospital",
        "lat": 25.6858,
        "lon": 85.2145,
        "region": "Vaishali",
        "capacity_factor": 0.90,
    },
]

# Day-of-week demand factors (Mon=0 .. Sun=6): higher on weekdays
WEEKLY_SEASONALITY = {
    0: 1.12,
    1: 1.15,
    2: 1.10,
    3: 1.10,
    4: 1.08,
    5: 0.88,
    6: 0.72,
}

EMERGENCY_TYPES = [
    "Mass Casualty / Highway Trauma",
    "Dengue Outbreak Cluster",
    "Festival Trauma Surge",
    "Obstetric / Surgical Emergency",
]


def _yearly_seasonality(d: date, component: str) -> float:
    """Compute Bihar seasonal demand factor (monsoon dengue spike, festival bump, summer dip)."""
    month = d.month
    # Base yearly wave peaking in Aug-Nov (monsoon/dengue + Chhath/Diwali festivals)
    if month in (8, 9, 10):
        # Monsoon / dengue season: platelets spike especially hard
        return 1.45 if component == "Platelets" else 1.25
    if month in (11, 12):
        # Post-monsoon / festival bump
        return 1.20 if component == "Platelets" else 1.16
    if month in (4, 5, 6):
        # Summer heat / lower elective volume & donation dip
        return 0.90
    return 1.02


def generate_synthetic_data(
    seed: int = SEED,
    reference_date: Optional[date] = None,
    days_history: int = 365,
    db_path: Optional[str | Path] = None,
) -> dict[str, pd.DataFrame]:
    """Generate and persist all synthetic tables to SQLite."""
    rng = np.random.default_rng(seed)
    today = reference_date or date.today()
    start_date = today - timedelta(days=days_history - 1)
    dates = [start_date + timedelta(days=i) for i in range(days_history)]

    facilities_df = pd.DataFrame(FACILITIES_DATA)

    # 1. Generate 17 emergency events (within 15-20 range), each 1-3 days with 3-5x multiplier
    n_events = 17
    event_rows: list[dict] = []
    event_lookup: dict[tuple[str, str], float] = {}

    candidate_start_indices = rng.choice(
        np.arange(10, days_history - 4), size=n_events, replace=False
    )
    for idx in sorted(candidate_start_indices):
        fac_id = str(rng.choice(facilities_df["id"].values))
        ev_type = str(rng.choice(EMERGENCY_TYPES))
        duration = int(rng.integers(1, 4))  # 1 to 3 days
        multiplier = float(np.round(rng.uniform(3.0, 5.0), 2))
        for offset in range(duration):
            ev_date = dates[int(idx) + offset].isoformat()
            event_rows.append(
                {
                    "date": ev_date,
                    "facility_id": fac_id,
                    "type": ev_type,
                    "multiplier": multiplier,
                }
            )
            key = (ev_date, fac_id)
            event_lookup[key] = max(event_lookup.get(key, 1.0), multiplier)

    emergency_df = pd.DataFrame(event_rows)

    # 2. Generate 12 months of daily consumption per facility x blood group x component
    base_total_daily = 36.0  # network-average daily units at capacity_factor=1.0
    consumption_rows: list[dict] = []

    for _, fac in facilities_df.iterrows():
        fac_id = fac["id"]
        cap = float(fac["capacity_factor"])
        # Blood banks distribute bulk demand across network, hospitals consume directly
        role_factor = 0.85 if fac["type"] == "blood_bank" else 1.05

        for d in dates:
            d_iso = d.isoformat()
            dow_factor = WEEKLY_SEASONALITY[d.weekday()]
            em_mult = event_lookup.get((d_iso, fac_id), 1.0)

            for bg, bg_w in BLOOD_GROUP_WEIGHTS.items():
                for comp in COMPONENTS:
                    comp_w = COMPONENT_WEIGHTS[comp]
                    yr_factor = _yearly_seasonality(d, comp)
                    # Floor mean slightly for rare groups so they still register occasional demand
                    lam = (
                        base_total_daily
                        * cap
                        * role_factor
                        * bg_w
                        * comp_w
                        * dow_factor
                        * yr_factor
                        * em_mult
                    )
                    lam = max(lam, 0.04)
                    units_used = int(rng.poisson(lam))
                    consumption_rows.append(
                        {
                            "date": d_iso,
                            "facility_id": fac_id,
                            "blood_group": bg,
                            "component": comp,
                            "units_used": units_used,
                        }
                    )

    consumption_df = pd.DataFrame(consumption_rows)

    # 3. Generate current inventory_units with deliberate demo scenarios
    # Critical shortages: FAC01 (O-, AB-), FAC02 (B-, O-), FAC06 (O-, A-)
    # Overstocked + near-expiry (<= 72h): FAC05 (Patna City BB), FAC07 (Bihar Red Cross BB)
    critical_short_pairs = {
        ("FAC01", "O-"): 1,
        ("FAC01", "AB-"): 0,
        ("FAC02", "B-"): 1,
        ("FAC02", "O-"): 1,
        ("FAC06", "O-"): 0,
        ("FAC06", "A-"): 1,
    }
    overstocked_facilities = {"FAC05", "FAC07"}

    # Recent 28-day mean daily consumption per facility x blood_group x component
    cutoff_28 = (today - timedelta(days=27)).isoformat()
    recent_cons = (
        consumption_df[consumption_df["date"] >= cutoff_28]
        .groupby(["facility_id", "blood_group", "component"])["units_used"]
        .mean()
        .to_dict()
    )

    inventory_rows: list[dict] = []
    unit_counter = 1

    def _add_unit(fac_id: str, bg: str, comp: str, expires_in_days: int) -> None:
        nonlocal unit_counter
        shelf = SHELF_LIFE_DAYS[comp]
        # Clamp expires_in_days to [1, shelf]
        exp_days = max(1, min(shelf, int(expires_in_days)))
        exp_date = today + timedelta(days=exp_days)
        coll_date = exp_date - timedelta(days=shelf)
        inventory_rows.append(
            {
                "id": f"UNIT-{unit_counter:05d}",
                "facility_id": fac_id,
                "blood_group": bg,
                "component": comp,
                "collected_date": coll_date.isoformat(),
                "expiry_date": exp_date.isoformat(),
            }
        )
        unit_counter += 1

    for _, fac in facilities_df.iterrows():
        fac_id = fac["id"]
        for bg in BLOOD_GROUPS:
            for comp in COMPONENTS:
                shelf = SHELF_LIFE_DAYS[comp]
                mean_daily = float(recent_cons.get((fac_id, bg, comp), 0.2))
                # Ensure rare groups at non-critical sites have enough to operate
                effective_daily = max(mean_daily, 0.35 if bg.endswith("-") else 0.6)

                if (fac_id, bg) in critical_short_pairs:
                    # Deliberate critical shortage (<2 days cover, concentrated in RBC)
                    target_units = critical_short_pairs[(fac_id, bg)] if comp == "RBC" else 0
                    for _ in range(target_units):
                        _add_unit(fac_id, bg, comp, int(rng.integers(5, shelf + 1)))
                elif fac_id in overstocked_facilities:
                    # Overstocked: 14-20 days of cover + deliberate <=72h expiring units
                    base_cover_days = int(rng.integers(14, 21))
                    target_units = max(4 if bg.endswith("-") else 8, int(round(effective_daily * base_cover_days)))
                    # Guarantee generous O-, B-, A-, AB- surplus at blood banks for transfers
                    if bg in ("O-", "B-", "A-", "AB-") and comp == "RBC":
                        target_units = max(target_units, 10 if bg == "O-" else 7)

                    # Put ~30-35% of units expiring within 1-3 days (<= 72 hours)
                    near_exp_count = max(2, int(round(target_units * 0.30)))
                    if comp == "Platelets":
                        near_exp_count = max(2, int(round(target_units * 0.45)))

                    for i in range(target_units):
                        if i < near_exp_count:
                            exp_in = int(rng.integers(1, 4))  # 1, 2, or 3 days (<=72h)
                        else:
                            low = 4 if shelf > 4 else 2
                            exp_in = int(rng.integers(low, shelf + 1))
                        _add_unit(fac_id, bg, comp, exp_in)
                else:
                    # Normal hospitals: 5-9 days cover, mostly healthy shelf life, occasional near-expiry
                    cover_days = float(rng.uniform(5.0, 9.0))
                    # Make one watch/high case at FAC03 (IGIMS) and FAC04 (NMCH) for O-/B+ variety
                    if fac_id == "FAC03" and bg == "O-":
                        cover_days = 2.6
                    elif fac_id == "FAC04" and bg == "AB-":
                        cover_days = 3.0
                    target_units = max(1 if bg.endswith("-") else 3, int(round(effective_daily * cover_days)))
                    for i in range(target_units):
                        if comp == "RBC":
                            exp_in = int(rng.integers(4, shelf + 1))
                        else:
                            exp_in = int(rng.integers(2, shelf + 1))
                        _add_unit(fac_id, bg, comp, exp_in)

    inventory_df = pd.DataFrame(inventory_rows)

    init_db(db_path)
    save_table(facilities_df, "facilities", db_path=db_path)
    save_table(inventory_df, "inventory_units", db_path=db_path)
    save_table(consumption_df, "daily_consumption", db_path=db_path)
    save_table(emergency_df, "emergency_events", db_path=db_path)

    return {
        "facilities": facilities_df,
        "inventory_units": inventory_df,
        "daily_consumption": consumption_df,
        "emergency_events": emergency_df,
    }


def print_summary(tables: dict[str, pd.DataFrame], reference_date: Optional[date] = None) -> None:
    """Print concise summary statistics of the generated dataset."""
    today = reference_date or date.today()
    fac = tables["facilities"]
    inv = tables["inventory_units"]
    cons = tables["daily_consumption"]
    emerg = tables["emergency_events"]

    exp_dates = pd.to_datetime(inv["expiry_date"]).dt.date
    exp_72h = int(((exp_dates - today).apply(lambda d: d.days) <= 3).sum())

    print("=== Synthetic Blood Network Data Summary ===")
    print(f"Facilities: {len(fac)} ({(fac['type']=='hospital').sum()} hospitals, {(fac['type']=='blood_bank').sum()} blood banks)")
    print(f"Daily consumption records: {len(cons):,} ({cons['date'].min()} to {cons['date'].max()})")
    print(f"Total historical units consumed: {int(cons['units_used'].sum()):,}")
    print(f"Emergency event-days logged: {len(emerg)} (mean multiplier: {emerg['multiplier'].mean():.2f}x)")
    print(f"Current inventory units: {len(inv):,} (RBC: {(inv['component']=='RBC').sum():,}, Platelets: {(inv['component']=='Platelets').sum():,})")
    print(f"Units expiring within 72h: {exp_72h}")

    print("\nInventory by blood group:")
    bg_counts = inv.groupby("blood_group")["id"].count().reindex(BLOOD_GROUPS, fill_value=0)
    print(bg_counts.to_string())

    print("\nRare group inventory across facilities (RBC + Platelets):")
    rare_pivot = (
        inv[inv["blood_group"].isin(["O-", "B-", "A-", "AB-"])]
        .pivot_table(index="facility_id", columns="blood_group", values="id", aggfunc="count", fill_value=0)
        .reindex(fac["id"], fill_value=0)
    )
    print(rare_pivot.to_string())


if __name__ == "__main__":
    generated = generate_synthetic_data(seed=SEED)
    print_summary(generated)
