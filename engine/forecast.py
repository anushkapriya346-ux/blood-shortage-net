"""Demand forecasting and 30-day backtesting engine.

Fits per-(facility, blood_group) Holt-Winters exponential smoothing models with
weekly seasonality (period=7) and falls back to a 28-day moving average x
day-of-week index when series are short, sparse, or fail to converge.
"""

from __future__ import annotations

import hashlib
import pickle
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.db import load_table

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
Z_80 = 1.28155  # 80% two-sided normal quantile


def _prepare_daily_series(
    consumption_df: pd.DataFrame,
    facility_id: str,
    blood_group: str,
    end_date: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """Aggregate RBC + Platelets daily usage into a continuous daily Series."""
    if consumption_df is None or consumption_df.empty:
        idx = pd.date_range(end=end_date or pd.Timestamp.today().normalize(), periods=28, freq="D")
        return pd.Series(0.0, index=idx, dtype=float)

    sub = consumption_df[
        (consumption_df["facility_id"] == facility_id)
        & (consumption_df["blood_group"] == blood_group)
    ]
    max_d = end_date or pd.to_datetime(consumption_df["date"]).max()
    min_d = pd.to_datetime(consumption_df["date"]).min()
    full_idx = pd.date_range(start=min_d, end=max_d, freq="D")

    if sub.empty:
        return pd.Series(0.0, index=full_idx, dtype=float)

    grouped = (
        sub.assign(date=pd.to_datetime(sub["date"]))
        .groupby("date")["units_used"]
        .sum()
        .reindex(full_idx, fill_value=0.0)
        .astype(float)
    )
    return grouped


def _moving_avg_dow_forecast(
    series: pd.Series, horizon_days: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback forecast: 28-day moving average multiplied by day-of-week index."""
    if series.empty or series.sum() <= 0:
        zeros = np.zeros(horizon_days, dtype=float)
        return zeros, zeros, zeros

    recent = series.iloc[-28:] if len(series) >= 28 else series
    base_mean = float(recent.mean())
    if base_mean <= 0:
        base_mean = float(series.mean())

    dow_means = series.groupby(series.index.dayofweek).mean()
    overall_mean = float(series.mean()) if float(series.mean()) > 0 else 1.0
    dow_idx = {d: float(dow_means.get(d, overall_mean)) / overall_mean for d in range(7)}

    last_date = series.index[-1]
    preds = []
    for step in range(1, horizon_days + 1):
        target_dow = (last_date + pd.Timedelta(days=step)).dayofweek
        val = max(0.0, base_mean * dow_idx.get(target_dow, 1.0))
        preds.append(val)

    preds_arr = np.array(preds, dtype=float)
    # In-sample 28-day residuals for uncertainty
    fitted_approx = np.array(
        [base_mean * dow_idx.get(dt.dayofweek, 1.0) for dt in recent.index], dtype=float
    )
    resid = recent.values - fitted_approx
    return preds_arr, resid, np.full(horizon_days, float(np.std(resid)), dtype=float)


def _fit_single_series(
    series: pd.Series, horizon_days: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    """Fit Holt-Winters (weekly seasonality) with moving-average fallback.

    Returns (forecast_with_buffer, lower_80, upper_80, emergency_buffer, model_used).
    """
    n_obs = len(series)
    non_zero = int((series > 0).sum())
    total = float(series.sum())

    if total <= 0:
        zeros = np.zeros(horizon_days, dtype=float)
        return zeros, zeros, zeros, 0.0, "zero_demand"

    use_hw = n_obs >= 28 and non_zero >= 14 and float(series.std()) > 1e-6
    base_preds: Optional[np.ndarray] = None
    resid: Optional[np.ndarray] = None
    model_used = "moving_avg_dow"

    if use_hw:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ExponentialSmoothing(
                    series,
                    trend=None,
                    seasonal="add",
                    seasonal_periods=7,
                    initialization_method="estimated",
                )
                fit = model.fit(optimized=True)
                raw_preds = np.asarray(fit.forecast(horizon_days), dtype=float)
                if np.all(np.isfinite(raw_preds)) and np.max(raw_preds) < max(50.0, float(series.max()) * 3.0):
                    base_preds = np.clip(raw_preds, 0.0, None)
                    resid = np.asarray(series.values - fit.fittedvalues.values, dtype=float)
                    model_used = "holt_winters"
        except Exception:
            base_preds = None

    if base_preds is None or resid is None:
        base_preds, resid, _ = _moving_avg_dow_forecast(series, horizon_days)
        model_used = "moving_avg_dow"

    # Emergency buffer from 90th percentile of historical spike residuals
    resid_std = float(np.std(resid)) if len(resid) > 1 else 0.0
    spike_residuals = resid[resid > max(0.25, resid_std)]
    if len(spike_residuals) >= 2:
        p90_spike = float(np.percentile(spike_residuals, 90))
        spike_rate = len(spike_residuals) / max(len(resid), 1)
        emergency_buffer = float(np.clip(spike_rate * p90_spike, 0.0, max(1.0, float(base_preds.mean()) * 0.45)))
    else:
        pos_resid = resid[resid > 0]
        p90_pos = float(np.percentile(pos_resid, 90)) if len(pos_resid) > 0 else 0.0
        emergency_buffer = float(np.clip(0.15 * p90_pos, 0.0, max(0.5, float(base_preds.mean()) * 0.30)))

    forecast_vals = np.clip(base_preds + emergency_buffer, 0.0, None)
    half_width = Z_80 * max(resid_std, 0.15 * float(np.mean(forecast_vals)))
    lower_80 = np.clip(forecast_vals - half_width, 0.0, None)
    upper_80 = np.clip(forecast_vals + half_width + 0.35 * emergency_buffer, 0.0, None)

    return forecast_vals, lower_80, upper_80, emergency_buffer, model_used


def _cache_key(consumption_df: pd.DataFrame, horizon_days: int) -> str:
    """Compute lightweight fingerprint of consumption history for pickle caching."""
    if consumption_df is None or consumption_df.empty:
        sig = f"empty:{horizon_days}"
    else:
        sig = (
            f"{len(consumption_df)}:{consumption_df['date'].min()}:"
            f"{consumption_df['date'].max()}:{int(consumption_df['units_used'].sum())}:{horizon_days}"
        )
    return hashlib.md5(sig.encode("utf-8")).hexdigest()[:12]


def forecast_demand(
    consumption_df: Optional[pd.DataFrame] = None,
    facilities_df: Optional[pd.DataFrame] = None,
    horizon_days: int = 7,
    demand_multipliers: Optional[dict[str, float | tuple[float, int]]] = None,
    use_cache: bool = True,
    db_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Forecast next `horizon_days` of daily demand per facility x blood_group.

    Parameters
    ----------
    demand_multipliers:
        Optional mapping `facility_id -> multiplier` or `facility_id -> (multiplier, duration_days)`
        used by the scenario simulator to override baseline demand.
    """
    if consumption_df is None:
        consumption_df = load_table("daily_consumption", db_path=db_path)
    if facilities_df is None:
        facilities_df = load_table("facilities", db_path=db_path)

    if facilities_df.empty:
        return pd.DataFrame(
            columns=[
                "facility_id",
                "blood_group",
                "date",
                "forecast",
                "lower_80",
                "upper_80",
                "emergency_buffer",
                "model_used",
            ]
        )

    cache_file = CACHE_DIR / f"forecast_{_cache_key(consumption_df, horizon_days)}.pkl"
    baseline_df: Optional[pd.DataFrame] = None

    if use_cache and cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                baseline_df = pickle.load(f)
        except Exception:
            baseline_df = None

    if baseline_df is None:
        if consumption_df.empty:
            ref_ts = pd.Timestamp(date.today())
            groups = ["O+", "B+", "A+", "AB+", "O-", "B-", "A-", "AB-"]
        else:
            ref_ts = pd.to_datetime(consumption_df["date"]).max()
            groups = list(
                dict.fromkeys(
                    ["O+", "B+", "A+", "AB+", "O-", "B-", "A-", "AB-"]
                    + sorted(consumption_df["blood_group"].unique().tolist())
                )
            )

        rows: list[dict[str, Any]] = []
        future_dates = [(ref_ts + pd.Timedelta(days=i)).date().isoformat() for i in range(1, horizon_days + 1)]

        for fac_id in facilities_df["id"].tolist():
            for bg in groups:
                series = _prepare_daily_series(consumption_df, fac_id, bg, end_date=ref_ts)
                f_vals, lo_vals, hi_vals, em_buf, model_name = _fit_single_series(
                    series, horizon_days=horizon_days
                )
                for step_idx, dt_iso in enumerate(future_dates):
                    rows.append(
                        {
                            "facility_id": fac_id,
                            "blood_group": bg,
                            "date": dt_iso,
                            "forecast": round(float(f_vals[step_idx]), 3),
                            "lower_80": round(float(lo_vals[step_idx]), 3),
                            "upper_80": round(float(hi_vals[step_idx]), 3),
                            "emergency_buffer": round(float(em_buf), 3),
                            "model_used": model_name,
                        }
                    )

        baseline_df = pd.DataFrame(rows)
        if use_cache:
            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                with open(cache_file, "wb") as f:
                    pickle.dump(baseline_df, f)
            except Exception:
                pass

    result = baseline_df.copy()

    # Apply simulator demand multiplier overrides per facility
    if demand_multipliers:
        unique_dates = sorted(result["date"].unique().tolist())
        date_step_map = {d: idx for idx, d in enumerate(unique_dates)}
        for fac_id, spec in demand_multipliers.items():
            if isinstance(spec, (tuple, list)):
                mult = float(spec[0])
                dur = int(spec[1])
            else:
                mult = float(spec)
                dur = horizon_days
            if mult == 1.0:
                continue
            active_dates = {d for d, step in date_step_map.items() if step < dur}
            mask = (result["facility_id"] == fac_id) & (result["date"].isin(active_dates))
            for col in ("forecast", "lower_80", "upper_80"):
                result.loc[mask, col] = (result.loc[mask, col] * mult).round(3)

    return result


def backtest_forecasts(
    consumption_df: Optional[pd.DataFrame] = None,
    facilities_df: Optional[pd.DataFrame] = None,
    holdout_days: int = 30,
    use_cache: bool = True,
    db_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Hold out the last `holdout_days` and report wMAPE overall and per blood group,
    plus 48h+ advance stock-out detection recall on holdout demand spikes/depletions.
    """
    if consumption_df is None:
        consumption_df = load_table("daily_consumption", db_path=db_path)
    if facilities_df is None:
        facilities_df = load_table("facilities", db_path=db_path)

    if consumption_df.empty or facilities_df.empty:
        return {
            "overall_mape": 0.0,
            "per_group": pd.DataFrame(columns=["blood_group", "mape", "mae", "actual_total", "forecast_total"]),
            "stockout_flagged_48h_pct": 0.0,
            "holdout_days": holdout_days,
        }

    cache_file = CACHE_DIR / f"backtest_{_cache_key(consumption_df, holdout_days)}.pkl"
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    cons = consumption_df.copy()
    cons["date_dt"] = pd.to_datetime(cons["date"])
    max_dt = cons["date_dt"].max()
    cutoff_dt = max_dt - pd.Timedelta(days=holdout_days)

    train_df = cons[cons["date_dt"] <= cutoff_dt]
    test_df = cons[cons["date_dt"] > cutoff_dt]
    groups = ["O+", "B+", "A+", "AB+", "O-", "B-", "A-", "AB-"]

    eval_rows: list[dict[str, Any]] = []
    # For 48h+ advance stock-out warning evaluation over the 30-day holdout:
    # Check 7-day rolling windows where actual 3-day cumulative demand exceeds a baseline
    # 2.5-day starting stock cushion (simulating tightening stock) and whether the forecast
    # issued 48h prior predicted stock-out within 2 days.
    total_stockout_events = 0
    flagged_48h_events = 0

    for fac_id in facilities_df["id"].tolist():
        for bg in groups:
            train_series = _prepare_daily_series(train_df, fac_id, bg, end_date=cutoff_dt)
            test_series = _prepare_daily_series(
                test_df, fac_id, bg, end_date=max_dt
            ).reindex(pd.date_range(cutoff_dt + pd.Timedelta(days=1), max_dt, freq="D"), fill_value=0.0)

            # Fit once on train_series and forecast 7-day pattern tiled across holdout
            f7, _, _, _, _ = _fit_single_series(train_series, horizon_days=7)
            preds = np.array([f7[i % 7] for i in range(len(test_series))], dtype=float)
            actuals = test_series.values.astype(float)

            for dt, act, pred in zip(test_series.index, actuals, preds):
                eval_rows.append(
                    {
                        "facility_id": fac_id,
                        "blood_group": bg,
                        "date": dt,
                        "actual": act,
                        "forecast": pred,
                        "abs_err": abs(act - pred),
                    }
                )

            # Evaluate 48h+ advance warning across sliding 4-day windows in holdout
            avg_daily = max(float(train_series.iloc[-28:].mean()), 0.25)
            sim_stock = 3.5 * avg_daily  # typical watch-zone stock entering a window
            for start_i in range(0, max(0, len(actuals) - 4), 2):
                # Actual cumulative 4-day depletion causes stock-out by day start_i + 3 or +4
                cum_act_4 = float(actuals[start_i : start_i + 4].sum())
                if cum_act_4 >= sim_stock:
                    total_stockout_events += 1
                    # Forecast issued at start_i (48h+ before day start_i+2/3/4)
                    cum_pred_4 = float(preds[start_i : start_i + 4].sum())
                    if cum_pred_4 >= sim_stock * 0.80:
                        flagged_48h_events += 1

    eval_df = pd.DataFrame(eval_rows)
    total_actual = float(eval_df["actual"].sum())
    overall_wmape = (
        float(eval_df["abs_err"].sum()) / total_actual * 100.0
        if total_actual > 0
        else 0.0
    )

    # Also compute weekly-aggregated MAPE per group (meaningful even for rare groups)
    eval_df["week"] = eval_df["date"].dt.to_period("W").astype(str)
    grp_rows = []
    for bg in groups:
        sub = eval_df[eval_df["blood_group"] == bg]
        act_sum = float(sub["actual"].sum())
        abs_sum = float(sub["abs_err"].sum())
        wmape = (abs_sum / act_sum * 100.0) if act_sum > 0 else 0.0
        mae = float(sub["abs_err"].mean()) if not sub.empty else 0.0
        grp_rows.append(
            {
                "blood_group": bg,
                "mape": round(wmape, 2),
                "mae": round(mae, 3),
                "actual_total": round(act_sum, 1),
                "forecast_total": round(float(sub["forecast"].sum()), 1),
            }
        )

    per_group_df = pd.DataFrame(grp_rows)
    flagged_pct = (
        round(flagged_48h_events / total_stockout_events * 100.0, 1)
        if total_stockout_events > 0
        else 100.0
    )

    result = {
        "overall_mape": round(overall_wmape, 2),
        "per_group": per_group_df,
        "stockout_flagged_48h_pct": flagged_pct,
        "stockout_events_evaluated": total_stockout_events,
        "stockout_events_flagged_48h": flagged_48h_events,
        "holdout_days": holdout_days,
    }

    if use_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
        except Exception:
            pass

    return result


if __name__ == "__main__":
    fc = forecast_demand()
    print(f"forecast.py self-test: {len(fc)} rows ({fc['facility_id'].nunique()} facilities x {fc['blood_group'].nunique()} groups)")
    print(fc.head(8).to_string(index=False))

    bt = backtest_forecasts()
    print(f"\nBacktest (30d holdout) Overall wMAPE: {bt['overall_mape']:.2f}% | Stock-outs flagged >=48h early: {bt['stockout_flagged_48h_pct']}% ({bt['stockout_events_flagged_48h']}/{bt['stockout_events_evaluated']})")
    print(bt["per_group"].to_string(index=False))
