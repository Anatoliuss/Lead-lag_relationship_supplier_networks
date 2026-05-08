"""Load events from Excel, determine reaction windows, measure primary CAR."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backtest.config import (
    FABLESS_PRIMARIES,
    FOUNDRY_PRIMARIES,
    WFE_PRIMARIES,
    ALL_PRIMARIES,
    EVENTS_FILE,
    EVENT_TIMING,
    MATERIALITY_CAR_2H,
    MATERIALITY_CAR_1D,
    MATERIALITY_VOLUME_RATIO,
    SECTOR_ETF,
    VOLUME_LOOKBACK_DAYS,
    BARS_PER_DAY,
)
from backtest.data_loader import _get_trading_days

log = logging.getLogger(__name__)


# ── Data class ─────────────────────────────────────────────────────────────

@dataclass
class EventResult:
    event_date:      pd.Timestamp
    event_hour:      Optional[pd.Timestamp]
    primary_ticker:  str
    event_type:      str
    event_description: str
    direction:       str          # "up" / "down" / "mixed"
    reaction_start:  pd.Timestamp

    # Cumulative abnormal returns of the primary
    car_1h:  float = np.nan
    car_2h:  float = np.nan
    car_3h:  float = np.nan
    car_4h:  float = np.nan
    car_1d:  float = np.nan
    car_2d:  float = np.nan
    car_5d:  float = np.nan

    volume_ratio: float = np.nan
    passed_materiality: bool = False

    raw: dict = field(default_factory=dict)   # original Excel row


# ── Excel loader ───────────────────────────────────────────────────────────

def load_events(events_file: Path = EVENTS_FILE) -> pd.DataFrame:
    """
    Load and clean the events Excel file.

    Returns a DataFrame with columns:
      date, primary, event_type, description, direction, primary_move,
      key_dependents, expected_lag_signal
    """
    # The workbook has two sheets: "Dependency Map" and "Event Log".
    # Events live on the Event Log sheet.
    xls = pd.ExcelFile(events_file, engine="openpyxl")
    sheet = next(
        (s for s in xls.sheet_names if "event" in s.lower()),
        xls.sheet_names[-1],
    )
    df = pd.read_excel(xls, sheet_name=sheet)

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Rename to expected names (adjust if your Excel has different column names)
    rename = {
        "date":               "date",
        "primary":            "primary",
        "event_type":         "event_type",
        "event_description":  "description",
        "direction":          "direction",
        "primary_move":       "primary_move",
        "key_dependents":     "key_dependents",
        "expected_lag_signal":"expected_lag_signal",
    }
    existing = {k: v for k, v in rename.items() if k in df.columns}
    df = df.rename(columns=existing)

    # Drop section headers (Primary is NaN or starts with "ALL")
    df = df.dropna(subset=["primary"])
    df = df[~df["primary"].astype(str).str.startswith("ALL")]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def expand_sector_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand rows where primary is a group token into one row per primary ticker.
    Supported tokens: ALL (all primaries), ALL-FAB (fabless), ALL-IDM (foundry/IDM),
    ALL-WFE (equipment). Legacy tokens ALL-E / ALL-D map to fabless / foundry.
    """
    group_map = {
        "ALL":     ALL_PRIMARIES,
        "ALL-FAB": FABLESS_PRIMARIES,
        "ALL-IDM": FOUNDRY_PRIMARIES,
        "ALL-WFE": WFE_PRIMARIES,
        "ALL-E":   FABLESS_PRIMARIES,    # legacy
        "ALL-D":   FOUNDRY_PRIMARIES,    # legacy
    }

    expanded_rows = []
    regular_rows  = []

    for _, row in df.iterrows():
        primary = str(row.get("primary", "")).strip().upper()
        if primary in group_map:
            for p in group_map[primary]:
                r = row.copy(); r["primary"] = p
                expanded_rows.append(r)
        else:
            regular_rows.append(row)

    all_rows = regular_rows + expanded_rows
    result = pd.DataFrame(all_rows).sort_values("date").reset_index(drop=True)
    return result


# ── Reaction-start logic ───────────────────────────────────────────────────

def get_reaction_start(
    event_date: pd.Timestamp,
    event_type: str,
    hourly_index: pd.DatetimeIndex,
) -> pd.Timestamp:
    """
    Determine the first hourly bar at which the primary's reaction is observable.

    after_close  → 09:30 of the next trading day
    pre_market   → 09:30 of event_date (gap open)
    intraday     → 09:30 of event_date (default; no intrabar timing available)
    mixed        → 09:30 of next trading day (conservative, avoid look-ahead)
    """
    timing = EVENT_TIMING.get(event_type, "mixed")
    event_day = event_date.normalize()

    if timing == "after_close" or timing == "mixed":
        target_day = _next_trading_day(event_day, hourly_index)
    else:  # pre_market, intraday
        # Check if event_date itself is a trading day
        day_bars = hourly_index[hourly_index.normalize() == event_day]
        if len(day_bars) > 0:
            target_day = event_day
        else:
            target_day = _next_trading_day(event_day, hourly_index)

    # Find the 09:30 bar on target_day
    open_bar = pd.Timestamp(f"{target_day.date()} 09:30")
    if open_bar in hourly_index:
        return open_bar

    # Fall back: first bar on that day
    day_bars = hourly_index[hourly_index.normalize() == target_day]
    if len(day_bars) > 0:
        return day_bars[0]

    # Last resort: first bar after event_date in the index
    future = hourly_index[hourly_index > event_date]
    if len(future) > 0:
        return future[0]

    raise ValueError(f"No hourly bar found after event {event_date}")


def _next_trading_day(
    ref_day: pd.Timestamp,
    hourly_index: pd.DatetimeIndex,
) -> pd.Timestamp:
    """Return the next date that has bars in hourly_index, after ref_day."""
    future_days = sorted(set(hourly_index[hourly_index > ref_day].normalize()))
    if not future_days:
        raise ValueError(f"No trading day found after {ref_day}")
    return future_days[0]


# ── CAR measurement ────────────────────────────────────────────────────────

def measure_car(
    prices:          pd.Series,
    benchmark_prices: pd.Series,
    start_bar:       pd.Timestamp,
    n_bars:          int,
) -> float:
    """
    Cumulative Abnormal Return of a stock vs. benchmark over n_bars starting at start_bar.

    Parameters
    ----------
    prices           : hourly close prices for the stock (indexed by bar_start Timestamp)
    benchmark_prices : hourly close prices for the benchmark ETF
    start_bar        : first bar of the reaction window (exclusive — we need the bar
                       just before as the base price)
    n_bars           : number of hourly bars to include

    Returns
    -------
    float : cumulative stock return minus cumulative benchmark return
    """
    try:
        idx       = prices.index.get_loc(start_bar)
    except KeyError:
        return np.nan

    if idx == 0:
        return np.nan

    base_idx  = idx - 1
    end_idx   = min(idx + n_bars - 1, len(prices) - 1)

    p0  = prices.iloc[base_idx]
    p1  = prices.iloc[end_idx]
    b0  = benchmark_prices.reindex(prices.index).iloc[base_idx]
    b1  = benchmark_prices.reindex(prices.index).iloc[end_idx]

    if p0 <= 0 or p1 <= 0 or b0 <= 0 or b1 <= 0:
        return np.nan

    stock_ret = (p1 - p0) / p0
    bench_ret = (b1 - b0) / b0
    return float(stock_ret - bench_ret)


def measure_volume_ratio(
    volumes:    pd.Series,
    start_bar:  pd.Timestamp,
    n_event_bars: int = 2,
    lookback_days: int = VOLUME_LOOKBACK_DAYS,
) -> float:
    """
    Compare volume in the first n_event_bars after start_bar to the 20-day average
    volume during the same intraday slots.
    """
    try:
        start_idx = volumes.index.get_loc(start_bar)
    except KeyError:
        return np.nan

    event_vol = volumes.iloc[start_idx: start_idx + n_event_bars].sum()

    # Same time-of-day bars over prior lookback_days trading days
    bar_times = [volumes.index[start_idx + i].time()
                 for i in range(n_event_bars)
                 if start_idx + i < len(volumes)]

    prior = volumes[
        volumes.index < start_bar
    ]
    same_slot_vol = prior[np.isin(prior.index.time, bar_times)]

    if same_slot_vol.empty:
        return np.nan

    # Restrict to lookback_days trading days
    trading_days_prior = sorted(set(prior.index.normalize()), reverse=True)
    cutoff_day = trading_days_prior[min(lookback_days, len(trading_days_prior)) - 1]
    same_slot_vol = same_slot_vol[same_slot_vol.index.normalize() >= cutoff_day]

    if same_slot_vol.empty:
        return np.nan

    avg_vol_per_event = same_slot_vol.sum() / max(lookback_days, 1) * n_event_bars
    if avg_vol_per_event <= 0:
        return np.nan

    return float(event_vol / avg_vol_per_event)


# ── Main detector ──────────────────────────────────────────────────────────

def detect_events(
    raw_events:   pd.DataFrame,
    bar_data:  dict[str, pd.DataFrame],
    daily_data:   dict[str, pd.DataFrame],
) -> list[EventResult]:
    """
    For each event row, measure primary reaction and apply materiality filter.

    Returns list of EventResult objects (both passing and failing materiality).
    """
    results: list[EventResult] = []
    events = expand_sector_events(raw_events)

    # Build a combined hourly index from all available tickers
    all_bars: list[pd.DatetimeIndex] = [
        df.index for df in bar_data.values() if not df.empty
    ]
    if not all_bars:
        log.error("No hourly data available.")
        return results
    from functools import reduce
    global_index = reduce(lambda a, b: a.union(b), all_bars)

    for _, row in events.iterrows():
        primary      = str(row["primary"]).strip().upper()
        event_type   = str(row.get("event_type", "")).strip()
        event_date   = pd.Timestamp(row["date"])
        description  = str(row.get("description", ""))
        direction    = str(row.get("direction", "")).lower()

        if primary not in bar_data:
            log.warning("No hourly data for primary %s, skipping event %s.", primary, event_date)
            continue

        primary_prices = bar_data[primary]["close"]
        primary_vols   = bar_data[primary]["volume"]

        etf_ticker = SECTOR_ETF.get(primary, "SPY")
        if etf_ticker in bar_data:
            benchmark_prices = bar_data[etf_ticker]["close"]
        elif primary in bar_data:
            log.warning("Benchmark ETF %s not loaded, using primary raw return.", etf_ticker)
            benchmark_prices = pd.Series(1.0, index=primary_prices.index)
        else:
            continue

        try:
            reaction_start = get_reaction_start(event_date, event_type, global_index)
        except ValueError as exc:
            log.warning("Cannot determine reaction start for %s %s: %s", primary, event_date, exc)
            continue

        er = EventResult(
            event_date=event_date,
            event_hour=reaction_start,
            primary_ticker=primary,
            event_type=event_type,
            event_description=description,
            direction=direction,
            reaction_start=reaction_start,
            raw=row.to_dict(),
        )

        # Measure CARs
        for attr, n in [("car_1h", 1), ("car_2h", 2), ("car_3h", 3), ("car_4h", 4)]:
            setattr(er, attr, measure_car(primary_prices, benchmark_prices, reaction_start, n))

        # Daily CARs (1d = 7 bars, 2d = 14, 5d = 35)
        for attr, n in [("car_1d", BARS_PER_DAY), ("car_2d", 2*BARS_PER_DAY), ("car_5d", 5*BARS_PER_DAY)]:
            setattr(er, attr, measure_car(primary_prices, benchmark_prices, reaction_start, n))

        er.volume_ratio = measure_volume_ratio(primary_vols, reaction_start)

        # Materiality — LOOK-AHEAD SAFE: only uses data observable within
        # entry_delay_hours of reaction_start. car_1d/2d/5d are measured for
        # post-hoc analytics only and must NOT feed the trade decision.
        car_2h_ok = abs(er.car_2h or 0) >= MATERIALITY_CAR_2H
        vol_ok    = (er.volume_ratio or 0) >= MATERIALITY_VOLUME_RATIO
        er.passed_materiality = bool(car_2h_ok or vol_ok)

        if not er.passed_materiality:
            log.debug(
                "Event %s %s failed materiality (car_2h=%.3f, vol_ratio=%.2f).",
                primary, event_date.date(), er.car_2h or 0, er.volume_ratio or 0,
            )

        results.append(er)

    log.info(
        "Event detection complete: %d total, %d passed materiality.",
        len(results),
        sum(1 for r in results if r.passed_materiality),
    )
    return results
