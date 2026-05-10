"""Beta estimation, expected-reaction models, and underreaction signal scoring."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from backtest.config import (
    BETA_LOOKBACK_DAYS,
    DEPENDENCY_MAP,
    HARD_TO_BORROW,
    ILLIQUID_HOURLY_THRESHOLD,
    MIN_EXPECTED_RETURN,
    TICKER_IPO_DATES,
    UNDERREACTION_THRESHOLD,
    ZERO_VOLUME_SKIP_BARS,
    BARS_PER_DAY,
)
from backtest.event_detector import EventResult, measure_car

log = logging.getLogger(__name__)


@dataclass
class Signal:
    event_date:       pd.Timestamp
    reaction_start:   pd.Timestamp
    entry_bar:        pd.Timestamp
    primary_ticker:   str
    dependent_ticker: str
    event_type:       str

    beta:       float
    r_squared:  float
    revenue_pct: float

    expected_return_model_a: float    # β × CAR_primary
    expected_return_model_b: float    # β × CAR_primary × revenue_pct
    actual_return: float

    score_model_a: float
    score_model_b: float

    # "long" if primary went up (dep should rise to catch up), "short" otherwise
    signal_direction: str
    primary_car_at_entry: float

    dependent_avg_hourly_volume: float
    is_liquid: bool
    is_hard_to_borrow: bool

    scores_by_horizon: dict[str, float] = field(default_factory=dict)
    actual_returns_by_horizon: dict[str, float] = field(default_factory=dict)

    # Only populated when model == "vol_based"
    volatility: float = 0.0
    expected_return_vol: float = 0.0
    score_model_vol: float = 0.0

    skipped: bool = False
    skip_reason: str = ""


def compute_beta(
    daily_data:   dict[str, pd.DataFrame],
    dep_ticker:   str,
    primary_ticker: str,
    as_of_date:   pd.Timestamp,
    lookback:     int = BETA_LOOKBACK_DAYS,
) -> tuple[float, float]:
    """OLS slope of dep daily returns on primary daily returns, before as_of_date."""
    if dep_ticker not in daily_data or primary_ticker not in daily_data:
        return 1.0, 0.0

    dep_df  = daily_data[dep_ticker]
    prim_df = daily_data[primary_ticker]

    dep_ret  = dep_df.loc[dep_df.index < as_of_date, "ret"].dropna()
    prim_ret = prim_df.loc[prim_df.index < as_of_date, "ret"].dropna()
    combined = pd.DataFrame({"dep": dep_ret, "prim": prim_ret}).dropna().tail(lookback)

    if len(combined) < 30:
        log.debug("Insufficient data for beta: %s vs %s", dep_ticker, primary_ticker)
        return 1.0, 0.0

    X = sm.add_constant(combined["prim"])
    y = combined["dep"]

    try:
        model = sm.OLS(y, X).fit()
        beta  = float(model.params.get("prim", 1.0))
        r2    = float(model.rsquared)
    except Exception as exc:
        log.warning("OLS failed for %s vs %s: %s", dep_ticker, primary_ticker, exc)
        return 1.0, 0.0

    return beta, r2


def _avg_hourly_volume(
    bar_data: dict[str, pd.DataFrame],
    ticker: str,
    start_bar: pd.Timestamp,
    n_bars: int = 4,
) -> float:
    if ticker not in bar_data:
        return 0.0
    df = bar_data[ticker]
    try:
        idx  = df.index.get_loc(start_bar)
    except KeyError:
        return 0.0
    chunk = df.iloc[idx: idx + n_bars]["volume"]
    return float(chunk.mean())


def _has_zero_volume_bars(
    bar_data: dict[str, pd.DataFrame],
    ticker: str,
    start_bar: pd.Timestamp,
    n_bars: int = ZERO_VOLUME_SKIP_BARS,
) -> bool:
    if ticker not in bar_data:
        return True
    df = bar_data[ticker]
    try:
        idx  = df.index.get_loc(start_bar)
    except KeyError:
        return True
    chunk = df.iloc[idx: idx + n_bars]["volume"]
    return bool((chunk == 0).any())


def get_entry_bar(
    reaction_start: pd.Timestamp,
    entry_delay_hours: int,
    index: pd.DatetimeIndex,
) -> Optional[pd.Timestamp]:
    """Bar that is entry_delay_hours bars after reaction_start (positional, not clock)."""
    try:
        start_pos = index.get_loc(reaction_start)
    except KeyError:
        future = index[index >= reaction_start]
        if len(future) == 0:
            return None
        start_pos = index.get_loc(future[0])

    target_pos = start_pos + entry_delay_hours
    if target_pos >= len(index):
        return None
    return index[target_pos]


def _score(expected: float, actual: float) -> float:
    if abs(expected) < MIN_EXPECTED_RETURN:
        return 0.0
    return (expected - actual) / abs(expected)


def _score_vol(expected: float, actual: float, sigma: float) -> float:
    """(expected − actual) / σ, in standard-deviation units."""
    if sigma <= 0:
        return 0.0
    return (expected - actual) / sigma


def _compute_n_bar_vol(
    prices: pd.Series,
    ref_bar: pd.Timestamp,
    n_bars: int,
    lookback_days: int = 20,
) -> Optional[float]:
    """σ of n-bar cumulative returns over the last `lookback_days` trading days
    before ref_bar.  Sample = close[open + n_bars - 1] / close[open - 1] - 1."""
    if ref_bar not in prices.index or n_bars <= 0:
        return None
    try:
        ref_idx = prices.index.get_loc(ref_bar)
    except KeyError:
        return None

    prior = prices.iloc[:ref_idx]
    if prior.empty:
        return None

    # Identify each trading day's first bar in the prior window
    bar_times = pd.Series(prior.index, index=prior.index)
    first_bar_per_day = bar_times.groupby(prior.index.date).first()
    open_bars = list(first_bar_per_day.tail(lookback_days).values)

    rets: list[float] = []
    for ob in open_bars:
        try:
            i_open = prices.index.get_loc(ob)
        except KeyError:
            continue
        if i_open == 0:
            continue
        i_end = i_open + n_bars - 1
        if i_end >= len(prices):
            continue
        p0 = prices.iloc[i_open - 1]
        p1 = prices.iloc[i_end]
        if p0 > 0 and p1 > 0:
            rets.append((p1 - p0) / p0)

    if len(rets) < 5:
        return None
    return float(np.std(rets, ddof=1))


def resolve_overlapping_signals(signals: list[Signal]) -> list[Signal]:
    """Same dep on same day with conflicting directions → skip both."""
    from collections import defaultdict
    day_dep: dict[tuple, list[Signal]] = defaultdict(list)

    for sig in signals:
        key = (sig.event_date.date(), sig.dependent_ticker)
        day_dep[key].append(sig)

    for key, group in day_dep.items():
        if len(group) <= 1:
            continue
        directions = set(s.signal_direction for s in group)
        if len(directions) > 1:
            for s in group:
                s.skipped = True
                s.skip_reason = "conflicting multi-primary signal"

    return signals


def generate_signals(
    event_results:    list[EventResult],
    bar_data:      dict[str, pd.DataFrame],
    daily_data:       dict[str, pd.DataFrame],
    entry_delay_hours: int = 2,
    model:            str = "beta_revenue",
    threshold:        float = UNDERREACTION_THRESHOLD,
) -> list[Signal]:
    """One Signal per (material event × dependent).
    model: "beta_only" | "beta_revenue" | "vol_based"."""
    signals: list[Signal] = []
    active_index: Optional[pd.DatetimeIndex] = None

    # Build a single sorted index across all hourly data
    all_indices = [df.index for df in bar_data.values() if not df.empty]
    if all_indices:
        active_index = all_indices[0]
        for idx in all_indices[1:]:
            active_index = active_index.union(idx)

    for event in event_results:
        if not event.passed_materiality:
            continue

        primary = event.primary_ticker
        deps    = DEPENDENCY_MAP.get(primary, [])

        if active_index is None:
            continue

        entry_bar = get_entry_bar(event.reaction_start, entry_delay_hours, active_index)
        if entry_bar is None:
            log.debug("No entry bar for %s %s", primary, event.event_date.date())
            continue

        # Primary CAR at entry bar
        if primary not in bar_data:
            continue
        primary_prices    = bar_data[primary]["close"]
        etf_ticker_key    = primary  # will get benchmark inside measure_car
        from backtest.config import SECTOR_ETF
        etf_ticker        = SECTOR_ETF.get(primary, "SPY")
        if etf_ticker in bar_data:
            benchmark_prices = bar_data[etf_ticker]["close"]
        else:
            benchmark_prices = pd.Series(1.0, index=primary_prices.index)

        primary_car_entry = measure_car(
            primary_prices, benchmark_prices,
            event.reaction_start,
            entry_delay_hours,
        )
        if primary_car_entry is None or np.isnan(primary_car_entry):
            continue

        for dep_info in deps:
            dep_ticker  = dep_info["ticker"]
            revenue_pct = dep_info["pct"]

            # Check IPO date
            ipo_str = TICKER_IPO_DATES.get(dep_ticker)
            if ipo_str and event.event_date < pd.Timestamp(ipo_str):
                log.debug("Skipping %s (pre-IPO).", dep_ticker)
                continue

            if dep_ticker not in bar_data:
                sig = _make_skipped(event, dep_info, primary_car_entry, entry_bar, "no hourly data")
                signals.append(sig)
                continue

            dep_prices = bar_data[dep_ticker]["close"]

            # Liquidity check
            if _has_zero_volume_bars(bar_data, dep_ticker, event.reaction_start):
                sig = _make_skipped(event, dep_info, primary_car_entry, entry_bar, "zero volume bars")
                signals.append(sig)
                continue

            avg_hvol = _avg_hourly_volume(bar_data, dep_ticker, event.reaction_start)

            # Beta
            beta, r2 = compute_beta(
                daily_data, dep_ticker, primary, event.event_date
            )

            # Expected returns
            expected_a = primary_car_entry * beta
            expected_b = primary_car_entry * beta * revenue_pct

            # Actual dependent CAR at entry_bar
            if etf_ticker in bar_data:
                dep_bench = bar_data[etf_ticker]["close"]
            else:
                dep_bench = pd.Series(1.0, index=dep_prices.index)

            actual_dep = measure_car(
                dep_prices, dep_bench,
                event.reaction_start,
                entry_delay_hours,
            )
            if actual_dep is None or np.isnan(actual_dep):
                sig = _make_skipped(event, dep_info, primary_car_entry, entry_bar, "no dependent CAR")
                signals.append(sig)
                continue

            score_a = _score(expected_a, actual_dep)
            score_b = _score(expected_b, actual_dep)

            # Vol-based model: σ of dep's n-bar return over a 20-day lookback.
            # expected_vol = sign(primary_CAR) × σ; score_vol = (expected - actual) / σ.
            sigma = _compute_n_bar_vol(
                dep_prices, event.reaction_start, entry_delay_hours, lookback_days=20
            )
            if sigma is None or sigma <= 0:
                expected_vol = 0.0
                score_vol = 0.0
            else:
                direction_sign = 1.0 if primary_car_entry >= 0 else -1.0
                expected_vol = direction_sign * sigma
                score_vol = _score_vol(expected_vol, actual_dep, sigma)

            if model == "beta_revenue":
                score = score_b
            elif model == "vol_based":
                score = score_vol
            else:
                score = score_a

            if abs(score) < threshold:
                sig = _make_skipped(event, dep_info, primary_car_entry, entry_bar, "below threshold")
                sig.score_model_a = score_a
                sig.score_model_b = score_b
                sig.score_model_vol = score_vol
                sig.volatility = sigma if sigma is not None else 0.0
                sig.expected_return_vol = expected_vol
                sig.beta           = beta
                sig.r_squared      = r2
                sig.actual_return  = actual_dep
                signals.append(sig)
                continue

            # If vol-based and σ unavailable, can't trade.
            if model == "vol_based" and (sigma is None or sigma <= 0):
                sig = _make_skipped(event, dep_info, primary_car_entry, entry_bar, "no volatility")
                signals.append(sig)
                continue

            # Direction: follow primary's direction
            direction = "long" if primary_car_entry > 0 else "short"
            # If score is negative (overreaction), reverse
            if score < 0:
                direction = "short" if direction == "long" else "long"

            is_hard = dep_ticker in HARD_TO_BORROW
            is_liquid = avg_hvol >= ILLIQUID_HOURLY_THRESHOLD

            # Compute underreaction scores at multiple horizons for analysis
            horizon_scores: dict[str, float] = {}
            horizon_returns: dict[str, float] = {}
            for n_bars in [1, 2, 3, 4, 7, 14, 21, 35, 49, 70]:
                ar = measure_car(dep_prices, dep_bench, event.reaction_start, n_bars)
                horizon_returns[f"h{n_bars}"] = ar if ar is not None else np.nan
                if ar is None or np.isnan(ar):
                    horizon_scores[f"h{n_bars}"] = np.nan
                    continue
                if model == "vol_based":
                    σ_h = _compute_n_bar_vol(dep_prices, event.reaction_start, n_bars, 20)
                    if σ_h and σ_h > 0:
                        sign_h = 1.0 if primary_car_entry >= 0 else -1.0
                        horizon_scores[f"h{n_bars}"] = _score_vol(sign_h * σ_h, ar, σ_h)
                    else:
                        horizon_scores[f"h{n_bars}"] = np.nan
                else:
                    ex = expected_b if model == "beta_revenue" else expected_a
                    horizon_scores[f"h{n_bars}"] = _score(ex, ar)

            sig = Signal(
                event_date=event.event_date,
                reaction_start=event.reaction_start,
                entry_bar=entry_bar,
                primary_ticker=primary,
                dependent_ticker=dep_ticker,
                event_type=event.event_type,
                beta=beta,
                r_squared=r2,
                revenue_pct=revenue_pct,
                expected_return_model_a=expected_a,
                expected_return_model_b=expected_b,
                actual_return=actual_dep,
                score_model_a=score_a,
                score_model_b=score_b,
                signal_direction=direction,
                primary_car_at_entry=primary_car_entry,
                dependent_avg_hourly_volume=avg_hvol,
                is_liquid=is_liquid,
                is_hard_to_borrow=is_hard,
                scores_by_horizon=horizon_scores,
                actual_returns_by_horizon=horizon_returns,
                volatility=sigma if sigma is not None else 0.0,
                expected_return_vol=expected_vol,
                score_model_vol=score_vol,
            )
            signals.append(sig)

    # Resolve overlapping multi-primary signals
    signals = resolve_overlapping_signals(signals)

    tradeable = sum(1 for s in signals if not s.skipped)
    log.info(
        "Signal generation complete: %d total, %d tradeable.",
        len(signals), tradeable,
    )
    return signals




def _make_skipped(
    event: EventResult,
    dep_info: dict,
    primary_car: float,
    entry_bar: pd.Timestamp,
    reason: str,
) -> Signal:
    return Signal(
        event_date=event.event_date,
        reaction_start=event.reaction_start,
        entry_bar=entry_bar,
        primary_ticker=event.primary_ticker,
        dependent_ticker=dep_info["ticker"],
        event_type=event.event_type,
        beta=1.0,
        r_squared=0.0,
        revenue_pct=dep_info["pct"],
        expected_return_model_a=0.0,
        expected_return_model_b=0.0,
        actual_return=np.nan,
        score_model_a=0.0,
        score_model_b=0.0,
        signal_direction="long",
        primary_car_at_entry=primary_car,
        dependent_avg_hourly_volume=0.0,
        is_liquid=False,
        is_hard_to_borrow=dep_info["ticker"] in HARD_TO_BORROW,
        skipped=True,
        skip_reason=reason,
    )
