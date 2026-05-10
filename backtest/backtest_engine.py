"""Main backtest loop: orchestrates events, signals, portfolio, and parameter sweep."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from backtest.config import (
    BACKTEST_END,
    BACKTEST_START,
    DEFAULT_ENTRY_DELAY,
    DEFAULT_HEDGE,
    DEFAULT_HOLDING_HOURS,
    DEFAULT_MODEL,
    DEFAULT_THRESHOLD,
    EXIT_CONVERGENCE_THRESHOLD,
    EXIT_END_OF_DAY,
    EXIT_MAX_HOLDING_HOURS,
    EXIT_STOP_LOSS_PCT,
    INITIAL_CAPITAL,
    MAX_CONCURRENT_POSITIONS,
    MAX_POSITION_PCT,
    SECTOR_ETF,
    SWEEP_PARAMS,
)
from backtest.data_loader import forward_fill_bars
from backtest.event_detector import EventResult, detect_events
from backtest.portfolio import Portfolio, PortfolioSnapshot
from backtest.signal_generator import Signal, generate_signals, _score

log = logging.getLogger(__name__)




@dataclass
class RunResult:
    params: dict
    portfolio: Portfolio
    signals: list[Signal]
    event_results: list[EventResult]
    hourly_snapshots: list[PortfolioSnapshot]
    daily_snapshots: list[PortfolioSnapshot]




def run_backtest(
    bar_data:       dict[str, pd.DataFrame],
    daily_data:        dict[str, pd.DataFrame],
    event_results:     list[EventResult],
    entry_delay_hours: int   = DEFAULT_ENTRY_DELAY,
    holding_hours:     int   = DEFAULT_HOLDING_HOURS,
    hedge_method:      str   = DEFAULT_HEDGE,
    signal_model:      str   = DEFAULT_MODEL,
    threshold:         float = DEFAULT_THRESHOLD,
    max_positions:     int   = MAX_CONCURRENT_POSITIONS,
    initial_capital:   float = INITIAL_CAPITAL,
    eod_exit:          bool  = EXIT_END_OF_DAY,
) -> RunResult:
    """Run one backtest. Per hourly bar: check exits, open new trades, snapshot."""
    # Fill gaps
    bar_data = forward_fill_bars(bar_data)

    portfolio = Portfolio(
        initial_capital=initial_capital,
        max_concurrent=max_positions,
        max_position_pct=MAX_POSITION_PCT,
        hedge_method=hedge_method,
    )

    # Generate all signals up-front (no look-ahead: signal uses only data ≤ entry_bar)
    signals = generate_signals(
        event_results,
        bar_data,
        daily_data,
        entry_delay_hours=entry_delay_hours,
        model=signal_model,
        threshold=threshold,
    )

    tradeable_signals = [s for s in signals if not s.skipped]
    # Sort signals by entry_bar for chronological processing
    tradeable_signals.sort(key=lambda s: s.entry_bar)

    # Build a sorted list of all unique hourly bars across all tickers
    all_bars = _build_global_bar_index(bar_data)

    hourly_snapshots: list[PortfolioSnapshot] = []
    daily_snapshots:  list[PortfolioSnapshot] = []
    prev_day: Optional[pd.Timestamp] = None

    # Pending signals queue: signals not yet opened
    signal_queue = list(tradeable_signals)
    signal_idx   = 0  # pointer into signal_queue (signals are sorted by entry_bar)

    log.info(
        "Backtest starting: %d bars, %d tradeable signals.",
        len(all_bars), len(tradeable_signals),
    )

    for bar in tqdm(all_bars, desc="Backtest", unit="bar", leave=False):
        # --- 1. Process exits ---
        # Build a quick lookup of current underreaction scores for convergence check
        score_lookup = _build_score_lookup(
            portfolio._open_trades,
            bar_data,
            daily_data,
            bar,
            signal_model,
        )
        closed = portfolio.check_exits(
            current_bar=bar,
            bar_data=bar_data,
            max_holding=holding_hours,
            eod_exit=eod_exit,
            convergence_threshold=EXIT_CONVERGENCE_THRESHOLD,
            stop_loss_pct=EXIT_STOP_LOSS_PCT,
            signal_lookup=score_lookup,
        )
        if closed:
            log.debug("Bar %s: closed %d trade(s).", bar, len(closed))

        # --- 2. Open new trades ---
        while signal_idx < len(signal_queue):
            sig = signal_queue[signal_idx]
            if sig.entry_bar > bar:
                break  # remaining signals are in the future
            if sig.entry_bar == bar:
                trade = portfolio.enter_trade(
                    signal=sig,
                    entry_bar=bar,
                    bar_data=bar_data,
                    sector_etf_map=SECTOR_ETF,
                )
                if trade is None:
                    log.debug("Could not open trade for %s at %s.", sig.dependent_ticker, bar)
            signal_idx += 1

        # --- 3. Snapshot ---
        snap = portfolio.take_snapshot(bar)
        hourly_snapshots.append(snap)

        day = bar.normalize()
        if day != prev_day:
            # With daily bars every bar is the day's only entry; always snapshot it.
            # With hourly bars only snapshot when we're at/past 15:30.
            is_last_bar_of_day = (bar.strftime("%H:%M") >= "15:30") or (bar.hour == 9 and bar.minute == 30)
            if is_last_bar_of_day:
                daily_snapshots.append(snap)
            prev_day = day

    # Force-close remaining at end
    if all_bars:
        portfolio.force_close_all(all_bars[-1], bar_data)

    log.info(
        "Backtest complete: %d trades, final NAV=%.0f.",
        len(portfolio.closed_trades), portfolio.cash,
    )

    params = dict(
        entry_delay_hours=entry_delay_hours,
        holding_hours=holding_hours,
        hedge_method=hedge_method,
        signal_model=signal_model,
        threshold=threshold,
    )
    return RunResult(
        params=params,
        portfolio=portfolio,
        signals=signals,
        event_results=event_results,
        hourly_snapshots=hourly_snapshots,
        daily_snapshots=daily_snapshots,
    )




def run_sweep(
    bar_data:   dict[str, pd.DataFrame],
    daily_data:    dict[str, pd.DataFrame],
    event_results: list[EventResult],
    sweep_params:  dict = None,
) -> pd.DataFrame:
    """Run every combination of sweep_params and return a summary DataFrame."""
    if sweep_params is None:
        sweep_params = SWEEP_PARAMS

    keys   = list(sweep_params.keys())
    values = list(sweep_params.values())
    combos = list(itertools.product(*values))

    log.info("Running parameter sweep: %d combinations.", len(combos))

    rows = []
    for combo in tqdm(combos, desc="Sweep", unit="run"):
        params = dict(zip(keys, combo))
        try:
            result = run_backtest(
                bar_data=bar_data,
                daily_data=daily_data,
                event_results=event_results,
                entry_delay_hours=params.get("entry_delay_hours", DEFAULT_ENTRY_DELAY),
                holding_hours=params.get("holding_period_hours", DEFAULT_HOLDING_HOURS),
                hedge_method=params.get("hedge_method", DEFAULT_HEDGE),
                signal_model=params.get("signal_model", DEFAULT_MODEL),
                threshold=params.get("underreaction_threshold", DEFAULT_THRESHOLD),
            )
            from backtest.analytics import compute_summary_stats
            stats = compute_summary_stats(result)
            row = {**params, **stats}
        except Exception as exc:
            log.error("Sweep run failed for %s: %s", params, exc)
            row = {**params, "sharpe": np.nan, "total_return": np.nan}
        rows.append(row)

    df = pd.DataFrame(rows)
    if "sharpe" in df.columns:
        df = df.sort_values("sharpe", ascending=False)
    return df




def _build_global_bar_index(
    bar_data: dict[str, pd.DataFrame],
) -> list[pd.Timestamp]:
    """Sorted list of all unique bar timestamps across all tickers."""
    all_ts: set[pd.Timestamp] = set()
    for df in bar_data.values():
        all_ts.update(df.index.tolist())
    return sorted(all_ts)


def _build_score_lookup(
    open_trades: dict,
    bar_data: dict[str, pd.DataFrame],
    daily_data:  dict[str, pd.DataFrame],
    current_bar: pd.Timestamp,
    model: str,
) -> dict[str, float]:
    """Current underreaction score per open position. Used for convergence exit."""
    from backtest.event_detector import measure_car
    from backtest.config import SECTOR_ETF
    scores: dict[str, float] = {}

    for dep_ticker, trade in open_trades.items():
        sig = trade.signal
        primary = sig.primary_ticker
        etf     = SECTOR_ETF.get(primary, "SPY")

        if dep_ticker not in bar_data or primary not in bar_data:
            continue

        dep_prices   = bar_data[dep_ticker]["close"]
        prim_prices  = bar_data[primary]["close"]
        bench_prices = bar_data[etf]["close"] if etf in bar_data else pd.Series(
            1.0, index=dep_prices.index
        )

        # Bars held since reaction_start
        try:
            start_pos = dep_prices.index.get_loc(sig.reaction_start)
            end_pos   = dep_prices.index.get_loc(current_bar)
            n_bars    = end_pos - start_pos
        except KeyError:
            continue

        actual_dep  = measure_car(dep_prices, bench_prices, sig.reaction_start, n_bars)
        actual_prim = measure_car(prim_prices, bench_prices, sig.reaction_start, n_bars)

        if actual_dep is None or actual_prim is None:
            continue

        if model == "vol_based":
            # σ is fixed at signal-gen time; direction follows primary's current sign
            sigma = sig.volatility
            if sigma <= 0:
                continue
            sign_now = 1.0 if actual_prim >= 0 else -1.0
            expected = sign_now * sigma
            scores[dep_ticker] = (expected - actual_dep) / sigma
        else:
            expected = (actual_prim * sig.beta * sig.revenue_pct
                        if model == "beta_revenue"
                        else actual_prim * sig.beta)
            scores[dep_ticker] = _score(expected, actual_dep)

    return scores
