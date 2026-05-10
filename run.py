#!/usr/bin/env python
"""Entry point: download TAQ data, detect events, run a single backtest."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backtest.config import (
    BACKTEST_END,
    BACKTEST_START,
    DEFAULT_ENTRY_DELAY,
    DEFAULT_HEDGE,
    DEFAULT_HOLDING_HOURS,
    DEFAULT_MODEL,
    DEFAULT_THRESHOLD,
    EVENTS_FILE,
    MAX_CONCURRENT_POSITIONS,
    OUTPUT_DIR,
)


def _setup_logging(verbose: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    fmt   = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    logging.basicConfig(
        level=level, format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(OUTPUT_DIR / "backtest.log", encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lead-lag information diffusion backtest")
    p.add_argument("--wrds-user",     default=os.environ.get("WRDS_USERNAME", ""))
    p.add_argument("--start-date",    default=BACKTEST_START)
    p.add_argument("--end-date",      default=BACKTEST_END)
    p.add_argument("--threshold",     type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--holding-hours", type=int,   default=DEFAULT_HOLDING_HOURS)
    p.add_argument("--entry-delay",   type=int,   default=DEFAULT_ENTRY_DELAY)
    p.add_argument("--hedge",         default=DEFAULT_HEDGE,
                                       choices=["primary", "etf", "none"])
    p.add_argument("--model",         default=DEFAULT_MODEL,
                                       choices=["beta_only", "beta_revenue", "vol_based"])
    p.add_argument("--max-positions", type=int,   default=MAX_CONCURRENT_POSITIONS)
    p.add_argument("--sweep",         action="store_true")
    p.add_argument("--no-cache",      action="store_true",
                                       help="Force re-download from WRDS")
    p.add_argument("--verbose",       action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.verbose)
    log = logging.getLogger("run")

    log.info("=" * 60)
    log.info("Lead-Lag Backtest  %s → %s", args.start_date, args.end_date)
    log.info("=" * 60)

    if not args.wrds_user:
        log.error("WRDS username required. Pass --wrds-user or set WRDS_USERNAME.")
        sys.exit(1)

    from backtest.data_loader import get_wrds_connection, load_event_window_data
    log.info("Connecting to WRDS as '%s' …", args.wrds_user)
    db = get_wrds_connection(args.wrds_user)

    if not EVENTS_FILE.exists():
        log.error("Events file not found: %s", EVENTS_FILE)
        sys.exit(1)

    from backtest.event_detector import load_events, detect_events
    log.info("Loading events from %s …", EVENTS_FILE)
    raw_events = load_events(EVENTS_FILE)

    log.info("Event-targeted TAQ download …")
    bar_data, daily_data = load_event_window_data(
        db=db, raw_events=raw_events,
        start_date=args.start_date, end_date=args.end_date,
        force=args.no_cache,
    )
    log.info("Loaded %d tickers of hourly TAQ bars.", len(bar_data))

    event_results = detect_events(raw_events, bar_data, daily_data)
    log.info("%d events loaded, %d passed materiality.",
             len(event_results),
             sum(1 for e in event_results if e.passed_materiality))

    from backtest.backtest_engine import run_backtest, run_sweep
    from backtest.analytics import save_all_outputs

    sweep_df = None
    if args.sweep:
        log.info("Running parameter sweep …")
        sweep_df = run_sweep(bar_data, daily_data, event_results)
        log.info("Sweep top 5:\n%s", sweep_df.head(5).to_string(index=False))
        best = sweep_df.iloc[0].to_dict()
        result = run_backtest(
            bar_data=bar_data, daily_data=daily_data, event_results=event_results,
            entry_delay_hours=int(best.get("entry_delay_hours", args.entry_delay)),
            holding_hours=int(best.get("holding_period_hours", args.holding_hours)),
            hedge_method=str(best.get("hedge_method", args.hedge)),
            signal_model=str(best.get("signal_model", args.model)),
            threshold=float(best.get("underreaction_threshold", args.threshold)),
            max_positions=args.max_positions,
        )
    else:
        log.info("Running: delay=%dh hold=%dh hedge=%s model=%s threshold=%.2f",
                 args.entry_delay, args.holding_hours, args.hedge,
                 args.model, args.threshold)
        result = run_backtest(
            bar_data=bar_data, daily_data=daily_data, event_results=event_results,
            entry_delay_hours=args.entry_delay,
            holding_hours=args.holding_hours,
            hedge_method=args.hedge,
            signal_model=args.model,
            threshold=args.threshold,
            max_positions=args.max_positions,
        )

    spy_daily = daily_data["SPY"]["close"] if "SPY" in daily_data else None
    save_all_outputs(result, sweep_df=sweep_df, spy_daily=spy_daily)
    log.info("Outputs in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
