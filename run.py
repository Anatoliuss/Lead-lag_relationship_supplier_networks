#!/usr/bin/env python
"""Entry point for the lead-lag information diffusion backtest."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Make sure `backtest/` is importable when running as `python run.py` from project root
sys.path.insert(0, str(Path(__file__).parent))

from backtest.config import (
    ALL_TICKERS,
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
        level=level,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(OUTPUT_DIR / "backtest.log", encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lead-lag information diffusion backtest (2016-2020)"
    )
    p.add_argument("--wrds-user",    default=os.environ.get("WRDS_USERNAME", ""),
                   help="WRDS username (or set WRDS_USERNAME env var)")
    p.add_argument("--start-date",   default=BACKTEST_START)
    p.add_argument("--end-date",     default=BACKTEST_END)
    p.add_argument("--threshold",    type=float, default=DEFAULT_THRESHOLD,
                   help="Underreaction threshold (default 0.30)")
    p.add_argument("--holding-hours",type=int,   default=DEFAULT_HOLDING_HOURS,
                   help="Holding period in hours (default 7 = 1 trading day)")
    p.add_argument("--entry-delay",  type=int,   default=DEFAULT_ENTRY_DELAY,
                   help="Entry delay in hourly bars after event (default 2)")
    p.add_argument("--hedge",        default=DEFAULT_HEDGE, choices=["primary", "etf", "none"],
                   help="Hedge leg: short primary, sector ETF, or none (long-only dependent)")
    p.add_argument("--model",        default=DEFAULT_MODEL,
                   choices=["beta_only", "beta_revenue"],
                   help="Signal model (default beta_revenue)")
    p.add_argument("--max-positions",type=int,   default=MAX_CONCURRENT_POSITIONS)
    p.add_argument("--sweep",        action="store_true",
                   help="Run full parameter sweep (~320 combinations)")
    p.add_argument("--no-cache",     action="store_true",
                   help="Force re-download from WRDS (ignore cached parquet files)")
    p.add_argument("--daily-only",   action="store_true",
                   help="Use daily bars only (faster, for debugging)")
    p.add_argument("--verbose",      action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.verbose)
    log = logging.getLogger("run")

    log.info("=" * 60)
    log.info("Lead-Lag Backtest  %s → %s", args.start_date, args.end_date)
    log.info("=" * 60)

    # ── 1. Connect to WRDS ────────────────────────────────────────────────
    if not args.wrds_user:
        log.error(
            "WRDS username required. Pass --wrds-user or set WRDS_USERNAME env var."
        )
        sys.exit(1)

    from backtest.data_loader import (
        get_wrds_connection, load_event_window_data, _detect_taq_format
    )
    log.info("Connecting to WRDS as '%s' …", args.wrds_user)
    db = get_wrds_connection(args.wrds_user)

    # ── 2. Load events FIRST so downloads can be event-targeted ───────────
    if not EVENTS_FILE.exists():
        log.error("Events file not found: %s", EVENTS_FILE)
        sys.exit(1)

    from backtest.event_detector import load_events, detect_events
    log.info("Loading events from %s …", EVENTS_FILE)
    raw_events = load_events(EVENTS_FILE)

    # ── 3. Download only the hourly bars events need ──────────────────────
    log.info("Event-targeted TAQ download …")
    bar_data, daily_data = load_event_window_data(
        db=db,
        raw_events=raw_events,
        start_date=args.start_date,
        end_date=args.end_date,
        force=args.no_cache,
    )

    import backtest.config as _cfg
    _cfg.BARS_PER_DAY = _cfg.BARS_PER_DAY_HOURLY
    log.info("Loaded %d tickers of hourly TAQ bars.", len(bar_data))

    event_results = detect_events(raw_events, bar_data, daily_data)
    log.info(
        "%d events loaded, %d passed materiality filter.",
        len(event_results),
        sum(1 for e in event_results if e.passed_materiality),
    )

    # ── 4. Run backtest ────────────────────────────────────────────────────
    from backtest.backtest_engine import run_backtest, run_sweep
    from backtest.analytics import save_all_outputs

    sweep_df = None

    if args.sweep:
        log.info("Running parameter sweep …")
        sweep_df = run_sweep(bar_data, daily_data, event_results)
        log.info("Sweep complete. Top 5 results:")
        log.info("\n%s", sweep_df.head(5).to_string(index=False))

        # Run the best single config for full analytics
        best = sweep_df.iloc[0].to_dict()
        log.info("Running best config: %s", best)
        result = run_backtest(
            bar_data=bar_data,
            daily_data=daily_data,
            event_results=event_results,
            entry_delay_hours=int(best.get("entry_delay_hours", args.entry_delay)),
            holding_hours=int(best.get("holding_period_hours", args.holding_hours)),
            hedge_method=str(best.get("hedge_method", args.hedge)),
            signal_model=str(best.get("signal_model", args.model)),
            threshold=float(best.get("underreaction_threshold", args.threshold)),
            max_positions=args.max_positions,
        )
    else:
        log.info(
            "Running single backtest: entry_delay=%dh, holding=%dh, hedge=%s, model=%s, threshold=%.2f",
            args.entry_delay, args.holding_hours, args.hedge, args.model, args.threshold,
        )
        result = run_backtest(
            bar_data=bar_data,
            daily_data=daily_data,
            event_results=event_results,
            entry_delay_hours=args.entry_delay,
            holding_hours=args.holding_hours,
            hedge_method=args.hedge,
            signal_model=args.model,
            threshold=args.threshold,
            max_positions=args.max_positions,
        )

    # ── 5. Analytics & output ──────────────────────────────────────────────
    spy_daily = None
    if "SPY" in daily_data:
        spy_daily = daily_data["SPY"]["close"]

    save_all_outputs(result, sweep_df=sweep_df, spy_daily=spy_daily)
    log.info("All outputs written to %s", OUTPUT_DIR)
    log.info("Done.")


if __name__ == "__main__":
    main()
