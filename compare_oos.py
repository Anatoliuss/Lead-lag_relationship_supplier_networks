"""Out-of-sample test: apply the 2016-2020 best strategy to 2021-2025 events.

Best strategy = trimmed universe (PAA/WTI/NGS dropped), 10-day hold,
no stop-loss, β-revenue model, threshold 0.30, hedge=none.

Universe trim was made on 2016-2020 data; this script tests whether it
generalizes.  Compares against a 5-day hold and against the hedged variant.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from backtest.config import EVENTS_FILE, OUTPUT_DIR
from backtest.analytics import save_all_outputs
from backtest import backtest_engine as _bt_engine
from backtest.backtest_engine import run_backtest
from backtest.data_loader import _bars_to_daily
from backtest.event_detector import detect_events, load_events
from compare import load_cached_bars  # reuse per-day cache reader


_ORIGINAL_STOP_LOSS = _bt_engine.EXIT_STOP_LOSS_PCT


def run_one(label, bar_data, daily_data, raw_events, **kw):
    log = logging.getLogger("compare_oos")
    log.info("=" * 60)
    log.info("Run: %s  %s", label, kw)
    log.info("=" * 60)

    stop_loss = kw.pop("stop_loss_pct", None)
    if stop_loss is not None:
        _bt_engine.EXIT_STOP_LOSS_PCT = stop_loss
    else:
        _bt_engine.EXIT_STOP_LOSS_PCT = _ORIGINAL_STOP_LOSS

    event_results = detect_events(raw_events, bar_data, daily_data)
    result = run_backtest(
        bar_data=bar_data,
        daily_data=daily_data,
        event_results=event_results,
        **kw,
    )
    save_all_outputs(result, sweep_df=None, spy_daily=None)
    src = OUTPUT_DIR / "summary_stats.txt"
    dst = OUTPUT_DIR / f"summary_oos_{label}.txt"
    if src.exists():
        shutil.copy(src, dst)
        log.info("Saved -> %s", dst)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("compare_oos")

    log.info("Loading cached hourly bars ...")
    bar_data = load_cached_bars()
    log.info("Loaded %d tickers from cache.", len(bar_data))

    daily_data = _bars_to_daily(bar_data)
    raw_events = load_events(EVENTS_FILE)
    log.info("Loaded %d events from %s", len(raw_events), EVENTS_FILE.name)

    common = dict(
        hedge_method="none",
        signal_model="beta_revenue",
        threshold=0.30,
    )
    common_hedged = dict(
        hedge_method="primary",
        signal_model="beta_revenue",
        threshold=0.30,
    )

    # The headline OOS test: exact in-sample best config
    run_one(
        "best_hold10d",
        bar_data, daily_data, raw_events,
        entry_delay_hours=2, holding_hours=70,
        stop_loss_pct=0.99, **common,
    )

    # Comparison: 5-day hold to see whether reversion timeline is consistent
    run_one(
        "hold5d",
        bar_data, daily_data, raw_events,
        entry_delay_hours=2, holding_hours=35,
        stop_loss_pct=0.99, **common,
    )

    # Comparison: 7-day intermediate
    run_one(
        "hold7d",
        bar_data, daily_data, raw_events,
        entry_delay_hours=2, holding_hours=49,
        stop_loss_pct=0.99, **common,
    )

    # Hedged variant for sanity (in-sample showed hedging hurt)
    run_one(
        "hold10d_hedged",
        bar_data, daily_data, raw_events,
        entry_delay_hours=2, holding_hours=70,
        stop_loss_pct=0.99, **common_hedged,
    )

    log.info("Done. OOS summaries in %s/summary_oos_*.txt", OUTPUT_DIR)


if __name__ == "__main__":
    main()
