"""Out-of-sample (2021-2025) test of the parameters chosen on 2016-2020.

Frozen config: 10-day hold, no stop-loss, β-revenue model, threshold 0.30, hedge=none.
Also tests 5d / 7d holds and a hedged variant for sanity.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from backtest.config import OUTPUT_DIR, ROOT_DIR
from backtest.analytics import save_all_outputs
from backtest import backtest_engine as _bt_engine
from backtest.backtest_engine import run_backtest
from backtest.data_loader import _bars_to_daily
from backtest.event_detector import detect_events, load_events
from compare import load_cached_bars


_ORIGINAL_STOP_LOSS = _bt_engine.EXIT_STOP_LOSS_PCT
EVENTS_FILE = ROOT_DIR / "backtest_events_2021_2025.xlsx"


def run_one(label, bar_data, daily_data, raw_events, **kw):
    log = logging.getLogger("compare_oos")
    log.info("=" * 60)
    log.info("Run: %s  %s", label, kw)
    log.info("=" * 60)

    stop_loss = kw.pop("stop_loss_pct", None)
    _bt_engine.EXIT_STOP_LOSS_PCT = stop_loss if stop_loss is not None else _ORIGINAL_STOP_LOSS

    event_results = detect_events(raw_events, bar_data, daily_data)
    result = run_backtest(
        bar_data=bar_data, daily_data=daily_data,
        event_results=event_results, **kw,
    )
    save_all_outputs(result, sweep_df=None, spy_daily=None)

    src = OUTPUT_DIR / "summary_stats.txt"
    dst = OUTPUT_DIR / f"summary_oos_{label}.txt"
    if src.exists():
        shutil.copy(src, dst)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("compare_oos")

    log.info("Loading cached hourly bars …")
    bar_data = load_cached_bars()
    log.info("Loaded %d tickers from cache.", len(bar_data))

    daily_data = _bars_to_daily(bar_data)
    raw_events = load_events(EVENTS_FILE)
    log.info("Loaded %d events from %s", len(raw_events), EVENTS_FILE.name)

    common = dict(hedge_method="none", signal_model="beta_revenue", threshold=0.30)
    common_hedged = dict(hedge_method="primary", signal_model="beta_revenue", threshold=0.30)

    run_one("best_hold10d",
            bar_data, daily_data, raw_events,
            entry_delay_hours=2, holding_hours=70, stop_loss_pct=0.99, **common)
    run_one("hold5d",
            bar_data, daily_data, raw_events,
            entry_delay_hours=2, holding_hours=35, stop_loss_pct=0.99, **common)
    run_one("hold7d",
            bar_data, daily_data, raw_events,
            entry_delay_hours=2, holding_hours=49, stop_loss_pct=0.99, **common)
    run_one("hold10d_hedged",
            bar_data, daily_data, raw_events,
            entry_delay_hours=2, holding_hours=70, stop_loss_pct=0.99, **common_hedged)

    log.info("Done. OOS summaries in %s/summary_oos_*.txt", OUTPUT_DIR)


if __name__ == "__main__":
    main()
