"""Regenerate IS (2016-2020) and OOS (2021-2025) plots side-by-side.

Runs the best config on each events file and saves charts as *_is.png / *_oos.png.
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

PLOTS = ["equity_curve.png", "drawdown.png", "pnl_distribution.png",
         "monthly_heatmap.png", "rolling_sharpe.png", "decay_curve.png",
         "win_rate_by_holding.png", "score_vs_pnl.png", "sector_curves.png"]


def _clip_bars(bars_dict: dict, start: str, end: str) -> dict:
    """Return a new ticker→DataFrame dict with rows restricted to [start, end]."""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end) + pd.Timedelta(days=1)   # inclusive end
    out = {}
    for t, df in bars_dict.items():
        clip = df.loc[(df.index >= s) & (df.index < e)]
        if not clip.empty:
            out[t] = clip
    return out


def run_and_snapshot(label: str, events_file: Path,
                     start: str, end: str,
                     bar_data, daily_data) -> None:
    log = logging.getLogger("regen")
    log.info("=" * 60)
    log.info("Run: %s  (events=%s, period=%s..%s)", label, events_file.name, start, end)
    log.info("=" * 60)

    # Clip bar_data and daily_data so NAV snapshots only span this run's period
    bars_clip  = _clip_bars(bar_data, start, end)
    daily_clip = _clip_bars(daily_data, start, end)
    log.info("Clipped to %d tickers (%d daily) for %s..%s",
             len(bars_clip), len(daily_clip), start, end)

    raw_events = load_events(events_file)

    _bt_engine.EXIT_STOP_LOSS_PCT = 0.99   # disable stop loss
    event_results = detect_events(raw_events, bars_clip, daily_clip)
    result = run_backtest(
        bar_data=bars_clip,
        daily_data=daily_clip,
        event_results=event_results,
        entry_delay_hours=2,
        holding_hours=70,
        hedge_method="none",
        signal_model="beta_revenue",
        threshold=0.30,
    )
    save_all_outputs(result, sweep_df=None, spy_daily=None)

    # Snapshot all plots and trades.csv with the label suffix
    for p in PLOTS:
        src = OUTPUT_DIR / p
        if not src.exists():
            continue
        dst = OUTPUT_DIR / f"{src.stem}_{label}{src.suffix}"
        shutil.copy(src, dst)
        log.info("Saved %s", dst.name)
    trades_src = OUTPUT_DIR / "trades.csv"
    if trades_src.exists():
        shutil.copy(trades_src, OUTPUT_DIR / f"trades_{label}.csv")
    summary_src = OUTPUT_DIR / "summary_stats.txt"
    if summary_src.exists():
        shutil.copy(summary_src, OUTPUT_DIR / f"summary_{label}.txt")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("regen")

    log.info("Loading cached hourly bars (full 2016-2025) ...")
    bar_data = load_cached_bars()
    log.info("Loaded %d tickers from cache.", len(bar_data))
    daily_data = _bars_to_daily(bar_data)

    is_file  = ROOT_DIR / "backtest_events_2016_2020_v2.xlsx"
    oos_file = ROOT_DIR / "backtest_events_2021_2025.xlsx"

    run_and_snapshot("is",  is_file,  "2016-01-01", "2020-12-31", bar_data, daily_data)
    run_and_snapshot("oos", oos_file, "2021-01-01", "2025-05-15", bar_data, daily_data)

    log.info("Done. Plots saved as *_is.png and *_oos.png in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
