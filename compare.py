"""Run three backtest configs back-to-back using cached hourly CSVs."""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from backtest.config import DATA_DIR, EVENTS_FILE, OUTPUT_DIR
from backtest.analytics import save_all_outputs
from backtest import backtest_engine as _bt_engine
from backtest.backtest_engine import run_backtest
from backtest.data_loader import _bars_to_daily
from backtest.event_detector import detect_events, load_events


_ORIGINAL_STOP_LOSS = _bt_engine.EXIT_STOP_LOSS_PCT


def load_cached_bars() -> dict[str, pd.DataFrame]:
    """Load hourly bars from per-day cache: data/hourly/<ticker>/<YYYY-MM-DD>.csv.

    Falls back to a combined data/hourly/<ticker>.csv if present.
    """
    out: dict[str, pd.DataFrame] = {}
    hourly_dir = DATA_DIR / "hourly"
    if not hourly_dir.exists():
        return out

    for ticker_dir in sorted(hourly_dir.iterdir()):
        if ticker_dir.is_dir():
            frames = []
            for day_csv in sorted(ticker_dir.glob("*.csv")):
                df = pd.read_csv(day_csv, index_col=0, parse_dates=True)
                if not df.empty:
                    frames.append(df)
            if frames:
                combined = pd.concat(frames).sort_index()
                combined = combined[~combined.index.duplicated(keep="last")]
                out[ticker_dir.name] = combined
        elif ticker_dir.suffix == ".csv":
            df = pd.read_csv(ticker_dir, index_col="bar_start", parse_dates=True)
            if not df.empty and ticker_dir.stem not in out:
                out[ticker_dir.stem] = df
    return out


def run_one(
    label: str,
    bar_data: dict[str, pd.DataFrame],
    daily_data: dict[str, pd.DataFrame],
    raw_events: pd.DataFrame,
    **bt_kwargs,
) -> None:
    log = logging.getLogger("compare")
    log.info("=" * 60)
    log.info("Run: %s  %s", label, bt_kwargs)
    log.info("=" * 60)

    # Filter events by start_date / end_date if passed
    start_date = bt_kwargs.pop("start_date", None)
    end_date   = bt_kwargs.pop("end_date", None)
    # Optional per-run stop-loss override
    stop_loss = bt_kwargs.pop("stop_loss_pct", None)
    if stop_loss is not None:
        _bt_engine.EXIT_STOP_LOSS_PCT = stop_loss
        log.info("Overriding stop-loss to %.2f%%", stop_loss * 100)
    else:
        _bt_engine.EXIT_STOP_LOSS_PCT = _ORIGINAL_STOP_LOSS

    events = raw_events
    if start_date:
        events = events[events["date"] >= pd.Timestamp(start_date)]
        log.info("Filtered to %d events starting %s", len(events), start_date)
    if end_date:
        events = events[events["date"] <= pd.Timestamp(end_date)]
        log.info("Filtered to %d events ending %s", len(events), end_date)

    event_results = detect_events(events, bar_data, daily_data)
    result = run_backtest(
        bar_data=bar_data,
        daily_data=daily_data,
        event_results=event_results,
        **bt_kwargs,
    )

    # Snapshot output/summary_stats.txt to a per-label name
    save_all_outputs(result, sweep_df=None, spy_daily=None)
    src = OUTPUT_DIR / "summary_stats.txt"
    dst = OUTPUT_DIR / f"summary_{label}.txt"
    if src.exists():
        shutil.copy(src, dst)
        log.info("Saved → %s", dst)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("compare")

    log.info("Loading cached hourly bars …")
    bar_data = load_cached_bars()
    log.info("Loaded %d tickers from cache.", len(bar_data))

    daily_data = _bars_to_daily(bar_data)
    raw_events = load_events(EVENTS_FILE)

    common = dict(
        hedge_method="none",
        signal_model="beta_revenue",
        threshold=0.30,
    )

    # Hedged variant — short the primary alongside long-dependent
    common_hedged = dict(
        hedge_method="primary",
        signal_model="beta_revenue",
        threshold=0.30,
    )

    # 2017-01-01 onward, trimmed universe (PAA/WTI/NGS removed)
    period = dict(start_date="2017-01-01")

    run_one(
        "trimmed_2017on_hold5d",
        bar_data, daily_data, raw_events,
        entry_delay_hours=2, holding_hours=35,
        stop_loss_pct=0.99, **common, **period,
    )

    run_one(
        "trimmed_2017on_hold7d",
        bar_data, daily_data, raw_events,
        entry_delay_hours=2, holding_hours=49,
        stop_loss_pct=0.99, **common, **period,
    )

    run_one(
        "trimmed_2017on_hold10d",
        bar_data, daily_data, raw_events,
        entry_delay_hours=2, holding_hours=70,
        stop_loss_pct=0.99, **common, **period,
    )

    run_one(
        "trimmed_2017on_hold10d_hedged",
        bar_data, daily_data, raw_events,
        entry_delay_hours=2, holding_hours=70,
        stop_loss_pct=0.99, **common_hedged, **period,
    )

    # Full-universe 2016-2020 baseline for reference
    run_one(
        "baseline_full_2016_2020_hold10d",
        bar_data, daily_data, raw_events,
        entry_delay_hours=2, holding_hours=70,
        stop_loss_pct=0.99, **common,
    )

    log.info("Done. Summaries in %s/summary_<label>.txt", OUTPUT_DIR)


if __name__ == "__main__":
    main()
