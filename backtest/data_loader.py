"""WRDS TAQ download, per-day caching, and hourly/daily bar building."""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from backtest.config import (
    BACKTEST_END,
    BACKTEST_START,
    DATA_DIR,
    DEPENDENCY_MAP,
    EVENT_WINDOW_POST_DAYS,
    EVENT_WINDOW_PRE_DAYS,
    SECTOR_ETF,
    TICKER_IPO_DATES,
    TICKER_MAPPING,
)

log = logging.getLogger(__name__)


def get_wrds_connection(wrds_username: Optional[str] = None):
    import wrds
    username = wrds_username or os.environ.get("WRDS_USERNAME", "")
    if not username:
        raise ValueError("Set WRDS_USERNAME env var or pass --wrds-user.")
    return wrds.Connection(wrds_username=username)


def resolve_ticker(ticker: str, as_of_date: pd.Timestamp) -> str:
    """Map RTX → UTX before merger date."""
    mapping = TICKER_MAPPING.get(ticker)
    if mapping and as_of_date < pd.Timestamp(mapping["merger_date"]):
        return mapping["pre_merger_ticker"]
    return ticker


def is_ticker_active(ticker: str, as_of_date: pd.Timestamp) -> bool:
    ipo_str = TICKER_IPO_DATES.get(ticker)
    if ipo_str and as_of_date < pd.Timestamp(ipo_str):
        return False
    return True


# Hourly bar bins (NYSE regular trading hours)
_HOUR_BINS = [
    ("09:30", "10:30"),
    ("10:30", "11:30"),
    ("11:30", "12:30"),
    ("12:30", "13:30"),
    ("13:30", "14:30"),
    ("14:30", "15:30"),
    ("15:30", "16:00"),
]


def _aggregate_trades_to_hourly(trades: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    """Bin a day's TAQ trades into 7 hourly OHLCV bars."""
    if trades.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    trades = trades.copy()
    if pd.api.types.is_datetime64_any_dtype(trades["time_m"]):
        trades["dt"] = trades["time_m"]
    else:
        date_str = str(trade_date)
        # TAQ mixes "HH:MM:SS" and "HH:MM:SS.fff" strings — let pandas infer
        trades["dt"] = pd.to_datetime(
            date_str + " " + trades["time_m"].astype(str),
            format="mixed", errors="coerce",
        )
        trades = trades.dropna(subset=["dt"])

    bars = []
    for bar_open_str, bar_close_str in _HOUR_BINS:
        t_open  = pd.Timestamp(f"{trade_date} {bar_open_str}")
        t_close = pd.Timestamp(f"{trade_date} {bar_close_str}")
        chunk = trades[(trades["dt"] >= t_open) & (trades["dt"] < t_close)]
        if chunk.empty:
            bars.append({"bar_start": t_open, "open": np.nan, "high": np.nan,
                         "low": np.nan, "close": np.nan, "volume": 0})
        else:
            bars.append({
                "bar_start": t_open,
                "open":   chunk.iloc[0]["price"],
                "high":   chunk["price"].max(),
                "low":    chunk["price"].min(),
                "close":  chunk.iloc[-1]["price"],
                "volume": chunk["size"].sum(),
            })
    return pd.DataFrame(bars).set_index("bar_start")


_TAQ_FORMATS = [
    ("taqmsec", "ctm_{date}"),
    ("taq",     "ctm_{date}"),
    ("taq",     "ct_{date}"),
]
_taq_format_cache: dict[int, Optional[tuple[str, str]]] = {}


def _detect_taq_format(db) -> Optional[tuple[str, str]]:
    """Probe WRDS for the available TAQ table format. Cached per-connection."""
    key = id(db)
    if key in _taq_format_cache:
        return _taq_format_cache[key]

    probe_date = "20190102"
    probe_sym  = "CVX"

    for lib, tmpl in _TAQ_FORMATS:
        table = f"{lib}.{tmpl.format(date=probe_date)}"
        time_col = "time_m" if "ctm" in tmpl else "time"

        try:
            db.raw_sql(f"SELECT 1 FROM {table} LIMIT 1")
        except Exception:
            continue

        sql = f"""
            SELECT COUNT(*) AS n FROM {table}
            WHERE sym_root = '{probe_sym}' AND price > 0 AND size > 0
              AND {time_col} BETWEEN '09:30:00' AND '16:00:00'
        """
        try:
            n = int(db.raw_sql(sql).iloc[0, 0])
        except Exception:
            n = 0

        if n > 0:
            log.info("TAQ format confirmed: %s (%d %s trades on probe day).",
                     table, n, probe_sym)
            _taq_format_cache[key] = (lib, tmpl)
            return (lib, tmpl)

    log.warning("TAQ intraday data not accessible on this WRDS subscription.")
    _taq_format_cache[key] = None
    return None


def _query_taq_daily(db, ticker: str, trade_date: date) -> pd.DataFrame:
    """Pull one ticker's trades for one day from TAQ."""
    fmt = _detect_taq_format(db)
    if fmt is None:
        return pd.DataFrame(columns=["time_m", "price", "size"])

    lib, tmpl = fmt
    date_str = pd.Timestamp(trade_date).strftime("%Y%m%d")
    table    = f"{lib}.{tmpl.format(date=date_str)}"
    sym      = resolve_ticker(ticker, pd.Timestamp(trade_date))
    time_col = "time_m" if "ctm" in tmpl else "time"

    sql = f"""
        SELECT {time_col} AS time_m, price, size, sym_suffix, tr_corr
        FROM {table}
        WHERE sym_root = '{sym}' AND price > 0 AND size > 0
          AND {time_col} BETWEEN '09:30:00' AND '16:00:00'
        ORDER BY {time_col}
    """
    try:
        df = db.raw_sql(sql)
    except Exception as exc:
        log.debug("TAQ query skipped for %s on %s: %s", ticker, trade_date, exc)
        return pd.DataFrame(columns=["time_m", "price", "size"])

    if df.empty:
        return df

    # Drop only explicit cancels — keeping "00", blanks, late prints
    if "tr_corr" in df.columns:
        corr = df["tr_corr"].astype(str).str.strip()
        df = df[~corr.isin(["08", "09"])]
    return df[["time_m", "price", "size"]]


def build_hourly_bars_from_taq(
    db,
    ticker: str,
    start_date: str,
    end_date:   str,
    force: bool = False,
    trading_days: Optional[list[date]] = None,
) -> pd.DataFrame:
    """Build hourly bars for ticker over the given days. Per-day CSV cached."""
    if _detect_taq_format(db) is None:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if trading_days is None:
        trading_days = _get_trading_days(pd.Timestamp(start_date), pd.Timestamp(end_date))

    s, e = pd.Timestamp(start_date).date(), pd.Timestamp(end_date).date()
    trading_days = sorted({d for d in trading_days if s <= d <= e})
    if not trading_days:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    ticker_dir = DATA_DIR / "hourly" / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    days_downloaded = 0
    for tday in trading_days:
        day_cache = ticker_dir / f"{tday.isoformat()}.csv"
        if day_cache.exists() and not force:
            df = pd.read_csv(day_cache, index_col=0, parse_dates=True)
            if not df.empty:
                frames.append(df)
            continue

        trades   = _query_taq_daily(db, ticker, tday)
        day_bars = _aggregate_trades_to_hourly(trades, tday)
        # Always write — even empty — so we don't re-query next run
        day_bars.to_csv(day_cache, index_label="bar_start")
        if not day_bars.empty:
            frames.append(day_bars)
        days_downloaded += 1

    log.info("%s: downloaded %d day(s), %d cached — total %d bars.",
             ticker, days_downloaded, len(trading_days) - days_downloaded,
             sum(len(f) for f in frames))

    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    combined = pd.concat(frames).sort_index()
    combined.to_csv(DATA_DIR / "hourly" / f"{ticker}.csv", index_label="bar_start")
    return combined


def plan_event_windows(
    raw_events: pd.DataFrame,
    pre_days:   int = EVENT_WINDOW_PRE_DAYS,
    post_days:  int = EVENT_WINDOW_POST_DAYS,
) -> dict[str, set[date]]:
    """{ticker: set of dates} we need TAQ data for, ±N days around each event."""
    from backtest.event_detector import expand_sector_events

    events = expand_sector_events(raw_events)
    needs: dict[str, set[date]] = {}
    all_event_days: set[date] = set()

    for _, row in events.iterrows():
        primary = str(row.get("primary", "")).strip().upper()
        if not primary:
            continue
        try:
            event_day = pd.Timestamp(row["date"]).normalize().date()
        except Exception:
            continue

        window = _window_trading_days(event_day, pre_days, post_days)
        all_event_days.update(window)

        tickers = {primary}
        tickers.update(dep["ticker"] for dep in DEPENDENCY_MAP.get(primary, []))
        etf = SECTOR_ETF.get(primary)
        if etf:
            tickers.add(etf)

        for t in tickers:
            needs.setdefault(t, set()).update(window)

    if all_event_days:
        needs.setdefault("SPY", set()).update(all_event_days)
    return needs


def _window_trading_days(event_day: date, pre_days: int, post_days: int) -> list[date]:
    start = pd.Timestamp(event_day) - pd.offsets.BDay(pre_days)
    end   = pd.Timestamp(event_day) + pd.offsets.BDay(post_days)
    return [d.date() for d in pd.bdate_range(start, end)]


def load_event_window_data(
    db,
    raw_events: pd.DataFrame,
    start_date: str = BACKTEST_START,
    end_date:   str = BACKTEST_END,
    force:      bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Event-targeted download. Pull only ticker-days actually needed."""
    if _detect_taq_format(db) is None:
        raise RuntimeError("TAQ intraday data not accessible.")

    plan = plan_event_windows(raw_events)
    log.info("Event-targeted download plan: %d tickers, %d ticker-days.",
             len(plan), sum(len(v) for v in plan.values()))

    bar_data: dict[str, pd.DataFrame] = {}
    for ticker in sorted(plan.keys()):
        days = sorted(plan[ticker])
        try:
            bars = build_hourly_bars_from_taq(
                db, ticker, start_date, end_date, force=force, trading_days=days,
            )
        except Exception as exc:
            log.error("Error loading TAQ bars for %s: %s", ticker, exc)
            continue
        if bars.empty:
            log.info("%s: no hourly TAQ bars across %d event days.", ticker, len(days))
            continue
        bar_data[ticker] = bars

    daily_data = _bars_to_daily(bar_data)
    return bar_data, daily_data


def _bars_to_daily(bar_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Resample hourly bars to daily OHLCV (used for β computation)."""
    out: dict[str, pd.DataFrame] = {}
    for ticker, df in bar_data.items():
        if df.empty:
            continue
        g = df.groupby(df.index.normalize())
        daily = pd.DataFrame({
            "open":   g["open"].first(),
            "high":   g["high"].max(),
            "low":    g["low"].min(),
            "close":  g["close"].last(),
            "volume": g["volume"].sum(),
        })
        daily["ret"] = daily["close"].pct_change()
        out[ticker] = daily
    return out


def _get_trading_days(start: pd.Timestamp, end: pd.Timestamp) -> list[date]:
    return [d.date() for d in pd.bdate_range(start, end)]


def forward_fill_bars(bar_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Forward-fill within-day price gaps (overnight gaps are NOT filled)."""
    filled = {}
    for ticker, df in bar_data.items():
        df = df.copy()
        df["close"]  = df["close"].ffill()
        df["open"]   = df["open"].fillna(df["close"])
        df["high"]   = df["high"].fillna(df["close"])
        df["low"]    = df["low"].fillna(df["close"])
        df["volume"] = df["volume"].fillna(0)
        filled[ticker] = df
    return filled
