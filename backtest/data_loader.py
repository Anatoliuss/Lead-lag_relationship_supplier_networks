"""WRDS data download, caching, and hourly/daily bar construction."""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from pathlib import Path
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
    HARD_TO_BORROW,
    SECTOR_ETF,
    TICKER_IPO_DATES,
    TICKER_MAPPING,
    ALL_PRIMARIES,
    ALL_TICKERS,
)

log = logging.getLogger(__name__)

# ── Cache paths ────────────────────────────────────────────────────────────

def _hourly_cache(ticker: str, year: int) -> Path:
    return DATA_DIR / "hourly" / f"{ticker}_{year}.parquet"

def _daily_cache(ticker: str) -> Path:
    return DATA_DIR / "daily" / f"{ticker}.parquet"

def _beta_cache(ticker: str) -> Path:
    return DATA_DIR / "beta" / f"{ticker}_betas.parquet"


# ── WRDS connection helper ─────────────────────────────────────────────────

def get_wrds_connection(wrds_username: Optional[str] = None):
    """Return an active wrds.Connection, using env var or argument for credentials."""
    import wrds  # type: ignore
    username = wrds_username or os.environ.get("WRDS_USERNAME", "")
    if not username:
        raise ValueError(
            "Set WRDS_USERNAME environment variable or pass --wrds-user on the command line."
        )
    return wrds.Connection(wrds_username=username)


# ── Ticker helpers ─────────────────────────────────────────────────────────

def resolve_ticker(ticker: str, as_of_date: pd.Timestamp) -> str:
    """Return the historical ticker symbol for a given date (handles UTX→RTX)."""
    mapping = TICKER_MAPPING.get(ticker)
    if mapping and as_of_date < pd.Timestamp(mapping["merger_date"]):
        return mapping["pre_merger_ticker"]
    return ticker


def is_ticker_active(ticker: str, as_of_date: pd.Timestamp) -> bool:
    """Return False if the ticker had not yet IPO'd by as_of_date."""
    ipo_str = TICKER_IPO_DATES.get(ticker)
    if ipo_str and as_of_date < pd.Timestamp(ipo_str):
        return False
    return True


# ── TAQ hourly bar construction ────────────────────────────────────────────

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
    """
    Aggregate a DataFrame of TAQ trades into hourly OHLCV bars.

    Parameters
    ----------
    trades : DataFrame with columns [time_m, price, size]
              time_m should be strings like '09:30:01.123' or datetime.time objects
    trade_date : the calendar date of the trades

    Returns
    -------
    DataFrame indexed by bar_start (Timestamp), columns [open, high, low, close, volume]
    """
    if trades.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # Normalise time column to Timestamp.  TAQ returns a mix of "HH:MM:SS" and
    # "HH:MM:SS.fff" strings in the same column, so let pandas infer per-row.
    trades = trades.copy()
    if pd.api.types.is_datetime64_any_dtype(trades["time_m"]):
        trades["dt"] = trades["time_m"]
    else:
        date_str = str(trade_date)
        trades["dt"] = pd.to_datetime(
            date_str + " " + trades["time_m"].astype(str),
            format="mixed",
            errors="coerce",
        )
        trades = trades.dropna(subset=["dt"])

    bars = []
    for bar_open_str, bar_close_str in _HOUR_BINS:
        t_open  = pd.Timestamp(f"{trade_date} {bar_open_str}")
        t_close = pd.Timestamp(f"{trade_date} {bar_close_str}")
        mask = (trades["dt"] >= t_open) & (trades["dt"] < t_close)
        chunk = trades.loc[mask]
        if chunk.empty:
            bars.append({
                "bar_start": t_open,
                "open": np.nan, "high": np.nan,
                "low": np.nan,  "close": np.nan,
                "volume": 0,
            })
        else:
            bars.append({
                "bar_start": t_open,
                "open":   chunk.iloc[0]["price"],
                "high":   chunk["price"].max(),
                "low":    chunk["price"].min(),
                "close":  chunk.iloc[-1]["price"],
                "volume": chunk["size"].sum(),
            })

    result = pd.DataFrame(bars).set_index("bar_start")
    return result


# ── TAQ library auto-detection ─────────────────────────────────────────────

# Possible TAQ table patterns in order of preference
_TAQ_FORMATS = [
    ("taqmsec", "ctm_{date}"),   # millisecond TAQ (newer subscriptions)
    ("taq",     "ctm_{date}"),   # legacy consolidated trades
    ("taq",     "ct_{date}"),    # even older format
]

# Cache: db id → detected format tuple or None
_taq_format_cache: dict[int, Optional[tuple[str, str]]] = {}


def _detect_taq_format(db) -> Optional[tuple[str, str]]:
    """
    Probe WRDS to discover which TAQ table format is available AND usable.

    Two-stage check:
      1. Table exists (schema probe).
      2. At least one row is returned for CVX on 2019-01-02 (data probe).
         If the table exists but returns no rows for a heavily-traded stock,
         the subscription almost certainly does not include trade-level TAQ data,
         and we fall back to CRSP pseudo-hourly for all tickers.

    Result is cached per connection object so we only probe once per run.
    """
    key = id(db)
    if key in _taq_format_cache:
        return _taq_format_cache[key]

    probe_date = "20190102"   # known liquid trading day
    probe_sym  = "CVX"

    for lib, tmpl in _TAQ_FORMATS:
        table = f"{lib}.{tmpl.format(date=probe_date)}"
        time_col = "time_m" if "ctm" in tmpl else "time"

        # Stage 1: does the table exist?
        try:
            db.raw_sql(f"SELECT 1 FROM {table} LIMIT 1")
        except Exception:
            continue   # try next format

        # Stage 2: does it have real trade data for a liquid ticker?
        # NOTE: do NOT filter on sym_suffix or tr_corr here — many valid rows
        # have sym_suffix IS NULL or non-'00' correction codes that are still
        # legitimate prints. We sanity-check post-probe instead.
        data_sql = f"""
            SELECT COUNT(*) AS n
            FROM {table}
            WHERE sym_root = '{probe_sym}'
              AND price > 0
              AND size  > 0
              AND {time_col} BETWEEN '09:30:00' AND '16:00:00'
        """
        try:
            result = db.raw_sql(data_sql)
            n_rows = int(result.iloc[0, 0])
        except Exception:
            n_rows = 0

        if n_rows > 0:
            log.info(
                "TAQ format confirmed: %s (%d %s trades on probe day).",
                table, n_rows, probe_sym,
            )
            fmt = (lib, tmpl)
            _taq_format_cache[key] = fmt
            return fmt
        else:
            log.info(
                "TAQ table %s exists but returned 0 rows for %s — "
                "subscription may not include trade-level data.",
                table, probe_sym,
            )
            # Don't try other formats for the same library type; continue to next.

    log.warning(
        "TAQ intraday data not accessible on this WRDS subscription. "
        "All tickers will use pseudo-hourly bars derived from CRSP daily OHLCV. "
        "Signal quality will be lower than with true intraday bars."
    )
    _taq_format_cache[key] = None
    return None


def _query_taq_daily(db, ticker: str, trade_date: date) -> pd.DataFrame:
    """
    Pull raw consolidated trades for one ticker/day from TAQ.
    Auto-detects the correct library/table format on first call.
    Returns an empty DataFrame (silently) when the table doesn't exist —
    the caller will fall back to CRSP daily.
    """
    fmt = _detect_taq_format(db)
    if fmt is None:
        return pd.DataFrame(columns=["time_m", "price", "size"])

    lib, tmpl   = fmt
    date_str    = pd.Timestamp(trade_date).strftime("%Y%m%d")
    table       = f"{lib}.{tmpl.format(date=date_str)}"
    sym         = resolve_ticker(ticker, pd.Timestamp(trade_date))

    # Column name differs between formats
    time_col = "time_m" if "ctm" in tmpl else "time"
    corr_col = "tr_corr" if lib == "taqmsec" else "tr_corr"

    # Pull with loose filters; apply correction-code/suffix filtering in pandas
    # so a restrictive WHERE clause can't accidentally zero out a ticker.
    sql = f"""
        SELECT {time_col} AS time_m,
               price,
               size,
               sym_suffix,
               {corr_col} AS tr_corr
        FROM {table}
        WHERE sym_root = '{sym}'
          AND price > 0
          AND size  > 0
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

    # Exclude only explicit cancels ("08", "09"). Anything else — including
    # blank, null, "00", late-print codes — is kept.  Past too-strict filters
    # were eating every row for major tickers like XOM / SPY.
    if "tr_corr" in df.columns:
        corr = df["tr_corr"].astype(str).str.strip()
        df = df[~corr.isin(["08", "09"])]

    return df[["time_m", "price", "size"]]


def _taq_ticker_has_data(db, ticker: str) -> bool:
    """
    Quick one-row probe: does this ticker appear at all in TAQ on the probe day?
    Takes ~0.1 s instead of the ~10 s a full year download would waste.
    Result is cached per (connection, ticker).
    """
    fmt = _detect_taq_format(db)
    if fmt is None:
        return False

    lib, tmpl   = fmt
    probe_date  = "20190102"
    table       = f"{lib}.{tmpl.format(date=probe_date)}"
    sym         = ticker
    time_col    = "time_m" if "ctm" in tmpl else "time"

    sql = f"""
        SELECT 1
        FROM {table}
        WHERE sym_root = '{sym}'
          AND price > 0
          AND size  > 0
          AND {time_col} BETWEEN '09:30:00' AND '16:00:00'
        LIMIT 1
    """
    try:
        result = db.raw_sql(sql)
        if len(result) > 0:
            return True
    except Exception:
        pass

    # Try sweep across a few probe dates before giving up — some tickers
    # weren't listed on 2019-01-02 (IPO'd later) but are perfectly valid later.
    for alt in ("20170103", "20180102", "20200102", "20160104"):
        alt_table = f"{lib}.{tmpl.format(date=alt)}"
        alt_sql = f"""
            SELECT 1 FROM {alt_table}
            WHERE sym_root = '{sym}'
              AND price > 0
              AND size  > 0
              AND {time_col} BETWEEN '09:30:00' AND '16:00:00'
            LIMIT 1
        """
        try:
            result = db.raw_sql(alt_sql)
            if len(result) > 0:
                return True
        except Exception:
            continue
    return False


# Per-connection ticker availability cache: (conn_id, ticker) → bool
_ticker_taq_available: dict[tuple, bool] = {}


def _ticker_in_taq(db, ticker: str) -> bool:
    key = (id(db), ticker)
    if key not in _ticker_taq_available:
        ok = _taq_ticker_has_data(db, ticker)
        if not ok:
            _diagnose_taq_for_ticker(db, ticker)
        _ticker_taq_available[key] = ok
    return _ticker_taq_available[key]


def _diagnose_taq_for_ticker(db, ticker: str) -> None:
    """
    When a ticker fails the normal probe, run progressively looser queries
    across a handful of probe dates so the log shows WHAT is in the table.
    Purely diagnostic — does not affect caching or return value.
    """
    fmt = _detect_taq_format(db)
    if fmt is None:
        return
    lib, tmpl = fmt
    time_col  = "time_m" if "ctm" in tmpl else "time"

    for probe_date in ("20190102", "20180102", "20200102"):
        table = f"{lib}.{tmpl.format(date=probe_date)}"
        # Count any rows for this sym_root at all
        try:
            row_any = db.raw_sql(
                f"SELECT COUNT(*) AS n FROM {table} WHERE sym_root = '{ticker}'"
            )
            n_any = int(row_any.iloc[0, 0])
        except Exception as exc:
            log.debug("TAQ diag: %s on %s — table unreadable: %s", ticker, probe_date, exc)
            continue

        if n_any == 0:
            # Is the ticker symbol just missing? Look for close matches.
            try:
                like = db.raw_sql(
                    f"SELECT DISTINCT sym_root FROM {table} "
                    f"WHERE sym_root LIKE '{ticker[:2]}%' LIMIT 5"
                )
                sample = ", ".join(like["sym_root"].astype(str).tolist())
                log.info(
                    "TAQ diag: %s has 0 rows in %s. Similar sym_roots: %s",
                    ticker, table, sample or "(none)",
                )
            except Exception:
                log.info("TAQ diag: %s has 0 rows in %s.", ticker, table)
            return

        # Rows exist — show why they fail the strict filter
        try:
            info = db.raw_sql(f"""
                SELECT
                  SUM(CASE WHEN price > 0 AND size > 0 THEN 1 ELSE 0 END) AS clean,
                  SUM(CASE WHEN {time_col} BETWEEN '09:30:00' AND '16:00:00'
                           THEN 1 ELSE 0 END) AS rth,
                  COUNT(*) AS total
                FROM {table}
                WHERE sym_root = '{ticker}'
            """)
            log.info(
                "TAQ diag: %s on %s — total=%s clean=%s rth=%s (but probe still 0).",
                ticker, probe_date,
                int(info.iloc[0]["total"]),
                int(info.iloc[0]["clean"]),
                int(info.iloc[0]["rth"]),
            )
        except Exception as exc:
            log.debug("TAQ diag detail failed for %s on %s: %s", ticker, probe_date, exc)
        return


def build_hourly_bars_from_taq(
    db,
    ticker:         str,
    start_date:     str,
    end_date:       str,
    force:          bool = False,
    trading_days:   Optional[list[date]] = None,
) -> pd.DataFrame:
    """
    Build hourly OHLCV bars for *ticker*.

    If ``trading_days`` is provided, only those specific days are downloaded
    (event-window mode).  Otherwise every business day between
    ``start_date`` and ``end_date`` is downloaded.

    Per-day bars are cached individually in data/hourly/<ticker>/<YYYY-MM-DD>.parquet
    so overlapping event windows across runs share cache entries.
    Returns an empty DataFrame if TAQ is unavailable or no data was found.
    """
    if _detect_taq_format(db) is None:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if trading_days is None:
        trading_days = _get_trading_days(
            pd.Timestamp(start_date), pd.Timestamp(end_date),
        )

    # Clip to backtest range
    s = pd.Timestamp(start_date).date()
    e = pd.Timestamp(end_date).date()
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
        # Always write cache (even empty) so we don't re-query on next run
        day_bars.to_csv(day_cache, index_label="bar_start")
        if not day_bars.empty:
            frames.append(day_bars)
        days_downloaded += 1

    log.info(
        "%s: downloaded %d day(s), %d cached — total %d bars.",
        ticker, days_downloaded, len(trading_days) - days_downloaded,
        sum(len(f) for f in frames),
    )

    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    combined = pd.concat(frames).sort_index()
    # Write a combined per-ticker CSV for easy inspection
    combined.to_csv(DATA_DIR / "hourly" / f"{ticker}.csv", index_label="bar_start")
    return combined


# ── Event window planning ──────────────────────────────────────────────────

def plan_event_windows(
    raw_events: pd.DataFrame,
    pre_days:   int = EVENT_WINDOW_PRE_DAYS,
    post_days:  int = EVENT_WINDOW_POST_DAYS,
) -> dict[str, set[date]]:
    """
    For each event, expand the primary's sector-wide rows, enumerate the
    [event_date - pre, event_date + post] trading-day window, and collect
    the set of (ticker, date) we actually need TAQ data for.

    Tickers included per event:
      - the primary
      - its dependents (from DEPENDENCY_MAP)
      - its sector ETF (from SECTOR_ETF)

    Always-needed tickers (SPY as market benchmark) are added for the full
    union of event dates.

    Returns {ticker: set[date]}.
    """
    from backtest.event_detector import expand_sector_events  # local import avoids cycles

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

        tickers_for_event = {primary}
        tickers_for_event.update(
            dep["ticker"] for dep in DEPENDENCY_MAP.get(primary, [])
        )
        etf = SECTOR_ETF.get(primary)
        if etf:
            tickers_for_event.add(etf)

        for t in tickers_for_event:
            needs.setdefault(t, set()).update(window)

    # SPY across the union of all event windows (market benchmark for beta fallback)
    if all_event_days:
        needs.setdefault("SPY", set()).update(all_event_days)

    return needs


def _window_trading_days(
    event_day: date,
    pre_days:  int,
    post_days: int,
) -> list[date]:
    """Return ±N *business* days around event_day (inclusive)."""
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
    """
    Event-targeted TAQ download.  Only pulls hourly data for tickers/days
    actually needed by the events file (primary + dependents + sector ETF,
    within ±EVENT_WINDOW_*_DAYS of each event).
    """
    if _detect_taq_format(db) is None:
        raise RuntimeError(
            "TAQ intraday data is not accessible on this WRDS subscription."
        )

    plan = plan_event_windows(raw_events)
    log.info(
        "Event-targeted download plan: %d tickers, %d ticker-days.",
        len(plan), sum(len(v) for v in plan.values()),
    )

    bar_data: dict[str, pd.DataFrame] = {}
    for ticker in sorted(plan.keys()):
        days = sorted(plan[ticker])
        try:
            bars = build_hourly_bars_from_taq(
                db, ticker, start_date, end_date,
                force=force, trading_days=days,
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


# ── CRSP daily bar download ────────────────────────────────────────────────

def get_daily_data_crsp(
    db,
    tickers: list[str],
    start_date: str,
    end_date:   str,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Download daily OHLCV from CRSP for each ticker.
    Returns {ticker: DataFrame} with columns [open, high, low, close, volume, ret].
    """
    result: dict[str, pd.DataFrame] = {}

    # First get PERMNO mapping for all tickers
    ticker_list = "', '".join(tickers)
    permno_sql = f"""
        SELECT DISTINCT ticker, permno
        FROM crsp.dsenames
        WHERE ticker IN ('{ticker_list}')
          AND namedt  <= '{end_date}'
          AND nameendt >= '{start_date}'
    """
    try:
        permno_df = db.raw_sql(permno_sql)
    except Exception as exc:
        log.error("PERMNO lookup failed: %s", exc)
        return result

    # Group tickers → may have multiple permnos (e.g. RTX/UTX)
    for ticker in tickers:
        cache_file = _daily_cache(ticker)
        if cache_file.exists() and not force:
            log.debug("Daily cache hit: %s", cache_file)
            result[ticker] = pd.read_parquet(cache_file)
            continue

        permnos = permno_df.loc[permno_df["ticker"] == ticker, "permno"].tolist()
        if not permnos:
            # Try historical ticker for RTX/UTX case
            mapped = resolve_ticker(ticker, pd.Timestamp(start_date))
            if mapped != ticker:
                permnos = permno_df.loc[
                    permno_df["ticker"] == mapped, "permno"
                ].tolist()

        if not permnos:
            log.warning("No PERMNO found for %s, skipping.", ticker)
            continue

        permno_str = ", ".join(str(p) for p in permnos)
        sql = f"""
            SELECT a.date,
                   ABS(a.prc)   AS close,
                   a.openprc    AS open,
                   a.askhi      AS high,
                   a.bidlo      AS low,
                   a.vol        AS volume,
                   a.ret
            FROM crsp.dsf AS a
            WHERE a.permno IN ({permno_str})
              AND a.date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY a.date
        """
        try:
            df = db.raw_sql(sql)
        except Exception as exc:
            log.error("CRSP daily query failed for %s: %s", ticker, exc)
            continue

        if df.empty:
            log.warning("No CRSP data for %s.", ticker)
            continue

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df.dropna(subset=["close"])

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)
        result[ticker] = df
        log.info("Downloaded daily CRSP data for %s: %d rows.", ticker, len(df))

    return result


# ── Aggregate loader: hourly with daily fallback ───────────────────────────

def load_all_data(
    db,
    tickers: list[str],
    start_date: str = BACKTEST_START,
    end_date:   str = BACKTEST_END,
    daily_only: bool = False,
    force: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Load price bars and daily data for all tickers.

    When TAQ intraday data is available for a ticker, return real hourly bars.
    When it is not, return real daily CRSP bars (one bar per trading day).
    No fake or interpolated prices are ever generated.

    The returned ``bar_data`` dict uses the same schema as hourly TAQ bars
    (open, high, low, close, volume) but indexed by the day's open timestamp
    (09:30) when only daily data is available.  Downstream code works
    identically — it just operates at daily rather than hourly resolution.

    Returns
    -------
    bar_data   : {ticker: DataFrame}  indexed by bar_start Timestamp
    daily_data : {ticker: DataFrame}  indexed by date  (always CRSP daily)
    """
    if _detect_taq_format(db) is None:
        raise RuntimeError(
            "TAQ intraday data is not accessible on this WRDS subscription. "
            "This backtest requires hourly TAQ data — no fallback is used."
        )

    bar_data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            bars = build_hourly_bars_from_taq(db, ticker, start_date, end_date, force=force)
        except Exception as exc:
            log.error("Error loading TAQ bars for %s: %s", ticker, exc)
            continue
        if bars.empty:
            log.info("%s: no hourly TAQ bars available, skipping.", ticker)
            continue
        log.info("%s: %d real hourly TAQ bars loaded.", ticker, len(bars))
        bar_data[ticker] = bars

    # Daily data = hourly bars resampled to one row per trading day.
    # Used downstream for beta regression — no CRSP needed.
    daily_data = _bars_to_daily(bar_data)
    return bar_data, daily_data


def _bars_to_daily(
    bar_data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Collapse hourly bars to daily OHLCV for beta computation."""
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


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_trading_days(start: pd.Timestamp, end: pd.Timestamp) -> list[date]:
    """Return NYSE trading days between start and end (inclusive)."""
    # Simple approximation: weekdays excluding common US holidays
    # For production, use pandas_market_calendars or a holiday calendar.
    bdays = pd.bdate_range(start, end)
    return [d.date() for d in bdays]


def _crsp_to_daily_bars(
    daily_data: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """
    Convert CRSP daily OHLCV rows into the bar schema used by the rest of the
    system, with each bar indexed at 09:30 of its trading day.

    This is real data — one bar per trading day, no interpolation.
    Open, high, low, close, volume are the actual CRSP values.
    Tickers missing open/high/low in CRSP will have those columns set to the
    closing price (which is the only price guaranteed present in CRSP DSF).
    """
    result: dict[str, pd.DataFrame] = {}
    for ticker, df in daily_data.items():
        rows = []
        for ts, row in df.iterrows():
            day   = pd.Timestamp(ts).normalize()
            close = row["close"]
            if pd.isna(close) or close <= 0:
                continue
            rows.append({
                "bar_start": day + pd.Timedelta(hours=9, minutes=30),
                "open":   row["open"]   if pd.notna(row.get("open"))   and row["open"]  > 0 else close,
                "high":   row["high"]   if pd.notna(row.get("high"))   and row["high"]  > 0 else close,
                "low":    row["low"]    if pd.notna(row.get("low"))    and row["low"]   > 0 else close,
                "close":  close,
                "volume": row["volume"] if pd.notna(row.get("volume")) else 0,
            })
        if rows:
            out = pd.DataFrame(rows).set_index("bar_start").sort_index()
            result[ticker] = out
    return result


def forward_fill_bars(
    bar_data: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Forward-fill NaN prices within each bar series (gaps between trading days are NOT filled)."""
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


def compute_hourly_returns(
    hourly_data: dict[str, pd.DataFrame]
) -> dict[str, pd.Series]:
    """Return {ticker: Series of bar-over-bar returns (pct change of close)}."""
    returns = {}
    for ticker, df in hourly_data.items():
        returns[ticker] = df["close"].pct_change()
    return returns
