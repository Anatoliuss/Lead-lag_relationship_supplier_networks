"""Performance metrics, breakdown tables, charts, and output files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from backtest.config import OUTPUT_DIR, RISK_FREE_RATE, INITIAL_CAPITAL

if TYPE_CHECKING:
    from backtest.backtest_engine import RunResult

log = logging.getLogger(__name__)


# ── Core performance metrics ───────────────────────────────────────────────

def compute_summary_stats(result: "RunResult") -> dict:
    """Return a dict of scalar performance metrics."""
    trades = result.portfolio.closed_trades
    snaps  = result.hourly_snapshots

    if not trades:
        return _empty_stats()

    net_pnls = [t.net_pnl for t in trades]

    # NAV series from hourly snapshots
    if snaps:
        nav = pd.Series(
            [s.portfolio_value for s in snaps],
            index=[s.bar for s in snaps],
        )
    else:
        nav = pd.Series([INITIAL_CAPITAL], dtype=float)

    total_return = (nav.iloc[-1] / nav.iloc[0]) - 1.0
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 0.01)
    ann_return = (1 + total_return) ** (1 / years) - 1

    daily_nav = nav.resample("1D").last().dropna()
    daily_ret = daily_nav.pct_change().dropna()

    sharpe   = _sharpe(daily_ret, RISK_FREE_RATE)
    sortino  = _sortino(daily_ret, RISK_FREE_RATE)
    max_dd   = _max_drawdown(nav)
    calmar   = ann_return / abs(max_dd) if max_dd != 0 else np.nan

    wins  = [p for p in net_pnls if p > 0]
    loses = [p for p in net_pnls if p <= 0]
    win_rate     = len(wins) / len(net_pnls) if net_pnls else 0.0
    avg_win      = np.mean(wins)  if wins  else 0.0
    avg_loss     = np.mean(loses) if loses else 0.0
    profit_factor= sum(wins) / abs(sum(loses)) if loses else np.nan

    avg_holding = np.mean([t.holding_hours for t in trades])

    return dict(
        total_return=total_return,
        ann_return=ann_return,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=calmar,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        n_trades=len(trades),
        avg_holding_hours=avg_holding,
        avg_net_pnl=np.mean(net_pnls),
        final_nav=nav.iloc[-1],
    )


def _empty_stats() -> dict:
    keys = [
        "total_return", "ann_return", "sharpe", "sortino", "max_drawdown",
        "calmar", "win_rate", "avg_win", "avg_loss", "profit_factor",
        "n_trades", "avg_holding_hours", "avg_net_pnl", "final_nav",
    ]
    return {k: np.nan for k in keys}


def _sharpe(daily_ret: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    if daily_ret.empty or daily_ret.std() == 0:
        return np.nan
    excess = daily_ret - rf / 252
    return float(excess.mean() / excess.std() * np.sqrt(252))


def _sortino(daily_ret: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    if daily_ret.empty:
        return np.nan
    excess = daily_ret - rf / 252
    downside = excess[excess < 0]
    if downside.empty or downside.std() == 0:
        return np.nan
    return float(excess.mean() / downside.std() * np.sqrt(252))


def _max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    dd   = (nav - peak) / peak
    return float(dd.min())


# ── Breakdown tables ───────────────────────────────────────────────────────

def breakdown_by(result: "RunResult", by: str) -> pd.DataFrame:
    """
    Aggregate closed-trade P&L by a given grouping key.
    by: "primary" | "dependent" | "event_type" | "sector" | "year" | "holding_period"
    """
    trades = result.portfolio.closed_trades
    if not trades:
        return pd.DataFrame()

    rows = []
    for t in trades:
        sig = t.signal
        rows.append({
            "primary":        sig.primary_ticker,
            "dependent":      sig.dependent_ticker,
            "event_type":     sig.event_type,
            "sector":         _sector(sig.primary_ticker),
            "year":           sig.event_date.year,
            "holding_period": t.holding_hours,
            "net_pnl":        t.net_pnl,
            "gross_pnl":      t.gross_pnl,
            "win":            int(t.net_pnl > 0),
        })

    df = pd.DataFrame(rows)
    grp = df.groupby(by)
    summary = grp.agg(
        n_trades=("net_pnl", "count"),
        total_pnl=("net_pnl", "sum"),
        avg_pnl=("net_pnl", "mean"),
        win_rate=("win", "mean"),
    ).reset_index()
    return summary.sort_values("total_pnl", ascending=False)


def _sector(primary: str) -> str:
    from backtest.config import FABLESS_PRIMARIES, FOUNDRY_PRIMARIES, WFE_PRIMARIES
    if primary in FABLESS_PRIMARIES:
        return "Fabless"
    if primary in FOUNDRY_PRIMARIES:
        return "IDM/Foundry"
    if primary in WFE_PRIMARIES:
        return "WFE"
    return "Other"


def monthly_returns(result: "RunResult") -> pd.DataFrame:
    """Return a Year × Month pivot of monthly returns (as %)."""
    snaps = result.daily_snapshots
    if not snaps:
        return pd.DataFrame()

    nav = pd.Series(
        [s.portfolio_value for s in snaps],
        index=pd.DatetimeIndex([s.bar for s in snaps]),
    )
    monthly = nav.resample("ME").last().pct_change().dropna() * 100
    pivot = monthly.to_frame("ret")
    pivot["year"]  = pivot.index.year
    pivot["month"] = pivot.index.month
    table = pivot.pivot(index="year", columns="month", values="ret")
    table.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ][:len(table.columns)]
    return table


def slippage_sensitivity(result: "RunResult") -> pd.DataFrame:
    """Compute net Sharpe at different slippage assumptions."""
    trades = result.portfolio.closed_trades
    if not trades:
        return pd.DataFrame()

    rows = []
    for extra_bps in [0, 10, 25, 50, 100]:
        adjusted_pnls = []
        for t in trades:
            extra_cost = (
                t.position_size * (1 + t.hedge_ratio)
                * extra_bps / 10_000 * 2  # entry + exit
            )
            adjusted_pnls.append(t.net_pnl - extra_cost)

        nav_adj = INITIAL_CAPITAL + np.cumsum(adjusted_pnls)
        daily_ret = pd.Series(nav_adj).pct_change().dropna()
        rows.append({
            "extra_slippage_bps": extra_bps,
            "total_net_pnl":      sum(adjusted_pnls),
            "sharpe":             _sharpe(daily_ret),
        })

    return pd.DataFrame(rows)


def underreaction_decay_curve(result: "RunResult") -> pd.DataFrame:
    """
    For each tradeable signal, collect scores by horizon.
    Returns a DataFrame with mean and 95% CI for each horizon.
    """
    tradeable = [s for s in result.signals if not s.skipped]
    if not tradeable:
        return pd.DataFrame()

    records = []
    for sig in tradeable:
        for key, score in sig.scores_by_horizon.items():
            records.append({"horizon": key, "score": score})

    df = pd.DataFrame(records).dropna()
    if df.empty:
        return pd.DataFrame()

    # Sort horizon labels numerically
    df["h_num"] = df["horizon"].str.replace("h", "").astype(int)
    df = df.sort_values("h_num")

    summary = df.groupby("h_num")["score"].agg(
        mean="mean",
        std="std",
        n="count",
    ).reset_index()
    summary["se"]       = summary["std"] / np.sqrt(summary["n"])
    summary["ci95_lo"]  = summary["mean"] - 1.96 * summary["se"]
    summary["ci95_hi"]  = summary["mean"] + 1.96 * summary["se"]
    return summary


# ── Charts ─────────────────────────────────────────────────────────────────

def _ensure_output() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def plot_equity_curve(result: "RunResult", spy_daily: pd.Series = None) -> None:
    snaps = result.daily_snapshots
    if not snaps:
        return
    nav = pd.Series(
        [s.portfolio_value for s in snaps],
        index=pd.DatetimeIndex([s.bar for s in snaps]),
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    (nav / nav.iloc[0]).plot(ax=ax, label="Strategy", linewidth=1.5)
    if spy_daily is not None and not spy_daily.empty:
        spy_norm = spy_daily.reindex(nav.index, method="ffill").dropna()
        (spy_norm / spy_norm.iloc[0]).plot(ax=ax, label="SPY", linewidth=1, linestyle="--")
    ax.set_title("Equity Curve")
    ax.set_ylabel("Cumulative Return (normalised)")
    ax.legend()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    plt.tight_layout()
    plt.savefig(_ensure_output() / "equity_curve.png", dpi=150)
    plt.close(fig)


def plot_drawdown(result: "RunResult") -> None:
    snaps = result.daily_snapshots
    if not snaps:
        return
    nav = pd.Series(
        [s.portfolio_value for s in snaps],
        index=pd.DatetimeIndex([s.bar for s in snaps]),
    )
    peak = nav.cummax()
    dd   = (nav - peak) / peak

    fig, ax = plt.subplots(figsize=(12, 4))
    dd.plot(ax=ax, color="red", linewidth=1)
    ax.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
    ax.set_title("Drawdown")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    plt.tight_layout()
    plt.savefig(_ensure_output() / "drawdown.png", dpi=150)
    plt.close(fig)


def plot_monthly_heatmap(result: "RunResult") -> None:
    table = monthly_returns(result)
    if table.empty:
        return

    fig, ax = plt.subplots(figsize=(14, len(table) * 0.7 + 1))
    im = ax.imshow(table.values, cmap="RdYlGn", aspect="auto", vmin=-5, vmax=5)
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    for i in range(len(table.index)):
        for j in range(len(table.columns)):
            val = table.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="Return (%)")
    ax.set_title("Monthly Returns Heatmap")
    plt.tight_layout()
    plt.savefig(_ensure_output() / "monthly_heatmap.png", dpi=150)
    plt.close(fig)


def plot_pnl_distribution(result: "RunResult") -> None:
    pnls = [t.net_pnl for t in result.portfolio.closed_trades]
    if not pnls:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(pnls, bins=40, edgecolor="black", color="steelblue")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_title("Net P&L per Trade Distribution")
    ax.set_xlabel("Net P&L ($)")
    plt.tight_layout()
    plt.savefig(_ensure_output() / "pnl_distribution.png", dpi=150)
    plt.close(fig)


def plot_decay_curve(result: "RunResult") -> None:
    df = underreaction_decay_curve(result)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["h_num"], df["mean"], marker="o", linewidth=1.5, label="Avg score")
    ax.fill_between(df["h_num"], df["ci95_lo"], df["ci95_hi"], alpha=0.2, label="95% CI")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Hours after event")
    ax.set_ylabel("Underreaction score")
    ax.set_title("Underreaction Decay Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(_ensure_output() / "decay_curve.png", dpi=150)
    plt.close(fig)


def plot_win_rate_by_holding(result: "RunResult") -> None:
    trades = result.portfolio.closed_trades
    if not trades:
        return
    df = pd.DataFrame([{"h": t.holding_hours, "win": int(t.net_pnl > 0)} for t in trades])
    bins = [0, 2, 5, 8, 15, 22, 36, 9999]
    labels = ["1-2h", "3-5h", "6-8h", "9-15h", "16-22h", "23-35h", ">35h"]
    df["bucket"] = pd.cut(df["h"], bins=bins, labels=labels)
    summary = df.groupby("bucket", observed=True)["win"].agg(win_rate="mean", n="count").reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(summary["bucket"].astype(str), summary["win_rate"], color="steelblue", edgecolor="black")
    ax.set_title("Win Rate by Holding Period")
    ax.set_ylabel("Win Rate")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    plt.tight_layout()
    plt.savefig(_ensure_output() / "win_rate_by_holding.png", dpi=150)
    plt.close(fig)


def plot_rolling_sharpe(result: "RunResult", window: int = 50) -> None:
    trades = result.portfolio.closed_trades
    if len(trades) < window:
        return
    pnls = pd.Series([t.net_pnl for t in trades])
    roll_mean = pnls.rolling(window).mean()
    roll_std  = pnls.rolling(window).std()
    roll_sharpe = roll_mean / roll_std * np.sqrt(252 * 7)  # annualise from hourly

    fig, ax = plt.subplots(figsize=(10, 4))
    roll_sharpe.plot(ax=ax, linewidth=1.5)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title(f"Rolling {window}-Trade Sharpe")
    ax.set_ylabel("Sharpe")
    plt.tight_layout()
    plt.savefig(_ensure_output() / "rolling_sharpe.png", dpi=150)
    plt.close(fig)


def plot_score_vs_pnl(result: "RunResult") -> None:
    tradeable = [s for s in result.signals if not s.skipped]
    trades     = {t.signal.dependent_ticker + str(t.signal.event_date): t
                  for t in result.portfolio.closed_trades}

    xs, ys = [], []
    for sig in tradeable:
        key = sig.dependent_ticker + str(sig.event_date)
        if key in trades:
            xs.append(abs(sig.score_model_b))
            ys.append(trades[key].net_pnl)

    if not xs:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, alpha=0.5, s=20)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Underreaction score at entry")
    ax.set_ylabel("Net P&L ($)")
    ax.set_title("Score at Entry vs Net P&L")
    plt.tight_layout()
    plt.savefig(_ensure_output() / "score_vs_pnl.png", dpi=150)
    plt.close(fig)


def plot_sector_curves(result: "RunResult") -> None:
    """Separate equity curves by semi sub-industry (Fabless / IDM-Foundry / WFE)."""
    buckets: dict[str, list[float]] = {"Fabless": [], "IDM/Foundry": [], "WFE": []}
    for t in result.portfolio.closed_trades:
        buckets[_sector(t.signal.primary_ticker)].append(t.net_pnl)
    if not any(buckets.values()):
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, pnls in buckets.items():
        if not pnls:
            continue
        cum = INITIAL_CAPITAL + np.cumsum(pnls)
        ax.plot(cum / INITIAL_CAPITAL, label=label)
    ax.axhline(1, color="black", linestyle="--", linewidth=0.5)
    ax.set_title("Sector Sub-Portfolio Equity Curves")
    ax.set_ylabel("NAV / Initial Capital")
    ax.legend()
    plt.tight_layout()
    plt.savefig(_ensure_output() / "sector_curves.png", dpi=150)
    plt.close(fig)


# ── Output files ───────────────────────────────────────────────────────────

def save_all_outputs(
    result: "RunResult",
    sweep_df: pd.DataFrame = None,
    spy_daily: pd.Series = None,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = compute_summary_stats(result)

    _save_summary_stats(stats, result)
    _save_trades_csv(result)
    _save_signals_csv(result)
    _save_event_analysis_csv(result)
    _save_portfolio_csvs(result)
    if sweep_df is not None and not sweep_df.empty:
        sweep_df.to_csv(OUTPUT_DIR / "parameter_sweep.csv", index=False)
        log.info("Saved parameter_sweep.csv")
    _save_slippage_sensitivity(result)

    # Charts
    try:
        plot_equity_curve(result, spy_daily)
        plot_drawdown(result)
        plot_monthly_heatmap(result)
        plot_pnl_distribution(result)
        plot_decay_curve(result)
        plot_win_rate_by_holding(result)
        plot_rolling_sharpe(result)
        plot_score_vs_pnl(result)
        plot_sector_curves(result)
        log.info("All charts saved to %s.", OUTPUT_DIR)
    except Exception as exc:
        log.warning("Chart generation error: %s", exc)


def _save_summary_stats(stats: dict, result: "RunResult") -> None:
    lines = [
        "=" * 60,
        "  LEAD-LAG BACKTEST — SUMMARY STATISTICS",
        "=" * 60,
        f"  Period              : {result.params}",
        f"  Total Return        : {stats.get('total_return', np.nan):.2%}",
        f"  Annualised Return   : {stats.get('ann_return', np.nan):.2%}",
        f"  Sharpe Ratio        : {stats.get('sharpe', np.nan):.2f}",
        f"  Sortino Ratio       : {stats.get('sortino', np.nan):.2f}",
        f"  Max Drawdown        : {stats.get('max_drawdown', np.nan):.2%}",
        f"  Calmar Ratio        : {stats.get('calmar', np.nan):.2f}",
        f"  Win Rate            : {stats.get('win_rate', np.nan):.2%}",
        f"  Profit Factor       : {stats.get('profit_factor', np.nan):.2f}",
        f"  N Trades            : {stats.get('n_trades', 0):.0f}",
        f"  Avg Holding (hrs)   : {stats.get('avg_holding_hours', np.nan):.1f}",
        f"  Avg Net P&L/Trade   : ${stats.get('avg_net_pnl', np.nan):.0f}",
        f"  Final NAV           : ${stats.get('final_nav', np.nan):.0f}",
        "",
    ]

    # Breakdown tables
    for by in ["primary", "dependent", "event_type", "sector", "year"]:
        try:
            tbl = breakdown_by(result, by)
            if not tbl.empty:
                lines += ["", f"--- By {by.upper()} ---", tbl.to_string(index=False)]
        except Exception:
            pass

    # Slippage sensitivity
    slip_df = slippage_sensitivity(result)
    if not slip_df.empty:
        lines += ["", "--- SLIPPAGE SENSITIVITY ---", slip_df.to_string(index=False)]

    text = "\n".join(lines)
    out_path = OUTPUT_DIR / "summary_stats.txt"
    out_path.write_text(text, encoding="utf-8")
    log.info("Saved summary_stats.txt")
    print(text)


def _save_trades_csv(result: "RunResult") -> None:
    rows = []
    for t in result.portfolio.closed_trades:
        sig = t.signal
        rows.append({
            "trade_id":       t.trade_id,
            "entry_bar":      t.entry_bar,
            "exit_bar":       t.exit_bar,
            "primary":        sig.primary_ticker,
            "dependent":      sig.dependent_ticker,
            "direction":      t.direction,
            "hedge_ticker":   t.hedge_ticker,
            "event_type":     sig.event_type,
            "event_date":     sig.event_date,
            "position_size":  t.position_size,
            "entry_price_dep":t.entry_price_dep,
            "exit_price_dep": t.exit_price_dep,
            "beta":           sig.beta,
            "r2":             sig.r_squared,
            "score_at_entry": sig.score_model_b,
            "score_at_exit":  t.score_at_exit,
            "holding_hours":  t.holding_hours,
            "gross_pnl":      t.gross_pnl,
            "txn_cost":       t.txn_cost,
            "borrow_cost":    t.borrow_cost,
            "net_pnl":        t.net_pnl,
            "exit_reason":    t.exit_reason,
            "dep_avg_hvol":   sig.dependent_avg_hourly_volume,
        })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "trades.csv", index=False)
    log.info("Saved trades.csv (%d rows).", len(rows))


def _save_signals_csv(result: "RunResult") -> None:
    rows = []
    for sig in result.signals:
        rows.append({
            "event_date":     sig.event_date,
            "primary":        sig.primary_ticker,
            "dependent":      sig.dependent_ticker,
            "event_type":     sig.event_type,
            "beta":           sig.beta,
            "r2":             sig.r_squared,
            "revenue_pct":    sig.revenue_pct,
            "expected_a":     sig.expected_return_model_a,
            "expected_b":     sig.expected_return_model_b,
            "actual_return":  sig.actual_return,
            "score_a":        sig.score_model_a,
            "score_b":        sig.score_model_b,
            "direction":      sig.signal_direction,
            "is_liquid":      sig.is_liquid,
            "is_htb":         sig.is_hard_to_borrow,
            "skipped":        sig.skipped,
            "skip_reason":    sig.skip_reason,
            **{f"score_{k}": v for k, v in sig.scores_by_horizon.items()},
        })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "signal_analysis.csv", index=False)
    log.info("Saved signal_analysis.csv (%d rows).", len(rows))


def _save_event_analysis_csv(result: "RunResult") -> None:
    rows = []
    for ev in result.event_results:
        rows.append({
            "event_date":          ev.event_date,
            "primary":             ev.primary_ticker,
            "event_type":          ev.event_type,
            "description":         ev.event_description,
            "reaction_start":      ev.reaction_start,
            "car_1h":              ev.car_1h,
            "car_2h":              ev.car_2h,
            "car_3h":              ev.car_3h,
            "car_4h":              ev.car_4h,
            "car_1d":              ev.car_1d,
            "car_2d":              ev.car_2d,
            "car_5d":              ev.car_5d,
            "volume_ratio":        ev.volume_ratio,
            "passed_materiality":  ev.passed_materiality,
        })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "event_analysis.csv", index=False)
    log.info("Saved event_analysis.csv (%d rows).", len(rows))


def _save_portfolio_csvs(result: "RunResult") -> None:
    if result.hourly_snapshots:
        rows = [{"bar": s.bar, "nav": s.portfolio_value, "n_pos": s.open_positions}
                for s in result.hourly_snapshots]
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "hourly_portfolio.csv", index=False)
        log.info("Saved hourly_portfolio.csv")

    if result.daily_snapshots:
        rows = [{"date": s.bar, "nav": s.portfolio_value, "gross_exp": s.gross_exposure,
                 "n_pos": s.open_positions}
                for s in result.daily_snapshots]
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "daily_portfolio.csv", index=False)
        log.info("Saved daily_portfolio.csv")


def _save_slippage_sensitivity(result: "RunResult") -> None:
    df = slippage_sensitivity(result)
    if not df.empty:
        df.to_csv(OUTPUT_DIR / "slippage_sensitivity.csv", index=False)
        log.info("Saved slippage_sensitivity.csv")
