"""
Threshold sensitivity analysis — in-sample data only (2016-2020).
Tests thresholds from 0.05 to 1.50 and plots key metrics vs threshold.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))

from compare import load_cached_bars
from backtest.data_loader import _bars_to_daily
from backtest.event_detector import detect_events, load_events
from backtest.backtest_engine import run_backtest
from backtest.analytics import compute_summary_stats

IS_EVENTS_FILE = Path("backtest_events_2016_2020_v2.xlsx")
OUTPUT_DIR = Path("output")

logging.basicConfig(level=logging.WARNING)   # suppress noise during sweep

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00, 1.25, 1.50]

FIXED_PARAMS = dict(
    entry_delay_hours=2,
    holding_hours=70,
    hedge_method="none",
    signal_model="beta_revenue",
)



print("Loading cached bars …")
bar_data   = load_cached_bars()
daily_data = _bars_to_daily(bar_data)
raw_events = load_events(IS_EVENTS_FILE)
event_results = detect_events(raw_events, bar_data, daily_data)
print(f"  {len(bar_data)} tickers  |  {len(raw_events)} events")



rows = []
for thr in THRESHOLDS:
    result = run_backtest(
        bar_data=bar_data,
        daily_data=daily_data,
        event_results=event_results,
        threshold=thr,
        **FIXED_PARAMS,
    )
    stats = compute_summary_stats(result)
    rows.append({
        "threshold":    thr,
        "n_trades":     stats["n_trades"],
        "total_return": stats["total_return"] * 100,
        "ann_return":   stats["ann_return"] * 100,
        "sharpe":       stats["sharpe"],
        "sortino":      stats["sortino"],
        "max_dd":       stats["max_drawdown"] * 100,
        "win_rate":     stats["win_rate"] * 100,
        "profit_factor":stats["profit_factor"],
        "avg_pnl":      stats["avg_net_pnl"],
    })
    print(f"  thr={thr:.2f}  n={stats['n_trades']:>4}  "
          f"ret={stats['total_return']*100:+.2f}%  "
          f"sharpe={stats['sharpe']:+.3f}  "
          f"wr={stats['win_rate']*100:.1f}%")

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_DIR / "threshold_sensitivity_is.csv", index=False)



fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.32)

current_thr = 0.30

def vline(ax):
    ax.axvline(current_thr, color="#e74c3c", lw=1.4, linestyle="--", alpha=0.7, label="Current (0.30)")

# 1. Sharpe
ax = fig.add_subplot(gs[0, 0])
ax.plot(df["threshold"], df["sharpe"], "o-", color="#3498db", lw=2, ms=6)
ax.axhline(0, color="gray", lw=0.8, linestyle="--")
vline(ax)
ax.set_title("Sharpe Ratio", fontweight="bold")
ax.set_xlabel("Threshold"); ax.set_ylabel("Sharpe")
ax.legend(fontsize=8)

# 2. Total Return
ax = fig.add_subplot(gs[0, 1])
bar_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df["total_return"]]
ax.bar(df["threshold"], df["total_return"], color=bar_colors, alpha=0.8, width=0.03, edgecolor="white")
ax.axhline(0, color="gray", lw=0.8, linestyle="--")
vline(ax)
ax.set_title("Total Return (%)", fontweight="bold")
ax.set_xlabel("Threshold"); ax.set_ylabel("Return %")

# 3. N Trades
ax = fig.add_subplot(gs[0, 2])
ax.plot(df["threshold"], df["n_trades"], "s-", color="#9b59b6", lw=2, ms=6)
vline(ax)
ax.set_title("Number of Trades", fontweight="bold")
ax.set_xlabel("Threshold"); ax.set_ylabel("N Trades")

# 4. Win Rate
ax = fig.add_subplot(gs[1, 0])
ax.plot(df["threshold"], df["win_rate"], "o-", color="#e67e22", lw=2, ms=6)
ax.axhline(50, color="gray", lw=0.8, linestyle="--", label="50%")
vline(ax)
ax.set_title("Win Rate (%)", fontweight="bold")
ax.set_xlabel("Threshold"); ax.set_ylabel("Win Rate %")
ax.legend(fontsize=8)

# 5. Profit Factor
ax = fig.add_subplot(gs[1, 1])
ax.plot(df["threshold"], df["profit_factor"], "o-", color="#1abc9c", lw=2, ms=6)
ax.axhline(1, color="gray", lw=0.8, linestyle="--", label="Break-even")
vline(ax)
ax.set_title("Profit Factor", fontweight="bold")
ax.set_xlabel("Threshold"); ax.set_ylabel("Profit Factor")
ax.legend(fontsize=8)

# 6. Max Drawdown
ax = fig.add_subplot(gs[1, 2])
ax.plot(df["threshold"], df["max_dd"], "o-", color="#e74c3c", lw=2, ms=6)
vline(ax)
ax.set_title("Max Drawdown (%)", fontweight="bold")
ax.set_xlabel("Threshold"); ax.set_ylabel("Drawdown %")

fig.suptitle("Threshold Sensitivity Analysis — In-Sample (2016–2020)", fontsize=14, fontweight="bold")

out_path = OUTPUT_DIR / "threshold_sensitivity_is.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved → {out_path}")
print("\n=== Full Results ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

best_sharpe = df.loc[df["sharpe"].idxmax()]
best_return = df.loc[df["total_return"].idxmax()]
print(f"\nBest Sharpe : threshold={best_sharpe['threshold']:.2f}  sharpe={best_sharpe['sharpe']:.3f}")
print(f"Best Return : threshold={best_return['threshold']:.2f}  return={best_return['total_return']:.2f}%")
