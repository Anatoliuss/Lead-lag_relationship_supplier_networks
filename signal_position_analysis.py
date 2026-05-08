"""
Scatter plot: signal strength vs trade return.
Evaluates whether dynamic position sizing (proportional to |score|) adds value.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pathlib import Path

OUTPUT_DIR = Path("output")
TRADES_CSV = OUTPUT_DIR / "trades.csv"

# ── Load ──────────────────────────────────────────────────────────────────────

df = pd.read_csv(TRADES_CSV, parse_dates=["entry_bar", "exit_bar", "event_date"])

# Return as % of position size (gross return on capital deployed)
df["return_pct"] = df["net_pnl"] / df["position_size"] * 100

# Absolute score (magnitude of underreaction signal)
df["abs_score"] = df["score_at_entry"].abs()

# Cap extreme scores for readability (99th percentile)
score_cap = df["abs_score"].quantile(0.99)
df_plot = df[df["abs_score"] <= score_cap].copy()

# ── Regression ────────────────────────────────────────────────────────────────

slope, intercept, r_value, p_value, se = stats.linregress(
    df_plot["abs_score"], df_plot["return_pct"]
)
x_line = np.linspace(df_plot["abs_score"].min(), df_plot["abs_score"].max(), 200)
y_line = slope * x_line + intercept

# ── Bucket analysis ───────────────────────────────────────────────────────────

df_plot["score_bucket"] = pd.qcut(df_plot["abs_score"], q=5, labels=["Q1\n(weakest)", "Q2", "Q3", "Q4", "Q5\n(strongest)"])
bucket_stats = df_plot.groupby("score_bucket", observed=True)["return_pct"].agg(
    avg_return="mean",
    median_return="median",
    win_rate=lambda x: (x > 0).mean(),
    n="count",
)

# ── Dynamic sizing simulation ─────────────────────────────────────────────────
# Flat: 2% NAV per trade (position_size already reflects this)
# Dynamic: scale position by abs_score / median_score, capped at 4%

INITIAL_NAV = 1_000_000.0
MAX_ALLOC_PCT = 0.04  # cap at 4% per trade
BASE_ALLOC_PCT = 0.02

median_score = df_plot["abs_score"].median()

df_plot = df_plot.copy()
df_plot["dynamic_alloc_pct"] = (df_plot["abs_score"] / median_score * BASE_ALLOC_PCT).clip(
    upper=MAX_ALLOC_PCT
)
df_plot["dynamic_position"] = df_plot["dynamic_alloc_pct"] * INITIAL_NAV
df_plot["dynamic_pnl"] = df_plot["return_pct"] / 100 * df_plot["dynamic_position"]

flat_total = df_plot["net_pnl"].sum()
dynamic_total = df_plot["dynamic_pnl"].sum()

flat_cumulative = df_plot.sort_values("entry_bar")["net_pnl"].cumsum()
dynamic_cumulative = df_plot.sort_values("entry_bar")["dynamic_pnl"].cumsum()

# ── Plot ──────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

# 1. Scatter: |score| vs return %
ax1 = fig.add_subplot(gs[0, 0])
colors = df_plot["return_pct"].apply(lambda x: "#2ecc71" if x > 0 else "#e74c3c")
ax1.scatter(df_plot["abs_score"], df_plot["return_pct"], c=colors, alpha=0.55, s=30, edgecolors="none")
ax1.plot(x_line, y_line, color="#2c3e50", lw=1.8, label=f"r={r_value:.3f}, p={p_value:.3f}")
ax1.axhline(0, color="gray", lw=0.8, linestyle="--")
ax1.set_xlabel("|Score at Entry|  (underreaction magnitude)", fontsize=10)
ax1.set_ylabel("Net Return % (per trade)", fontsize=10)
ax1.set_title("Signal Strength vs Trade Return", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)
ax1.annotate(f"slope={slope:.4f}\nn={len(df_plot)}", xy=(0.97, 0.05),
             xycoords="axes fraction", ha="right", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

# 2. Bucket bar chart
ax2 = fig.add_subplot(gs[0, 1])
bucket_labels = [str(b) for b in bucket_stats.index]
bar_colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in bucket_stats["avg_return"]]
bars = ax2.bar(bucket_labels, bucket_stats["avg_return"], color=bar_colors, alpha=0.8, edgecolor="white")
ax2.axhline(0, color="gray", lw=0.8, linestyle="--")
ax2.set_xlabel("Score Quintile", fontsize=10)
ax2.set_ylabel("Avg Net Return %", fontsize=10)
ax2.set_title("Avg Return by Signal Quintile", fontsize=12, fontweight="bold")
for bar, (_, row) in zip(bars, bucket_stats.iterrows()):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"n={int(row['n'])}\nWR={row['win_rate']:.0%}", ha="center", va="bottom", fontsize=8)

# 3. Cumulative P&L: flat vs dynamic
ax3 = fig.add_subplot(gs[1, 0])
trade_idx = range(len(flat_cumulative))
ax3.plot(trade_idx, flat_cumulative.values, color="#3498db", lw=1.8, label=f"Flat 2%  (total ${flat_total:,.0f})")
ax3.plot(trade_idx, dynamic_cumulative.values, color="#e67e22", lw=1.8, linestyle="--",
         label=f"Dynamic (total ${dynamic_total:,.0f})")
ax3.axhline(0, color="gray", lw=0.8, linestyle="--")
ax3.set_xlabel("Trade #", fontsize=10)
ax3.set_ylabel("Cumulative Net P&L ($)", fontsize=10)
ax3.set_title("Flat vs Dynamic Sizing — Cumulative P&L", fontsize=12, fontweight="bold")
ax3.legend(fontsize=9)

# 4. Dynamic allocation distribution
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(df_plot["dynamic_alloc_pct"] * 100, bins=20, color="#9b59b6", alpha=0.75, edgecolor="white")
ax4.axvline(BASE_ALLOC_PCT * 100, color="#2c3e50", lw=1.5, linestyle="--", label="Flat 2%")
ax4.set_xlabel("Dynamic Allocation % of NAV", fontsize=10)
ax4.set_ylabel("Number of Trades", fontsize=10)
ax4.set_title("Distribution of Dynamic Position Sizes", fontsize=12, fontweight="bold")
ax4.legend(fontsize=9)

score_range_text = (
    f"Score range: [{df_plot['abs_score'].min():.2f}, {df_plot['abs_score'].quantile(0.95):.2f}] (95th pct)\n"
    f"Median score: {median_score:.2f}  →  Median alloc: {BASE_ALLOC_PCT*100:.1f}%\n"
    f"Cap at {MAX_ALLOC_PCT*100:.0f}%: {(df_plot['dynamic_alloc_pct'] >= MAX_ALLOC_PCT).mean():.0%} of trades"
)
ax4.text(0.97, 0.97, score_range_text, transform=ax4.transAxes, ha="right", va="top",
         fontsize=8, bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

fig.suptitle("Signal Strength → Dynamic Position Sizing Analysis", fontsize=14, fontweight="bold", y=0.98)

out_path = OUTPUT_DIR / "signal_dynamic_sizing.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out_path}")

# ── Console summary ───────────────────────────────────────────────────────────

print("\n=== Signal vs Return Regression ===")
print(f"  Pearson r    : {r_value:.4f}")
print(f"  p-value      : {p_value:.4f}  {'*** significant' if p_value < 0.05 else '(not significant)'}")
print(f"  Slope        : {slope:.5f}  (return % per unit of |score|)")
print(f"  Intercept    : {intercept:.4f}%")

print("\n=== Return by Score Quintile ===")
print(bucket_stats.to_string())

print(f"\n=== Flat vs Dynamic Sizing ===")
print(f"  Flat 2% total P&L    : ${flat_total:>10,.0f}")
print(f"  Dynamic total P&L    : ${dynamic_total:>10,.0f}")
print(f"  Improvement          : ${dynamic_total - flat_total:>+10,.0f}  ({(dynamic_total/flat_total - 1)*100:+.1f}%)")
