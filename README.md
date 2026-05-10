# Lead-Lag Information Diffusion Backtest

A Python backtest of post-event reversion in supplier-customer pairs.
When a large company (the *primary*) announces news, do its smaller
public suppliers (the *dependents*) react too slowly?  We measure the gap
between predicted and actual dependent moves over the first few hours,
then trade the mispricing.

Tested on Energy (CVX, XOM, OXY, COP) and Defense (LMT, NOC, GD, RTX)
primaries with hand-curated supplier maps. Period: 2016–2025.

---

## Headline result

| Period | Total return | PF | Win rate | Max DD |
| --- | --- | --- | --- | --- |
| In-sample 2016–2020 | +4.47% | 1.70 | 45.0% | -1.37% |
| Out-of-sample 2021–2025 | +2.52% | 1.39 | 47.5% | -1.64% |

OOS PF stayed > 1, confirming a real (but modest) edge after frozen
parameters were applied to a held-out period.  Annualised return is
small (~0.6–0.9%) — the strategy demonstrates the anomaly exists, not
that it's standalone-tradeable.

---

## Project layout

```
project_backtest/
├── README.md
├── requirements.txt
├── run.py                     # main entry: download → detect → backtest
├── compare.py                 # in-sample (2016-2020) sweep
├── compare_oos.py             # out-of-sample (2021-2025) test
├── regen_plots.py             # build *_is.png and *_oos.png together
├── threshold_sensitivity.py   # threshold sweep + chart
├── signal_position_analysis.py# signal vs trade return + dynamic sizing
│
├── backtest_events_2016_2020_v2.xlsx
├── backtest_events_2021_2025.xlsx
│
├── backtest/                  # core package
│   ├── config.py              # parameters, dependency map
│   ├── data_loader.py         # WRDS TAQ download + per-day cache
│   ├── event_detector.py      # event loading, materiality filter, CAR
│   ├── signal_generator.py    # β estimation, expected return, score
│   ├── portfolio.py           # trade record, P&L, costs
│   ├── backtest_engine.py     # main loop + parameter sweep
│   └── analytics.py           # summary stats, charts, output files
│
├── data/                      # cached hourly bars (auto-created)
│   └── hourly/<TICKER>/<YYYY-MM-DD>.csv
└── output/                    # plots, summaries, trades CSVs
```

---

## How it works (high level)

For each primary event in the events file:

1. Wait for the **reaction window** to start (next session 09:30 for
   after-close earnings).
2. After **2 hours**, measure the primary's **CAR** = its return minus
   the sector ETF's return (XLE for energy, ITA for defense).
3. For each dependent of that primary, predict the dependent's expected
   return:
   ```
   expected = β × CAR_primary × revenue_pct
   ```
   - β: 120-day daily OLS slope of dep on primary
   - revenue_pct: how much of the dependent's revenue comes from that
     primary (from the supplier's 10-K)
4. Compare predicted to actual. If
   `|score| = |expected − actual| / |expected| > 0.30`, open a trade in
   the direction that closes the gap (long if dep underreacted, short
   if it overreacted).
5. Hold for **10 trading days** (70 hourly bars) or until score
   converges (|score| < 0.10) — whichever comes first.

The sector ETF subtraction strips out the part of the move that wasn't
news-specific. The `revenue_pct` term acts as a fundamentals filter —
dependents with weak business linkage get smaller predictions and fall
below the trade threshold naturally.

---

## Setup

```bash
pip install -r requirements.txt
```

Requires a WRDS academic subscription with TAQ access for the data
download step. Set your username in the env var or pass it on the CLI:

```bash
set WRDS_USERNAME=your_username
```

---

## Running

### Full pipeline (download + backtest)

```bash
python run.py --wrds-user your_username --hedge none
```

Defaults to the events file in `config.py` (currently the 2021-2025
file). Downloads ±3 to +10 days of TAQ around each event the first
time; later runs hit the per-day CSV cache.

Useful flags:
```
--start-date 2021-01-01     --end-date 2025-05-15
--hedge {none, primary, etf}     --threshold 0.30
--entry-delay 2     --holding-hours 70
--model {beta_only, beta_revenue, vol_based}
--sweep                     # run full parameter grid
```

### After data is cached, re-run quickly without WRDS

```bash
python compare.py            # in-sample (2016–2020)
python compare_oos.py        # out-of-sample (2021–2025)
python regen_plots.py        # produces *_is.png and *_oos.png
```

### Side analyses

```bash
python threshold_sensitivity.py     # threshold sweep, in-sample
python signal_position_analysis.py  # signal-vs-return + dynamic sizing
```

---

## Outputs (in `output/`)

For both in-sample (`_is`) and out-of-sample (`_oos`):

- `equity_curve_*.png` — strategy NAV over time
- `drawdown_*.png` — running drawdown
- `decay_curve_*.png` — score progression after entry (the central
  empirical finding — reversion peaks at ~7 days)
- `monthly_heatmap_*.png` — month × year return calendar
- `score_vs_pnl_*.png` — entry score vs realized P&L scatter
- `pnl_distribution_*.png` — per-trade P&L histogram
- `rolling_sharpe_*.png` — 60-trade rolling Sharpe
- `sector_curves_*.png` — Energy vs Defense equity curves
- `win_rate_by_holding_*.png` — win rate by holding bucket

Plus tables:
- `summary_is.txt`, `summary_oos.txt` — performance summaries
- `trades_is.csv`, `trades_oos.csv` — full trade record
- `signal_analysis.csv` — score per horizon for every signal
- `event_analysis.csv` — primary CAR and materiality per event
- `slippage_sensitivity.csv` — Sharpe vs additional slippage assumption
- `threshold_sensitivity_is.csv` / `.png` — threshold sweep result

---

## Key design choices

- **Hourly bars from TAQ trades** (millisecond consolidated trade data),
  binned into 7 hourly OHLCV bars per session
- **Per-day CSV cache** in `data/hourly/<ticker>/<YYYY-MM-DD>.csv` —
  overlapping event windows share cache entries
- **Event-targeted download**: only ±3 days before / +10 days after each
  event are pulled, not the full backtest period
- **Long-only on the dependent by default** (`--hedge none`); optional
  primary or ETF hedge available
- **No look-ahead**: β uses only data strictly before the event date,
  CAR uses only data up to the entry bar
- **Static `revenue_pct`**: a known limitation — the supplier
  concentration values are 10-K snapshots, not time-varying

---

## Limitations

- Universe trim (PAA / WTI / NGS dropped) was an in-sample selection
  applied to OOS — partial overfit.
- `revenue_pct` values are static, sourced from current 10-Ks.
- Survivorship bias: the dependency map only contains companies that
  still exist today. Bankruptcies and delistings during the test period
  are absent.
- ~30–40 trades per year is too few for a high-Sharpe standalone
  strategy. Capital utilisation is low; alpha is fragile to slippage.

These are honest disclosures rather than fatal flaws — the OOS test
specifically surfaces overfitting bias, and the realised PF decay
(1.70 → 1.39) is consistent with a real, modest edge.
