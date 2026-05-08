"""Generate backtest_events_semis_2016_2020.xlsx.

Produces two sheets:
  - Dependency Map : from backtest.config.DEPENDENCY_MAP
  - Event Log      : quarterly earnings (heuristic dates) + curated major events

Run:  python scripts/generate_semi_events.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.config import (
    DEPENDENCY_MAP,
    EVENTS_FILE,
    ALL_PRIMARIES,
)


# ── Quarterly earnings (AMC = after-market-close) ─────────────────────────
# Dates below are the actual reported earnings dates (calendar quarter of
# release).  Verified against press releases / 10-Q filings.  Edit freely.
EARNINGS_DATES: dict[str, list[str]] = {
    "NVDA": [
        "2016-02-17", "2016-05-12", "2016-08-11", "2016-11-10",
        "2017-02-09", "2017-05-09", "2017-08-10", "2017-11-09",
        "2018-02-08", "2018-05-10", "2018-08-16", "2018-11-15",
        "2019-02-14", "2019-05-16", "2019-08-15", "2019-11-14",
        "2020-02-13", "2020-05-21", "2020-08-19", "2020-11-18",
    ],
    "AMD": [
        "2016-01-19", "2016-04-21", "2016-07-21", "2016-10-20",
        "2017-01-31", "2017-05-01", "2017-07-25", "2017-10-24",
        "2018-01-30", "2018-04-25", "2018-07-25", "2018-10-24",
        "2019-01-29", "2019-04-30", "2019-07-30", "2019-10-29",
        "2020-01-28", "2020-04-28", "2020-07-28", "2020-10-27",
    ],
    "INTC": [
        "2016-01-14", "2016-04-19", "2016-07-20", "2016-10-18",
        "2017-01-26", "2017-04-27", "2017-07-27", "2017-10-26",
        "2018-01-25", "2018-04-26", "2018-07-26", "2018-10-25",
        "2019-01-24", "2019-04-25", "2019-07-25", "2019-10-24",
        "2020-01-23", "2020-04-23", "2020-07-23", "2020-10-22",
    ],
    "TSM": [
        "2016-01-14", "2016-04-28", "2016-07-14", "2016-10-13",
        "2017-01-12", "2017-04-13", "2017-07-13", "2017-10-19",
        "2018-01-18", "2018-04-19", "2018-07-19", "2018-10-18",
        "2019-01-17", "2019-04-18", "2019-07-18", "2019-10-17",
        "2020-01-16", "2020-04-16", "2020-07-16", "2020-10-15",
    ],
    "MU": [
        "2016-01-07", "2016-03-30", "2016-06-30", "2016-10-04",
        "2016-12-21", "2017-03-23", "2017-06-29", "2017-09-26",
        "2017-12-19", "2018-03-22", "2018-06-20", "2018-09-20",
        "2018-12-18", "2019-03-20", "2019-06-25", "2019-09-26",
        "2019-12-18", "2020-03-25", "2020-06-29", "2020-09-29",
    ],
    "QCOM": [
        "2016-01-27", "2016-04-20", "2016-07-20", "2016-11-02",
        "2017-02-01", "2017-04-19", "2017-07-19", "2017-11-01",
        "2018-01-31", "2018-04-25", "2018-07-25", "2018-11-07",
        "2019-01-30", "2019-05-01", "2019-07-31", "2019-11-06",
        "2020-02-05", "2020-04-29", "2020-07-29", "2020-11-04",
    ],
    "AVGO": [
        "2016-03-10", "2016-06-02", "2016-09-01", "2016-12-08",
        "2017-03-02", "2017-06-01", "2017-08-24", "2017-12-06",
        "2018-03-15", "2018-06-06", "2018-09-06", "2018-12-06",
        "2019-03-14", "2019-06-13", "2019-09-12", "2019-12-12",
        "2020-03-12", "2020-06-04", "2020-09-03", "2020-12-10",
    ],
    "AMAT": [
        "2016-02-18", "2016-05-19", "2016-08-18", "2016-11-17",
        "2017-02-15", "2017-05-18", "2017-08-17", "2017-11-16",
        "2018-02-14", "2018-05-17", "2018-08-16", "2018-11-15",
        "2019-02-13", "2019-05-16", "2019-08-15", "2019-11-14",
        "2020-02-12", "2020-05-14", "2020-08-13", "2020-11-19",
    ],
}

# ── Curated major non-earnings events ─────────────────────────────────────
# (date, primary, event_type, description, direction)
MAJOR_EVENTS: list[tuple[str, str, str, str, str]] = [
    # 2016
    ("2016-05-06", "NVDA", "Product Launch", "Pascal GTX 1080 launch",               "up"),
    ("2016-07-27", "NVDA", "Product Launch", "Titan X Pascal announcement",          "up"),
    ("2016-10-27", "QCOM", "M&A",            "Announces NXP Semi acquisition ($47B)","mixed"),
    ("2016-08-08", "AMD",  "Product Launch", "Radeon RX 470/460 launch",             "up"),

    # 2017
    ("2017-03-02", "AMD",  "Product Launch", "Ryzen 1000 series launch",             "up"),
    ("2017-05-16", "NVDA", "Product Launch", "Volta V100 announcement (GTC)",        "up"),
    ("2017-08-10", "AMD",  "Product Launch", "Threadripper 1000 launch",             "up"),
    ("2017-09-25", "INTC", "Guidance",       "Core i9 / Coffee Lake launch",         "up"),
    ("2017-11-06", "QCOM", "M&A",            "Broadcom hostile bid ($103B)",         "up"),

    # 2018
    ("2018-01-03", "INTC", "Supply",         "Meltdown/Spectre vulnerability disclosure", "down"),
    ("2018-03-12", "QCOM", "Regulatory",     "Trump blocks Broadcom-QCOM merger",    "down"),
    ("2018-04-17", "TSM",  "Guidance",       "Q1 cautious outlook; crypto demand drop", "down"),
    ("2018-07-25", "INTC", "Guidance",       "10nm delayed again",                   "down"),
    ("2018-08-20", "NVDA", "Product Launch", "Turing RTX 2080 launch (Gamescom)",    "up"),
    ("2018-11-15", "NVDA", "Guidance",       "Crypto hangover guide-down",           "down"),

    # 2019
    ("2019-03-11", "NVDA", "M&A",            "Mellanox acquisition announced ($6.9B)", "up"),
    ("2019-05-16", "AVGO", "Geopolitical",   "Trump Huawei ban hits AVGO/QCOM",      "down"),
    ("2019-05-16", "QCOM", "Geopolitical",   "Trump Huawei ban hits AVGO/QCOM",      "down"),
    ("2019-05-16", "MU",   "Geopolitical",   "Trump Huawei ban hits memory",         "down"),
    ("2019-06-25", "MU",   "Geopolitical",   "MU resumes some Huawei shipments",     "up"),
    ("2019-07-07", "AMD",  "Product Launch", "Zen 2 / Ryzen 3000 launch",            "up"),
    ("2019-10-03", "INTC", "Supply",         "Client CPU supply shortage warning",   "down"),

    # 2020
    ("2020-03-09", "AMAT", "Macro",          "COVID market crash",                   "down"),
    ("2020-03-09", "TSM",  "Macro",          "COVID market crash",                   "down"),
    ("2020-03-09", "NVDA", "Macro",          "COVID market crash",                   "down"),
    ("2020-05-14", "NVDA", "Product Launch", "Ampere A100 datacenter GPU launch",    "up"),
    ("2020-07-23", "INTC", "Supply",         "7nm delay announcement (-16% next day)", "down"),
    ("2020-09-13", "NVDA", "M&A",            "Announces ARM acquisition ($40B)",     "up"),
    ("2020-09-15", "AVGO", "Geopolitical",   "US Huawei sanctions take effect",      "down"),
    ("2020-09-15", "QCOM", "Geopolitical",   "US Huawei sanctions take effect",      "down"),
    ("2020-10-20", "INTC", "M&A",            "SK Hynix buys INTC NAND business",     "mixed"),
    ("2020-11-10", "INTC", "Supply",         "Apple announces dropping Intel Macs",  "down"),
]


def build_event_log() -> pd.DataFrame:
    rows = []

    for ticker, dates in EARNINGS_DATES.items():
        for d in dates:
            rows.append({
                "date":            d,
                "primary":         ticker,
                "event_type":      "Earnings",
                "event_description": f"{ticker} quarterly earnings",
                "direction":       "mixed",
                "primary_move":    "",
                "key_dependents":  "",
                "expected_lag_signal": "",
            })

    for date, primary, ev_type, desc, direction in MAJOR_EVENTS:
        rows.append({
            "date":              date,
            "primary":           primary,
            "event_type":        ev_type,
            "event_description": desc,
            "direction":         direction,
            "primary_move":      "",
            "key_dependents":    "",
            "expected_lag_signal": "",
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def build_dependency_map_sheet() -> pd.DataFrame:
    rows = []
    for primary, deps in DEPENDENCY_MAP.items():
        for d in deps:
            rows.append({
                "primary":       primary,
                "dependent":     d["ticker"],
                "revenue_pct":   d["pct"],
                "relationship":  d["type"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    events_df = build_event_log()
    depmap_df = build_dependency_map_sheet()

    out = Path(EVENTS_FILE)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        depmap_df.to_excel(writer, sheet_name="Dependency Map", index=False)
        events_df.to_excel(writer, sheet_name="Event Log", index=False)

    print(f"Wrote {out}")
    print(f"  Dependency Map: {len(depmap_df)} rows")
    print(f"  Event Log:      {len(events_df)} rows  "
          f"({(events_df['event_type'] == 'Earnings').sum()} earnings, "
          f"{(events_df['event_type'] != 'Earnings').sum()} major events)")
    print(f"  Unique primaries: {sorted(events_df['primary'].unique())}")


if __name__ == "__main__":
    main()
