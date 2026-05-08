"""All strategy parameters, dependency map, and event configuration."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
EVENTS_FILE = ROOT_DIR / "backtest_events_2021_2025.xlsx"

# ── Universe ───────────────────────────────────────────────────────────────
BACKTEST_START = "2021-01-01"
BACKTEST_END = "2025-05-15"

ENERGY_PRIMARIES = ["CVX", "XOM", "OXY", "COP"]
DEFENSE_PRIMARIES = ["LMT", "NOC", "GD", "RTX"]
ALL_PRIMARIES = ENERGY_PRIMARIES + DEFENSE_PRIMARIES

# Back-compat aliases for semi-era code paths that moved to these names
FABLESS_PRIMARIES = ENERGY_PRIMARIES
FOUNDRY_PRIMARIES = DEFENSE_PRIMARIES
WFE_PRIMARIES: list[str] = []

SECTOR_ETF = {
    "CVX": "XLE", "XOM": "XLE", "OXY": "XLE", "COP": "XLE",
    "LMT": "ITA", "NOC": "ITA", "GD": "ITA", "RTX": "ITA",
}

DEPENDENCY_MAP: dict[str, list[dict]] = {
    "CVX": [
        {"ticker": "DXC",  "pct": 0.310, "type": "IT services"},
        {"ticker": "MUR",  "pct": 0.190, "type": "E&P peer"},
        {"ticker": "RIG",  "pct": 0.149, "type": "offshore driller"},
        {"ticker": "TPL",  "pct": 0.110, "type": "Permian royalties"},
    ],
    "XOM": [
        {"ticker": "PAGP", "pct": 0.300,  "type": "PAA GP"},
        {"ticker": "IMO",  "pct": 0.2994, "type": "Canadian subsidiary"},
        {"ticker": "PUMP", "pct": 0.249,  "type": "Permian frac"},
        {"ticker": "PVL",  "pct": 0.230,  "type": "royalty trust"},
    ],
    "OXY": [
        {"ticker": "TPL",  "pct": 0.190,  "type": "Permian royalties"},
        {"ticker": "PUMP", "pct": 0.137,  "type": "Permian frac"},
        {"ticker": "STR",  "pct": 0.100,  "type": "mineral rights"},
    ],
    "COP": [
        {"ticker": "ARIS", "pct": 0.2167, "type": "water disposal"},
        {"ticker": "CRGY", "pct": 0.135,  "type": "E&P"},
        {"ticker": "SFL",  "pct": 0.130,  "type": "shipping"},
        {"ticker": "STR",  "pct": 0.110,  "type": "mineral rights"},
        {"ticker": "TPL",  "pct": 0.100,  "type": "Permian royalties"},
    ],
    "LMT": [
        {"ticker": "AIRI", "pct": 0.302,  "type": "aerostructures"},
        {"ticker": "CVU",  "pct": 0.240,  "type": "aircraft assemblies"},
        {"ticker": "EMKR", "pct": 0.2391, "type": "fiber optics"},
        {"ticker": "SIF",  "pct": 0.1108, "type": "forgings"},
        {"ticker": "TTMI", "pct": 0.1105, "type": "PCBs"},
    ],
    "NOC": [
        {"ticker": "FEIM", "pct": 0.3959, "type": "timing systems"},
        {"ticker": "SYPR", "pct": 0.230,  "type": "electronics"},
        {"ticker": "MRCY", "pct": 0.120,  "type": "mission computing"},
        {"ticker": "CVU",  "pct": 0.080,  "type": "aircraft assemblies"},
    ],
    "GD": [
        {"ticker": "OPXS", "pct": 0.1848, "type": "optical sighting"},
        {"ticker": "GHM",  "pct": 0.1819, "type": "heat exchangers"},
    ],
    "RTX": [
        {"ticker": "AIRI", "pct": 0.382,  "type": "aerostructures"},
        {"ticker": "CVU",  "pct": 0.360,  "type": "aircraft assemblies"},
        {"ticker": "MTX",  "pct": 0.2475, "type": "defense materials"},
        {"ticker": "MRCY", "pct": 0.190,  "type": "mission computing"},
    ],
}

# All unique tickers needed
ALL_DEPENDENTS: list[str] = sorted({
    dep["ticker"]
    for deps in DEPENDENCY_MAP.values()
    for dep in deps
})
ALL_TICKERS = ALL_PRIMARIES + ALL_DEPENDENTS + ["SPY", "XLE", "ITA"]

# ── Event timing ───────────────────────────────────────────────────────────
EVENT_TIMING: dict[str, str] = {
    "Earnings":     "after_close",
    "Oil Price":    "intraday",
    "Oil Price War":"pre_market",
    "Regulatory":   "intraday",
    "Geopolitical": "pre_market",
    "Corporate":    "mixed",
    "Macro":        "intraday",
    "M&A":          "mixed",
    "Contract":     "intraday",
}

# ── Ticker edge-cases ──────────────────────────────────────────────────────
TICKER_MAPPING = {
    "RTX": {"pre_merger_ticker": "UTX", "merger_date": "2020-04-03"},
}

TICKER_IPO_DATES: dict[str, str] = {
    "PUMP": "2017-03-17",
    "TALO": "2017-05-11",
    "ARIS": "2021-10-22",
    "CRGY": "2021-12-07",
}

HARD_TO_BORROW = ["AIRI", "CVU", "FEIM", "SYPR", "OPXS", "NGS", "SIF", "PVL", "EMKR"]

# ── Bar / time parameters ──────────────────────────────────────────────────
BAR_FREQUENCY = "1H"
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"
BARS_PER_DAY_HOURLY = 7
BARS_PER_DAY_DAILY  = 1
BARS_PER_DAY = BARS_PER_DAY_HOURLY

# ── Signal thresholds ──────────────────────────────────────────────────────
UNDERREACTION_THRESHOLD = 0.30
OVERREACTION_THRESHOLD = -0.30
MIN_EXPECTED_RETURN = 0.001

# ── Materiality filter ─────────────────────────────────────────────────────
MATERIALITY_CAR_2H = 0.010
MATERIALITY_CAR_1D = 0.015
MATERIALITY_VOLUME_RATIO = 2.5
VOLUME_LOOKBACK_DAYS = 20

# ── Entry / exit ───────────────────────────────────────────────────────────
ENTRY_DELAY_HOURS = [1, 2, 3, 4, 7]
HOLDING_PERIODS_HOURS = [1, 2, 3, 4, 7, 14, 21, 35, 49, 70]
EXIT_CONVERGENCE_THRESHOLD = 0.10
EXIT_STOP_LOSS_PCT = 0.03
EXIT_MAX_HOLDING_HOURS = 70     # 10 trading days × 7 bars
EXIT_END_OF_DAY = False         # ← hold across days so we can see reversion

# ── Position sizing ────────────────────────────────────────────────────────
MAX_POSITION_PCT = 0.02
MAX_CONCURRENT_POSITIONS = 10
INITIAL_CAPITAL = 1_000_000

# ── Beta ───────────────────────────────────────────────────────────────────
BETA_LOOKBACK_DAYS = 120

# ── Event-targeted download window ─────────────────────────────────────────
# Need ≥ holding_hours/7 trading days AFTER event to observe the full tail.
EVENT_WINDOW_PRE_DAYS  = 3
EVENT_WINDOW_POST_DAYS = 10     # ← extended from 5 to 10 to test late reversion

# ── Transaction costs ──────────────────────────────────────────────────────
COMMISSION_BPS = 5
SLIPPAGE_BPS = 10
SLIPPAGE_BPS_SMALLCAP = 25
SMALLCAP_VOLUME_THRESHOLD = 100_000
ILLIQUID_HOURLY_THRESHOLD = 5_000
ZERO_VOLUME_SKIP_BARS = 4

SHORT_BORROW_ANNUAL_BPS = 50
SHORT_BORROW_ANNUAL_BPS_HARDTOBORROW = 300

# ── Parameter sweep ────────────────────────────────────────────────────────
SWEEP_PARAMS: dict[str, list] = {
    "entry_delay_hours":        [1, 2, 4, 7],
    "holding_period_hours":     [4, 7, 14, 21, 35],
    "hedge_method":             ["primary", "etf"],
    "signal_model":             ["beta_only", "beta_revenue", "vol_based"],
    "underreaction_threshold":  [0.20, 0.30, 0.40, 0.50],
}

# ── Default run parameters ─────────────────────────────────────────────────
DEFAULT_ENTRY_DELAY = 2
DEFAULT_HOLDING_HOURS = 35      # ← was 7; now full 5-day observation
DEFAULT_HEDGE = "primary"
DEFAULT_MODEL = "beta_revenue"
DEFAULT_THRESHOLD = 0.30
RISK_FREE_RATE = 0.02
