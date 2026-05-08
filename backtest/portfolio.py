"""Position tracking, P&L calculation, and portfolio-level bookkeeping."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from backtest.config import (
    BARS_PER_DAY,
    COMMISSION_BPS,
    HARD_TO_BORROW,
    INITIAL_CAPITAL,
    MAX_CONCURRENT_POSITIONS,
    MAX_POSITION_PCT,
    SHORT_BORROW_ANNUAL_BPS,
    SHORT_BORROW_ANNUAL_BPS_HARDTOBORROW,
    SLIPPAGE_BPS,
    SLIPPAGE_BPS_SMALLCAP,
    SMALLCAP_VOLUME_THRESHOLD,
    EXIT_CONVERGENCE_THRESHOLD,
    EXIT_STOP_LOSS_PCT,
    EXIT_MAX_HOLDING_HOURS,
)
from backtest.signal_generator import Signal

log = logging.getLogger(__name__)


# ── Trade record ───────────────────────────────────────────────────────────

@dataclass
class Trade:
    trade_id:          int
    signal:            Signal

    # Entry
    entry_bar:         pd.Timestamp
    entry_price_dep:   float
    entry_price_prim:  float   # price of hedge leg (primary or ETF)
    hedge_ticker:      str

    position_size:     float   # notional $ in long leg
    hedge_ratio:       float   # $ short per $ long

    # Slippage / borrow rates applied at entry
    slippage_bps:      int
    borrow_rate_bps:   int

    # Exit
    exit_bar:          Optional[pd.Timestamp] = None
    exit_price_dep:    float = np.nan
    exit_price_prim:   float = np.nan
    exit_reason:       str = ""
    holding_hours:     int = 0

    # P&L
    gross_pnl:         float = 0.0
    txn_cost:          float = 0.0
    borrow_cost:       float = 0.0
    net_pnl:           float = 0.0

    # Convergence score at exit
    score_at_exit:     float = np.nan

    @property
    def is_open(self) -> bool:
        return self.exit_bar is None

    @property
    def direction(self) -> str:
        return self.signal.signal_direction

    def compute_gross_pnl(
        self,
        exit_price_dep:  float,
        exit_price_prim: float,
    ) -> float:
        """
        Gross P&L from entry to exit.
        Long dep, short hedge — dollar neutral.
        """
        if self.direction == "long":
            dep_return  =  (exit_price_dep  - self.entry_price_dep)  / self.entry_price_dep
            prim_return = -(exit_price_prim - self.entry_price_prim) / self.entry_price_prim
        else:  # short the dependent, long the hedge
            dep_return  = -(exit_price_dep  - self.entry_price_dep)  / self.entry_price_dep
            prim_return =  (exit_price_prim - self.entry_price_prim) / self.entry_price_prim

        long_leg_pnl  = dep_return  * self.position_size
        short_leg_pnl = prim_return * self.position_size * self.hedge_ratio
        return long_leg_pnl + short_leg_pnl

    def compute_txn_cost(self) -> float:
        """Round-trip transaction cost across both legs."""
        cost_per_side_bps = COMMISSION_BPS + self.slippage_bps
        # entry + exit, 2 legs each
        return (self.position_size * (1 + self.hedge_ratio)
                * cost_per_side_bps / 10_000 * 2)

    def compute_borrow_cost(self, holding_hours: int) -> float:
        """Short borrow cost prorated to holding period."""
        holding_years = holding_hours / (BARS_PER_DAY * 252)
        short_value = self.position_size * self.hedge_ratio
        return short_value * self.borrow_rate_bps / 10_000 * holding_years

    def close(
        self,
        exit_bar:        pd.Timestamp,
        exit_price_dep:  float,
        exit_price_prim: float,
        reason:          str,
        score_at_exit:   float = np.nan,
    ) -> None:
        self.exit_bar       = exit_bar
        self.exit_price_dep = exit_price_dep
        self.exit_price_prim= exit_price_prim
        self.exit_reason    = reason
        self.score_at_exit  = score_at_exit

        # Holding hours = number of bars between entry and exit
        self.holding_hours  = int(
            (exit_bar - self.entry_bar).total_seconds() / 3600
        )

        self.gross_pnl  = self.compute_gross_pnl(exit_price_dep, exit_price_prim)
        self.txn_cost   = self.compute_txn_cost()
        self.borrow_cost= self.compute_borrow_cost(self.holding_hours)
        self.net_pnl    = self.gross_pnl - self.txn_cost - self.borrow_cost

        log.debug(
            "Trade %d closed: dep=%s dir=%s net_pnl=%.2f reason=%s",
            self.trade_id, self.signal.dependent_ticker,
            self.direction, self.net_pnl, reason,
        )


# ── Portfolio ──────────────────────────────────────────────────────────────

@dataclass
class PortfolioSnapshot:
    bar:              pd.Timestamp
    portfolio_value:  float
    cash:             float
    gross_exposure:   float
    net_exposure:     float
    open_positions:   int


class Portfolio:
    """
    Tracks cash, open positions, and records all trades.
    """

    def __init__(
        self,
        initial_capital:      float = INITIAL_CAPITAL,
        max_concurrent:       int   = MAX_CONCURRENT_POSITIONS,
        max_position_pct:     float = MAX_POSITION_PCT,
        hedge_method:         str   = "primary",    # "primary" or "etf"
    ) -> None:
        self.initial_capital  = initial_capital
        self.cash             = initial_capital
        self.max_concurrent   = max_concurrent
        self.max_position_pct = max_position_pct
        self.hedge_method     = hedge_method

        self._open_trades:  dict[str, Trade] = {}  # keyed by dependent_ticker
        self._all_trades:   list[Trade]      = []
        self._next_id:      int              = 1

        self.hourly_snapshots: list[PortfolioSnapshot] = []
        self.daily_snapshots:  list[PortfolioSnapshot] = []

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def open_position_count(self) -> int:
        return len(self._open_trades)

    @property
    def portfolio_value(self) -> float:
        """Cash + mark-to-market of open positions (unrealised P&L already in cash on entry)."""
        # Positions are dollar-neutral; unrealised P&L not tracked separately — simplified.
        return self.cash

    def has_position(self, ticker: str) -> bool:
        return ticker in self._open_trades

    # ── Entry ───────────────────────────────────────────────────────────────

    def enter_trade(
        self,
        signal:           Signal,
        entry_bar:        pd.Timestamp,
        bar_data:      dict[str, pd.DataFrame],
        sector_etf_map:   dict[str, str],
    ) -> Optional[Trade]:
        """
        Open a new trade for *signal* at *entry_bar*.
        Returns the Trade object, or None if entry is not possible.
        """
        dep_ticker = signal.dependent_ticker
        primary    = signal.primary_ticker

        if self.open_position_count >= self.max_concurrent:
            log.debug("At max capacity, skipping %s.", dep_ticker)
            return None

        if self.has_position(dep_ticker):
            log.debug("Already holding %s, skipping.", dep_ticker)
            return None

        # Entry prices
        dep_price  = _get_close(bar_data, dep_ticker, entry_bar)
        if dep_price is None or dep_price <= 0:
            log.debug("No entry price for %s at %s.", dep_ticker, entry_bar)
            return None

        long_only = self.hedge_method == "none"
        if self.hedge_method == "etf":
            hedge_ticker = sector_etf_map.get(primary, "SPY")
        else:
            hedge_ticker = primary

        if long_only:
            hedge_price = dep_price   # unused but keeps Trade schema happy
        else:
            hedge_price = _get_close(bar_data, hedge_ticker, entry_bar)
            if hedge_price is None or hedge_price <= 0:
                log.debug("No hedge price for %s at %s.", hedge_ticker, entry_bar)
                return None

        # Position size (notional)
        pos_size = self.portfolio_value * self.max_position_pct
        if pos_size <= 0:
            return None

        # Hedge ratio: dollar-neutral by beta (0 when long-only)
        beta         = max(abs(signal.beta), 0.1)
        hedge_ratio  = 0.0 if long_only else beta

        # Slippage
        avg_daily_vol = _avg_daily_volume(bar_data, dep_ticker, entry_bar)
        slippage = (
            SLIPPAGE_BPS_SMALLCAP
            if (avg_daily_vol < SMALLCAP_VOLUME_THRESHOLD or dep_ticker in HARD_TO_BORROW)
            else SLIPPAGE_BPS
        )

        borrow_rate = (
            SHORT_BORROW_ANNUAL_BPS_HARDTOBORROW
            if dep_ticker in HARD_TO_BORROW
            else SHORT_BORROW_ANNUAL_BPS
        )

        # Deduct entry transaction cost from cash immediately
        txn_entry = (pos_size * (1 + hedge_ratio)
                     * (COMMISSION_BPS + slippage) / 10_000)
        self.cash -= txn_entry

        trade = Trade(
            trade_id=self._next_id,
            signal=signal,
            entry_bar=entry_bar,
            entry_price_dep=dep_price,
            entry_price_prim=hedge_price,
            hedge_ticker=hedge_ticker,
            position_size=pos_size,
            hedge_ratio=hedge_ratio,
            slippage_bps=slippage,
            borrow_rate_bps=borrow_rate,
        )
        self._next_id += 1
        self._open_trades[dep_ticker] = trade
        self._all_trades.append(trade)

        log.debug(
            "Trade %d entered: %s dir=%s size=%.0f entry=%.2f",
            trade.trade_id, dep_ticker, signal.signal_direction,
            pos_size, dep_price,
        )
        return trade

    # ── Exit ────────────────────────────────────────────────────────────────

    def check_exits(
        self,
        current_bar:   pd.Timestamp,
        bar_data:   dict[str, pd.DataFrame],
        max_holding:   int   = EXIT_MAX_HOLDING_HOURS,
        eod_exit:      bool  = True,
        convergence_threshold: float = EXIT_CONVERGENCE_THRESHOLD,
        stop_loss_pct: float = EXIT_STOP_LOSS_PCT,
        signal_lookup: Optional[dict] = None,  # dep_ticker → latest underreaction score
    ) -> list[Trade]:
        """
        Evaluate exit conditions for all open trades.  Returns closed trades.
        """
        closed: list[Trade] = []
        to_close: list[tuple[str, str]] = []  # (dep_ticker, reason)

        # EOD = last bar of the session (15:30 for hourly, whatever for daily).
        is_eod_bar = current_bar.strftime("%H:%M") >= "15:30"

        for dep_ticker, trade in list(self._open_trades.items()):
            # Count elapsed *trading bars*, not calendar hours — overnight gaps
            # must not inflate holding duration past max_holding.
            dep_index = bar_data[dep_ticker].index if dep_ticker in bar_data else None
            if dep_index is not None:
                try:
                    entry_pos   = dep_index.get_loc(trade.entry_bar)
                    current_pos = dep_index.get_loc(current_bar)
                    hours_held  = max(1, current_pos - entry_pos)
                except KeyError:
                    hours_held = max(
                        1,
                        int((current_bar - trade.entry_bar).total_seconds() / 3600),
                    )
            else:
                hours_held = max(
                    1,
                    int((current_bar - trade.entry_bar).total_seconds() / 3600),
                )

            dep_price  = _get_close(bar_data, dep_ticker, current_bar)
            hedge_price= _get_close(bar_data, trade.hedge_ticker, current_bar)

            if dep_price is None or hedge_price is None:
                continue  # no data yet

            # Unrealised P&L (gross, before exit costs)
            gross = trade.compute_gross_pnl(dep_price, hedge_price)
            unrealised_pct = gross / trade.position_size

            reason = ""
            if hours_held >= max_holding:
                reason = "max_holding"
            elif eod_exit and is_eod_bar:
                reason = "eod"
            elif unrealised_pct <= -stop_loss_pct:
                reason = "stop_loss"
            elif signal_lookup:
                current_score = signal_lookup.get(dep_ticker, np.nan)
                if not np.isnan(current_score) and abs(current_score) < convergence_threshold:
                    reason = "convergence"

            if reason:
                to_close.append((dep_ticker, reason))

        for dep_ticker, reason in to_close:
            trade = self._open_trades.pop(dep_ticker)
            dep_price   = _get_close(bar_data, dep_ticker, current_bar) or trade.entry_price_dep
            hedge_price = _get_close(bar_data, trade.hedge_ticker, current_bar) or trade.entry_price_prim
            trade.close(current_bar, dep_price, hedge_price, reason)
            # Credit net P&L to cash (add back gross, exit txn cost already deducted inside close())
            exit_txn = (trade.position_size * (1 + trade.hedge_ratio)
                        * (COMMISSION_BPS + trade.slippage_bps) / 10_000)
            self.cash += trade.gross_pnl - exit_txn - trade.borrow_cost
            closed.append(trade)

        return closed

    def force_close_all(
        self,
        current_bar: pd.Timestamp,
        bar_data: dict[str, pd.DataFrame],
    ) -> None:
        """Close all remaining positions at the end of the backtest."""
        for dep_ticker in list(self._open_trades.keys()):
            trade      = self._open_trades.pop(dep_ticker)
            dep_price  = _get_close(bar_data, dep_ticker, current_bar) or trade.entry_price_dep
            hedge_price= _get_close(bar_data, trade.hedge_ticker, current_bar) or trade.entry_price_prim
            trade.close(current_bar, dep_price, hedge_price, "backtest_end")
            exit_txn = (trade.position_size * (1 + trade.hedge_ratio)
                        * (COMMISSION_BPS + trade.slippage_bps) / 10_000)
            self.cash += trade.gross_pnl - exit_txn - trade.borrow_cost

    # ── Snapshot ─────────────────────────────────────────────────────────────

    def take_snapshot(self, bar: pd.Timestamp) -> PortfolioSnapshot:
        gross_exp = sum(
            t.position_size * (1 + t.hedge_ratio)
            for t in self._open_trades.values()
        )
        snap = PortfolioSnapshot(
            bar=bar,
            portfolio_value=self.cash,
            cash=self.cash,
            gross_exposure=gross_exp,
            net_exposure=0.0,   # dollar-neutral by design
            open_positions=self.open_position_count,
        )
        return snap

    # ── Accessors ─────────────────────────────────────────────────────────

    @property
    def closed_trades(self) -> list[Trade]:
        return [t for t in self._all_trades if not t.is_open]

    @property
    def all_trades(self) -> list[Trade]:
        return self._all_trades


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_close(
    bar_data: dict[str, pd.DataFrame],
    ticker: str,
    bar: pd.Timestamp,
) -> Optional[float]:
    if ticker not in bar_data:
        return None
    df = bar_data[ticker]
    if bar in df.index:
        v = df.loc[bar, "close"]
        return float(v) if pd.notna(v) and v > 0 else None
    # Try ffill: last available bar up to this timestamp
    prior = df.loc[df.index <= bar, "close"].dropna()
    if prior.empty:
        return None
    return float(prior.iloc[-1])


def _avg_daily_volume(
    bar_data: dict[str, pd.DataFrame],
    ticker: str,
    as_of: pd.Timestamp,
    lookback: int = 20,
) -> float:
    if ticker not in bar_data:
        return 0.0
    df = bar_data[ticker]
    prior = df.loc[df.index < as_of, "volume"]
    if prior.empty:
        return 0.0
    daily_vol = prior.resample("1D").sum()
    daily_vol = daily_vol[daily_vol > 0].tail(lookback)
    return float(daily_vol.mean()) if not daily_vol.empty else 0.0
