"""
XEMM Perpetual Tiered Scanner v3 — Auto-Discovery
===================================================

Automatically discovers all USDT perpetual pairs available on BOTH
maker and taker exchanges, then scans spreads across the full
synchronized universe.

Auto-discovery
--------------
On startup ``init_markets`` calls exchange REST endpoints to fetch
all active USDT perpetual contracts, finds the intersection (pairs
available on BOTH exchanges), and registers them.

Endpoints used (synchronous, one-time at startup):
  - Binance: fapi/v1/exchangeInfo
  - KuCoin:  api/v1/contracts/active
  - Bitget:  api/v2/mix/market/contracts?productType=USDT-FUTURES

``scan_pairs`` and ``rest_discovery_pairs`` config fields become
optional overrides: if set, they are merged with the auto-discovered
set.  If empty, only auto-discovered pairs are used.

Two-tier scanning
-----------------
ALL discovered pairs get WS order book subscriptions via init_markets.
A background REST scanner polls bookTicker endpoints to provide a
coarse spread ranking signal.  Pairs are activated for trading based
on this ranking, using real-time WS data for execution.

Lazy per-pair setup, auto-promote, pair rotation, hysteresis, imbalance
cap, hanging-hedge quarantine, drawdown halt, and perp-setup validation
are identical to v2.

Sizing model
------------
    total_notional = total_amount_quote * leverage * budget_utilization
    per_pair       = total_notional / max_active_pairs
    per_side       = per_pair / 2
    level_notional = per_side * (level_weight / sum(level_weights_on_that_side))
"""

import asyncio
import logging
import os
import threading
import time
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple, Union

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from pydantic import Field, field_validator

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import (
    OrderType,
    PositionAction,
    PositionMode,
    PriceType,
    TradeType,
)
from hummingbot.core.data_type.order_candidate import PerpetualOrderCandidate
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase
from hummingbot.strategy_v2.executors.data_types import ConnectorPair
from hummingbot.strategy_v2.executors.xemm_executor.data_types import XEMMExecutorConfig
from hummingbot.strategy_v2.executors.xemm_executor.xemm_executor import XEMMExecutor
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executor_actions import (
    CreateExecutorAction,
    ExecutorAction,
    StopExecutorAction,
)
from hummingbot.strategy_v2.models.executors import CloseType, TrackedOrder


_log = logging.getLogger(__name__)

# Profit bounds around target profitability in _build_executor_action
PROFIT_BOUND_LOWER_BPS = Decimal("0.001")  # 10 bps below target
PROFIT_BOUND_UPPER_BPS = Decimal("0.003")  # 30 bps above target

# ---------------------------------------------------------------------------
# Auto-discovery: async fetch via aiohttp, sync wrapper for init_markets
# ---------------------------------------------------------------------------

async def _aiohttp_get_json(session: aiohttp.ClientSession, url: str) -> Optional[dict]:
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                _log.warning(f"[DISCOVERY] HTTP {resp.status} from {url}")
                return None
            return await resp.json(content_type=None)
    except Exception as e:
        _log.warning(f"[DISCOVERY] aiohttp error {url}: {e}")
        return None


async def _async_fetch_binance_perp_pairs(session: aiohttp.ClientSession) -> Set[str]:
    data = await _aiohttp_get_json(session, "https://fapi.binance.com/fapi/v1/exchangeInfo")
    if not data:
        return set()
    pairs: Set[str] = set()
    for sym in data.get("symbols", []):
        if (
            sym.get("quoteAsset") == "USDT"
            and sym.get("status") == "TRADING"
            and sym.get("contractType") == "PERPETUAL"
        ):
            base = sym.get("baseAsset", "")
            if base:
                pairs.add(f"{base}-USDT")
    return pairs


async def _async_fetch_kucoin_perp_pairs(session: aiohttp.ClientSession) -> Set[str]:
    data = await _aiohttp_get_json(session, "https://api-futures.kucoin.com/api/v1/contracts/active")
    if not isinstance(data, dict) or data.get("code") != "200000":
        return set()
    pairs: Set[str] = set()
    for item in data.get("data", []):
        if item.get("quoteCurrency") == "USDT" and item.get("status") == "Open":
            base = item.get("baseCurrency", "")
            if base:
                base = "BTC" if base == "XBT" else base
                pairs.add(f"{base}-USDT")
    return pairs


async def _async_fetch_bitget_perp_pairs(session: aiohttp.ClientSession) -> Set[str]:
    data = await _aiohttp_get_json(
        session, "https://api.bitget.com/api/v2/mix/market/contracts?productType=USDT-FUTURES"
    )
    if not isinstance(data, dict) or data.get("code") != "00000":
        return set()
    pairs: Set[str] = set()
    for item in data.get("data", []):
        if item.get("quoteCoin") == "USDT":
            base = item.get("baseCoin", "")
            if base:
                pairs.add(f"{base}-USDT")
    return pairs


async def _async_fetch_exchange_pairs(connector_name: str, session: aiohttp.ClientSession) -> Set[str]:
    cn = connector_name.lower()
    if "binance" in cn and "perpetual" in cn:
        return await _async_fetch_binance_perp_pairs(session)
    if "kucoin" in cn and "perpetual" in cn:
        return await _async_fetch_kucoin_perp_pairs(session)
    if "bitget" in cn and "perpetual" in cn:
        return await _async_fetch_bitget_perp_pairs(session)
    _log.warning(f"[DISCOVERY] No parser for {connector_name}")
    return set()


# Helpers for prescan spreads (used in discovery phase)
async def _async_fetch_binance_bookTicker(session: aiohttp.ClientSession) -> Dict[str, Tuple[float, float]]:
    result = {}
    data = await _aiohttp_get_json(session, "https://fapi.binance.com/fapi/v1/ticker/bookTicker")
    if isinstance(data, list):
        for t in data:
            sym = t.get("symbol", "")
            if sym.endswith("USDT"):
                base = sym[:-4]
                pair = f"{base}-USDT"
                bid = float(t.get("bidPrice", 0) or 0)
                ask = float(t.get("askPrice", 0) or 0)
                if bid > 0 and ask > 0:
                    result[pair] = (bid, ask)
    return result


async def _async_fetch_kucoin_allTickers(session: aiohttp.ClientSession) -> Dict[str, Tuple[float, float]]:
    result = {}
    data = await _aiohttp_get_json(session, "https://api-futures.kucoin.com/api/v1/allTickers")
    if isinstance(data, dict) and data.get("code") == "200000":
        for item in data.get("data", []):
            sym = item.get("symbol", "")
            if sym.endswith("USDT"):
                base = sym[:-4]
                pair = f"{base}-USDT"
                bid = float(item.get("bestBidPrice", 0) or 0)
                ask = float(item.get("bestAskPrice", 0) or 0)
                if bid > 0 and ask > 0:
                    result[pair] = (bid, ask)
    return result


async def _async_fetch_bitget_tickers(session: aiohttp.ClientSession) -> Dict[str, Tuple[float, float]]:
    result = {}
    data = await _aiohttp_get_json(
        session, "https://api.bitget.com/api/v2/mix/market/tickers?productType=USDT-FUTURES"
    )
    if isinstance(data, dict) and data.get("code") == "00000":
        for item in data.get("data", []):
            base = item.get("baseCoin", "")
            pair = f"{base}-USDT"
            bid = float(item.get("bidPr", 0) or 0)
            ask = float(item.get("askPr", 0) or 0)
            if bid > 0 and ask > 0:
                result[pair] = (bid, ask)
    return result


async def _async_prescan_spreads(
    session: aiohttp.ClientSession,
    maker_connector: str, taker_connector: str,
    common_pairs: Set[str],
) -> List[Tuple[str, float]]:
    """Fetch bookTicker from both exchanges, compute spread, return sorted desc."""
    maker_books: Dict[str, Tuple[float, float]] = {}
    taker_books: Dict[str, Tuple[float, float]] = {}

    # Fetch maker exchange tickers
    maker_cn = maker_connector.lower()
    if "binance" in maker_cn and "perpetual" in maker_cn:
        maker_books = await _async_fetch_binance_bookTicker(session)
    elif "kucoin" in maker_cn and "perpetual" in maker_cn:
        maker_books = await _async_fetch_kucoin_allTickers(session)
    elif "bitget" in maker_cn and "perpetual" in maker_cn:
        maker_books = await _async_fetch_bitget_tickers(session)

    # Fetch taker exchange tickers
    taker_cn = taker_connector.lower()
    if "binance" in taker_cn and "perpetual" in taker_cn:
        taker_books = await _async_fetch_binance_bookTicker(session)
    elif "kucoin" in taker_cn and "perpetual" in taker_cn:
        taker_books = await _async_fetch_kucoin_allTickers(session)
    elif "bitget" in taker_cn and "perpetual" in taker_cn:
        taker_books = await _async_fetch_bitget_tickers(session)

    spreads: List[Tuple[str, float]] = []
    for pair in common_pairs:
        mk = maker_books.get(pair)
        tk = taker_books.get(pair)
        if mk is None or tk is None:
            continue
        mk_bid, mk_ask = mk
        tk_bid, tk_ask = tk
        buy_spread = (tk_bid - mk_ask) / mk_ask if mk_ask > 0 else 0
        sell_spread = (mk_bid - tk_ask) / tk_ask if tk_ask > 0 else 0
        best = max(buy_spread, sell_spread)
        if best > 0:
            spreads.append((pair, best))

    spreads.sort(key=lambda x: x[1], reverse=True)
    return spreads


async def _async_discover_common_pairs(
    maker_connector: str, taker_connector: str,
    extra_pairs: Optional[Set[str]] = None,
) -> Tuple[Set[str], List[Tuple[str, float]]]:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
        maker_pairs, taker_pairs = await asyncio.gather(
            _async_fetch_exchange_pairs(maker_connector, session),
            _async_fetch_exchange_pairs(taker_connector, session),
        )
        common = maker_pairs & taker_pairs
        if extra_pairs:
            common |= extra_pairs & maker_pairs & taker_pairs
        spread_rank = await _async_prescan_spreads(
            session, maker_connector, taker_connector, common
        )
    _log.info(
        f"[DISCOVERY] maker={len(maker_pairs)} taker={len(taker_pairs)} "
        f"common={len(common)} pairs, spread data for {len(spread_rank)}"
    )
    return common, spread_rank


def _discover_pairs_sync(
    maker_connector: str, taker_connector: str,
    extra_pairs: Optional[Set[str]] = None,
) -> Tuple[Set[str], List[Tuple[str, float]]]:
    """Runs aiohttp discovery in a background thread with its own event loop."""
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp is required for auto_discover_pairs")

    result: list = [(set(), [])]
    exc: list = [None]

    def _worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result[0] = loop.run_until_complete(
                _async_discover_common_pairs(maker_connector, taker_connector, extra_pairs)
            )
        except Exception as e:
            exc[0] = e
        finally:
            loop.close()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=30)
    if exc[0] is not None:
        raise exc[0]
    return result[0]


class PerpXEMMExecutor(XEMMExecutor):

    @classmethod
    def logger(cls):
        if cls._logger is None:
            cls._logger = logging.getLogger("hummingbot.strategy.strategy_v2_base")
        return cls._logger

    @property
    def filled_amount_quote(self) -> Decimal:
        if self.maker_order is None:
            return Decimal("0")
        try:
            executed_quote = self.maker_order.executed_amount_quote
            if executed_quote and executed_quote > 0:
                return executed_quote
            executed_base = self.maker_order.executed_amount_base or Decimal("0")
            avg_price = self.maker_order.average_executed_price or Decimal("0")
            return executed_base * avg_price
        except Exception:
            return Decimal("0")

    async def validate_sufficient_balance(self):
        mid_price = self.get_price(
            self.maker_connector, self.maker_trading_pair, price_type=PriceType.MidPrice
        )
        maker_leverage = self._get_leverage(self.maker_connector, self.maker_trading_pair)
        taker_leverage = self._get_leverage(self.taker_connector, self.taker_trading_pair)

        maker_candidate = PerpetualOrderCandidate(
            trading_pair=self.maker_trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=self.maker_order_side,
            amount=self.config.order_amount,
            price=mid_price,
            leverage=Decimal(str(maker_leverage)),
            position_close=False,
        )
        taker_candidate = PerpetualOrderCandidate(
            trading_pair=self.taker_trading_pair,
            is_maker=False,
            order_type=OrderType.MARKET,
            order_side=self.taker_order_side,
            amount=self.config.order_amount,
            price=mid_price,
            leverage=Decimal(str(taker_leverage)),
            position_close=False,
        )
        maker_adj = self.adjust_order_candidates(self.maker_connector, [maker_candidate])[0]
        taker_adj = self.adjust_order_candidates(self.taker_connector, [taker_candidate])[0]
        if maker_adj.amount == Decimal("0") or taker_adj.amount == Decimal("0"):
            self.close_type = CloseType.INSUFFICIENT_BALANCE
            self.logger().error(
                f"Not enough margin to open PerpXEMM "
                f"({self.maker_connector} {self.maker_trading_pair} / "
                f"{self.taker_connector} {self.taker_trading_pair})."
            )
            self.stop()

    def _get_leverage(self, connector_name: str, trading_pair: str) -> int:
        connector = self.connectors.get(connector_name)
        if connector is None:
            return 1
        try:
            return int(connector.get_leverage(trading_pair))
        except Exception:
            return 1

    async def get_tx_cost_in_asset(
        self,
        exchange: str,
        trading_pair: str,
        is_buy: bool,
        order_amount: Decimal,
        asset: str,
        order_type: OrderType = OrderType.MARKET,
    ):
        connector = self.connectors[exchange]
        if self.is_amm_connector(exchange=exchange):
            return await super().get_tx_cost_in_asset(
                exchange=exchange,
                trading_pair=trading_pair,
                is_buy=is_buy,
                order_amount=order_amount,
                asset=asset,
                order_type=order_type,
            )

        fee = connector.get_fee(
            base_currency=asset,
            quote_currency=trading_pair.split("-")[1],
            order_type=order_type,
            order_side=TradeType.BUY if is_buy else TradeType.SELL,
            position_action=PositionAction.OPEN,
            amount=order_amount,
            price=self._taker_result_price,
            is_maker=order_type.is_limit_type(),
        )
        return fee.fee_amount_in_token(
            trading_pair=trading_pair,
            price=self._taker_result_price,
            order_amount=order_amount,
            token=asset,
        )

    async def create_maker_order(self):
        order_id = self.place_order(
            connector_name=self.maker_connector,
            trading_pair=self.maker_trading_pair,
            order_type=OrderType.LIMIT,
            side=self.maker_order_side,
            amount=self.config.order_amount,
            position_action=PositionAction.OPEN,
            price=self._maker_target_price,
        )
        self.maker_order = TrackedOrder(order_id=order_id)
        self.logger().info(
            f"[PerpXEMM] Maker {self.maker_order_side.name} {self.maker_trading_pair} "
            f"@ {self._maker_target_price} on {self.maker_connector} (id={order_id})."
        )

    def place_taker_order(self):
        taker_order_id = self.place_order(
            connector_name=self.taker_connector,
            trading_pair=self.taker_trading_pair,
            order_type=OrderType.MARKET,
            side=self.taker_order_side,
            amount=self.config.order_amount,
            position_action=PositionAction.OPEN,
        )
        self.taker_order = TrackedOrder(order_id=taker_order_id)
        self.logger().info(
            f"[PerpXEMM] Taker {self.taker_order_side.name} {self.taker_trading_pair} "
            f"MKT on {self.taker_connector} (id={taker_order_id})."
        )

    def process_order_failed_event(self, _, market, event):
        margin_mode_errors = (
            "margin mode does not match",
            "marginMode",
            "330005",
        )
        err_msg = getattr(event, "error_message", "") or ""
        is_margin_error = any(tok in err_msg for tok in margin_mode_errors)

        if is_margin_error:
            self.close_type = CloseType.FAILED
            self.logger().error(
                f"[PerpXEMM] Order {event.order_id} failed with margin mode error: "
                f"{err_msg}. Stopping executor to prevent retry spam."
            )
            self.stop()
            return

        super().process_order_failed_event(_, market, event)

    def early_stop(self, keep_position: bool = False):
        super().early_stop(keep_position=keep_position)

    async def control_shutdown_process(self):
        m_done = self.maker_order.is_done if self.maker_order else False
        t_done = self.taker_order.is_done if self.taker_order else False
        self.logger().info(
            f"[PerpXEMM] control_shutdown_process: "
            f"maker_done={m_done} taker_done={t_done}"
        )
        if m_done and t_done:
            self.close_type = CloseType.COMPLETED
            self.logger().info(
                f"[PerpXEMM] Both orders done → COMPLETED. "
                f"{self.maker_trading_pair}"
            )
            self.stop()
        else:
            self.logger().info(
                f"[PerpXEMM] Waiting for orders: "
                f"maker={self.maker_order.order_id[:12] if self.maker_order else 'None'} "
                f"taker={self.taker_order.order_id[:12] if self.taker_order else 'None'}"
            )

    def get_custom_info(self) -> Dict:
        info = super().get_custom_info()
        try:
            if self.maker_order and self.maker_order.order:
                info["maker_avg_fill_price"] = self.maker_order.average_executed_price
            if self.taker_order and self.taker_order.order:
                info["taker_avg_fill_price"] = self.taker_order.average_executed_price
        except Exception:
            pass
        return info


class XEMMPerpTieredScannerConfig(StrategyV2ConfigBase):
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    candles_config: List[CandlesConfig] = Field(default_factory=list)
    controllers_config: List[str] = Field(default_factory=list)
    markets: Dict[str, Set[str]] = Field(default_factory=dict)

    # Fields to store discovered pairs (avoid class attributes)
    discovered_pairs: Set[str] = Field(default_factory=set, exclude=True)
    rest_only_pairs: Set[str] = Field(default_factory=set, exclude=True)

    maker_connector: str = Field(
        default="kucoin_perpetual",
        client_data=ClientFieldData(
            prompt=lambda e: "Maker perpetual connector (limit orders): ",
            prompt_on_new=True,
        ))
    taker_connector: str = Field(
        default="binance_perpetual",
        client_data=ClientFieldData(
            prompt=lambda e: "Taker perpetual connector (hedge): ",
            prompt_on_new=True,
        ))
    auto_discover_pairs: bool = Field(
        default=True,
        client_data=ClientFieldData(
            prompt=lambda e: "Auto-discover USDT perps from both exchanges (True/False): ",
            prompt_on_new=True,
        ))
    scan_pairs: str = Field(
        default="",
        client_data=ClientFieldData(
            prompt=lambda e: "Extra WS-tier pairs to ADD to auto-discovered set (comma-separated, empty=none): ",
            prompt_on_new=True,
        ))
    rest_discovery_pairs: str = Field(
        default="",
        client_data=ClientFieldData(
            prompt=lambda e: "Extra REST-tier pairs to ADD to auto-discovered set (comma-separated, empty=none): ",
            prompt_on_new=True,
        ))
    excluded_pairs: str = Field(
        default="",
        client_data=ClientFieldData(
            prompt=lambda e: "Pairs to exclude from both tiers (comma-separated): ",
            prompt_on_new=True,
        ))
    max_ws_pairs: int = Field(
        default=200, gt=0,
        client_data=ClientFieldData(
            prompt=lambda e: "Max pairs to register for WS subscriptions (exchange limit ~1000): ",
            prompt_on_new=True,
        ))
    leverage: int = Field(
        default=5, gt=0,
        client_data=ClientFieldData(
            prompt=lambda e: "Leverage (e.g. 5 for 5x): ",
            prompt_on_new=True,
        ))
    position_mode: PositionMode = Field(
        default=PositionMode.HEDGE,
        client_data=ClientFieldData(
            prompt=lambda e: "Position mode (HEDGE/ONEWAY). HEDGE recommended: ",
            prompt_on_new=True,
        ))
    margin_mode: str = Field(
        default="CROSS",
        client_data=ClientFieldData(
            prompt=lambda e: "Margin mode (CROSS/ISOLATED). CROSS recommended for XEMM: ",
            prompt_on_new=True,
        ))
    total_amount_quote: Decimal = Field(
        default=Decimal("100"),
        client_data=ClientFieldData(
            prompt=lambda e: "Total portfolio MARGIN budget in quote (e.g. 100 USDT): ",
            prompt_on_new=True,
        ))
    budget_utilization: Decimal = Field(
        default=Decimal("0.9"),
        client_data=ClientFieldData(
            prompt=lambda e: "Fraction of budget committed (0.9 = 90%): ",
            prompt_on_new=True,
        ))
    buy_levels_targets_amount: str = Field(
        default="0.002,1-0.004,2-0.006,3",
        client_data=ClientFieldData(
            prompt=lambda e: "Buy levels 'NET_profit,weight' joined by '-': ",
            prompt_on_new=True,
        ))
    sell_levels_targets_amount: str = Field(
        default="0.002,1-0.004,2-0.006,3",
        client_data=ClientFieldData(
            prompt=lambda e: "Sell levels 'NET_profit,weight' joined by '-': ",
            prompt_on_new=True,
        ))
    maker_fee_pct: Decimal = Field(
        default=Decimal("0.0002"),
        client_data=ClientFieldData(
            prompt=lambda e: "Maker fee pct (0.0002 = 0.02%): ",
            prompt_on_new=True,
        ))
    taker_fee_pct: Decimal = Field(
        default=Decimal("0.0005"),
        client_data=ClientFieldData(
            prompt=lambda e: "Taker fee pct (0.0005 = 0.05%): ",
            prompt_on_new=True,
        ))
    slippage_buffer_pct: Decimal = Field(
        default=Decimal("0.0005"),
        client_data=ClientFieldData(
            prompt=lambda e: "Extra slippage buffer pct (0.0005 = 0.05%): ",
            prompt_on_new=True,
        ))
    funding_rate_buffer_pct: Decimal = Field(
        default=Decimal("0.0003"),
        client_data=ClientFieldData(
            prompt=lambda e: "Static fallback funding reserve pct (0.0003 = 0.03%): ",
            prompt_on_new=True,
        ))
    use_dynamic_funding: bool = Field(
        default=True,
        client_data=ClientFieldData(
            prompt=lambda e: "Use live funding rates (True/False): ",
            prompt_on_new=True,
        ))
    max_active_pairs: int = Field(
        default=4,
        client_data=ClientFieldData(
            prompt=lambda e: "Max pairs to trade simultaneously: ",
            prompt_on_new=True,
        ))
    scan_interval: int = Field(
        default=10, gt=0,
        client_data=ClientFieldData(
            prompt=lambda e: "WS-tier scan interval (seconds): ",
            prompt_on_new=True,
        ))
    rest_scan_interval: int = Field(
        default=30, gt=0,
        client_data=ClientFieldData(
            prompt=lambda e: "REST-tier poll interval (seconds). Higher = less API load: ",
            prompt_on_new=True,
        ))
    min_spread_to_switch: Decimal = Field(
        default=Decimal("0.001"),
        client_data=ClientFieldData(
            prompt=lambda e: "Min spread advantage to switch pairs (0.001 = 0.1%): ",
            prompt_on_new=True,
        ))
    min_holding_time: int = Field(
        default=10,
        client_data=ClientFieldData(
            prompt=lambda e: "Min holding time (sec) before checking close trigger: ",
            prompt_on_new=True,
        ))
    max_holding_time: int = Field(
        default=120,
        client_data=ClientFieldData(
            prompt=lambda e: "Max holding time (sec) before forced MARKET exit (HFT rotation): ",
            prompt_on_new=True,
        ))
    min_close_profit_pct: Decimal = Field(
        default=Decimal("0.0003"),
        client_data=ClientFieldData(
            prompt=lambda e: "Min net profit pct to trigger LIMIT close (0.0003 = 0.03%): ",
            prompt_on_new=True,
        ))
    close_limit_tick_shift: int = Field(
        default=1, ge=0, le=5,
        client_data=ClientFieldData(
            prompt=lambda e: "Ticks to shift LIMIT close orders toward market (0=mid, 1-2 for HFT): ",
            prompt_on_new=True,
        ))
    close_order_timeout_sec: int = Field(
        default=15, ge=0,
        client_data=ClientFieldData(
            prompt=lambda e: "Timeout (sec) to cancel unfilled LIMIT close orders (0=disable): ",
            prompt_on_new=True,
        ))
    max_notional_imbalance_pct: Decimal = Field(
        default=Decimal("0.4"),
        client_data=ClientFieldData(
            prompt=lambda e: "Max BUY-vs-SELL imbalance per pair (0.4 = 40%): ",
            prompt_on_new=True,
        ))
    min_order_lifetime: int = Field(
        default=90,
        client_data=ClientFieldData(
            prompt=lambda e: "Cancel idle maker executor after N seconds (0 = never): ",
            prompt_on_new=True,
        ))
    taker_depth_check: bool = Field(
        default=True,
        client_data=ClientFieldData(
            prompt=lambda e: "Pre-check taker order book depth via VWAP (True/False): ",
            prompt_on_new=True,
        ))
    max_mark_divergence_pct: Decimal = Field(
        default=Decimal("0.005"),
        client_data=ClientFieldData(
            prompt=lambda e: "Max mid-price divergence before pair is skipped (0.005 = 0.5%): ",
            prompt_on_new=True,
        ))
    hanging_hedge_alert_sec: int = Field(
        default=60,
        client_data=ClientFieldData(
            prompt=lambda e: "Seconds in SHUTTING_DOWN before CRITICAL alert: ",
            prompt_on_new=True,
        ))
    hanging_hedge_hard_timeout_sec: int = Field(
        default=300,
        client_data=ClientFieldData(
            prompt=lambda e: "Hard timeout (sec) for SHUTTING_DOWN → pair QUARANTINED (0=disable): ",
            prompt_on_new=True,
        ))
    max_quarantines_per_hour: int = Field(
        default=3,
        client_data=ClientFieldData(
            prompt=lambda e: "Kill-switch: N quarantines in 1h → halt all (0=disable): ",
            prompt_on_new=True,
        ))
    max_drawdown_pct: Decimal = Field(
        default=Decimal("0.05"),
        client_data=ClientFieldData(
            prompt=lambda e: "Max drawdown vs budget before halting (0.05 = 5%, 0=disable): ",
            prompt_on_new=True,
        ))
    perp_setup_validation_after_ticks: int = Field(
        default=10,
        client_data=ClientFieldData(
            prompt=lambda e: "Ticks before verifying position_mode + leverage (0=never): ",
            prompt_on_new=True,
        ))
    lazy_setup: bool = Field(
        default=True,
        client_data=ClientFieldData(
            prompt=lambda e: "Lazy setup: only call set_leverage when a pair is activated, "
                             "not for all pairs at startup (True/False): ",
            prompt_on_new=True,
        ))
    auto_promote: bool = Field(
        default=True,
        client_data=ClientFieldData(
            prompt=lambda e: "Auto-promote REST-tier pairs to trading when "
                             "they rank high — uses REST pricing (True/False): ",
            prompt_on_new=True,
        ))

    @field_validator("position_mode", mode="before")
    @classmethod
    def validate_position_mode(cls, v: Union[str, PositionMode]) -> PositionMode:
        if isinstance(v, PositionMode):
            return v
        if isinstance(v, str):
            try:
                return PositionMode[v.upper()]
            except KeyError:
                raise ValueError(f"Invalid position_mode: {v}. Use HEDGE or ONEWAY.")
        raise ValueError(f"Invalid position_mode type: {type(v)}")

    @field_validator("total_amount_quote", mode="before")
    @classmethod
    def validate_total_amount_quote(cls, v) -> Decimal:
        d = Decimal(str(v))
        if d <= 0:
            raise ValueError("total_amount_quote must be positive")
        return d

    @field_validator("budget_utilization", mode="before")
    @classmethod
    def validate_budget_utilization(cls, v) -> Decimal:
        d = Decimal(str(v))
        if d <= 0 or d > 1:
            raise ValueError("budget_utilization must be in (0, 1]")
        return d

    @field_validator("buy_levels_targets_amount", "sell_levels_targets_amount", mode="before")
    @classmethod
    def validate_levels(cls, v):
        if isinstance(v, str):
            parts = v.split("-")
            for part in parts:
                vals = part.split(",")
                if len(vals) != 2:
                    raise ValueError(
                        f"Invalid level format: {part!r}. Expected 'net_profit,amount'"
                    )
                p = Decimal(vals[0])
                a = Decimal(vals[1])
                if p <= 0 or a <= 0:
                    raise ValueError(f"Level values must be positive: {part!r}")
        return v

    def parse_levels(self) -> Tuple[List[Tuple[Decimal, Decimal]], List[Tuple[Decimal, Decimal]]]:
        return (
            _parse_level_string(self.buy_levels_targets_amount),
            _parse_level_string(self.sell_levels_targets_amount),
        )

    def parse_scan_pairs(self) -> List[str]:
        raw = [p.strip().upper() for p in self.scan_pairs.split(",") if p.strip()]
        excluded = {p.strip().upper() for p in self.excluded_pairs.split(",") if p.strip()}
        return list(dict.fromkeys(p for p in raw if p not in excluded))

    def parse_rest_discovery_pairs(self) -> List[str]:
        raw = [p.strip().upper() for p in self.rest_discovery_pairs.split(",") if p.strip()]
        excluded = {p.strip().upper() for p in self.excluded_pairs.split(",") if p.strip()}
        ws_pairs = set(self.parse_scan_pairs())
        return list(dict.fromkeys(p for p in raw if p not in excluded and p not in ws_pairs))


def _parse_level_string(s: str) -> List[Tuple[Decimal, Decimal]]:
    levels: List[Tuple[Decimal, Decimal]] = []
    for part in s.split("-"):
        profit_str, amount_str = part.split(",")
        levels.append((Decimal(profit_str), Decimal(amount_str)))
    return sorted(levels, key=lambda x: x[0])


class XEMMPerpTieredScanner(StrategyV2Base):

    @classmethod
    def init_markets(cls, config: XEMMPerpTieredScannerConfig):
        if not cls._is_perpetual(config.maker_connector):
            raise ValueError(
                f"maker_connector {config.maker_connector!r} must be a perpetual connector. "
                "Use v2_xemm_fast_scanner.py for spot."
            )
        if not cls._is_perpetual(config.taker_connector):
            raise ValueError(
                f"taker_connector {config.taker_connector!r} must be a perpetual connector. "
                "Use v2_xemm_fast_scanner.py for spot."
            )

        excluded = {p.strip().upper() for p in config.excluded_pairs.split(",") if p.strip()}
        extra_ws = set(config.parse_scan_pairs())
        extra_rest = set(config.parse_rest_discovery_pairs())

        if config.auto_discover_pairs:
            all_pairs, spread_rank = _discover_pairs_sync(
                config.maker_connector,
                config.taker_connector,
                extra_pairs=(extra_ws | extra_rest) or None,
            )
            all_pairs -= excluded
            if not all_pairs:
                raise ValueError(
                    "Auto-discovery found 0 common USDT perpetual pairs. "
                    "Check connector names and network connectivity."
                )
            if spread_rank:
                ranked_pairs = [p for p, _ in spread_rank]
                fallback = all_pairs - set(ranked_pairs)
                ranked_pairs.extend(sorted(fallback))
            else:
                ranked_pairs = sorted(all_pairs)
            ws_subset = set(ranked_pairs[: config.max_ws_pairs])
            _log.info(
                f"[DISCOVERY] WS tier (top {len(ws_subset)} by spread): "
                + ", ".join(f"{p}={s*100:.3f}%" for p, s in spread_rank[:10])
            )
        else:
            all_pairs = extra_ws | extra_rest
            if not all_pairs:
                raise ValueError(
                    "auto_discover_pairs=False and no manual pairs specified. "
                    "Provide scan_pairs / rest_discovery_pairs or enable auto_discover_pairs."
                )
            ws_subset = all_pairs

        # Store discovered pairs in config instead of class attributes
        config.discovered_pairs = all_pairs
        config.rest_only_pairs = all_pairs - ws_subset

        cls.markets = {
            config.maker_connector: ws_subset,
            config.taker_connector: ws_subset,
        }

    @staticmethod
    def _is_perpetual(connector_name: str) -> bool:
        return "perpetual" in connector_name.lower()

    def __init__(self, connectors: Dict[str, ConnectorBase], config: XEMMPerpTieredScannerConfig):
        super().__init__(connectors, config)
        self.config: XEMMPerpTieredScannerConfig = config

        self.executor_orchestrator._executor_mapping = {
            **self.executor_orchestrator._executor_mapping,
            "xemm_executor": PerpXEMMExecutor,
        }

        buy_levels, sell_levels = config.parse_levels()
        self._parsed_buy_levels: List[Tuple[Decimal, Decimal]] = buy_levels
        self._parsed_sell_levels: List[Tuple[Decimal, Decimal]] = sell_levels

        self._buy_weight_sum: Decimal = sum(
            (w for _, w in buy_levels), start=Decimal("0")
        ) or Decimal("1")
        self._sell_weight_sum: Decimal = sum(
            (w for _, w in sell_levels), start=Decimal("0")
        ) or Decimal("1")

        all_registered: Set[str] = set()
        for pairs in self.markets.values():
            all_registered |= pairs

        self._ws_universe: List[str] = sorted(all_registered)
        self._all_discovered_pairs: Set[str] = self.config.discovered_pairs or all_registered
        rest_only: Set[str] = self.config.rest_only_pairs or set()
        self._rest_universe: List[str] = sorted(rest_only)
        self._spread_ranking: List[Tuple[str, Decimal, Decimal, Decimal, str]] = []
        self._last_scan_ts: float = 0.0
        self._last_scan_log_ts: float = 0.0

        self._active_pairs: Set[str] = set()
        self._switching_pairs: Set[str] = set()
        self._pair_activated_ts: Dict[str, float] = {}
        self._pair_force_close_sent: Set[str] = set()

        self._initial_setting_applied: bool = False
        self._tick_counter: int = 0
        self._perp_setup_validated: bool = False
        self._perp_setup_failed: bool = False
        self._executor_shutting_down_since: Dict[str, float] = {}
        self._hanging_hedge_executor_ids: Set[str] = set()
        self._trading_halted: bool = False
        self._halt_reason: Optional[str] = None
        self._halt_ts: float = 0.0
        self._expected_position_modes: Dict[str, PositionMode] = {}
        self._consecutive_order_failures: int = 0
        self._last_failure_ts: float = 0.0
        self._failure_spam_threshold: int = 5
        self._quarantined_pairs: Dict[str, float] = {}
        self._quarantined_executor_ids: Set[str] = set()
        self._quarantine_events: List[float] = []

        self._setup_pairs: Set[str] = set()
        self._closed_executor_ids: Set[str] = set()
        self._completed_hedges: Dict[str, Dict] = {}
        self._pending_close_eids: Set[str] = set()
        self._pending_close_orders: Dict[str, Dict] = {}

        self._rest_ticker_cache: Dict[str, Dict[str, Tuple[Decimal, Decimal]]] = {}
        self._rest_session: Optional[aiohttp.ClientSession] = None
        self._rest_scan_task: Optional[asyncio.Task] = None
        self._rest_scan_restart_ts: float = 0
        self._last_rest_scan_ts: float = 0.0
        self._rest_promoted_pairs: Set[str] = set()
        self._pending_ob_subscriptions: Set[str] = set()
        self._ob_subscribed_pairs: Set[str] = set()
        self._ob_subscribe_task: Optional[asyncio.Task] = None
        self._rest_errors: int = 0
        self._symbol_map: Dict[Tuple[str, str], str] = {}
        self._promote_logged_ts: Dict[str, float] = {}

    async def _resolve_exchange_symbol(self, connector_name: str, trading_pair: str) -> str:
        key = (connector_name, trading_pair)
        if key in self._symbol_map:
            return self._symbol_map[key]
        connector = self.connectors.get(connector_name)
        if connector and hasattr(connector, 'exchange_symbol_associated_to_pair'):
            try:
                sym = await connector.exchange_symbol_associated_to_pair(trading_pair)
                if sym:
                    self._symbol_map[key] = sym
                    return sym
            except Exception:
                pass
        sym = trading_pair.replace("-", "")
        self._symbol_map[key] = sym
        return sym

    def apply_initial_setting(self):
        if self.config.lazy_setup:
            self._apply_position_mode_only()
            self.logger().info(
                f"[LAZY SETUP] Position mode set. Leverage will be configured "
                f"per-pair when activated (only {self.config.max_active_pairs} pairs "
                f"at a time instead of {len(self._all_discovered_pairs)})."
            )
        else:
            self._apply_full_setup()
        self._initial_setting_applied = True

    def _apply_position_mode_only(self):
        for connector_name in (self.config.maker_connector, self.config.taker_connector):
            connector = self.connectors.get(connector_name)
            if connector is None:
                continue
            if not self.is_perpetual(connector_name):
                continue
            expected_mode = self.config.position_mode
            try:
                connector.set_position_mode(expected_mode)
            except Exception:
                if expected_mode != PositionMode.ONEWAY:
                    try:
                        connector.set_position_mode(PositionMode.ONEWAY)
                        expected_mode = PositionMode.ONEWAY
                        self.logger().warning(
                            f"{connector_name} does not support "
                            f"{self.config.position_mode.name}, using ONEWAY"
                        )
                    except Exception as e2:
                        self.logger().warning(
                            f"Could not set position mode on {connector_name}: {e2}"
                        )
                else:
                    self.logger().warning(
                        f"Could not set position mode on {connector_name}"
                    )
            self._expected_position_modes[connector_name] = expected_mode

    def _apply_full_setup(self):
        pairs = list(self._all_discovered_pairs)
        for connector_name in (self.config.maker_connector, self.config.taker_connector):
            connector = self.connectors.get(connector_name)
            if connector is None:
                continue
            if not self.is_perpetual(connector_name):
                continue
            expected_mode = self.config.position_mode
            try:
                connector.set_position_mode(expected_mode)
            except Exception:
                if expected_mode != PositionMode.ONEWAY:
                    try:
                        connector.set_position_mode(PositionMode.ONEWAY)
                        expected_mode = PositionMode.ONEWAY
                        self.logger().warning(
                            f"{connector_name} does not support "
                            f"{self.config.position_mode.name}, using ONEWAY"
                        )
                    except Exception as e2:
                        self.logger().warning(
                            f"Could not set position mode on {connector_name}: {e2}"
                        )
                else:
                    self.logger().warning(
                        f"Could not set position mode on {connector_name}"
                    )
            self._expected_position_modes[connector_name] = expected_mode
            for pair in pairs:
                try:
                    connector.set_leverage(pair, self.config.leverage)
                    self._setup_pairs.add(pair)
                except Exception as e:
                    self.logger().warning(
                        f"Could not set leverage {self.config.leverage}x on "
                        f"{connector_name} {pair}: {e}"
                    )

    def _lazy_setup_pair(self, trading_pair: str) -> bool:
        if trading_pair in self._setup_pairs:
            return True
        success = True
        for connector_name in (self.config.maker_connector, self.config.taker_connector):
            connector = self.connectors.get(connector_name)
            if connector is None:
                continue
            if not self.is_perpetual(connector_name):
                continue
            try:
                connector.set_leverage(trading_pair, self.config.leverage)
                self.logger().info(
                    f"[LAZY SETUP] {connector_name} {trading_pair}: "
                    f"leverage set to {self.config.leverage}x"
                )
            except Exception as e:
                err_msg = str(e).lower()
                if "active positions exist" in err_msg or "position" in err_msg:
                    self.logger().warning(
                        f"[LAZY SETUP] {connector_name} {trading_pair}: "
                        f"leverage change skipped (active positions): {e}"
                    )
                else:
                    self.logger().error(
                        f"[LAZY SETUP] FAILED {connector_name} {trading_pair}: {e}"
                    )
                    success = False
        if success:
            self._set_margin_mode_for_pair(trading_pair)
        self._setup_pairs.add(trading_pair)
        return success

    def _validate_perp_setup(self) -> None:
        if self._perp_setup_validated or self._perp_setup_failed:
            return
        if self.config.perp_setup_validation_after_ticks <= 0:
            self._perp_setup_validated = True
            return
        if self._tick_counter < self.config.perp_setup_validation_after_ticks:
            return

        problems: List[str] = []
        leverage_skipped = False
        for cn in (self.config.maker_connector, self.config.taker_connector):
            connector = self.connectors.get(cn)
            if connector is None:
                continue
            expected_mode = self._expected_position_modes.get(cn, self.config.position_mode)
            try:
                actual_mode = connector.position_mode
                if actual_mode != expected_mode:
                    problems.append(
                        f"{cn}: position_mode={actual_mode} expected={expected_mode}"
                    )
            except Exception as ex:
                problems.append(f"{cn}: cannot read position_mode ({ex})")

            if hasattr(connector, "margin_mode"):
                try:
                    actual_margin = connector.margin_mode
                    if actual_margin is not None and str(actual_margin.value) != self.config.margin_mode.upper():
                        problems.append(
                            f"{cn}: margin_mode={actual_margin.value}, "
                            f"expected={self.config.margin_mode.upper()}"
                        )
                except Exception:
                    pass

            sample_pairs = list(self._setup_pairs)[:5] if self._setup_pairs else []
            if not sample_pairs and self.config.lazy_setup:
                leverage_skipped = True
                continue
            for pair in sample_pairs:
                try:
                    actual_lev = int(connector.get_leverage(pair))
                    if actual_lev != self.config.leverage:
                        problems.append(
                            f"{cn} {pair}: leverage={actual_lev}x "
                            f"expected={self.config.leverage}x"
                        )
                        break
                except Exception:
                    continue

        if problems:
            self._perp_setup_failed = True
            self._trading_halted = True
            self._halt_reason = "perp setup mismatch"
            for p in problems:
                self.logger().critical(f"[PERP SETUP] {p}")
            self.logger().critical(
                "[PERP SETUP] Halting NEW orders. Fix exchange-side configuration "
                "(margin mode / position mode / leverage) and restart the strategy."
            )
        elif leverage_skipped:
            self.logger().info(
                f"[PERP SETUP] Position mode & margin OK. Leverage validation "
                f"deferred (lazy_setup=True, no pairs activated yet)."
            )
        else:
            self._perp_setup_validated = True
            self.logger().info(
                f"[PERP SETUP] Verified position_mode={self.config.position_mode.name} "
                f"and leverage={self.config.leverage}x on sampled pairs."
            )

    @staticmethod
    def _resolve_margin_mode_enum(connector, desired_mode: str):
        import importlib
        try:
            connector_module = type(connector).__module__
            pkg = connector_module.rsplit(".", 1)[0]
            mod_name = pkg.split(".")[-1]
            constants_path = f"{pkg}.{mod_name}_constants"
            mod = importlib.import_module(constants_path)
            mm_enum = getattr(mod, "MarginMode", None)
            if mm_enum is not None:
                return mm_enum[desired_mode.upper()]
        except Exception:
            pass
        return None

    def _set_margin_mode_for_pair(self, trading_pair: str):
        for connector_name in (self.config.maker_connector, self.config.taker_connector):
            if "bitget" not in connector_name.lower():
                continue
            connector = self.connectors.get(connector_name)
            if connector is None:
                continue
            margin_mode_enum = self._resolve_margin_mode_enum(connector, self.config.margin_mode)
            if margin_mode_enum is None:
                continue
            asyncio.ensure_future(
                self._set_single_pair_margin_mode_bitget(
                    connector, connector_name, trading_pair, margin_mode_enum
                )
            )

    async def _set_single_pair_margin_mode_bitget(
        self, connector, connector_name: str, trading_pair: str, mode
    ):
        try:
            from hummingbot.connector.derivative.bitget_perpetual import (
                bitget_perpetual_constants as CONSTANTS,
            )
            margin_mode_str = CONSTANTS.MARGIN_MODE_TYPES.get(mode, "crossed")
            exchange_symbol = await self._resolve_exchange_symbol(connector_name, trading_pair)
            product_type = await connector.product_type_associated_to_trading_pair(trading_pair)
            margin_coin = connector.get_buy_collateral_token(trading_pair)

            response = await connector._api_post(
                path_url=CONSTANTS.SET_MARGIN_MODE_ENDPOINT,
                data={
                    "symbol": exchange_symbol,
                    "productType": product_type,
                    "marginMode": margin_mode_str,
                    "marginCoin": margin_coin,
                },
                is_auth_required=True,
            )

            code = response.get("code")
            if code == CONSTANTS.RET_CODE_OK:
                self.logger().info(
                    f"[MARGIN MODE] {connector_name} {trading_pair}: "
                    f"set to {margin_mode_str}"
                )
            else:
                already_ok = (
                    "same as current" in str(response.get("msg", "")).lower()
                    or "no need" in str(response.get("msg", "")).lower()
                )
                if already_ok:
                    self.logger().debug(
                        f"[MARGIN MODE] {connector_name} {trading_pair}: "
                        f"already {margin_mode_str}"
                    )
                else:
                    self.logger().warning(
                        f"[MARGIN MODE] {connector_name} {trading_pair}: "
                        f"failed (code={code}, msg={response.get('msg', '')})"
                    )
        except Exception as e:
            self.logger().warning(
                f"[MARGIN MODE] {connector_name} {trading_pair}: error: {e}"
            )

    # -------------------------------------------------- dynamic OB subscription

    async def _subscribe_orderbooks(self, pairs: Set[str]) -> None:
        if not pairs:
            return
        for pair in pairs:
            try:
                ok_maker = await self.market_data_provider.initialize_order_book(
                    self.config.maker_connector, pair
                )
                ok_taker = await self.market_data_provider.initialize_order_book(
                    self.config.taker_connector, pair
                )
                if ok_maker and ok_taker:
                    self._ob_subscribed_pairs.add(pair)
                    self.logger().info(
                        f"[OB-SUB] {pair}: orderbook subscribed on both connectors"
                    )
                else:
                    self.logger().warning(
                        f"[OB-SUB] {pair}: subscription failed "
                        f"(maker={ok_maker}, taker={ok_taker})"
                    )
            except Exception as e:
                self.logger().error(f"[OB-SUB] {pair}: {e}")

    # ---------------------------------------------------------------- REST tier

    async def _rest_scan_loop(self):
        if not AIOHTTP_AVAILABLE:
            self.logger().warning(
                "[REST SCAN] aiohttp not installed - REST discovery disabled."
            )
            return
        self.logger().info(
            f"[REST SCAN] Starting background scanner for "
            f"{len(self._rest_universe)} pairs, interval={self.config.rest_scan_interval}s"
        )
        while not self._is_stop_triggered:
            try:
                await self._fetch_rest_tickers()
            except Exception as e:
                self._rest_errors += 1
                self.logger().error(f"[REST SCAN] Error: {e}")
            await asyncio.sleep(self.config.rest_scan_interval)

    async def _fetch_rest_tickers(self):
        if not self._rest_universe:
            return
        if self._rest_session is None or self._rest_session.closed:
            timeout = aiohttp.ClientTimeout(total=15)
            self._rest_session = aiohttp.ClientSession(timeout=timeout)

        maker_cache: Dict[str, Tuple[Decimal, Decimal]] = {}
        taker_cache: Dict[str, Tuple[Decimal, Decimal]] = {}

        try:
            maker_data, taker_data = await asyncio.gather(
                self._fetch_exchange_tickers(self.config.maker_connector),
                self._fetch_exchange_tickers(self.config.taker_connector),
            )
            if maker_data:
                maker_cache = maker_data
            if taker_data:
                taker_cache = taker_data
        except Exception as e:
            self.logger().debug(f"[REST SCAN] Fetch failed: {e}")

        self._rest_ticker_cache = {
            self.config.maker_connector: maker_cache,
            self.config.taker_connector: taker_cache,
        }
        self._last_rest_scan_ts = time.time()

    async def _fetch_exchange_tickers(
        self, connector_name: str
    ) -> Dict[str, Tuple[Decimal, Decimal]]:
        result: Dict[str, Tuple[Decimal, Decimal]] = {}
        cn_lower = connector_name.lower()

        if "binance" in cn_lower and "perpetual" in cn_lower:
            result = await self._fetch_binance_perp_tickers()
        elif "kucoin" in cn_lower and "perpetual" in cn_lower:
            result = await self._fetch_kucoin_perp_tickers()
        elif "bitget" in cn_lower and "perpetual" in cn_lower:
            result = await self._fetch_bitget_perp_tickers()
        else:
            self.logger().debug(
                f"[REST SCAN] No REST parser for {connector_name}, skipping"
            )
        return result

    async def _fetch_binance_perp_tickers(self) -> Dict[str, Tuple[Decimal, Decimal]]:
        result: Dict[str, Tuple[Decimal, Decimal]] = {}
        url = "https://fapi.binance.com/fapi/v1/ticker/bookTicker"
        try:
            async with self._rest_session.get(url) as resp:
                if resp.status != 200:
                    self.logger().warning(
                        f"[REST SCAN] Binance bookTicker HTTP {resp.status}"
                    )
                    return result
                data = await resp.json()
        except Exception as e:
            self.logger().debug(f"[REST SCAN] Binance fetch error: {e}")
            return result

        target_symbols = set()
        for pair in self._rest_universe:
            exchange_sym = await self._resolve_exchange_symbol(
                self.config.taker_connector
                if "binance" in self.config.taker_connector.lower()
                else self.config.maker_connector,
                pair
            )
            target_symbols.add(exchange_sym)

        for item in data:
            sym = item.get("symbol", "")
            if sym not in target_symbols:
                continue
            try:
                bid = Decimal(item.get("bidPrice", "0"))
                ask = Decimal(item.get("askPrice", "0"))
                if bid > 0 and ask > 0:
                    result[sym] = (bid, ask)
            except Exception:
                continue
        return result

    async def _fetch_kucoin_perp_tickers(self) -> Dict[str, Tuple[Decimal, Decimal]]:
        result: Dict[str, Tuple[Decimal, Decimal]] = {}
        url = "https://api-futures.kucoin.com/api/v1/allTickers"
        try:
            async with self._rest_session.get(url) as resp:
                if resp.status != 200:
                    self.logger().warning(
                        f"[REST SCAN] KuCoin allTickers HTTP {resp.status}"
                    )
                    return result
                payload = await resp.json()
        except Exception as e:
            self.logger().debug(f"[REST SCAN] KuCoin fetch error: {e}")
            return result

        if not isinstance(payload, dict) or payload.get("code") != "200000":
            self.logger().debug(
                f"[REST SCAN] KuCoin API error: code={payload.get('code') if isinstance(payload, dict) else 'N/A'}"
            )
            return result
        data = payload.get("data", [])
        if not data:
            return result

        kucoin_connector = (
            self.config.maker_connector
            if "kucoin" in self.config.maker_connector.lower()
            else self.config.taker_connector
        )
        target_symbols = set()
        for pair in self._rest_universe:
            exchange_sym = await self._resolve_exchange_symbol(kucoin_connector, pair)
            target_symbols.add(exchange_sym)

        for item in data:
            sym = item.get("symbol", "")
            if sym not in target_symbols:
                continue
            try:
                bid = Decimal(str(item.get("bestBidPrice", "0") or "0"))
                ask = Decimal(str(item.get("bestAskPrice", "0") or "0"))
                if bid > 0 and ask > 0:
                    result[sym] = (bid, ask)
            except Exception:
                continue
        return result

    async def _fetch_bitget_perp_tickers(self) -> Dict[str, Tuple[Decimal, Decimal]]:
        result: Dict[str, Tuple[Decimal, Decimal]] = {}
        url = "https://api.bitget.com/api/v2/mix/market/tickers?productType=USDT-FUTURES"
        try:
            async with self._rest_session.get(url) as resp:
                if resp.status != 200:
                    self.logger().warning(
                        f"[REST SCAN] Bitget tickers HTTP {resp.status}"
                    )
                    return result
                payload = await resp.json()
        except Exception as e:
            self.logger().debug(f"[REST SCAN] Bitget fetch error: {e}")
            return result

        if not isinstance(payload, dict) or payload.get("code") != "00000":
            self.logger().debug(
                f"[REST SCAN] Bitget API error: code={payload.get('code') if isinstance(payload, dict) else 'N/A'}"
            )
            return result
        data = payload.get("data", [])
        if not data:
            return result

        bitget_connector = (
            self.config.maker_connector
            if "bitget" in self.config.maker_connector.lower()
            else self.config.taker_connector
        )
        target_symbols = set()
        for pair in self._rest_universe:
            exchange_sym = await self._resolve_exchange_symbol(bitget_connector, pair)
            target_symbols.add(exchange_sym)

        for item in data:
            sym = item.get("symbol", "")
            if sym not in target_symbols:
                continue
            try:
                bid = Decimal(str(item.get("bidPr", "0") or "0"))
                ask = Decimal(str(item.get("askPr", "0") or "0"))
                if bid > 0 and ask > 0:
                    result[sym] = (bid, ask)
            except Exception:
                continue
        return result

    async def _calculate_spread_from_rest(
        self, trading_pair: str
    ) -> Optional[Tuple[Decimal, Decimal, Decimal, Decimal]]:
        maker_cn = self.config.maker_connector
        taker_cn = self.config.taker_connector
        maker_cache = self._rest_ticker_cache.get(maker_cn, {})
        taker_cache = self._rest_ticker_cache.get(taker_cn, {})

        maker_sym = await self._resolve_exchange_symbol(maker_cn, trading_pair)
        taker_sym = await self._resolve_exchange_symbol(taker_cn, trading_pair)

        maker_data = maker_cache.get(maker_sym)
        taker_data = taker_cache.get(taker_sym)

        if maker_data is None or taker_data is None:
            return None

        maker_bid, maker_ask = maker_data
        taker_bid, taker_ask = taker_data

        maker_mid = (maker_bid + maker_ask) / Decimal("2")
        taker_mid = (taker_bid + taker_ask) / Decimal("2")

        if maker_mid <= 0 or taker_mid <= 0:
            return None

        divergence = abs(maker_mid - taker_mid) / maker_mid
        if divergence > self.config.max_mark_divergence_pct:
            return None

        if maker_ask <= 0 or taker_ask <= 0:
            return None
        buy_spread = (taker_bid - maker_ask) / maker_ask
        sell_spread = (maker_bid - taker_ask) / taker_ask
        best_spread = max(buy_spread, sell_spread)
        return best_spread, buy_spread, sell_spread, maker_mid

    # ---------------------------------------------------------------- spreads

    def _spread_floor(self) -> Decimal:
        min_buy = min(t[0] for t in self._parsed_buy_levels)
        min_sell = min(t[0] for t in self._parsed_sell_levels)
        return min(min_buy, min_sell)

    def _funding_cost_pct(
        self, trading_pair: Optional[str] = None, maker_side: Optional[TradeType] = None
    ) -> Decimal:
        static = self.config.funding_rate_buffer_pct
        if not self.config.use_dynamic_funding or trading_pair is None:
            return static
        try:
            mk = self.market_data_provider.get_funding_info(
                self.config.maker_connector, trading_pair
            )
            tk = self.market_data_provider.get_funding_info(
                self.config.taker_connector, trading_pair
            )
        except Exception:
            return static
        if mk is None or tk is None:
            return static
        try:
            mk_rate = Decimal(str(mk.rate or 0))
            tk_rate = Decimal(str(tk.rate or 0))
        except Exception:
            return static

        def _signed(side: TradeType) -> Decimal:
            if side == TradeType.BUY:
                return mk_rate - tk_rate
            return tk_rate - mk_rate

        if maker_side is not None:
            return max(Decimal("0"), _signed(maker_side))
        return max(_signed(TradeType.BUY), _signed(TradeType.SELL), Decimal("0"))

    def _cost_pct(
        self, trading_pair: Optional[str] = None, maker_side: Optional[TradeType] = None
    ) -> Decimal:
        return (
            self.config.maker_fee_pct
            + self.config.taker_fee_pct
            + self.config.slippage_buffer_pct
            + self._funding_cost_pct(trading_pair, maker_side)
        )

    def _min_eligible_raw_spread(self, trading_pair: Optional[str] = None) -> Decimal:
        return self._spread_floor() + self._cost_pct(trading_pair)

    def _calculate_spread(
        self, trading_pair: str
    ) -> Optional[Tuple[Decimal, Decimal, Decimal, Decimal]]:
        try:
            maker_mid = self.market_data_provider.get_price_by_type(
                self.config.maker_connector, trading_pair, PriceType.MidPrice
            )
            taker_mid = self.market_data_provider.get_price_by_type(
                self.config.taker_connector, trading_pair, PriceType.MidPrice
            )
            if maker_mid is None or taker_mid is None:
                return None
            if maker_mid <= 0 or taker_mid <= 0:
                return None

            divergence = abs(maker_mid - taker_mid) / maker_mid
            if divergence > self.config.max_mark_divergence_pct:
                self.logger().debug(
                    f"{trading_pair}: skipped, mid divergence "
                    f"{divergence * 100:.3f}% > "
                    f"{self.config.max_mark_divergence_pct * 100:.3f}%"
                )
                return None

            maker_ask = self.market_data_provider.get_price_by_type(
                self.config.maker_connector, trading_pair, PriceType.BestAsk
            )
            maker_bid = self.market_data_provider.get_price_by_type(
                self.config.maker_connector, trading_pair, PriceType.BestBid
            )
            taker_ask = self.market_data_provider.get_price_by_type(
                self.config.taker_connector, trading_pair, PriceType.BestAsk
            )
            taker_bid = self.market_data_provider.get_price_by_type(
                self.config.taker_connector, trading_pair, PriceType.BestBid
            )
            if any(p is None or p <= 0 for p in (maker_ask, maker_bid, taker_ask, taker_bid)):
                return None

            buy_spread = (taker_bid - maker_ask) / maker_ask
            sell_spread = (maker_bid - taker_ask) / taker_ask
            best_spread = max(buy_spread, sell_spread)
            return best_spread, buy_spread, sell_spread, maker_mid
        except Exception as e:
            self.logger().debug(f"Could not calculate spread for {trading_pair}: {e}")
            return None

    def _calculate_spread_from_rest_sync(
        self, trading_pair: str
    ) -> Optional[Tuple[Decimal, Decimal, Decimal, Decimal]]:
        maker_cn = self.config.maker_connector
        taker_cn = self.config.taker_connector
        maker_sym = self._symbol_map.get(
            (maker_cn, trading_pair), trading_pair.replace("-", "")
        )
        taker_sym = self._symbol_map.get(
            (taker_cn, trading_pair), trading_pair.replace("-", "")
        )
        maker_data = self._rest_ticker_cache.get(maker_cn, {}).get(maker_sym)
        taker_data = self._rest_ticker_cache.get(taker_cn, {}).get(taker_sym)
        if maker_data is None or taker_data is None:
            return None
        maker_bid, maker_ask = maker_data
        taker_bid, taker_ask = taker_data
        maker_mid = (maker_bid + maker_ask) / Decimal("2")
        if maker_mid <= 0:
            return None
        if maker_ask <= 0 or taker_ask <= 0:
            return None
        buy_spread = (taker_bid - maker_ask) / maker_ask
        sell_spread = (maker_bid - taker_ask) / taker_ask
        best_spread = max(buy_spread, sell_spread)
        return best_spread, buy_spread, sell_spread, maker_mid

    def _taker_depth_ok(
        self, trading_pair: str, maker_side: TradeType, amount_base: Decimal
    ) -> bool:
        if not self.config.taker_depth_check or amount_base <= 0:
            return True
        try:
            ob = self.market_data_provider.get_order_book(
                self.config.taker_connector, trading_pair
            )
            if ob is None:
                return True
            taker_is_buy = (maker_side == TradeType.SELL)
            ref_price_type = PriceType.BestAsk if taker_is_buy else PriceType.BestBid
            ref_price = self.market_data_provider.get_price_by_type(
                self.config.taker_connector, trading_pair, ref_price_type
            )
            if not ref_price or ref_price <= 0:
                return True
            res = ob.get_vwap_for_volume(taker_is_buy, float(amount_base))
            vwap = Decimal(str(res.result_price)) if res and res.result_price else Decimal("0")
            if vwap <= 0:
                self.logger().info(
                    f"{trading_pair}: taker book has no VWAP for {amount_base} base "
                    "- insufficient depth, skipping level."
                )
                return False
            slippage = abs(vwap - ref_price) / ref_price
            if slippage > self.config.slippage_buffer_pct:
                self.logger().info(
                    f"{trading_pair} {maker_side.name}: taker VWAP slippage "
                    f"{slippage * 100:.3f}% > buffer "
                    f"{self.config.slippage_buffer_pct * 100:.3f}% "
                    f"(amount={amount_base}, ref={ref_price_type.name}={ref_price}); skipping level."
                )
                return False
            return True
        except Exception as ex:
            self.logger().debug(
                f"_taker_depth_ok({trading_pair}) failed: {ex}; allowing order"
            )
            return True

    def _scan_and_rank_pairs(self) -> None:
        results: List[Tuple[str, Decimal, Decimal, Decimal, str]] = []

        for pair in self._ws_universe:
            data = self._calculate_spread(pair)
            if data is None:
                continue
            best, buy, sell, _ = data
            results.append((pair, best, buy, sell, "WS"))

        for pair in self._rest_universe:
            data = self._calculate_spread_from_rest_sync(pair)
            if data is None:
                continue
            best, buy, sell, _ = data
            results.append((pair, best, buy, sell, "REST"))

        if not results and self._rest_universe and not self._rest_ticker_cache:
            self.logger().debug(
                "[REST SCAN] No REST ticker cache yet — REST tier pairs "
                "will appear after first background scan completes."
            )

        results.sort(key=lambda x: x[1], reverse=True)
        self._spread_ranking = results

        now = self.current_timestamp
        if now - self._last_scan_log_ts >= 60:
            self._last_scan_log_ts = now
            min_eligible = self._min_eligible_raw_spread()
            ws_ranked = [(p, s) for p, s, _, _, t in results if t == "WS"]
            rest_ranked = [(p, s) for p, s, _, _, t in results if t == "REST"]
            eligible = [(p, s) for p, s, _, _, _ in results if s >= min_eligible]
            self.logger().info(
                f"Scan: WS={len(ws_ranked)} REST={len(rest_ranked)} "
                f"total={len(results)} pairs, {len(eligible)} eligible "
                f"(>= {min_eligible*100:.3f}%). "
                f"Cost: floor={self._spread_floor()*100:.3f}% "
                f"+ mkr={self.config.maker_fee_pct*100:.3f}% "
                f"+ tkr={self.config.taker_fee_pct*100:.3f}% "
                f"+ slip={self.config.slippage_buffer_pct*100:.3f}% "
                f"+ fund={self._funding_cost_pct()*100:.3f}% "
                f"= {min_eligible*100:.3f}%. Top: "
                + ", ".join(f"{p}={s*100:.3f}%" for p, s, _, _, _ in results[:3])
            )

    # --------------------------------------------------------------- routing

    def _get_top_pairs(self) -> List[str]:
        eligible = [
            (p, s, tier)
            for p, s, _, _, tier in self._spread_ranking
            if s >= self._min_eligible_raw_spread(p)
            and p not in self._quarantined_pairs
        ]

        current_active = self._active_pairs - self._switching_pairs
        now = self.current_timestamp
        to_add: List[str] = []

        rank_map = {p: s for p, s, _, _, _ in self._spread_ranking}
        for pair, spread, tier in eligible:
            if pair in current_active or pair in to_add:
                continue

            if tier == "REST":
                if not self.config.auto_promote:
                    last_log = self._promote_logged_ts.get(pair, 0)
                    if now - last_log >= 60:
                        self._promote_logged_ts[pair] = now
                        self.logger().info(
                            f"[PROMOTE] {pair}: REST tier pair with spread "
                            f"{spread * 100:.3f}% ranks in top but auto_promote=False. "
                            f"Enable auto_promote to trade."
                        )
                    continue
                if pair not in self._ws_universe:
                    self._rest_promoted_pairs.add(pair)
                    self._ws_universe.append(pair)
                    self._ws_universe.sort()
                    if pair not in self._ob_subscribed_pairs:
                        self._pending_ob_subscriptions.add(pair)
                        self.logger().info(
                            f"[PROMOTE] {pair}: scheduling orderbook subscription"
                        )

            if len(current_active) + len(to_add) < self.config.max_active_pairs:
                to_add.append(pair)
            elif current_active:
                switchable = [
                    (cp, rank_map.get(cp, Decimal("-1")))
                    for cp in current_active
                    if now - self._pair_activated_ts.get(cp, 0) >= self.config.min_holding_time
                    and cp not in self._switching_pairs
                ]
                if not switchable:
                    break
                worst_pair, worst_spread = min(switchable, key=lambda x: x[1])
                if spread > worst_spread + self.config.min_spread_to_switch:
                    to_add.append(pair)
                    self._switching_pairs.add(worst_pair)
                else:
                    break
            else:
                break
        return to_add

    # --------------------------------------------------------------- helpers

    def _get_executors_for_pair(self, trading_pair: str):
        return self.filter_executors(
            executors=self.get_all_executors(),
            filter_func=lambda e: e.config.buying_market.trading_pair == trading_pair
            or e.config.selling_market.trading_pair == trading_pair,
        )

    def _get_active_executors_for_pair(self, trading_pair: str):
        return [e for e in self._get_executors_for_pair(trading_pair) if not e.is_done]

    def _get_active_buy_executors_by_level(self, trading_pair: str) -> Dict[Decimal, list]:
        result: Dict[Decimal, list] = {}
        actives = [
            e for e in self._get_active_executors_for_pair(trading_pair)
            if e.config.maker_side == TradeType.BUY
        ]
        for target_profit, _ in self._parsed_buy_levels:
            result[target_profit] = [
                e for e in actives
                if Decimal(str(e.config.target_profitability)) == target_profit
            ]
        return result

    def _get_active_sell_executors_by_level(self, trading_pair: str) -> Dict[Decimal, list]:
        result: Dict[Decimal, list] = {}
        actives = [
            e for e in self._get_active_executors_for_pair(trading_pair)
            if e.config.maker_side == TradeType.SELL
        ]
        for target_profit, _ in self._parsed_sell_levels:
            result[target_profit] = [
                e for e in actives
                if Decimal(str(e.config.target_profitability)) == target_profit
            ]
        return result

    @staticmethod
    def _executor_has_filled(e) -> bool:
        filled_qte = getattr(e, "filled_amount_quote", None) or Decimal("0")
        if filled_qte > 0:
            return True
        if getattr(e, "close_type", None) == CloseType.COMPLETED:
            return True
        return False

    def _filled_notional(self, e) -> Decimal:
        v = getattr(e, "filled_amount_quote", None) or Decimal("0")
        return v if v > 0 else Decimal("0")

    def _stop_idle_executors(self, trading_pair: str) -> List[StopExecutorAction]:
        return [
            StopExecutorAction(executor_id=e.id)
            for e in self._get_active_executors_for_pair(trading_pair)
            if not self._executor_has_filled(e)
        ]

    def _cancel_stale_idle_executors(self, trading_pair: str) -> List[StopExecutorAction]:
        if self.config.min_order_lifetime <= 0:
            return []
        actions: List[StopExecutorAction] = []
        for e in self._get_active_executors_for_pair(trading_pair):
            age = getattr(e, "age", None)
            if not age or age < self.config.min_order_lifetime:
                continue
            if self._executor_has_filled(e):
                continue
            self.logger().info(
                f"Cancelling stale idle executor {e.id[:8]} for {trading_pair} "
                f"(age={age:.0f}s)"
            )
            actions.append(StopExecutorAction(executor_id=e.id))
        return actions

    def _notional_imbalance(self, trading_pair: str) -> Decimal:
        buy_filled = Decimal("0")
        sell_filled = Decimal("0")
        for e in self._get_executors_for_pair(trading_pair):
            if not self._executor_has_filled(e):
                continue
            v = self._filled_notional(e)
            if e.config.maker_side == TradeType.BUY:
                buy_filled += v
            else:
                sell_filled += v
        return buy_filled - sell_filled

    def _per_pair_notional_budget(self) -> Decimal:
        return self._total_notional_budget() / Decimal(max(1, self.config.max_active_pairs))

    def _imbalance_cap(self) -> Decimal:
        return self._per_pair_notional_budget() * self.config.max_notional_imbalance_pct

    # --------------------------------------------- tick-size helpers for LIMIT close
    def _get_tick_size(self, connector_name: str, pair: str) -> Decimal:
        try:
            connector = self.connectors.get(connector_name)
            if connector and hasattr(connector, "get_trading_rules"):
                rules = connector.get_trading_rules(pair)
                if rules and hasattr(rules, "min_price_increment") and rules.min_price_increment > 0:
                    return Decimal(str(rules.min_price_increment))
        except Exception:
            pass
        mid = self.market_data_provider.get_price_by_type(connector_name, pair, PriceType.MidPrice)
        return (mid or Decimal("1")) * Decimal("0.0001")

    def _tick_adjusted_price(
        self, connector_name: str, pair: str, base_price: Decimal,
        side: TradeType, ticks: int = 1
    ) -> Decimal:
        tick = self._get_tick_size(connector_name, pair)
        shift = tick * ticks
        try:
            bid = self.market_data_provider.get_price_by_type(connector_name, pair, PriceType.BestBid)
            ask = self.market_data_provider.get_price_by_type(connector_name, pair, PriceType.BestAsk)
            if bid and ask and ask > bid:
                spread = ask - bid
                shift = min(shift, spread * Decimal("0.8"))
        except Exception:
            pass

        if side == TradeType.SELL:
            return max(base_price - shift, tick)
        return base_price + shift

    # --------------------------------------------- completed hedge closer (HFT)
    def _close_hedges_by_executable_pnl(self) -> None:
        now = self.current_timestamp
        executors = self.get_all_executors()
        min_hold = float(self.config.min_holding_time)
        max_hold = float(self.config.max_holding_time)
        min_profit = self.config.min_close_profit_pct
        tick_shift = self.config.close_limit_tick_shift

        # Clean up pending eids where LIMIT close orders already filled
        for eid in list(self._pending_close_eids):
            info = self._completed_hedges.get(eid)
            if not info:
                self._pending_close_eids.discard(eid)
                continue
            try:
                long_pos = self.connectors[info["long_cn"]].get_position(info["pair"])
                short_pos = self.connectors[info["short_cn"]].get_position(info["pair"])
                long_sz = abs(float(long_pos.amount)) if long_pos else 0.0
                short_sz = abs(float(short_pos.amount)) if short_pos else 0.0
                if long_sz < 1e-6 and short_sz < 1e-6:
                    self._closed_executor_ids.add(eid)
                    self._completed_hedges.pop(eid, None)
                    self._pending_close_eids.discard(eid)
            except Exception:
                pass

        # 1. Register new completed hedges with entry prices
        for e in executors:
            eid = str(e.id)
            if eid in self._closed_executor_ids or eid in self._completed_hedges:
                continue
            if getattr(e, "close_type", None) != CloseType.COMPLETED or not e.is_done:
                continue

            # Get fill prices from custom_info (set by PerpXEMMExecutor.get_custom_info)
            ci = e.custom_info or {}
            maker_avg = Decimal(str(ci.get("maker_avg_fill_price", 0) or 0))
            taker_avg = Decimal(str(ci.get("taker_avg_fill_price", 0) or 0))
            # Fallback to target/taker prices if actual fills not available
            if maker_avg <= 0:
                maker_avg = Decimal(str(ci.get("maker_target_price", 0) or 0))
            if taker_avg <= 0:
                taker_avg = Decimal(str(ci.get("taker_price", 0) or 0))
            if maker_avg <= 0 or taker_avg <= 0:
                self.logger().warning(f"[CLOSE-HFT] eid={eid[:8]} no fill prices in custom_info, skipping")
                continue

            pair = e.config.buying_market.trading_pair
            maker_cn = e.config.buying_market.connector_name
            taker_cn = e.config.selling_market.connector_name
            maker_side = e.config.maker_side
            amount = e.config.order_amount
            created = float(getattr(e.config, "timestamp", 0) or 0)

            if maker_side == TradeType.BUY:
                long_cn, long_entry = maker_cn, maker_avg
                short_cn, short_entry = taker_cn, taker_avg
            else:
                long_cn, long_entry = taker_cn, taker_avg
                short_cn, short_entry = maker_cn, maker_avg

            self._completed_hedges[eid] = {
                "pair": pair, "amount": amount, "created": created,
                "long_cn": long_cn, "long_entry": long_entry,
                "short_cn": short_cn, "short_entry": short_entry,
            }

        # 2. Evaluate and close
        to_remove = []
        for eid, info in self._completed_hedges.items():
            holding_sec = now - info["created"]
            if holding_sec < min_hold:
                continue
            if eid in self._pending_close_eids:
                continue

            pair = info["pair"]
            amount = info["amount"]
            long_cn, short_cn = info["long_cn"], info["short_cn"]
            long_entry, short_entry = info["long_entry"], info["short_entry"]

            try:
                long_bid = self.market_data_provider.get_price_by_type(long_cn, pair, PriceType.BestBid)
                short_ask = self.market_data_provider.get_price_by_type(short_cn, pair, PriceType.BestAsk)
                long_mid = self.market_data_provider.get_price_by_type(long_cn, pair, PriceType.MidPrice)
                short_mid = self.market_data_provider.get_price_by_type(short_cn, pair, PriceType.MidPrice)
                if any(p is None or p <= 0 for p in (long_bid, short_ask, long_mid, short_mid)):
                    continue
            except Exception:
                continue

            # Executable PnL: LONG exits at BID, SHORT exits at ASK
            pnl_long = (long_bid - long_entry) / long_entry
            pnl_short = (short_entry - short_ask) / short_entry
            executable_net = pnl_long + pnl_short

            # Dynamic close cost (NO funding, HFT-appropriate)
            long_ask = self.market_data_provider.get_price_by_type(long_cn, pair, PriceType.BestAsk)
            short_bid = self.market_data_provider.get_price_by_type(short_cn, pair, PriceType.BestBid)
            long_spread = Decimal("0")
            short_spread = Decimal("0")
            if long_ask and long_bid and long_mid and long_mid > 0:
                long_spread = (long_ask - long_bid) / long_mid
            if short_ask and short_bid and short_mid and short_mid > 0:
                short_spread = (short_ask - short_bid) / short_mid

            close_cost = (
                self.config.taker_fee_pct * 2
                + (long_spread + short_spread) * Decimal("0.5")
                + self.config.slippage_buffer_pct
            )

            threshold = close_cost + min_profit
            force_close = holding_sec >= max_hold

            if executable_net >= threshold or force_close:
                reason = "TIMEOUT" if force_close else "PNL_TRIGGER"
                order_type = OrderType.LIMIT if not force_close else OrderType.MARKET

                if order_type == OrderType.LIMIT:
                    exit_price_long = self._tick_adjusted_price(long_cn, pair, long_mid, TradeType.SELL, ticks=tick_shift)
                    exit_price_short = self._tick_adjusted_price(short_cn, pair, short_mid, TradeType.BUY, ticks=tick_shift)
                else:
                    exit_price_long = None
                    exit_price_short = None

                try:
                    long_oid = self.connectors[long_cn].sell(
                        pair, amount,
                        order_type=order_type,
                        position_action=PositionAction.CLOSE,
                        price=exit_price_long if order_type == OrderType.LIMIT else None,
                    )
                    short_oid = self.connectors[short_cn].buy(
                        pair, amount,
                        order_type=order_type,
                        position_action=PositionAction.CLOSE,
                        price=exit_price_short if order_type == OrderType.LIMIT else None,
                    )

                    if order_type == OrderType.LIMIT:
                        if long_oid:
                            self._pending_close_orders[long_oid] = {
                                "cn": long_cn, "pair": pair, "ts": now, "side": "SELL", "eid": eid
                            }
                        if short_oid:
                            self._pending_close_orders[short_oid] = {
                                "cn": short_cn, "pair": pair, "ts": now, "side": "BUY", "eid": eid
                            }
                        # Keep hedge data for re-evaluation if LIMIT is cancelled
                        self._pending_close_eids.add(eid)
                    else:
                        # MARKET close is final
                        self._closed_executor_ids.add(eid)
                        to_remove.append(eid)

                    self.logger().info(
                        f"[CLOSE-HFT] {pair} {reason} after {holding_sec:.0f}s | "
                        f"NetPnL={executable_net * 100:.3f}% >= Thr={threshold * 100:.3f}% | "
                        f"Cost={close_cost * 100:.3f}% | Type={order_type.name} | "
                        f"LONG@{long_cn} {long_entry:.5f}->{'MKT' if exit_price_long is None else f'{exit_price_long:.5f}'} | "
                        f"SHORT@{short_cn} {short_entry:.5f}->{'MKT' if exit_price_short is None else f'{exit_price_short:.5f}'}"
                    )
                except Exception as ex:
                    self.logger().warning(f"[CLOSE-HFT] Failed to close {pair}: {ex}")

        for eid in to_remove:
            self._completed_hedges.pop(eid, None)

    # ------------------------------------------------- hanging hedge monitor
    def _monitor_hanging_hedges(self) -> None:
        alert_th = max(1, int(self.config.hanging_hedge_alert_sec))
        hard_th = int(self.config.hanging_hedge_hard_timeout_sec or 0)
        now = self.current_timestamp
        live_ids: Set[str] = set()
        for e in self.get_all_executors():
            status = getattr(e, "status", None)
            if status != RunnableStatus.SHUTTING_DOWN:
                continue
            live_ids.add(e.id)
            since = self._executor_shutting_down_since.get(e.id)
            if since is None:
                self._executor_shutting_down_since[e.id] = now
                continue
            elapsed = now - since
            pair = e.config.buying_market.trading_pair
            side = e.config.maker_side.name
            filled = self._filled_notional(e)

            if elapsed >= alert_th and e.id not in self._hanging_hedge_executor_ids:
                self._hanging_hedge_executor_ids.add(e.id)
                self.logger().critical(
                    f"[HANGING HEDGE] executor={e.id[:8]} {pair} maker={side} "
                    f"in SHUTTING_DOWN for {elapsed:.0f}s - maker filled "
                    f"~${filled:.2f} but taker hedge not done. "
                    f"Inspect taker connector and close position manually if needed."
                )

            if (
                hard_th > 0
                and elapsed >= hard_th
                and e.id not in self._quarantined_executor_ids
            ):
                self._quarantined_executor_ids.add(e.id)
                self._quarantine_pair(pair, e, elapsed)

        stale = [
            eid for eid in self._executor_shutting_down_since if eid not in live_ids
        ]
        for eid in stale:
            self._executor_shutting_down_since.pop(eid, None)
            self._hanging_hedge_executor_ids.discard(eid)
            self._quarantined_executor_ids.discard(eid)

    def _quarantine_pair(self, pair: str, executor, elapsed: float) -> None:
        now = self.current_timestamp
        if pair not in self._quarantined_pairs:
            self._quarantined_pairs[pair] = now
            self.logger().critical(
                f"[QUARANTINE] {pair} executor={executor.id[:8]} stuck "
                f"{elapsed:.0f}s > hard_timeout "
                f"{self.config.hanging_hedge_hard_timeout_sec}s. "
                f"NEW orders on this pair are disabled until restart."
            )
            self._active_pairs.discard(pair)
            self._switching_pairs.discard(pair)
            self._pair_force_close_sent.discard(pair)

        self._quarantine_events.append(now)
        self._quarantine_events = [t for t in self._quarantine_events if now - t <= 3600]

        max_q = int(self.config.max_quarantines_per_hour or 0)
        if (
            max_q > 0
            and len(self._quarantine_events) >= max_q
            and not (self._trading_halted and self._halt_reason == "kill_switch")
        ):
            self._trading_halted = True
            self._halt_reason = "kill_switch"
            self.logger().critical(
                f"[KILL-SWITCH] {len(self._quarantine_events)} pair quarantines "
                f"within 1h >= {max_q}. Halting NEW orders on ALL pairs."
            )

    # --------------------------------------------------- drawdown halt
    def _realized_pnl_quote(self) -> Decimal:
        total = Decimal("0")
        for e in self.get_all_executors():
            if not getattr(e, "is_done", False):
                continue
            v = getattr(e, "net_pnl_quote", None) or Decimal("0")
            try:
                total += Decimal(str(v))
            except Exception:
                continue
        return total

    def _check_drawdown(self) -> None:
        drawdown_cooldown = 60
        if self._trading_halted and self._halt_reason == "drawdown":
            now = self.current_timestamp
            if now - self._halt_ts >= drawdown_cooldown:
                self._trading_halted = False
                self._halt_reason = None
                self.logger().info(
                    f"[DRAWDOWN] Cooldown elapsed ({drawdown_cooldown}s), "
                    f"resuming trading."
                )
            else:
                return
        if self.config.max_drawdown_pct <= 0:
            return
        budget = self.config.total_amount_quote
        if budget <= 0:
            return
        pnl = self._realized_pnl_quote()
        threshold = -(budget * self.config.max_drawdown_pct)
        if pnl <= threshold:
            self._trading_halted = True
            self._halt_reason = "drawdown"
            self._halt_ts = self.current_timestamp
            self.logger().critical(
                f"[DRAWDOWN HALT] Realized PnL {pnl:+.2f} USDT <= "
                f"-{self.config.max_drawdown_pct * 100:.2f}% of "
                f"budget {budget:.2f} USDT. NEW orders halted for {drawdown_cooldown}s."
            )

    # ------------------------------------------------- consecutive failure detection
    def _check_consecutive_order_failures(self) -> None:
        if self._trading_halted:
            return
        now = self.current_timestamp
        recent_failures = 0
        for e in self.get_all_executors():
            if not getattr(e, "is_done", False):
                continue
            ct = getattr(e, "close_type", None)
            if ct != CloseType.FAILED:
                continue
            created = getattr(e.config, "timestamp", 0) or 0
            if now - created <= 60:
                recent_failures += 1

        if recent_failures >= self._failure_spam_threshold:
            self._trading_halted = True
            self._halt_reason = "order failures"
            self.logger().critical(
                f"[FAILURE HALT] {recent_failures} executors failed with "
                f"CloseType.FAILED within the last 60 seconds. "
                f"NEW orders halted. Check exchange margin mode / position mode "
                f"settings and restart the strategy."
            )

    # --------------------------------------------------------------- on_stop

    async def on_stop(self):
        self.logger().info("[STOP] Cancelling idle executors...")
        for pair in list(self._active_pairs):
            for e in self._get_active_executors_for_pair(pair):
                if not self._executor_has_filled(e):
                    try:
                        self.executor_orchestrator.execute_action(
                            StopExecutorAction(executor_id=e.id)
                        )
                    except Exception as ex:
                        self.logger().warning(
                            f"[STOP] Could not stop executor {str(e.id)[:8]}: {ex}"
                        )

        if self._rest_scan_task and not self._rest_scan_task.done():
            self._rest_scan_task.cancel()
            self._rest_scan_task = None

        if self._rest_session is not None and not self._rest_session.closed:
            await self._rest_session.close()
            self._rest_session = None

    # --------------------------------------------------------- stale LIMIT close monitor
    def _monitor_and_cancel_stale_close_orders(self) -> None:
        timeout = self.config.close_order_timeout_sec
        if timeout <= 0:
            return

        now = self.current_timestamp
        to_remove = []
        requeue_eids: Set[str] = set()

        for oid, info in list(self._pending_close_orders.items()):
            age = now - info["ts"]
            if age >= timeout:
                cn = info["cn"]
                pair = info["pair"]
                eid = info["eid"]
                connector = self.connectors.get(cn)
                if connector:
                    try:
                        connector.cancel(pair, oid)
                        self.logger().info(
                            f"[CLOSE-CANCEL] Cancelled stale LIMIT close {oid[:12]} "
                            f"on {cn} {pair} (age={age:.0f}s, side={info['side']})"
                        )
                    except Exception as ex:
                        self.logger().debug(
                            f"[CLOSE-CANCEL] Order {oid[:12]} already filled/closed or cancel skipped: {ex}"
                        )
                requeue_eids.add(eid)
                to_remove.append(oid)

        for oid in to_remove:
            self._pending_close_orders.pop(oid, None)

        # Unmark cancelled hedges so they get re-evaluated on next tick
        for eid in requeue_eids:
            self._pending_close_eids.discard(eid)
            self.logger().info(f"[CLOSE-REQUEUE] {eid[:8]} requeued for re-evaluation")

    # --------------------------------------------------------- on_tick logic

    def on_tick(self):
        if self._is_stop_triggered:
            return
        self._tick_counter += 1
        if not self.market_data_provider.ready:
            if self._tick_counter % 60 == 0:
                self.logger().warning(
                    "market_data_provider not ready — skipping tick. "
                    f"Connectors: maker={self.config.maker_connector}, taker={self.config.taker_connector}"
                )
            return
        self.update_executors_info()
        self._validate_perp_setup()
        self._close_hedges_by_executable_pnl()
        self._monitor_and_cancel_stale_close_orders()
        self._monitor_hanging_hedges()
        self._check_drawdown()
        self._check_consecutive_order_failures()

        if self._rest_universe and AIOHTTP_AVAILABLE:
            if self._rest_scan_task is not None and self._rest_scan_task.done():
                try:
                    self._rest_scan_task.result()
                except Exception as exc:
                    self.logger().error(f"[REST SCAN] Background task crashed: {exc}")
                self._rest_scan_task = None
                self._rest_scan_restart_ts = time.time()
            if (
                self._rest_scan_task is None
                and time.time() - self._rest_scan_restart_ts >= 30
            ):
                self._rest_scan_task = asyncio.ensure_future(self._rest_scan_loop())

        if self._pending_ob_subscriptions and (
            self._ob_subscribe_task is None or self._ob_subscribe_task.done()
        ):
            if self._ob_subscribe_task is not None:
                try:
                    self._ob_subscribe_task.result()
                except Exception as exc:
                    self.logger().error(f"[OB-SUB] Subscription task crashed: {exc}")
            pending = set(self._pending_ob_subscriptions)
            self._pending_ob_subscriptions.clear()
            self._ob_subscribe_task = asyncio.ensure_future(
                self._subscribe_orderbooks(pending)
            )

        actions = self.determine_executor_actions()
        if actions:
            self.logger().info(f"on_tick: executing {len(actions)} executor action(s)")
        for action in actions:
            self.executor_orchestrator.execute_action(action)

    def determine_executor_actions(self) -> List[ExecutorAction]:
        executor_actions: List[ExecutorAction] = []
        now = self.current_timestamp

        if now - self._last_scan_ts >= self.config.scan_interval:
            self._scan_and_rank_pairs()
            self._last_scan_ts = now

            if not self._trading_halted:
                for new_pair in self._get_top_pairs():
                    if self.config.lazy_setup:
                        if not self._lazy_setup_pair(new_pair):
                            self.logger().warning(
                                f"Skipping {new_pair}: lazy setup failed"
                            )
                            continue
                    self._active_pairs.add(new_pair)
                    self._pair_activated_ts[new_pair] = now
                    self._pair_force_close_sent.discard(new_pair)

            finished_switching: Set[str] = set()
            for pair in self._switching_pairs:
                if not self._get_active_executors_for_pair(pair):
                    finished_switching.add(pair)
                    self._active_pairs.discard(pair)
                    self._pair_force_close_sent.discard(pair)
            self._switching_pairs -= finished_switching

        imbalance_cap = self._imbalance_cap()

        for pair in list(self._active_pairs):
            spread_data = self._calculate_spread(pair)
            has_ws_data = spread_data is not None
            if spread_data is None and pair in self._rest_universe:
                spread_data = self._calculate_spread_from_rest_sync(pair)
            if spread_data is None:
                continue
            best_spread, buy_spread, sell_spread, mid_price = spread_data

            min_eligible = self._min_eligible_raw_spread(pair)
            if best_spread < min_eligible * Decimal("0.5"):
                executor_actions.extend(self._stop_idle_executors(pair))
                self._active_pairs.discard(pair)
                self._switching_pairs.discard(pair)
                self._pair_force_close_sent.discard(pair)
                continue

            if pair in self._switching_pairs:
                if pair not in self._pair_force_close_sent:
                    stop_acts = self._stop_idle_executors(pair)
                    if stop_acts:
                        self.logger().info(
                            f"Rotating out {pair}: cancelling {len(stop_acts)} idle executors"
                        )
                        executor_actions.extend(stop_acts)
                    self._pair_force_close_sent.add(pair)
                continue

            executor_actions.extend(self._cancel_stale_idle_executors(pair))

            if self._trading_halted:
                continue
            if pair in self._quarantined_pairs:
                continue
            if not has_ws_data:
                continue

            notional_imbalance = self._notional_imbalance(pair)
            buy_cost = self._cost_pct(pair, TradeType.BUY)
            sell_cost = self._cost_pct(pair, TradeType.SELL)
            active_buy = self._get_active_buy_executors_by_level(pair)
            active_sell = self._get_active_sell_executors_by_level(pair)

            for target_profit, amount_quote in self._parsed_buy_levels:
                if active_buy.get(target_profit):
                    continue
                if notional_imbalance >= imbalance_cap:
                    break
                if buy_spread < target_profit + buy_cost:
                    continue
                level_notional = self._level_notional(TradeType.BUY, amount_quote)
                amount_base = level_notional / mid_price
                if not self._taker_depth_ok(pair, TradeType.BUY, amount_base):
                    continue
                executor_actions.append(
                    self._build_executor_action(
                        pair, mid_price, TradeType.BUY, target_profit,
                        amount_quote, amount_base,
                    )
                )

            for target_profit, amount_quote in self._parsed_sell_levels:
                if active_sell.get(target_profit):
                    continue
                if notional_imbalance <= -imbalance_cap:
                    break
                if sell_spread < target_profit + sell_cost:
                    continue
                level_notional = self._level_notional(TradeType.SELL, amount_quote)
                amount_base = level_notional / mid_price
                if not self._taker_depth_ok(pair, TradeType.SELL, amount_base):
                    continue
                executor_actions.append(
                    self._build_executor_action(
                        pair, mid_price, TradeType.SELL, target_profit,
                        amount_quote, amount_base,
                    )
                )

        return executor_actions

    # -------------------------------------------------------- executor build

    def _total_notional_budget(self) -> Decimal:
        return (
            self.config.total_amount_quote
            * Decimal(self.config.leverage)
            * self.config.budget_utilization
        )

    def _level_notional(self, maker_side: TradeType, weight: Decimal) -> Decimal:
        per_pair = self._total_notional_budget() / Decimal(self.config.max_active_pairs)
        per_side = per_pair / Decimal("2")
        weight_sum = (
            self._buy_weight_sum if maker_side == TradeType.BUY else self._sell_weight_sum
        )
        return per_side * (weight / weight_sum)

    def _build_executor_action(
        self,
        pair: str,
        mid_price: Decimal,
        maker_side: TradeType,
        target_profit: Decimal,
        amount_quote: Decimal,
        order_amount_base: Optional[Decimal] = None,
    ) -> CreateExecutorAction:
        min_profit = max(target_profit - PROFIT_BOUND_LOWER_BPS, Decimal("0"))
        max_profit = target_profit + PROFIT_BOUND_UPPER_BPS
        if order_amount_base is None:
            level_notional = self._level_notional(maker_side, amount_quote)
            order_amount_base = level_notional / mid_price

        if maker_side == TradeType.BUY:
            buying = ConnectorPair(
                connector_name=self.config.maker_connector, trading_pair=pair
            )
            selling = ConnectorPair(
                connector_name=self.config.taker_connector, trading_pair=pair
            )
        else:
            buying = ConnectorPair(
                connector_name=self.config.taker_connector, trading_pair=pair
            )
            selling = ConnectorPair(
                connector_name=self.config.maker_connector, trading_pair=pair
            )

        config = XEMMExecutorConfig(
            timestamp=self.current_timestamp,
            buying_market=buying,
            selling_market=selling,
            maker_side=maker_side,
            order_amount=order_amount_base,
            min_profitability=min_profit,
            target_profitability=target_profit,
            max_profitability=max_profit,
        )
        return CreateExecutorAction(executor_config=config)

    # ------------------------------------------------------------ formatting

    def format_status(self) -> str:
        original = super().format_status()
        lines: List[str] = []
        lines.append(f"\n{'=' * 90}")
        lines.append("  XEMM PERPETUAL TIERED SCANNER V3")
        lines.append(f"{'=' * 90}")
        lines.append(
            f"  Maker: {self.config.maker_connector} | Taker: {self.config.taker_connector}"
        )
        lines.append(
            f"  Leverage: {self.config.leverage}x | Position mode: {self.config.position_mode.value} "
            f"| Margin mode: {self.config.margin_mode}"
        )
        total_notional = self._total_notional_budget()
        per_pair = total_notional / Decimal(self.config.max_active_pairs)
        per_side = per_pair / Decimal("2")
        lines.append(
            f"  Budget: margin ${self.config.total_amount_quote:.2f} x "
            f"{self.config.leverage}x x {self.config.budget_utilization * 100:.0f}% = "
            f"${total_notional:.2f} notional"
        )
        lines.append(
            f"  Per pair: ${per_pair:.2f} (BUY ${per_side:.2f} + SELL ${per_side:.2f})"
        )
        lines.append(
            f"  Discovery: auto={self.config.auto_discover_pairs} | "
            f"Total={len(self._all_discovered_pairs)} pairs "
            f"(WS={len(self._ws_universe)} + REST={len(self._rest_universe)}) | "
            f"Active: {', '.join(sorted(self._active_pairs)) or 'None'} | "
            f"Max: {self.config.max_active_pairs}"
        )
        lines.append(
            f"  Lazy setup: {self.config.lazy_setup} | "
            f"Setup done: {len(self._setup_pairs)}/{len(self._all_discovered_pairs)} | "
            f"Auto-promote: {self.config.auto_promote}"
        )
        lines.append(
            f"  Costs (mkr+tkr+slip+fund): {self._cost_pct() * 100:.3f}% | "
            f"Min eligible spread: {self._min_eligible_raw_spread() * 100:.3f}% | "
            f"Scan: WS={self.config.scan_interval}s REST={self.config.rest_scan_interval}s"
        )

        halt_str = (
            f"HALTED ({self._halt_reason})" if self._trading_halted else "OK"
        )
        setup_str = (
            "VALIDATED" if self._perp_setup_validated
            else ("FAILED" if self._perp_setup_failed else "pending")
        )
        realized = self._realized_pnl_quote()
        dd_threshold = -self.config.total_amount_quote * self.config.max_drawdown_pct
        rest_age = time.time() - self._last_rest_scan_ts if self._last_rest_scan_ts > 0 else -1
        lines.append(
            f"  Trading: {halt_str} | Perp setup: {setup_str} | "
            f"PnL: {realized:+.4f} USDT (DD halt @ {dd_threshold:.2f}) | "
            f"Hanging: {len(self._hanging_hedge_executor_ids)}"
        )
        lines.append(
            f"  Imbalance cap: ${self._imbalance_cap():.2f} per pair | "
            f"Mark-div cutoff: {self.config.max_mark_divergence_pct * 100:.3f}% | "
            f"Dynamic funding: {self.config.use_dynamic_funding}"
        )
        lines.append(
            f"  REST scan: {'active' if (self._rest_scan_task and not self._rest_scan_task.done()) else 'off'} | "
            f"Last: {rest_age:.0f}s ago | Errors: {self._rest_errors} | "
            f"Promoted: {len(self._rest_promoted_pairs)} | "
            f"OB-sub: {len(self._ob_subscribed_pairs)} pending: {len(self._pending_ob_subscriptions)}"
        )
        if self._quarantined_pairs or self._quarantine_events:
            now = self.current_timestamp
            recent_q = [t for t in self._quarantine_events if now - t <= 3600]
            q_pairs = ", ".join(sorted(self._quarantined_pairs)) or "-"
            lines.append(
                f"  QUARANTINED ({len(self._quarantined_pairs)}): {q_pairs} | "
                f"Events 1h: {len(recent_q)}/{self.config.max_quarantines_per_hour}"
            )

        lines.append("\n  BUY LEVELS (net target, weight -> USDT notional):")
        for i, (p, w) in enumerate(self._parsed_buy_levels, 1):
            notional = self._level_notional(TradeType.BUY, w)
            lines.append(f"    L{i}: {p * 100:.3f}% | w={w:g} -> ${notional:.2f}")
        lines.append("  SELL LEVELS:")
        for i, (p, w) in enumerate(self._parsed_sell_levels, 1):
            notional = self._level_notional(TradeType.SELL, w)
            lines.append(f"    L{i}: {p * 100:.3f}% | w={w:g} -> ${notional:.2f}")

        if self._spread_ranking:
            ws_ranked = [(p, b, buy, sell) for p, b, buy, sell, t in self._spread_ranking if t == "WS"]
            rest_ranked = [(p, b, buy, sell) for p, b, buy, sell, t in self._spread_ranking if t == "REST"]

            if ws_ranked:
                lines.append(f"\n  WS TIER TOP 15 (of {len(ws_ranked)}):")
                lines.append(
                    f"  {'#':<3} {'Pair':<16} {'Best%':<9} {'Buy%':<9} {'Sell%':<9} {'Status':<10}"
                )
                lines.append(
                    f"  {'-' * 3:<3} {'-' * 16:<16} {'-' * 9:<9} {'-' * 9:<9} "
                    f"{'-' * 9:<9} {'-' * 10:<10}"
                )
                for i, (pair, best, buy, sell) in enumerate(ws_ranked[:15], 1):
                    min_eligible = self._min_eligible_raw_spread(pair)
                    if pair in self._quarantined_pairs:
                        st = "QUARANTINE"
                    elif pair in self._switching_pairs:
                        st = "SWITCH"
                    elif pair in self._active_pairs:
                        st = "ACTIVE"
                    elif best >= min_eligible:
                        st = "ELIGIBLE"
                    else:
                        st = "-"
                    lines.append(
                        f"  {i:<3} {pair:<16} {best * 100:>6.3f}%  "
                        f"{buy * 100:>6.3f}%  {sell * 100:>6.3f}%  {st:<10}"
                    )

            if rest_ranked:
                lines.append(f"\n  REST TIER TOP 15 (of {len(rest_ranked)}):")
                lines.append(
                    f"  {'#':<3} {'Pair':<16} {'Best%':<9} {'Buy%':<9} {'Sell%':<9} {'Note':<10}"
                )
                for i, (pair, best, buy, sell) in enumerate(rest_ranked[:15], 1):
                    min_eligible = self._min_eligible_raw_spread(pair)
                    if pair in self._rest_promoted_pairs:
                        note = "PROMOTED"
                    elif best >= min_eligible:
                        note = "ELIGIBLE*"
                    else:
                        note = "-"
                    lines.append(
                        f"  {i:<3} {pair:<16} {best * 100:>6.3f}%  "
                        f"{buy * 100:>6.3f}%  {sell * 100:>6.3f}%  {note:<10}"
                    )
                lines.append(
                    "  * ELIGIBLE REST pairs: add to scan_pairs and restart to trade"
                )

        all_execs = self.get_all_executors()
        active = [e for e in all_execs if not e.is_done]
        done = [e for e in all_execs if e.is_done and self._executor_has_filled(e)]

        if active:
            lines.append(f"\n  ACTIVE EXECUTORS ({len(active)}):")
            lines.append(
                f"  {'ID':<10} {'Pair':<16} {'Side':<5} {'Target%':<9} {'Age':<8} {'Status':<14}"
            )
            for e in active[:20]:
                pair = e.config.buying_market.trading_pair
                side = "BUY" if e.config.maker_side == TradeType.BUY else "SELL"
                target = f"{e.config.target_profitability * 100:.3f}%"
                age = f"{e.age:.0f}s" if getattr(e, "age", None) else "N/A"
                status = getattr(e, "status", None)
                status_name = status.name if status else "-"
                marker = "!HEDGE" if e.id in self._hanging_hedge_executor_ids else ""
                lines.append(
                    f"  {str(e.id)[:8]:<10} {pair:<16} {side:<5} {target:<9} {age:<8} "
                    f"{status_name:<14} {marker}"
                )

        if done:
            total_pnl = sum(
                getattr(e, "net_pnl_quote", Decimal("0")) or Decimal("0") for e in done
            )
            lines.append(
                f"\n  COMPLETED ({len(done)}): Total Net PnL = {total_pnl:+.4f} USDT"
            )
            recent = sorted(
                done, key=lambda e: getattr(e, "close_timestamp", 0) or 0, reverse=True
            )[:5]
            for e in recent:
                pnl = getattr(e, "net_pnl_quote", Decimal("0")) or Decimal("0")
                lines.append(
                    f"    {e.config.buying_market.trading_pair}: {pnl:+.4f} USDT"
                )

        lines.append(f"{'=' * 90}\n")
        return original + "\n".join(lines)
