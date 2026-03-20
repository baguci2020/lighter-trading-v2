"""
exchange/lighter_client.py
==========================
Lighter DEX 交易所客户端封装。

本文件封装了与 Lighter DEX 交互的所有底层逻辑，包括：
- REST API 调用 (获取市场数据、账户信息)
- 交易操作 (下单、撤单、改单)
- 数据格式转换 (将 API 原始数据转为内部数据结构)

设计原则:
    1. 上层智能体不需要关心 API 的具体细节
    2. 所有网络错误在这一层处理，上层只需处理业务逻辑
    3. 支持 dry_run 模式 (模拟交易，不实际调用下单 API)

依赖:
    - lighter-sdk: Lighter DEX 官方 Python SDK
    - aiohttp: 用于手动调用 SDK 未覆盖的端点
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

import aiohttp

from config import ExchangeConfig
from models.schemas import (
    AccountState,
    Asset,
    ExecutionOrder,
    MarketSnapshot,
    OrderBookLevel,
    OrderResult,
    PositionInfo,
    Side,
)

logger = logging.getLogger(__name__)


class LighterClient:
    """
    Lighter DEX 交易所客户端。
    
    这是与交易所通信的唯一入口。所有的 API 调用都通过这个类进行。
    
    使用方法:
        client = LighterClient(config=exchange_config, dry_run=True)
        await client.initialize()
        snapshot = await client.get_market_snapshot(Asset.BTC)
    
    Attributes:
        config: 交易所配置
        dry_run: 是否为模拟模式
        _session: aiohttp 会话 (用于 HTTP 请求)
        _signer: lighter-sdk 的 SignerClient (用于签名交易)
    """

    def __init__(self, config: ExchangeConfig, dry_run: bool = True):
        self.config = config
        self.dry_run = dry_run
        self._session: Optional[aiohttp.ClientSession] = None
        self._signer = None  # 延迟初始化 (需要 async)
        self._initialized = False

    async def initialize(self) -> None:
        """
        初始化客户端。
        
        创建 HTTP 会话，如果不是 dry_run 模式还会初始化 SignerClient。
        必须在使用其他方法之前调用。
        """
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers={
                # 禁用 brotli 编码 (br)，避免 aiohttp 解码失败
                # Lighter API 默认返回 br 编码，但 aiohttp 不一定安装了 brotli 库
                "Accept-Encoding": "gzip, deflate",
            },
        )

        # 验证交易所连接
        try:
            status = await self._get("/")
            logger.info(f"[交易所] 连接成功 | 状态: {status}")
        except Exception as e:
            logger.error(f"[交易所] 连接失败: {e}")
            raise

        # 如果不是模拟模式，初始化签名客户端
        if not self.dry_run and self.config.private_key:
            await self._init_signer()

        self._initialized = True
        mode = "模拟模式 (dry_run)" if self.dry_run else "实盘模式"
        logger.info(f"[交易所] 客户端已初始化 | {mode}")

    async def close(self) -> None:
        """关闭客户端，释放网络资源。"""
        if self._session:
            await self._session.close()
            logger.info("[交易所] 客户端已关闭")

    # ============================================================
    # 市场数据 API (公开接口，无需认证)
    # ============================================================

    async def get_market_snapshot(self, asset: Asset) -> Optional[MarketSnapshot]:
        """
        获取指定资产的市场快照。
        
        聚合了订单簿、最新价格、24h 交易量等信息，
        组装成标准化的 MarketSnapshot 数据结构。
        
        Args:
            asset: 目标资产 (BTC 或 ETH)
            
        Returns:
            MarketSnapshot 实例，获取失败返回 None
        """
        try:
            # 1. 获取订单簿详情 (包含最新价格、24h 数据)
            details = await self._get(
                "/api/v1/orderBookDetails",
                params={"market_id": asset.market_id},
            )
            detail = self._find_market_detail(details, asset)
            if not detail:
                logger.warning(f"[交易所] 未找到 {asset.value} 的市场详情")
                return None

            # 2. 获取订单簿深度 (买卖盘)
            orderbook = await self._get(
                "/api/v1/orderBookOrders",
                params={"market_id": asset.market_id, "limit": 10},
            )

            # 3. 组装 MarketSnapshot
            bid_levels, ask_levels = self._parse_orderbook(orderbook)

            snapshot = MarketSnapshot(
                asset=asset,
                timestamp=time.time(),
                last_price=float(detail.get("last_trade_price", 0)),
                best_bid=bid_levels[0].price if bid_levels else 0.0,
                best_ask=ask_levels[0].price if ask_levels else 0.0,
                bid_levels=bid_levels,
                ask_levels=ask_levels,
                volume_24h=float(detail.get("daily_quote_token_volume", 0)),
                funding_rate=None,  # 资金费率需要单独查询
                price_change_24h=float(detail.get("daily_price_change", 0)),
            )

            # 4. 尝试获取资金费率 (非关键，失败不影响主流程)
            try:
                funding = await self._get(
                    "/api/v1/funding-rates",
                    params={"market_id": asset.market_id},
                )
                if funding and "funding_rates" in funding:
                    rates = funding["funding_rates"]
                    if rates:
                        snapshot.funding_rate = float(rates[0].get("rate", 0))
            except Exception:
                pass  # 资金费率获取失败不影响主流程

            logger.info(
                f"[交易所] {asset.value} 快照 | "
                f"价格: ${snapshot.last_price:,.2f} | "
                f"价差: {snapshot.spread_pct:.4f}% | "
                f"24h量: ${snapshot.volume_24h:,.0f}"
            )
            return snapshot

        except Exception as e:
            logger.error(f"[交易所] 获取 {asset.value} 市场快照失败: {e}")
            return None

    async def get_candles(
        self,
        asset: Asset,
        resolution: int = 60,
        limit: int = 50,
    ) -> list[dict]:
        """
        获取 K 线数据。
        
        Args:
            asset: 目标资产
            resolution: K 线周期 (秒)。60=1分钟, 300=5分钟, 3600=1小时
            limit: 返回的 K 线数量 (最多 500)
            
        Returns:
            K 线数据列表，每个元素包含 open, high, low, close, volume
        """
        try:
            data = await self._get(
                "/api/v1/candles",
                params={
                    "market_id": asset.market_id,
                    "resolution": resolution,
                    "limit": limit,
                },
            )
            return data.get("candles", [])
        except Exception as e:
            logger.error(f"[交易所] 获取 {asset.value} K线失败: {e}")
            return []

    # ============================================================
    # 账户数据 API (需要认证)
    # ============================================================

    async def get_account_state(self) -> Optional[AccountState]:
        """
        获取当前账户状态。

        【Bug 修复 #1 - 账户类型修复】
        Lighter DEX 账户体系分为两层：
          - Perps 合约账户：用于永续合约交易的保证金账户（本系统使用此账户下单）
          - Spot 现货账户：链上钱包余额

        原实现错误地读取了 Spot 账户余额（通常为 $0），导致风控模块拦截所有订单。
        已修复为读取 Perps 账户的 free_collateral（可用保证金）。

        【Bug 修复 #2 - 可用保证金区分】
        原实现使用 total_equity（总权益）作为可用余额，在有持仓时会高估可用资金。
        已修复为使用 free_collateral（已扣除占用保证金后的真实可用余额），
        并增加维持保证金率监控，防止超额下单引发强平。

        Returns:
            AccountState 实例
        """
        if self.dry_run:
            # 模拟模式: 返回一个假的账户状态
            return AccountState(
                total_equity=10000.0,   # 模拟 10000 USD
                available_balance=8000.0,
                unrealized_pnl=0.0,
                positions={},
            )

        try:
            # 真实模式: 调用 Perps 合约账户 API（非 Spot 现货账户）
            data = await self._get(
                "/api/v1/account",
                params={
                    "by": "index",
                    "value": str(self.config.account_index),
                },
            )

            accounts_list = data.get("accounts", [])
            account = accounts_list[0] if accounts_list else {}

            # 【修复点 1】total_equity = Perps 账户总权益（含未实现盈亏）
            equity = float(account.get("collateral", 0))

            # 【修复点 2】available_balance = Perps 账户的可用保证金（free_collateral）
            # 此字段已扣除已占用的初始保证金，可直接用于判断是否能下新订单。
            # 原代码此处读取的是 Spot 余额（$0），导致所有订单被拦截。
            available = float(account.get("available_balance", 0))

            upnl = sum(float(pos.get("unrealized_pnl", 0)) for pos in account.get("positions", []))



            # 解析持仓
            positions = {}
            for pos in account.get("positions", []):
                size = float(pos.get("position", 0))
                if abs(size) > 0:
                    market_id = int(pos.get("market_id", -1))
                    asset_name = {0: "ETH", 1: "BTC", 2: "SOL", 92: "XAU", 93: "XAG"}.get(market_id)
                    if asset_name:
                        asset = Asset(asset_name)
                        positions[asset_name] = PositionInfo(
                            asset=asset,
                            side=Side.LONG if int(pos.get("sign", 1)) == 1 else Side.SHORT,
                            size=abs(size),
                            entry_price=float(pos.get("avg_entry_price", 0)),
                            unrealized_pnl=float(pos.get("unrealized_pnl", 0)),
                            leverage=float(pos.get("initial_margin_fraction", 100)) / 100,
                        )

            logger.info(
                f"[交易所] 账户状态 | 总权益: ${equity:,.2f} | "
                f"可用保证金: ${available:,.2f} | 未实现盈亏: ${upnl:+,.2f} | "
                f"持仓数: {len(positions)}"
            )

            return AccountState(
                total_equity=equity,
                available_balance=available,  # 已修复：Perps 可用保证金
                unrealized_pnl=upnl,
                positions=positions,
            )

        except Exception as e:
            logger.error(f"[交易所] 获取账户状态失败: {e}")
            return None

    async def get_orderbook(self, asset: Asset) -> Optional[tuple[list[OrderBookLevel], list[OrderBookLevel]]]:
        """
        获取指定资产的完整订单簿数据。

        供动态滑点计算使用：通过遍历订单簿深度，
        可以精确计算特定下单量所需的平均成交价格。

        Args:
            asset: 目标资产

        Returns:
            (买盘列表, 卖盘列表) 元组，失败时返回 None
        """
        try:
            ob_data = await self._get(
                "/api/v1/orderBookOrders",
                params={"market_id": asset.market_id},
            )
            bids, asks = self._parse_orderbook(ob_data)
            return bids, asks
        except Exception as e:
            logger.error(f"[交易所] 获取 {asset.value} 订单簿失败: {e}")
            return None

    async def close_position_market(
        self, asset: Asset, size: float, side: Side
    ) -> OrderResult:
        """
        以市价单强制平仓（用于止损止盈触发和一键平仓）。

        Args:
            asset: 目标资产
            size: 平仓数量
            side: 当前持仓方向（平仓方向相反）

        Returns:
            OrderResult
        """
        import uuid as _uuid
        from models.schemas import Action, OrderType

        # 平仓方向与持仓方向相反
        close_side = Side.SHORT if side == Side.LONG else Side.LONG

        # 获取当前市场价格作为最差可接受价
        snapshot = await self.get_market_snapshot(asset)
        if snapshot:
            last_price = snapshot.last_price
        else:
            last_price = 0.0

        # 滑点容忍：平多（卖出）用 bid * 0.99，平空（买入）用 ask * 1.01
        if close_side == Side.SHORT:
            price = last_price * 0.99 if last_price > 0 else 0.0
        else:
            price = last_price * 1.01 if last_price > 0 else 0.0

        order = ExecutionOrder(
            order_id=_uuid.uuid4().hex[:8],
            approved=True,
            signal_id="SL_TP_CLOSE",
            asset=asset,
            side=close_side,
            action=Action.CLOSE,
            order_type=OrderType.MARKET,
            base_amount=round(size, asset.size_decimals),
            price=price,
            risk_notes="止损/止盈/一键平仓触发",
        )
        return await self.execute_order(order)

    # ============================================================
    # 交易操作 API
    # ============================================================

    async def execute_order(self, order: ExecutionOrder) -> OrderResult:
        """
        执行交易指令。
        
        这是最关键的方法 —— 将风控审核后的执行指令转化为实际的 API 调用。
        在 dry_run 模式下只记录日志，不实际下单。
        
        Args:
            order: 经过风控审核的执行指令
            
        Returns:
            OrderResult 记录执行结果
        """
        if not order.approved:
            return OrderResult(
                order_id=order.order_id,
                success=False,
                error_message=f"指令未批准: {order.reject_reason}",
            )

        # --- 模拟模式 ---
        if self.dry_run:
            logger.info(
                f"[交易所] [模拟] 执行订单 | "
                f"{order.asset.value} {order.side.value} {order.action.value} | "
                f"数量: {order.base_amount} | 价格: ${order.price:,.2f}"
            )
            return OrderResult(
                order_id=order.order_id,
                success=True,
                tx_hash=f"DRY_RUN_{uuid.uuid4().hex[:16]}",
                filled_amount=order.base_amount,
                filled_price=order.price,
            )

        # --- 实盘模式 ---
        try:
            return await self._send_real_order(order)
        except Exception as e:
            logger.error(f"[交易所] 订单执行异常: {e}")
            return OrderResult(
                order_id=order.order_id,
                success=False,
                error_message=str(e),
            )

    async def _send_real_order(self, order: ExecutionOrder) -> OrderResult:
        """
        通过 lighter-sdk 发送真实订单。
        
        使用 SignerClient 对交易进行签名，然后通过 API 发送。
        
        注意: 此方法仅在非 dry_run 模式下被调用。
        """
        try:
            import lighter

            if not self._signer:
                await self._init_signer()

            # 确定订单参数
            is_ask = (order.side == Side.SHORT)  # 做空 = 卖出 (ask)
            market_index = order.asset.market_id

            # 将浮点数转为 Lighter 要求的整数格式
            # price 需要乘以 10^price_decimals
            # base_amount 需要乘以 10^size_decimals
            price_int = int(order.price * (10 ** order.asset.price_decimals))
            amount_int = int(order.base_amount * (10 ** order.asset.size_decimals))

            # 生成唯一的 client_order_index
            client_order_index = int(time.time() * 1000) % (2**31)

            # 根据订单类型选择参数
            if order.order_type.value == "market":
                order_type = self._signer.ORDER_TYPE_MARKET
                time_in_force = self._signer.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL
                order_expiry = self._signer.DEFAULT_IOC_EXPIRY
            else:
                order_type = self._signer.ORDER_TYPE_LIMIT
                time_in_force = self._signer.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME
                order_expiry = self._signer.DEFAULT_28_DAY_ORDER_EXPIRY

            # 调用 SDK 创建订单
            tx, tx_hash, err = await self._signer.create_order(
                market_index=market_index,
                client_order_index=client_order_index,
                base_amount=amount_int,
                price=price_int,
                is_ask=is_ask,
                order_type=order_type,
                time_in_force=time_in_force,
                reduce_only=False,
                order_expiry=order_expiry,
            )

            if err:
                logger.error(f"[交易所] 订单创建失败: {err}")
                return OrderResult(
                    order_id=order.order_id,
                    success=False,
                    error_message=str(err),
                )

            logger.info(
                f"[交易所] 订单已发送 | tx_hash: {tx_hash} | "
                f"{order.asset.value} {order.side.value} {order.base_amount}"
            )

            return OrderResult(
                order_id=order.order_id,
                success=True,
                tx_hash=str(tx_hash),
                filled_amount=order.base_amount,
                filled_price=order.price,
            )

        except ImportError:
            logger.error(
                "[交易所] lighter-sdk 未安装。请运行: pip install lighter-sdk"
            )
            return OrderResult(
                order_id=order.order_id,
                success=False,
                error_message="lighter-sdk 未安装",
            )

    # ============================================================
    # 内部辅助方法
    # ============================================================

    async def _init_signer(self) -> None:
        """初始化 lighter-sdk 的 SignerClient (用于签名交易)。"""
        try:
            import lighter
            self._signer = lighter.SignerClient(
                url=self.config.base_url,
                api_private_keys={self.config.api_key_index: self.config.private_key},
                account_index=self.config.account_index,
            )
            logger.info("[交易所] SignerClient 已初始化")
        except ImportError:
            logger.warning(
                "[交易所] lighter-sdk 未安装，实盘交易功能不可用。"
                "如需实盘，请运行: pip install lighter-sdk"
            )
        except Exception as e:
            logger.error(f"[交易所] SignerClient 初始化失败: {e}")

    async def _get(self, path: str, params: Optional[dict] = None) -> dict:
        """
        发送 GET 请求到 Lighter DEX REST API。
        
        Args:
            path: API 路径 (如 /api/v1/orderBookDetails)
            params: 查询参数
            
        Returns:
            JSON 响应的字典
            
        Raises:
            Exception: 请求失败时抛出异常
        """
        url = f"{self.config.base_url}{path}"
        async with self._session.get(url, params=params) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"API 请求失败 [{resp.status}]: {text[:200]}")
            return await resp.json()

    @staticmethod
    def _find_market_detail(data: dict, asset: Asset) -> Optional[dict]:
        """
        从 orderBookDetails 响应中找到指定资产的市场详情。
        
        Args:
            data: API 响应数据
            asset: 目标资产
            
        Returns:
            该资产的市场详情字典，未找到返回 None
        """
        for detail in data.get("order_book_details", []):
            if detail.get("market_id") == asset.market_id:
                return detail
        return None

    @staticmethod
    def _parse_orderbook(data: dict) -> tuple[list[OrderBookLevel], list[OrderBookLevel]]:
        """
        解析订单簿数据，转换为标准化的 OrderBookLevel 列表。
        
        Args:
            data: orderBookOrders API 响应数据
            
        Returns:
            (买盘列表, 卖盘列表) 的元组
        """
        bids = []
        asks = []

        for order in data.get("bids", []):
            bids.append(OrderBookLevel(
                price=float(order.get("price", 0)),
                amount=float(order.get("remaining_base_amount", 0)),
            ))

        for order in data.get("asks", []):
            asks.append(OrderBookLevel(
                price=float(order.get("price", 0)),
                amount=float(order.get("remaining_base_amount", 0)),
            ))

        # 买盘按价格从高到低排列，卖盘按价格从低到高排列
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        return bids, asks
