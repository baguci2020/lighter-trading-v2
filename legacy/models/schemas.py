"""
models/schemas.py
=================
核心数据模型定义。

本文件定义了所有智能体之间传递的标准化数据结构。
使用 Python dataclass 而非 Pydantic，保持轻量依赖。
所有字段都有清晰的中文注释，方便后续开发者理解。

数据流向:
    市场原始数据 -> MarketSnapshot
    MarketSnapshot -> MarketAnalysis (市场感知智能体输出)
    MarketAnalysis -> TradingSignal (策略分析智能体输出)
    TradingSignal + AccountState -> ExecutionOrder (风险控制智能体输出)
    ExecutionOrder -> OrderResult (订单执行智能体输出)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ============================================================
# 枚举类型
# ============================================================

class Side(str, Enum):
    """交易方向"""
    LONG = "long"    # 做多 (买入)
    SHORT = "short"  # 做空 (卖出)


class Action(str, Enum):
    """交易动作"""
    OPEN = "open"    # 开仓
    CLOSE = "close"  # 平仓
    HOLD = "hold"    # 观望，不操作


class Asset(str, Enum):
    """交易资产"""
    BTC = "BTC"
    ETH = "ETH"
    SOL = "SOL"
    XAU = "XAU"
    XAG = "XAG"

    @property
    def market_id(self) -> int:
        """
        返回该资产在 Lighter DEX 上的 market_id。
        ETH=0, BTC=1, SOL=2, XAU=92, XAG=93
        """
        return {"ETH": 0, "BTC": 1, "SOL": 2, "XAU": 92, "XAG": 93}[self.value]

    @property
    def size_decimals(self) -> int:
        """返回该资产的数量精度（小数位数）"""
        return {"ETH": 4, "BTC": 5, "SOL": 3, "XAU": 4, "XAG": 2}[self.value]

    @property
    def price_decimals(self) -> int:
        """返回该资产的价格精度（小数位数）"""
        return {"ETH": 2, "BTC": 1, "SOL": 3, "XAU": 2, "XAG": 4}[self.value]

    @property
    def min_base_amount(self) -> float:
        """返回该资产的最小下单量"""
        return {"ETH": 0.005, "BTC": 0.0002, "SOL": 0.05, "XAU": 0.003, "XAG": 0.15}[self.value]


class OrderType(str, Enum):
    """订单类型"""
    MARKET = "market"  # 市价单：以当前市场最优价格立即成交
    LIMIT = "limit"    # 限价单：以指定价格挂单等待成交


class SignalConfidence(str, Enum):
    """信号置信度等级"""
    HIGH = "high"      # 高置信度：强烈建议执行
    MEDIUM = "medium"  # 中置信度：可以执行但需谨慎
    LOW = "low"        # 低置信度：建议观望


# ============================================================
# 市场数据相关
# ============================================================

@dataclass
class OrderBookLevel:
    """
    订单簿的单个价格档位。
    
    订单簿由多个价格档位组成，每个档位包含一个价格和该价格上的挂单总量。
    例如：价格 69000.0，数量 1.5 表示在 69000 美元有 1.5 BTC 的买单/卖单。
    """
    price: float   # 价格 (USD)
    amount: float  # 该价格上的挂单总量


@dataclass
class MarketSnapshot:
    """
    市场快照 —— 某一时刻的完整市场状态。
    
    由市场感知智能体从 Lighter DEX 获取原始数据后组装而成，
    是所有后续分析的基础数据源。
    """
    asset: Asset                          # 资产类型 (BTC/ETH)
    timestamp: float                      # 快照时间戳 (Unix 秒)
    last_price: float                     # 最新成交价 (USD)
    best_bid: float                       # 买一价 (最高买价)
    best_ask: float                       # 卖一价 (最低卖价)
    bid_levels: list[OrderBookLevel]      # 买盘深度 (从高到低排列)
    ask_levels: list[OrderBookLevel]      # 卖盘深度 (从低到高排列)
    volume_24h: float                     # 24小时交易量 (USD)
    funding_rate: Optional[float] = None  # 当前资金费率 (可能暂时获取不到)
    price_change_24h: Optional[float] = None  # 24小时价格变化百分比

    @property
    def spread(self) -> float:
        """买卖价差 (USD)"""
        return self.best_ask - self.best_bid

    @property
    def spread_pct(self) -> float:
        """买卖价差百分比"""
        mid = (self.best_ask + self.best_bid) / 2
        return (self.spread / mid) * 100 if mid > 0 else 0.0

    @property
    def mid_price(self) -> float:
        """中间价"""
        return (self.best_ask + self.best_bid) / 2

    def to_llm_text(self) -> str:
        """
        将市场快照转化为 LLM 可读的文本格式。
        
        这个方法非常重要 —— 它决定了 LLM 能"看到"什么样的市场数据。
        格式设计原则：简洁、结构化、包含关键信息。
        """
        # 取前5档买卖盘
        bids_text = "\n".join(
            f"    价格: ${level.price:.2f}, 数量: {level.amount}"
            for level in self.bid_levels[:5]
        )
        asks_text = "\n".join(
            f"    价格: ${level.price:.2f}, 数量: {level.amount}"
            for level in self.ask_levels[:5]
        )

        return (
            f"=== {self.asset.value} 市场快照 ===\n"
            f"最新价格: ${self.last_price:,.2f}\n"
            f"买一价: ${self.best_bid:,.2f} | 卖一价: ${self.best_ask:,.2f}\n"
            f"价差: ${self.spread:.2f} ({self.spread_pct:.4f}%)\n"
            f"24h 交易量: ${self.volume_24h:,.2f}\n"
            f"24h 价格变化: {self.price_change_24h:.2f}%\n"
            f"资金费率: {self.funding_rate if self.funding_rate else '暂无'}\n"
            f"买盘深度 (前5档):\n{bids_text}\n"
            f"卖盘深度 (前5档):\n{asks_text}"
        )


# ============================================================
# 智能体输出数据结构
# ============================================================

@dataclass
class MarketAnalysis:
    """
    市场分析结果 —— 市场感知智能体的输出。
    
    由 LLM 对原始市场数据进行分析后生成，
    包含了对市场情绪、趋势和异常的判断。
    """
    asset: Asset                    # 分析的资产
    timestamp: float                # 分析时间戳
    sentiment: str                  # 市场情绪: "bullish"(看涨) / "bearish"(看跌) / "neutral"(中性)
    sentiment_score: float          # 情绪评分: -1.0(极度看跌) 到 1.0(极度看涨)
    short_term_trend: str           # 短期趋势描述 (如 "上升通道", "震荡整理")
    volatility_level: str           # 波动率水平: "high" / "medium" / "low"
    key_observations: list[str]     # 关键观察点列表 (如 "买盘深度远大于卖盘")
    support_price: Optional[float] = None   # 支撑位价格
    resistance_price: Optional[float] = None  # 阻力位价格
    raw_snapshot: Optional[MarketSnapshot] = None  # 原始快照数据 (供下游参考)


@dataclass
class TradingSignal:
    """
    交易信号 —— 策略分析智能体的输出。
    
    当策略智能体认为存在交易机会时生成此信号。
    注意：这只是一个"建议"，还需要经过风控智能体的审核。
    """
    signal_id: str                  # 信号唯一ID (用于追踪)
    asset: Asset                    # 目标资产
    action: Action                  # 交易动作: 开仓/平仓/观望
    side: Optional[Side] = None     # 交易方向: 做多/做空 (观望时为 None)
    confidence: SignalConfidence = SignalConfidence.LOW  # 置信度
    entry_price: Optional[float] = None   # 建议入场价格
    stop_loss: Optional[float] = None     # 建议止损价格
    take_profit: Optional[float] = None   # 建议止盈价格
    reasoning: str = ""                   # LLM 给出的决策理由
    timestamp: float = field(default_factory=time.time)

    @staticmethod
    def hold(asset: Asset, reasoning: str = "当前无明确交易机会") -> TradingSignal:
        """快捷方法：生成一个"观望"信号"""
        return TradingSignal(
            signal_id=str(uuid.uuid4())[:8],
            asset=asset,
            action=Action.HOLD,
            reasoning=reasoning,
        )


@dataclass
class AccountState:
    """
    账户状态 —— 从 Lighter DEX 获取的实时账户信息。
    
    风控智能体需要这些信息来计算仓位大小和评估风险。
    """
    total_equity: float                  # 总权益 (USD)
    available_balance: float             # 可用余额 (USD)
    unrealized_pnl: float                # 未实现盈亏 (USD)
    positions: dict[str, PositionInfo]   # 当前持仓 {资产名: 持仓信息}


@dataclass
class PositionInfo:
    """单个资产的持仓信息"""
    asset: Asset           # 资产类型
    side: Side             # 持仓方向
    size: float            # 持仓数量
    entry_price: float     # 平均入场价格
    unrealized_pnl: float  # 未实现盈亏
    leverage: float        # 杠杆倍数


@dataclass
class ExecutionOrder:
    """
    执行指令 —— 风险控制智能体的输出。
    
    这是经过风控审核后的最终交易指令，包含了具体的下单参数。
    订单执行智能体将直接根据此指令调用 Lighter DEX API。
    """
    order_id: str                   # 指令唯一ID
    approved: bool                  # 是否批准执行
    signal_id: str                  # 对应的交易信号ID (用于追踪)
    asset: Optional[Asset] = None   # 目标资产
    side: Optional[Side] = None     # 交易方向
    action: Optional[Action] = None # 交易动作
    order_type: OrderType = OrderType.MARKET  # 订单类型
    base_amount: Optional[float] = None  # 下单数量 (资产数量, 如 0.01 BTC)
    price: Optional[float] = None        # 限价单价格 / 市价单的最差可接受价格
    stop_loss: Optional[float] = None    # 止损价格
    take_profit: Optional[float] = None  # 止盈价格
    reject_reason: str = ""              # 拒绝理由 (approved=False 时填写)
    risk_notes: str = ""                 # 风控备注
    timestamp: float = field(default_factory=time.time)

    @staticmethod
    def reject(signal_id: str, reason: str) -> ExecutionOrder:
        """快捷方法：生成一个"拒绝"指令"""
        return ExecutionOrder(
            order_id=str(uuid.uuid4())[:8],
            approved=False,
            signal_id=signal_id,
            reject_reason=reason,
        )


@dataclass
class OrderResult:
    """
    订单结果 —— 订单执行智能体的输出。
    
    记录了订单在 Lighter DEX 上的最终执行情况。
    """
    order_id: str              # 对应的执行指令ID
    success: bool              # 是否执行成功
    tx_hash: Optional[str] = None       # 交易哈希 (成功时有值)
    filled_amount: Optional[float] = None  # 实际成交数量
    filled_price: Optional[float] = None   # 实际成交价格
    error_message: str = ""                # 错误信息 (失败时有值)
    timestamp: float = field(default_factory=time.time)
