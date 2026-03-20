"""
utils/slippage.py
=================
动态滑点计算工具。

优化内容 (Phase 2):
    原系统使用固定滑点（如 0.1%），在流动性差时会导致订单无法成交
    或以远差于预期的价格成交。

    本模块通过遍历订单簿深度，精确计算特定下单量所需的
    加权平均成交价格（VWAP），从而动态估算实际滑点。

    动态滑点策略:
        1. 从订单簿中按价格顺序累积成交量，直到满足目标数量
        2. 计算加权平均成交价格（VWAP）
        3. 与 best bid/ask 比较，得出实际滑点百分比
        4. 在此基础上增加安全缓冲（默认 0.05%），作为最终限价单价格

使用示例:
    from utils.slippage import calculate_dynamic_slippage, apply_slippage

    # 计算买入 1 BTC 的预期滑点
    slippage_pct = calculate_dynamic_slippage(
        order_size=1.0,
        side="buy",
        asks=snapshot.ask_levels,
    )

    # 计算实际下单价格（在 VWAP 基础上加安全缓冲）
    limit_price = apply_slippage(
        mid_price=50000.0,
        side="buy",
        slippage_pct=slippage_pct,
    )
"""
from __future__ import annotations

from typing import Optional

from models.schemas import OrderBookLevel, Side
from utils.logger import get_logger

logger = get_logger("slippage")

# 最小/最大滑点限制（防止极端值）
MIN_SLIPPAGE_PCT = 0.0005   # 0.05%（最小滑点，即使流动性极好）
MAX_SLIPPAGE_PCT = 0.01     # 1.0%（最大滑点，超过此值拒绝下单）
SAFETY_BUFFER_PCT = 0.0005  # 0.05% 安全缓冲（在 VWAP 基础上额外加）


def calculate_dynamic_slippage(
    order_size: float,
    side: Side,
    bids: Optional[list[OrderBookLevel]] = None,
    asks: Optional[list[OrderBookLevel]] = None,
    fallback_pct: float = 0.002,
) -> float:
    """
    根据订单簿深度动态计算预期滑点。

    通过遍历订单簿，模拟市价单的成交过程，
    计算加权平均成交价格（VWAP）与最优价格的偏差。

    Args:
        order_size: 目标下单数量（基础资产，如 BTC 数量）
        side: 交易方向（买入用 asks，卖出用 bids）
        bids: 买盘列表（按价格从高到低排列）
        asks: 卖盘列表（按价格从低到高排列）
        fallback_pct: 无法获取订单簿时的默认滑点，默认 0.2%

    Returns:
        预期滑点百分比（0.002 表示 0.2%）
    """
    # 根据交易方向选择对应的订单簿一侧
    if side == Side.LONG:
        levels = asks  # 买入消耗卖盘
        best_price = asks[0].price if asks else None
    else:
        levels = bids  # 卖出消耗买盘
        best_price = bids[0].price if bids else None

    if not levels or not best_price or best_price <= 0:
        logger.warning(
            f"[滑点] 无法获取 {side.value} 方向的订单簿数据，"
            f"使用默认滑点 {fallback_pct*100:.2f}%"
        )
        return fallback_pct

    # 遍历订单簿，计算加权平均成交价格（VWAP）
    remaining = order_size
    total_cost = 0.0
    total_filled = 0.0

    for level in levels:
        if remaining <= 0:
            break

        fill_amount = min(remaining, level.amount)
        total_cost += fill_amount * level.price
        total_filled += fill_amount
        remaining -= fill_amount

    if total_filled <= 0:
        return fallback_pct

    # 如果订单簿深度不足，说明流动性极差
    if remaining > 0:
        unfilled_pct = remaining / order_size * 100
        logger.warning(
            f"[滑点] 订单簿深度不足！目标: {order_size:.4f}，"
            f"可成交: {total_filled:.4f}，"
            f"未成交: {remaining:.4f} ({unfilled_pct:.1f}%)"
        )

    # 计算 VWAP
    vwap = total_cost / total_filled

    # 计算滑点（VWAP 与最优价格的偏差）
    if side == Side.LONG:
        slippage = (vwap - best_price) / best_price
    else:
        slippage = (best_price - vwap) / best_price

    # 加上安全缓冲
    total_slippage = max(slippage + SAFETY_BUFFER_PCT, MIN_SLIPPAGE_PCT)
    total_slippage = min(total_slippage, MAX_SLIPPAGE_PCT)

    logger.info(
        f"[滑点] {side.value} {order_size:.4f} | "
        f"最优价: ${best_price:,.4f} | VWAP: ${vwap:,.4f} | "
        f"市场滑点: {slippage*100:.3f}% | "
        f"含缓冲: {total_slippage*100:.3f}%"
    )

    return total_slippage


def apply_slippage(
    mid_price: float,
    side: Side,
    slippage_pct: float,
) -> float:
    """
    将滑点应用到中间价，得到实际限价单价格。

    买入时：限价 = mid_price * (1 + slippage_pct)（愿意支付更高价格）
    卖出时：限价 = mid_price * (1 - slippage_pct)（接受更低价格）

    Args:
        mid_price: 当前中间价（通常为 (best_bid + best_ask) / 2）
        side: 交易方向
        slippage_pct: 滑点百分比（由 calculate_dynamic_slippage 返回）

    Returns:
        考虑滑点后的限价单价格
    """
    if side == Side.LONG:
        price = mid_price * (1 + slippage_pct)
    else:
        price = mid_price * (1 - slippage_pct)

    logger.debug(
        f"[滑点] 价格计算 | 中间价: ${mid_price:,.4f} | "
        f"方向: {side.value} | 滑点: {slippage_pct*100:.3f}% | "
        f"限价: ${price:,.4f}"
    )
    return price


def check_slippage_acceptable(
    slippage_pct: float,
    max_allowed: float = MAX_SLIPPAGE_PCT,
) -> bool:
    """
    检查滑点是否在可接受范围内。

    如果预期滑点超过最大允许值，风控模块应拒绝该订单。

    Args:
        slippage_pct: 预期滑点百分比
        max_allowed: 最大允许滑点，默认 1.0%

    Returns:
        True 表示滑点可接受，False 表示滑点过大应拒绝
    """
    if slippage_pct > max_allowed:
        logger.warning(
            f"[滑点] ⚠️ 滑点过大！预期: {slippage_pct*100:.3f}% | "
            f"最大允许: {max_allowed*100:.3f}%，建议拒绝下单"
        )
        return False
    return True
