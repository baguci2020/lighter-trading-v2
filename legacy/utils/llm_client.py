"""
utils/llm_client.py
===================
通用 LLM 客户端。

本文件封装了与第三方中转站 API 的通信逻辑。
使用 OpenAI 兼容格式 (openai Python 库)，支持：
- 异步调用 (async/await)
- 强制 JSON 输出 (通过 system prompt 约束)
- 自动重试 (遇到网络错误或格式错误时)
- 详细的日志输出 (方便调试 LLM 的输入输出)

使用方法:
    from utils.llm_client import LLMClient
    from config import LLMConfig
    
    client = LLMClient(config=my_llm_config, agent_name="策略分析")
    result = await client.ask_json(
        system_prompt="你是一个交易员...",
        user_message="请分析以下数据...",
    )
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any, Optional

from openai import AsyncOpenAI

from config import LLMConfig
from utils.logger import get_logger


class LLMClient:
    """
    通用 LLM 客户端 —— 每个智能体持有一个实例。
    
    核心功能:
    1. 将 system prompt + 用户消息发送给 LLM
    2. 解析 LLM 返回的 JSON 结果
    3. 如果 LLM 返回了非法 JSON，自动重试 (最多 3 次)
    
    Attributes:
        config: LLM 配置 (base_url, api_key, model 等)
        agent_name: 所属智能体的名称 (用于日志标识)
        client: OpenAI 异步客户端实例
    """

    def __init__(self, config: LLMConfig, agent_name: str = "Agent"):
        self.config = config
        self.agent_name = agent_name

        # 初始化 OpenAI 兼容客户端
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )

        # 降级模型（可选）：主模型失败时使用低成本模型
        self._fallback_model: Optional[str] = getattr(config, "fallback_model", None)
        # 连续失败计数（用于触发模型降级）
        self._consecutive_failures: int = 0
        self._failure_threshold: int = 2  # 连续失败 2 次后切换到降级模型

        _logger = get_logger(agent_name)
        _logger.info(
            f"[{self.agent_name}] LLM 客户端已初始化 | "
            f"主模型: {config.model} | "
            f"降级模型: {self._fallback_model or '未配置'} | "
            f"端点: {config.base_url}"
        )

    def _select_model(self) -> str:
        """选择当前应使用的模型（根据失败次数决定是否降级）。"""
        if (
            self._fallback_model
            and self._consecutive_failures >= self._failure_threshold
        ):
            return self._fallback_model
        return self.config.model

    async def ask_json(
        self,
        system_prompt: str,
        user_message: str,
        max_retries: int = 4,
        base_delay: float = 1.0,
    ) -> Optional[dict[str, Any]]:
        """
        向 LLM 发送请求并解析 JSON 响应。

        这是智能体调用 LLM 的核心方法。它会：
        1. 在 system prompt 中强制要求 LLM 输出纯 JSON
        2. 发送请求并等待响应
        3. 尝试解析 JSON，如果失败则按指数退避策略重试
        4. 主模型连续失败时自动降级到低成本模型

        指数退避策略:
            第 1 次失败：等待 1s + 随机抖动
            第 2 次失败：等待 2s + 随机抖动
            第 3 次失败：等待 4s + 随机抖动
            第 4 次失败：等待 8s + 随机抖动（最大 30s）

        Args:
            system_prompt: 系统提示词 (定义 LLM 的角色和输出格式)
            user_message: 用户消息 (包含需要分析的数据)
            max_retries: 最大重试次数 (默认 4 次)
            base_delay: 退避基础延迟（秒），默认 1.0s

        Returns:
            解析后的 JSON 字典，如果所有重试都失败则返回 None
        """
        _logger = get_logger(self.agent_name)

        # 在 system prompt 末尾追加 JSON 格式约束
        full_system_prompt = (
            f"{system_prompt}\n\n"
            "【输出格式要求】\n"
            "你必须且只能输出一个合法的 JSON 对象，不要输出任何其他文字、"
            "解释或 markdown 代码块标记。直接输出 JSON 即可。"
        )

        model = self._select_model()

        for attempt in range(1, max_retries + 1):
            try:
                _logger.debug(
                    f"[{self.agent_name}] 第 {attempt}/{max_retries} 次调用 LLM "
                    f"(模型: {model})..."
                )
                call_start = time.time()

                # 调用 LLM API
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": full_system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )

                call_elapsed = time.time() - call_start
                raw_text = response.choices[0].message.content.strip()

                _logger.debug(
                    f"[{self.agent_name}] LLM 原始输出 ({call_elapsed:.1f}s):\n"
                    f"{raw_text[:500]}"
                )

                # 尝试解析 JSON
                result = self._extract_json(raw_text)
                if result is not None:
                    _logger.info(
                        f"[{self.agent_name}] LLM 调用成功 "
                        f"(第 {attempt} 次, {call_elapsed:.1f}s, 模型: {model})"
                    )
                    self._consecutive_failures = 0
                    return result

                _logger.warning(
                    f"[{self.agent_name}] LLM 输出不是合法 JSON (第 {attempt} 次), "
                    f"原始输出: {raw_text[:200]}..."
                )

            except Exception as e:
                _logger.error(
                    f"[{self.agent_name}] LLM 调用异常 (第 {attempt} 次): {e}"
                )

            # 指数退避等待（最后一次不等待）
            if attempt < max_retries:
                delay = min(base_delay * (2 ** (attempt - 1)), 30.0)
                jitter = delay * 0.2 * (2 * random.random() - 1)
                wait_time = max(0.1, delay + jitter)
                _logger.info(
                    f"[{self.agent_name}] 第 {attempt} 次失败，"
                    f"{wait_time:.1f}s 后重试..."
                )
                await asyncio.sleep(wait_time)

                # 检查是否需要切换到降级模型
                self._consecutive_failures += 1
                if (
                    self._fallback_model
                    and self._consecutive_failures >= self._failure_threshold
                    and model != self._fallback_model
                ):
                    _logger.warning(
                        f"[{self.agent_name}] 主模型 {model} 连续失败 "
                        f"{self._consecutive_failures} 次，"
                        f"切换到降级模型: {self._fallback_model}"
                    )
                    model = self._fallback_model

        self._consecutive_failures += 1
        _logger.error(
            f"[{self.agent_name}] LLM 调用失败，已达最大重试次数 {max_retries}"
        )
        return None

    async def ask_text(
        self,
        system_prompt: str,
        user_message: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> Optional[str]:
        """
        向 LLM 发送请求并返回纯文本响应（不要求 JSON 格式）。

        同样支持指数退避重试机制。

        Args:
            system_prompt: 系统提示词
            user_message: 用户消息
            max_retries: 最大重试次数
            base_delay: 退避基础延迟（秒）

        Returns:
            LLM 的文本回复，失败时返回 None
        """
        _logger = get_logger(self.agent_name)
        model = self._select_model()

        for attempt in range(1, max_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                self._consecutive_failures = 0
                return response.choices[0].message.content.strip()
            except Exception as e:
                _logger.error(f"[{self.agent_name}] LLM 文本调用异常 (第 {attempt} 次): {e}")
                if attempt < max_retries:
                    delay = min(base_delay * (2 ** (attempt - 1)), 30.0)
                    await asyncio.sleep(delay)
                    self._consecutive_failures += 1
        return None

    @staticmethod
    def _extract_json(text: str) -> Optional[dict[str, Any]]:
        """
        从 LLM 的输出文本中提取 JSON 对象。
        
        LLM 有时会在 JSON 前后添加额外的文字或 markdown 标记，
        这个方法会尝试多种方式来提取有效的 JSON：
        1. 直接解析整个文本
        2. 去除 markdown 代码块标记后解析
        3. 查找第一个 { 和最后一个 } 之间的内容
        
        Args:
            text: LLM 的原始输出文本
            
        Returns:
            解析后的字典，解析失败返回 None
        """
        # 方法1: 直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 方法2: 去除 markdown 代码块标记 (```json ... ```)
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # 去掉第一行 (```json) 和最后一行 (```)
            lines = cleaned.split("\n")
            lines = lines[1:]  # 去掉 ```json
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]  # 去掉 ```
            cleaned = "\n".join(lines).strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

        # 方法3: 查找 { ... } 范围
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        return None
