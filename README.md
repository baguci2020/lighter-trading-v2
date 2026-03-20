# lighter-trading-v2

> 可视化 AI 策略平台 —— 支持多 Agent、多时间框架、策略实验室

## 项目简介

本项目是 [lighter-trading-system](https://github.com/baguci2020/lighter-trading-system) 的全新重写版本，基于可视化策略实验室理念设计，用户可直接在前端创建、配置、测试和发布 AI 交易 Agent，无需修改任何代码。

## 核心特性

- **策略实验室**：可视化编辑 Prompt、参数、K 线周期和技术指标
- **沙盒测试**：实时查看 AI 的完整思考过程和决策输出
- **多 Agent 管理**：每个 Agent 绑定独立账户，完全隔离
- **多时间框架**：支持 1m / 5m / 15m / 1h / 4h 独立策略
- **AI 决策看板**：透明化展示每次分析的推理过程

## 目录结构

```
lighter-trading-v2/
├── legacy/                 # 从 v1 保留的可复用代码
│   ├── exchange/           # Lighter DEX API 客户端（已调试）
│   ├── models/             # 数据模型（Asset/Side/PositionInfo 等）
│   └── utils/              # LLM 客户端、日志、滑点计算
├── docs/                   # 设计文档
│   └── upgrade_plan_v2.md  # 升级计划书
├── backend/                # 新后端（待开发）
├── frontend/               # 新前端（待开发）
└── docker/                 # 容器配置（待开发）
```

## 开发计划

详见 [docs/upgrade_plan_v2.md](docs/upgrade_plan_v2.md)

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | Python 3.11 + FastAPI + APScheduler |
| 前端 | React + TypeScript + TailwindCSS |
| 数据库 | MySQL 8.0 |
| 数据源 | Binance API（K 线）+ Lighter DEX（交易执行）|
| 代理 | Clash（Mihomo）|
