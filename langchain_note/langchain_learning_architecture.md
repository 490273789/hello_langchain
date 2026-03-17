# LangChain 学习架构图说明

这张图聚焦于官方文档中的“模块化架构”思想，目标是帮助你快速建立全局心智模型。

## 图中模块含义

- Agent Runtime：执行中枢，负责状态管理、决策和编排。
- Prompts + Messages：组织系统/用户/工具消息，形成可执行输入。
- Model Providers：实际推理能力来源（不同模型供应商可替换）。
- Tools：外部动作能力（API、函数、数据库、搜索等）。
- Retrieval Pipeline：典型 RAG 流程（Loader -> Splitter -> Embeddings -> Vector Store -> Retriever -> Context）。
- Memory / Checkpointer：会话状态和历史持久化。
- Middleware：在运行时插入策略（例如 HITL、总结、护栏）。
- LangSmith：可观测性与评测（Tracing / Eval）。

## 推荐学习路径

1. 先学 Prompts + Messages
2. 再学模型调用和输出结构
3. 再学 LCEL/Chains 组合
4. 然后学 Tools + Agent ReAct 回路
5. 再接 Retrieval + Memory
6. 最后学 Middleware 与 LangSmith 观测优化

## 文档依据（官方概念）

- LangChain modular architecture（LLMs / Prompts / Chains / Indexes / Memory / Agents）
- LangChain Agents 与 Tools 概念
- create_agent 与 middleware 用法
- LangSmith observability/tracing
