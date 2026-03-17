# LangChain 1.x 知识体系总览

更新时间：2026-03-15  
适用范围：LangChain 1.x（以 Python 官方文档为主，补充 LangGraph / LangSmith / Deep Agents 的关系）

---

## 1. 先给一个总判断

LangChain 1.x 不再适合按“零散功能点”去记忆，而更适合按下面 5 层来理解：

1. **表达层**：模型输入输出如何被统一表达  
   典型概念：`messages`、`content blocks`、`prompts`、`structured output`
2. **能力层**：模型和外部能力如何接入  
   典型概念：`models`、`tools`、`embeddings`、`document loaders`、`vector stores`
3. **编排层**：智能体如何做决策与执行  
   典型概念：`agents`、`middleware`、`runtime`、`tool calling`
4. **知识层**：外部知识如何进入推理过程  
   典型概念：`documents`、`text splitters`、`retrievers`、`RAG`
5. **生产层**：系统如何可观测、可持续、可控  
   典型概念：`memory`、`streaming`、`persistence`、`human-in-the-loop`、`observability`

如果只背一个公式，可以背这个：

```text
用户输入
-> Prompt / Messages
-> Model
-> Agent decision
-> Tools / Retrieval / Memory
-> Structured Output / Streaming
-> Observability / Persistence
```

---

## 2. LangChain 1.x 在整个 LangChain 生态里的位置

### 2.1 生态分层

LangChain 官方现在更强调下面几个产品之间的职责边界：

1. **LangChain**
   面向应用开发者的高层框架，重点是快速构建 `agent`。
2. **LangGraph**
   更底层的编排与运行时框架，负责状态、节点、边、持久化、人工介入、耐久执行。
3. **LangSmith**
   调试、追踪、评估、监控平台。
4. **Deep Agents**
   构建在 LangChain agent 之上的更完整 agent 方案，带有更强的上下文管理和工作流能力。

### 2.2 关系理解

可以把它理解成：

```text
LangSmith     -> 观测与评估
LangGraph     -> 底层运行时与编排
LangChain     -> 面向 agent 的高层抽象
Deep Agents   -> 更完整的上层 agent 产品形态
```

其中，**LangChain 1.x 的 `create_agent` 底层运行在 LangGraph runtime 上**。所以你在 LangChain 中看到的：

- streaming
- persistence
- human-in-the-loop
- state
- runtime context

本质上都和 LangGraph 有直接关系。

---

## 3. LangChain 1.x 的完整知识体系分类

下面给一个适合学习和做项目时使用的“完整分类树”。

```text
LangChain 1.x
├── A. 核心定位与包结构
│   ├── langchain
│   ├── langchain-core
│   ├── provider integrations
│   ├── langchain-classic
│   └── LangGraph / LangSmith / Deep Agents
├── B. 模型表达层
│   ├── messages
│   ├── content blocks
│   ├── prompts
│   └── structured output
├── C. 模型能力层
│   ├── chat models
│   ├── embeddings
│   ├── multimodality
│   ├── reasoning
│   └── model profiles
├── D. 动作执行层
│   ├── tools
│   ├── tool calling
│   ├── toolkits
│   └── server-side tools
├── E. Agent 编排层
│   ├── agents
│   ├── middleware
│   ├── runtime
│   ├── context engineering
│   └── state schema
├── F. 知识检索层
│   ├── documents
│   ├── document loaders
│   ├── text splitters
│   ├── vector stores
│   ├── retrievers
│   └── RAG / agentic RAG
├── G. 记忆与状态层
│   ├── short-term memory
│   ├── long-term memory
│   ├── checkpoints
│   ├── store
│   └── persistence
├── H. 输出与交互层
│   ├── streaming
│   ├── tool messages
│   ├── response metadata
│   └── structured responses
├── I. 生产工程层
│   ├── observability
│   ├── evaluation
│   ├── guardrails / moderation
│   ├── retries / fallback
│   └── human-in-the-loop
└── J. 集成生态层
    ├── model providers
    ├── embedding providers
    ├── vector DB providers
    ├── loaders / retrievers
    └── MCP / external systems
```

---

## 4. 你提到的几个核心概念，应该怎么分类

这是最关键的一部分。

| 概念 | 在知识体系中的位置 | 它解决的问题 | 典型角色 |
|---|---|---|---|
| `agent` | Agent 编排层 | 决定何时调用模型、何时调用工具、何时结束 | 大脑 + 调度器 |
| `middleware` | Agent 编排层 / 生产工程层 | 在模型调用、工具调用前后注入控制逻辑 | 横切控制层 |
| `models` | 模型能力层 | 提供推理、生成、工具调用、结构化输出能力 | 推理引擎 |
| `messages` | 模型表达层 | 标准化对话上下文的表达方式 | 上下文载体 |
| `tools` | 动作执行层 | 让模型调用外部能力 | 外部执行器 |
| `embeddings` | 模型能力层 / 知识检索层 | 把文本映射为向量供检索使用 | 语义编码器 |
| `prompt` | 模型表达层 | 组织模型输入上下文与指令 | 输入模板 / 指令层 |

一句话概括它们的关系：

```text
Prompt 组织 Messages
Messages 送给 Model
Model 在 Agent 中做决策
Agent 通过 Tools 行动
Embeddings 支持 Retrieval / RAG
Middleware 负责在整个调用链中施加控制
```

---

## 5. A 层：核心定位与包结构

### 5.1 `langchain`

`langchain` 是 1.x 的高层包，重点放在 **agent 构建** 上。官方强调 1.x 的目标是“更聚焦、面向生产、以 agent 为中心”。

重点变化：

1. `create_agent` 成为标准入口。
2. `content_blocks` 成为跨 provider 的统一内容表达。
3. 命名空间被精简，旧能力迁移到 `langchain-classic`。

### 5.2 `langchain-core`

这是底层抽象层，定义标准接口和基类。很多集成实现最终都遵循这里的抽象。

常见基础接口包括：

- `BaseChatModel`
- `Embeddings`
- `BaseTool`
- `BaseRetriever`
- `VectorStore`

### 5.3 provider integrations

LangChain 官方现在大力推动“provider 独立包”。

例如：

- `langchain-openai`
- `langchain-anthropic`
- `langchain-google-genai`
- 各类 vector store / loader / retriever 对应 provider 包

这样做的原因：

1. 版本管理更清晰。
2. 依赖更轻。
3. 测试与维护边界更明确。

### 5.4 `langchain-classic`

这是旧式 API 的安置区。你如果在旧教程里看到大量 `chains`、老式 agent executor、旧 memory 方案，很可能属于 classic 时代。

学习建议：

1. 新项目优先学 1.x 的 `create_agent`、`middleware`、`runtime`。
2. 遇到旧教程时，把它识别为 classic 风格，不要和 1.x 混着记。

---

## 6. B 层：模型表达层

这一层回答一个问题：**模型的输入输出在 LangChain 里如何被标准化表达？**

### 6.1 Messages

`messages` 是 LangChain 中最基础的上下文单位。每条消息通常包含：

1. `role`
2. `content`
3. `metadata`

消息不是简单字符串，而是对“对话状态”的统一抽象。

常见消息类型：

- `SystemMessage`
- `HumanMessage`
- `AIMessage`
- `ToolMessage`

#### 作用

1. 统一不同模型供应商的消息格式。
2. 支撑多轮对话。
3. 支撑工具调用结果回注。
4. 支撑多模态内容。

### 6.2 Content Blocks

这是 1.x 的重点升级。`content_blocks` 用来统一表达更现代的模型内容能力，例如：

- 文本
- 图像
- 音频
- reasoning 内容
- citations
- server-side tool use

理解要点：

1. `message.content` 更偏原始载体。
2. `message.content_blocks` 更偏统一、类型化、跨 provider 的访问方式。

### 6.3 Prompts

`prompt` 是组织输入给模型的模板系统。它不是模型，也不是消息本身，而是**生成模型上下文的方式**。

职责：

1. 注入系统指令。
2. 拼装变量。
3. 组合上下文。
4. 形成最终 messages 或字符串输入。

你可以把 prompt 理解为：

```text
Prompt = 构造输入上下文的方法
Messages = 构造后的标准上下文对象
```

### 6.4 Structured Output

结构化输出是 1.x 的核心能力之一，用来约束模型返回 JSON / Pydantic schema / typed object。

它的重要性非常高，因为它直接决定了：

1. agent 输出是否可程序消费
2. tool routing 是否稳定
3. 业务字段抽取是否可靠

---

## 7. C 层：模型能力层

这一层回答：**真正负责“理解和生成”的能力是什么？**

### 7.1 Chat Models

1.x 里最核心的是 `chat model` 抽象。LangChain 倾向把现代 LLM 统一到聊天模型接口上，而不是老式纯文本 completion 接口。

模型能力通常包括：

1. 文本生成
2. 工具调用
3. 结构化输出
4. 多模态处理
5. reasoning

### 7.2 Embeddings

`embeddings` 负责把文本、文档等内容编码成向量，主要服务于：

1. 语义搜索
2. 检索增强生成（RAG）
3. 相似度匹配
4. 聚类与去重

关键理解：

- `chat model` 负责“推理与生成”
- `embedding model` 负责“语义表示”

它们都属于模型层，但用途完全不同。

### 7.3 Multimodality

现代模型不再只处理文本，还可能输入或输出：

- 图片
- 音频
- 文档
- 视频相关内容

LangChain 1.x 通过 message content 和 content blocks 尝试统一这类差异。

### 7.4 Reasoning

部分模型支持更强的推理能力。LangChain 1.x 关注的是如何用统一接口暴露这些能力，而不是只绑定单一 provider。

### 7.5 Model Profiles

较新的文档里提到 `model.profile`，用于暴露模型支持的能力与特征。这个方向的意义是：

1. 能根据模型能力做动态策略。
2. middleware 可以根据模型 profile 调整行为。
3. structured output、summarization 等能力可以更自动化地适配。

---

## 8. D 层：动作执行层

这一层回答：**模型如何作用于外部世界？**

### 8.1 Tools

`tools` 是 LangChain agent 的外部能力接口。它本质上让模型具备“调用函数 / API / 数据库 / 检索器”的能力。

工具的两大职责：

1. 向模型声明“可用操作”及其参数 schema
2. 真正执行外部动作，并把结果回传给模型

### 8.2 Tool Calling

这是现代 agent 的关键机制。

基本流程：

```text
用户问题
-> 模型判断需要调用哪个工具
-> 生成 tool call 参数
-> LangChain 执行工具
-> 返回 ToolMessage
-> 模型继续推理或给出最终答案
```

### 8.3 Toolkits

`toolkit` 可以理解成一组围绕某领域组织好的工具集合，例如数据库工具集、搜索工具集等。

### 8.4 Server-side Tool Use

部分 provider 支持服务端工具循环，也就是模型在 provider 端直接完成某些工具使用。LangChain 1.x 会把这类结果统一映射到 content blocks 里。

这意味着：

1. 不是所有 tool use 都发生在你的本地 agent loop 中。
2. 有些 tool result 会直接作为模型响应的一部分返回。

---

## 9. E 层：Agent 编排层

这是 LangChain 1.x 的中心层。

### 9.1 Agents

`agent` 是对“模型 + 工具 + 状态 + 控制逻辑”的高层封装。它的核心不是单次调用模型，而是一个**面向任务完成的决策循环**。

agent 主要负责：

1. 接收用户输入
2. 维护消息状态
3. 调用模型进行决策
4. 判断是否调用工具
5. 执行工具
6. 汇总结果并结束

### 9.2 Middleware

`middleware` 是 1.x 很关键的新增主线。它是 agent 的横切控制机制，允许你在模型调用或工具调用前后插入逻辑。

middleware 常见用途：

1. 动态 system prompt
2. 模型路由
3. 重试
4. 内容安全检查
5. 摘要压缩
6. 审计与日志
7. 人工审批

如果你从 Web 框架角度理解，middleware 很像请求处理链；如果你从 AOP 角度理解，它就是 agent 的切面控制层。

### 9.3 Runtime

`runtime` 是 LangChain 1.x 背后的运行时概念，底层来自 LangGraph。

官方文档强调 runtime 中常见的几个东西：

1. `Context`
2. `Store`
3. `Stream writer`

#### Runtime 的核心价值

1. 给 tools 和 middleware 注入依赖。
2. 避免全局变量。
3. 支持测试和复用。
4. 为 memory / streaming / persistence 提供基础设施。

### 9.4 Context Engineering

LangChain 现在更强调“上下文工程”而不是“只写 prompt”。

上下文工程包括：

1. 系统提示设计
2. 消息历史管理
3. 外部知识注入
4. 状态到 prompt 的映射
5. 工具结果如何回填到上下文

### 9.5 State Schema

在 agent / LangGraph 风格下，`state` 不只是聊天历史，它是运行过程中可被读写的动态数据结构。

例如：

- 用户名
- 当前任务阶段
- 临时检索结果
- 中间决策产物

---

## 10. F 层：知识检索层

这是做 RAG 时最重要的一层。

### 10.1 Documents

`Document` 是知识处理中的基本对象，一般包含：

1. `page_content`
2. `metadata`

### 10.2 Document Loaders

负责从外部源加载原始内容，例如：

- PDF
- 网页
- Markdown
- 数据库
- Notion / Confluence / 本地文件

### 10.3 Text Splitters

负责把长文档切成适合 embedding 和检索的 chunk。

它的重要性不低于向量库本身，因为 chunk 策略会直接影响：

1. 检索命中率
2. 上下文完整性
3. RAG 的最终回答质量

### 10.4 Vector Stores

向量库负责：

1. 存储 embedding 后的向量
2. 索引文档
3. 基于相似度执行搜索

注意：

- vector store 不是 embedding model
- vector store 也不是 retriever

它们关系是：

```text
Document
-> Splitter
-> Embeddings
-> VectorStore
-> Retriever
-> Model / Agent
```

### 10.5 Retrievers

`retriever` 是“取回相关文档”的抽象接口。它可能基于：

- 向量搜索
- 关键词搜索
- 混合检索
- API 检索

LangChain 中，retriever 比 vector store 更靠近“应用使用层”。

### 10.6 RAG

RAG 是把检索结果作为上下文注入模型进行推理的模式。

### 10.7 Agentic RAG

Agentic RAG 比普通 RAG 更进一步：**让 agent 自主判断是否需要检索、检索几次、是否改写 query、如何使用结果**。

所以：

- 普通 RAG 更像固定管道
- Agentic RAG 更像带决策的检索工作流

---

## 11. G 层：记忆与状态层

这一层最容易和旧版 LangChain 的 memory 概念混淆。

### 11.1 Short-term Memory

短期记忆通常指一次运行或一段会话中的状态，常见表现：

- 对话历史
- 当前任务状态
- 工具执行中间结果

在新体系里，它和 `state`、LangGraph runtime 的关系非常紧密。

### 11.2 Long-term Memory

长期记忆通常需要借助 store 或外部存储，把跨会话的信息保存下来，例如：

- 用户画像
- 历史偏好
- 之前完成的任务摘要

### 11.3 Checkpoints

checkpoint 是运行过程中的持久化快照，用于：

1. 断点恢复
2. 长任务持续执行
3. 审计与回放

### 11.4 Store

runtime / graph 中的 `store` 更偏长期存储抽象，不等于聊天历史本身。

### 11.5 Persistence

持久化是“系统能力”，不是单一模块。它可能包含：

- state 持久化
- 运行快照
- 长期 memory
- 可恢复执行

---

## 12. H 层：输出与交互层

### 12.1 Streaming

流式输出是 1.x 的一等能力，既包括：

1. 模型 token 流
2. agent 执行事件流
3. 自定义 stream writer 输出

### 12.2 Tool Messages

工具执行结果通常以 `ToolMessage` 的形式回到消息流中，继续参与后续推理。

### 12.3 Response Metadata

LangChain 统一保留模型响应里的元数据，例如：

- token 使用量
- 响应 id
- provider 原始附加信息

### 12.4 Structured Responses

面向程序消费时，最终输出应该优先考虑结构化，而不是仅自然语言文本。

---

## 13. I 层：生产工程层

### 13.1 Observability

官方推荐使用 LangSmith 进行：

1. tracing
2. 调试 tool 调用链
3. 分析 prompt 与模型行为
4. 监控生产流量

### 13.2 Evaluation

评估是把 agent 从“能跑”推进到“可靠”的关键步骤，包括：

- 正确性评估
- 工具选择评估
- RAG 命中评估
- 回归测试

### 13.3 Guardrails / Moderation

1.x 的 middleware 体系已经明显把安全控制纳入一等公民，例如：

- 输入审查
- 输出审查
- 工具结果审查

### 13.4 Retries / Fallback

生产环境里模型和工具都会失败，所以需要：

- 重试
- fallback 模型
- 超时与错误处理

### 13.5 Human-in-the-loop

人工审批、人工恢复、人工确认等机制，更多依托 LangGraph / runtime 能力实现。

---

## 14. J 层：集成生态层

LangChain 不是一个“只做 prompt 的库”，它更像是一个统一接口生态。

### 14.1 集成大类

官方文档明确把 provider 集成分成很多组件类型，包括但不限于：

1. chat models
2. embedding models
3. tools and toolkits
4. document loaders
5. vector stores
6. middleware
7. checkpointers
8. sandboxes

### 14.2 这层的本质

LangChain 的真正价值之一，是把第三方能力挂接到统一抽象之下，让你可以替换 provider 而不重写整个应用。

---

## 15. 这些概念之间的依赖关系

### 15.1 Agent 主链路

```text
User Input
-> Prompt
-> Messages
-> Model
-> Agent loop
-> Tool calling
-> ToolMessage
-> Model
-> Final Response
```

### 15.2 RAG 主链路

```text
Raw Data
-> Document Loader
-> Documents
-> Text Splitter
-> Embeddings
-> Vector Store
-> Retriever
-> Prompt / Messages
-> Model
```

### 15.3 Runtime / Memory 主链路

```text
Invocation
-> Runtime Context
-> State / Short-term Memory
-> Middleware / Tools read-write state
-> Store / Checkpoint / Long-term Memory
-> Streaming / Observability
```

---

## 16. 你学习 LangChain 1.x 时最容易混淆的边界

### 16.1 `prompt` 和 `message` 不是一回事

- `prompt` 是构造输入的方法
- `message` 是输入上下文的标准对象

### 16.2 `model` 和 `agent` 不是一回事

- `model` 负责生成与推理
- `agent` 负责多步决策、工具使用和任务完成

### 16.3 `tool` 和 `retriever` 不是一回事

- `retriever` 是“取文档”的抽象
- `tool` 是“可被 agent 调用的外部动作”抽象
- retriever 可以被包装成 tool

### 16.4 `embedding` 和 `chat model` 不是一回事

- `embedding` 做向量表示
- `chat model` 做推理生成

### 16.5 `memory` 和 `history` 不是一回事

- `history` 更偏消息历史
- `memory` 在 1.x / LangGraph 语境里更偏状态与持久化体系

### 16.6 `middleware` 不是业务工具

- `tool` 解决“去做什么”
- `middleware` 解决“调用链如何被控制”

---

## 17. 按学习顺序组织的一条推荐路线

如果你要构建完整知识体系，建议按下面顺序学习。

### 第一阶段：输入输出表达

1. `messages`
2. `prompts`
3. `models`
4. `structured output`

### 第二阶段：外部能力接入

1. `tools`
2. `tool calling`
3. `embeddings`
4. `documents / loaders / splitters / vector stores / retrievers`

### 第三阶段：智能体系统

1. `agents`
2. `middleware`
3. `runtime`
4. `state`

### 第四阶段：生产能力

1. `memory / persistence`
2. `streaming`
3. `observability`
4. `evaluation`
5. `human-in-the-loop`

### 第五阶段：进阶编排

1. LangChain agent
2. LangGraph graph/state/nodes/edges
3. Agentic RAG
4. 多 agent / 深度代理系统

---

## 18. 与你当前仓库目录的对应关系

结合你当前仓库结构，可以大致映射成下面这样：

| 仓库目录 | 对应 LangChain 知识层 |
|---|---|
| `models/` | 模型能力层 |
| `pkg_langchain/messages/` | 模型表达层中的 messages |
| `pkg_langchain/tools/` | 动作执行层中的 tools |
| `pkg_langchain/middleware/` | Agent 编排层中的 middleware |
| `pkg_langchain/retrieval/` | 知识检索层 |
| `pkg_langchain/structured_output/` | 模型表达层 / 输出层 |
| `pkg_langchain/agents/` | Agent 编排层 |
| `base/memory/` | 记忆与状态层 |
| `base/model_io/` | prompt / model input-output 基础层 |
| `project1/01_RAG_project.py` | RAG 实战 |
| `project2/semantic-search-indexing.py` | embeddings + vector store + retrieval |
| `note/` | 知识笔记层 |

这说明你的仓库已经覆盖了 LangChain 学习的主要板块，但目前更像“按功能散放的练习集合”，还没有完全收束为一张统一知识地图。这份文档就是把这些碎片统一起来。

---

## 19. 一张适合记忆的最终总图

```text
LangChain 1.x
├── 基础表达
│   ├── Messages
│   ├── Content Blocks
│   ├── Prompts
│   └── Structured Output
├── 模型能力
│   ├── Chat Models
│   ├── Embeddings
│   ├── Multimodality
│   └── Reasoning
├── 动作系统
│   ├── Tools
│   ├── Tool Calling
│   └── Toolkits
├── Agent 系统
│   ├── Agents
│   ├── Middleware
│   ├── Runtime
│   ├── State
│   └── Context Engineering
├── 知识系统
│   ├── Documents
│   ├── Loaders
│   ├── Splitters
│   ├── Vector Stores
│   ├── Retrievers
│   └── RAG
├── 记忆系统
│   ├── Short-term Memory
│   ├── Long-term Memory
│   ├── Store
│   ├── Checkpoint
│   └── Persistence
├── 交互输出
│   ├── Streaming
│   ├── Tool Messages
│   └── Response Metadata
└── 生产工程
    ├── Observability
    ├── Evaluation
    ├── Guardrails
    ├── Retry / Fallback
    └── Human-in-the-loop
```

---

## 20. 一个项目视角下的落地理解

如果你真的做一个 LangChain 1.x 项目，通常不是“选择某一个模块”，而是同时组合下面这些层：

1. **用 prompt + messages 组织输入**
2. **用 model 负责推理**
3. **用 tools 连接外部行动**
4. **用 agent 组织多步决策**
5. **用 middleware 控制调用链**
6. **用 embeddings + vector store + retriever 构建知识检索**
7. **用 memory / persistence 管理状态**
8. **用 streaming + structured output 优化交互**
9. **用 LangSmith 做 tracing 和评估**

所以，LangChain 1.x 最合理的理解方式不是“一个库有很多模块”，而是：

> **它是一套围绕 agent 构建的统一抽象系统。**

---

## 21. 本文档参考的官方资料方向

本文内容基于 LangChain 官方文档中以下主题整理：

1. LangChain overview
2. LangChain v1 release notes
3. Models
4. Messages
5. Runtime
6. Dynamic runtime context
7. Integrations providers overview
8. Knowledge base / semantic search tutorial
9. LangGraph agentic RAG
10. LangSmith observability

如果后续你要继续整理，我建议下一步分别拆成独立专题：

1. `LangChain 1.x Agent体系.md`
2. `LangChain 1.x Retrieval与RAG体系.md`
3. `LangChain 1.x Middleware与Runtime体系.md`
4. `LangChain 1.x Message与Prompt体系.md`
5. `LangChain 1.x Memory与Persistence体系.md`
