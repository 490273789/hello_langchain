# LangChain Memory 模块学习笔记

## 1. Memory 概述

Memory（记忆）是一个能够记住先前交互信息的系统。对于 AI Agent 来说，Memory 至关重要，因为它能让 Agent：
- 记住之前的对话内容
- 从反馈中学习
- 适应用户偏好

随着 Agent 处理越来越复杂的任务和大量用户交互，这种能力对于效率和用户满意度变得至关重要。

## 2. Memory 的两种类型

### 2.1 短期记忆 (Short-term Memory)

短期记忆让应用程序能够记住**单个会话（thread）内**的先前交互。对话历史是短期记忆最常见的形式。

**核心挑战**：长对话对当前的 LLM 是一个挑战，完整的对话历史可能无法放入 LLM 的上下文窗口，导致上下文丢失或错误。

#### 使用 InMemorySaver 实现短期记忆

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    "gpt-4",
    [get_user_info],
    checkpointer=InMemorySaver(),  # 内存检查点
)

# 使用 thread_id 来标识会话
agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
    {"configurable": {"thread_id": "1"}},
)
```

#### 使用 PostgreSQL 持久化短期记忆（生产环境）

```python
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # 自动创建表
    agent = create_agent(
        "gpt-4",
        [get_user_info],
        checkpointer=checkpointer,
    )
```

### 2.2 长期记忆 (Long-term Memory)

长期记忆用于**跨会话**存储信息，每个记忆都在自定义的 `namespace`（类似文件夹）和独特的 `key`（类似文件名）下组织。

#### 使用 InMemoryStore 管理长期记忆

```python
from langgraph.store.memory import InMemoryStore

def embed(texts: list[str]) -> list[list[float]]:
    # 替换为实际的嵌入函数
    return [[1.0, 2.0] * len(texts)]

store = InMemoryStore(index={"embed": embed, "dims": 2})
user_id = "my-user"
application_context = "chitchat"
namespace = (user_id, application_context)

# 存储数据
store.put(
    namespace,
    "a-memory",
    {
        "rules": [
            "User likes short, direct language",
            "User only speaks English & python",
        ],
        "my-key": "my-value",
    },
)

# 获取数据
item = store.get(namespace, "a-memory")

# 搜索数据
items = store.search(
    namespace, filter={"my-key": "my-value"}, query="language preferences"
)
```

## 3. 消息管理策略

### 3.1 使用 Message 对象管理对话历史

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

conversation = [
    SystemMessage("You are a helpful assistant that translates English to French."),
    HumanMessage("Translate: I love programming."),
    AIMessage("J'adore la programmation."),
    HumanMessage("Translate: I love building applications.")
]

response = model.invoke(conversation)
```

也可以使用字典格式：

```python
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"}
]

response = model.invoke(conversation)
```

### 3.2 裁剪消息 (Trim Messages)

当对话历史过长时，可以通过裁剪来管理上下文窗口：

```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model

@before_model
def trim_messages(state: AgentState, runtime) -> dict | None:
    """只保留最近的几条消息以适应上下文窗口"""
    messages = state["messages"]
    
    if len(messages) <= 3:
        return None  # 不需要修改
    
    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages
    
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

agent = create_agent(
    model,
    tools=tools,
    middleware=[trim_messages],
    checkpointer=InMemorySaver(),
)
```

### 3.3 删除所有消息

```python
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.messages import RemoveMessage

def delete_messages(state):
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
```

### 3.4 自动删除旧消息

```python
from langchain.messages import RemoveMessage
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langgraph.checkpoint.memory import InMemorySaver

@after_model
def delete_old_messages(state: AgentState, runtime) -> dict | None:
    """删除旧消息以保持对话可管理"""
    messages = state["messages"]
    if len(messages) > 2:
        # 删除最早的两条消息
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
    return None

agent = create_agent(
    "gpt-4",
    tools=[],
    system_prompt="Please be concise.",
    middleware=[delete_old_messages],
    checkpointer=InMemorySaver(),
)
```

### 3.5 消息摘要 (Summarization)

使用 `SummarizationMiddleware` 在达到 token 限制时自动总结对话历史：

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            max_tokens_before_summary=4000,  # 在 4000 tokens 时触发摘要
            messages_to_keep=20,  # 摘要后保留最后 20 条消息
        )
    ],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "1"}}
agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)
# Agent 仍能记住用户名字是 Bob
```

## 4. 在 Tools 中访问和更新 Memory

### 4.1 从 Store 读取数据

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
    user_id: str

store = InMemoryStore()

# 预填充数据
store.put(
    ("users",),
    "user_123",
    {"name": "John Smith", "language": "English"}
)

@tool
def get_user_info(runtime: ToolRuntime[Context]) -> str:
    """查询用户信息"""
    store = runtime.store
    user_id = runtime.context.user_id
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

agent = create_agent(
    model="gpt-4",
    tools=[get_user_info],
    store=store,
    context_schema=Context
)

agent.invoke(
    {"messages": [{"role": "user", "content": "look up user information"}]},
    context=Context(user_id="user_123")
)
```

### 4.2 向 Store 写入数据

```python
from typing_extensions import TypedDict

class UserInfo(TypedDict):
    name: str

@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[Context]) -> str:
    """保存用户信息"""
    store = runtime.store
    user_id = runtime.context.user_id
    store.put(("users",), user_id, user_info)
    return "Successfully saved user info."

agent = create_agent(
    model="gpt-4",
    tools=[save_user_info],
    store=store,
    context_schema=Context
)
```

### 4.3 保存用户偏好

```python
@tool
def save_preference(
    preference_key: str,
    preference_value: str,
    runtime: ToolRuntime[Context]
) -> str:
    """保存用户偏好到 Store"""
    user_id = runtime.context.user_id
    store = runtime.store
    
    # 读取现有偏好
    existing_prefs = store.get(("preferences",), user_id)
    
    # 合并新偏好
    prefs = existing_prefs.value if existing_prefs else {}
    prefs[preference_key] = preference_value
    
    # 写入 Store
    store.put(("preferences",), user_id, prefs)
    
    return f"Saved preference: {preference_key} = {preference_value}"
```

## 5. 基于 Memory 自定义 Prompt

可以根据 Store 中的用户偏好动态调整系统提示：

```python
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langgraph.store.memory import InMemoryStore

@dynamic_prompt
def store_aware_prompt(request: ModelRequest) -> str:
    user_id = request.runtime.context.user_id
    store = request.runtime.store
    user_prefs = store.get(("preferences",), user_id)
    
    base = "You are a helpful assistant."
    
    if user_prefs:
        style = user_prefs.value.get("communication_style", "balanced")
        base += f"\nUser prefers {style} responses."
    
    return base

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[store_aware_prompt],
    context_schema=Context,
    store=InMemoryStore()
)
```

## 6. 总结

| 类型 | 用途 | 存储方式 | 适用场景 |
|------|------|----------|----------|
| 短期记忆 | 单会话内的对话历史 | InMemorySaver / PostgresSaver | 对话上下文保持 |
| 长期记忆 | 跨会话的用户数据 | InMemoryStore / DB Store | 用户偏好、个性化 |

**关键概念**：
- **Checkpointer**：用于保存对话状态（短期记忆）
- **Store**：用于保存持久化数据（长期记忆）
- **thread_id**：标识不同的对话会话
- **namespace + key**：组织和访问 Store 中的数据

**消息管理策略**：
- **裁剪 (Trim)**：保留首条和最近的 N 条消息
- **删除 (Delete)**：删除过旧的消息
- **摘要 (Summarize)**：将历史消息压缩为摘要
