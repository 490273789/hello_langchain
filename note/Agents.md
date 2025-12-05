# LangChain Agents 学习笔记

## 1. 什么是 Agent？

Agent（代理）是 LangChain 中的核心概念之一。简单来说：

> **Agent 是一个在循环中运行工具以实现目标的 LLM 系统**

Agent 会持续运行直到满足停止条件：
- 模型输出最终响应
- 达到迭代次数限制

### 1.1 Agent 的核心循环

Agent 的执行遵循一个循环模式：

1. **模型调用** - 使用提示词和可用工具调用 LLM，返回响应或请求执行工具
2. **工具执行** - 执行 LLM 请求的工具，返回工具结果
3. 重复以上步骤直到完成任务

这种模式源自 [ReAct 论文](https://arxiv.org/abs/2210.03629)（ReAct = Reasoning and Acting），通过推理和行动相结合来解决问题。

## 2. 创建基础 Agent

### 2.1 基本用法

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="claude-sonnet-4-5-20250929",  # 指定模型
    tools=[get_weather],                  # 绑定工具
    system_prompt="You are a helpful assistant",  # 系统提示词
)

# 运行 Agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

### 2.2 使用 @tool 装饰器定义工具

```python
from langchain.tools import tool
from langchain.agents import create_agent

@tool
def search(query: str) -> str:
    """搜索信息"""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """获取指定地点的天气信息"""
    return f"Weather in {location}: Sunny, 72°F"

agent = create_agent(model, tools=[search, get_weather])
```

## 3. Agent 的核心组件

### 3.1 模型 (Model)

模型是 Agent 的推理引擎，可以通过多种方式指定：

```python
# 方式1：使用字符串标识符（静态模型）
agent = create_agent("gpt-5", tools=tools)

# 方式2：使用模型实例
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o")
agent = create_agent(model, tools=tools)
```

### 3.2 工具 (Tools)

工具是 Agent 与外部世界交互的方式，LLM 可以请求执行工具来完成任务。

### 3.3 系统提示词 (System Prompt)

系统提示词设置 LLM 的行为和能力：

```python
agent = create_agent(
    model,
    tools=[retrieve_context],
    system_prompt=(
        "You have access to a tool that retrieves context from a blog post. "
        "Use the tool to help answer user queries."
    )
)
```

## 4. 短期记忆 (Short-term Memory)

### 4.1 使用 Checkpointer 保持对话状态

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    "gpt-5",
    tools=[get_user_info],
    checkpointer=InMemorySaver(),  # 内存检查点
)

# 使用 thread_id 维护会话
agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
    {"configurable": {"thread_id": "1"}},
)
```

### 4.2 生产环境使用 PostgreSQL 持久化

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    agent = create_agent(
        "gpt-5",
        tools=[get_user_info],
        checkpointer=checkpointer,
    )
```

### 4.3 扩展 Agent 状态

```python
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver

class CustomAgentState(AgentState):
    user_id: str
    preferences: dict

agent = create_agent(
    "gpt-5",
    tools=[get_user_info],
    state_schema=CustomAgentState,
    checkpointer=InMemorySaver(),
)

# 调用时传入自定义状态
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user_123",
        "preferences": {"theme": "dark"}
    },
    {"configurable": {"thread_id": "1"}}
)
```

## 5. 中间件 (Middleware)

中间件允许你在 Agent 执行过程中插入自定义逻辑：

- 在模型调用前处理状态（消息裁剪、上下文注入）
- 修改或验证模型响应（护栏、内容过滤）
- 处理工具执行错误
- 实现动态模型选择
- 添加日志、监控或分析

### 5.1 消息裁剪中间件

```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model

@before_model
def trim_messages(state: AgentState, runtime) -> dict | None:
    """只保留最近几条消息以适应上下文窗口"""
    messages = state["messages"]
    if len(messages) <= 3:
        return None
    
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

### 5.2 工具错误处理中间件

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    """处理工具执行错误并返回自定义消息"""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model="gpt-4o",
    tools=[search, get_weather],
    middleware=[handle_tool_errors]
)
```

### 5.3 消息摘要中间件

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            max_tokens_before_summary=4000,  # 达到4000 tokens时触发摘要
            messages_to_keep=20,  # 摘要后保留最近20条消息
        )
    ],
    checkpointer=checkpointer,
)
```

## 6. 人机协作 (Human-in-the-Loop)

### 6.1 配置人工审批中间件

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-4o",
    tools=[write_file_tool, execute_sql_tool, read_data_tool],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file": True,  # 允许所有决策（批准、编辑、拒绝）
                "execute_sql": {"allowed_decisions": ["approve", "reject"]},  # 不允许编辑
                "read_data": False  # 安全操作，无需审批
            },
            description_prefix="Tool execution pending approval"
        )
    ],
    checkpointer=InMemorySaver()
)
```

### 6.2 处理中断并恢复执行

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "some_id"}}

# 运行 Agent 直到遇到中断
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Delete old records from the database"}]},
    config=config
)

# 检查中断信息
print(result['__interrupt__'])

# 恢复执行（批准操作）
agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config
)
```

## 7. 结构化输出 (Structured Output)

### 7.1 使用 ToolStrategy

```python
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_tool],
    response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

print(result["structured_response"])
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
```

## 8. 多 Agent 系统

### 8.1 使用子 Agent 作为工具

```python
from langchain.tools import tool
from langchain.agents import create_agent

# 创建子 Agent
subagent1 = create_agent(model="...", tools=[...])

@tool("subagent1_name", description="subagent1_description")
def call_subagent1(query: str):
    result = subagent1.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content

# 主 Agent 使用子 Agent
agent = create_agent(model="...", tools=[call_subagent1])
```

### 8.2 Supervisor 模式

创建一个监督者 Agent 来协调多个专业 Agent：

```python
# 创建专业 Agent
calendar_agent = create_agent(
    model,
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=CALENDAR_AGENT_PROMPT,
)

email_agent = create_agent(
    model,
    tools=[send_email],
    system_prompt=EMAIL_AGENT_PROMPT,
)

# 将 Agent 包装为工具
@tool
def schedule_event(request: str) -> str:
    """使用自然语言安排日历事件"""
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text

@tool
def manage_email(request: str) -> str:
    """使用自然语言发送邮件"""
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text

# 创建监督者 Agent
supervisor_agent = create_agent(
    model,
    tools=[schedule_event, manage_email],
    system_prompt=(
        "You are a helpful personal assistant. "
        "You can schedule calendar events and send emails. "
        "Break down user requests into appropriate tool calls and coordinate the results."
    )
)
```

## 9. 使用场景

### 9.1 何时需要多 Agent

- 单个 Agent 工具过多，决策质量下降
- 上下文或记忆对单个 Agent 来说太大
- 任务需要**专业化**（如规划者、研究者、数学专家）

### 9.2 RAG Agent

```python
from langchain.agents import create_agent

tools = [retrieve_context]
prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)
agent = create_agent(model, tools, system_prompt=prompt)
```

## 10. 最佳实践

1. **明确定义工具描述** - 好的工具描述帮助 LLM 正确选择工具
2. **合理使用记忆** - 开发时用 InMemorySaver，生产环境用持久化存储
3. **添加错误处理** - 使用中间件捕获和处理工具错误
4. **敏感操作加审批** - 对写入、删除等操作使用人机协作
5. **控制上下文长度** - 使用消息裁剪或摘要防止超出上下文窗口

## 11. Agent 失败的常见原因

Agent 失败通常是因为 LLM 调用采取了错误的行动。LLM 失败的两个主要原因：

1. **上下文不足** - 缺少必要的信息来做出正确决策
2. **指令不清晰** - 系统提示词没有清楚说明如何使用工具

解决方案：优化系统提示词，确保提供足够的上下文信息。
