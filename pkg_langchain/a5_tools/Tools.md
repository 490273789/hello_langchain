# LangChain Tools 学习笔记

## 1. 什么是 Tool？

Tool（工具）是 LangChain 中的核心概念，它是 Agent 与外部世界交互的方式。

> **Tool 是一个可以被 LLM 调用的函数，用于执行特定任务**

Tool 的关键特点：
- 有明确的名称和描述
- 有定义好的输入参数
- 返回执行结果给模型

### 1.1 Tool 的工作流程

1. **模型调用** - LLM 分析用户请求，决定是否需要调用工具
2. **工具选择** - LLM 选择合适的工具并生成调用参数
3. **工具执行** - 执行选中的工具，获取结果 
4. **结果处理** - 将工具结果返回给 LLM，继续推理

## 2. 创建 Tool 的方式

### 2.1 使用 @tool 装饰器（最简单）

```python
from langchain.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    return f"Found {limit} results for '{query}'"
```

**要点：**
- 函数的 docstring 自动成为工具描述
- 类型提示（type hints）定义输入 schema
- 描述帮助模型理解何时使用该工具

### 2.2 使用 parse_docstring 解析参数描述

```python
from langchain.tools import tool

@tool(parse_docstring=True)
def search_orders(
    user_id: str,
    status: str,
    limit: int = 10
) -> str:
    """Search for user orders by status.

    Use this when the user asks about order history or wants to check
    order status. Always filter by the provided status.

    Args:
        user_id: Unique identifier for the user
        status: Order status: 'pending', 'shipped', or 'delivered'
        limit: Maximum number of results to return
    """
    # Implementation here
    pass
```

### 2.3 使用 Pydantic 定义复杂输入 Schema

```python
from pydantic import BaseModel, Field
from typing import Literal
from langchain.tools import tool

class WeatherInput(BaseModel):
    """Input for weather queries."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result
```

### 2.4 使用 JSON Schema 定义输入

```python
weather_schema = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "units": {"type": "string"},
        "include_forecast": {"type": "boolean"}
    },
    "required": ["location", "units", "include_forecast"]
}

@tool(args_schema=weather_schema)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    pass
```

### 2.5 使用 Pydantic 类作为工具（bind_tools）

```python
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

class GetWeather(BaseModel):
    """Get the current weather in a given location"""
    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

class GetPopulation(BaseModel):
    """Get the current population in a given location"""
    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

model = init_chat_model(temperature=0)
model_with_tools = model.bind_tools([GetWeather, GetPopulation])
```

## 3. 将 Tool 绑定到模型

### 3.1 基本绑定

```python
from langchain.tools import tool
from langchain.chat_models import init_chat_model

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."

model = init_chat_model("gpt-4o")
model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke("What's the weather like in Boston?")
for tool_call in response.tool_calls:
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
```

### 3.2 处理并行工具调用

模型可能一次请求多个工具调用：

```python
model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke("What's the weather in Boston and Tokyo?")

# 模型可能生成多个工具调用
print(response.tool_calls)
# [
#   {'name': 'get_weather', 'args': {'location': 'Boston'}, 'id': 'call_1'},
#   {'name': 'get_weather', 'args': {'location': 'Tokyo'}, 'id': 'call_2'},
# ]

# 执行所有工具（可以用 async 并行执行）
results = []
for tool_call in response.tool_calls:
    if tool_call['name'] == 'get_weather':
        result = get_weather.invoke(tool_call)
    results.append(result)
```

## 4. 在 Agent 中使用 Tool

### 4.1 创建带工具的 Agent

```python
from langchain.tools import tool
from langchain.agents import create_agent

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

agent = create_agent(
    model="gpt-4o",
    tools=[search, get_weather]
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in SF?"}]
})
```

### 4.2 复杂工具示例

```python
from langchain_core.tools import tool

@tool
def create_calendar_event(
    title: str,
    start_time: str,       # ISO format: "2024-01-15T14:00:00"
    end_time: str,         # ISO format: "2024-01-15T15:00:00"
    attendees: list[str],  # email addresses
    location: str = ""
) -> str:
    """Create a calendar event. Requires exact ISO datetime format."""
    return f"Event created: {title} from {start_time} to {end_time} with {len(attendees)} attendees"

@tool
def send_email(
    to: list[str],
    subject: str,
    body: str,
    cc: list[str] = []
) -> str:
    """Send an email via email API. Requires properly formatted addresses."""
    return f"Email sent to {', '.join(to)} - Subject: {subject}"

@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,  # ISO format: "2024-01-15"
    duration_minutes: int
) -> list[str]:
    """Check calendar availability for given attendees on a specific date."""
    return ["09:00", "14:00", "16:00"]
```

## 5. Tool 访问运行时状态

### 5.1 使用 ToolRuntime 访问状态

```python
from langchain.tools import tool, ToolRuntime

@tool
def summarize_conversation(runtime: ToolRuntime) -> str:
    """Summarize the conversation so far."""
    messages = runtime.state["messages"]

    human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    tool_msgs = sum(1 for m in messages if m.__class__.__name__ == "ToolMessage")

    return f"Conversation has {human_msgs} user messages, {ai_msgs} AI responses, and {tool_msgs} tool results"

# 访问自定义状态字段
@tool
def get_user_preference(pref_name: str, runtime: ToolRuntime) -> str:
    """Get a user preference value."""
    preferences = runtime.state.get("user_preferences", {})
    return preferences.get(pref_name, "Not set")
```

**注意：** `ToolRuntime` 参数对模型不可见，不会出现在工具的参数 schema 中。

### 5.2 从 Store 读取用户偏好

```python
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
    user_id: str

@tool
def get_preference(preference_key: str, runtime: ToolRuntime[Context]) -> str:
    """Get user preference from Store."""
    user_id = runtime.context.user_id

    store = runtime.store
    existing_prefs = store.get(("preferences",), user_id)

    if existing_prefs:
        value = existing_prefs.value.get(preference_key)
        return f"{preference_key}: {value}" if value else f"No preference set for {preference_key}"
    else:
        return "No preferences found"

agent = create_agent(
    model="gpt-4o",
    tools=[get_preference],
    context_schema=Context,
    store=InMemoryStore()
)
```

## 6. Tool 错误处理

### 6.1 使用中间件处理工具错误

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
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

### 6.2 工具重试中间件

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware

# 基本配置
agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, database_tool],
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
        ),
    ],
)

# 高级配置
agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, database_tool, api_tool],
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
            max_delay=60.0,
            jitter=True,
            tools=["api_tool"],  # 只对特定工具重试
            retry_on=(ConnectionError, TimeoutError),  # 只重试特定异常
            on_failure="return_message",
        ),
    ],
)
```

## 7. 动态工具选择

### 7.1 基于状态过滤工具

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@wrap_model_call
def state_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on conversation State."""
    state = request.state
    is_authenticated = state.get("authenticated", False)
    message_count = len(state["messages"])

    # 未认证时只启用公开工具
    if not is_authenticated:
        tools = [t for t in request.tools if t.name.startswith("public_")]
        request = request.override(tools=tools)
    elif message_count < 5:
        # 对话初期限制工具
        tools = [t for t in request.tools if t.name != "advanced_search"]
        request = request.override(tools=tools)

    return handler(request)

agent = create_agent(
    model="gpt-4o",
    tools=[public_search, private_search, advanced_search],
    middleware=[state_based_tools]
)
```

### 7.2 基于 Store 过滤工具

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
    user_id: str

@wrap_model_call
def store_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on Store preferences."""
    user_id = request.runtime.context.user_id

    store = request.runtime.store
    feature_flags = store.get(("features",), user_id)

    if feature_flags:
        enabled_features = feature_flags.value.get("enabled_tools", [])
        tools = [t for t in request.tools if t.name in enabled_features]
        request = request.override(tools=tools)

    return handler(request)

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, analysis_tool, export_tool],
    middleware=[store_based_tools],
    context_schema=Context,
    store=InMemoryStore()
)
```

## 8. ToolMessage 的使用

当模型请求工具调用后，需要返回 `ToolMessage` 给模型：

```python
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

# 模型发起工具调用后
ai_message = AIMessage(
    content=[],
    tool_calls=[{
        "name": "get_weather",
        "args": {"location": "San Francisco"},
        "id": "call_123"
    }]
)

# 执行工具并创建结果消息
weather_result = "Sunny, 72°F"
tool_message = ToolMessage(
    content=weather_result,
    tool_call_id="call_123"  # 必须匹配调用 ID
)

# 继续对话
messages = [
    HumanMessage("What's the weather in San Francisco?"),
    ai_message,      # 模型的工具调用
    tool_message     # 工具执行结果
]
response = model.invoke(messages)
```

## 9. 使用 SQLDatabaseToolkit（内置工具集）

```python
from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=model)

tools = toolkit.get_tools()

for tool in tools:
    print(f"{tool.name}: {tool.description}\n")
```

## 10. 将 Sub-Agent 作为工具

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

# 主 Agent 使用子 Agent 作为工具
agent = create_agent(model="...", tools=[call_subagent1])
```

## 11. 最佳实践

### 11.1 工具描述要清晰

- 使用清晰、具体的 docstring
- 说明工具的用途和使用场景
- 描述参数的格式要求

### 11.2 动态选择工具

- 工具太多会使模型困惑，增加错误
- 工具太少会限制功能
- 根据上下文动态过滤工具

### 11.3 错误处理

- 使用中间件统一处理工具错误
- 返回有意义的错误消息给模型
- 对瞬态错误使用重试机制

### 11.4 类型安全

- 使用 Pydantic 定义复杂输入
- 添加参数验证
- 提供默认值

## 12. 常见问题

### 12.1 工具调用失败的常见原因

1. **描述不清晰** - 模型不知道何时使用工具
2. **参数格式错误** - 模型提供了错误格式的参数
3. **工具太多** - 模型无法有效选择

### 12.2 调试建议

1. 检查工具的 docstring 是否清晰
2. 验证参数类型提示是否正确
3. 使用日志记录工具调用和结果
4. 考虑使用中间件添加监控
