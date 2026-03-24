# MCP（Model Context Protocol）知识总结

## 1. 什么是 MCP？

MCP（模型上下文协议）是 Anthropic 提出的一套开放标准协议，用于让 LLM 应用与外部工具/数据源进行标准化通信。

**核心思想：** 将工具、资源、提示词从 LLM 应用中解耦，以标准协议的形式暴露给任意 AI 客户端（Claude、Cursor、LangChain Agent 等）。

**组成：**
- **MCP Server** — 暴露工具（Tool）、资源（Resource）、提示词（Prompt）
- **MCP Client** — 连接 Server，获取并调用工具
- **传输层（Transport）** — Server 与 Client 之间的通信方式

---

## 2. 传输方式（Transport）

| 传输方式 | 场景 | 特点 |
|---|---|---|
| `stdio` | 本地进程间通信 | Client 启动 Server 子进程，通过 stdin/stdout 通信 |
| `streamable-http` | 网络通信 | Server 暴露 HTTP 接口（默认路径 `/mcp`），支持流式 |
| `sse` | 网络通信（旧） | Server-Sent Events，阿里云 DashScope 等平台使用 |

---

## 3. MCP Server（使用 FastMCP）

### 3.1 创建 Server — stdio 模式

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="demo_mcp")

@mcp.tool()
def add_two_number(a: int, b: int):
    return a + b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### 3.2 创建 Server — HTTP 模式

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="demo_mcp_http")

@mcp.tool()
def add_two_number(a: int, b: int):
    return a + b

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
    # 默认监听 http://127.0.0.1:8000/mcp
```

### 3.3 注册工具（Tool）

- 使用 `@mcp.tool()` 装饰器
- **函数签名** 自动生成 `inputSchema`
- **docstring** 自动生成工具 `description`（支持 Args: 块描述每个参数）

```python
@mcp.tool()
def query_user(user_id: str) -> str:
    """
    查询数据库中的用户信息。

    Args:
        user_id: 用户 ID，例如: 001, 002, 003
    """
    ...
```

### 3.4 注册资源（Resource）

资源用于提供只读的上下文数据（如文档、配置、静态内容）：

```python
@mcp.resource("docs://guide")
def get_guide() -> str:
    """MCP Server 使用文档"""
    return "这是使用指南内容..."
```

### 3.5 挂载到 FastAPI（ASGI 模式，多 MCP 实例）

当需要在同一个服务中运行多个 MCP 实例时，可挂载到 FastAPI：

```python
import contextlib
import uvicorn
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP

tool_mcp = FastMCP("tool server")
resource_mcp = FastMCP("resource server")

tool_mcp.settings.streamable_http_path = "/"
resource_mcp.settings.streamable_http_path = "/"

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(tool_mcp.session_manager.run())
        await stack.enter_async_context(resource_mcp.session_manager.run())
        yield

app = FastAPI(lifespan=lifespan)
app.mount("/tool", tool_mcp.streamable_http_app())
app.mount("/resource", resource_mcp.streamable_http_app())

if __name__ == "__main__":
    uvicorn.run(app)
```

---

## 4. MCP Client（原生 SDK）

### 4.1 stdio Client

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["path/to/server.py"]   # 建议使用绝对路径
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()       # 获取工具列表
            result = await session.call_tool("add_two_number", {"a": 1, "b": 2})

asyncio.run(main())
```

### 4.2 streamable-http Client

```python
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

async def main():
    url = "http://127.0.0.1:8000/mcp"   # /mcp 是固定路径
    async with streamable_http_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool("add_two_number", {"a": 1, "b": 2})

asyncio.run(main())
```

---

## 5. LangChain 集成 MCP（langchain-mcp-adapters）

### 5.1 安装

```bash
pip install langchain-mcp-adapters
```

### 5.2 MultiServerMCPClient — 连接多个 MCP Server

`MultiServerMCPClient` 是 LangChain 与 MCP 集成的核心类，支持同时连接多个传输方式各异的 MCP Server，并将所有工具统一转换为 LangChain Tool 对象。

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

mcp_client = MultiServerMCPClient({
    # SSE 传输（适用于阿里云 DashScope 等平台）
    "WebSearch": {
        "transport": "sse",
        "url": "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/sse",
        "headers": {"Authorization": f"Bearer {API_KEY}"},
    },
    # HTTP 传输（streamable-http）
    "Maps": {
        "transport": "http",
        "url": f"https://mcp.amap.com/mcp?key={MAP_KEY}",
    },
    # stdio 传输（本地进程）
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
    },
})

# 获取所有工具（转换为 LangChain Tool 对象列表）
tools = asyncio.run(mcp_client.get_tools())
```

### 5.3 与 Agent 结合使用

```python
import asyncio
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

mcp_client = MultiServerMCPClient({
    "WebSearch": {
        "transport": "sse",
        "url": "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/sse",
        "headers": {"Authorization": f"Bearer {API_KEY}"},
    },
})

tools = asyncio.run(mcp_client.get_tools())

llm = init_chat_model(model="gpt-4o-mini", model_provider="openai")
agent = create_agent(model=llm, tools=tools)

async def main():
    async for chunk in agent.astream({
        "messages": [
            {"role": "system", "content": "你是一位助手，需要调用工具来帮助用户。"},
            {"role": "user", "content": "北京今天天气怎么样？"},
        ]
    }):
        print(chunk, end="\n\n")

asyncio.run(main())
```

### 5.4 传输方式配置对照

| 传输类型 | 配置 key | 说明 |
|---|---|---|
| SSE | `"transport": "sse"` | 适合云端 MCP 服务（阿里云 DashScope 等） |
| HTTP | `"transport": "http"` | streamable-http，适合自建 HTTP MCP 服务 |
| stdio | `"command"` + `"args"` | 本地子进程，无需 `transport` key |

---

## 6. 多 Agent 架构（Supervisor Pattern）

MCP 工具可以分配给不同子 Agent，各自专注于特定能力：

```
用户请求
   └── Supervisor Agent（路由）
          ├── SearchSubAgent（含 WebSearch、RailService、Maps MCP）
          └── EmailSubAgent（含自定义 send_email Tool）
```

```python
class SearchSubAgent:
    def __init__(self):
        self.tools = asyncio.run(
            MultiServerMCPClient({
                "WebSearch": { "transport": "sse", "url": "...", "headers": {...} },
                "Maps": { "transport": "http", "url": "..." },
                "filesystem": { "command": "npx", "args": [...] },
            }).get_tools()
        )
        self.agent = create_agent(model=llm, tools=self.tools)

    async def __call__(self, input: str) -> str:
        return await self.agent.ainvoke({"messages": [{"role": "user", "content": input}]})
```

---

## 7. 在 Cursor / IDE 中配置本地 MCP Server

在 `.vscode/mcp.json` 或 Cursor 的 MCP 配置中：

```json
{
  "servers": {
    "my-mcp-server": {
      "command": "/path/to/.venv/bin/python",
      "args": ["/absolute/path/to/server.py"]
    }
  }
}
```

> **注意：** 必须使用虚拟环境的绝对路径，否则依赖无法加载。

---

## 8. 注意事项 & 常见坑

| 问题 | 解决方案 |
|---|---|
| `FastMCP` 不支持 `version=` 参数 | 只传 `name`：`FastMCP(name="xxx")` |
| stdio 客户端找不到脚本 | 使用绝对路径，或基于 `__file__` 构建路径 |
| stdio 传输配置缺 `"transport": "stdio"` | MultiServerMCPClient 中 stdio 用 `command`+`args`，不需要写 `transport` |
| `langchain_mcp_adapters.tools.get_tools` 不存在 | 使用 `MultiServerMCPClient(...).get_tools()` |
| DashScope Embeddings 返回 400 | 使用 `OpenAIEmbeddings` 时设置 `check_embedding_ctx_length=False` |

---

## 9. 完整工作流

```
FastMCP Server
    ↓ 注册 Tool / Resource / Prompt
Transport（stdio / streamable-http / sse）
    ↓ 协议通信
MCP Client（原生 SDK 或 MultiServerMCPClient）
    ↓ 转换为 LangChain Tool
LangChain Agent（create_agent）
    ↓ 工具调用
用户响应
```
