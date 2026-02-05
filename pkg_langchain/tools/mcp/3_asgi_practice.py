import contextlib

import uvicorn
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP

# 创建MCP实例
tool_mcp = FastMCP("tool server")
resource_mcp = FastMCP("resource server")
prompt_mcp = FastMCP("prompt server")

# 为tool_mcp 实例添加工具


@tool_mcp.tool()
def add(a: int, b: int):
    return a + b


# 为resource_mcp实例添加资源
@resource_mcp.tool()
def get_greeting() -> str:
    return "Hello from static resource!"


# 为prompt_mcp添加工具
@prompt_mcp.tool()
def greet_user(name: str, style: str = "friendly") -> str:
    styles = {
        "friendly": "写一句友善的问候",
        "formal": "写一句正式的问候",
        "casual": "写一句轻松的问候",
    }
    return f"为{name}{styles.get(style, styles['friendly'])}"


# 设置 MCP 的 HTTP 根路径
tool_mcp.settings.streamable_http_path = "/"
resource_mcp.settings.streamable_http_path = "/"
prompt_mcp.settings.streamable_http_path = "/"


# 创建一个组合生命周期来管理会话管理器
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(tool_mcp.session_manager.run())
        await stack.enter_async_context(resource_mcp.session_manager.run())
        await stack.enter_async_context(prompt_mcp.session_manager.run())
        yield


app = FastAPI(lifespan=lifespan)

# 挂载 MCP 服务器
app.mount("/tool", tool_mcp.streamable_http_app())
app.mount("/resource", resource_mcp.streamable_http_app())
app.mount("/prompt", prompt_mcp.streamable_http_app())

if __name__ == "__main__":
    uvicorn.run(app)
