import asyncio
import os
import re
from pathlib import Path

# from pathlib import Path
import dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools.base import ToolException

# MCP 适配器
from langchain_mcp_adapters.client import MultiServerMCPClient

from models import qwen
from tools.format_print import pretty_print_ai

dotenv.load_dotenv()


# /Users/ethan/workspace/Python/hello_langchain/pkg_langchain/tools/mcp/7-1_sear_database_client.py
# /Users/ethan/workspace/Python/hello_langchain/pkg_langchain/tools/mcp/7_search_database.py
class Colors:
    BG_GREEN = "\033[42m"
    BG_BLUE = "\033[44m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_status(message, color_bg=None):
    if color_bg:
        print(f"{color_bg} {message} {Colors.RESET}")
    else:
        print(message)


def extract_tool_result_text(tool_result) -> str:
    if isinstance(tool_result, str):
        return tool_result

    if isinstance(tool_result, dict):
        text = tool_result.get("text")
        if isinstance(text, str):
            return text
        return str(tool_result)

    if isinstance(tool_result, list):
        # MCP/LLM tool calls may return content blocks like
        # [{"type": "text", "text": "..."}, ...].
        text_parts = []
        for item in tool_result:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                item_text = item.get("text")
                if isinstance(item_text, str):
                    text_parts.append(item_text)

        if text_parts:
            return "\n".join(text_parts)
        return str(tool_result)

    return str(tool_result)


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


def find_tool_by_name(tools, name: str):
    return next((t for t in tools if t.name == name), None)


def coerce_select_page_args(tool_name: str, tool_args):
    if tool_name != "select_page" or not isinstance(tool_args, dict):
        return tool_args

    args = dict(tool_args)
    if "pageId" not in args:
        for alias in ["page_id", "page", "id", "index", "pageIndex", "page_idx"]:
            if alias in args:
                args["pageId"] = args[alias]
                break

    page_id = args.get("pageId")
    if isinstance(page_id, str):
        stripped = page_id.strip()
        if stripped.isdigit():
            args["pageId"] = int(stripped)

    if isinstance(args.get("pageId"), int) and args["pageId"] <= 0:
        args["pageId"] = 1

    return args


def extract_page_iqwen(value) -> list[int]:
    text = extract_tool_result_text(value)
    iqwen = [int(m.group(1)) for m in re.finditer(r"(?m)^\s*(\d+)\s*:", text)]
    return sorted(set(iqwen))


async def main():
    # print(*(p for p in os.getenv("ALLOWED_PATHS").split(",") if p))
    # path = [p for p in os.getenv("ALLOWED_PATHS").split(",") if p]
    # server_file = Path(__file__).with_name("7_search_database.py")

    # 3. 初始化 MCP 客户端 (对应 MultiServerMCPClient)
    # 注意：command 和 args 需要适配 Python 环境或保留 node 调用

    mcp_client = MultiServerMCPClient(
        {
            # "my-mcp-server": {
            #     "transport": "stdio",
            #     "command": "/Users/ethan/workspace/Python/hello_langchain/.venv/bin/python",
            #     # 请确保此路径指向您之前创建的 server.js 或编译后的文件
            #     # 如果您将之前的代码改写成了 Python (server.py)，这里应改为:
            #     # "command": "python",
            #     # "args": ["/Users/guang/code/tool-test/src/my-mcp-server.py"]
            #     "args": [str(server_file)],
            # },
            "amap-maps-streamableHTTP": {
                "transport": "http",
                "url": f"https://mcp.amap.com/mcp?key={os.getenv('AMAP_MAPS_API_KEY')}",
            },
            "filesystem": {
                "transport": "stdio",
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    *[p for p in os.getenv("ALLOWED_PATHS", "").split(",") if p],
                ],
            },
            "chrome-devtools": {
                "transport": "stdio",
                "command": "npx",
                "args": [
                    "-y",
                    "chrome-devtools-mcp@latest",
                ],
            },
        }
    )

    # 4. 获取工具
    # langchain-mcp-adapters 新版通过 client.get_tools() 获取工具。
    tools = await mcp_client.get_tools()

    # 绑定工具到模型
    model_with_tools = qwen.bind_tools(tools)

    # 5. 运行 Agent 逻辑
    # query = "北京南站附近的2个酒店，以及去的路线，路线规划生成文档保存到 /Users/ethan/Desktop 的一个 md 文件"
    query = "北京南站附近的酒店，最近的 2 个酒店，拿到酒店图片，打开浏览器，展示每个酒店的图片，每个 tab 一个 url 展示，并且在把那个页面标题改为酒店名"
    messages = [HumanMessage(content=query)]
    max_iterations = 30
    for i in range(max_iterations):
        print_status("⏳ 正在等待 AI 思考...", Colors.BG_GREEN)
        response = await model_with_tools.ainvoke(messages)
        messages.append(response)

        # 检查是否有工具调用
        # LangChain Python 版中，tool_calls 通常在 response.additional_kwargs 或直接作为属性存在
        tool_calls = getattr(response, "tool_calls", [])

        if not tool_calls:
            pretty_print_ai(messages, view="message")
            print(f"\n✨ AI 最终回复:\n{response.content}\n")
            return response.content

        print_status(f"🔍 检测到 {len(tool_calls)} 个工具调用", Colors.BG_BLUE)
        tool_names = [tc["name"] for tc in tool_calls]
        print_status(f"🔍 工具调用: {', '.join(tool_names)}", Colors.BG_BLUE)

        # 执行工具调用
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            # 查找对应的工具对象
            found_tool = find_tool_by_name(tools, tool_name)

            if found_tool:
                try:
                    # invoke 工具
                    # 注意：LangChain 工具 invoke 通常接受 input 字典
                    tool_result = await found_tool.ainvoke(tool_args)
                    print(f"tool_result: {tool_result}")

                    # 将结果添加入消息历史
                    messages.append(
                        ToolMessage(content=str(tool_result), tool_call_id=tool_call_id)
                    )
                except ToolException as exc:
                    error_text = str(exc)
                    print(f"⚠️ 工具调用失败 {tool_name}: {error_text}")

                    # 将错误反馈给模型，让下一轮自行修正参数。
                    messages.append(
                        ToolMessage(
                            content=f"ToolError: {error_text}",
                            tool_call_id=tool_call_id,
                        )
                    )
            else:
                print(f"⚠️ 未找到工具: {tool_name}")


if __name__ == "__main__":
    asyncio.run(main())
