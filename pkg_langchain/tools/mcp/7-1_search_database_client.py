import asyncio
from pathlib import Path

from langchain_core.messages import HumanMessage, ToolMessage

# MCP 适配器
from langchain_mcp_adapters.client import MultiServerMCPClient

from models import ds
from tools.format import pretty_print_response


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


async def main():
    server_file = Path(__file__).with_name("7_search_database.py")

    # 3. 初始化 MCP 客户端 (对应 MultiServerMCPClient)
    # 注意：command 和 args 需要适配 Python 环境或保留 node 调用
    mcp_client = MultiServerMCPClient(
        {
            "my-mcp-server": {
                "transport": "stdio",
                "command": "/Users/ethan/workspace/Python/hello_langchain/.venv/bin/python",
                # 请确保此路径指向您之前创建的 server.js 或编译后的文件
                # 如果您将之前的代码改写成了 Python (server.py)，这里应改为:
                # "command": "python",
                # "args": ["/Users/guang/code/tool-test/src/my-mcp-server.py"]
                "args": [str(server_file)],
            }
        }
    )

    # 4. 获取工具
    # langchain-mcp-adapters 新版通过 client.get_tools() 获取工具。
    tools = await mcp_client.get_tools()

    # 绑定工具到模型
    model_with_tools = ds.bind_tools(tools)

    # 5. 运行 Agent 逻辑
    query = "查一下用户 002 的信息"
    messages = [HumanMessage(content=query)]
    max_iterations = 30

    for i in range(max_iterations):
        print_status("⏳ 正在等待 AI 思考...", Colors.BG_GREEN)
        pretty_print_response(messages, view="full")
        response = await model_with_tools.ainvoke(messages)

        messages.append(response)

        # 检查是否有工具调用
        # LangChain Python 版中，tool_calls 通常在 response.additional_kwargs 或直接作为属性存在
        tool_calls = getattr(response, "tool_calls", [])

        if not tool_calls:
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
            found_tool = next((t for t in tools if t.name == tool_name), None)

            if found_tool:
                #  invoke 工具
                # 注意：LangChain 工具 invoke 通常接受 input 字典
                tool_result = await found_tool.ainvoke(tool_args)

                # 将结果添加入消息历史
                messages.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tool_call_id)
                )
            else:
                print(f"⚠️ 未找到工具: {tool_name}")


if __name__ == "__main__":
    asyncio.run(main())
