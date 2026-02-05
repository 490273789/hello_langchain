import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def stdio_run():
    # 告诉客户端怎么启动服务端
    server_params = StdioServerParameters(
        command="python", args=["pkg_langchain/tools/mcp/1_mcp_stdio_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化链接
            await session.initialize()

            # 获取可用工具
            tools = await session.list_tools()
            print(f"tools: {tools}")

            # 调用工具
            call_res = await session.call_tool("add_two_number", {"a": 1, "b": 2})
            print(f"call_res: {call_res}")


asyncio.run(stdio_run())
