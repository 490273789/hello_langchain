import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def streamable_http_run():
    # 告诉客户端怎么启动服务端
    url = "http://127.0.0.1:8000/mcp"  # /mcp是一个固定的路径
    # headers = {"Authorization": "Bearer sk-atguigu"}
    async with streamable_http_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            # 初始化链接
            await session.initialize()

            # 获取可用工具
            tools = await session.list_tools()
            print(f"tools: {tools}")

            # 调用工具
            call_res = await session.call_tool("add_two_number", {"a": 1, "b": 2})
            print(f"call_res: {call_res}")


asyncio.run(streamable_http_run())
