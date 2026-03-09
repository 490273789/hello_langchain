# "servers": {
#         "my-mcp-server": {
#             "command": "/Users/ethan/workspace/Python/hello_langchain/.venv/bin/python",
#             "args": [
#                 "/Users/ethan/workspace/Python/hello_langchain/7.search_database.py"
#             ]
#         }
#     }
from mcp.server.fastmcp import FastMCP

# 初始化 MCP 服务器
mcp = FastMCP(name="my-mcp-server")

# 模拟数据库
DATABASE = {
    "users": {
        "001": {
            "id": "001",
            "name": "张三",
            "email": "zhangsan@example.com",
            "role": "admin",
        },
        "002": {
            "id": "002",
            "name": "李四",
            "email": "lisi@example.com",
            "role": "user",
        },
        "003": {
            "id": "003",
            "name": "王五",
            "email": "wangwu@example.com",
            "role": "user",
        },
    }
}


# 注册工具：查询用户信息
# FastMCP 会自动根据函数签名和 docstring 生成 inputSchema 和 description
@mcp.tool()
def query_user(user_id: str) -> str:
    """
    查询数据库中的用户信息。输入用户 ID，返回该用户的详细信息（姓名、邮箱、角色）。

    Args:
        user_id: 用户 ID，例如: 001, 002, 003
    """
    user = DATABASE["users"].get(user_id)

    if not user:
        available_ids = ", ".join(DATABASE["users"].keys())
        return f"用户 ID {user_id} 不存在。可用的 ID: {available_ids}"

    return (
        f"用户信息：\n"
        f"- ID: {user['id']}\n"
        f"- 姓名：{user['name']}\n"
        f"- 邮箱：{user['email']}\n"
        f"- 角色：{user['role']}"
    )


# 注册资源：使用指南
# 资源通常用于提供静态或动态的只读数据上下文
@mcp.resource("docs://guide")
def get_guide() -> str:
    """MCP Server 使用文档"""
    return """MCP Server 使用指南

功能：提供用户查询等工具。

使用：在 Cursor 等 MCP Client 中通过自然语言对话，Cursor 会自动调用相应工具。"""


if __name__ == "__main__":
    # 启动服务器，默认使用 Stdio 传输模式（与 JS 版的 StdioServerTransport 对应）
    # 这将阻塞进程，等待客户端通过 stdin/stdout 通信
    mcp.run()
