import asyncio
import os

from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage

from models import qwen
from tools.cus_tools import execute_command, list_directory, read_file, write_file
from tools.format import pretty_print_response

tools = [read_file, write_file, execute_command, list_directory]


async def run_agent(query):
    messages = [
        SystemMessage(
            "你是一个项目管理助手，使用工具完成任务。\n\n"
            f"当前工作目录: {os.getcwd()}\n\n"
            "工具：\n"
            "1. read_file: 读取文件\n"
            "2. write_file: 写入文件\n"
            "3. execute_command: 执行命令（支持 working_directory 参数）\n"
            "4. list_directory: 列出目录\n\n"
            "重要规则 - execute_command：\n"
            "- working_directory 参数会自动切换到指定目录\n"
            "- 当使用 working_directory 时，绝对不要在 command 中使用 cd\n"
            "- 错误示例: { command: 'cd react-todo-app && pnpm install', working_directory: 'react-todo-app' }\n"
            "这是错误的！因为 working_directory 已经在 react-todo-app 目录了，再 cd react-todo-app 会找不到目录\n"
            "- 正确示例: { command: 'pnpm install', working_directory: 'react-todo-app' }\n"
            "这样就对了！working_directory 已经切换到 react-todo-app，直接执行命令即可\n\n"
            "回复要简洁，只说做了什么"
        ),
        HumanMessage(content=query),
    ]

    agent = create_agent(model=qwen, tools=tools, middleware=[])
    return await agent.ainvoke({"messages": messages})


query = """
创建一个功能丰富的 React TodoList 应用：

1. 创建项目：echo -e "n\nn" | pnpm create vite react-todo-app --template react-ts
2. 修改 src/App.tsx，实现完整功能的 TodoList：
 - 添加、删除、编辑、标记完成
 - 分类筛选（全部/进行中/已完成）
 - 统计信息显示
 - localStorage 数据持久化
3. 添加复杂样式：
 - 渐变背景（蓝到紫）
 - 卡片阴影、圆角
 - 悬停效果
4. 添加动画：
 - 添加/删除时的过渡动画
 - 使用 CSS transitions
5. 列出目录确认

注意：使用 pnpm，功能要完整，样式要美观，要有动画效果

之后在 react-todo-app 项目中：
1. 使用 pnpm install 安装依赖
2. 使用 pnpm run dev 启动服务器
"""
# print(run_agent())
pretty_print_response(asyncio.run(run_agent(query)), view="zen")
