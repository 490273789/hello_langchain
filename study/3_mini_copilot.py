import os

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from models import ds
from tools.all_tools import TOOLS
from tools.format import pretty_print

# model = ChatOpenAI(
#     model="qwen-plus",
#     api_key=os.getenv("OPENAI_API_KEY"),
#     temperature=0,
#     base_url=os.getenv("OPENAI_BASE_URL"),
# )

model = ds.bind_tools(TOOLS)

# model_with_tools = model.bind_tools(TOOLS)


def run_agent_with_tools(query: str, max_iterations: int = 30) -> str:
    messages = [
        SystemMessage(
            content=(
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
            )
        ),
        HumanMessage(content=query),
    ]

    for _ in range(max_iterations):
        print("\033[30;43m[agent] ⏳ 正在等待 AI 思考...\033[0m")
        response = model.invoke(messages)
        print("\033[30;43m[agent] ⏳ 思考完成...\033[0m")
        pretty_print(response, view="ms")
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            print(f"\n[agent] AI 最终回复:\n{response.content}\n")
            return str(response.content)

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id", "")

            found_tool = next((tool for tool in TOOLS if tool.name == tool_name), None)
            if found_tool is None:
                messages.append(
                    ToolMessage(
                        content=f"工具不存在: {tool_name}",
                        tool_call_id=tool_id,
                    )
                )
                continue

            tool_result = found_tool.invoke(tool_args)
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_id,
                )
            )

    return str(messages[-1].content)


if __name__ == "__main__":
    run_agent_with_tools("""创建一个功能丰富的 React TodoList 应用：

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
2. 使用 pnpm run dev 启动服务器""")
