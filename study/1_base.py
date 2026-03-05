from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from pydantic import BaseModel, Field

from models import qwen
from tools.format import pretty_print_response

# from langchain_core.tools import tool
# from langchain_core.messages import HumanMessage,AIMessage,ToolMessage

# model = init_chat_model(model="deepseek-chat", model_provider="deepseek")


class ReadFileArgs(BaseModel):
    file_path: str = Field(description="'要读取的文件路径'")


@tool(
    "read_file",
    args_schema=ReadFileArgs,
    description="用此工具来读取文件内容。当用户要求读取文件、查看代码、分析文件内容时，调用此工具。输入文件路径（可以是相对路径或绝对路径）。",
)
def read_file(file_path: str):
    with open(file_path, "r") as file:
        content = file.read()
        print(f"[tool] read_file({file_path}) - read {len(content)} bytes")
        return content


tools = [read_file]

messages = [
    SystemMessage(""" 你是一个代码助手，可以用工具读取文件并解释代码。

工作流程：
1. 用户要求读取文件时，立即调用 read_file 工具
2. 等待工具返回文件内容
3. 基于问价内容进行分析和解释

可用工具：
- read_file: 读取文件内容（使用此工具来获取文件内容）
"""),
    HumanMessage("请读取 test/ai.txt 文件内容并输出"),
]


def invoke(is_agent: bool = False):
    if is_agent:
        agent = create_agent(model=qwen, tools=tools, middleware=[])
        response = agent.invoke({"messages": messages})
    else:
        model = qwen.bind_tools(tools)
        response = model.invoke(messages)
    return response


print(invoke())
pretty_print_response(invoke(), view="zen")
