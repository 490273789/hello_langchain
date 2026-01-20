from uuid import uuid4

from deepagents import create_deep_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from models import qwen


# 定义工具
@tool
def delete_file(path: str) -> str:
    """从文件系统删除一个文件."""
    print("Tool: delete_file")
    return f"已删除 {path}"


@tool
def read_file(path: str) -> str:
    """从文件系统读取一个文件."""
    print("Tool: read_file")
    return f"内容 {path}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件."""
    print("Tool: send_email")
    return f"发送邮件到 {to}"


# HumanInTheLoopMiddleware
# 人机交互中间件：在工具调用之前执行，暂停代理执行，以便人工审批、编辑或拒绝工具调用

# agent = create_agent(
#     model=qwen,
#     tools=[delete_file, read_file, send_email],
#     middleware=[
#         HumanInTheLoopMiddleware(
#             interrupt_on={
#                 "delete_file": True,
#                 "read_file": False,
#                 "send_email": {"allowed_decisions": ["approve", "reject"]},
#             }
#         )
#     ],
# )

# 必须支持多轮会话
checkpointer = MemorySaver()

deep_agent = create_deep_agent(
    model=qwen,
    tools=[delete_file, read_file, send_email],
    interrupt_on={
        "delete_file": True,  # default: approve edit reject
        "read_file": False,
        "send_email": {"allowed_decisions": ["approve", "reject"]},  # custom
    },
    checkpointer=checkpointer,
)

# 创建config 带thread_id 保持会话状态
config = {"configurable": {"thread_id": str(uuid4())}}

result = deep_agent.invoke(
    {"messages": [{"role": "user", "content": "删除当前目录下的文件 temp.txt"}]},
    config=config,
)

if result.get("__interrupt__"):
    # 了解内部机制------------------
    print("-" * 10)
    print(result)
    # 提取 中断 信息
    interrupts = result["__interrupt__"][0].value
    action_requests = interrupts["action_requests"]
    review_configs = interrupts["review_configs"]

    # 创建一个从工具名称到检查配置的查找映射
    config_map = {cfg["action_name"]: cfg for cfg in review_configs}

    for action in action_requests:
        review_config = config_map[action["name"]]
        print(f"Tool: {action['name']}")
        print(f"Arguments: {action['args']}")
        print(f"Allowed decisions: {review_config['allowed_decisions']}")
    # --------------------------
    # 获取用户决策（每个action_request一个， 按顺序）
    decisions = [
        {"type": "approve"},  # 用户同意删除
    ]

    # 通过决策恢复执行
    result = deep_agent.invoke(Command(resume={"decisions": decisions}), config=config)

    print()
    print(f"result: {result}")
    print(result["messages"][-1].content)


# 多次工具调用
#   当Agent调用多个需要审批的工具时，所有中断都会被合并成一个中断处理。你必须按顺序对每个中断做出决策。
