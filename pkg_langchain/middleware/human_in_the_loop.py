import models
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
# HumanInTheLoopMiddleware
# 人机交互中间件：在工具调用之前执行，暂停代理执行，以便人工审批、编辑或拒绝工具调用

agent = create_agent(
    model=models.qwen,
    tools=[],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file":True,
                "execute_sql":{"allowed_decisions": ["approve", "reject"]},
                "read_data": False
            }
        )
    ]
)