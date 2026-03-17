from models import qwen
from langchain.agents import create_agent
from typing import TypedDict
from langchain.agents.middleware import dynamic_prompt, ModelRequest


class Context(TypedDict):
    user_role: str


@dynamic_prompt
def user_role_prompt(request: ModelRequest):
    """根据用户角色生成系统提示词"""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "你是一个有用的助手"
    print(f"user_role: {user_role}")
    if user_role == "expert":
        return f"{base_prompt} 提供详细的技术回应。"
    elif user_role == "beginner":
        return f"{base_prompt} 简单的解释概念，避免行话。"

    return base_prompt


agent = create_agent(
    model=qwen, tools=[], middleware=[user_role_prompt], context_schema=Context
)

result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "解释一下什么是机器学习？"}],
    },
    context={"user_role": "expert"},
)
print(f"result: {result}")