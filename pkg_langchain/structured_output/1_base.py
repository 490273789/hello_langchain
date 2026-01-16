from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel

from models import qwen


class ContractInfo(BaseModel):
    name: str
    email: str
    phone: str


agent = create_agent(model=qwen, tools=[], response_format=ToolStrategy(ContractInfo))


result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "从：Jeff，jeff@123.com, 16263668888 提取联系信息",
            }
        ]
    }
)

print(result)
# print(type(result["structured_response"]))  # <class '__main__.ContactInfo'>
print(
    result["structured_response"]
)  # name='Jeff' email='jeff@123.com' phone='16263668888'
