from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from models import qwen


class BaseModel(BaseModel):
    name = str


checkpointer = MemorySaver()

agent = create_agent(
    model=qwen,
    tools=[],
    response_format=BaseModel,
    context_schema=[],
    checkpointer=checkpointer,
)
