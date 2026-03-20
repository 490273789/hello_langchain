from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from models import qwen

checkpointer = MemorySaver()

agent = create_agent(
    model=qwen,
    tools=[],
    response_format=any,
    context_schema=[],
    checkpointer=checkpointer,
)
