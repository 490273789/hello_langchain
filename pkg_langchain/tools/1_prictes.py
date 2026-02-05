import dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import (
    InMemorySaver,  # 推荐开发测试使用，不推荐用到生产环境中
)

dotenv.load_dotenv()

llm = init_chat_model(model="gpt-4o-mini")

tavily_search_tool = TavilySearch(max_results=5)

tools = [tavily_search_tool]

config = {"configurable": {"thread_id": 1}}

agent = create_agent(
    model=llm,
    # tools=tools,
    checkpointer=InMemorySaver(),
    system_prompt="你是一个智能助手，能够选择合适的工具帮助用户解决问题。",
)

while True:
    query = input("User:")
    if query == "q":
        break

    res = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
    )
    print(res["messages"][-1].content)
