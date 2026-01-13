from operator import itemgetter
from langchain.chat_models import init_chat_model
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages

# 初始化模型
model = init_chat_model(model="gpt-4o", model_provider="openai")

# 1. 创建提示模板
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是聊天机器人，请根据对应的上下文回复用户问题"),
        MessagesPlaceholder("history"),
        ("human", "{query}"),
    ]
)

# 2. 定义消息修剪器 (替代 BufferWindowMemory)
# 现代 LangChain 推荐使用 trim_messages 来管理上下文长度
# 这会保留最新的 max_tokens 内容，自动丢弃旧消息
trimmer = trim_messages(
    max_tokens=300,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# 3. 构建基础链
# 使用 RunnablePassthrough.assign 在 prompt 之前修剪 history
# RunnableWithMessageHistory 会注入完整的 history，我们在这里进行修剪
chain = (
    RunnablePassthrough.assign(history=itemgetter("history") | trimmer)
    | prompt
    | model
    | StrOutputParser()
)

# 4. 管理聊天记录
# 使用 InMemoryChatMessageHistory 在内存中存储记录
store = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 使用 RunnableWithMessageHistory 包装基础链
# 它会自动处理 history 的读取和写入
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="query",
    history_messages_key="history",
)

# 5. 对话循环
print("开始对话 (输入 'q' 退出):")
session_id = "user_1"

while True:
    query = input("Human: ")

    if query == "q":
        break

    # 调用链
    response = chain_with_history.stream(
        {"query": query}, config={"configurable": {"session_id": session_id}}
    )

    print("AI: ", flush=True, end="")
    for chunk in response:
        print(chunk, flush=True, end="")
    print("\n")
