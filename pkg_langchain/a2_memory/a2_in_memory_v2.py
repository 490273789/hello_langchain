from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig, RunnableWithMessageHistory

from tools import init_llm_client, logger, pretty_print

prompt = ChatPromptTemplate.from_messages(
    [MessagesPlaceholder(variable_name="history"), ("human", "{input}")]
)

model = init_llm_client()

parse = StrOutputParser()

chain = prompt | model | parse

# 定义全局的“会话存储”，用来保存每个 session 的聊天历史
#    （真实项目中可改为 Redis、SQLite 等）
store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个友好的中文助理，会根据上下文回答问题。"),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

runnable = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


config = RunnableConfig(configurable={"session_id": "user_001"})


pretty_print("我叫张三，我爱好学习。", "提问1")
logger.info("模型思考中。。。")
pretty_print(runnable.invoke({"input": "我叫张三，我爱好学习。"}, config), "回答1")

pretty_print("我是谁？我爱好什么？", "提问2")
logger.info("模型思考中。。。")
pretty_print(runnable.invoke({"input": "我是谁？我爱好什么？"}, config), "回答2")
