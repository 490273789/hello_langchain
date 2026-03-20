from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig, RunnableWithMessageHistory

from tools import init_llm_client, pretty_print

prompt = ChatPromptTemplate.from_messages(
    [MessagesPlaceholder(variable_name="history"), ("human", "{input}")]
)

model = init_llm_client()

parse = StrOutputParser()

chain = prompt | model | parse

history = InMemoryChatMessageHistory()

runnable = RunnableWithMessageHistory(
    chain,
    get_session_history=lambda session_id: history,
    input_messages_key="input",
    history_messages_key="history",
)

history.clear()

config = RunnableConfig(configurable={"session_id": "user_001"})

pretty_print("模型思考中。。。")
pretty_print(runnable.invoke({"input": "我叫张三，我爱好学习。"}, config), "回答1")
pretty_print("模型思考中。。。")
pretty_print(runnable.invoke({"input": "我是谁？我爱好什么？"}, config), "回答2")
