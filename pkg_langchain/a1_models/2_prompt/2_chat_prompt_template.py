from langchain_core.prompts import ChatPromptTemplate

chat_prompt_template = ChatPromptTemplate(
    [("system", "你是一个AI开发工程师，你的名字是{name}.")]
)
