"""
字符串解析器StrOutputParser
它是LangChain中最简单的输出解析器，它可以简单地将任何输入转换为字符串。
从结果中提取content字段转换为字符串输出。
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from tools import init_llm_client, logger, pretty_print, pretty_print_ai

# 创建聊天提示模板，包含系统角色设定和用户问题输入
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个{role}，请简短回答我提出的问题"),
        ("human", "请回答:{question}"),
    ]
)

# 使用指定的角色和问题生成具体的提示内容
prompt = chat_prompt.invoke(
    {"role": "AI助手", "question": "什么是LangChain，简洁回答100字以内"}
)

pretty_print(prompt, "prompt")

# 初始化聊天模型
model = init_llm_client()

# 调用模型获取回答结果
result = model.invoke(prompt)
logger.info(f"模型原始输出:\n{result}")
pretty_print_ai(result, view="message")
# 创建字符串输出解析器，用于解析模型返回的结果
parser = StrOutputParser()

# 打印解析后的结构化结果
response = parser.invoke(result)
pretty_print(response, "解析后的结构化结果")
logger.info("\n")
pretty_print(type(response), "结果类型")
