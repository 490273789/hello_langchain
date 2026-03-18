from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from tools import init_llm_client, logger, pretty_print, pretty_print_ai

"""
JsonOutputParser，即JSON输出解析器，
是一种用于将大模型的自由文本输出转换为结构化JSON数据的工具。

本案例是：借助JsonOutputParser的get_format_instructions() ，
生成格式说明，指导模型输出JSON 结构
"""


class Person(BaseModel):
    """
    使用pydantic定义一个新闻结构化的数据模型类
    属性:
        time (str): 新闻发生的时间
        person (str): 新闻涉及的人物
        event (str): 发生的具体事件
    """

    time: str = Field(description="时间")  #
    person: str = Field(description="人物")
    event: str = Field(description="事件")


# 创建JSON输出解析器，用于将model输出解析为Person对象
parser = JsonOutputParser(pydantic_object=Person)

# 获取格式化指令，告诉model如何输出符合要求的JSON格式
format_instructions = parser.get_format_instructions()
pretty_print(format_instructions, "parser.get_format_instructions")

# 创建聊天提示模板，定义系统角色和用户输入格式
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个AI助手，你只能输出结构化JSON数据。"),
        ("human", "请生成一个关于{topic}的新闻。{format_instructions}"),
    ]
)

# 格式化提示词，填入具体主题和格式化指令
prompt = chat_prompt.format_messages(
    topic="小米su7跑车", format_instructions=format_instructions
)

pretty_print_ai(prompt, view="message")

model = init_llm_client()

result = model.invoke(prompt)
pretty_print_ai(result, view="message")

# 使用解析器将模型输出解析为结构化数据
response = parser.invoke(result)
pretty_print(response, "解析后的结构化结果")

# 打印类型
logger.info(f"结果类型: {type(response)}")
