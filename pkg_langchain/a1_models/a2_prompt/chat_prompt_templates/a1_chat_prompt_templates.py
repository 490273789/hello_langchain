from langchain_core.prompts import ChatPromptTemplate

from tools import pretty_print

"""
1. 使用ChatPromptTemplate构造方法直接实例化
实例化时需要传入messages: Sequence[MessageLikeRepresentation]
messages 参数支持如下格式：
	tuple 构成的列表，格式为[(role, content)]
	dict 构成的列表，格式为[{“role”:... , “content”:...}]
	Message 类构成的列表
"""
chat_prompt_template = ChatPromptTemplate(
    [
        ("system", "你是一个AI开发工程师，你的名字是{name}。"),
        ("human", "你能帮我做什么?"),
        ("ai", "我能开发很多{thing}。"),
        ("human", "{user_input}"),
    ]
)

prompt1 = chat_prompt_template.format_messages(
    name="小谷AI", thing="AI", user_input="7 + 5等于多少"
)
pretty_print(prompt1, "prompt1 - Constructor")

"""
2. from_messages
作用：将模板变量替换后，直接生成消息列表（List[BaseMessage]），
一般包含：SystemMessage``HumanMessage``AIMessage
常用场景：用于手动查看或调试 Prompt 的最终“消息结构”或者自己拼接进 Chain。

实例化时需要传入messages: Sequence[MessageLikeRepresentation]
messages 参数支持如下格式：
	tuple 构成的列表，格式为[(role, content)]
	dict 构成的列表，格式为[{“role”:... , “content”:...}]
	Message 类构成的列表
"""


# 系统消息定义了AI助手的角色，人类消息定义了用户问题的格式
chat_prompt_template = ChatPromptTemplate.from_messages(
    [("system", "你是一个{role}，请回答我提出的问题"), ("human", "请回答:{question}")]
)

# prompt_value = chat_prompt.format_messages(role="python开发工程师", question="冒泡排序怎么写")
prompt_value = chat_prompt_template.format_messages(
    **{"role": "python开发工程师", "question": "堆排序怎么写"}
)
pretty_print(prompt_value, "prompt_value - format_messages")

print()

# 使用指定的角色和问题参数填充模板，生成具体的提示内容
prompt_value2 = chat_prompt_template.invoke(
    {"role": "python开发工程师", "question": "堆排序怎么写"}
)
# 输出生成的提示内容
pretty_print(prompt_value2.to_string(), "prompt_value2 - invoke - to_string")

print()

prompt_value3 = chat_prompt_template.format(
    **{"role": "python开发工程师", "question": "快速排序怎么写"}
)
# 输出生成的提示内容
pretty_print(prompt_value3, "prompt_value3 - format")
