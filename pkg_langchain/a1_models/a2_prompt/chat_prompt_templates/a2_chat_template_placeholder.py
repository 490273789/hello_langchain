from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools import pretty_print

"""
1. 显式使用MessagesPlaceholder
如果我们不确定消息何时生成，也不确定要插入几条消息，比如在提示词中添加聊天历史记忆这种场景，
可以在ChatPromptTemplate添加MessagesPlaceholder占位符，在调用invoke时，在占位符处插入消息。
"""
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个资深的Python工程师，请回答我提出的的问题",
        ),
        # 插入 memory 占位符，用于填充历史对话记录（如多轮对话上下文）
        MessagesPlaceholder("memory"),
        ("human", "{question}"),
    ]
)

# 调用 prompt.invoke 来格式化整个 Prompt 模板
# - memory：是一组历史消息，表示之前的对话内容（多轮上下文）
prompt_value1 = prompt.invoke(
    {
        "memory": [
            # 用户第一轮说的话
            HumanMessage("我的名字叫亮仔，是一名程序员111"),
            # AI 第一轮的回应
            AIMessage("好的，亮仔你好222"),
        ],
        "question": "请问我的名字叫什么？",
    }
)
pretty_print(prompt_value1.to_string(), "prompt_value1.to_string")


"""
2. "placeholder" 是 ("placeholder", "{memory}") 的简写语法，
等价于 MessagesPlaceholder("memory")。

隐式使用MessagesPlaceholder
"""
prompt = ChatPromptTemplate.from_messages(
    [
        # 占位符，用于插入对话“记忆”内容，即之前的聊天记录（历史上下文）
        ("placeholder", "{memory}"),
        (
            "system",
            "你是一个资深的Python工程师，请回答我提出的Python相关的问题",
        ),
        ("human", "{question}"),
    ]
)

prompt_value2 = prompt.invoke(
    {
        # memory：是之前的对话上下文，会被插入到 {memory} 的位置
        "memory": [
            # 用户第一轮对话
            HumanMessage("我的名字叫亮仔，是一名程序员"),
            # AI 第一轮回答
            AIMessage("好的，亮仔你好"),
        ],
        "question": "请问我的名字叫什么？",
    }
)
# 使用 .to_string() 将格式化后的对话链转换成纯文本字符串，方便查看输出
pretty_print(prompt_value2.to_string(), "prompt_value2.to_string")
