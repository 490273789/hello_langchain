from langchain_core.prompts import PromptTemplate

from tools import pretty_print

# 创建一个PromptTemplate对象，用于生成格式化的提示词模板
# 模板中包含两个占位符：{role}表示角色，{question}表示问题
template = PromptTemplate.from_template(
    "你是一个专业的{role}工程师，请回答我的问题给出回答，我的问题是：{question}"
)

# 1. invoke() 是LCEL 的统一执行入口，用于执行任意可运行对象（Runnable ）。
# 使用invoke方法填充模板中的占位符，生成具体的提示词
# 参数：字典类型，包含role和question两个键值对
# 返回的是一个 PromptValue 对象，可以用 .to_string() 或 .to_messages() 查看内容
prompt_invoke = template.invoke({"role": "python开发", "question": "冒泡排序怎么写？"})
pretty_print(type(prompt_invoke), "prompt_invoke type")
pretty_print(prompt_invoke, "prompt_invoke")
print()

# 将PromptValue对象转换为字符串并打印
# to_string()方法将PromptValue转换为可读的字符串格式
pretty_print(type(prompt_invoke.to_string()), "prompt_invoke.to_string type")
pretty_print(prompt_invoke.to_string(), "to_string")
print()

pretty_print(type(prompt_invoke.to_messages()), "prompt_invoke.to_messages type")
pretty_print(prompt_invoke.to_messages(), "prompt_invoke.to_messages")
print()

# 2. format() 使用指定的角色和问题参数来格式化模板，生成最终的提示词字符串
prompt_format = template.format(role="python开发", question="二分查找算法怎么写？")
pretty_print(type(prompt_format), "prompt_format type")
pretty_print(prompt_format, "prompt_format")
print()

# 3. partial()方法可以格式化部分变量，并且继续返回一个模板，通常在部分提示词模板场景下使用
partial = template.partial(role="python开发")

# 打印partial对象及其类型信息
pretty_print(type(partial), "partial type")
pretty_print(partial, "partial")
print()

# 使用format方法填充question参数，生成最终的提示词字符串
# 此时所有占位符都已填充完毕，返回完整的提示词文本
prompt_partial = partial.format(question="冒泡排序怎么写？")

# 输出生成的提示词
print(type(prompt_partial), "prompt_partial type")
pretty_print(prompt_partial, "prompt_partial")
