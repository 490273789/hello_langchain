# 方式1：使用构造方法实例化提示词模板
from langchain_core.prompts import PromptTemplate

from tools import pretty_print

# 1. 使用构造器
# 创建一个PromptTemplate对象，用于生成格式化的提示词模板
# 该模板包含两个变量：role（角色）和question（问题）
template1 = PromptTemplate(
    template="你是一个专业的{role}工程师，请回答我的问题给出回答，我的问题是：{question}",
    input_variables=["role", "question"],
)

# 使用模板格式化具体的提示词内容
# 将role替换为"python开发"，question替换为"冒泡排序怎么写？"
prompt1 = template1.format(
    role="python开发", question="冒泡排序怎么写,只要代码其它不要，简洁"
)
pretty_print(prompt1, "构造器")

# 2. 使用from_template方法
# 创建一个PromptTemplate对象，用于生成格式化的提示词模板
# 模板包含两个占位符：{role}表示角色，{question}表示问题
template2 = PromptTemplate.from_template(
    "你是一个专业的{role}工程师，请回答我的问题给出回答，我的问题是：{question}"
)

# 使用指定的角色和问题参数来格式化模板，生成最终的提示词字符串
# role: 工程师角色描述
# question: 具体的技术问题
prompt2 = template2.format(role="python开发", question="快速排序怎么写？")
pretty_print(prompt2, "from_template")

# 3. 组合两种方式
template3 = (
    PromptTemplate.from_template("请用一句话介绍{topic}，要求通俗易懂\n")
    + "内容不超过{length}个字"
)
# 使用format方法填充模板中的占位符，生成具体的提示词
prompt3 = template3.format(topic="LangChain", length=100)
pretty_print(prompt3, "prompt3 组合方式")

# 分别创建两个独立的PromptTemplate模板
prompt_a = PromptTemplate.from_template("请用一句话介绍{topic}，要求通俗易懂\n")
prompt_b = PromptTemplate.from_template("内容不超过{length}个字")
# 将两个模板进行拼接组合
prompt_all = prompt_a + prompt_b
# 填充组合后模板的占位符，生成最终的提示词
prompt4 = prompt_all.format(topic="LangChain", length=200)
pretty_print(prompt4, "prompt4 组合方式")
