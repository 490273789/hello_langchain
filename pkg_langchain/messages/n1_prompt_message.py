from langchain_core.prompts import ChatPromptTemplate

from models import qwen

# ChatPromptTemplate  角色设置
# system 系统角色  user human 用户角色 assistant 大模型回复

prompt_template = ChatPromptTemplate(
    [("system", "把用户输入翻译成{language}"), ("user", "{text}")]
)

prompt = prompt_template.format(language="英文", text="你好")

print(f"prompt: {prompt}")

print("-" * 50)
result = qwen.invoke(prompt)
print(f"result: {result}")
