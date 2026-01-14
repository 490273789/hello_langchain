
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


from langchain_core.prompts import ChatPromptTemplate
import models

# ChatPromptTemplate  角色设置
# system 系统角色  user human 用户角色 assistant 大模型回复

prompt_template = ChatPromptTemplate([
    ("system", "把用户输入翻译成{language}"),
    ("user", "{text}")
])

prompt = prompt_template.format(language="英文", text="你好")

print(f"prompt: {prompt}")

print("-" * 50)
print(models.models)
result = models.models.qwen.invoke(prompt)
print(f"result: {result}")