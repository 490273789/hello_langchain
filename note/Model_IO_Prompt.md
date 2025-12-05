# LangChain Model I/O - Prompt 学习笔记

## 概述

Model I/O 是 LangChain 中与语言模型交互的核心模块，主要包含三个部分：
- **Prompts（提示词）**: 模板化模型输入
- **Language Models（语言模型）**: 通过通用接口调用语言模型
- **Output Parsers（输出解析器）**: 从模型输出中提取信息

本笔记重点总结 Prompt 相关知识。

---

## Prompt Template 类型总览

LangChain 提供了多种 Prompt Template 类型：

| 模板类型 | 用途 | 适用场景 |
|---------|------|---------|
| `PromptTemplate` | 基础字符串模板 | 简单的文本提示词 |
| `ChatPromptTemplate` | 对话消息模板 | 多轮对话、角色设定 |
| `MessagesPlaceholder` | 消息占位符 | 动态插入对话历史 |
| `FewShotPromptTemplate` | 少样本学习模板 | 提供示例引导模型 |
| `FewShotChatMessagePromptTemplate` | 对话式少样本模板 | 对话场景下的少样本学习 |
| `PipelinePromptTemplate` | 管道模板 | 组合复用多个模板 |
| `HumanMessagePromptTemplate` | 用户消息模板 | 构建用户消息 |
| `SystemMessagePromptTemplate` | 系统消息模板 | 构建系统消息 |
| `AIMessagePromptTemplate` | AI消息模板 | 构建AI消息 |

---

## 1. 消息类型（Messages）

LangChain 支持多种消息类型，用于构建与大模型的对话：

### 1.1 消息类

```python
from langchain.messages import SystemMessage, HumanMessage, AIMessage

# SystemMessage - 系统消息，用于设置AI的行为和角色
system_msg = SystemMessage(content="你是一个智能助手，帮用户解决问题，你的名字叫一把手")

# HumanMessage - 用户消息
human_msg = HumanMessage(content="你是谁？")

# AIMessage - AI的回复消息
ai_msg = AIMessage(content="我是一把手，很高兴为您服务！")
```

### 1.2 消息构建方式

**方式一：使用消息类列表**
```python
from langchain.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="你是一个智能助手"),
    HumanMessage(content="你是谁？"),
]
response = model.invoke(messages)
```

**方式二：使用字典格式（OpenAI 兼容格式）**
```python
messages = [
    {"role": "system", "content": "你是一个智能助手"},
    {"role": "user", "content": "你是谁？"},
]
response = model.invoke(messages)
```

**方式三：直接使用文本（简单场景）**
```python
# 适用于单次独立请求，不需要对话历史
response = model.invoke("写一首关于春天的诗")
```

### 1.3 消息角色说明

| 角色 | 类 | 字典role值 | 说明 |
|------|-----|----------|------|
| 系统 | `SystemMessage` | `system` | 设置AI的行为和角色 |
| 用户 | `HumanMessage` | `user`/`human` | 用户的输入 |
| AI | `AIMessage` | `assistant`/`ai` | AI的回复 |

---

## 2. PromptTemplate - 基础提示词模板

**核心概念**: 模板 + 变量值 = 完整的提示词

### 2.1 创建 PromptTemplate

**方式一：直接实例化**
```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="你是一个翻译助手，帮助用户将{content}翻译成语言：{lang}",
    input_variables=["content", "lang"],
)
```

**方式二：使用类方法 `from_template`（推荐）**
```python
template = PromptTemplate.from_template(
    template="你是一个翻译助手，帮助用户将{content}翻译成语言：{lang}",
)
```

> 💡 使用 `from_template` 方法会自动推断 `input_variables`，无需手动指定。

### 2.2 使用 PromptTemplate

**调用方式一：`format()` 方法**
```python
prompt = template.format(content="你好", lang="法语")
print(prompt)
# 输出：你是一个翻译助手，帮助用户将你好翻译成语言：法语
```

**调用方式二：`invoke()` 方法**
```python
result = template.invoke({"content": "你好", "lang": "法语"})

# 转换为字符串
print(result.to_string())

# 转换为消息列表
print(result.to_messages())
```

### 2.3 模板格式

LangChain 支持两种模板格式：

**f-string 格式（默认）**
```python
template = PromptTemplate.from_template("Hello, {name}!")
```

**Mustache 格式**
```python
template = PromptTemplate.from_template(
    "Hello, {{name}}!",
    template_format="mustache"
)
```

### 2.4 部分变量（Partial Variables）

预先填充部分变量，后续只需提供剩余变量：

```python
template = PromptTemplate(
    template="你是{role}，请用{lang}回答问题：{question}",
    input_variables=["question"],
    partial_variables={"role": "一个AI助手", "lang": "中文"}
)

# 使用时只需提供 question
prompt = template.format(question="什么是机器学习？")
```

---

## 3. ChatPromptTemplate - 对话提示词模板

专门用于构建多轮对话的提示词模板。

### 3.1 创建 ChatPromptTemplate

**方式一：使用元组列表**
```python
from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate([
    ("system", "你是我的小助手，你叫{name}"),
    ("human", "你叫什么名字？")
])
```

**方式二：使用 `from_messages` 方法**
```python
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手"),
    ("human", "请介绍一下{topic}")
])
```

### 3.2 使用 ChatPromptTemplate

```python
# 格式化提示词
prompt = chat_prompt.format_prompt(name="小南")

# 转换为消息列表
messages = prompt.to_messages()
print(messages)
# 输出：[SystemMessage(content='你是我的小助手，你叫小南'), HumanMessage(content='你叫什么名字？')]
```

### 3.3 与模型链式调用

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(model="gpt-4o-mini", model_provider="openai")

# 创建链
chain = chat_prompt | model

# 调用
response = chain.invoke({"topic": "人工智能"})
```

### 3.4 常用角色标识

| 角色标识 | 对应消息类 | 说明 |
|---------|-----------|------|
| `system` | `SystemMessage` | 系统指令 |
| `human`/`user` | `HumanMessage` | 用户输入 |
| `ai`/`assistant` | `AIMessage` | AI回复 |
| `placeholder` | - | 用于插入动态消息列表 |

---

## 4. 消息模板类（Message Prompt Templates）

用于构建单条消息的模板。

### 4.1 HumanMessagePromptTemplate

```python
from langchain_core.prompts import HumanMessagePromptTemplate

human_template = HumanMessagePromptTemplate.from_template(
    "请帮我翻译以下内容：{content}"
)

message = human_template.format(content="Hello World")
print(message)
# 输出：HumanMessage(content='请帮我翻译以下内容：Hello World')
```

### 4.2 SystemMessagePromptTemplate

```python
from langchain_core.prompts import SystemMessagePromptTemplate

system_template = SystemMessagePromptTemplate.from_template(
    "你是一个{role}，专注于{domain}领域"
)

message = system_template.format(role="专家", domain="机器学习")
# 输出：SystemMessage(content='你是一个专家，专注于机器学习领域')
```

### 4.3 AIMessagePromptTemplate

```python
from langchain_core.prompts import AIMessagePromptTemplate

ai_template = AIMessagePromptTemplate.from_template(
    "好的，我来帮你处理关于{topic}的问题"
)

message = ai_template.format(topic="数据分析")
# 输出：AIMessage(content='好的，我来帮你处理关于数据分析的问题')
```

### 4.4 组合使用

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是一个{role}"),
    HumanMessagePromptTemplate.from_template("{question}")
])

messages = chat_prompt.format_messages(role="翻译专家", question="Hello用中文怎么说？")
```

---

## 5. MessagesPlaceholder - 消息占位符

用于在模板中动态插入消息列表，常用于对话历史。

### 5.1 基本用法

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# 调用时传入对话历史
messages = prompt.invoke({
    "history": [
        HumanMessage(content="你好"),
        AIMessage(content="你好！有什么可以帮你的？")
    ],
    "question": "今天天气怎么样？"
})
```

### 5.2 简写语法

```python
# 使用 placeholder 元组简写
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手"),
    ("placeholder", "{history}"),  # 简写形式
    ("human", "{question}")
])
```

### 5.3 可选占位符

```python
# 设置为可选，如果不提供则不会报错
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手"),
    MessagesPlaceholder(variable_name="history", optional=True),
    ("human", "{question}")
])

# 不传 history 也可以正常工作
messages = prompt.invoke({"question": "你好"})
```

### 5.4 完整对话历史示例

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手"),
    MessagesPlaceholder("history"),
    ("human", "{question}")
])

result = prompt.invoke({
    "history": [
        ("human", "5 + 2 等于多少"),
        ("ai", "5 + 2 等于 7")
    ],
    "question": "再乘以 4 呢？"
})

# 输出：
# ChatPromptValue(messages=[
#     SystemMessage(content="你是一个有帮助的助手"),
#     HumanMessage(content="5 + 2 等于多少"),
#     AIMessage(content="5 + 2 等于 7"),
#     HumanMessage(content="再乘以 4 呢？"),
# ])
```

---

## 6. FewShotPromptTemplate - 少样本学习模板

通过提供示例来引导模型学习特定的输入输出模式。

### 6.1 基本用法

```python
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

# 定义示例
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "sunny", "output": "cloudy"},
]

# 定义单个示例的格式模板
example_prompt = PromptTemplate(
    template="输入: {input}\n输出: {output}",
    input_variables=["input", "output"]
)

# 创建 FewShotPromptTemplate
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="给出每个输入词的反义词。",
    suffix="输入: {word}\n输出:",
    input_variables=["word"]
)

prompt = few_shot_prompt.format(word="big")
print(prompt)
```

**输出：**
```
给出每个输入词的反义词。

输入: happy
输出: sad

输入: tall
输出: short

输入: sunny
输出: cloudy

输入: big
输出:
```

### 6.2 使用 Example Selector 动态选择示例

根据输入动态选择最相关的示例：

```python
from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 更多示例
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
    {"input": "2+4", "output": "6"},
    {"input": "What is 2+2?", "output": "4"},
    {"input": "What is 2+3?", "output": "5"},
]

# 创建语义相似度选择器
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    Chroma,
    k=2  # 选择最相似的2个示例
)

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="你是一个计算器。",
    suffix="输入: {input}\n输出:",
    input_variables=["input"]
)
```

---

## 7. FewShotChatMessagePromptTemplate - 对话式少样本模板

专门为 Chat 模型设计的少样本学习模板。

### 7.1 固定示例

```python
from langchain_core.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate
)

# 定义示例
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
]

# 定义示例格式（对话形式）
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

# 创建少样本模板
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)

# 组合成完整的对话模板
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个数学助手"),
    few_shot_prompt,
    ("human", "{input}")
])

# 使用
messages = final_prompt.format_messages(input="4+4等于多少？")
```

### 7.2 动态示例选择

```python
from langchain_core.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain_core.example_selectors import SemanticSimilarityExampleSelector

# 使用语义相似度选择器
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embeddings,
    vectorstore_cls,
    k=2
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    input_variables=["input"],
    example_selector=example_selector,
    example_prompt=(
        HumanMessagePromptTemplate.from_template("{input}") +
        AIMessagePromptTemplate.from_template("{output}")
    ),
)

# 完整模板
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手"),
    few_shot_prompt,
    ("human", "{input}")
])
```

---

## 8. PipelinePromptTemplate - 管道模板

用于组合和复用多个提示词模板。

### 8.1 基本概念

PipelinePromptTemplate 允许将多个模板组合在一起，前一个模板的输出可以作为后一个模板的输入。

### 8.2 使用示例

```python
from langchain_core.prompts import PromptTemplate, PipelinePromptTemplate

# 定义子模板
introduction_template = PromptTemplate.from_template(
    "你是一个{role}。"
)

example_template = PromptTemplate.from_template(
    """这是一个示例对话：
用户：{example_q}
助手：{example_a}"""
)

start_template = PromptTemplate.from_template(
    """现在，请回答用户的问题。
用户：{input}
助手："""
)

# 最终模板
final_template = PromptTemplate.from_template(
    """{introduction}

{example}

{start}"""
)

# 创建管道模板
pipeline_prompt = PipelinePromptTemplate(
    final_prompt=final_template,
    pipeline_prompts=[
        ("introduction", introduction_template),
        ("example", example_template),
        ("start", start_template)
    ]
)

# 使用
prompt = pipeline_prompt.format(
    role="翻译专家",
    example_q="Hello",
    example_a="你好",
    input="Good morning"
)
```

**输出：**
```
你是一个翻译专家。

这是一个示例对话：
用户：Hello
助手：你好

现在，请回答用户的问题。
用户：Good morning
助手：
```

### 8.3 应用场景

- 模板复用：将常用的模板片段独立出来
- 模块化管理：不同部分可以独立修改
- 条件组合：根据场景选择不同的子模板

---

## 9. 模型调用方式

### 4.1 获取模型实例

```python
from langchain.chat_models import init_chat_model

# 调用 OpenAI
model = init_chat_model(model="gpt-4o-mini", model_provider="openai")

# 调用 DeepSeek
model = init_chat_model(model="deepseek-chat", model_provider="deepseek")
```

### 4.2 调用方式对比

| 调用方式 | 方法 | 特点 |
|---------|------|------|
| 非流式调用 | `invoke()` | 等待完整响应返回 |
| 流式调用 | `stream()` | 返回生成器，逐步获取响应 |
| 批次调用 | `batch()` | 多线程并行处理多个请求 |
| 异步调用 | `ainvoke()` | 协程方式，单线程高并发 |

**非流式调用**
```python
response = model.invoke(messages)
print(response.content)
```

**流式调用**
```python
for chunk in model.stream("你是谁？"):
    print(chunk.content, end="")
```

**批次调用**
```python
messages_list = [
    [{"role": "user", "content": "写一句关于春天的诗。"}],
    [{"role": "user", "content": "写一句关于夏天的诗。"}],
    [{"role": "user", "content": "写一句关于秋天的诗。"}],
]
responses = model.batch(messages_list)
```

**异步调用**
```python
import asyncio

async def gather_task(messages: list):
    tasks = [model.ainvoke(message) for message in messages]
    res = await asyncio.gather(*tasks)
    return res

await gather_task(messages_list)
```

---

## 10. 动态提示词（高级用法）

在 LangChain Agent 中，可以使用中间件动态生成系统提示词：

```python
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from typing import TypedDict

class Context(TypedDict):
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """根据用户角色生成不同的系统提示词"""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "你是一个有帮助的助手。"
    
    if user_role == "expert":
        return f"{base_prompt} 请提供详细的技术响应。"
    elif user_role == "beginner":
        return f"{base_prompt} 请用简单的语言解释，避免使用术语。"
    
    return base_prompt

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[user_role_prompt],
    context_schema=Context
)
```

---

## 11. 最佳实践

### 11.1 提示词设计原则

1. **明确角色**: 使用 SystemMessage 清晰定义 AI 的角色和行为
2. **具体指令**: 提供具体、清晰的任务指令
3. **提供示例**: 对于复杂任务，可以提供少量示例（Few-shot learning）
4. **设置边界**: 明确告诉 AI 不应该做什么

### 11.2 模板使用建议

1. 使用 `from_template` 方法创建模板，自动推断变量
2. 对于多轮对话，使用 `ChatPromptTemplate`
3. 将常用提示词定义为常量，便于复用

```python
TRANSLATOR_PROMPT = """
你是一个专业的翻译助手。
请将以下内容翻译成{target_lang}：

{content}

要求：
1. 保持原意
2. 使用地道的表达
3. 注意语法正确
"""

template = PromptTemplate.from_template(TRANSLATOR_PROMPT)
```

### 11.3 安全考虑

对于涉及数据库操作等敏感任务，在提示词中明确安全约束：

```python
system_prompt = """
你是一个数据库查询助手。
注意：禁止执行任何 DML 语句（INSERT, UPDATE, DELETE, DROP 等）。
在执行查询前，请先检查表结构。
"""
```

---

## 12. 常用类和方法速查

| 类/方法 | 用途 |
|---------|------|
| `PromptTemplate` | 创建基础字符串提示词模板 |
| `ChatPromptTemplate` | 创建对话提示词模板 |
| `MessagesPlaceholder` | 动态插入消息列表（如对话历史） |
| `FewShotPromptTemplate` | 少样本学习模板（字符串版） |
| `FewShotChatMessagePromptTemplate` | 少样本学习模板（对话版） |
| `PipelinePromptTemplate` | 组合多个模板 |
| `HumanMessagePromptTemplate` | 构建用户消息模板 |
| `SystemMessagePromptTemplate` | 构建系统消息模板 |
| `AIMessagePromptTemplate` | 构建AI消息模板 |
| `SystemMessage` | 系统消息 |
| `HumanMessage` | 用户消息 |
| `AIMessage` | AI消息 |
| `template.format()` | 填充变量，返回字符串 |
| `template.invoke()` | 填充变量，返回 PromptValue |
| `prompt.to_string()` | 转换为字符串 |
| `prompt.to_messages()` | 转换为消息列表 |

---

## 参考资料

- [LangChain 官方文档](https://python.langchain.com/)
- [LangChain Prompt Templates](https://python.langchain.com/docs/modules/model_io/prompts/)
