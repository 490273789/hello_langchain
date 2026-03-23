# LangChain Chains 模块学习笔记

## 1. 什么是 Chains（链）

Chains（链）是 LangChain 中用于将多个组件组合在一起的核心概念。它允许我们将 Prompt、LLM、输出解析器等组件串联起来，形成一个完整的处理流程。

### 核心思想
- 将复杂任务分解为多个简单步骤
- 每个步骤的输出作为下一个步骤的输入
- 实现模块化、可复用的处理流程

---

## 2. 传统 Chain 方式（Legacy）

### 2.1 LLMChain

最基础的链类型，将 Prompt 和 LLM 组合在一起：

```python
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

# 创建 prompt
prompt = ChatPromptTemplate.from_template(
    "Give me a small report about {topic}"
)

# 创建 LLM
model = ChatOpenAI(model="gpt-3.5-turbo")

# 创建输出解析器
output_parser = StrOutputParser()

# 创建 LLMChain
chain = LLMChain(
    prompt=prompt,
    llm=model,
    output_parser=output_parser
)

# 运行链
result = chain.run(topic="Artificial Intelligence")
print(result)
```

> ⚠️ **注意**: `LLMChain` 是传统方式，现在推荐使用 LCEL（LangChain Expression Language）

---

## 3. LCEL（LangChain Expression Language）

### 3.1 什么是 LCEL

LCEL 是 LangChain 表达式语言，是一种更现代、更简洁的方式来构建链。它使用 **管道操作符 `|`** 来连接组件。

### 3.2 LCEL 的优势

1. **超快的开发速度** - 更简洁的代码
2. **高级特性支持** - 流式输出、异步、并行执行
3. **易于集成** - 与 LangSmith、LangServe 无缝集成
4. **统一接口** - 所有链都有相同的调用方式

### 3.3 基本语法

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "Tell me a short fact about {topic}"
)
model = ChatOpenAI()
output_parser = StrOutputParser()

# 使用 LCEL 语法（管道操作符）
chain = prompt | model | output_parser

# 调用链
result = chain.invoke({"topic": "Artificial Intelligence"})
print(result)
```

### 3.4 管道操作符 `|` 的原理

管道操作符的工作原理是利用 Python 的 `__or__` 魔术方法：

```python
# 这两种写法是等价的
chain = a | b
chain = a.__or__(b)
```

简单理解：**左边的输出** → **右边的输入**

```python
# 自定义 Runnable 类示例
class Runnable:
    def __init__(self, func):
        self.func = func

    def __or__(self, other):
        def chained_func(*args, **kwargs):
            # 将当前函数的结果传递给下一个函数
            return other(self.func(*args, **kwargs))
        return Runnable(chained_func)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

# 使用示例
def add_five(x):
    return x + 5

def multiply_by_two(x):
    return x * 2

add_five = Runnable(add_five)
multiply_by_two = Runnable(multiply_by_two)

chain = add_five | multiply_by_two
result = chain(3)  # (3 + 5) * 2 = 16
```

---

## 4. Runnable 组件

### 4.1 RunnableLambda

将普通 Python 函数转换为可链接的组件：

```python
from langchain_core.runnables import RunnableLambda

def add_five(x):
    return x + 5

def multiply_by_two(x):
    return x * 2

# 包装函数
add_five = RunnableLambda(add_five)
multiply_by_two = RunnableLambda(multiply_by_two)

# 创建链
chain = add_five | multiply_by_two

# 调用（注意：必须使用 invoke）
result = chain.invoke(3)  # 返回 16
```

**实际应用示例**：自定义输出处理

```python
from langchain_core.runnables import RunnableLambda

def extract_fact(x):
    """提取事实，去掉前缀"""
    if "\n\n" in x:
        return "\n".join(x.split("\n\n")[1:])
    else:
        return x

get_fact = RunnableLambda(extract_fact)

# 在链中使用
chain = prompt | model | output_parser | get_fact
result = chain.invoke({"topic": "AI"})
```

### 4.2 RunnableParallel

并行执行多个操作，常用于 RAG 场景：

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

retriever = vectorstore.as_retriever()

prompt_str = """Answer the question using the context:

Context: {context}

Question: {question}

Answer: """
prompt = ChatPromptTemplate.from_template(prompt_str)

# 并行处理：同时获取 context 和传递 question
retrieval = RunnableParallel(
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
)

chain = retrieval | prompt | model | output_parser
result = chain.invoke("What is LangChain?")
```

### 4.3 RunnablePassthrough

直接传递输入到输出，不做任何修改：

```python
from langchain_core.runnables import RunnablePassthrough

# 将输入直接传递给 "question" 键
retrieval = RunnableParallel(
    {
        "context": retriever,           # 执行检索
        "question": RunnablePassthrough()  # 直接传递原始输入
    }
)
```

### 4.4 多数据源并行检索

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

retriever_a = vectorstore_a.as_retriever()
retriever_b = vectorstore_b.as_retriever()

prompt_str = """Answer the question using the context:

Context:
{context_a}
{context_b}

Question: {question}

Answer: """
prompt = ChatPromptTemplate.from_template(prompt_str)

# 从多个数据源并行检索
retrieval = RunnableParallel(
    {
        "context_a": retriever_a,
        "context_b": retriever_b,
        "question": RunnablePassthrough()
    }
)

chain = retrieval | prompt | model | output_parser
result = chain.invoke("When was James born?")
```

---

## 5. 链的调用方法

LCEL 链支持多种调用方式：

| 方法 | 说明 | 示例 |
|------|------|------|
| `invoke()` | 同步调用，返回完整结果 | `chain.invoke({"topic": "AI"})` |
| `ainvoke()` | 异步调用 | `await chain.ainvoke({"topic": "AI"})` |
| `stream()` | 流式输出 | `for chunk in chain.stream({"topic": "AI"}): print(chunk)` |
| `astream()` | 异步流式输出 | `async for chunk in chain.astream({"topic": "AI"}): print(chunk)` |
| `batch()` | 批量调用 | `chain.batch([{"topic": "AI"}, {"topic": "ML"}])` |
| `abatch()` | 异步批量调用 | `await chain.abatch([{"topic": "AI"}, {"topic": "ML"}])` |

### 流式输出示例

```python
chain = prompt | model | output_parser

# 流式输出
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)
```

---

## 6. 常见 Chain 类型总结

| Chain 类型 | 用途 | LCEL 等价写法 |
|-----------|------|--------------|
| LLMChain | 基础的 Prompt + LLM | `prompt \| model \| output_parser` |
| SequentialChain | 顺序执行多个链 | `chain1 \| chain2 \| chain3` |
| SimpleSequentialChain | 简单顺序链 | `chain1 \| chain2` |
| RouterChain | 根据条件路由到不同链 | 使用 `RunnableBranch` |
| TransformChain | 数据转换 | 使用 `RunnableLambda` |
| RetrievalQA | 检索问答 | `retriever \| prompt \| model` |

---

## 7. 最佳实践

### 7.1 使用 LCEL 而非传统 Chain

```python
# ❌ 传统方式（不推荐）
from langchain.chains import LLMChain
chain = LLMChain(prompt=prompt, llm=model)

# ✅ LCEL 方式（推荐）
chain = prompt | model | output_parser
```

### 7.2 模块化设计

```python
# 将复杂链拆分为可复用的子链
format_chain = prompt | model | output_parser
process_chain = RunnableLambda(process_data)
final_chain = format_chain | process_chain
```

### 7.3 错误处理

```python
from langchain_core.runnables import RunnableLambda

def safe_process(x):
    try:
        return process(x)
    except Exception as e:
        return f"Error: {str(e)}"

safe_chain = chain | RunnableLambda(safe_process)
```

---

## 8. 总结

| 概念 | 说明 |
|------|------|
| **Chains** | 将多个组件串联起来的处理流程 |
| **LCEL** | LangChain 表达式语言，使用 `\|` 操作符构建链 |
| **RunnableLambda** | 将普通函数转为可链接组件 |
| **RunnableParallel** | 并行执行多个操作 |
| **RunnablePassthrough** | 直接传递输入 |
| **invoke/stream/batch** | 链的不同调用方式 |

### 核心要点

1. **LCEL 是现代 LangChain 的核心** - 简洁、强大、统一
2. **管道操作符 `|`** - 左边输出 → 右边输入
3. **Runnable 系列组件** - 灵活组合各种操作
4. **多种调用方式** - 同步、异步、流式、批量
