# Model

model I/O: 与大模型进行交互的核心组件，标准化各个大模型的输入和输出，包含输入模版，模型本身和格式化输出

## 三件套
输入、处理、输出
### 输入提示 - Format
指代Prompts Template 提示词模版，通过模版管理大模型的输入输出，将原始数据格式化成模型可以处理的形式，插入到一个模版问题中心，然后送入模型进行处理

### 调用模型 - Predict
指代models，使用通用接口调用不同的大语言模型。接受被送进来的问题，然后基于这个问题进行预测或生成回答

### 输出解析 - Parse
指代Output Parser，用来从模型的推理中提取信息，并按照预先设定好的模版来规范化输出，比如，格式化成一个结构化的json对象

## 模型分类
### LLM - Large Language Model
- 输出形式：纯文本字符串
- 输出形式：文本字符串
- 主要特点
    - 最基础的文本模型
    - 无上下文记忆
    - 高速轻量
- 典型适用场景：单论问答、摘要生成、文本改写、指令执行

### 对话模型
- 输出形式：消息列表（List[BaseMessage]），包括HumanMessage\AIMessage\SystemMessage\ToolMessage
- 输出形式：AIMessage
- 主要特点
    - 面向对话场景
    - 支持多轮上下文
    - 更贴近人类对话逻辑
- 典型适用场景：智能助手、、文本改写、指令执行

### 向量模型
- 输出形式：纯文本字符串
- 输出形式：文本字符串
- 主要特点
    - 最基础的文本模型
    - 无上下文记忆
    - 高速轻量
- 典型适用场景：单论问答、摘要生成、文本改写、指令执行


## 创建模型

参数名
- model: 指定使用的大语言模型名称(如"gpt-4"、等gpt-3.5-turbo")
- temperature: 温度，温度越高，输出内容越随机;温度越低，输出内容越确定
- timeout: 请求超时时间
- max_tokens: 生成内容的最大token数
- stop: 模型在生成时遇到这些"停止词”将立刻停止生成，常用于控制输出的边界。
- max_retries: 最大重试请求次数
- api_key: 大模型供应商提供的API秘钥
- base_url: 大模型供应商API请求地址

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="qwen-flash",
    model_provider="openai",
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)
```

## 调用模型
### 普通调用
invoke: 普通调用，处理单条输入，等待LLM完全推理完成后再返回调用结果
ainvoke: 在 异步环境（async/await）中高效并行地执行模型推理。它的核心作用是：让你同时调用多个模型请求而不阻塞主线程

### 流式调用
stream: 一种逐步返回大模型生成结果的技术，生成一点返回一点，允许服务器将响应内容分批次实时传输给客户端，而不是等待全部内容生成完毕后再一次性返回
astream: 异步流式响应

### 批处理
batch: 处理批量输入，一次性向模型提交多个输入并并行处理，从而显著提升吞吐量
abatch: 异步处理批量输入

## Message组件
SystemMessage
- 系统提示词
HumanMessage
- 人类发送的消息类型
AIMessage
- 大模型回复的消息类型
ToolMessage
- 工具消息类型

## 提示词 - Prompt
### 分类
#### PromptTemplate - 文本提示词模版
针对文本生成模型的提示词模板，也是LangChain提供的最基础的模板，通过格式化字符串生成提示词，在执行invoke时将变量格式化到提示词模板中。

PromptTemplate()主要参数：
- template：定义提示词模板的字符串，其中包含文本和变量占位符（如{name}） ；
- input_variables： 列表，指定了模板中使用的变量名称，在调用模板时被替换；
- partial_variables：字典，用于定义模板中一些固定的变量名。这些值不需要再每次调用时被替换；

PromptTemplate.from_template()参数：
- 字符串模版

函数方法：
- format()：给input_variables变量赋值，并返回提示词。利用format() 进行格式化时就一定要赋值，否则会报错。当在template中未设置input_variables，则会自动忽略。
- invoke：格式化提示词模板为PromptValue，返回的是一个 PromptValue 对象，可以用 .to_string() 或 .to_messages() 查看内容
- partial：格式化提示词模板为一个新的提示词模板，可以继续进行格式化

#### ChatPromptTemplate - 对话提示词模版
ChatPromptTemplate 是 LangChain 中专门用于**结构化聊天对话提示**的核心组件，它比普通 `PromptTemplate` 更适合处理多角色、多轮次的对话场景。为与现代聊天模型的交互提供了一种上下文丰富和会话友好的方式

参数：
- 列表参数格式是tuple类型(role: "ai | human | system", content: "你好")
- 元组的格式为：("ai | human | system", "你好")

ChatMessagePromptTemplate

SystemMessagePromptTemplate

HumanMessagePromptTemplate

AIMessagePromptTemplate

#### FewShotPromptTemplate - 少样本学习提示词模板
构建一个Prompt其中包含多个示例，可以自动将这些示例格式化并插入到主Prompt 中形成样本提示模板，通过在给模型的最终输入中筛入一些示例，来教模型如何回答

#### PipelinePrompt - 管道提示词模版
管道提示词模板，用于把几个提示词组合在一起使用