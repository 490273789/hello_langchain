from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

# 1. 初始化模型
# 这里的 model 用于对话，summary_model 用于生成摘要（可以用同一个）
model = ChatOpenAI(model="gpt-4o")


# 2. 定义状态 (State)
# 我们继承 MessagesState，它自动包含 'messages' 列表
# 我们额外增加一个 'summary' 字段来存储摘要
class State(MessagesState):
    summary: str


# 3. 定义节点逻辑


def call_model(state: State):
    """主对话节点：调用模型回复用户"""
    summary = state.get("summary", "")
    messages = state["messages"]

    # 如果有摘要，将其作为 SystemMessage 插入到上下文最前面
    if summary:
        system_msg = SystemMessage(content=f"以下是之前的对话摘要：{summary}")
        # 注意：这里只是临时组合给模型看，不会存入 state["messages"]
        messages_with_summary = [system_msg] + messages
    else:
        messages_with_summary = messages

    response = model.invoke(messages_with_summary)
    # 返回的内容会自动 append 到 state["messages"]
    return {"messages": [response]}


def summarize_conversation(state: State):
    """总结节点：压缩旧消息"""
    summary = state.get("summary", "")
    messages = state["messages"]

    # 策略：保留最后 2 条消息（一问一答），总结其余所有的
    if summary:
        summary_message = (
            f"这是目前的对话摘要: {summary}\n\n"
            "请结合上面的摘要，将以下新的对话内容延伸到摘要中："
        )
    else:
        summary_message = "请将以下对话总结为一个简短的摘要："

    # 拿出除了最后 2 条之外的所有消息进行总结
    messages_to_summarize = messages[:-2]

    # 调用模型生成新摘要
    prompt = ChatPromptTemplate.from_messages(
        [("system", summary_message), MessagesPlaceholder(variable_name="history")]
    )
    chain = prompt | model
    response = chain.invoke({"history": messages_to_summarize})

    # 构建删除操作：删除由于已经总结而不再需要的旧消息
    # RemoveMessage 是 LangGraph 特有的更新操作
    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize]

    return {"summary": response.content, "messages": delete_messages}


def should_summarize(state: State) -> Literal["summarize_conversation", END]:
    """条件边：判断是否需要总结"""
    messages = state["messages"]

    # 模拟 max_token_limit，这里简单用消息数量 > 6 来演示
    # 如果消息超过 6 条，就去总结；否则结束本次运行
    if len(messages) > 6:
        return "summarize_conversation"
    return END


# 4. 构建图 (Graph)
workflow = StateGraph(State)

# 添加节点
workflow.add_node("conversation", call_model)
workflow.add_node("summarize_conversation", summarize_conversation)

# 定义边
# START -> 对话
workflow.add_edge(START, "conversation")

# 对话 -> 判断是否总结 -> (总结 或 结束)
workflow.add_conditional_edges(
    "conversation",
    should_summarize,
)

# 总结 -> 结束
workflow.add_edge("summarize_conversation", END)

# 5. 编译图，并挂载记忆保存器 (Checkpointer)
# MemorySaver 替代了旧代码中的 memory 对象，用于持久化 thread_id 对应的状态
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# 6. 运行对话循环
print("开始对话 (输入 'q' 退出)...")
thread_id = "thread-1"  # 用于标识当前会话 ID
config = {"configurable": {"thread_id": thread_id}}

while True:
    user_input = input("Human: ")
    if user_input.lower() == "q":
        break

    # 在 LangGraph 中，我们直接把新消息传入图
    input_message = HumanMessage(content=user_input)

    # stream 模式
    print("AI: ", end="", flush=True)
    # stream_mode="values" 会流式返回 state 的更新，我们只打印最后一条 AI 的回复
    final_response = None

    # 这里为了简单演示流式效果，我们只取 call_model 的输出
    # 实际生产中可以使用 app.stream_events 来获得更细粒度的 token 流
    for event in app.stream({"messages": [input_message]}, config=config):
        if "conversation" in event:
            final_response = event["conversation"]["messages"][-1].content

    if final_response:
        print(final_response)

    # 调试：查看当前的 State 状态（看看消息是否被删除了，摘要是否更新了）
    snapshot = app.get_state(config)
    print(
        f"\n[System Info] 当前消息数: {len(snapshot.values['messages'])}, 摘要长度: {len(snapshot.values.get('summary', ''))}"
    )
    print("-" * 30)
