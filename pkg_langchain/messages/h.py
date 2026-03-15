import json
import os

import tiktoken
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string,
)
from langchain_openai import ChatOpenAI

load_dotenv()


model = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
    base_url=os.getenv("OPENAI_BASE_URL"),
)


def _message_content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def count_tokens(messages, encoder) -> int:
    total = 0
    for msg in messages:
        content = _message_content_to_text(msg.content)
        total += len(encoder.encode(content))
    return total


def summarize_history(messages) -> str:
    if not messages:
        return ""

    conversation_text = get_buffer_string(
        messages,
        human_prefix="用户",
        ai_prefix="助手",
    )

    summary_prompt = f"""请总结以下对话的核心内容，保留重要信息：

{conversation_text}

总结："""

    summary_response = model.invoke([SystemMessage(content=summary_prompt)])
    return _message_content_to_text(summary_response.content)


def summarization_memory_demo() -> None:
    history = InMemoryChatMessageHistory()
    max_tokens = 200
    keep_recent_tokens = 80

    enc = tiktoken.get_encoding("cl100k_base")

    messages = [
        {"type": "human", "content": "我想学做红烧肉，你能教我吗？"},
        {
            "type": "ai",
            "content": "当然可以！红烧肉是一道经典的中式菜肴。首先需要准备五花肉、冰糖、生抽、老抽、料酒等材料。",
        },
        {"type": "human", "content": "五花肉需要切多大块？"},
        {
            "type": "ai",
            "content": "建议切成3-4厘米见方的块，这样既容易入味，口感也更好。切好后可以用开水焯一下去除血沫。",
        },
        {"type": "human", "content": "炒糖色的时候有什么技巧吗？"},
        {
            "type": "ai",
            "content": "炒糖色是关键步骤。用小火慢慢炒，等冰糖完全融化变成焦糖色，冒小泡时就可以下肉了。注意不要炒过头，否则会发苦。",
        },
        {"type": "human", "content": "需要炖多长时间？"},
        {
            "type": "ai",
            "content": "一般需要炖40-60分钟，用小火慢炖，直到肉变得软糯入味。可以用筷子戳一下，能轻松戳透就说明好了。",
        },
        {"type": "human", "content": "最后收汁的时候要注意什么？"},
        {
            "type": "ai",
            "content": "收汁时要用大火，不断翻动，让汤汁均匀包裹在肉块上。看到汤汁变得浓稠，颜色红亮就可以出锅了。",
        },
    ]

    for msg in messages:
        if msg["type"] == "human":
            history.add_message(HumanMessage(content=msg["content"]))
        else:
            history.add_message(AIMessage(content=msg["content"]))

    all_messages = history.messages
    total_tokens = count_tokens(all_messages, enc)

    if total_tokens >= max_tokens:
        recent_messages = []
        recent_tokens = 0

        for msg in reversed(all_messages):
            content = _message_content_to_text(msg.content)
            msg_tokens = len(enc.encode(content))

            if recent_tokens + msg_tokens <= keep_recent_tokens:
                recent_messages.insert(0, msg)
                recent_tokens += msg_tokens
            else:
                break

        messages_to_summarize = all_messages[: len(all_messages) - len(recent_messages)]
        summarize_tokens = count_tokens(messages_to_summarize, enc)

        print("\nToken 数量超过阈值，开始总结...")
        print(
            f"将被总结的消息数量: {len(messages_to_summarize)} ({summarize_tokens} tokens)"
        )
        print(f"将被保留的消息数量: {len(recent_messages)} ({recent_tokens} tokens)")

        summary = summarize_history(messages_to_summarize)

        history.clear()
        for msg in recent_messages:
            history.add_message(msg)

        print(f"\n保留消息数量: {len(recent_messages)}")
        print("保留的消息:")
        for m in recent_messages:
            content = _message_content_to_text(m.content)
            tokens = len(enc.encode(content))
            print(f"  {m.__class__.__name__} ({tokens} tokens): {content}")

        print(f"\n总结内容（不包含保留的消息）: {summary}")
    else:
        print(f"\nToken 数量 ({total_tokens}) 未超过阈值 ({max_tokens})，无需总结")


if __name__ == "__main__":
    summarization_memory_demo()
