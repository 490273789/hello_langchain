import json

import tiktoken
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import convert_to_messages, get_buffer_string

from tools import init_llm_client

llm = init_llm_client()

messages = [
    ("human", "我想学做红烧肉，你能教我吗？"),
    (
        "ai",
        "当然可以！红烧肉是一道经典的中式菜肴。首先需要准备五花肉、冰糖、生抽、老抽、料酒等材料。",
    ),
    ("human", "五花肉需要切多大块？"),
    (
        "ai",
        "建议切成3-4厘米见方的块，这样既容易入味，口感也更好。切好后可以用开水焯一下去除血沫。",
    ),
    ("human", "炒糖色的时候有什么技巧吗？"),
    (
        "ai",
        "炒糖色是关键步骤。用小火慢慢炒，等冰糖完全融化变成焦糖色，冒小泡时就可以下肉了。注意不要炒过头，否则会发苦。",
    ),
    ("human", "需要炖多长时间？"),
    (
        "ai",
        "一般需要炖40-60分钟，用小火慢炖，直到肉变得软糯入味。可以用筷子戳一下，能轻松戳透就说明好了。",
    ),
    ("human", "最后收汁的时候要注意什么？"),
    (
        "ai",
        "收汁时要用大火，不断翻动，让汤汁均匀包裹在肉块上。看到汤汁变得浓稠，颜色红亮就可以出锅了。",
    ),
]


def _message_content_to_text(content):
    return (
        content if isinstance(content, str) else json.dumps(content, ensure_ascii=True)
    )


def count_tokens(messages, encoder):
    total = 0
    for msg in messages:
        content = _message_content_to_text(msg[1])
        total += len(encoder.encode(content))
    return total


def summarize_history(messages):
    if not len(messages):
        return ""
    # print(convert_to_messages(messages))
    conversation_text = get_buffer_string(
        convert_to_messages(messages), human_prefix="用户", ai_prefix="助手"
    )
    print(conversation_text)
    summary_prompt = f"""请总结以下对话的核心内容，保留重要信息：

{conversation_text}

总结："""

    summary_response = llm.invoke([("system", summary_prompt)])

    return _message_content_to_text(summary_response.content)


def summarization_memory_demo() -> None:
    history = InMemoryChatMessageHistory()
    max_tokens = 200
    keep_recent_tokens = 80

    enc = tiktoken.get_encoding("cl100k_base")

    history.add_messages(messages)
    history_msgs = history.messages

    total_tokens = count_tokens(history_msgs, enc)

    if total_tokens >= max_tokens:
        recent_messages = []
        recent_tokens = 0

        for i in reversed(history_msgs):
            msg_tokens = len(enc.encode(i[1]))
            if recent_tokens + msg_tokens <= keep_recent_tokens:
                recent_messages.insert(0, i)
                recent_tokens += msg_tokens
            else:
                break
        message_to_summarize = history_msgs[: len(history_msgs) - len(recent_messages)]
        summarize_tokens = count_tokens(message_to_summarize, enc)

        print(
            f"将要被总结的消息数量： {len(message_to_summarize)}长度， {summarize_tokens} tokens"
        )
        print(
            f"将要被保留的消息数量： {len(recent_messages)}长度， {recent_tokens} tokens"
        )

        summary = summarize_history(message_to_summarize)

        history.clear()
        history.add_messages(recent_messages)

        print(f"\n保留消息数量: {recent_messages}")
        print("保留的消息:")
        for m in recent_messages:
            content = _message_content_to_text(m[1])
            tokens = len(enc.encode(content))
            print(f"  {m.__class__.__name__} ({tokens} tokens): {content}")

        print(f"\n总结内容（不包含保留的消息）: {summary}")
    else:
        print(f"\nToken 数量 ({total_tokens}) 未超过阈值 ({max_tokens})，无需总结")


# summarization_memory_demo()
