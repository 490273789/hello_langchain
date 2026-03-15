import json

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


def _message_content_to_text(content):
    return content if isinstance(content, str) else json.dumps(content)


def count_tokens(messages, encoder):
    total = 0
    for msg in messages:
        content = _message_content_to_text(msg["content"])
        total += len(encoder.encode(content))
    return total
