from time import sleep
from typing import Any
from openai import OpenAI


class ConversationSummaryBufferMemory:
    """摘要缓冲混合记忆类"""

    # 1. max_tokens 用于判断是否需要生成新的摘要
    # 2. summary 用于存储摘要的信息
    # 3. chat_histories 用于存储历史对话
    # 4. get_num_token 用于计算传入的文本的token数
    # 5. save_context 用于存储新的交流对话
    # 6. get_buffer_string 用于将历史对话转换为字符串
    # 7. load_memory_variables 用于加载记忆变量信息
    # 8. summary_text 用于将旧的摘要和传入的对话生成新的摘要

    def __init__(
        self, summary: str = "", chat_histories: list = None, max_token: int = 300
    ):
        self.summary = summary
        self.chat_histories = chat_histories
        self.max_token = max_token
        self._client = OpenAI()

    @classmethod
    def get_num_tokens(cls, query: str) -> int:
        """用于计算传入的文本的token数"""
        return len(query)

    def save_context(self, human_query: str, ai_content: str) -> None:
        """
        存储新的交流对话
        在大模型回复，后处理，下次提问就可以直接使用了
        """
        self.chat_histories.append({"human": human_query, "ai": ai_content})
        buffer_string = self.get_buffer_string()

        tokens = self.get_num_tokens(buffer_string)

        # 如果历史对话的tokens超过最大值，将部分历史对话转化为摘要
        if tokens > self.max_token:
            print("摘要生成中")
            first_chat = self.chat_histories[0]
            self.summary = self.summary_text(
                self.summary,
                f"Human: {first_chat.get('human')}\nAI: {first_chat.get('ai')}",
            )
            print(f"新生成的摘要: {self.summary}")
            del self.chat_histories[0]

    def get_buffer_string(self) -> str:
        """将历史对话转换为字符串"""
        buffer: str = ""
        for chat in self.chat_histories:
            buffer += f"Human: {chat.get('human')}\nAI: {chat.get('ai')}\n\n"
        return buffer.strip()

    def load_memory_variables(self) -> dict[str, Any]:
        """加载记忆变量信息"""
        buffer_string = self.get_buffer_string()
        return {"chat_history": f"摘要：{self.summary}\n\n历史消息：{buffer_string}\n"}

    def summary_text(self, origin_summary: str, new_line: str) -> str:
        """将旧的摘要和传入的对话生成新的摘要"""
        prompt = f"""你是一个强大的聊天机器人，请根据用户提供的谈话内容，总结摘要，并将其添加到先前提供的摘要中，返回一个新的摘要。除了新摘要其他任何数据都不要生成。
请不要将<example>标签俩面的数据当成实际的数据，这里的数据只是一个示例数据，告诉你如何生成新摘要
<example>
当前摘要：人类会问人工智能对人工智能的看法，人工智能认为人工智能是一股善良的力量

新的对话：
Human：为什么会认为人工智能是一股善良的力量？
AI：因为人工智能会帮助人类充分发挥潜力。

新摘要：人类会问人工智能对人工智能的看法，人工智能认为人工智能是一股善良的力量，因为人工智能会帮助人类充分发挥潜力。
</example>

当前摘要：{origin_summary}
新的对话：{new_line}

新摘要："""

        summary = self._client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}]
        )
        return summary.choices[0].message.content


client = OpenAI()

memory = ConversationSummaryBufferMemory("", [], 300)

while True:
    query = input("Human:")
    if query == "q":
        break

    memory_variables = memory.load_memory_variables()

    answer_prompt = (
        "你是一个强大的聊天机器人，请根据对应的上下文和用户提问解决问题。\n\n"
        f"{memory_variables.get('chat_history')}"
        f"用户提问：{query}"
    )
    print(answer_prompt)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": answer_prompt}],
        stream=True,
    )

    ai_content = ""
    for chunk in response:
        if len(chunk.choices) == 0:
            continue
        content = chunk.choices[0].delta.content
        if content is None:
            break
        ai_content += content
        print(content, flush=True, end="")
    print("")
    memory.save_context(query, ai_content)
