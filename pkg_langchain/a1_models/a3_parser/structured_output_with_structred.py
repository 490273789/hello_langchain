from typing import Annotated, TypedDict

from tools import init_llm_client, pretty_print_ai

llm = init_llm_client()


class Animal(TypedDict):
    animal: Annotated[str, "动物"]
    emoji: Annotated[str, "表情"]


class AnimalList(TypedDict):
    animals: Annotated[list[Animal], "动物与表情列表"]  # List<Animal>


messages = [
    {
        "role": "human",
        "content": "任意生成三种动物，以及他们的 emoji 表情，以 JSON 格式返回",
    }
]

llm = llm.with_structured_output(AnimalList)
resp = llm.invoke(messages)
print(resp)
pretty_print_ai(resp, "message")
