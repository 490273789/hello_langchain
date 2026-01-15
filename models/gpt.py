
from langchain.chat_models import init_chat_model

gpt4o = init_chat_model(model="gpt-4o", model_provider="openai")
gpt4mini = init_chat_model(model="gpt-4o-mini", model_provider="openai")




# res1 = gpt4o.invoke('你是谁？')
# print(f"res1: {res1}")
# print("-" * 20)

# res2 = qwen.invoke("你是谁？")
# print(f"res2: {res2}")
# print("-" * 20)

# res3 = ds.invoke("你是谁？")
# print(f"res3: {res3}")
# print("-" * 20)
