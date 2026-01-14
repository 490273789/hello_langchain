from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
import models

# v1.0
agentV1 = create_agent(
    model=models.qwen,
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model=models.qwen,
            max_tokens_before_summary=400,
            message_to_keep=20,
            summary_prompt="custom prompt for summarization"
        )
    ]
)


# v1.1
agent = create_agent(
    model=models.qwen,
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model=models.qwen,
            trigger=("tokens", 400),
            keep=("messages", 10),
            summary_prompt="custom prompt for summarization"
        )
    ]
)

