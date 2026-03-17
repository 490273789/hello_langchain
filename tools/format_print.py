import json
from pprint import pprint
from typing import Any


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert objects into JSON-serializable structures."""
    if hasattr(obj, "model_dump"):
        return _to_jsonable(obj.model_dump())

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]

    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def pretty_print(message: Any, view: str = "all") -> None:
    """Pretty print LangChain-style response objects.

    Args:
        message: Usually an AIMessage/ChatMessage instance.
        view: One of "all", "content", "meta", "json".
    """

    # Support both a single message object and agent state dict:
    # {"messages": [SystemMessage, HumanMessage, AIMessage, ToolMessage, ...]}
    def is_ai_message(msg: Any) -> bool:
        return msg.__class__.__name__ == "AIMessage"

    # state_messages = message.get("messages", []) if isinstance(message, dict) else []
    content = getattr(message, "content", "")
    response_metadata = getattr(message, "response_metadata", {})
    usage_metadata = getattr(message, "usage_metadata", None)
    has_model_dump = hasattr(message, "model_dump")
    # ai_messages = [m for m in state_messages if is_ai_message(m)]
    # final_ai_message = ai_messages[-1] if ai_messages else None

    # Optional dependency: rich for better terminal readability.
    try:
        from rich.console import Console
        from rich.json import JSON
        from rich.panel import Panel
        from rich.pretty import Pretty

        console = Console()

        # if view in {"all", "zen", "content"}:
        #     if final_ai_message is not None:
        #         console.print(
        #             Panel.fit(
        #                 str(getattr(final_ai_message, "content", "")),
        #                 title="final_ai_content",
        #                 border_style="cyan",
        #             )
        #         )
        #     else:
        #         console.print(
        #             Panel.fit(
        #                 "<no AIMessage>", title="final_ai_content", border_style="red"
        #             )
        #         )

        if view in {"all", "zen", "content"} and content:
            console.print(Panel.fit(str(content), title="content", border_style="cyan"))

        if view in {"all", "zen", "meta"} and response_metadata:
            console.print(
                Panel.fit(
                    Pretty(response_metadata),
                    title="response_metadata",
                    border_style="green",
                )
            )
            console.print(
                Panel.fit(
                    Pretty(usage_metadata),
                    title="usage_metadata",
                    border_style="yellow",
                )
            )

        if view in {"all", "zen", "json", "full"}:
            if has_model_dump:
                payload = _to_jsonable(message.model_dump())
                console.print(
                    Panel.fit(
                        JSON.from_data(payload),
                        title="full_message_json",
                        border_style="magenta",
                    )
                )
            else:
                console.print(
                    Panel.fit(
                        Pretty(message),
                        title="full_message_repr",
                        border_style="magenta",
                    )
                )

        if view in {"all", "json"}:
            console.print(
                Panel.fit(
                    JSON.from_data(_to_jsonable(message)),
                    title="full_state_json",
                    border_style="magenta",
                )
            )

    except ImportError:
        print(
            "\n[提示] 未安装 rich，已自动使用 pprint 输出。可执行: uv pip install rich"
        )

        if view in {"all", "content"}:
            print("\n=== content ===")
            print(content)

        if view in {"all", "meta"}:
            print("\n=== response_metadata ===")
            pprint(response_metadata, sort_dicts=False)
            print("\n=== usage_metadata ===")
            pprint(usage_metadata, sort_dicts=False)

        if view in {"all", "json"}:
            print("\n=== full_message_json ===")
            if has_model_dump:
                print(
                    json.dumps(
                        _to_jsonable(message.model_dump()),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                pprint(message)

        if view in {"all", "json"}:
            print("\n=== full_state_json ===")
            print(json.dumps(_to_jsonable(message), ensure_ascii=False, indent=2))
