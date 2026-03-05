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


def pretty_print_response(message: Any, view: str = "all") -> None:
    """Pretty print LangChain-style response objects.

    Args:
        message: Usually an AIMessage/ChatMessage instance.
        view: One of "all", "content", "meta", "json".
    """
    # Support both a single message object and agent state dict:
    # {"messages": [SystemMessage, HumanMessage, AIMessage, ToolMessage, ...]}

    content = getattr(message, "content", "")
    response_metadata = getattr(message, "response_metadata", {})
    usage_metadata = getattr(message, "usage_metadata", None)
    has_model_dump = hasattr(message, "model_dump")

    # Optional dependency: rich for better terminal readability.
    try:
        from rich.console import Console
        from rich.json import JSON
        from rich.panel import Panel
        from rich.pretty import Pretty

        console = Console()

        if view in {"all", "content"}:
            console.print(Panel.fit(str(content), title="content", border_style="cyan"))

        if view in {"all", "meta"}:
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

        if view in {"all", "json"}:
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
    if isinstance(message, dict) and isinstance(message.get("messages"), list):
        _pretty_print_agent_state(message, view=view)


def _pretty_print_agent_state(state: dict, view: str = "all") -> None:
    messages = state.get("messages", [])

    def is_ai_message(msg: Any) -> bool:
        return msg.__class__.__name__ == "AIMessage"

    ai_messages = [m for m in messages if is_ai_message(m)]
    tool_call_messages = [m for m in ai_messages if getattr(m, "tool_calls", [])]
    final_ai_message = ai_messages[-1] if ai_messages else None

    try:
        from rich.console import Console
        from rich.json import JSON
        from rich.panel import Panel
        from rich.pretty import Pretty

        console = Console()

        if view in {"all", "content"}:
            if final_ai_message is not None:
                console.print(
                    Panel.fit(
                        str(getattr(final_ai_message, "content", "")),
                        title="final_ai_content",
                        border_style="cyan",
                    )
                )
            else:
                console.print(
                    Panel.fit(
                        "<no AIMessage>", title="final_ai_content", border_style="red"
                    )
                )

        if view in {"all", "meta"}:
            if final_ai_message is not None:
                console.print(
                    Panel.fit(
                        Pretty(getattr(final_ai_message, "response_metadata", {})),
                        title="final_response_metadata",
                        border_style="green",
                    )
                )
                console.print(
                    Panel.fit(
                        Pretty(getattr(final_ai_message, "usage_metadata", None)),
                        title="final_usage_metadata",
                        border_style="yellow",
                    )
                )

        if view in {"all", "json"}:
            console.print(
                Panel.fit(
                    JSON.from_data(_to_jsonable(state)),
                    title="full_state_json",
                    border_style="magenta",
                )
            )

    except ImportError:
        print(
            "\n[提示] 未安装 rich，已自动使用 pprint 输出。可执行: uv pip install rich"
        )

        if view in {"all", "content"}:
            print("\n=== final_ai_content ===")
            if final_ai_message is not None:
                print(getattr(final_ai_message, "content", ""))
            else:
                print("<no AIMessage>")

        if view in {"all", "meta"}:
            print("\n=== tool_calls ===")
            if tool_call_messages:
                for idx, msg in enumerate(tool_call_messages, start=1):
                    print(f"\n--- tool_calls_ai_message_{idx} ---")
                    pprint(getattr(msg, "tool_calls", []), sort_dicts=False)
                    print(f"\n--- tool_call_response_metadata_{idx} ---")
                    pprint(getattr(msg, "response_metadata", {}), sort_dicts=False)
            else:
                print([])

            print("\n=== final_response_metadata ===")
            if final_ai_message is not None:
                pprint(
                    getattr(final_ai_message, "response_metadata", {}), sort_dicts=False
                )
                print("\n=== final_usage_metadata ===")
                pprint(
                    getattr(final_ai_message, "usage_metadata", None), sort_dicts=False
                )
            else:
                pprint({}, sort_dicts=False)
                print("\n=== final_usage_metadata ===")
                pprint(None)

        if view in {"all", "json"}:
            print("\n=== full_state_json ===")
            print(json.dumps(_to_jsonable(state), ensure_ascii=False, indent=2))
