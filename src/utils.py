import json
from typing import Callable
from rich.table import Table
from rich.console import Console
from langchain.agents.middleware import (
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    wrap_model_call,
)
import pandas as pd
from langchain.messages import AIMessage
from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

console = Console()
def show_prompt(prompt_text: str, title: str = "Prompt", border_style: str = "white"):
    """Display a prompt with rich formatting and XML tag highlighting.

    Args:
        prompt_text: The prompt string to display
        title: Title for the panel (default: "Prompt")
        border_style: Border color style (default: "blue")
    """
    # Create a formatted display of the prompt
    formatted_text = Text(prompt_text)
    formatted_text.highlight_regex(r"<[^>]+>", style="bold blue")  # Highlight XML tags
    formatted_text.highlight_regex(
        r"##[^#\n]+", style="bold magenta"
    )  # Highlight headers
    formatted_text.highlight_regex(
        r"###[^#\n]+", style="bold cyan"
    )  # Highlight sub-headers

    # Display in a panel for better presentation
    console.print(
        Panel(
            formatted_text,
            title=f"[bold green]{title}[/bold green]",
            border_style=border_style,
            padding=(1, 2),
        )
    )


    """Utility functions for displaying messages and prompts in Jupyter notebooks."""



console = Console()


def format_message_content(message):
    """Convert message content to displayable string."""
    parts = []
    tool_calls_processed = False

    # Handle main content
    if isinstance(message.content, str):
        parts.append(message.content)
    elif isinstance(message.content, list):
        # Handle complex content like tool calls (Anthropic format)
        for item in message.content:
            if item.get("type") == "text":
                parts.append(item["text"])
            elif item.get("type") == "tool_use":
                parts.append(f"\n🔧 Tool Call: {item['name']}")
                parts.append(f"   Args: {json.dumps(item['input'], indent=2, ensure_ascii=False)}")
                parts.append(f"   ID: {item.get('id', 'N/A')}")
                tool_calls_processed = True
    else:
        parts.append(str(message.content))

    # Handle tool calls attached to the message (OpenAI format) - only if not already processed
    if (
        not tool_calls_processed
        and hasattr(message, "tool_calls")
        and message.tool_calls
    ):
        for tool_call in message.tool_calls:
            parts.append(f"\n🔧 Tool Call: {tool_call['name']}")
            parts.append(f"   Args: {json.dumps(tool_call['args'], indent=2, ensure_ascii=False)}")
            parts.append(f"   ID: {tool_call['id']}")

    return "\n".join(parts)


def format_messages(messages):
    """Format and display a list of messages with Rich formatting."""
    for m in messages:
        msg_type = m.__class__.__name__.replace("Message", "")
        content = format_message_content(m)

        if msg_type == "Human":
            console.print(Panel(Markdown(content), title="🧑 Human", border_style="blue"))
        elif msg_type == "AI" or msg_type == 'model':
            console.print(Panel(Markdown(content), title="🤖 Assistant", border_style="green"))
        elif msg_type == "Tool" or msg_type == "tools":
            console.print(Panel(Markdown(content), title="🔧 Tool Output", border_style="yellow"))
        else:
            console.print(Panel(Markdown(content), title=f"📝 {msg_type}", border_style="white"))


def format_message(messages):
    """Alias for format_messages for backward compatibility."""
    return format_messages(messages)



# more expressive runner
async def stream_agent(agent, query, config=None):
    async for graph_name, stream_mode, event in agent.astream(
        query,
        stream_mode=["updates", "values"], 
        subgraphs=True,
        config=config
    ):
        if stream_mode == "updates":
            print(f'Graph: {graph_name if len(graph_name) > 0 else "root"}')
            
            node, result = list(event.items())[0]
            print(f'Node: {node}')
            
            for key in result.keys():
                if "messages" in key:
                    # print(f"Messages key: {key}")
                    format_messages(result[key])
                    break
        elif stream_mode == "values":
            current_state = event

    return current_state

def run_dialog(agent: CompiledStateGraph, mas_array: list[dict[str, list[dict[str, str]]]], config: dict) -> None:
    """
    Run a dialog with the agent using a series of messages.

    Args:
        agent: The compiled state graph agent
        mas_array: List of message dictionaries
        thread_id: Thread identifier for the conversation
    
    """
    for mes in mas_array:
        # config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 350 }
        console.print(Panel(mes['messages'][0]['content'], title="🧑 Human", border_style="white"))
        for chunk in agent.stream(mes, config):
            if 'model' in chunk:
                format_messages(chunk['model']['messages'])
            if 'tools' in chunk:
                format_messages(chunk['tools']['messages'])


def print_eval(df:pd.DataFrame) -> None:
    """
    Печатает DataFrame в виде таблицы с помощью rich
    """
    c = Console()
    t = Table(*df.columns.to_list(), show_lines=True, style = "dim")
    for _, row in df.iterrows():
        t.add_row(*[str(val) for val in row.tolist()])
    c.print(t)


# добавляем функцию в middleware для обработки случая, когда файл уже обработан
@wrap_model_call
def file_aware_middleware(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    state = request.state


    path = state.get("path")
    report = state.get("report")

    already_processed = bool(path and report)

    if already_processed:
        request = request.override(
            system_prompt=(
                "Текущий файл уже обработан. "
                "Используй данные из state и не запускай повторную обработку файла."
            )
        )

    return handler(request)