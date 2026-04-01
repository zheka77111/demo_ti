import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_gigachat import GigaChat
from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from blanks.prompts import DIALOG_PROMPT
from src.state import CustomState
from src.tools import extract_data_tool, extract_metrics_tool, glob_files, rag_search
from src.utils import file_aware_middleware, format_messages, run_dialog

os.environ["CUDA_VISIBLE_DEVICES"] = ""

load_dotenv()
console = Console()

# --- Настройки (берём из переменных окружения или используем значения по умолчанию) ---
API_KEY = os.environ.get("GIGACHAT_API_KEY", "")
MODEL = os.environ.get("GIGACHAT_MODEL", "GigaChat-2-MAX")
HISTORY_FILE = Path("history") / os.environ.get("HISTORY_FILE", "history.json")
DATASET_PATH = os.environ.get("DATASET_PATH", "input/dataset.json")
TEMPERATURE = os.environ.get("TEMPERATURE", 0.8)
# Сохранение данных агента в память
checkpointer = MemorySaver()


# --- Вспомогательные функции ---

def safe_text(text) -> str:
    """
    Безопасно преобразует входное значение в строку с корректной UTF-8 кодировкой.
    Функция обрабатывает различные типы входных данных и гарантирует возврат
    валидной UTF-8 строки, заменяя некорректные символы на безопасные аналоги.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return text.encode("utf-8", errors="replace").decode("utf-8")


def message_to_dict(msg):
    """Преобразует сообщение в словарь."""
    if isinstance(msg, dict):
        return msg
    return {
        "role": getattr(msg, "type", msg.__class__.__name__.lower()),
        "content": getattr(msg, "content", ""),
        "name": getattr(msg, "name", None),
    }


def messages_to_dict(messages):
    """Преобразует список сообщений в список словарей."""
    return [message_to_dict(m) for m in messages]

def save_history(path: Path, messages: list[dict[str, Any]]) -> None:
    """Сохраняет историю в файл."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(messages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# --- Основной цикл ---


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Интерактивный агент отчёта (GigaChat)")
    p.add_argument(
        "--print-eval",
        action="store_true",
        dest="is_print_eval",
        default=False,
        help="Оценить полученный результат в сравнении с эталонным отчетом. По умолчанию False",
    )
    p.add_argument(
        "--etalon",
        type=str,
        default="input/etalon.md",
        metavar="PATH",
        dest="etalon_file",
        help="Путь к файлу эталону, по умолчанию input/etalon.md",
    )
    return p.parse_args()


def main(is_print_eval: bool = False,
         etalon_file: str = 'input/etalon.md'
        ):
    llm = GigaChat(
        credentials=API_KEY,
        model=MODEL,
        scope="GIGACHAT_API_CORP",
        temperature=TEMPERATURE,
        verify_ssl_certs=False,
        profanity_check=False,
        max_tokens=25000,
        timeout=300,
    )

    agent = create_agent(
        model=llm,
        system_prompt=DIALOG_PROMPT,
        checkpointer=checkpointer,
        state_schema=CustomState,
        tools=[extract_data_tool, glob_files, extract_metrics_tool, rag_search],
        middleware=[file_aware_middleware],
    )

    messages = []
    thread_id = "1"
    console.print(Markdown("Бот запущен. Для анализа файлов pdf поместите их в папку `input`"))
    console.print(Markdown("Команды: */exit*, */reset*, */history*, */dataset*"))

    # цикл общения с пользователем
    while True:
        try:
            user_text = console.input(prompt ="you> ").strip()
            user_text = safe_text(user_text)
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not user_text:
            continue

        if user_text in {"/exit", "exit", "quit", ":q"}:
            print("bye")
            break
        
        config = {"configurable": {"thread_id": thread_id,
                                   "is_print_eval": is_print_eval,
                                   "etalon_file": etalon_file}, "recursion_limit": 350}
        
        if user_text == "/dataset":
            with open(DATASET_PATH, "r", encoding="utf-8") as f:
                dataset_data = json.load(f)
            run_dialog(agent, dataset_data, config = config)
            continue

        if user_text == "/reset":
            messages = []
            save_history(HISTORY_FILE, messages)
            console.print(Panel("История очищена", title="Reset", border_style="blue"))
            continue

        if user_text == "/history":
            print(json.dumps(messages, ensure_ascii=False, indent=2))
            continue

        console.print(Panel(safe_text(user_text), title="🧑 Human", border_style="white"))
        messages.append({"role": "user", "content": user_text})
        

        try:
            for chunk in agent.stream({"messages": {"role": "user", "content": user_text}}, config):
                if "model" in chunk:
                    format_messages(chunk["model"]["messages"])
                    messages.extend(messages_to_dict(chunk["model"]["messages"]))
                if "tools" in chunk:
                    format_messages(chunk["tools"]["messages"])
                    messages.extend(messages_to_dict(chunk["tools"]["messages"]))
        except Exception as e:
            print(safe_text(f"error: {e}"))
            messages.pop()
            continue

        save_history(HISTORY_FILE, messages)


if __name__ == "__main__":
    _args = parse_args()
    main(is_print_eval=_args.is_print_eval, etalon_file=_args.etalon_file)
