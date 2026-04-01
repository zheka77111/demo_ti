
import json
import os
import pickle
import tempfile
from glob import glob
from pathlib import Path
from typing import Annotated, Any, Callable, Dict


from blanks.prompts import (
    DEFAULT_IMAGE_PROMPT,
    IMAGE_ANALYSIS_PROMPT,
    REDUCE_REPORT_PROMPT,
    STEP2_SUMMARY_PROMPT,
    METRICS_STRUCT,
    METRICS_2TEXT
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument
from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import ToolRuntime
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import InjectedToolCallId, tool
from langchain_core.tools.base import ToolException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph, RunnableConfig
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, Send
from rich import print
from src.gigachat_api import analyze_image_langchain
from src.llm import embeddings, llm, parse_llm
from src.evaluator import eval_report, EntityType
from src.utils import print_eval
from src.state import GraphState_parallel as GraphState, CustomState

# from src.llm import llm


load_dotenv()
key = os.environ.get('GIGACHAT_API_KEY')

def export_md(text:str, path:str) -> None:
    """
    Export text to markdown file
    """
    with open(path, 'w') as f:
        f.write(text)

def load_from_exist_file(file:str)-> DoclingDocument | str | None:
    """
    Load from existing file
    """

    if Path(file).as_posix().endswith('json'):
        return json.loads(file)
    if Path(file).as_posix().endswith('.md'):
        with open( file, 'r' ) as f:
            return f.read()
 



def make_parse_pdf_gigachat_postprocess(prompt: str = DEFAULT_IMAGE_PROMPT) -> Callable[[dict[str, Any]], dict[str, str]]:
    """
    Фабричная функция. Возвращает узел LangGraph для парсинга PDF
    с постобработкой изображений через GigaChat (без прокси-сервера).

    Принцип работы:
      1. docling парсит PDF и сохраняет позицию каждой картинки как <!-- image --> в markdown.
      2. doc.pictures и <!-- image --> в markdown идут в одном порядке (1:1).
      3. Для каждого PictureItem вызывается analyze_image_langchain().
      4. Первое вхождение <!-- image --> заменяется описанием (count=1),
         затем берётся следующая картинка — так описания встают строго на свои места.

    Args:
        prompt: промпт для анализа изображений через GigaChat

    Returns:
        Функция-узел LangGraph с сигнатурой (state: dict) -> dict
    """
    def _node(state: dict) -> GraphState:
        pdf_path = state["pdf_path"]
        name = Path(pdf_path).stem
        out_dir = Path("work")
        out_dir.mkdir(parents=True, exist_ok=True)

        md_path = out_dir / f"{name}_gigachat_v2.md"

        # Кэш: если файлы уже есть — не перепарсиваем
        if md_path.exists():
            return {
                # "docling_doc": load_from_exist_file(pkl_path.as_posix()), # чекпоинтер не работает, если передавать несерриализуемый объект, избавляемся от docling_doc
                "doc_markdown": load_from_exist_file(md_path.as_posix()),
            }

        # Парсим PDF с извлечением PIL-изображений (без вызова внешнего LLM)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_picture_images = True

        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        res = converter.convert(str(pdf_path))
        doc = res.document

        # Базовый markdown: каждое изображение → <!-- image --> на своём месте в потоке текста
        doc_md = doc.export_to_markdown(page_break_placeholder="<!-- PAGE_BREAK -->")

        # Постобработка: описываем каждую картинку и вставляем на её место
        # doc.pictures и вхождения <!-- image --> в markdown совпадают по порядку
        for picture in doc.pictures:
            pil_image = picture.get_image(doc)
            if pil_image is None:
                # нет данных изображения — оставляем плейсхолдер нетронутым
                continue

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                pil_image.save(tmp.name)
                tmp_path = tmp.name

            try:
                description = analyze_image_langchain(tmp_path, prompt)
                replacement = f"\n**[Изображение]**: {description}\n"
            except Exception as e:
                replacement = f"<!-- image: ошибка описания — {e} -->"
            finally:
                os.unlink(tmp_path)

            # count=1: заменяем строго первое вхождение — это и есть текущая картинка в тексте
            doc_md = doc_md.replace("<!-- image -->", replacement, 1)

        # export_pickle(doc, pkl_path.as_posix()) # чекпоинтер не работает, если передавать несерриализуемый объект, избавляемся от docling_doc
        export_md(doc_md, md_path.as_posix())
        return {"doc_markdown": doc_md}

    return _node


# функции для работы графа обработки документа

parse_with_gigachat_images = make_parse_pdf_gigachat_postprocess(prompt=IMAGE_ANALYSIS_PROMPT)

def rag_prepare_data(state: GraphState):
    """
    Подготовка и загрузка данных в векторную базу для RAG
    Args: state - состояние графа
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1600,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=True,
        separators=["<!-- PAGE_BREAK -->", "[Изображение]", "##", '\n\n\n\n','\n\n']
    )
  
    filename = Path(state.get("pdf_path")).name
    not_splited_text = state.get("doc_markdown")
    

    vectorstore = Chroma(
        collection_name="reports_rag",
        persist_directory="chroma_reports_1",
        embedding_function=embeddings,
        
    )
    existing = vectorstore.get(
        where={"filename": filename},
        include=[],
    )
    if existing and existing.get("ids"):
        return {
            "rag_ready": {
                "collection_name": "reports_rag",
                "persist_directory": "chroma_reports_1",
                "filename": filename,
                "already_indexed": True,
            }
        }
    
    splited_texts = text_splitter.create_documents(
        [not_splited_text],
        metadatas=[{"filename": filename}],
    )

    vectorstore.add_documents(splited_texts)

    return {
        "rag_ready": {
            "collection_name": "reports_rag",
            "persist_directory": "chroma_reports_1",
            "filename": filename,
            "already_indexed": False,
        }
    }

def summarize_step2(state: GraphState) -> Dict[str, Any]:
    """
    Кратко резюмирует содержание документа в md формате.
    Args: state - состояние графа
    """
    messages = [
        SystemMessage(content=STEP2_SUMMARY_PROMPT),
        HumanMessage(content=state["doc_markdown"]),
    ]
    result = llm.invoke(messages)
    return {"summary": result.content}

def prepare_pages(state: GraphState) -> Dict[str, Any]:
    """
    Разбиваем документ в md формате на страницы по маркеру <!-- PAGE_BREAK -->
    Args: state - состояние графа
    """
    pages_md = [{"page_no": i, "md": x.strip()}
                for i, x in enumerate(state["doc_markdown"].split("<!-- PAGE_BREAK -->"))
                if x.strip()]
    return {"pages": pages_md, "page_metrics": []}


def fanout_pages(state: GraphState):
    """
    На каждую страницу создаём отдельный запуск узла-воркера — распараллеливаем процесс
    Args: state - состояние графа
    Return: List[]
    """
    return [
        Send("extract_metrics_page", {"page": page})
        for page in state["pages"]
    ]


def extract_metrics_page(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Извлекаем метрики для одной страницы. Запускается параллельно для всех страниц.
    """
    page = state["page"]
    messages = [
        SystemMessage(content=METRICS_STRUCT),
        HumanMessage(content=f'Документ:\n{page["md"]}'),
    ]

    result = parse_llm.invoke(messages)
    
    messages = [
        SystemMessage(content=METRICS_2TEXT),
        HumanMessage(content=f'Документ:\n{result.content}'),
    ]
    result = parse_llm.invoke(messages)


    return {
        "page_metrics": [{"page_no": page["page_no"], "summary": result.content if result.content else ''}]
    }


def reduce_reports(state: GraphState) -> Dict[str, Any]:
    """Объединяем результаты работы LLM по всем страницам в один отчет."""
    items = state.get("page_metrics", [])
    # Сортируем по page_no, потому что параллельные апдейты могут прийти в разном порядке
    items = sorted(items, key=lambda x: (x.get("page_no") is None, x.get("page_no")))
    combined = "\n\n".join(
        i['summary'] for i in items if 'Отчет не содержит' not in i["summary"]
    )
    messages = [
        SystemMessage(content=REDUCE_REPORT_PROMPT),
        HumanMessage(content=f'Документ:\n{combined}'),
    ]

    result = parse_llm.invoke(messages)
    return {"metrics": result.content}


# Граф для извлечения метрик из документа
# ------------------------------------------------
extract_metrics_graph = StateGraph(GraphState)
extract_metrics_graph.add_node("prepare_pages", prepare_pages)
extract_metrics_graph.add_node("extract_metrics_page", extract_metrics_page)
extract_metrics_graph.add_node("reduce_reports", reduce_reports)
extract_metrics_graph.add_edge(START, "prepare_pages")
extract_metrics_graph.add_conditional_edges("prepare_pages", fanout_pages, ["extract_metrics_page"])
extract_metrics_graph.add_edge("extract_metrics_page", "reduce_reports")
extract_metrics_graph.add_edge("reduce_reports", END)
extract_metrics = extract_metrics_graph.compile()

# Граф для обработки документов
# ------------------------------------------------
extract_text_rag_graph = StateGraph(GraphState)
extract_text_rag_graph.add_node("parse_pdf_docling", parse_with_gigachat_images)
extract_text_rag_graph.add_node("rag_prepare_data", rag_prepare_data)
extract_text_rag_graph.add_node("summarize", summarize_step2)
extract_text_rag_graph.add_edge(START, "parse_pdf_docling")
extract_text_rag_graph.add_edge("parse_pdf_docling", "summarize")
extract_text_rag_graph.add_edge("summarize", "rag_prepare_data")
extract_text_rag_graph.add_edge("rag_prepare_data", END)
extract_text_with_rag: CompiledStateGraph = extract_text_rag_graph.compile()

# ------------------------------------------------
# Инструменты для чат агента


@tool
def extract_metrics_tool(config: RunnableConfig,
                        runtime: ToolRuntime[None, CustomState],
                        ) -> Command:
    """
    Извлекает метрики из отчета, хранящегося в state для ответа пользователю. 
    ВНИМАНИЕ: если ранее в сессии не вызывался инструмент extract_data_tool, то extract_metrics_tool вызывать нельзя!!!
    В этом случае попросить пользователя загрузить документ для анализа сначала
    Args:
        path: Путь к файлу
    Result: Command, обновляет состояние агента с полем report и metrics
    """
    state = runtime.state
    doc_markdown = state.get("report", "")
    result = extract_metrics.invoke(
        {"doc_markdown": doc_markdown},
        config={"max_concurrency": 3},
    )
    metrics = result.get("metrics", "")

    if config['configurable']['is_print_eval']:
        with open(config['configurable']['etalon_file']) as f:
            etalon = f.read()
        df, ans, _ = eval_report(etalon, metrics, entity=EntityType.METRICS)
        print_eval(df)
        print(f'{ans}')

    return Command(update={
        "metrics": metrics,
        "messages": [
            ToolMessage(content=f"Метрики извлечены: {metrics}. В ответе напиши только эти метрики как есть!" if metrics else "Метрики не удалось извлечь.", 
                        tool_call_id=runtime.tool_call_id, name="extract_metrics_tool")
        ],
    })


@tool
def extract_data_tool(path: str,
                      runtime: ToolRuntime[None, CustomState],
                    ) -> Command:
    """
    Читает отчет из файла Работает с pdf файлами, готовит RAG для ответа на вопросы.
    Args:
        path: Путь к файлу pdf
    Result: Command  обновляет состояние агента с полем doc_markdown (содержимое файла в md формате) и summary ()
    # """

    query = path.strip()
    candidates = []
    
    if os.path.exists(query):
        candidates = [query]
    else:
        candidates = glob(os.path.join("input", query))
        if not candidates and not query.endswith(".pdf"):
            candidates = glob(os.path.join("input", f"*{query}*.pdf"))

    if not candidates:
        msg = "Файл не найден в папке input. Пожалуйста, пришли файл или уточни название."
        return Command(update={
            "messages": [ToolMessage(content=msg, tool_call_id=runtime.tool_call_id, name="subgraph_agent")]
        })
    
    if len(candidates) > 1:
        msg = "Найдено несколько файлов:\n" + "\n".join(candidates) + "\nУточни, какой именно нужен."
        return Command(update={
            "messages": [ToolMessage(content=msg, tool_call_id=runtime.tool_call_id, name="subgraph_agent")]
        })
    state = {"pdf_path": candidates[0]}
    result = extract_text_with_rag.invoke(state, config={"max_concurrency": 3})

    data = result.get("doc_markdown", "")
    summary = result.get("summary", "")
    rag_ready = result.get("rag_ready")
    return Command(update={
        "path": path,
        "report": data,
        "summary":summary,
        "rag_ready": rag_ready,
        "messages": [
            ToolMessage(content=f"Данные извлечены. В ответе напиши только этот текст: {summary}" if data else "Данные не удалось извлечь.", 
                        tool_call_id=runtime.tool_call_id, name="extract_data_tool")
        ],
    })


@tool
def glob_files(path: str = 'input', pattern: str = "*.pdf"  ) -> str:
    """Находит файлы в папке по маске. Используется, когда пользователь не знает название файла.
    Args:
    path: str - путь папки для поиска, по умолчанию ищем в папке input
    pattern: str - паттерн для поиска файлов, ищем файлы pdf
    Result: str - список файлов"""
    files = glob(os.path.join(path, pattern))
    return str(files)

@tool
def rag_search(query: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Используетсы для ответов на вопрос пользователя, ищет данные в базе RAG.
    Используется, когда нужно найти информацию в документах.
    Args: 
    query: str - поисковый запрос
    Result: str - релевантный текст для ответа
    """
    if not state.get('rag_ready'):
        raise ToolException( "RAG не инициализирован. Сначала нужно обработать отчет командой /extract_data.")
    rag_ready = state['rag_ready']
    filename = Path(state.get("path")).name
    retriever = Chroma(persist_directory=rag_ready["persist_directory"],
                       collection_name=rag_ready["collection_name"],
                       embedding_function=embeddings).as_retriever(search_kwargs={
                                                "k": 5,
                                                "filter": {"filename": {"$eq": filename}} # ищи только чанки, которые получены из загруженного документа
                                            })
    

    config={"max_concurrency": 3}
    docs = retriever.invoke(query, config)
    if not docs:
        return "Релевантных фрагментов в базе RAG не найдено."

    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source") or d.metadata.get("filename") or "unknown"
        parts.append(f"\n[# Фрагмент {i} | source={src}]\n{d.page_content}")
    return "На основе предоставленной информации составь ответ пользователю. Передай точные числа и показатели в ответе. Вот информация: "+"\n\n".join(parts)