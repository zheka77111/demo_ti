import operator
from enum import StrEnum
from typing import Annotated, Any, Dict, Generic, Iterable, List, Literal, TypeVar

from langchain.agents import AgentState
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, create_model
from typing_extensions import NotRequired, TypedDict

class CustomState(AgentState):
    """
    Состояние агента, расширенное полями report и metrics.
    report - полный текст отчета
    metrics - извлеченные метрики
    """

    path: NotRequired[str]
    rag_ready: NotRequired[dict]
    report: NotRequired[str]
    summary: NotRequired[str]
    metrics: NotRequired[str]


class GraphState_parallel(TypedDict, total=False):
    pdf_path: str
    rag_ready: dict 
    doc_markdown: str
    summary: str
    # map-reduce поля:
    pages: List[Dict[str, Any]]  # [{page_no: int, md: str}, ...]
    page_metrics: Annotated[List[Dict[str, Any]], operator.add]  # reducer: append
    metrics: NotRequired[str]
    # trends: NotRequired[str] # поле, аналогичное metrics для хранения извлеченных трендов
    # forecast: NotRequired[str] # поле, аналогичное metrics для хранения извлеченных предиктов
    # statistics: NotRequired[str] # поле, аналогичное metrics для хранения извлеченных статистик
    # technical_details: NotRequired[str] # поле, аналогичное metrics для хранения извлеченных технических деталей



