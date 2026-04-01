import operator
import os
from enum import StrEnum
from typing import Annotated, Any, Dict, Generic, Iterable, List, Literal, TypeVar

import pandas as pd
from langchain.agents import AgentState, create_agent
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, create_model
from typing_extensions import NotRequired, TypedDict

# ============================================================
# 1. Опциональный конфиг входных сущностей
# ============================================================


class EntityAliases(TypedDict):
    metrics: NotRequired[str]
    trends: NotRequired[str]
    forecast: NotRequired[str]
    statistics: NotRequired[str]
    tehnical_details: NotRequired[str]


# ============================================================
# 2. Реестр сущностей
# ============================================================


class EntityType(StrEnum):
    METRICS = "metrics"
    TRENDS = "trends"
    FORECAST = "forecast"
    STATISTICS = "statistics"
    TEHNICAL_DETAILS = "tehnical_details"


# ============================================================
# 3. Конфиг текстов для schema / structured output
# ============================================================


class EntityTextConfig(BaseModel):
    entity_key: str
    display_name_ru: str

    category_alias: str
    category_description: str

    reference_alias: str
    reference_description: str

    exists_alias: str
    exists_description: str

    extracted_alias: str
    extracted_description: str


ENTITY_TEXTS: dict[EntityType, EntityTextConfig] = {
    EntityType.METRICS: EntityTextConfig(
        entity_key="metrics",
        display_name_ru="метрики",
        category_alias="Категория метрик",
        category_description="Тематическая категория метрики",
        reference_alias="Метрики из отчета Notebook_lm",
        reference_description="Эталонная формулировка метрики из референсного отчета NotebookLM",
        exists_alias="Есть ли метрика в отчете Гига",
        exists_description="Присутствует ли соответствующая метрика в извлечении Гигачата",
        extracted_alias="Метрики Гигачата",
        extracted_description="Формулировка соответствующей метрики, найденной Гигачатом; если метрики нет, ставь —",
    ),
    EntityType.TRENDS: EntityTextConfig(
        entity_key="trends",
        display_name_ru="тренды",
        category_alias="Категория трендов",
        category_description="Тематическая категория тренда",
        reference_alias="Тренды из отчета Notebook_lm",
        reference_description="Эталонная формулировка тренда из референсного отчета NotebookLM",
        exists_alias="Есть ли тренд в отчете Гига",
        exists_description="Присутствует ли соответствующий тренд в извлечении Гигачата",
        extracted_alias="Тренды Гигачата",
        extracted_description="Формулировка соответствующего тренда, найденного Гигачатом; если тренда нет, ставь —",
    ),
    EntityType.FORECAST: EntityTextConfig(
        entity_key="forecast",
        display_name_ru="прогнозы",
        category_alias="Категория прогнозов",
        category_description="Тематическая категория прогноза",
        reference_alias="Прогнозы из отчета Notebook_lm",
        reference_description="Эталонная формулировка прогноза из референсного отчета NotebookLM",
        exists_alias="Есть ли прогноз в отчете Гига",
        exists_description="Присутствует ли соответствующий прогноз в извлечении Гигачата",
        extracted_alias="Прогнозы Гигачата",
        extracted_description="Формулировка соответствующего прогноза, найденного Гигачатом; если прогноза нет, ставь —",
    ),
    EntityType.STATISTICS: EntityTextConfig(
        entity_key="statistics",
        display_name_ru="статистики",
        category_alias="Категория статистики",
        category_description="Тематическая категория статистики",
        reference_alias="Статистики из отчета Notebook_lm",
        reference_description="Эталонная формулировка статистики из референсного отчета NotebookLM",
        exists_alias="Есть ли статистика в отчете Гига",
        exists_description="Присутствует ли соответствующая статистика в извлечении Гигачата",
        extracted_alias="Статистики Гигачата",
        extracted_description="Формулировка соответствующей статистики, найденной Гигачатом; если статистики нет, ставь —",
    ),
    EntityType.TEHNICAL_DETAILS: EntityTextConfig(
        entity_key="tehnical_details",
        display_name_ru="технические детали",
        category_alias="Категория технических деталей",
        category_description="Тематическая категория технической детали",
        reference_alias="Технические детали из отчета Notebook_lm",
        reference_description="Эталонная формулировка технической детали из референсного отчета NotebookLM",
        exists_alias="Есть ли техническая деталь в отчете Гига",
        exists_description="Присутствует ли соответствующая техническая деталь в извлечении Гигачата",
        extracted_alias="Технические детали Гигачата",
        extracted_description="Формулировка соответствующей технической детали, найденной Гигачатом; если детали нет, ставь —",
    ),
}


def _coerce_exists_in_giga(v: object) -> object:
    """LLM иногда кладёт «—» в exists_*; нормализуем к Да/Нет."""
    if v is None:
        return "Нет"
    if isinstance(v, str):
        s = v.strip()
        if not s or s in ("—", "–", "-", "―", "−"):
            return "Нет"
        low = s.lower()
        if low in ("да", "yes", "true", "1", "y"):
            return "Да"
        if low in ("нет", "no", "false", "0", "n"):
            return "Нет"
    return v


ExistsDaNet = Annotated[
    Literal["Да", "Нет"],
    BeforeValidator(_coerce_exists_in_giga),
]


# ============================================================
# 4. Базовая модель
# ============================================================


class StructuredOutputModel(BaseModel):
    # В structured output ключи JSON — английские имена полей (category, …);
    # русские заголовки только через serialization_alias при model_dump(by_alias=True).
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )


# ============================================================
# 5. Generic контейнер для таблицы
# ============================================================

RowT = TypeVar("RowT", bound=BaseModel)


class ComparisonTable(StructuredOutputModel, Generic[RowT]):
    rows: list[RowT] = Field(
        description="Список строк таблицы сравнения",
    )


# ============================================================
# 6. Фабрика строки
# ============================================================


def build_comparison_row(entity: EntityType) -> type[BaseModel]:
    cfg = ENTITY_TEXTS[entity]
    model_name = f"{entity.name.title().replace('_', '')}ComparisonRow"

    return create_model(
        model_name,
        __base__=StructuredOutputModel,
        category=(
            str,
            Field(
                serialization_alias=cfg.category_alias,
                description=(
                    f"{cfg.category_description}. "
                    "JSON key must be exactly: category"
                ),
            ),
        ),
        reference_item=(
            str,
            Field(
                serialization_alias=cfg.reference_alias,
                description=(
                    f"{cfg.reference_description}. "
                    "JSON key must be exactly: reference_item"
                ),
            ),
        ),
        exists_in_giga_report=(
            ExistsDaNet,
            Field(
                serialization_alias=cfg.exists_alias,
                description=(
                    f"{cfg.exists_description} "
                    "Строго «Да» или «Нет» (латиница да/нет тоже допустима). "
                    "Символ «—» сюда не ставить — только в giga_item. "
                    "JSON key must be exactly: exists_in_giga_report"
                ),
                default="Нет",
            ),
        ),
        giga_item=(
            str,
            Field(
                serialization_alias=cfg.extracted_alias,
                description=(
                    f"{cfg.extracted_description} "
                    "JSON key must be exactly: giga_item"
                ),
                default="",
            ),
        ),
    )


# ============================================================
# 7. Фабрика таблицы под конкретную сущность
# ============================================================


def build_comparison_table(entity: EntityType) -> type[BaseModel]:
    row_model = build_comparison_row(entity)
    table_name = f"{entity.name.title().replace('_', '')}ComparisonTable"

    return create_model(
        table_name,
        __base__=ComparisonTable[row_model],
    )


# ============================================================
# 8. Общая схема оценщика на несколько сущностей сразу
# ============================================================


def build_evaluator_schema(
    entities: Iterable[EntityType],
    *,
    model_name: str = "EvaluatorSchema",
) -> type[BaseModel]:
    field_definitions: dict[str, tuple[Any, Field]] = {}

    for entity in entities:
        table_model = build_comparison_table(entity)
        cfg = ENTITY_TEXTS[entity]

        field_definitions[entity.value] = (
            table_model | None,
            Field(
                default=None,
                description=f"Таблица сравнения для сущности '{cfg.display_name_ru}'",
            ),
        )

    return create_model(
        model_name,
        __base__=StructuredOutputModel,
        **field_definitions,
    )


# ============================================================
# 9. Утилита: преобразование TypedDict-конфига в enum-список
# ============================================================


def entities_from_aliases(config: EntityAliases) -> list[EntityType]:
    return [EntityType(key) for key in config]


def comparison_eval_system_message(entity: EntityType) -> str:
    cfg = ENTITY_TEXTS[entity]
    return (
        f"Тебе на вход подаются {cfg.display_name_ru} из эталонного отчёта NotebookLM "
        f"и из оцениваемого отчёта Гигачата. Для каждого существенного элемента из эталона "
        f"найди семантическое соответствие в ответе Гигачата.\n"
        f"В structured output используй ТОЛЬКО английские ключи полей строки: "
        f"category, reference_item, exists_in_giga_report, giga_item "
        f"(без перевода и без опечаток в русских названиях).\n"
        f"exists_in_giga_report — строго «Да» или «Нет». "
        f"Если соответствия в отчёте Гигачата нет: exists_in_giga_report=«Нет», "
        f"giga_item=«—» (длинное тире) или пустая строка. "
        f"Текст эталона — в reference_item; категория — в category."
    )


def comparison_eval_user_message(entity: EntityType, etalon: str, giga: str) -> str:
    cfg = ENTITY_TEXTS[entity]
    return (
        f"Сопоставь {cfg.display_name_ru}.\n\n"
        f"=== Эталон NotebookLM ===\n{etalon}\n\n"
        f"=== Отчёт Гигачата ===\n{giga}"
    )


# ============================================================
# 10. Готовые схемы
# ============================================================

MetricsRow = build_comparison_row(EntityType.METRICS)
TrendsRow = build_comparison_row(EntityType.TRENDS)

MetricsTable = build_comparison_table(EntityType.METRICS)
TrendsTable = build_comparison_table(EntityType.TRENDS)

EvaluatorSchema = build_evaluator_schema(
    [
        EntityType.METRICS,
        EntityType.TRENDS,
        EntityType.FORECAST,
        EntityType.STATISTICS,
        EntityType.TEHNICAL_DETAILS,
    ],
)

# Старые имена классов / «универсальные» алиасы (по умолчанию — метрики)
ReferenceMetricRow = MetricsRow
ReferenceMetricTable = MetricsTable
ReferenceComparisonRow = MetricsRow
ReferenceComparisonTable = MetricsTable


# ============================================================
# Функция для вызова
# ============================================================

def eval_report(
    etalon_metrics: str,
    giga_report: str,
    entity: EntityType = EntityType.METRICS,
) -> tuple[pd.DataFrame, str, tuple[int, int, int]]:
    schema_model: type[BaseModel] = build_comparison_table(entity)
    cfg = ENTITY_TEXTS[entity]

    key = os.environ.get("OPENAI")
    model = ChatOpenAI(
        model="gpt-5.4-nano",
        api_key=key,
        temperature=0,
        top_p=0.5,
    )

    system_prompt = SystemMessage(content=comparison_eval_system_message(entity))
    reference_report_agent = create_agent(
        model=model,
        system_prompt=system_prompt,
        response_format=schema_model,
    )
    result = reference_report_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": comparison_eval_user_message(
                        entity, etalon_metrics, giga_report
                    ),
                }
            ],
        }
    )

    table = result["structured_response"]
    rows = [r.model_dump(by_alias=True) for r in table.rows]
    df = pd.DataFrame(rows)
    df[cfg.exists_alias] = df[cfg.extracted_alias].apply(
        lambda x: "Нет" if x == "—" or (isinstance(x, str) and not x.strip()) else "Да"
    )
    all_n = int(df[cfg.exists_alias].describe()["count"])
    right_n = int((df[cfg.exists_alias] == "Да").sum())
    miss = all_n - right_n
    return (
        df,
        f"Всего: {all_n}, правильно извлеченных: {right_n}, не извлеченных: {miss}",
        (all_n, right_n, miss),
    )

