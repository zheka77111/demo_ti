
## Интерактивный агент `agent_report.py`

CLI-диалог с GigaChat: парсинг PDF из `input/`, извлечение метрик, RAG-поиск по документу. История дописывается в JSON после каждого ответа.

### Запуск

Из каталога `agents/TI` (где `pyproject.toml` / виртуальное окружение):

Убедитесь, что установлен пакетный менеджер [`uv`](https://github.com/astral-sh/uv):

```bash
pip install uv
```

Затем подтяните зависимости по `pyproject.toml` и `uv.lock`:

```bash
uv sync
```

Создайте папку `input/`, если её нет:
```bash
mkdir -p input
```



```bash
cd agents/TI
uv run python agent_report.py
uv run python agent_report.py --print-eval
uv run python agent_report.py --etalon path/to/etalon.md
uv run python agent_report.py --print-eval --etalon input/etalon.md
```

Справка по флагам:

```bash
uv run python agent_report.py -h
```

### Аргументы командной строки

| Аргумент | Значение по умолчанию | Назначение |
|----------|----------------------|------------|
| `--print-eval` | выключен | В `config` агента в `configurable` передаётся `is_print_eval=True` (используется middleware / eval-логика, если настроена). |
| `--etalon PATH` | `input/etalon.md` | Путь к файлу эталона; попадает в `configurable["etalon_file"]` при стриминге и в `/dataset`. |

### Переменные окружения

Подхватываются из `.env` (`load_dotenv()`).

| Переменная | Назначение | Значение по умолчанию в коде |
|------------|------------|------------------------------|
| `GIGACHAT_API_KEY` | Учётные данные GigaChat | пусто (нужно задать) |
| `GIGACHAT_MODEL` | Идентификатор модели | `GigaChat-2-MAX` |
| `TEMPERATURE` | Температура сэмплирования | `0.8` |
| `HISTORY_FILE` | Имя файла истории **внутри** каталога `history/` | `history.json` |
| `DATASET_PATH` | JSON со сценариями для команды `/dataset` | `input/dataset.json` |

Итоговый путь истории: `history/<HISTORY_FILE>`.

### Команды в чате

| Ввод | Действие |
|------|----------|
| `/exit`, `exit`, `quit`, `:q` | Выход из программы |
| `/reset` | Очистить историю в памяти и перезаписать файл истории пустым списком |
| `/history` | Вывести накопленные сообщения в stdout (JSON) |
| `/dataset` | Загрузить `DATASET_PATH` и прогнать диалог через `run_dialog` с тем же `config` (включая `is_print_eval`, `etalon_file`) |

Любой другой текст — запрос агенту (инструменты: `extract_data_tool`, `glob_files`, `extract_metrics_tool`, `rag_search`).

### Зависимости и окружение

- В начале скрипта выставляется `CUDA_VISIBLE_DEVICES=""`.
- `scope` GigaChat: `GIGACHAT_API_CORP`, `verify_ssl_certs=False` (как в коде).

### Программный вызов

```python
from agent_report import main

main(is_print_eval=False, etalon_file="input/etalon.md")
```
