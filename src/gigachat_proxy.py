import os
import json
import time
import base64
import io
from typing import Any, Dict, List

import requests
from fastapi import FastAPI, Request, Response

from src.gigachat_api import get_oauth_token, GIGACHAT_BASE_URL


app = FastAPI()


@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request) -> Response:
    """
    Простой HTTP-прокси для GigaChat, совместимый с OpenAI /v1/chat/completions.

    Docling будет стучаться сюда по http://localhost:<PORT>/v1/chat/completions
    без TLS и без заголовка авторизации, а прокси уже сам:
    - получает access token через get_oauth_token,
    - форвардит запрос на https://gigachat.devices.sberbank.ru/api/v1/chat/completions
      с verify=False (игнорируем корпоративный self-signed сертификат),
    - возвращает тело ответа как есть.
    """
    body = await request.body()

    # Разбираем входящий JSON от Docling
    try:
        parsed_payload: Dict[str, Any] = json.loads(body.decode("utf-8"))
    except Exception:
        return Response(
            content=b'{"status":400,"message":"Invalid JSON in request to proxy."}',
            status_code=400,
            media_type="application/json",
        )

    messages: List[Dict[str, Any]] = parsed_payload.get("messages", [])
    model_name = parsed_payload.get("model", "GigaChat-2-Max")
    temperature = parsed_payload.get("temperature", 0.0)
    max_tokens = parsed_payload.get("max_completion_tokens", 200)

    # Проверяем, есть ли в запросе OpenAI-совместимое описание изображения через "image_url"
    has_image_url = False
    image_url_value = None
    if messages:
        first_msg = messages[0]
        content = first_msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    img = item.get("image_url", {})
                    image_url_value = img.get("url")
                    if isinstance(image_url_value, str) and image_url_value.startswith("data:image/"):
                        has_image_url = True
                        break

    access_token = get_oauth_token()

    if has_image_url and image_url_value is not None:
        # Путь 1: Docling прислал картинку в формате OpenAI (data:image/...;base64,...).
        # Для GigaChat нужно сначала загрузить файл, затем дернуть /chat/completions с attachments.
        try:
            header, b64_data = image_url_value.split(",", 1)
            mime_type = "image/png"
            if header.startswith("data:") and ";base64" in header:
                mime_type = header[len("data:") : header.index(";base64")]

            img_bytes = base64.b64decode(b64_data.encode("utf-8"))

            files = {
                "file": ("docling_image", io.BytesIO(img_bytes), mime_type),
            }
            data = {"purpose": "general"}

            upload_resp = requests.post(
                f"{GIGACHAT_BASE_URL}/files",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json",
                },
                files=files,
                data=data,
                verify=False,
            )
            upload_resp.raise_for_status()
            file_id = upload_resp.json()["id"]

            # Строим payload максимально близко к рабочему примеру analyze_image из gigachat_api.py:
            # attachments — на уровне сообщения, а не на верхнем уровне.

            # Берем текстовый промпт, если Docling его прислал; иначе — дефолтный.
            prompt_text = "Подробно опиши данные на изображении. Если есть текст/числа — извлеки их максимально дословно."
            try:
                first_msg = messages[0] if messages else {}
                content = first_msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            txt = item.get("text")
                            if isinstance(txt, str) and txt.strip():
                                prompt_text = txt
                                break
            except Exception:
                # если не смогли извлечь текст — остаемся на дефолтном prompt_text
                pass

            gc_payload: Dict[str, Any] = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt_text,
                        "attachments": [file_id],
                    }
                ],
                "stream": False,
                "temperature": temperature,
            }
            if max_tokens is not None:
                gc_payload["max_tokens"] = max_tokens
        except Exception:
            return Response(
                content=b'{"status":400,"message":"Error while handling image for GigaChat proxy."}',
                status_code=400,
                media_type="application/json",
            )
    else:
        # Путь 2: обычный текстовый запрос — упрощенная адаптация OpenAI-совместимого payload.
        gc_payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            gc_payload["max_tokens"] = max_tokens

    resp = requests.post(
        f"{GIGACHAT_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=gc_payload,
        verify=False,
    )

    # Пытаемся привести ответ GigaChat к максимально OpenAI-совместимому формату
    raw_content = resp.content
    transformed = False
    try:
        data = resp.json()
        if isinstance(data, dict):
            # GigaChat не присылает поле id, а Pydantic OpenAiApiResponse его требует
            if "id" not in data:
                data["id"] = f"chatcmpl-{int(time.time() * 1000)}"
                transformed = True
            # На всякий случай гарантируем наличие ключевых полей
            if "created" not in data:
                data["created"] = int(time.time())
                transformed = True
            if "choices" not in data:
                data["choices"] = []
                transformed = True
            if "usage" not in data:
                data["usage"] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
                transformed = True
        if transformed:
            raw_content = json.dumps(data, ensure_ascii=False).encode("utf-8")
    except Exception:
        # Если не удалось распарсить JSON — возвращаем как есть
        pass

    return Response(
        content=raw_content,
        status_code=resp.status_code,
        media_type="application/json",
    )


if __name__ == "__main__":
    # Локальный запуск: uvicorn src.gigachat_proxy:app --host 127.0.0.1 --port 8001
    import uvicorn

    host = os.environ.get("GIGACHAT_PROXY_HOST", "127.0.0.1")
    port = int(os.environ.get("GIGACHAT_PROXY_PORT", "8001"))
    uvicorn.run(app, host=host, port=port)

