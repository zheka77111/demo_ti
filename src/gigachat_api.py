import os
import requests
import uuid
from dotenv import load_dotenv
import base64
from pathlib import Path
from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import HumanMessage

load_dotenv()
GIGACHAT_API_KEY = os.environ.get('GIGACHAT_API_KEY')
NGW_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"
CLIENT_ID = os.getenv("GIGACHAT_CLIENT_ID")
CLIENT_SECRET = os.getenv("GIGACHAT_CLIENT_SECRET")
SCOPE = os.getenv("GIGACHAT_SCOPE")

def get_oauth_token(scope: str = SCOPE) -> str:
    # Функция получает OAUTH токен для дальнейших запросов в Гигачат
    rq_uid = str(uuid.uuid4())
    auth = requests.auth.HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET)

    resp = requests.post(
        NGW_URL,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": rq_uid,
        },
        data={"scope": scope},
        auth=auth,
        verify=False,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]

def analyze_image(image_path: str, prompt: str, model: str = "GigaChat-2-Max") -> str:
    # Функция анализиврует загруженое изображение по заданному промпту с помощью LLM GigaChat. Используется библиотека requests
    access_token = get_oauth_token()

    # 1. загрузка файла
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        data = {"purpose": "general"}
        resp = requests.post(
            f"{GIGACHAT_BASE_URL}/files",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            },
            files=files,
            data=data,
            verify=False,
        )
    resp.raise_for_status()
    file_id = resp.json()["id"]

    # 2. запрос в чат с привязанным изображением
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "attachments": [file_id],
            }
        ],
        "stream": False,
        "temperature": 0.1,
    }
    chat_resp = requests.post(
        f"{GIGACHAT_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=payload,
        verify=False,
    )
    chat_resp.raise_for_status()
    return chat_resp.json()["choices"][0]["message"]["content"]

def analyze_image_langchain(image_path: str, prompt: str) -> str:
    img_bytes = Path(image_path).read_bytes()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    data_uri = f"data:image/png;base64,{img_b64}"  # подставьте нужный mime: png/jpeg

    chat = GigaChat(
        credentials=GIGACHAT_API_KEY,
        model="GigaChat-2-Max",
        scope=SCOPE,
        temperature=0.87,
        verify_ssl_certs=False,
        profanity_check=False,
        timeout=300,
        auto_upload_images=True,
    )

    # ВАЖНО: используем стандартный формат content для LangChain
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": data_uri
                },
            },
        ]
    )

    response = chat.invoke([message])
    return response.content



if __name__ == "__main__":
    # Проводим анализ изображения через requests
    answer = analyze_image(
        image_path="/home/pai/Рабочий стол/image006.png",
        prompt="Опиши, что изображено на картинке и в каком стиле она выполнена",
    )
    print('================================================\nАнализ изображения через requests\n', answer)

    # Проводим анализ изображения через langchain_gigachat
    answer = analyze_image_langchain(
        image_path="/home/pai/Рабочий стол/image006.png",
        prompt="Опиши, что на картинке и в каком стиле выполнено изображение",
    )
    print('\n================================================\nАнализ изображения через langchain_gigachat\n', answer)