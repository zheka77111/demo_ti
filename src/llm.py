import os



from dotenv import load_dotenv

from langchain_gigachat.chat_models import GigaChat
from langgraph.checkpoint.memory import MemorySaver
from langchain_gigachat import GigaChatEmbeddings
checkpointer = MemorySaver()

load_dotenv()
key = os.environ.get('GIGACHAT_API_KEY')


load_dotenv()

key_openai = os.environ.get("OPENAI")





llm = GigaChat(
    credentials=key,
    model='GigaChat-2-Max',
    scope='GIGACHAT_API_CORP',
    temperature=0.87,
    verify_ssl_certs=False,
    profanity_check=False,
    max_tokens=25000,
    timeout=300,
)

# для парсинга метрик выставляем температуру в 0
parse_llm = GigaChat(
    credentials=key,
    model='GigaChat-2-Max',
    scope='GIGACHAT_API_CORP',
    temperature=0,
    top_p=0.8,
    verify_ssl_certs=False,
    profanity_check=False,
    max_tokens=25000,
    timeout=300,
)

embeddings = GigaChatEmbeddings(credentials=key,
                scope='GIGACHAT_API_CORP',
                model = "EmbeddingsGigaR",
                verify_ssl_certs = False,
                
)