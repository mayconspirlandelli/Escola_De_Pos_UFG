from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings  # Ajustado para Pydantic v2
from dotenv import load_dotenv
import os


# Obter a chave da API
load_dotenv(override=True) 
load_dotenv(os.path.join("..", ".env"), override=True)
GEMINI_CHAT_ESCOLA_KEY = os.getenv("GEMINI_CHAT_ESCOLA_KEY")

if not GEMINI_CHAT_ESCOLA_KEY:
    GEMINI_CHAT_ESCOLA_KEY = os.getenv("GEMINI_CHAT_ESCOLA_KEY")

MODEL_NAME = os.getenv("GEMINI_MODEL")


# Iniciando a API
app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas as origens (use origens específicas em produção)
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos os métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todos os cabeçalhos
)