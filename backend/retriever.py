#Importando bibliotecas

from langchain_google_genai import ChatGoogleGenerativeAI
from rich import print
from rich.panel import Panel
from langchain.document_loaders.csv_loader import CSVLoader
from typing import List
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

class GemmaEmbeddings(Embeddings):
    def __init__(self, model_name: str = "google/embeddinggemma-300m"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode_document(texts)
        return embeddings.tolist() if hasattr(embeddings, "tolist") else [list(e) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode_query(text)
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)


# Obter a chave da API
load_dotenv(override=True)
load_dotenv(os.path.join("..", ".env"), override=True)
GEMINI_CHAT_ESCOLA_KEY = os.getenv("GEMINI_CHAT_ESCOLA_KEY")

if not GEMINI_CHAT_ESCOLA_KEY:
    GEMINI_CHAT_ESCOLA_KEY = os.getenv("GEMINI_CHAT_ESCOLA_KEY")



def retriever():
        current_dir = os.path.dirname(os.path.abspath(__file__))

        #Embeddings
        embeddings = GemmaEmbeddings(model_name="google/embeddinggemma-300m")

        # Caminho do índice local do FAISS
        index_path = os.path.join(current_dir, "faiss_index")

        if os.path.exists(index_path):
            # Carrega o banco de dados existente de forma instantânea
            vectordb = FAISS.load_local(
                index_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            # Importa o banco de dados
            csv_path = os.path.join(current_dir, 'informacoe_escola_pos.csv')
            loader = CSVLoader(file_path=csv_path, source_column="PERGUNTAS", encoding="ISO-8859-1")
            data = loader.load()

            # Cria o banco e calcula os embeddings (apenas na primeira vez)
            vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
            
            # Salva o índice localmente para as próximas inicializações
            vectordb.save_local(index_path)

        #Criando Retriever:
        retriever = vectordb.as_retriever(score_threshold = 0.7)

        return retriever