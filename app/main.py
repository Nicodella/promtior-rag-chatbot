import os
from dotenv import load_dotenv
load_dotenv()

# LangChain usa USER_AGENT para identificar requests
os.environ.setdefault("USER_AGENT", "PromtiorRAG/1.0")

from fastapi import FastAPI
from langserve import add_routes

from app.rag_chain import create_rag_chain

#crear la app web
app = FastAPI(
    title="Promtior RAG Chatbot",
    version="1.0",
    description="Chatbot using RAG + LangChain"
)

#crear el chain (cerebro)
qa_chain = create_rag_chain()

#exponer endpoint con LangServe
add_routes(
    app,
    qa_chain,
    path="/chat"
)

#endpoint de prueba
@app.get("/")
def root():
    return {"status": "running"}
