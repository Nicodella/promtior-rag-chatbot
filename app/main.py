from fastapi import FastAPI
from langserve import add_routes

from app.rag_chain import create_rag_chain

#from dotenv import load_dotenv
#load_dotenv()

#crear la app web
app = FastAPI(
    title="Promtior RAG Chatbot",
    version="1.0",
    description="Chatbot using RAG + LangChain + Ollama"
)


#crear el chain (cerebro)
qa_chain = create_rag_chain()


#exponer endpoint automáticamente con LangServe
add_routes(
    app,
    qa_chain,
    path="/chat"
)


#endpoint simple de prueba
@app.get("/")
def root():
    return {"status": "running"}
