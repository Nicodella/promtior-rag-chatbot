import os

# Core
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Text splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector stores
from langchain_community.vectorstores import FAISS

# Embeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

# LLMs
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI

from app.loaders import load_documents


USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def create_rag_chain():
    #Cargamos las fuentes de conocimiento
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = splitter.split_documents(docs)

    if USE_OLLAMA: 
        embeddings = OllamaEmbeddings(model="nomic-embed-text") 
        llm = Ollama(model="llama3")
    else:
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        llm = ChatOpenAI(model="gpt-4", temperature=0)

    vectorstore = FAISS.from_documents(splits, embeddings) 
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template("""
        You are a question-answering assistant.

        STRICT RULES:
        - Use ONLY the information contained in the provided context.
        - Do NOT use prior knowledge.
        - Do NOT guess or invent information.
        - If the answer is not explicitly present in the context, reply EXACTLY:
        "I don't have information about that in the provided documents."

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """)


    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
