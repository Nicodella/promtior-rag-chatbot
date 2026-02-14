import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from app.loaders import load_documents


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
# text-embedding-3-large rankea mejor que small (web + PDF se mezclan bien)
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain():
    docs = load_documents()
    splits = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0, api_key=OPENAI_API_KEY)

    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(k=10)

    prompt = ChatPromptTemplate.from_template("""
You are a question-answering assistant.

RULES:
- Use ONLY the information in the context below.
- If the answer is not in the context, reply: "I don't have information about that in the provided documents."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""")

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
