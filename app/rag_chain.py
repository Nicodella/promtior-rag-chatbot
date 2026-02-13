from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA

from app.loaders import load_documents

def create_rag_chain():
    #Cargamos las fuentes de conocimiento
    docs = load_documents()


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    splits = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    llm = Ollama(model="llama3")

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
