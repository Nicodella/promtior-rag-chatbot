from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader

def load_documents():
    """
    Carga documentos desde la web de Promtior.
    Podés agregar más fuentes después (PDFs, etc.)
    """
    docs = []
    urls = [
        "https://promtior.ai/", 
        "https://www.promtior.ai/service"
    ]

    for url in urls:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())

    pdf_loader = DirectoryLoader(
        "data",
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    docs.extend(pdf_loader.load())
    return docs
