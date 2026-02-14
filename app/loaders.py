import os
import warnings
import concurrent.futures
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    DirectoryLoader,
)
from langchain_core.documents import Document
from bs4 import BeautifulSoup

# URLs de Promtior (sitio con JavaScript; hace falta Playwright para ver el contenido real)
PROMTIOR_URLS = [
    "https://promtior.ai/",
    "https://www.promtior.ai/service",
]

WEB_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def _load_web_playwright(urls):
    """
    Carga páginas con Playwright en modo SÍNCRONO (sync_playwright).
    Evita 'asyncio.run() cannot be called from a running event loop' con uvicorn.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    docs = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            for url in urls:
                try:
                    page = browser.new_page()
                    page.goto(url, wait_until="networkidle", timeout=20000)
                    html = page.content()
                    page.close()
                    if html and "Error:" not in html[:200]:
                        text = BeautifulSoup(html, "html.parser").get_text(
                            separator="\n", strip=True
                        )
                        if len(text) > 100:
                            docs.append(Document(page_content=text, metadata={"source": url}))
                except Exception as e:
                    warnings.warn(f"Playwright no pudo cargar {url}: {e}")
            browser.close()
        return docs if docs else None
    except Exception as e:
        warnings.warn(f"Playwright falló: {e}")
        return None


def _load_web_fallback(urls):
    """Fallback sin JavaScript (puede devolver poco contenido en sitios SPA)."""
    docs = []
    for url in urls:
        try:
            loader = WebBaseLoader(
                url,
                header_template=WEB_HEADERS,
                requests_kwargs={"timeout": 15},
            )
            docs.extend(loader.load())
        except Exception as e:
            warnings.warn(f"No se pudo cargar {url}: {e}")
    return docs


def load_documents():
    """
    Carga documentos desde la web de Promtior (con Playwright si está instalado)
    y PDFs en data/.
    """
    docs = []

    # En Railway no hay Chromium; usar solo fallback evita el warning de Playwright
    skip_playwright = os.environ.get("RAILWAY") or os.environ.get("DISABLE_PLAYWRIGHT")
    if not skip_playwright:
        try:
            import asyncio
            asyncio.get_running_loop()
            in_async = True
        except RuntimeError:
            in_async = False
        if in_async:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                f = pool.submit(_load_web_playwright, PROMTIOR_URLS)
                web_docs = f.result(timeout=60)
        else:
            web_docs = _load_web_playwright(PROMTIOR_URLS)
    else:
        web_docs = None
    if web_docs:
        docs.extend(web_docs)
    else:
        docs.extend(_load_web_fallback(PROMTIOR_URLS))

    # 2) PDFs (si existe la carpeta data/)
    if os.path.isdir("data"):
        try:
            pdf_loader = DirectoryLoader(
                "data",
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
            )
            docs.extend(pdf_loader.load())
        except Exception as e:
            warnings.warn(f"No se pudieron cargar PDFs: {e}")

    return docs
