# Despliegue en Railway

## Variables de entorno (obligatorias en Railway)

En el proyecto de Railway → **Variables** añade:

| Variable | Obligatoria | Ejemplo | Descripción |
|----------|-------------|---------|-------------|
| `OPENAI_API_KEY` | **Sí** | `sk-proj-...` | API key de OpenAI |
| `OPENAI_CHAT_MODEL` | No | `gpt-4o-mini` | Modelo de chat (default: gpt-4o-mini) |
| `OPENAI_EMBEDDING_MODEL` | No | `text-embedding-3-large` | Modelo de embeddings (default: text-embedding-3-large) |
| `USER_AGENT` | No | `PromtiorRAG/1.0` | Opcional, para logs |

**PORT** la define Railway; no hace falta configurarla.

## Pasos para subir

1. Sube el repo a GitHub (si no está ya).
2. En [Railway](https://railway.app): **New Project** → **Deploy from GitHub repo** → elige el repo.
3. En **Variables** del servicio, añade al menos `OPENAI_API_KEY`.
4. Railway usará el `Procfile` y hará `pip install -r requirements.txt`; el comando de inicio será `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
5. Genera un dominio público en **Settings** → **Networking** → **Generate Domain**.

## Notas

- **Playwright**: En Railway normalmente no hay Chromium instalado. La app usará el fallback (WebBaseLoader) para la web; el contenido de las URLs puede ser más limitado que en local. Los PDFs y la API de OpenAI funcionan igual.
- **PDFs**: Incluye la carpeta `data/` con tus PDFs en el repo para que se carguen en el despliegue. Si no existe `data/`, la app arranca igual pero solo con el contenido web.
