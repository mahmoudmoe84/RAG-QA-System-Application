# RAG Q&A System

> **⚠️ Work in Progress**: This project is currently under active development. Some features may be incomplete or subject to change. Contributions and feedback are welcome!

A production-ready Retrieval-Augmented Generation (RAG) question-answering system built with FastAPI, LangChain, and Qdrant. This system enables intelligent document search and conversational AI capabilities with automatic response evaluation.

## ✨ Features

- 📄 **Document Processing**: Supports multiple formats (PDF, CSV, TXT) with intelligent chunking
- 🔍 **Semantic Search**: Powered by OpenAI embeddings and Qdrant vector database
- 🤖 **AI-Powered Responses**: Uses GPT-4o-mini for accurate, context-aware answers
- 📊 **Automatic Evaluation**: Built-in RAGAS metrics for response quality assessment
- 🎨 **Modern Web Interface**: Clean, responsive UI for easy interaction
- 🐳 **Docker Ready**: Containerized deployment with Docker support
- 📝 **Comprehensive Logging**: Built-in logging with file and console output
- ⚙️ **Highly Configurable**: Environment-based configuration management

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.13+
- **AI/ML**: LangChain, OpenAI (GPT-4o-mini, text-embedding-3-small)
- **Vector Database**: Qdrant Cloud
- **Document Processing**: pypdf, python-docx (CSV, TXT, PDF support)
- **Evaluation**: RAGAS (Retrieval-Augmented Generation Assessment)
- **Frontend**: HTML, CSS, JavaScript (Vanilla)

## 📋 Prerequisites

# RAG Q&A System

**Status: Complete**

This repository contains a working Retrieval-Augmented Generation (RAG) question-answering system built with FastAPI, LangChain, and Qdrant. The project includes document ingestion, embedding storage, semantic retrieval, a simple frontend, and automatic RAGAS-based evaluation.

## ✨ Highlights

- Project marked as **Complete**: core features implemented and runnable locally or via Docker.
- Document ingestion and chunking pipeline
- Semantic search using OpenAI embeddings + Qdrant
- RAG pipeline producing context-grounded responses
- Built-in RAGAS evaluation for automatic response quality assessment
- Dockerized for easy deployment

## Quick Start

1. Create a `.env` in the project root with your keys (see `app/config.py` for variables).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3a. Run locally (development):

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

3b. Or run with Docker Compose:

```bash
docker-compose up --build
```

4. Open the app: http://localhost:8000 — API docs at `/docs`.

## Project Structure

Top-level layout:

```
app/                # FastAPI app, core logic and utilities
static/             # Frontend assets
notebooks/          # Experiments and notes
data/               # Raw/sample data
tests/              # Tests
docker-compose.yml
Dockerfile
requirements.txt
README.md
```

Key files:

- `app/main.py`: FastAPI entrypoint
- `app/config.py`: configuration and env variables
- `app/core/document_processor.py`: chunking & preprocessing
- `app/core/embedding.py`: embedding generation
- `app/core/vector_store.py`: Qdrant interactions
- `app/core/rag_chain.py`: RAG orchestration

## Configuration

Edit environment variables (or `app/config.py`) to set:

- `OPENAI_API_KEY`, `QDRANT_*` settings
- `COLLECTION_NAME`, `CHUNK_SIZE`, `CHUNK_OVERLAP`
- `llm_model`, `embedding_model`, `top_k_retrieval`

## API Examples

Upload a document:

```bash
curl -X POST "http://localhost:8000/api/upload" -F "file=@/path/to/doc.pdf"
```

Ask a question:

```bash
curl -X POST "http://localhost:8000/api/query" -H "Content-Type: application/json" -d '{"question":"Summarize the document."}'
```

## Tests & Quality

Run tests:

```bash
pytest tests/
```

Format and lint:

```bash
black app/ && ruff check app/ && mypy app/
```

## Docker

Build and run:

```bash
docker-compose up -d --build
docker-compose logs -f
docker-compose down
```

## Contributing

This repository is currently marked complete. If you find bugs or want to contribute improvements, open an issue or submit a Pull Request.

## License

MIT

---

If you'd like, I can also commit these changes, run tests, or update the project version. Which would you prefer next?
```
