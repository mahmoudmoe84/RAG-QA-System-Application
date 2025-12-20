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

- Python 3.13 or higher
- OpenAI API key
- Qdrant Cloud account (or local Qdrant instance)
- Docker and Docker Compose (optional, for containerized deployment)

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd "RAG QA System Application"
```

### 2. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Qdrant Configuration (Cloud)
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key

# Collection Settings
COLLECTION_NAME=rag_documents

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Model Configuration
llm_model=gpt-4o-mini
llm_temperature=0.0
embedding_model=text-embedding-3-small

# Retrieval Settings
top_k_retrieval=5

# RAGAS Evaluation
enable_ragas_evaluation=true
ragas_timeout_seconds=30.0
ragas_log_results=true

# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# Application Info
APP_NAME=RAG Q&A System
APP_VERSION=0.1.0
```

### 3. Install Dependencies

#### Using pip:

```bash
pip install -r requirements.txt
```

#### Using uv (recommended):

```bash
uv pip install -r requirements.txt
```

### 4. Run the Application

#### Local Development:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Using Docker:

```bash
docker-compose up --build
```

### 5. Access the Application

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

## 📂 Project Structure

```
RAG QA System Application/
├── app/
│   ├── __init__.py
│   ├── config.py              # Application configuration
│   ├── main.py                # FastAPI application entry point
│   ├── api/                   # API routes and endpoints
│   ├── core/                  # Core business logic
│   │   ├── document_processor.py  # Document chunking and processing
│   │   ├── embedding.py           # Embedding generation
│   │   ├── rag_chain.py           # RAG pipeline implementation
│   │   └── vector_store.py        # Vector database operations
│   └── utils/
│       └── logger.py          # Logging configuration
├── static/                    # Frontend assets
│   ├── index.html
│   ├── css/
│   └── js/
├── notebooks/                 # Jupyter notebooks for experimentation
│   ├── exp.ipynb
│   └── data/
│       └── raw/
├── tests/                     # Test files
├── sample_data/              # Sample documents for testing
├── docker-compose.yml        # Docker Compose configuration
├── Dockerfile               # Docker image configuration
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Project metadata
└── README.md               # This file
```

## 🔧 Configuration

The application uses environment variables for configuration. Key settings include:

- **Document Processing**: Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` for optimal text splitting
- **Model Selection**: Choose between different OpenAI models via `llm_model` and `embedding_model`
- **Retrieval**: Control the number of retrieved documents with `top_k_retrieval`
- **Evaluation**: Enable/disable RAGAS evaluation with `enable_ragas_evaluation`

## 📖 API Usage

### Upload Documents

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@your_document.pdf"
```

### Ask Questions

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

## 🧪 Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black app/

# Lint code
ruff check app/

# Type checking
mypy app/
```

## 📊 Evaluation Metrics

The system uses RAGAS (Retrieval-Augmented Generation Assessment) to automatically evaluate:

- **Context Relevancy**: How relevant retrieved documents are to the query
- **Answer Relevancy**: How well the answer addresses the question
- **Faithfulness**: How grounded the answer is in the retrieved context
- **Answer Correctness**: Overall quality of the generated response

## 🐳 Docker Deployment

Build and run with Docker Compose:

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/)
- Vector storage by [Qdrant](https://qdrant.tech/)
- Powered by [OpenAI](https://openai.com/)
- Evaluation framework by [RAGAS](https://github.com/explodinggradients/ragas)

## 📧 Contact

For questions or support, please open an issue in the repository.

---

**Note**: This is a naive RAG implementation (p1-naive-rag-run1). Future iterations may include advanced features like hybrid search, re-ranking, and multi-modal support.
