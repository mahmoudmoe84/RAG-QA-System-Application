"""FASTAPI app main entry point."""

# IMPORTANT: Load .env file FIRST, before any LangChain imports
# This ensures LangSmith environment variables are available for tracing
# ruff: noqa: E402, I001

from dotenv import load_dotenv
load_dotenv

from contextlib import asynccontextmanager
#contextlib is a module that provides utilities for working with context managers and the asynccontextmanager is a decorator that allows you to define asynchronous context managers using async functions.
#example is that it allows you to set up and tear down resources that require asynchronous operations, such as database connections or network connections, in a clean and efficient manner.
from fastapi import FastAPI , Request 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
#this imports the __version__ variable from the app package. This variable typically contains the version number of the application, which can be useful for logging, debugging, or displaying version information in API responses.
from app.api.routes import documents , health, query 
from app.config import get_settings
from app.utils.logger import get_logger , setup_logging

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """application lifespan manager"""
    
    #startup code 
    setup_logging(settings.LOG_LEVEL)
    logger = get_logger(__name__)
    logger.info(f"Starting up {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"logging level set to {settings.LOG_LEVEL}")
    
    yield
    #yield here is intended to separate the startup code from the shutdown code in an asynchronous context manager.
    #When the context manager is entered, the code before the yield statement is executed (startup code). When the context manager is exited, the code after the yield statement is executed (shutdown code).
    #shutdown code
    logger.info(f"Shutting down {settings.APP_NAME} v{settings.APP_VERSION}")

#create the FastAPI app instance
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""sumary_line
## RAG Q&A System API

A Retrieval-Augmented Generation (RAG) question-answering system built with:
- **FastAPI** for the API layer
- **LangChain** for RAG orchestration
- **Qdrant Cloud** for vector storage
- **OpenAI** for embeddings and LLM

### Features
- Upload PDF, TXT, and CSV documents
- Ask questions and get AI-powered answers
- View source documents for transparency
- Streaming responses for real-time feedback
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

#add cors middle ware this will allow cross origin requests from any domain meaning that any website can make requests to this API without being blocked by the browser's same-origin policy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

#mount static files meaning that any requests to the /static URL path will be served from the app/static directory on the server.
app.mount("/static", StaticFiles(directory="app/static"), name="static")

#include API routers
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)

@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def root():
    """serve the main UI"""
    with open("static/index.html") as f:
        return f.read()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to catch unhandled exceptions."""
    logger = get_logger(__name__)
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error",
                 "message": str(exc)
        }
    )
#the above route is a global exception handler for a FastAPI application. It catches any unhandled exceptions that occur during the processing of requests and logs the error details using a logger. It then returns a JSON response with a 500 status code (Internal Server Error) and includes a generic error message along with the specific exception message.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", 
                host=settings.API_HOST, 
                port=settings.API_PORT, 
                reload=True)    