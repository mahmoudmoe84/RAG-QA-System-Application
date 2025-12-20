"""Embeddings Generation Module using OpenAI API."""

from functools import lru_cache 

from langchain_openai import OpenAIEmbeddings
from app.config import get_settings
from app.utils.logger import get_logger 

logger = get_logger(__name__)

@lru_cache()
def get_embeddings() -> OpenAIEmbeddings: 
    """
    get cached openai embeddings instance

    Returns:
        OpenAIEmbeddings: OpenAI embeddings instance
    """
    settings = get_settings()
    logger.info("Creating OpenAI Embeddings instance")
    
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.OPENAI_API_KEY
    )
    
    logger.info("OpenAI Embeddings instance created successfully")
    return embeddings 


class EmbeddingService:
    """Service for Generating Embeddings
    """
    def __init__(self):
        """initialize EmbeddingService
        """
        settings = get_settings()
        self.embeddings = get_embeddings()
        
    def embed_query(self,query:str) ->list[float]:
        """Generate embedding for a query string

        Args:
            query (str): input query string
        Returns:
            list[float]: embedding vector
        """
        logger.debug(f"Generating embedding for query: {query[:100]}...")
        return self.embeddings.embed_query(query)
    
    def embed_documents(self,documents:list[str]) ->list[list[float]]:
        """Generate embeddings for a list of documents

        Args:
            documents (list[str]): list of input document strings
        Returns:
            list[list[float]]: list of embedding vectors
        """
        logger.debug(f"Generating embeddings for {len(documents)} documents...")
        return self.embeddings.embed_documents(documents)
    
    
# list[float] | list[list[float]] can be used for type hinting in Python 3.9+
# it is equivalent to typing.List[float] | typing.List[typing.List[float]]