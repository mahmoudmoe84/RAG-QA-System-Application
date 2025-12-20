"""Vector store module for qdrant operations"""
from functools import lru_cache 
from typing import Any
from uuid import uuid4
#uuid4 generates a random UUID

from langchain_core.documents import Document 
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams

from app.config import get_settings
from app.utils.logger import get_logger
from app.core.embedding import get_embeddings

logger = get_logger(__name__)
settings = get_settings()

#embedding dimension for text-embedding-3-small model
EMBEDDING_DIMENSION = 1536


@lru_cache
def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client with caching"""
    logger.info(f"Creating Qdrant client at: {settings.QDRANT_URL}")
    client = QdrantClient(
        url =settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )   
    logger.info("Qdrant client created successfully")
    return client


class VectorStoreService: 
    """Service for managing Qdrant vector store operations"""
    
    
    def __init__(self, collection_name:str |None =None) -> None:
        
        """initialize vector store service"""
        
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self.client = get_qdrant_client()
        self.embeddings = get_embeddings()
        self._ensure_collection()
        
        #initialize Qdrant vector store 
        
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        
        logger.info(f"VectorStoreService initialized for collection: {self.collection_name}")
    
    def _ensure_collection(self):
        """ensure the Qdrant collection exists"""
        try:
            collection_info = self.client.get_collection(
                self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists with"
                        f"with {collection_info.points_count} points")
            #collection points count is the number of vectors stored in the collection
        except UnexpectedResponse:
            logger.info(f"Creation collection: {self.collection_name}")
            self.client.create_collection(collection_name=self.collection_name,
                                          vectors_config = VectorParams(
                                          size=EMBEDDING_DIMENSION,
                                          distance=Distance.COSINE))
            logger.info(f"Collection '{self.collection_name}' created successfully")
    
    def add_documents(self,documents: list[Document],ids=ids) -> None:
        """add documents to the vector store"""
        if not documents:
            logger.warning("Non Documents to add to vector store")
            return []
        
        logger.info(f"adding {len(documents)} documents to vector store")
        
        ids = [str(uuid4()) for _ in documents]
        
        self.vector_store.add_documents(documents,ids=ids)
        logger.info("Documents added successfully")
        return ids
    
    def search(self,query:str , k:int | None=None)->list[Document]:
        """search the vector store for similar documents"""
        
        k = k or settings.top_k_retrieval
        logger.info(f"Searching for top {k} documents similar to query")
        
        results = self.vector_store.similarity_search(
            query,k=k)
        
        logger.debug(f"Search results: {results}" )
        return results
    
    def search_with_scores(self,query:str , k:int |None =None) ->list[tuple[Document, float]]:
        """search for similar documents with relevance scores"""
        k = k or settings.top_k_retrieval
        logger.info(f"Searching for top {k} documents similar to query with scores")
        
        results = self.vector_store.similarity_search_with_score(
            query,k=k)
        
        logger.debug(f"Search results with scores: {results}")
        return results
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        logger.warning(f"Deleting collection: {self.collection_name}")
        self.client.delete_collection(self.collection_name)
        logger.info(f"Collection '{self.collection_name}' deleted")

    def get_collection_info(self) -> dict:
        """Get information about the collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status.value,
            }
        except UnexpectedResponse:
            return {
                "name": self.collection_name,
                "points_count": 0,
                "indexed_vectors_count": 0,
                "status": "not_found",
            }

    def health_check(self) -> bool:
        """Check if vector store is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return False
    
    def get_retriever(self, k: int | None = None) -> Any:
        """Get a retriever for the vector store.

        Args:
            k: Number of documents to retrieve

        Returns:
            LangChain retriever object
        """
        k = k or settings.top_k_retrieval

        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
    )