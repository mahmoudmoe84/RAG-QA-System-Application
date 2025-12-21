"""RAG Chain Module."""

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from app.utils.logger import get_logger
from app.config import get_settings 
from app.core.vector_store import VectorStoreService

logger = get_logger(__name__)
settings = get_settings()

RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question based on the provided context.

If you cannot answer the question based on the context, say "I don't have enough information to answer that question."

Do not make up information. Only use the context provided.

Context:
{context}

Question: {question}

Answer:"""

def format_docs(docs:list[Document]):
    """format documents for RAG prompt."""
    return "\n\n".join([doc.page_content for doc in docs])


class RAGChain:
    """rag chain class."""
    
    def __init(self,vector_store_service:VectorStoreService | None = None):
        
        self.vector_store = vector_store_service or VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        
        self._evaluator = None 
        
        # initialize llm
        self.llm = ChatOpenAI(
            model= settings.llm_model,
            temperature= settings.llm_temperature,
            openai_api_key= settings.openai_api_key)
        
        #create prompt
        self.prompt= ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        
        self.chain = (
            {"context":self.retriever |format_docs,
             "question": RunnablePassthrough()}
            |self.prompt
            |self.llm 
            | StrOutputParser()
        )
        
        logger.info(f"RAG Chain initialized successfully. with model {settings.llm_model}")
    
    @property
    def evaluator(self):
        """get or create RAGAS evaluator instance"""
        if self._evaluator is None:
            from app.core.ragas_evaluator import RAGASEvaluator
            self._evaluator = RAGASEvaluator()
        return self._evaluator
    
    
    def query(self , question:str) -> str:
        """query RAG chain"""
        logger.info(f"Processing RAG query: {question}")
        
        try:
            answer = self.chain.invoke({"question": question})
            logger.info("RAG query processed successfully.")
            return answer
        except Exception as e:
            logger.error(f"Error processing RAG query: {e}")
            raise
    
    def query_with_sources(self, question:str) -> dict:
        """query rag chain and return answer with sources."""
        logger.info(f"Processing RAG query with sources: {question[:50]}...")
        
        try:
            answer =self.chain.invoke({"question": question})
            source_docs = self.retriever.invoke(question)
            
            sources = [
                {
                    "content": (
                        doc.page_content[:500] + "..." 
                        if len(doc.page_content) > 500 
                        else doc.page_content
                    ),
                    "metadata": doc.metadata
                } for doc in source_docs]
            
            logger.info("RAG query with sources processed successfully.")
            
            return {
                "answer": answer,
                "sources": sources}
        
        except Exception as e:
            logger.error(f"Error processing RAG query with sources: {e}")
            raise
        
    async def aquery(self, question:str) -> str:
        """execute async query."""
        logger.info(f"Processing async RAG query: {question}")
        
        try:
            answer = await self.chain.ainvoke({"question": question})
            logger.info("Async RAG query processed successfully.")
            return answer
        except Exception as e:
            logger.error(f"Error processing async RAG query: {e}")
            raise 
    async def aquery_with_sources(self, question: str) -> dict:
        """Execute an async RAG query and return sources.

        Args:
            question: User question

        Returns:
            Dictionary with answer and source documents
        """
        logger.info(f"Processing async query with sources: {question[:100]}...")

        try:
            # Get answer
            answer = await self.chain.ainvoke(question)

            # Get source documents (sync operation)
            source_docs = self.retriever.invoke(question)

            # Format sources
            sources = [
                {
                    "content": (
                        doc.page_content[:500] + "..."
                        if len(doc.page_content) > 500
                        else doc.page_content
                    ),
                    "metadata": doc.metadata,
                }
                for doc in source_docs
            ]

            logger.info(f"Async query processed with {len(sources)} sources")

            return {
                "answer": answer,
                "sources": sources,
            }
        except Exception as e:
            logger.error(f"Error processing async query with sources: {e}")
            raise

    async def aquery_with_evaluation(self, question: str, include_sources: bool = True) -> dict:
        """Execute async RAG query with RAGAS evaluation.

        Args:
            question: User question
            include_sources: Whether to include sources in response

        Returns:
            Dictionary with answer, sources, and evaluation scores
        """
        logger.info(f"Processing query with evaluation: {question[:100]}...")

        try:
            # Get answer and sources
            result = await self.aquery_with_sources(question)
            answer = result["answer"]
            sources = result["sources"]

            # Prepare contexts for evaluation
            contexts = [source["content"] for source in sources]

            # Run evaluation with error handling
            try:
                evaluation = await self.evaluator.aevaluate(
                    question=question, answer=answer, contexts=contexts
                )
                logger.info(
                    f"Evaluation completed - "
                    f"faithfulness={evaluation.get('faithfulness', 'N/A')}, "
                    f"answer_relevancy={evaluation.get('answer_relevancy', 'N/A')}"
                )
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}", exc_info=True)
                evaluation = {
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "evaluation_time_ms": None,
                    "error": str(e),
                }

            return {"answer": answer, "sources": sources, "evaluation": evaluation}

        except Exception as e:
            logger.error(f"Error in query with evaluation: {e}")
            raise

    def stream(self, question: str):
        """Stream RAG response.

        Args:
            question: User question

        Yields:
            Response chunks
        """
        logger.info(f"Streaming query: {question[:100]}...")

        try:
            for chunk in self.chain.stream(question):
                yield chunk
        except Exception as e:
            logger.error(f"Error streaming query: {e}")
            raise    
    