"""RAGAS evaluation module for RAG quality assessment."""

import asyncio 
import time 
from typing import Any 

from datasets import Dataset 
#datasets is a library for handling datasets in machine learning from Hugging Face
# It provides tools for loading, processing, and evaluating datasets.
# with ragas-evaluator, we can evaluate the quality of retrieval-augmented generation (RAG) systems.
#datasets imports for ragas is used to load and manage datasets for evaluating RAG systems.

from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
from ragas import evaluate 
from ragas.metrics import answer_relevancy ,faithfulness 

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RAGASEvaluator: 
    """initialize a RAGAS evaluator instance."""
    def __init__(self):
        """initialization method."""
        
        self.settings = get_settings()
        
        #use ragas-specific llm setting if provided otherwise fallback mechanism 
        eval_llm_model = self.settings.ragas_llm_model or self.settings.llm_model
        eval_llm_temperature = self.settings.ragas_llm_temperature if self.settings.ragas_llm_temperature is not None else self.settings.llm_temperature
        
        eval_embedding_model =self.settings.ragas_embedding_model or self.settings.embedding_model
        
        #initialize llm for evaluation 
        self.llm = ChatOpenAI(
            model_name=eval_llm_model,
            temperature=eval_llm_temperature,
            openai_api_key=self.settings.OPENAI_API_KEY,)
        
        self.embedding_model = OpenAIEmbeddings(
            model=eval_embedding_model,
            openai_api_key=self.settings.OPENAI_API_KEY,)
        
        self.metrics =[faithfulness, answer_relevancy]
        
        logger.info(f"RAGAS evaluator initialized"
                    f"LLM Model: {eval_llm_model}, "
                    f"Embedding Model: {eval_embedding_model}"
                    f"Metrics: {[metric.name for metric in self.metrics]}")
        
        #with ragas we need to use asynchronous context for evaluation
        #if async is not available ragas will not function properly and will raise errors.
        
    async def aevaluate(self,question:str,
                        answer:str,
                        contexts:list[str]) -> dict[str,any]:
        """execute async ragas evaluation"""
        logger.debug(f"Starting RAGAS evaluation...for question: {question[:100]}")
        start_time = time.time()
        
        try: 
            #prep dataset for ragas
            dataset = self._prepare_dataset(question,answer,contexts)
            
            #run evaluation in thread pool to avoid blocking event loop
            result = await asyncio.to_thread(
                self._evaluate_with_timeout,
                dataset,)
            
            evaluation_time_ms = (time.time() - start_time) * 1000
            
            #extract scores 
            scores = {"faithfulness": float(result["faithfulness"] if "faithfulness" in result else None),
                        "answer_relevancy": float(result["answer_relevancy"] if "answer_relevancy" in result else None),
                        "evaluation_time_ms": round(evaluation_time_ms,2),
                        "error": None}
            if self.settings.ragas_log_results:
                logger.info(
                    f"Evaluation completed - "
                    f"faithfulness={scores['faithfulness']}, "
                    f"answer_relevancy={scores['answer_relevancy']}, "
                    f"time={scores['evaluation_time_ms']}ms"
                )
   

        except Exception as e:
            logger.warning(f"Evaluation failed: {e}", exc_info=True)
            return self._handle_evaluation_error(e)
        return scores

    def _prepare_dataset(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> Dataset:
        """Convert RAG output to RAGAS Dataset format.

        Args:
            question: The user's question
            answer: The generated answer
            contexts: List of retrieved context documents

        Returns:
            Dataset object for RAGAS evaluation
        """
        # RAGAS expects data in specific format
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],  # List of lists
        }

        logger.debug(
            f"Prepared dataset with {len(contexts)} contexts " f"for question: {question[:50]}..."
        )

        return Dataset.from_dict(data)

    def _evaluate_with_timeout(self, dataset: Dataset) -> dict[str, Any]:
        """Execute RAGAS evaluation with timeout.

        Args:
            dataset: Prepared RAGAS dataset

        Returns:
            Evaluation results dictionary

        Raises:
            TimeoutError: If evaluation exceeds timeout
        """
        # Note: asyncio.timeout would be ideal, but RAGAS evaluate() is sync
        # For now, we rely on the async wrapper and trust RAGAS to complete
        # In production, consider using signal.alarm or threading.Timer
        result = evaluate(
            dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embedding_model,
        )

        # Convert to dictionary and extract scores
        return result.to_pandas().to_dict("records")[0]

    def _handle_evaluation_error(self, error: Exception) -> dict[str, Any]:
        """Return safe fallback scores on error.

        Args:
            error: The exception that occurred

        Returns:
            Dictionary with null scores and error message
        """
        logger.error(f"Returning fallback scores due to error: {error}")

        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "evaluation_time_ms": None,
            "error": str(error),
        }                    