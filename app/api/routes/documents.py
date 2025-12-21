from fastapi import APIRouter , File , HTTPException ,UploadFile 

from app.api.schemas import (
    DocumentListResponse, DocumentUploadResponse,
    ErrorResponse 
)

from app.core.document_processor import DocumentProcessor
from app.core.vector_store import VectorStoreService 
from app.utils.logger import get_logger


logger = get_logger(__name__)
router = APIRouter(prefix="/documents",tags=['Documents'])

@router.post("/upload",response_model=DocumentUploadResponse,
             responses={400:{"model":ErrorResponse,"description":"invalid file type"},
                        500:{"model":ErrorResponse,"description":"processing error"},},
             summary="upload and ingest a document",
             description="Upload and process a document file")
async def upload_document(
    file:UploadFile = File(...,description="Document file to upload")
    ) -> DocumentUploadResponse:    
    """API endpoint to upload and ingest a document."""

    logger.info(f"Received upload request for file: {file.filename}")
    #file.filename will work here because UploadFile has a filename attribute
    
    #validation for filename as its critical for processing
    if not file.filename:
        logger.error("No filename provided in the upload request.")
        raise HTTPException(status_code=400,detail="Filename is required.")
    
    try:
        #process document
        processor = DocumentProcessor()
        chunks = await processor.process_upload(file.file,file.filename)
        
        if not chunks:
            logger.error("No chunks were created from the uploaded document.")
            raise HTTPException(status_code=400,detail="No content was extracted from the document.")
        
        # add to vector store
        vector_store = VectorStoreService()
        document_ids = await vector_store.add_documents(chunks)
        
        logger.info(f"Document '{file.filename}' uploaded and processed successfully."
                    f"Chunks created: {len(chunks)}")
        
        return DocumentUploadResponse(
            message="Document uploaded and processed successfully.",
            filename=file.filename,
            chunks_created=len(chunks),
            document_ids=document_ids,)
    
    except ValueError as e:
        logger.warning(f"error processing the document: {e}")
        raise HTTPException(status_code=400,detail=str(e))
    except Exception as e:
        logger.error(f"error processing the document: {e}", exc_info=True)
        raise HTTPException(status_code=500,detail="Internal server error during document upload.")
    
@router.get(
    "/info",
    response_model=DocumentListResponse,
    summary="Get collection information",
    description="Get information about the document collection.",
)
async def get_collection_info() -> DocumentListResponse:
    """Get information about the document collection."""
    logger.debug("Collection info requested")

    try:
        vector_store = VectorStoreService()
        info = vector_store.get_collection_info()

        return DocumentListResponse(
            collection_name=info["name"],
            total_documents=info["points_count"],
            status=info["status"],
        )
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting collection info: {str(e)}",
        )


@router.delete(
    "/collection",
    responses={
        200: {"description": "Collection deleted successfully"},
        500: {"model": ErrorResponse, "description": "Deletion error"},
    },
    summary="Delete the entire collection",
    description="Delete all documents from the vector store. Use with caution!",
)
async def delete_collection() -> dict:
    """Delete the entire document collection."""
    logger.warning("Collection deletion requested")

    try:
        vector_store = VectorStoreService()
        vector_store.delete_collection()

        return {"message": "Collection deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting collection: {str(e)}",
        )
