"""Document processor module to process and chunk documents"""


import tempfile 
from pathlib import Path 
from typing import BinaryIO 


from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,     
)

from langchain_core.documents import Document
from langchain_core.text_splitter import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

class DocumentProcessor:
    """Process documents for RAG pipeline"""
    SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.csv'}
    
    def __init__(self,chunk_size:int | None =None ,
                 chunk_overlap:int |None =None):
        # Initialize text splitter with provided or default settings
        #chunk_size is of type int if not int it is set to None and default value is used
        settings = get_settings()
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        
        logger.info(f"DocumentProcessor initialized with chunk_size={self.chunk_size} and chunk_overlap={self.chunk_overlap}")  
    
    def load_pdf(self, file_path :str | Path) -> list[Document]:
        """load pdf documents from file path"""
        file_path = Path(file_path)
        logger.info(f"Loading PDF document from {file_path.name}")
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from PDF document")
        
        return documents 
    
    def load_txt(self,file_path:str |Path) -> list[Document]:
        """load text documents from file path"""
        file_path = Path(file_path)
        logger.info(f"Loading text document from {file_path.name}")
        loader = TextLoader(str(file_path), encoding='utf-8')
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} text documents")
        
        return documents
        
    def load_csv(self,file_path:str |Path) -> list[Document]:
        """load csv documents from file path"""
        file_path = Path(file_path)
        logger.info(f"Loading CSV document from {file_path.name}")
        loader = CSVLoader(str(file_path))
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} rows from CSV document")
        
        return documents    
    
    def load_file(self,file_path:str | Path) -> list[Document]:
        """Load document based on file extension"""
        file_path = Path(file_path)
        extensions = file_path.suffix.lower()
        
        if extensions not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {extensions}. Supported extensions are: {self.SUPPORTED_EXTENSIONS}"
            )
        
        loaders = {
            '.pdf': self.load_pdf,
            '.txt': self.load_txt,
            '.csv': self.load_csv,}
        
        return loaders[extensions](file_path)
    
    def load_from_upload(self,
                         file: BinaryIO,
                         file_name:str) -> list[Document]:
        """Load document from uploaded file"""
        
        #we use file type BInaryIO to read the file content
        #since file while uploaded does not have a path we create a temporary file to save the content
        #uploaded file is read and written toa temporary file
        #uploaded file is transferred as a binary stream to the temporary file
        # Binary IO is not the only way to handle file uploads but it is a common approach in web applications
        # another approach is to use frameworks that handle file uploads and provide file-like objects directly such as FASTAPI or use base64 encoding 
        # example using base64 encoding is b64encode(file.read()).decode('utf-8') , full example function if we use this approach is below
        #example function using base64 encoding
        # """base64 example
        # import base64
        # def load_from_upload_base64(file: BinaryIO, file_name: str) -> list[Document]:
        #     file_content = base64.b64encode(file.read()).decode('utf-8')
        #     # Further processing using file_content
        # """
        
        extension = Path(file_name).suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {extension}"
                f"Supported extensions are: {self.SUPPORTED_EXTENSIONS}"
                
            )
        
        #save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix=extension) as temp_file:
            temp_file.write(file.read())
            temp_path = temp_file.name

        try:
            documents = self.load_file(temp_path)
            
            for doc in documents:
                doc.metadata['source'] = file_name
            
            return documents 
        
        finally:
            #clean up temporary file
            #Path.unlink is used to delete the temporary file
            # unlike(missing_ok=True) ensures no error is raised if the file is already deleted
            Path(temp_path).unlink(missing_ok=True)
        
    
    def split_documents(self,documents:list[Document])-> list[Document]:
        """Split documents into smaller chunks"""
        
        logger.info(f"Splitting {len(documents)} documents into chunks")
        
        chunks = self.text_splitter.split_documents(documents)
        
        logger.info(f"Created {len(chunks)} document chunks")
        return chunks
    
    
    def process_file(self, file_path: str | Path) -> list[Document]:
        """Load and split a file in one step.

        Args:
            file_path: Path to file

        Returns:
            List of chunked Document objects
        """
        documents = self.load_file(file_path)
        return self.split_documents(documents)

    def process_upload(
        self,
        file: BinaryIO,
        filename: str,
    ) -> list[Document]:
        """Load and split an uploaded file.

        Args:
            file: File-like object
            filename: Original filename

        Returns:
            List of chunked Document objects
        """
        documents = self.load_from_upload(file, filename)
        return self.split_documents(documents)