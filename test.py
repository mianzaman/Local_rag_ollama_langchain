import logging
import yaml
import os
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# Updated imports for LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_application.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration class for RAG application parameters"""
    pdf_dir: str
    chunk_size: int
    chunk_overlap: int
    embedding_model_name: str
    llm_model_name: str
    temperature: float
    retriever_k: int
    max_workers: int
    faiss_index_path: str
    batch_size: int

    @classmethod
    def from_yaml(cls, config_path: str) -> 'RAGConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

class SentenceTransformerEmbeddings(Embeddings):
    """LangChain Embeddings interface implementation using sentence-transformers."""
    
    def __init__(self, model_name: str):
        """Initialize the embedding model.
        
        Args:
            model_name (str): Name of the sentence-transformers model to use
        """
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents.
        
        Args:
            texts (List[str]): List of text documents to embed
            
        Returns:
            List[List[float]]: List of embeddings, one for each document
        """
        embeddings = self.model.encode(texts, batch_size=32, convert_to_tensor=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text.
        
        Args:
            text (str): Query text to embed
            
        Returns:
            List[float]: Embedding vector for the query
        """
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

class DocumentProcessor:
    """Handles document loading and processing"""
    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )

    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Process a single PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List[Document]: List of processed document chunks
        """
        try:
            logger.info(f"Processing file: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            # Ensure metadata is preserved during splitting
            splits = self.text_splitter.split_documents(docs)
            for split in splits:
                split.metadata.update({
                    'source': Path(pdf_path).name,
                    'page': split.metadata.get('page', None)
                })
            
            return splits
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return []

    def process_all_documents(self) -> List[Document]:
        """Process all PDF documents in parallel"""
        pdf_files = [
            os.path.join(self.config.pdf_dir, f)
            for f in os.listdir(self.config.pdf_dir)
            if f.endswith('.pdf')
        ]
        
        doc_splits = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            results = list(executor.map(self.process_pdf, pdf_files))
            for splits in results:
                doc_splits.extend(splits)
                
        logger.info(f"Processed {len(doc_splits)} total document chunks")
        return doc_splits

class VectorStoreManager:
    """Manages vector store operations"""
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedding_function = SentenceTransformerEmbeddings(config.embedding_model_name)
        
    def load_or_create(self, documents: List[Document]) -> FAISS:
        """Load existing vector store or create new one"""
        store_path = Path(self.config.faiss_index_path)
        index_path = store_path / "index.faiss"
        store_path.mkdir(exist_ok=True)
        
        if index_path.exists():
            try:
                vectorstore = FAISS.load_local(
                    store_path.as_posix(),
                    self.embedding_function,
                    "index.faiss"
                )
                logger.info("FAISS index loaded successfully")
                return vectorstore
            except Exception as e:
                logger.error(f"Error loading FAISS index: {str(e)}")
                
        logger.info("Creating new FAISS index")
        vectorstore = FAISS.from_documents(
            documents,
            self.embedding_function
        )
        
        vectorstore.save_local(store_path.as_posix(), "index.faiss")
        logger.info("FAISS index created and saved successfully")
        
        return vectorstore

class RAGApplication:
    """Main RAG application class"""
    def __init__(self, config_path: str):
        self.config = RAGConfig.from_yaml(config_path)
        self.setup_components()
        
    def setup_components(self):
        """Initialize all components"""
        # Process documents
        doc_processor = DocumentProcessor(self.config)
        documents = doc_processor.process_all_documents()
        
        # Setup vector store
        vector_store_manager = VectorStoreManager(self.config)
        vectorstore = vector_store_manager.load_or_create(documents)
        
        # Setup retriever
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config.retriever_k}
        )
        
        # Setup LLM and prompt
        prompt = PromptTemplate(
            template="""You are an assistant for question-answering tasks.
            Use the following documents to answer the question.
            If you don't know the answer, just say that you don't know.
            Use three sentences maximum and keep the answer concise.
            Include confidence score (low/medium/high) based on document relevance:
            Question: {question}
            Documents: {documents}
            Answer:
            """,
            input_variables=["question", "documents"]
        )
        
        llm = ChatOllama(
            model=self.config.llm_model_name,
            temperature=self.config.temperature
        )
        
        self.chain = prompt | llm | StrOutputParser()
        
    @lru_cache(maxsize=100)
    def get_answer(self, question: str) -> Dict[str, Any]:
        """Get answer for a question with caching"""
        try:
            # Retrieve relevant documents
            results = self.retriever.invoke(question)
            doc_texts = "\n".join([doc.page_content for doc in results])
            
            # Get answer from LLM
            answer = self.chain.invoke({
                "question": question,
                "documents": doc_texts
            })
            
            return {
                "answer": answer,
                "source_documents": [doc.metadata for doc in results],
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                "answer": "An error occurred while processing your question.",
                "error": str(e),
                "status": "error"
            }

if __name__ == "__main__":
    # Initialize and use the RAG application
    rag_app = RAGApplication("config.yaml")
    
    # Example question
    question = "Under what conditions can the Central Weapons Office issue a joint permit for temporary movement of weapons for sports shooters?"
    result = rag_app.get_answer(question)
    
    if result["status"] == "success":
        print(f"Question: {question}")
        print(f"Answer: {result['answer']}")
        print("Source documents:", result["source_documents"])
    else:
        print(f"Error: {result['error']}")