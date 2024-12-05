import logging
import yaml
import os
import pickle
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.schema import Document

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

class DocumentProcessor:
    """Handles document loading and processing"""
    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )

    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Process a single PDF file"""
        try:
            logger.info(f"Processing file: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            return self.text_splitter.split_documents(docs)
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

class LocalEmbeddingWrapper:
    """Wrapper for local embedding model"""
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Batch embed documents"""
        return self.model.encode(texts, batch_size=32, convert_to_tensor=False)
        
    @lru_cache(maxsize=1000)
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query with caching"""
        return self.model.encode(text, convert_to_tensor=False)
        
    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)

class VectorStoreManager:
    """Manages vector store operations"""
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedding_function = LocalEmbeddingWrapper(config.embedding_model_name)
        
    def load_or_create(self, documents: List[Document]) -> FAISS:
        """Load existing vector store or create new one"""
        store_path = Path(self.config.faiss_index_path) / "faiss_store.pkl"
        
        if store_path.exists() and store_path.stat().st_size > 0:
            try:
                with open(store_path, "rb") as f:
                    vectorstore = pickle.load(f)
                logger.info("FAISS index loaded successfully")
                return vectorstore
            except Exception as e:
                logger.error(f"Error loading FAISS index: {str(e)}")
                
        # Create new index if loading fails or file doesn't exist
        logger.info("Creating new FAISS index")
        vectorstore = FAISS.from_texts(
            [doc.page_content for doc in documents],
            self.embedding_function,
            batch_size=self.config.batch_size
        )
        
        # Save new index
        store_path.parent.mkdir(exist_ok=True)
        with open(store_path, "wb") as f:
            pickle.dump(vectorstore, f)
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

# Example configuration file (config.yaml):
"""
pdf_dir: "E:/allinweb/Local_rag_ollama_langchain/data"
chunk_size: 150
chunk_overlap: 50
embedding_model_name: "all-MiniLM-L6-v2"
llm_model_name: "llama3.1"
temperature: 0
retriever_k: 4
max_workers: 4
faiss_index_path: "faiss_index"
batch_size: 32
"""

# Example usage:
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