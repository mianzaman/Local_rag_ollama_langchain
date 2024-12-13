import logging
import yaml
import os
import torch
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from transformers import AutoTokenizer, AutoModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer, models
import re
import spacy
from PyPDF2 import PdfReader


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_tax_application.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    logger.warning("CUDA not available. Running on CPU.")

@dataclass
class RAGConfig:
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
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        try:
            # Load the model directly using SentenceTransformers
            self.model = SentenceTransformer(model_name)
        except Exception:
            print(f"Model '{model_name}' not found in SentenceTransformers. Loading manually from Hugging Face...")
            # Load the transformer and tokenizer from Hugging Face
            transformer_model = AutoModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Create SentenceTransformer model with pooling
            word_embedding_model = models.Transformer(model_name, model=transformer_model, tokenizer=tokenizer)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
            self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        self.batch_size = 32

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, batch_size=self.batch_size, convert_to_tensor=True).cpu().numpy().tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_tensor=True).cpu().numpy()[0].tolist()


class GPUDocumentProcessor:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", ". ", "\n", " "]
        )
        # Load NLP model for entity recognition
        self.nlp = spacy.load("en_core_web_sm")

    def extract_metadata(self, pdf_path: str, text: str) -> Dict[str, any]:
        """Dynamically extract metadata from a PDF."""
        metadata = {
            "file_name": Path(pdf_path).name,
            "file_path": pdf_path,
        }
        
        # Extract case number
        case_number_match = re.search(r"\d{2}\.\d{4}\.\d{3}", text)
        metadata["case_number"] = case_number_match.group(0) if case_number_match else None

        # Extract dates
        date_match = re.findall(r"\d{2}\.\d{2}\.\d{4}", text)
        metadata["dates"] = date_match if date_match else None

        # Extract named entities for parties
        doc = self.nlp(text)
        parties = {"plaintiff": None, "defendant": None}
        for ent in doc.ents:
            if ent.label_ == "ORG":  # Adjust as per document content
                if not parties["plaintiff"]:
                    parties["plaintiff"] = ent.text
                elif not parties["defendant"]:
                    parties["defendant"] = ent.text
        metadata["parties"] = parties

        # Extract referenced laws
        laws_match = re.findall(r"art\.\s\d+", text, re.IGNORECASE)
        metadata["laws_referenced"] = laws_match if laws_match else None

        return metadata

    def process_pdf(self, pdf_path: str) -> List[Document]:
        try:
            logger.info(f"Processing file: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            combined_text = " ".join([doc.page_content for doc in docs])

            # Extract metadata dynamically
            metadata = self.extract_metadata(pdf_path, combined_text)

            # Split documents into chunks
            splits = self.text_splitter.split_documents(docs)
            for split in splits:
                split.metadata.update(metadata)  # Add metadata to each chunk
                split.metadata.update({
                    'source': Path(pdf_path).name,
                    'page': split.metadata.get('page', None)
                })
            return splits
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return []

    def process_all_documents(self) -> List[Document]:
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


class GPUVectorStoreManager:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedding_function = SentenceTransformerEmbeddings(config.embedding_model_name)

    def load_or_create(self, documents: List[Document]) -> FAISS:
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
    def __init__(self, config_path: str):
        self.config = RAGConfig.from_yaml(config_path)
        self.setup_components()

    def setup_components(self):
        doc_processor = GPUDocumentProcessor(self.config)
        documents = doc_processor.process_all_documents()

        vector_store_manager = GPUVectorStoreManager(self.config)
        vectorstore = vector_store_manager.load_or_create(documents)

        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config.retriever_k}
        )

        prompt = PromptTemplate(
            template="""
                You are a legal assistant specialized in tax law. Use the provided legal documents to answer the user's question.
                Provide concise, accurate answers based on tax-related cases. Include the case name and page number if applicable.
                Use the provided documents to answer the user's question.
                
                Question: {question}
                Documents: {documents}
                If no relevant information is found in the documents, respond: 'No information about restitution payments was found in the provided documents.'
                Answer:
                """,
            input_variables=["question", "documents"]
        )

        llm = ChatOllama(
            model=self.config.llm_model_name,
            temperature=self.config.temperature,
            max_tokens=150,
            gpu=True
        )

        self.chain = prompt | llm | StrOutputParser()

    @lru_cache(maxsize=100)
    def get_answer(self, question: str) -> Dict[str, Any]:
        try:
            results = self.retriever.invoke(question)
            doc_texts = "\n".join([doc.page_content[:500] for doc in results[:5]])

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
    rag_app = RAGApplication("config.yaml")

    question = "In which cases were the defendants ordered to make restitution payments?"
    result = rag_app.get_answer(question)

    if result["status"] == "success":
        print(f"Question: {question}")
        print(f"Answer: {result['answer']}")
        print("\nSource documents:")
        for source in result["source_documents"]:
            print(f"- File: {source['source']}, Page: {source.get('page', 'N/A')}")
    else:
        print(f"Error: {result['error']}")
