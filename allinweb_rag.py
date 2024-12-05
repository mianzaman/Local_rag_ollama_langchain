from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import os
import pickle

# Define the directory containing PDF files
pdf_dir = r"E:\allinweb\Local_rag_ollama_langchain\data"

# Initialize a text splitter with specified chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=150, chunk_overlap=50
)

# Initialize variables for processing
doc_splits = []

# Process each PDF file individually to avoid memory overflow
for file in os.listdir(pdf_dir):
    if file.endswith(".pdf"):
        print(f"Processing file: {file}")
        loader = PyPDFLoader(os.path.join(pdf_dir, file))
        docs = loader.load()  # Load documents from the PDF
        # Split documents into smaller chunks
        doc_splits.extend(text_splitter.split_documents(docs))

print(f"Total document chunks: {len(doc_splits)}")

# Use a local embedding model (e.g., 'all-MiniLM-L6-v2')
local_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define a wrapper class for embeddings
class LocalEmbeddingWrapper:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        # Method for embedding documents
        return self.model.encode(texts, batch_size=32, convert_to_tensor=False)

    def embed_query(self, text):
        # Method for embedding a query
        return self.model.encode(text, convert_to_tensor=False)

    # Make the class instance callable, so FAISS can invoke it directly
    def __call__(self, text):
        return self.embed_query(text)

# Instantiate the embedding wrapper
embedding_function = LocalEmbeddingWrapper(local_embedding_model)

# Create or load the FAISS vector database
faiss_index_path = "faiss_index"
faiss_store_path = f"{faiss_index_path}/faiss_store.pkl"

# Check if FAISS index file exists and is not empty
if os.path.exists(faiss_store_path) and os.path.getsize(faiss_store_path) > 0:
    # Load the existing FAISS index
    with open(faiss_store_path, "rb") as f:
        vectorstore = pickle.load(f)
    print("FAISS index loaded successfully.")
else:
    # Create a new FAISS index and save it
    vectorstore = FAISS.from_texts([chunk.page_content for chunk in doc_splits], embedding_function)
    # Create the directory if it doesn't exist
    os.makedirs(faiss_index_path, exist_ok=True)
    with open(f"{faiss_index_path}/faiss_store.pkl", "wb") as f:
        pickle.dump(vectorstore, f)
    print("FAISS index created and saved successfully.")

# Define a retriever using FAISS
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Define the prompt template for the LLM
prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks.
    Use the following documents to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise:
    Question: {question}
    Documents: {documents}
    Answer:
    """,
    input_variables=["question", "documents"],
)

# Initialize the LLM with a local model
llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)

# Create a chain combining the prompt template and LLM
rag_chain = prompt | llm | StrOutputParser()

# Define the RAG application class
class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain

    def run(self, question):
        # Retrieve relevant documents
        results = self.retriever.invoke(question)
        # Extract content from retrieved documents (get the page_content from Document objects)
        doc_texts = "\n".join([doc.page_content for doc in results])
        # Get the answer from the language model
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        return answer

# Initialize the RAG application
rag_application = RAGApplication(retriever, rag_chain)

# Example usage
question = "What are the guest categories that organizers can form under the amended Article 11 of the Military Sports Ordinance?"
answer = rag_application.run(question)
print("Question:", question)
print("Answer:", answer)
