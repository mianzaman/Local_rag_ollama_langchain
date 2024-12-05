from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS - Important for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class QuestionRequest(BaseModel):
    question: str

class SourceDocument(BaseModel):
    page: Optional[int] = None
    source: Optional[str] = None

class QuestionResponse(BaseModel):
    answer: str
    source_documents: List[SourceDocument]
    status: str = "success"
    error: Optional[str] = None

# Initialize RAG application at startup
@app.on_event("startup")
async def startup_event():
    global rag_app
    try:
        rag_app = RAGApplication("config.yaml")
        logger.info("RAG application initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG application: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "rag_status": "initialized" if 'rag_app' in globals() else "not initialized"}

@app.post("/api/chat", response_model=QuestionResponse)
async def chat_endpoint(request: QuestionRequest):
    """Main endpoint for chat functionality"""
    logger.info(f"Received question: {request.question}")
    
    try:
        # Get response from RAG
        result = rag_app.get_answer(request.question)
        
        if result["status"] == "error":
            logger.error(f"RAG error: {result['error']}")
            raise HTTPException(status_code=500, detail=result["error"])
            
        logger.info("Successfully processed question")
        return {
            "answer": result["answer"],
            "source_documents": result["source_documents"],
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return {
            "answer": "An error occurred while processing your question.",
            "source_documents": [],
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)