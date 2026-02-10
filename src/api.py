"""
api.py - FastAPI REST API for RAG System
Run with: uvicorn src.api:app --reload
"""

import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from retrieval import RAGRetriever

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG Q&A API",
    description="REST API for Retrieval Augmented Generation Question Answering",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global retriever instance
retriever: Optional[RAGRetriever] = None


# Pydantic models
class QueryRequest(BaseModel):
    """Request model for queries"""
    question: str = Field(..., min_length=1, description="The question to ask")
    top_k: Optional[int] = Field(None, ge=1, le=10, description="Number of documents to retrieve")


class SourceDocument(BaseModel):
    """Model for source document"""
    content: str
    source: str
    page: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for queries"""
    answer: str
    sources: List[SourceDocument]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str


class StatsResponse(BaseModel):
    """System statistics response"""
    vectorstore_path: str
    embedding_model: str
    llm_model: str
    top_k: int
    temperature: float


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global retriever
    
    try:
        print("🚀 Initializing RAG system...")
        
        vectorstore_path = os.getenv("VECTORSTORE_PATH", "./vectorstore")
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        llm_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
        top_k = int(os.getenv("TOP_K_RESULTS", "3"))
        
        retriever = RAGRetriever(
            vectorstore_path=vectorstore_path,
            embedding_model=embedding_model,
            llm_model=llm_model,
            temperature=temperature,
            top_k=top_k
        )
        
        retriever.initialize()
        print("✅ RAG system initialized successfully")
        
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        print("   Make sure you've run 'python src/indexing.py' first!")
        retriever = None


# Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Q&A API",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "Ask a question",
            "GET /health": "Health check",
            "GET /stats": "System statistics"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized"
        )
    
    return HealthResponse(
        status="healthy",
        message="RAG system is running"
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics"""
    if retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized"
        )
    
    return StatsResponse(
        vectorstore_path=retriever.vectorstore_path,
        embedding_model=retriever.embedding_model,
        llm_model=retriever.llm_model,
        top_k=retriever.top_k,
        temperature=retriever.temperature
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system
    
    Args:
        request: QueryRequest containing the question
        
    Returns:
        QueryResponse with answer and sources
    """
    if retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized. Please check server logs."
        )
    
    try:
        # Override top_k if specified
        if request.top_k:
            original_top_k = retriever.top_k
            retriever.top_k = request.top_k
        
        # Get answer
        answer, sources = retriever.query(request.question)
        
        # Restore original top_k
        if request.top_k:
            retriever.top_k = original_top_k
        
        # Format response
        source_docs = [
            SourceDocument(
                content=doc.page_content,
                source=doc.metadata.get('source', 'Unknown'),
                page=str(doc.metadata.get('page', 'N/A'))
            )
            for doc in sources
        ]
        
        return QueryResponse(
            answer=answer,
            sources=source_docs
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


# Run with: uvicorn src.api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
