import time
import logging
from contextlib import asynccontextmanager
from typing import List, Literal

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from execution.search_core import search, get_vstore
from execution.config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("search_api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load embeddings and FAISS index into memory
    logger.info("Starting up Search Engine API...")
    try:
        get_vstore()
        logger.info("Successfully loaded VectorStore models and FAISS index.")
    except Exception as e:
        logger.error(f"Failed to load VectorStore: {e}")
        
    yield
    
    # Shutdown
    logger.info("Shutting down Search Engine API...")

app = FastAPI(
    title="AntiSearch API", 
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "message": str(exc)},
    )

# Models
class SearchResult(BaseModel):
    doc_id: int
    url: str
    title: str
    snippet: str
    score: float
    bm25_score: float
    pagerank: float

class SearchResponse(BaseModel):
    query: str
    mode: str
    results: List[SearchResult]
    total_results: int
    execution_time_ms: float

@app.get("/api/health")
def api_health():
    """Health check endpoint for Docker and orchestrators."""
    return {"status": "ok", "version": app.version}

@app.get("/api/search", response_model=SearchResponse)
def api_search(
    q: str = Query(..., min_length=1, max_length=200, description="Search query"),
    mode: Literal["bm25", "semantic", "hybrid"] = Query("hybrid", description="Search mode")
):
    start_time = time.time()
    
    try:
        results = search(q, mode=mode, limit=20)
    except Exception as e:
        logger.error(f"Search error for query '{q}': {e}")
        raise HTTPException(status_code=500, detail="Error executing search")
        
    execution_time_ms = (time.time() - start_time) * 1000
    
    logger.info(f"Query: '{q}', Mode: {mode}, Results: {len(results)}, Latency: {execution_time_ms:.1f}ms")
    
    return SearchResponse(
        query=q,
        mode=mode,
        results=results,
        total_results=len(results),
        execution_time_ms=execution_time_ms
    )

@app.get("/api/stats")
def api_stats():
    import sqlite3
    try:
        conn = sqlite3.connect(settings.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM corpus_stats")
        stats = dict(cursor.fetchall())
        conn.close()
        return stats
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve stats")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("execution.api:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
