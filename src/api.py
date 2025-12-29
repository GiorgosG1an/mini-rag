import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from src.retrieval import VectorStoreManager
from src.generation import RAGGenerator, GenerationConfig

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager.
    Stores models in 'app.state' instead of global variables.
    """
    # --- STARTUP LOGIC ---
    logger.info("Loading system models...")
    
    # 1. Load Vector Store
    try:
        vector_store = VectorStoreManager()
        vector_store.load_index("data/vector_store")
        app.state.vector_store = vector_store # Attach to App State
        logger.info("Vector Store loaded successfully.")
    except Exception as e:
        logger.error(f"FATAL: Could not load index. {e}")
        app.state.vector_store = None

    # 2. Load Generator
    try:
        gen_config = GenerationConfig()
        generator = RAGGenerator(gen_config)
        app.state.generator = generator # Attach to App State
        logger.info("Generator loaded successfully.")
    except Exception as e:
        logger.error(f"FATAL: Could not load generator. {e}")
        app.state.generator = None
    
    yield
    
    # --- SHUTDOWN LOGIC ---
    logger.info("Shutting down. Cleaning up resources...")
    app.state.vector_store = None
    app.state.generator = None
    logger.info("System shutdown complete.")

# Initialize App with lifespan
app = FastAPI(title="Mini-RAG API", version="1.0.0", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    fetch_k: int = 20

class QueryResponse(BaseModel):
    answer: str
    sources: list
    latency_seconds: float

@app.post("/predict", response_model=QueryResponse)
async def predict(payload: QueryRequest, request: Request):
    start_time = time.time()
    
    # Access models safely from App State
    vector_store = getattr(request.app.state, "vector_store", None)
    generator = getattr(request.app.state, "generator", None)
    
    # Explicit check
    if not vector_store:
        logger.error("Attempted to predict but Vector Store is missing.")
        raise HTTPException(status_code=503, detail="Vector Store not loaded")
    
    if not generator:
        logger.error("Attempted to predict but Generator is missing.")
        raise HTTPException(status_code=503, detail="Generator not loaded")

    # 1. Retrieve
    try:
        results = vector_store.search_with_rerank(payload.query, k=payload.top_k, fetch_k=20)
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Retrieval failed")

    if not results or results[0][1] < 0.0:
        logger.info("Low confidence retrieval. Aborting generation.")
        return QueryResponse(
            answer="I cannot find information about that in the document.",
            sources=[],
            latency_seconds=time.time() - start_time
        )
    context_strs = [r[0].text for r in results]
    
    # 2. Generate
    try:
        answer = generator.generate(payload.query, context_strs)
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail="Generation failed")
    
    # 3. Format Response
    sources = [
        {"page": r[0].page_number, "score": float(r[1]), "preview": r[0].text[:50]} 
        for r in results
    ]
    
    latency = time.time() - start_time
    logger.info(f"Processed query in {latency:.2f}s")
    
    return QueryResponse(
        answer=answer,
        sources=sources,
        latency_seconds=latency
    )