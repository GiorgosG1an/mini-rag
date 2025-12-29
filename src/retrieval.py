import logging
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer

from src.ingestion import ProcessedChunk

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Manages the lifecycle of the FAISS index and embeddings.

    Implements the 'Build Once, Read Many' pattern.
    """

    def __init__(self, model_name: str= 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        
        # Lazy loading
        self._model = None

        self.index = None
        self.chunks: List[ProcessedChunk] = []

    @property
    def model(self):
        """Singleton access to the model to save memory."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def create_index(self, chunks: List[ProcessedChunk]):
        """
        1. Embeds all text chunks.
        2. Normalizes vectors (crucial for Cosine Similarity).
        3. Builds a FAISS index.
        """

        if not chunks:
            logger.warning("No chunks provided to index.")
            return
        
        texts = [chunk.text for chunk in chunks]

        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        # Normalize vectors to unit length
        faiss.normalize_L2(embeddings)
        
        # dimension check, should be 384
        d = embeddings.shape[1]
        logger.info(f"Embedding dimension: {d}")

        # create FAISS Index
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)

        self.chunks = chunks
        logger.info("Index built successfully.")

    def save_index(self, folder_path: str):
        """
        Persists the index and metadata to disk.
        """
        path = Path(folder_path)
        path.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(path / "index.faiss"))

        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        logger.info(f"Vector store saved to {folder_path}")

    def load_index(self, folder_path: str):
        """
        Loads the index from disk (Fast startup).
        """

        path = Path(folder_path)
        if not (path / "index.faiss").exists():
            raise FileNotFoundError("Index file not found. Run create_index first.")
        
        self.index = faiss.read_index(str(path / "index.faiss"))

        with open(path / "metadata.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        logger.info(f"Loaded index with {self.index.ntotal} vectors.")


    def search(self, query: str, k: int = 3) -> List[Tuple[ProcessedChunk, float]]:
        """
        Semantic search:
        1. Embed query.
        2. Normalize.
        3. Search index.
        4. Return top-k chunks with scores.
        """

        if self.index is None:
            raise ValueError("Index not loaded.")
        
        # Embed and normalize query
        query_vector = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)

        # Search
        # D = distances (scores), I = indices of nearest neighbors
        D, I = self.index.search(query_vector, k)

        results = []
        for i in range(k):
            idx = I[0][i] # ID of the chunk
            score = D[0][i] # the similarity score

            if idx != -1: # FAISS returns -1 if not found
                results.append((self.chunks[idx], float(score)))
        
        return results

if __name__ == "__main__":
    from ingestion import IngestionConfig, PDFIngestionPipeline
    
    # 1. Re-run ingestion (or load from cache if you had one)
    config = IngestionConfig()
    pipeline = PDFIngestionPipeline(config)
    chunks = pipeline.process("data/raw/attention_paper.pdf") # data\raw\attention_paper.pdf
    
    # 2. Build Vector Store
    vector_store = VectorStoreManager()
    vector_store.create_index(chunks)
    
    # 3. Save it (Simulating the 'Build' phase)
    vector_store.save_index("data/vector_store")
    
    # 4. Test Search
    test_query = "How is the scaled dot-product attention calculated?"
    print(f"\nQuery: {test_query}")
    
    results = vector_store.search(test_query, k=2)
    
    for i, (chunk, score) in enumerate(results):
        print(f"\nResult {i+1} (Score: {score:.4f}):")
        print(f"Source: Page {chunk.page_number}")
        print(f"Excerpt: {chunk.text[:150]}...")