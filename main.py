import logging

from src.ingestion import IngestionConfig, PDFIngestionPipeline
from src.retrieval import VectorStoreManager
from src.generation import GenerationConfig, RAGGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

def run_rag_pipeline(pdf_path: str, query: str):
    # 1. Ingestion (Check if we need to ingest)
    vector_store = VectorStoreManager()

    try:
        vector_store.load_index('data/vector_store')
    except Exception:
        print("Index not found. Building...")
        ingest_config = IngestionConfig()
        pipeline = PDFIngestionPipeline(ingest_config)
        chunks = pipeline.process(pdf_path)
        vector_store.create_index(chunks)
        vector_store.save_index("data/vector_store")

    # 2. Retrieval
    print(f"\nSearching for: '{query}'...")
    retrieved_items = vector_store.search(query, k=3)

    context_chunks = [item[0].text for item in retrieved_items]

    # 3. Generation
    gen_config = GenerationConfig()
    generator = RAGGenerator(gen_config)

    answer = generator.generate(query, context_chunks)

    # 4. Display Results
    print("\n" + "="*50)
    print(f"🤖 QUESTION: {query}")
    print("="*50)
    print(f"💡 ANSWER: {answer}")
    print("-" * 50)
    print("📚 SOURCES:")

    for i, (chunk, score) in enumerate(retrieved_items):
        print(f"[{i+1}] Page {chunk.page_number} (Score: {score:.4f})")
    print("="*50)

if __name__ == "__main__":
       
    my_query = "What is the formula for Scaled Dot-Product Attention?"
        
    run_rag_pipeline("data/raw/attention_paper.pdf", my_query)