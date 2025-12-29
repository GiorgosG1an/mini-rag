import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

class IngestionConfig(BaseModel):
    """
    Configuration for the ingestion pipeline.\n
    Allows easy experimentation with chunk sizes.
    """

    chunk_size: int = Field(default=500, description='Target size of each text chunk')
    chunk_overlap: int = Field(default=50, description="Overlap between chunks to preserve context")
    separators: List[str] = Field(
        default_factory=lambda: ["\n\n", "\n", " ", ""],
        description="Priority list of separators for recursive splitting"
    )

@dataclass
class ProcessedChunk:
    """
    Standardized data class for a chunk of text.\n
    Includes metadata for citation/grounding later.
    """
    text: str
    source: str
    page_number: int
    chunk_id: int

class PDFIngestionPipeline:
    """
    Production-grade PDF ingestion pipeline.\n
    Handles loading, cleaning, and chunking.
    """
    def __init__(self, config: IngestionConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators
        )
    
    def load_pdf(self, file_path: str) -> List[dict]:
        """
        Loads a PDF and extracts text page by page.\n
        Returns a list of dicts with text and metadata.
        """

        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading PDF: {file_path}")

        try:
            reader = PdfReader(str(path))
            raw_pages = []

            for i, page in enumerate(reader.pages):
                text = page.extract_text()

                if text:
                    clean_text = " ".join(text.split()) # remove excessive whitespaces
                    raw_pages.append({'text': clean_text, 'page': i + 1})

            logger.info(f"Successfully extracted {len(raw_pages)} pages.")
            return raw_pages

        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            raise


    def process(self, file_path: str) -> List[ProcessedChunk]:
        """
        Orchestrates the full ingestion flow:

        Load -> Split -> Wrap.
        """
        raw_pages = self.load_pdf(file_path)
        final_chunks = []
        global_chunk_id = 0

        logger.info(f"Chunking with size={self.config.chunk_size}, overlap={self.config.chunk_overlap}")

        for page_data in raw_pages:
            split_texts = self.text_splitter.split_text(page_data['text'])

            for text_segment in split_texts:
                chunk = ProcessedChunk(
                    text=text_segment,
                    source=Path(file_path).name,
                    page_number=page_data['page'],
                    chunk_id=global_chunk_id
                )

                final_chunks.append(chunk)
                global_chunk_id += 1

        logger.info(f"Ingestion complete. Created {len(final_chunks)} chunks.")

        return final_chunks

if __name__ == "__main__":
    config = IngestionConfig(chunk_size=512, chunk_overlap=64)
    pipeline = PDFIngestionPipeline(config)

    pdf_path = "data/raw/attention_paper.pdf"

    try:
        chunks = pipeline.process(pdf_path)
        
        # Inspect the first chunk to verify quality
        print("\n--- Sample Chunk 0 ---")
        print(f"Source: {chunks[0].source} | Page: {chunks[0].page_number}")
        print(f"Content: {chunks[0].text[:200]}...") # Print first 200 chars
        print("-" * 30)
        
    except FileNotFoundError:
        print("Please ensure you downloaded the PDF in the previous step!")