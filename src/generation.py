import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TF warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Fix the floating point warning

import logging
import torch
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class GenerationConfig(BaseModel):
    model_id: str = "google/flan-t5-large"
    max_length: int = 512
    temperature: float = 0.0  # 0.0 for deterministic facts, higher for creativity
    device: str = Field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

class RAGGenerator:
    """
    The Generation component.

    Wraps an HF model to perform Context-Aware Question Answering.
    """
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.pipeline = self._load_model()

    def _load_model(self):
        logger.info(f"Loading Generator Model: {self.config.model_id} on {self.config.device}")

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_id)

        return pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if self.config.device == "cuda" else -1,
            max_length=self.config.max_length
        )
    
    def _format_prompt(self, question: str, context_chunks: List[str]) -> str:
        """
        Constructs the prompt
        """
        context_text = "\n---\n".join(context_chunks)

        prompt = f"""
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context:
        {context_text}

        Question: {question}
        
        Answer:
        """

        return prompt
    
    def generate(self, question: str, context_chunks: List[str]) -> str:
        """
        1. Formats prompt
        2. Runs inference
        3. Cleans output
        """

        prompt = self._format_prompt(question, context_chunks)

        gen_kwargs = {
            'do_sample' : False if self.config.temperature == 0 else True,
            'temperature' : self.config.temperature if self.config.temperature > 0 else None,
            "max_new_tokens": 200 # Restrict answer length
        }

        result = self.pipeline(prompt, **gen_kwargs)

        return result[0]['generated_text']

if __name__ == "__main__":
    # Mock data to test generation in isolation (Unit Test style)
    mock_context = [
        "The transformer model uses self-attention mechanisms.",
        "It was introduced in the paper 'Attention Is All You Need' by Google researchers."
    ]
    
    config = GenerationConfig()
    generator = RAGGenerator(config)
    
    q = "Who introduced the transformer model?"
    print(f"\nQuestion: {q}")
    print(f"Answer: {generator.generate(q, mock_context)}")
