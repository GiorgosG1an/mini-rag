import json
import logging
import nltk
from typing import List

from src.retrieval import VectorStoreManager
from src.generation import RAGGenerator, GenerationConfig

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(self, vector_store: VectorStoreManager, generator: RAGGenerator):
        self.vector_store = vector_store
        self.generator = generator

    def evaluate_retrieval(self, question: str, expected_page: int, k=3) -> int:
        """
        Returns 1 if the expected page is in the top-k chunks, 0 otherwise.
        """
        results = self.vector_store.search(question, k=k)
        retrieved_pages = [chunk.page_number for chunk, _ in results]

        hit = 1 if expected_page in retrieved_pages else 0
        logger.info(f"Q: '{question[:30]}...' | Expected Page: {expected_page} | Found: {retrieved_pages} | Hit: {hit}")

        return hit
    
    def evaluate_faithfullness_llm(self, answer: str, context: List[str]) -> int:
        """
        LLM-as-a-judge.

        Uses the FLAN-T5 model to check if the premise entails the hypothesis.

        Returns 1 (Yes) or 0 (No).
        """
        context_text = " ".join(context)
        prompt = (
            f"Premise: {context_text}\n\n"
            f"Hypothesis: {answer}\n\n"
            "Does the premise entail the hypothesis? Answer Yes or No."
        )

        result = self.generator.pipeline(prompt, max_new_tokens=5)
        generated_text = result[0]['generated_text'].lower().strip()

        if "yes" in generated_text:
            return 1
        return 0
    
    def run_benchmark(self, dataset_path: str):
        """
        Runs the full evaluation suite.
        """
        with open(dataset_path, 'r') as f:
            data = json.load(f)

        total_queries = len(data)
        retrieval_hits = 0
        faithfulness_scores = []

        print("\n--- 📊 Starting Benchmark ---")

        for item in data:
            q = item['question']
            
            # 1. Retrieval Step
            results = self.vector_store.search(q, k=3)
            context_chunks = [r[0].text for r in results]
            retrieved_pages = [r[0].page_number for r in results]

            # 2. Generation Step
            answer = self.generator.generate(q, context_chunks)

            # 3. Score
            if item["expected_page_match"] in retrieved_pages:
                retrieval_hits += 1
            
            faith_score = self.evaluate_faithfullness_llm(answer, context_chunks)
            faithfulness_scores.append(faith_score)

            print(f"Q: {q}")
            print(f"A: {answer}")
            print(f"Create Faithfulness: {faith_score:.2f} | Page Hit: {item['expected_page_match'] in retrieved_pages}")
            print("-" * 30)
        
        # Summary Stats
        avg_faithfulness = sum(faithfulness_scores) / total_queries
        print("\n=== 🏆 FINAL REPORT ===")
        print(f"Retrieval Hit Rate (Recall@3): {retrieval_hits}/{total_queries} ({retrieval_hits/total_queries:.1%})")
        print(f"Avg Faithfulness Score: {avg_faithfulness:.2f}")
        print("=======================")

if __name__ == "__main__":
    # Load your existing system components
    vector_store = VectorStoreManager()
    vector_store.load_index("data/vector_store")
    
    gen_config = GenerationConfig() # defaults
    generator = RAGGenerator(gen_config)
    
    # Run Eval
    evaluator = RAGEvaluator(vector_store, generator)
    evaluator.run_benchmark("data/golden_dataset.json")