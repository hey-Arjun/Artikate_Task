import os
import pandas as pd
from fpdf import FPDF
from datetime import datetime
from pipeline import RAGPipeline

def run_evaluation():
    pipeline = RAGPipeline()

    test_set = [
        {"question": "What is the difference between an agent and a workflow?", "expected_doc": "Agentic Orchestration.pdf"},
        {"question": "What are common reasons for RAG failure?", "expected_doc": "Langchian , RAG INTVW QUESTIONS.pdf"},
        {"question": "How does memory work in multi-agent designs?", "expected_doc": "Agentic Orchestration.pdf"},
        {"question": "Why should tools be registered dynamically?", "expected_doc": "Langchian , RAG INTVW QUESTIONS.pdf"},
        {"question": "What is the role of a router chain?", "expected_doc": "Agentic Orchestration.pdf"}
    ]

    detailed_results = []
    hits = 0
    total = len(test_set)

    print(f"Starting Section 2 Evaluation (Precision@3)...")

    for i, item in enumerate(test_set):
        result = pipeline.query(item['question'])
        retrieved_docs = [s["document"] for s in result["sources"]]
        
        is_hit = item["expected_doc"] in retrieved_docs
        if is_hit: hits += 1

        detailed_results.append({
            "Test ID": i + 1,
            "Question": item['question'],
            "Expected": item['expected_doc'],
            "Retrieved": ", ".join(retrieved_docs),
            "Status": "PASS" if is_hit else "FAIL"
        })

    df = pd.DataFrame(detailed_results)
    csv_path = "Results/eval_report.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    precision_at_3 = hits / total

    print(f"\nFinal Score: {precision_at_3:.2f}")
    print(f"Reports saved to Results/")

if __name__ == "__main__":
    run_evaluation()