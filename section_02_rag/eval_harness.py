import os
from pipeline import RAGPipeline

def run_evaluation():
    pipeline = RAGPipeline()

    # 1. Define the Ground Truth Test Set
    # We define the question and the 'Expected' document that contains the answer.
    test_set = [
        {
            "question": "What is the difference between an agent and a workflow?", 
            "expected_doc": "Agentic Orchestration.pdf"
        },
        {
            "question": "What are common reasons for RAG failure?", 
            "expected_doc": "Langchian , RAG INTVW QUESTIONS.pdf"
        },
        {
            "question": "How does memory work in multi-agent designs?", 
            "expected_doc": "Agentic Orchestration.pdf"
        },
        {
            "question": "Why should tools be registered dynamically?", 
            "expected_doc": "Langchian , RAG INTVW QUESTIONS.pdf"
        },
        {
            "question": "What is the role of a router chain?", 
            "expected_doc": "Agentic Orchestration.pdf"
        }
    ]

    hits = 0
    total = len(test_set)

    print(f"Start Section 2, Evaluation (precision@3)")
    print(f"Total Test Case: {total}\n")

    for i, item in enumerate(test_set):
        print(f"Test {i+1}: {item['question']}")
        result = pipeline.query(item['question'])

        # Extract the names of the top 3 documents retrieved
        retrieved_docs = [s["document"] for s in result["sources"]]

        # Check if the expected document is in the retrieved list (Precision@k logic)
        if item["expected_doc"] in retrieved_docs:
            hits += 1
            print(f"PASS: Found in {retrieved_docs}")
        else:
            print(f"FAIL: Expected {item['expected_doc']} but got {retrieved_docs}")
        print("-" * 30)

    # calculate final metric
    precision_at_3 = hits / total

    print(f"\nFINAL EVALUATION REPORT")
    print(f"Metric: Precision@3")
    print(f"Score: {precision_at_3:.2f}")
    print(f"Status: {'Acceptable' if precision_at_3 >= 0.8 else 'Requires Tuning'}")

if __name__ == "__main__":
    run_evaluation()