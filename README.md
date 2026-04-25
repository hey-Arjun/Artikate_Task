# Artikate_Task

## Section 01: Q&A & Logic

Focuses on the application of Large Language Models (LLMs) to solve complex reasoning tasks and direct query handling.

Key Deliverable: section_01_Q&A/ANSWERS.md

## Section 02: RAG Implementation

Architecting a Retrieval-Augmented Generation pipeline. This section explores vector embeddings, document chunking strategies, and retrieval precision.

Key Deliverable: section_02_rag/DESIGN.md

## Section 03: Support Ticket Classifier (SLA Focused)

Development of a localized support ticket classification system using a fine-tuned DistilBERT model.

Performance Highlights: 86% Evaluation Accuracy and ~52ms Average Latency.

Local Inference: Engineered for CPU-only environments to meet strict <500ms SLA requirements.

Artifacts: See section_03_classifier/Results_classifier/ for visual performance metrics.

Model Weights Note: Due to repository health and size constraints (247MB), the model.safetensors file is excluded from version control.

Accessing Model Weights: The weights are hosted externally on Google Drive. Please reach out for the access link to run the classifier locally.

# 🛠 Setup & Installation

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Section_02 RAG Working

1. First you need to upload docs file to  **section_02/rag/Data/contracts/**
2. make .env file where u need to store your OPEN_API_KEY
3. you need to write query in pipeline.py inside "__main__" function 
4. Run the command
```
python pipeline.py
```
5. If you want to check the harness, you need to make a test_set a set of queries and place it in "eval_harness.py"
6. Run the command
```
python eval_harness.py
```
