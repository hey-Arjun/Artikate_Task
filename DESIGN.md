# 1. Architectural Overview
The implemented system follows a Modular RAG (Retrieval-Augmented Generation) pattern designed to process dense legal contracts. Unlike generic RAG systems, this architecture prioritizes page-level citation accuracy and strict refusal mechanisms to prevent hallucination in high-stakes legal contexts.

# 2. Technical Trade-offs & Decisions

### A. Ingestion & Page-Aware Parsing

Choice: PyMuPDF (fitz) for extraction.

Reasoning: Legal documents rely heavily on page-specific references (e.g., "See Page 14, Section 2"). Standard PDF loaders often lose page boundaries during extraction. By using fitz, we preserve the page index as metadata in each chunk, ensuring that the final output can cite the exact location of the information.

### B. Chunking Strategy

Choice: Recursive Character Text Splitting (1000 chars, 100 char overlap).

Reasoning: Legal clauses are often lengthy and recursive. A smaller chunk size (e.g., 500) would risk splitting a single indemnity clause into two, losing the semantic "binding" of the paragraph. The 10% overlap acts as a buffer to ensure that context is maintained across split boundaries, which is critical for identifying "Except as otherwise provided" clauses that modify previous text.

### C. Vector Storage & Embedding

Choice: ChromaDB with OpenAI text-embedding-3-small.

Reasoning: Given the "Under 5-minute setup" constraint, ChromaDB provides a lightweight, local-first vector store. I selected text-embedding-3-small for its high dimensionality and lower cost, providing a superior cost-to-performance ratio for semantic search in structured technical English.

# 3. Hallucination Mitigation Strategy
The system implements a Dual-Gate Grounding strategy:

Similarity Thresholding: We utilize similarity_search_with_relevance_scores. Any retrieval with a Euclidean distance score below 0.7 is automatically rejected. This prevents the LLM from "guessing" based on loosely related text.

Strict Instruction Prompting: The prompt uses a "Closed-Domain" instruction, explicitly forbidding the model from using its internal pre-training knowledge. If the answer is not in the context, the model is instructed to output a standardized refusal message.

# 4. Evaluation Methodology
The system is measured using Precision@3.

Why Precision@3? In legal workflows, a lawyer rarely reviews more than 3 snippets to verify an answer. If the correct information is not in the top 3 retrieved results, the system has failed its primary utility.

Results: The current implementation achieved a 1.00 Precision@3 score across the internal test set, validating the current chunking and retrieval parameters.

# 5. Scaling to 50,000 Documents
To transition from the current 3-document prototype to a 50,000-document production system, the following remedies are required:

A. Storage -> Local ChromaDB (It managed VectorDB (eg: pinecone, pilvus) to handle HNSW indexing at scaling without RAM exhaustion)

B. Ingestion -> Sequestion Processing (Asynchronous Task Queue (e.g., Celery/Redis) to parallelize PDF parsing across multiple workers.)

C. Retrieval -> Vector Search Only (Hybrid Search (BM25 + Vector) to capture specific legal IDs/clause numbers that semantic search might miss.)

D. Compute -> Single CPU (Load-Balanced Inference and Request Batching to maintain sub-second response times.)