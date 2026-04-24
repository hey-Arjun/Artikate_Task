# Design Documentation: Section 03 - Support Ticket Classifier

## 1. Overview
The goal of this section was to build a robust, high-speed machine learning classifier that operates locally on a single CPU without external API dependencies.

## 2. Dataset Generation
- **Methodology:** We generated a synthetic dataset of approximately 700 support tickets.
- **Verification:** Categories were manually verified to ensure high-quality training labels.
- **Evaluation Set:** A 100-row hold-out set was used for the final validation to ensure the model wasn't just "memorizing" data.

## 3. Implementation Logic
### Why DistilBERT?
We chose **DistilBERT** because it is a lightweight version of the BERT model. It is optimized for speed and size, making it the "Goldilocks" choice for local CPU inference where we must return results in under 500ms.

### Training Workflow
1. **Colab T4 GPU:** Due to local hardware limitations and time constraints, we utilized Google Colab's T4 GPU for training.
2. **Model Export:** After training, we bundled the `model.safetensors`, `config.json`, and `tokenizer.json` into a zip file.
3. **Local Inference:** The zip was downloaded and unzipped into `section_03_classifier/model/`. All predictions are made locally using `torch`.

## 4. Performance Metrics
All artifacts can be found in `section_03_classifier/Results/`.

- **Accuracy:** 86% on the final evaluation set.
- **F1-Score:** 0.85 (Macro Average).
- **Latency Result:**
    - **Average:** ~52.5ms per ticket.
    - **Max (Cold Start):** 430.36ms.
    - **SLA Status:** **PASSED** (All tickets under 500ms).

## 5. Justification for Metrics
Despite a time-constrained training window (resulting in a 76% training match), the model demonstrated strong generalization capabilities, reaching 86% accuracy on unseen data. The discrepancy in F1-scores for 'Complaints' versus 'Feature Requests' is documented as a result of semantic overlap, which is a known challenge in NLP.

## 6. Directory Structure
- `section_03_classifier/model/`: Local model weights.
- `section_03_classifier/Results/`: Contains `latency_report.csv` and `confusion_matrix.png`.
- `section_03_classifier/Briefing_section_03`: Contains `Overview in detail for Section_03`.