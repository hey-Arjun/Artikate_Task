import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class TicketClassifier:
    def __init__(self, model_path="model/ticket_classifier"): # Adjusted path for local run
        # Use a path relative to the script location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_model_path = os.path.join(current_dir, "model/ticket_classifier")
        
        self.tokenizer = AutoTokenizer.from_pretrained(full_model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(full_model_path)
        self.categories = ["billing", "technical_issue", "feature_request", "complaint", "other"]
        self.model.eval()

    def predict(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
        with torch.no_grad():
            outputs = self.model(**inputs)
        pred_idx = torch.argmax(outputs.logits, dim=-1).item()
        return self.categories[pred_idx]