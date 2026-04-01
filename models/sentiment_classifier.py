import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoTokenizer


class SentimentClassifier:
    def __init__(
        self,
        model,
        tokenizer: AutoTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        label_map: dict = {0: "negative", 1: "neutral", 2: "positive"},
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.label_map = label_map

        self.model.to(device)
        self.model.eval()

    def predict(self, texts: list, max_length: int = 512) -> list:
        results = []

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            logits, _, _ = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

        for i, text in enumerate(texts):
            pred_label = predictions[i].item()
            pred_prob = probs[i].tolist()

            results.append(
                {
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "label": self.label_map[pred_label],
                    "label_id": pred_label,
                    "confidence": pred_prob[pred_label],
                    "probabilities": {
                        self.label_map[j]: pred_prob[j] for j in range(3)
                    },
                }
            )

        return results

    def predict_single(self, text: str, max_length: int = 512) -> dict:
        return self.predict([text], max_length)[0]
