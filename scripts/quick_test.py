#!/usr/bin/env python3
import os
import sys
import logging
import yaml
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from training import Trainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleBERTClassifier(nn.Module):
    def __init__(self, bert_name: str, num_labels: int = 3, dropout: float = 0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits, pooled, pooled


class QuickDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe.head(100)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["text"][:200]
        label = row["label"]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def main():
    config_path = os.path.join(PROJECT_ROOT, "configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info("Loading tokenizer...")
    en_tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["english"]["bert_name"]
    )
    logger.info("Tokenizer loaded!")

    data_dir = os.path.join(PROJECT_ROOT, "data/processed")
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))

    logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    train_dataset = QuickDataset(train_df, en_tokenizer)
    val_dataset = QuickDataset(val_df, en_tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    logger.info("Creating model...")
    model = SimpleBERTClassifier(
        bert_name=config["model"]["english"]["bert_name"],
        num_labels=3,
        dropout=config["model"]["fusion"]["dropout"],
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        gradient_accumulation_steps=1,
        mixed_precision=False,
        early_stopping_patience=10,
    )

    logger.info("Starting quick training test...")
    history = trainer.train(num_epochs=1, save_best=False)

    logger.info("Quick test completed!")
    logger.info(f"Train metrics: {history['train']}")
    logger.info(f"Val metrics: {history['val']}")


if __name__ == "__main__":
    main()
