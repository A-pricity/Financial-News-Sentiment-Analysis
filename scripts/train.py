import os
import sys
import logging
import yaml
import argparse
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from models import FusionSentimentModel, BilingualFusionSentimentModel
from training import Trainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length: int = 512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["text"]
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


class BilingualSentimentDataset(Dataset):
    def __init__(self, dataframe, zh_tokenizer, en_tokenizer, max_length: int = 512):
        self.data = dataframe
        self.zh_tokenizer = zh_tokenizer
        self.en_tokenizer = en_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["text"]
        label = row["label"]
        language = row.get("language", "zh")

        if language == "en":
            tokenizer = self.en_tokenizer
        else:
            tokenizer = self.zh_tokenizer

        encoding = tokenizer(
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
            "language": language,
        }


def load_data(data_dir: str, tokenizer):
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    logger.info(
        f"Loaded: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    train_dataset = SentimentDataset(train_df, tokenizer)
    val_dataset = SentimentDataset(val_df, tokenizer)

    return train_dataset, val_dataset, test_df


def load_bilingual_data(
    data_dir: str, zh_tokenizer, en_tokenizer, max_samples: int = None
):
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    if max_samples and max_samples > 0:
        train_df = train_df.head(max_samples)
        val_df = val_df.head(max(max_samples // 10, 50))
        logger.info(f"Using subset: train={len(train_df)}, val={len(val_df)}")

    logger.info(
        f"Loaded: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    logger.info(
        f"Language distribution (train): {train_df['language'].value_counts().to_dict()}"
    )

    train_dataset = BilingualSentimentDataset(train_df, zh_tokenizer, en_tokenizer)
    val_dataset = BilingualSentimentDataset(val_df, zh_tokenizer, en_tokenizer)

    return train_dataset, val_dataset, test_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of epochs"
    )
    args = parser.parse_args()

    config_path = os.path.join(PROJECT_ROOT, "configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.epochs:
        config["training"]["epochs"] = args.epochs

    checkpoint_dir = "checkpoints"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info("Loading Chinese tokenizer...")
    zh_tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["chinese"]["bert_name"]
    )
    logger.info("Loading English tokenizer...")
    en_tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["english"]["bert_name"]
    )
    logger.info("Tokenizers loaded!")

    data_dir = os.path.join(PROJECT_ROOT, "data/processed")
    max_samples = config["training"].get("max_samples")
    train_dataset, val_dataset, test_dataset = load_bilingual_data(
        data_dir, zh_tokenizer, en_tokenizer, max_samples
    )

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    logger.info("Creating model...")
    model = BilingualFusionSentimentModel(
        zh_bert_name=config["model"]["chinese"]["bert_name"],
        en_bert_name=config["model"]["english"]["bert_name"],
        zh_textcnn_filter_sizes=config["model"]["chinese"]["textcnn_filter_sizes"],
        en_textcnn_filter_sizes=config["model"]["english"]["textcnn_filter_sizes"],
        textcnn_num_filters=config["model"]["chinese"]["textcnn_num_filters"],
        fusion_hidden_dim=config["model"]["fusion"]["hidden_dim"],
        dropout=config["model"]["fusion"]["dropout"],
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info("Using bilingual model (Chinese + English channels)")

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config["training"]["weight_decay"],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config["training"]["learning_rate_bert"],
    )

    total_steps = (
        len(train_loader)
        * config["training"]["epochs"]
        // config["training"]["gradient_accumulation_steps"]
    )
    warmup_steps = config["training"]["warmup_steps"]

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        mixed_precision=config["training"]["mixed_precision"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
        checkpoint_dir=checkpoint_dir,
    )

    if args.resume:
        checkpoint_path = args.resume
    else:
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)
        logger.info(f"Resumed from epoch with best F1: {trainer.best_val_f1:.4f}")

    history = trainer.train(num_epochs=config["training"]["epochs"])

    logger.info("Training completed!")
    logger.info(f"Best validation F1: {trainer.best_val_f1:.4f}")


if __name__ == "__main__":
    main()
