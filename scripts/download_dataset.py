#!/usr/bin/env python3
"""
Download and preprocess public financial sentiment datasets.
Supports multiple datasets from HuggingFace.
"""

import os
import logging
import json
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_lwrf42_dataset():
    """Load lwrf42/financial-sentiment-dataset (95.2k rows)"""
    logger.info("Loading lwrf42/financial-sentiment-dataset...")
    ds = load_dataset("lwrf42/financial-sentiment-dataset")

    # Combine train and validation
    train_df = ds["train"].to_pandas()
    val_df = ds["validation"].to_pandas()

    df = pd.concat([train_df, val_df], ignore_index=True)
    logger.info(f"Loaded {len(df)} samples")

    return df


def load_sjyuxyz_dataset():
    """Load sjyuxyz/financial-sentiment-analysis"""
    logger.info("Loading sjyuxyz/financial-sentiment-analysis...")
    ds = load_dataset("sjyuxyz/financial-sentiment-analysis")

    df = ds["train"].to_pandas()
    logger.info(f"Loaded {len(df)} samples")

    return df


def map_lwrf42_labels(df):
    """Map lwrf42 dataset labels to our format"""
    # The dataset has 'output' column with sentiment labels
    # Map: positive->1, negative->2, neutral->0
    label_mapping = {"positive": 1, "negative": 2, "neutral": 0}

    processed = []
    for _, row in df.iterrows():
        text = row.get("input", "") or row.get("instruction", "") + " " + row.get(
            "output", ""
        )
        label = label_mapping.get(row.get("output", ""), 0)

        # Detect language (simple heuristic)
        text_sample = text[:100] if len(text) > 100 else text
        has_chinese = any("\u4e00" <= c <= "\u9fff" for c in text_sample)
        language = "zh" if has_chinese else "en"

        processed.append(
            {
                "text": text[:512],  # Truncate to max length
                "language": language,
                "label": label,
                "confidence": 1.0,  # High confidence for labeled data
                "source": "lwrf42",
                "url": "",
            }
        )

    return pd.DataFrame(processed)


def map_sjyuxyz_labels(df):
    """Map sjyuxyz dataset labels to our format"""
    # Check columns
    logger.info(f"Columns: {df.columns.tolist()}")

    processed = []
    for _, row in df.iterrows():
        # The dataset might have different column names
        text = row.get("text", "") or row.get("input", "")
        label = row.get("label", 0) or row.get("sentiment", 0)

        if not text:
            continue

        # Detect language
        text_sample = text[:100] if len(text) > 100 else text
        has_chinese = any("\u4e00" <= c <= "\u9fff" for c in text_sample)
        language = "zh" if has_chinese else "en"

        processed.append(
            {
                "text": text[:512],
                "language": language,
                "label": int(label),
                "confidence": 1.0,
                "source": "sjyuxyz",
                "url": "",
            }
        )

    return pd.DataFrame(processed)


def split_and_save(df, output_dir):
    """Split into train/val/test and save as CSV"""
    logger.info(f"Total samples before split: {len(df)}")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")

    # Split: 80% train, 10% val, 10% test
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
    )

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Saved: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    return train_df, val_df, test_df


def download_and_process():
    """Main function to download and process datasets"""

    # Try to load the larger dataset first
    try:
        df = load_lwrf42_dataset()
        processed_df = map_lwrf42_labels(df)
    except Exception as e:
        logger.warning(f"Failed to load lwrf42 dataset: {e}")
        try:
            df = load_sjyuxyz_dataset()
            processed_df = map_sjyuxyz_labels(df)
        except Exception as e2:
            logger.error(f"Failed to load sjyuxyz dataset: {e2}")
            raise RuntimeError("No datasets could be loaded")

    # Save raw data as JSON for reference
    raw_path = os.path.join(OUTPUT_DIR, "raw_data_public.json")
    processed_df.to_json(raw_path, orient="records", force_ascii=False, indent=2)
    logger.info(f"Saved raw data to {raw_path}")

    # Split and save
    split_and_save(processed_df, OUTPUT_DIR)

    logger.info("Dataset download and preprocessing completed!")

    # Print summary
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Total samples: {len(processed_df)}")
    print(f"Label distribution:")
    for label, count in processed_df["label"].value_counts().items():
        label_name = {0: "neutral", 1: "positive", 2: "negative"}.get(label, "unknown")
        print(f"  {label_name} ({label}): {count}")
    print(f"Language distribution:")
    print(processed_df["language"].value_counts().to_string())
    print("=" * 50)


if __name__ == "__main__":
    download_and_process()
