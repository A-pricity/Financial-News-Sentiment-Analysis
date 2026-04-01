import os
import json
import logging
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split

from utils import LanguageDetector, TextCleaner, SentimentDictionary

logger = logging.getLogger(__name__)


class FinancialSentimentDataset:
    def __init__(
        self,
        config: dict,
        raw_data_file: str = None,
    ):
        self.config = config
        self.max_length = config.get("max_length", 512)

        self.lang_detector = LanguageDetector()
        self.text_cleaner = TextCleaner()
        self.sentiment_dict = SentimentDictionary()

        self.zh_threshold = 0.01
        self.en_threshold = 0.01
        self.high_conf_threshold = 0.01

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(project_root, "data/processed")
        os.makedirs(self.output_dir, exist_ok=True)

        self.raw_data = []
        if raw_data_file and os.path.exists(raw_data_file):
            self._load_raw_data(raw_data_file)

    def _load_raw_data(self, filepath: str):
        logger.info(f"Loading raw data from {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        logger.info(f"Loaded {len(self.raw_data)} articles")

    def process(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info("Processing data...")

        processed_articles = []

        for article in self.raw_data:
            text = f"{article.get('title', '')} {article.get('content', '')}"

            text = self.text_cleaner.clean_and_truncate(text, self.max_length)

            if not text or len(text) < 10:
                continue

            language = self.lang_detector.detect(text)

            if language == "unknown":
                continue

            threshold = self.zh_threshold if language == "zh" else self.en_threshold
            label, confidence = self.sentiment_dict.annotate(text, language, threshold)

            processed_articles.append(
                {
                    "text": text,
                    "language": language,
                    "label": label,
                    "confidence": confidence,
                    "source": article.get("source", "unknown"),
                    "url": article.get("url", ""),
                }
            )

        logger.info(f"Processed {len(processed_articles)} articles")

        df = pd.DataFrame(processed_articles)

        if len(df) == 0:
            logger.error("No articles to process!")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Label distribution:\n{df['label'].value_counts()}")

        high_conf_df = df

        train_df, temp_df = train_test_split(
            high_conf_df,
            test_size=self.config.get("test_ratio", 0.1)
            + self.config.get("val_ratio", 0.1),
            random_state=self.config.get("random_seed", 42),
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=self.config.get("test_ratio", 0.1)
            / (self.config.get("test_ratio", 0.1) + self.config.get("val_ratio", 0.1)),
            random_state=self.config.get("random_seed", 42),
        )

        train_df.to_csv(os.path.join(self.output_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(self.output_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(self.output_dir, "test.csv"), index=False)

        logger.info(
            f"Saved splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

        return train_df, val_df, test_df

    def load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_df = pd.read_csv(os.path.join(self.output_dir, "train.csv"))
        val_df = pd.read_csv(os.path.join(self.output_dir, "val.csv"))
        test_df = pd.read_csv(os.path.join(self.output_dir, "test.csv"))

        return train_df, val_df, test_df


def main():
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_dir = os.path.join(project_root, "data/processed")
    raw_files = [
        f
        for f in os.listdir(data_dir)
        if f.startswith("raw_data_") and f.endswith(".json")
    ]

    if not raw_files:
        logger.error("No raw data file found. Run crawl_data.py first.")
        return

    latest_file = sorted(raw_files)[-1]
    raw_data_file = os.path.join(data_dir, latest_file)

    dataset = FinancialSentimentDataset(config["data"], raw_data_file)
    dataset.process()


if __name__ == "__main__":
    main()
