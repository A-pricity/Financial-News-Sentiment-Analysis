import os
import sys
import logging
import yaml
import torch
import argparse
from transformers import AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from models import FusionSentimentModel
from models.sentiment_classifier import SentimentClassifier

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, config: dict, device: str = "cuda"):

    model = FusionSentimentModel(
        bert_name=config["model"]["chinese"]["bert_name"],
        textcnn_filter_sizes=config["model"]["chinese"]["textcnn_filter_sizes"],
        textcnn_num_filters=config["model"]["chinese"]["textcnn_num_filters"],
        fusion_hidden_dim=config["model"]["fusion"]["hidden_dim"],
        dropout=config["model"]["fusion"]["dropout"],
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info(f"Loaded model from {checkpoint_path}")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Financial Sentiment Analysis Inference"
    )
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--file", type=str, help="File containing texts to analyze")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length"
    )

    args = parser.parse_args()

    config_path = os.path.join(PROJECT_ROOT, "configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.checkpoint is None:
        args.checkpoint = os.path.join(PROJECT_ROOT, "checkpoints/best_model.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["chinese"]["bert_name"])

    model = load_model(args.checkpoint, config, device)

    classifier = SentimentClassifier(model, tokenizer, device)

    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        logger.error("Please provide either --text or --file argument")
        return

    results = classifier.predict(texts, max_length=args.max_length)

    for result in results:
        print(f"\nText: {result['text']}")
        print(f"Sentiment: {result['label']} (confidence: {result['confidence']:.4f})")
        print(f"Probabilities: {result['probabilities']}")


if __name__ == "__main__":
    main()
