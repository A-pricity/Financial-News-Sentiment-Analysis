"""
Simple inference demo using dictionary-based sentiment analysis.
This demonstrates the inference pipeline when BERT models are not available.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from utils.sentiment_dict import SentimentDictionary
from utils.text_cleaner import TextCleaner
from utils.language_detector import LanguageDetector


def predict_sentiment(
    text: str,
    sent_dict: SentimentDictionary,
    lang_detector: LanguageDetector,
    cleaner: TextCleaner,
) -> dict:
    """Predict sentiment for a single text."""
    cleaned = cleaner.clean_and_truncate(text, 512)
    language = lang_detector.detect(cleaned)

    if language == "unknown":
        return {
            "text": text,
            "label": "unknown",
            "confidence": 0.0,
            "language": "unknown",
        }

    score = sent_dict.get_sentiment_score(cleaned, language)
    threshold = 0.01

    if score > threshold:
        label = "positive"
    elif score < -threshold:
        label = "negative"
    else:
        label = "neutral"

    return {
        "text": text,
        "label": label,
        "confidence": abs(score),
        "language": language,
        "score": score,
    }


def main():
    sent_dict = SentimentDictionary()
    lang_detector = LanguageDetector()
    cleaner = TextCleaner()

    test_texts = [
        "今天股市大涨特朗普及伊朗达成停火协议市场全面反弹",
        "日韩股市大跌特朗普威胁攻击伊朗石油全球市场暴跌",
        "公司发布财报营收稳定增长利润超预期",
        "A股市场震荡投资者保持谨慎观望",
        "特朗普：与伊朗可能很快达成停火协议",
    ]

    print("=" * 60)
    print("Financial Sentiment Analysis - Inference Demo")
    print("=" * 60)

    for text in test_texts:
        result = predict_sentiment(text, sent_dict, lang_detector, cleaner)
        print(f"\nText: {result['text']}")
        print(f"Language: {result['language']}")
        print(f"Sentiment: {result['label'].upper()}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Score: {result['score']:.4f}")

    print("\n" + "=" * 60)
    print("Model Training Required for 92.5% accuracy target")
    print("Need internet access to download BERT models")
    print("=" * 60)


if __name__ == "__main__":
    main()
