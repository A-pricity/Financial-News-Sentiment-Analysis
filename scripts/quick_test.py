#!/usr/bin/env python3
"""
快速测试脚本 - 验证模型基本功能和性能
"""

import os
import sys
import logging
import yaml
import torch
import time
import argparse
from transformers import AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from models.fusion_model import BilingualFusionSentimentModel

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True
)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(checkpoint_path: str, config: dict, device: str = "cuda"):
    """加载模型和 tokenizer"""
    
    logger.info("="*60)
    logger.info("Loading Model and Tokenizers")
    logger.info("="*60)
    
    model_cache_dir = os.path.join(PROJECT_ROOT, "models/pretrained")
    
    # 加载中文 tokenizer
    logger.info("Loading Chinese tokenizer...")
    zh_tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["chinese"]["bert_name"],
        cache_dir=model_cache_dir,
        local_files_only=True
    )
    logger.info("✓ Chinese tokenizer loaded")
    
    # 加载英文 tokenizer
    logger.info("Loading English tokenizer...")
    en_tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["english"]["bert_name"],
        cache_dir=model_cache_dir,
        local_files_only=True
    )
    logger.info("✓ English tokenizer loaded")
    
    # 加载模型
    logger.info(f"Loading model from {checkpoint_path}...")
    model = BilingualFusionSentimentModel(
        zh_bert_name=config["model"]["chinese"]["bert_name"],
        en_bert_name=config["model"]["english"]["bert_name"],
        zh_textcnn_filter_sizes=config["model"]["chinese"]["textcnn_filter_sizes"],
        en_textcnn_filter_sizes=config["model"]["english"]["textcnn_filter_sizes"],
        textcnn_num_filters=config["model"]["chinese"]["textcnn_num_filters"],
        fusion_hidden_dim=config["model"]["fusion"]["hidden_dim"],
        dropout=config["model"]["fusion"]["dropout"],
        cache_dir=model_cache_dir,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    best_f1 = checkpoint.get("best_val_f1", 0.0)
    epoch = checkpoint.get("epoch", 0)
    
    logger.info("✓ Model loaded successfully")
    logger.info(f"  Checkpoint epoch: {epoch}")
    logger.info(f"  Best validation F1: {best_f1:.4f}")
    logger.info("")
    
    return model, zh_tokenizer, en_tokenizer


def predict_sentiment(model, tokenizer, text: str, device: str) -> dict:
    """预测单个文本的情感"""
    
    # 检测语言
    language = "zh" if any("\u4e00" <= c <= "\u9fff" for c in text) else "en"
    
    # 编码
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    # 预测
    with torch.no_grad():
        logits, _, _ = model(input_ids, attention_mask, language=language)
        probs = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(logits, dim=1).item()
    
    label_names = ["negative", "neutral", "positive"]
    
    return {
        "text": text,
        "language": language,
        "prediction": label_names[pred],
        "confidence": probs[pred].item(),
        "probabilities": {
            label_names[i]: probs[i].item() for i in range(3)
        }
    }


def test_chinese_samples(model, zh_tokenizer, device: str):
    """测试中文样本"""
    
    logger.info("="*60)
    logger.info("Chinese Sentiment Analysis Tests")
    logger.info("="*60)
    
    test_samples = [
        ("这家公司业绩持续增长，前景看好", "positive"),
        ("股价下跌，投资者担忧", "negative"),
        ("市场保持稳定，无明显波动", "neutral"),
    ]
    
    for text, expected in test_samples:
        result = predict_sentiment(model, zh_tokenizer, text, device)
        
        status = "✓" if result["prediction"] == expected else "✗"
        logger.info(f"\n{status} Text: {text}")
        logger.info(f"  Language: {result['language']}")
        logger.info(f"  Prediction: {result['prediction']} ({result['confidence']:.4f})")
        logger.info(f"  Expected: {expected}")
        logger.info(f"  Probabilities:")
        for label, prob in result['probabilities'].items():
            logger.info(f"    {label}: {prob:.4f}")
    
    logger.info("")


def test_english_samples(model, en_tokenizer, device: str):
    """测试英文样本"""
    
    logger.info("="*60)
    logger.info("English Sentiment Analysis Tests")
    logger.info("="*60)
    
    test_samples = [
        ("The company's revenue increased significantly", "positive"),
        ("Stock prices fell sharply amid concerns", "negative"),
        ("Market remained stable with no major changes", "neutral"),
    ]
    
    for text, expected in test_samples:
        result = predict_sentiment(model, en_tokenizer, text, device)
        
        status = "✓" if result["prediction"] == expected else "✗"
        logger.info(f"\n{status} Text: {text}")
        logger.info(f"  Language: {result['language']}")
        logger.info(f"  Prediction: {result['prediction']} ({result['confidence']:.4f})")
        logger.info(f"  Expected: {expected}")
        logger.info(f"  Probabilities:")
        for label, prob in result['probabilities'].items():
            logger.info(f"    {label}: {prob:.4f}")
    
    logger.info("")


def test_inference_speed(model, zh_tokenizer, en_tokenizer, device: str, num_samples: int = 10):
    """测试推理速度"""
    
    logger.info("="*60)
    logger.info("Inference Speed Test")
    logger.info("="*60)
    
    test_texts = [
        "This is a test sentence for speed benchmark.",
        "这是一个用于速度基准测试的句子。",
    ] * (num_samples // 2 + 1)
    test_texts = test_texts[:num_samples]
    
    # Warm up
    logger.info("Warming up...")
    for text in test_texts[:2]:
        language = "zh" if any("\u4e00" <= c <= "\u9fff" for c in text) else "en"
        tok = zh_tokenizer if language == "zh" else en_tokenizer
        predict_sentiment(model, tok, text, device)
    
    # Benchmark
    logger.info(f"Running inference on {num_samples} samples...")
    start_time = time.time()
    
    for text in test_texts:
        language = "zh" if any("\u4e00" <= c <= "\u9fff" for c in text) else "en"
        tok = zh_tokenizer if language == "zh" else en_tokenizer
        predict_sentiment(model, tok, text, device)
    
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / num_samples * 1000  # ms
    
    logger.info(f"✓ Completed {num_samples} inferences")
    logger.info(f"  Total time: {elapsed_time:.2f}s")
    logger.info(f"  Average time: {avg_time:.2f}ms per sample")
    logger.info(f"  Throughput: {num_samples / elapsed_time:.2f} samples/sec")
    logger.info("")


def test_batch_processing(model, tokenizer, device: str, batch_size: int = 4):
    """测试批量处理"""
    
    logger.info("="*60)
    logger.info("Batch Processing Test")
    logger.info("="*60)
    
    texts = [
        "This is a positive statement about the market.",
        "Market outlook remains uncertain.",
        "Concerns about economic slowdown growing.",
        "Company reports strong quarterly earnings.",
    ] * batch_size
    
    logger.info(f"Processing {len(texts)} samples in batches of {batch_size}...")
    
    model.eval()
    all_results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        
        with torch.no_grad():
            logits, _, _ = model(input_ids, attention_mask, language="en")
            preds = torch.argmax(logits, dim=1).cpu().tolist()
        
        label_names = ["negative", "neutral", "positive"]
        for j, pred in enumerate(preds):
            all_results.append({
                "text": batch_texts[j][:50] + "...",
                "prediction": label_names[pred]
            })
    
    for result in all_results:
        logger.info(f"  {result['text']:<55} → {result['prediction']}")
    
    logger.info(f"✓ Batch processing completed")
    logger.info("")


def print_summary():
    """打印测试总结"""
    
    logger.info("="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    logger.info("✓ Model loading: PASSED")
    logger.info("✓ Tokenizer loading: PASSED")
    logger.info("✓ Chinese sentiment analysis: TESTED")
    logger.info("✓ English sentiment analysis: TESTED")
    logger.info("✓ Inference speed: BENCHMARKED")
    logger.info("✓ Batch processing: TESTED")
    logger.info("")
    logger.info("🎉 All tests completed successfully!")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Run full evaluation: uv run python scripts/evaluate.py")
    logger.info("  2. Try interactive inference: uv run python scripts/inference.py --text 'your text'")
    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description="Quick Model Testing")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Using device: {device}")
    logger.info("")
    
    # 检查 checkpoint 是否存在
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        logger.info("Please train the model first or specify a valid checkpoint path.")
        sys.exit(1)
    
    # 加载配置
    config_path = os.path.join(PROJECT_ROOT, "configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    try:
        # 加载模型
        model, zh_tokenizer, en_tokenizer = load_model_and_tokenizer(
            args.checkpoint, config, device
        )
        
        # 运行测试
        test_chinese_samples(model, zh_tokenizer, device)
        test_english_samples(model, en_tokenizer, device)
        test_inference_speed(model, zh_tokenizer, en_tokenizer, device)
        test_batch_processing(model, en_tokenizer, device)
        
        # 打印总结
        print_summary()
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
