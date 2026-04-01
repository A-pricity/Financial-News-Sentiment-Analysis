#!/usr/bin/env python3
"""
Quick test to verify model loading and TextCNN fix.
"""

import os
import sys
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.textcnn import TextCNN
from models.fusion_model import BilingualFusionSentimentModel

def test_textcnn_with_bert_embeddings():
    """测试 TextCNN 使用 BERT embeddings 作为输入"""
    logger.info("Testing TextCNN with BERT embeddings...")
    
    # 创建 TextCNN 实例
    textcnn = TextCNN(
        vocab_size=30000,
        embedding_dim=768,
        filter_sizes=[2, 3, 4],
        num_filters=256,
        dropout=0.3,
        use_bert_embeddings=True
    )
    
    # 模拟 BERT 输出 (batch_size=2, seq_len=10, hidden_size=768)
    bert_embeddings = torch.randn(2, 10, 768)
    
    # 前向传播
    output = textcnn(bert_embeddings=bert_embeddings)
    
    logger.info(f"✓ TextCNN input shape: {bert_embeddings.shape}")
    logger.info(f"✓ TextCNN output shape: {output.shape}")
    
    expected_dim = 256 * 3  # num_filters * len(filter_sizes)
    assert output.shape == (2, expected_dim), f"Expected shape (2, {expected_dim}), got {output.shape}"
    
    logger.info("✓ TextCNN test passed!\n")


def test_fusion_model():
    """测试融合模型的前向传播"""
    logger.info("Testing BilingualFusionSentimentModel...")
    
    # 创建模型
    model = BilingualFusionSentimentModel(
        zh_bert_name="bert-base-chinese",
        en_bert_name="bert-base-uncased",
        zh_textcnn_filter_sizes=[2, 3, 4],
        en_textcnn_filter_sizes=[2, 3, 4, 5],
        textcnn_num_filters=256,
        fusion_hidden_dim=768,
        dropout=0.3,
    )
    
    # 模拟输入 (batch_size=2, seq_len=10)
    input_ids = torch.randint(0, 1000, (2, 10))
    attention_mask = torch.ones(2, 10)
    
    # 测试中文
    logger.info("Testing Chinese forward pass...")
    logits_zh, bert_out_zh, textcnn_out_zh = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        language="zh"
    )
    logger.info(f"✓ Chinese output shape: {logits_zh.shape}")
    
    # 测试英文
    logger.info("Testing English forward pass...")
    logits_en, bert_out_en, textcnn_out_en = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        language="en"
    )
    logger.info(f"✓ English output shape: {logits_en.shape}")
    
    # 验证输出维度
    assert logits_zh.shape == (2, 3), f"Expected (2, 3), got {logits_zh.shape}"
    assert logits_en.shape == (2, 3), f"Expected (2, 3), got {logits_en.shape}"
    
    logger.info("✓ Fusion model test passed!\n")


def main():
    logger.info("=" * 50)
    logger.info("Running model verification tests...")
    logger.info("=" * 50 + "\n")
    
    try:
        test_textcnn_with_bert_embeddings()
        test_fusion_model()
        
        logger.info("=" * 50)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("=" * 50)
        logger.info("\n模型修复成功！可以开始训练了。")
        
    except Exception as e:
        logger.error("=" * 50)
        logger.error(f"❌ TEST FAILED: {e}")
        logger.error("=" * 50)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
