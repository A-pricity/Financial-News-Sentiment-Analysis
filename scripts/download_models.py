#!/usr/bin/env python3
"""
Download and cache pretrained models to persistent storage.
This script downloads BERT models and tokenizers to a persistent directory
to avoid re-downloading after notebook restart.
"""

import os
import logging
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 持久化模型缓存目录
MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT, "models/pretrained")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

def set_hf_mirror():
    """设置 Hugging Face 镜像以加速下载"""
    if os.getenv("HF_ENDPOINT") is None:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        logger.info("使用 Hugging Face 镜像：https://hf-mirror.com")


def model_exists(model_name: str) -> bool:
    """检查模型是否已存在于缓存目录"""
    model_dir_name = f"models--{model_name.replace('/', '--')}"
    model_path = os.path.join(MODEL_CACHE_DIR, model_dir_name)
    
    if not os.path.exists(model_path):
        return False
    
    # 检查是否有实际的模型文件（不仅仅是元数据）
    has_model_file = False
    for root, dirs, files in os.walk(model_path):
        for f in files:
            if f.endswith('.safetensors') or f.endswith('.bin'):
                has_model_file = True
                break
    
    return has_model_file


def download_model(model_name: str, force_download: bool = False):
    """下载并缓存模型到持久化目录
    
    Args:
        model_name: 模型名称
        force_download: 是否强制重新下载
    """
    # 检查模型是否已存在
    if not force_download and model_exists(model_name):
        logger.info(f"✓ Model already cached: {model_name}")
        return True
    
    logger.info(f"Downloading {model_name}...")
    
    try:
        # 下载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=MODEL_CACHE_DIR
        )
        logger.info(f"✓ Tokenizer downloaded: {model_name}")
        
        # 下载模型
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=MODEL_CACHE_DIR
        )
        logger.info(f"✓ Model downloaded: {model_name}")
        
        # 验证文件存在
        model_files = []
        for root, dirs, files in os.walk(MODEL_CACHE_DIR):
            for f in files:
                if model_name.replace("/", "--") in root:
                    model_files.append(os.path.join(root, f))
        
        if model_files:
            total_size = sum(os.path.getsize(f) for f in model_files)
            logger.info(f"✓ Cached {len(model_files)} files ({total_size / 1024 / 1024:.2f} MB)")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {model_name}: {e}")
        return False


def main():
    """下载所有需要的模型"""
    set_hf_mirror()
    
    logger.info(f"模型缓存目录：{MODEL_CACHE_DIR}")
    
    # 需要下载的模型列表
    all_models = [
        "bert-base-chinese",      # 中文 BERT
        "bert-base-uncased",      # 英文 BERT
    ]
    
    # 检查哪些模型需要下载
    models_to_download = [m for m in all_models if not model_exists(m)]
    
    if not models_to_download:
        logger.info("所有模型已存在，无需下载")
        logger.info("=" * 50)
        logger.info(f"模型存储位置：{MODEL_CACHE_DIR}")
        logger.info("=" * 50)
        return
    
    logger.info(f"需要下载的模型：{models_to_download}")
    
    success_count = 0
    for model_name in models_to_download:
        if download_model(model_name):
            success_count += 1
    
    logger.info("=" * 50)
    logger.info(f"下载完成：{success_count}/{len(models_to_download)} 个模型")
    logger.info(f"模型存储位置：{MODEL_CACHE_DIR}")
    logger.info("=" * 50)
    
    if success_count == len(models_to_download):
        logger.info("✓ 所有模型下载成功！")
        logger.info("\n下次运行时，模型将自动从本地缓存加载，无需重新下载。")
    else:
        logger.warning("⚠ 部分模型下载失败，请检查网络连接后重试。")


if __name__ == "__main__":
    main()
