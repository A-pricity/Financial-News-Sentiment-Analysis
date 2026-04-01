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


def check_and_download_models():
    """检查并下载预训练模型"""
    model_cache_dir = os.path.join(PROJECT_ROOT, "models/pretrained")
    os.makedirs(model_cache_dir, exist_ok=True)
    
    # 需要下载的模型列表
    required_models = [
        "bert-base-chinese",
        "bert-base-uncased",
    ]
    
    missing_models = []
    for model_name in required_models:
        model_dir_name = f"models--{model_name.replace('/', '--')}"
        model_path = os.path.join(model_cache_dir, model_dir_name)
        if not os.path.exists(model_path):
            missing_models.append(model_name)
    
    if missing_models:
        logger.info(f"发现缺失的模型：{missing_models}")
        logger.info("正在运行模型下载脚本...")
        import subprocess
        script_path = os.path.join(PROJECT_ROOT, "scripts/download_models.py")
        result = subprocess.run(["uv", "run", "python", script_path], check=False)
        if result.returncode != 0:
            raise RuntimeError("模型下载失败，请手动运行：uv run python scripts/download_models.py")
        logger.info("模型下载完成！")
    else:
        logger.info("所有预训练模型已存在，无需下载")


def check_and_download_dataset():
    """检查并下载数据集"""
    data_dir = os.path.join(PROJECT_ROOT, "data/processed")
    required_files = ["train.csv", "val.csv", "test.csv"]
    
    missing_files = []
    for file_name in required_files:
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_name)
    
    if missing_files:
        logger.info(f"发现缺失的数据文件：{missing_files}")
        logger.info("正在运行数据集下载脚本...")
        import subprocess
        script_path = os.path.join(PROJECT_ROOT, "scripts/download_dataset.py")
        result = subprocess.run(["uv", "run", "python", script_path], check=False)
        if result.returncode != 0:
            raise RuntimeError("数据集下载失败，请手动运行：uv run python scripts/download_dataset.py")
        logger.info("数据集下载完成！")
    else:
        logger.info("所有数据文件已存在，无需下载")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of epochs"
    )
    parser.add_argument(
        "--save-every", type=int, default=None, 
        help="Save checkpoint every N epochs (optional)"
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

    # 1. 先检查并下载数据集
    check_and_download_dataset()
    
    # 2. 检查并下载模型
    check_and_download_models()
    
    # 3. 设置持久化模型缓存目录
    model_cache_dir = os.path.join(PROJECT_ROOT, "models/pretrained")
    os.makedirs(model_cache_dir, exist_ok=True)
    logger.info(f"模型缓存目录：{model_cache_dir}")

    # 4. 获取模型路径（优先使用本地配置）
    local_model_dir = config["model"].get("local_model_dir")

    def get_model_path(model_name: str) -> str:
        if local_model_dir:
            local_path = os.path.join(PROJECT_ROOT, local_model_dir, model_name)
            if os.path.exists(local_path):
                return local_path
        # 回退到默认缓存目录
        default_cache = os.path.join(model_cache_dir, f"models--{model_name.replace('/', '--')}")
        if os.path.exists(default_cache):
            return default_cache
        return model_name

    zh_model_path = get_model_path(config["model"]["chinese"]["bert_name"])
    en_model_path = get_model_path(config["model"]["english"]["bert_name"])

    logger.info(f"Loading Chinese tokenizer from: {zh_model_path}...")
    zh_tokenizer = AutoTokenizer.from_pretrained(
        zh_model_path,
        cache_dir=model_cache_dir,
        local_files_only=False
    )
    logger.info(f"Loading English tokenizer from: {en_model_path}...")
    en_tokenizer = AutoTokenizer.from_pretrained(
        en_model_path,
        cache_dir=model_cache_dir,
        local_files_only=False
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
        zh_bert_name=zh_model_path,
        en_bert_name=en_model_path,
        zh_textcnn_filter_sizes=config["model"]["chinese"]["textcnn_filter_sizes"],
        en_textcnn_filter_sizes=config["model"]["english"]["textcnn_filter_sizes"],
        textcnn_num_filters=config["model"]["chinese"]["textcnn_num_filters"],
        fusion_hidden_dim=config["model"]["fusion"]["hidden_dim"],
        dropout=config["model"]["fusion"]["dropout"],
        cache_dir=model_cache_dir,  # 传递缓存目录
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

    # 添加 scheduler 引用到 trainer
    trainer.scheduler = scheduler

    if args.resume:
        checkpoint_path = args.resume
    else:
        # 自动查找最新的 checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
        
        # 如果没有 best_model.pt，查找其他 checkpoint
        if not os.path.exists(checkpoint_path):
            import glob
            checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt"))
            if checkpoints:
                # 选择最新的 checkpoint
                checkpoint_path = max(checkpoints, key=os.path.getctime)
                logger.info(f"Found checkpoint: {checkpoint_path}")

    # 确定总 epoch 数
    num_epochs = config["training"]["epochs"]
    if args.epochs:
        num_epochs = args.epochs
    
    extra_info = {}
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        extra_info = trainer.load_checkpoint(checkpoint_path)
        logger.info(f"Resumed from epoch with best F1: {trainer.best_val_f1:.4f}")
        
        # 如果指定了 epochs，使用指定的；否则从 checkpoint 中恢复
        if args.epochs:
            num_epochs = args.epochs
        else:
            # 如果没有指定 epochs，继续训练剩余的 epochs
            logger.info(f"Will train for total {num_epochs} epochs")
    else:
        logger.info("No checkpoint found, starting from scratch")
        extra_info = {}

    # 确定起始 epoch
    start_epoch = extra_info.get("epoch", 0) + 1 if extra_info else 1
    logger.info(f"Starting from epoch {start_epoch}")

    history = trainer.train(
        num_epochs=num_epochs, 
        start_epoch=start_epoch,
        save_every_n_epochs=args.save_every
    )

    logger.info("Training completed!")
    logger.info(f"Best validation F1: {trainer.best_val_f1:.4f}")


if __name__ == "__main__":
    main()
