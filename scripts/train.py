import os
import sys
import logging
import logging.handlers
import yaml
import argparse
import torch
import pandas as pd

# Enable better CUDA error debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cudnn.benchmark = False
from datetime import datetime
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from models import FusionSentimentModel, BilingualFusionSentimentModel
from training import Trainer

logger = logging.getLogger(__name__)


def setup_logging(
    log_level_console=logging.INFO,
    log_level_file=logging.DEBUG,
    logs_dir="logs",
    max_logs=5,
):
    os.makedirs(logs_dir, exist_ok=True)

    existing_logs = sorted(Path(logs_dir).glob("train_*.log"))
    if len(existing_logs) >= max_logs:
        for old_log in existing_logs[: len(existing_logs) - max_logs + 1]:
            old_log.unlink()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"train_{timestamp}.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level_file)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level_console)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    return log_file


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
    data_dir: str,
    zh_tokenizer,
    en_tokenizer,
    max_samples: int = None,
    max_test_samples: int = None,
):
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    if max_samples and max_samples > 0:
        train_df = train_df.head(max_samples)
        val_df = val_df.head(max(max_samples // 10, 50))
        logger.info(f"Using subset: train={len(train_df)}, val={len(val_df)}")

    # 测试集大小控制
    if max_test_samples and max_test_samples > 0:
        test_df = test_df.head(max_test_samples)
    elif max_samples and max_samples > 0:
        # 默认使用与训练集相同的样本数
        test_df = test_df.head(max_samples)

    logger.info(
        f"Loaded: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    logger.info(
        f"Language distribution (train): {train_df['language'].value_counts().to_dict()}"
    )

    train_dataset = BilingualSentimentDataset(train_df, zh_tokenizer, en_tokenizer)
    val_dataset = BilingualSentimentDataset(val_df, zh_tokenizer, en_tokenizer)
    test_dataset = BilingualSentimentDataset(test_df, zh_tokenizer, en_tokenizer)

    return train_dataset, val_dataset, test_dataset


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
            raise RuntimeError(
                "模型下载失败，请手动运行：uv run python scripts/download_models.py"
            )
        logger.info("模型下载完成！")
    else:
        logger.info("所有预训练模型已存在，无需下载")


def check_and_download_dataset():
    """检查并下载数据集"""
    # 优先检查 datasets 目录
    import yaml

    config_path = os.path.join(PROJECT_ROOT, "configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config.get("data", {}).get("dataset_name", "financial_news")
    dataset_dir = config.get("data", {}).get("dataset_dir", "datasets")
    datasets_path = os.path.join(PROJECT_ROOT, dataset_dir, dataset_name)

    # 检查 datasets 目录
    data_file = os.path.join(datasets_path, "data.csv")
    train_file = os.path.join(datasets_path, "train.csv")
    val_file = os.path.join(datasets_path, "val.csv")
    test_file = os.path.join(datasets_path, "test.csv")

    has_split_files = (
        os.path.exists(train_file)
        and os.path.exists(val_file)
        and os.path.exists(test_file)
    )
    has_data_file = os.path.exists(data_file)

    if has_split_files or has_data_file:
        logger.info(f"数据集 '{dataset_name}' 已在 datasets/ 目录中准备就绪")
        return

    # 如果 datasets 中没有，检查 data/processed
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
            raise RuntimeError(
                "数据集下载失败，请手动运行：uv run python scripts/download_dataset.py"
            )
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
        "--save-every",
        type=int,
        default=None,
        help="Save checkpoint every N epochs (optional)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=True,
        help="训练完成后自动评估 (默认开启)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="禁用训练后自动评估",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=0,
        help="K折交叉验证 (例如: --cv 10)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="数据集名称 (datasets/ 下的子目录名)",
    )
    args = parser.parse_args()

    if args.no_eval:
        args.eval = False

    config_path = os.path.join(PROJECT_ROOT, "configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.epochs:
        config["training"]["epochs"] = args.epochs

    log_file = setup_logging()
    logger = logging.getLogger(__name__)

    logger.info(f"日志文件: {log_file}")
    logger.info("=" * 60)
    logger.info("训练开始")
    logger.info("=" * 60)

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
        default_cache = os.path.join(
            model_cache_dir, f"models--{model_name.replace('/', '--')}"
        )
        if os.path.exists(default_cache):
            snapshot_base = os.path.join(default_cache, "snapshots")
            if os.path.exists(snapshot_base):
                for snap in os.listdir(snapshot_base):
                    full_snap_path = os.path.join(snapshot_base, snap)
                    if os.path.isdir(full_snap_path):
                        if os.path.exists(
                            os.path.join(full_snap_path, "model.safetensors")
                        ) or os.path.exists(os.path.join(full_snap_path, "model.bin")):
                            return full_snap_path
        return default_cache

    zh_model_path = get_model_path(config["model"]["chinese"]["bert_name"])
    en_model_path = get_model_path(config["model"]["english"]["bert_name"])

    logger.info(f"Loading Chinese tokenizer from: {zh_model_path}...")
    zh_tokenizer = AutoTokenizer.from_pretrained(
        zh_model_path, cache_dir=model_cache_dir, local_files_only=False
    )
    logger.info(f"Loading English tokenizer from: {en_model_path}...")
    en_tokenizer = AutoTokenizer.from_pretrained(
        en_model_path, cache_dir=model_cache_dir, local_files_only=False
    )
    logger.info("Tokenizers loaded!")

    # 数据集配置
    dataset_name = (
        args.dataset
        if args.dataset
        else config["data"].get("dataset_name", "financial_news")
    )
    dataset_dir = config["data"].get("dataset_dir", "datasets")

    # 检查是否使用K-fold数据集
    kfold_dataset_name = f"{dataset_name}_kfold"
    kfold_check_path = os.path.join(PROJECT_ROOT, dataset_dir, kfold_dataset_name)

    use_kfold = args.cv > 1 and os.path.exists(kfold_check_path)

    # 加载数据集
    if use_kfold:
        from utils.dataset_loader import load_kfold_dataset

        train_df, val_df, test_df = load_kfold_dataset(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            fold=1,
            k=args.cv,
        )
        logger.info(
            f"Loaded K-fold dataset '{kfold_dataset_name}': train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )
    else:
        from utils.dataset_loader import load_dataset

        train_df, val_df, test_df = load_dataset(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            train_ratio=config["data"].get("train_ratio", 0.8),
            val_ratio=config["data"].get("val_ratio", 0.1),
            test_ratio=config["data"].get("test_ratio", 0.1),
            random_seed=config["data"].get("random_seed", 42),
        )
        logger.info(
            f"Loaded dataset '{dataset_name}': train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    # 限制样本数量
    max_samples = config["training"].get("max_samples")
    if max_samples and max_samples > 0:
        train_df = train_df.head(max_samples)
        val_df = val_df.head(max(max_samples // 10, 50))
        test_df = test_df.head(max_samples)
        logger.info(
            f"Using subset: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    # 创建数据集
    train_dataset = BilingualSentimentDataset(train_df, zh_tokenizer, en_tokenizer)
    val_dataset = BilingualSentimentDataset(val_df, zh_tokenizer, en_tokenizer)
    test_dataset = BilingualSentimentDataset(test_df, zh_tokenizer, en_tokenizer)

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
        verbose=True,
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

            checkpoints = glob.glob(
                os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt")
            )
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

    # 交叉验证模式
    if args.cv > 1:
        logger.info(f"Running {args.cv}-fold cross-validation...")
        import numpy as np

        if use_kfold:
            from utils.dataset_loader import load_kfold_dataset

            logger.info("Using pre-generated K-fold dataset splits from datasets/")
            fold_results = []

            for fold in range(1, args.cv + 1):
                logger.info(f"\n=== Fold {fold}/{args.cv} ===")

                train_fold_df, val_fold_df, test_df = load_kfold_dataset(
                    dataset_name=dataset_name,
                    dataset_dir=dataset_dir,
                    fold=fold,
                    k=args.cv,
                )
                logger.info(
                    f"Fold {fold}: train={len(train_fold_df)}, val={len(val_fold_df)}, test={len(test_df)}"
                )

                train_dataset = BilingualSentimentDataset(
                    train_fold_df, zh_tokenizer, en_tokenizer
                )
                val_dataset = BilingualSentimentDataset(
                    val_fold_df, zh_tokenizer, en_tokenizer
                )

                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )
                val_loader = DataLoader(val_dataset, batch_size=batch_size)

                # 创建新模型
                model = BilingualFusionSentimentModel(
                    zh_bert_name=zh_model_path,
                    en_bert_name=en_model_path,
                    zh_textcnn_filter_sizes=config["model"]["chinese"][
                        "textcnn_filter_sizes"
                    ],
                    en_textcnn_filter_sizes=config["model"]["english"][
                        "textcnn_filter_sizes"
                    ],
                    textcnn_num_filters=config["model"]["chinese"][
                        "textcnn_num_filters"
                    ],
                    fusion_hidden_dim=config["model"]["fusion"]["hidden_dim"],
                    dropout=config["model"]["fusion"]["dropout"],
                    cache_dir=model_cache_dir,
                )
                model.to(device)

                optimizer = AdamW(
                    model.parameters(), lr=config["training"]["learning_rate_bert"]
                )
                scheduler = None
                trainer_fold = Trainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    gradient_accumulation_steps=config["training"][
                        "gradient_accumulation_steps"
                    ],
                    mixed_precision=config["training"]["mixed_precision"],
                    early_stopping_patience=config["training"][
                        "early_stopping_patience"
                    ],
                    checkpoint_dir=os.path.join(checkpoint_dir, f"fold_{fold}"),
                    verbose=False,
                )

                history = trainer_fold.train(num_epochs=1)
                fold_results.append(
                    {
                        "fold": fold,
                        "val_f1": trainer_fold.best_val_f1,
                        "val_accuracy": history["val"][0]["accuracy"],
                    }
                )
                logger.info(f"Fold {fold} - Val F1: {trainer_fold.best_val_f1:.4f}")

        else:
            from sklearn.model_selection import KFold

            full_train_df = pd.concat([train_df, val_df], ignore_index=True)

            if max_samples:
                full_train_df = full_train_df.head(max_samples)

            kf = KFold(n_splits=args.cv, shuffle=True, random_state=42)
            fold_results = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(full_train_df)):
                logger.info(f"\n=== Fold {fold + 1}/{args.cv} ===")

                train_fold_df = full_train_df.iloc[train_idx]
                val_fold_df = full_train_df.iloc[val_idx]
                logger.info(
                    f"Fold {fold + 1}: train={len(train_fold_df)}, val={len(val_fold_df)}"
                )

                train_dataset = BilingualSentimentDataset(
                    train_fold_df, zh_tokenizer, en_tokenizer
                )
                val_dataset = BilingualSentimentDataset(
                    val_fold_df, zh_tokenizer, en_tokenizer
                )

                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )
                val_loader = DataLoader(val_dataset, batch_size=batch_size)

                # 创建新模型
                model = BilingualFusionSentimentModel(
                    zh_bert_name=zh_model_path,
                    en_bert_name=en_model_path,
                    zh_textcnn_filter_sizes=config["model"]["chinese"][
                        "textcnn_filter_sizes"
                    ],
                    en_textcnn_filter_sizes=config["model"]["english"][
                        "textcnn_filter_sizes"
                    ],
                    textcnn_num_filters=config["model"]["chinese"][
                        "textcnn_num_filters"
                    ],
                    fusion_hidden_dim=config["model"]["fusion"]["hidden_dim"],
                    dropout=config["model"]["fusion"]["dropout"],
                    cache_dir=model_cache_dir,
                )
                model.to(device)

                optimizer = AdamW(
                    model.parameters(), lr=config["training"]["learning_rate_bert"]
                )
                scheduler = None
                trainer_fold = Trainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    gradient_accumulation_steps=config["training"][
                        "gradient_accumulation_steps"
                    ],
                    mixed_precision=config["training"]["mixed_precision"],
                    early_stopping_patience=config["training"][
                        "early_stopping_patience"
                    ],
                    checkpoint_dir=os.path.join(checkpoint_dir, f"fold_{fold + 1}"),
                    verbose=False,
                )

                history = trainer_fold.train(num_epochs=1)
                fold_results.append(
                    {
                        "fold": fold + 1,
                        "val_f1": trainer_fold.best_val_f1,
                        "val_accuracy": history["val"][0]["accuracy"],
                    }
                )
                logger.info(f"Fold {fold + 1} - Val F1: {trainer_fold.best_val_f1:.4f}")

        # 输出交叉验证结果
        avg_f1 = np.mean([r["val_f1"] for r in fold_results])
        avg_acc = np.mean([r["val_accuracy"] for r in fold_results])
        logger.info(f"\n=== Cross-Validation Results ===")
        logger.info(f"Average Val F1: {avg_f1:.4f}")
        logger.info(f"Average Val Accuracy: {avg_acc:.4f}")
        for r in fold_results:
            logger.info(
                f"  Fold {r['fold']}: F1={r['val_f1']:.4f}, Acc={r['val_accuracy']:.4f}"
            )
    else:
        history = trainer.train(
            num_epochs=num_epochs,
            start_epoch=start_epoch,
            save_every_n_epochs=args.save_every,
        )

    logger.info("Training completed!")
    logger.info(f"Best validation F1: {trainer.best_val_f1:.4f}")

    if args.eval:
        import subprocess

        logger.info("Running evaluation on test set...")
        eval_cmd = [
            "python",
            os.path.join(SCRIPT_DIR, "evaluate.py"),
            "--checkpoint",
            os.path.join(checkpoint_dir, "best_model.pt"),
        ]
        subprocess.run(eval_cmd, check=False)

        results_dir = trainer.get_results_dir()
        if results_dir:
            try:
                plot_output = os.path.join(results_dir, "training_history.png")
                plot_cmd = [
                    "python",
                    os.path.join(SCRIPT_DIR, "plot_training.py"),
                    "--csv",
                    os.path.join(results_dir, "training_history.csv"),
                    "--output",
                    plot_output,
                ]
                subprocess.run(plot_cmd, check=False)
                logger.info(f"Generated training visualization: {plot_output}")
            except Exception as e:
                logger.warning(f"Could not generate plot: {e}")


if __name__ == "__main__":
    main()
