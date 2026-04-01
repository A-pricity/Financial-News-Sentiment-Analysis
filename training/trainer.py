import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer,
        scheduler,
        device: str = "cuda",
        gradient_accumulation_steps: int = 4,
        mixed_precision: bool = True,
        early_stopping_patience: int = 3,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.early_stopping_patience = early_stopping_patience

        self.model.to(device)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.scaler = GradScaler() if mixed_precision else None

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.best_val_f1 = 0.0
        self.patience_counter = 0

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        self.optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            language = batch.get("language", ["en"] * len(input_ids))[0]

            if self.mixed_precision:
                with autocast():
                    logits, _, _ = self.model(
                        input_ids, attention_mask, language=language
                    )
                loss = self.criterion(logits, labels)
                loss = loss / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                logits, _, _ = self.model(input_ids, attention_mask, language=language)
                loss = self.criterion(logits, labels)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item() * self.gradient_accumulation_steps

            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

            progress_bar.set_postfix(
                {"loss": loss.item() * self.gradient_accumulation_steps}
            )

        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return {"loss": avg_loss, "accuracy": accuracy}

    def validate(self) -> dict:
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                language = batch.get("language", ["en"] * len(input_ids))[0]

                logits, _, _ = self.model(input_ids, attention_mask, language=language)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        report = classification_report(
            all_labels,
            all_preds,
            target_names=["negative", "neutral", "positive"],
            output_dict=True,
        )

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1_macro": report["macro avg"]["f1-score"],
            "report": report,
        }

    def train(
        self,
        num_epochs: int,
        save_best: bool = True,
        start_epoch: int = 1,
        save_every_n_epochs: int = None,
    ) -> dict:
        history = {"train": [], "val": []}

        for epoch in range(start_epoch, num_epochs + 1):
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Epoch {epoch}/{num_epochs}")
            logger.info(f"{'=' * 50}")

            train_metrics = self.train_epoch(epoch)
            history["train"].append(train_metrics)
            logger.info(
                f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}"
            )

            val_metrics = self.validate()
            history["val"].append(val_metrics)
            logger.info(
                f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_macro']:.4f}"
            )

            logger.info("\nClassification Report:")
            for label in ["negative", "neutral", "positive"]:
                logger.info(
                    f" {label}: P={val_metrics['report'][label]['precision']:.4f}, "
                    f"R={val_metrics['report'][label]['recall']:.4f}, "
                    f"F1={val_metrics['report'][label]['f1-score']:.4f}"
                )

            if save_best and val_metrics["f1_macro"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1_macro"]
                self.patience_counter = 0
                # 保存最佳模型时包含当前 epoch 信息
                self.save_checkpoint(f"best_model.pt", epoch=epoch)
                logger.info(f"Saved best model with F1: {self.best_val_f1:.4f}")
            else:
                self.patience_counter += 1
            
            # 定期保存 checkpoint
            if save_every_n_epochs is not None and epoch % save_every_n_epochs == 0:
                checkpoint_filename = f"checkpoint_epoch_{epoch}.pt"
                self.save_checkpoint(checkpoint_filename, epoch=epoch)
                logger.info(f"Saved periodic checkpoint: {checkpoint_filename}")

            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

        return history

    def save_checkpoint(self, filename: str, epoch: int = None):
        """保存检查点
        
        Args:
            filename: 文件名
            epoch: 当前 epoch（可选）
        """
        filepath = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_f1": self.best_val_f1,
            "epoch": epoch if epoch is not None else 0,
        }
        
        # 保存 scheduler 状态（如果存在）
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # 保存随机种子状态
        checkpoint["rng_state"] = torch.get_rng_state()
        if torch.cuda.is_available():
            checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state()

        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")

    def load_checkpoint(self, filepath: str) -> dict:
        """加载检查点并返回额外信息
        
        Returns:
            包含额外信息的字典（如 epoch）
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_f1 = checkpoint.get("best_val_f1", 0.0)
        
        extra_info = {}
        
        # 加载 epoch
        if "epoch" in checkpoint:
            extra_info["epoch"] = checkpoint["epoch"]
            logger.info(f"Checkpoint from epoch {checkpoint['epoch']}")
        
        # 加载 scheduler 状态
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            if "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logger.info("Loaded scheduler state")
        
        # 加载随机种子状态
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"])
            if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
                torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
            logger.info("Loaded RNG state")

        logger.info(f"Loaded checkpoint from {filepath}")
        return extra_info
