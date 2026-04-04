import os
import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_model_path(model_name: str, cache_dir: str = None) -> str:
    """解析模型路径，优先使用本地缓存"""
    if cache_dir is None:
        return model_name

    model_cache = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")

    if not os.path.exists(model_cache):
        return model_name

    snapshot_base = os.path.join(model_cache, "snapshots")
    if not os.path.exists(snapshot_base):
        return model_name

    for snap in os.listdir(snapshot_base):
        full_snap_path = os.path.join(snapshot_base, snap)
        if os.path.isdir(full_snap_path):
            if os.path.exists(
                os.path.join(full_snap_path, "model.safetensors")
            ) or os.path.exists(os.path.join(full_snap_path, "model.bin")):
                return full_snap_path

    return model_name


class BERTEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        hidden_size: int = 768,
        freeze_bert: bool = False,
        cache_dir: str = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.hidden_size = hidden_size
        self.cache_dir = cache_dir

        resolved_path = resolve_model_path(model_name, cache_dir)

        logger.info(f"Loading BERT model: {model_name}...")

        # 如果解析到的是本地路径且存在，使用本地加载
        if resolved_path != model_name and os.path.exists(resolved_path):
            logger.info(f"Using cache directory: {cache_dir}")
            self.bert = AutoModel.from_pretrained(resolved_path, local_files_only=True)
        elif cache_dir and os.path.exists(cache_dir):
            logger.info(f"Using cache directory: {cache_dir}")
            self.bert = AutoModel.from_pretrained(
                model_name, cache_dir=cache_dir, local_files_only=True
            )
        else:
            self.bert = AutoModel.from_pretrained(model_name)

        logger.info(f"BERT model loaded: {model_name}")

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        if self.bert.config.hidden_size != hidden_size:
            self.projection = nn.Linear(self.bert.config.hidden_size, hidden_size)
        else:
            self.projection = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        cls_output = outputs.last_hidden_state[:, 0, :]

        if self.projection is not None:
            cls_output = self.projection(cls_output)

        return cls_output

    def get_output_dim(self) -> int:
        return self.hidden_size
