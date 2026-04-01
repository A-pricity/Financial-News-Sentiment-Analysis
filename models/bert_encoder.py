import os
import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional
import logging

logger = logging.getLogger(__name__)


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

        logger.info(f"Loading BERT model: {model_name}...")
        
        # 如果指定了 cache_dir，使用本地缓存；否则尝试从网络下载
        if cache_dir and os.path.exists(cache_dir):
            logger.info(f"Using cache directory: {cache_dir}")
            self.bert = AutoModel.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                local_files_only=True  # 强制使用本地文件
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
