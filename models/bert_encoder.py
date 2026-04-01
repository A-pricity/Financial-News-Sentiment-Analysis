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
    ):
        super().__init__()

        self.model_name = model_name
        self.hidden_size = hidden_size

        logger.info(f"Loading BERT model: {model_name}...")
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
