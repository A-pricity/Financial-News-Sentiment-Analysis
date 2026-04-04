import torch
import torch.nn as nn
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AttentionFusion(nn.Module):
    def __init__(
        self,
        bert_dim: int = 768,
        textcnn_dim: int = 768,
        hidden_dim: int = 768,
    ):
        super().__init__()

        self.bert_dim = bert_dim
        self.textcnn_dim = textcnn_dim
        self.hidden_dim = hidden_dim

        self.bert_projection = nn.Linear(bert_dim, hidden_dim)
        self.textcnn_projection = nn.Linear(textcnn_dim, hidden_dim)

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, bert_output: torch.Tensor, textcnn_output: torch.Tensor
    ) -> torch.Tensor:
        bert_proj = self.bert_projection(bert_output)
        textcnn_proj = self.textcnn_projection(textcnn_output)

        concat = torch.cat([bert_proj, textcnn_proj], dim=1)

        attention_scores = self.attention(concat)
        attention_weights = torch.sigmoid(attention_scores)

        fused = attention_weights * bert_proj + (1 - attention_weights) * textcnn_proj

        output = self.layer_norm(fused)

        return output


class FusionSentimentModel(nn.Module):
    def __init__(
        self,
        bert_name: str = "bert-base-chinese",
        textcnn_filter_sizes: list = [2, 3, 4],
        textcnn_num_filters: int = 256,
        fusion_hidden_dim: int = 768,
        dropout: float = 0.3,
    ):
        super().__init__()

        from .bert_encoder import BERTEncoder
        from .textcnn import TextCNN

        self.bert_encoder = BERTEncoder(
            model_name=bert_name,
            hidden_size=768,
        )

        self.textcnn = TextCNN(
            vocab_size=30000,
            embedding_dim=768,
            filter_sizes=textcnn_filter_sizes,
            num_filters=textcnn_num_filters,
            dropout=dropout,
        )

        self.fusion = AttentionFusion(
            bert_dim=768,
            textcnn_dim=textcnn_num_filters * len(textcnn_filter_sizes),
            hidden_dim=fusion_hidden_dim,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(256, 3),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bert_output = self.bert_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        textcnn_output = self.textcnn(input_ids)

        fused_output = self.fusion(bert_output, textcnn_output)

        logits = self.classifier(fused_output)

        return logits, bert_output, textcnn_output


class BilingualFusionSentimentModel(nn.Module):
    def __init__(
        self,
        zh_bert_name: str = "bert-base-chinese",
        en_bert_name: str = "yiyanghkust/finbert-tone",
        zh_textcnn_filter_sizes: list = [2, 3, 4],
        en_textcnn_filter_sizes: list = [2, 3, 4, 5],
        textcnn_num_filters: int = 256,
        fusion_hidden_dim: int = 768,
        dropout: float = 0.3,
        cache_dir: str = None,
    ):
        super().__init__()

        from .bert_encoder import BERTEncoder
        from .textcnn import TextCNN

        logger.info("Creating Chinese BERT encoder...")
        self.zh_bert = BERTEncoder(
            model_name=zh_bert_name, hidden_size=768, cache_dir=cache_dir
        )
        logger.info("Creating English BERT encoder...")
        self.en_bert = BERTEncoder(
            model_name=en_bert_name, hidden_size=768, cache_dir=cache_dir
        )
        logger.info("Creating TextCNN modules...")

        self.zh_textcnn = TextCNN(
            vocab_size=30000,
            embedding_dim=768,
            filter_sizes=zh_textcnn_filter_sizes,
            num_filters=textcnn_num_filters,
            dropout=dropout,
        )

        self.en_textcnn = TextCNN(
            vocab_size=30000,
            embedding_dim=768,
            filter_sizes=en_textcnn_filter_sizes,
            num_filters=textcnn_num_filters,
            dropout=dropout,
        )

        self.zh_fusion = AttentionFusion(
            bert_dim=768,
            textcnn_dim=textcnn_num_filters * len(zh_textcnn_filter_sizes),
            hidden_dim=fusion_hidden_dim,
        )

        self.en_fusion = AttentionFusion(
            bert_dim=768,
            textcnn_dim=textcnn_num_filters * len(en_textcnn_filter_sizes),
            hidden_dim=fusion_hidden_dim,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(256, 3),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        language: str = "zh",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected_vocab = 30522 if language == "en" else 21128
        input_ids = input_ids.clamp(0, expected_vocab - 1)

        if language == "en":
            # 获取完整的 BERT 输出
            outputs = self.en_bert.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            # 使用整个序列的 hidden states 给 TextCNN
            bert_sequence = outputs.last_hidden_state  # (batch, seq_len, hidden)
            # 使用 CLS token 用于融合
            bert_cls = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
            # TextCNN 处理整个序列
            textcnn_output = self.en_textcnn(bert_embeddings=bert_sequence)
            # 融合
            fused_output = self.en_fusion(bert_cls, textcnn_output)
        else:
            # 获取完整的 BERT 输出
            outputs = self.zh_bert.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            # 使用整个序列的 hidden states 给 TextCNN
            bert_sequence = outputs.last_hidden_state  # (batch, seq_len, hidden)
            # 使用 CLS token 用于融合
            bert_cls = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
            # TextCNN 处理整个序列
            textcnn_output = self.zh_textcnn(bert_embeddings=bert_sequence)
            # 融合
            fused_output = self.zh_fusion(bert_cls, textcnn_output)

        logits = self.classifier(fused_output)

        return logits, bert_cls, textcnn_output
