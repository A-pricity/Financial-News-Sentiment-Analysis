import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int = 21128,
        embedding_dim: int = 768,
        filter_sizes: list = [2, 3, 4],
        num_filters: int = 256,
        dropout: float = 0.3,
        use_bert_embeddings: bool = True,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        
        # 当使用 BERT embeddings 时，不需要自己的 embedding 层
        if not use_bert_embeddings:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        else:
            self.embedding = None  # 直接使用 BERT 的 hidden states

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=fs,
                )
                for fs in filter_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout)

        self.output_dim = num_filters * len(filter_sizes)

    def forward(self, input_ids: torch.Tensor = None, bert_embeddings: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs (仅当不使用 BERT embeddings 时使用)
            bert_embeddings: BERT 的 hidden states (可选，如果提供则直接使用)
                             期望形状：(batch_size, seq_len, hidden_size)
        """
        if bert_embeddings is not None:
            # 直接使用 BERT 的输出作为 TextCNN 的输入
            # BERT output shape: (batch_size, hidden_size) - CLS token
            # 需要转换为 (batch_size, 1, hidden_size) 以适应 Conv1d
            if bert_embeddings.dim() == 2:
                # 如果是 CLS token，添加 sequence 维度
                bert_embeddings = bert_embeddings.unsqueeze(1)
            embedded = bert_embeddings
        elif self.embedding is not None and input_ids is not None:
            # 使用自己的 embedding 层
            embedded = self.embedding(input_ids)
        else:
            raise ValueError("Either bert_embeddings or input_ids must be provided")

        embedded = embedded.transpose(1, 2)

        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)

        concat = torch.cat(conv_outputs, dim=1)

        output = self.dropout(concat)

        return output

    def get_output_dim(self) -> int:
        return self.output_dim
