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
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)

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
