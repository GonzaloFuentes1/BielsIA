"""Modelo GAT (Graph Attention Network) usando PyTorch Geometric."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    """Implementación simple de un GAT de dos capas.

    Igual que `GCN`, puede recibir un `Data` o `(x, edge_index)`.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            dropout=dropout,
            concat=False,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_or_data: torch.Tensor | Data,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  # type: ignore[override]
        if isinstance(x_or_data, Data):
            x = x_or_data.x
            edge_index = x_or_data.edge_index
        else:
            x = x_or_data
            if edge_index is None:
                raise ValueError("edge_index debe proporcionarse si no se pasa un Data.")

        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x
