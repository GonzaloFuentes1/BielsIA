"""Modelo Graphormer muy simplificado.

No pretende ser una implementación fiel del paper, pero sí ofrece una
interfaz coherente con el resto de modelos y sirve como baseline
transformer para grafos.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data


class Graphormer(nn.Module):
    """Graphormer simplificado usando un Transformer encoder sobre nodos.

    Toma las features de nodos como secuencia y aplica auto-atención para
    producir embeddings de nodos del mismo tamaño.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x_or_data: torch.Tensor | Data,
        edge_index: Optional[torch.Tensor] = None,  # se ignora por ahora
    ) -> torch.Tensor:  # type: ignore[override]
        if isinstance(x_or_data, Data):
            x = x_or_data.x
        else:
            x = x_or_data

        # x: (N, in_channels) -> tratamos nodos como secuencia
        x = self.input_proj(x).unsqueeze(0)  # (1, N, hidden_dim)
        x = self.encoder(x)  # (1, N, hidden_dim)
        x = self.output_proj(x)
        return x.squeeze(0)
