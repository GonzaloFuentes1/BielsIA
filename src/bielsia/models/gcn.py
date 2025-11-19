"""Modelo GCN (Graph Convolutional Network) usando PyTorch Geometric.

Este modelo sirve como implementación base para el pipeline de BielsIA.
Produce embeddings de nodos a partir de un grafo de jugadores.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """Implementación simple de un GCN de dos capas.

    Interfaz pensada para ser compatible con el resto del pipeline:
    - Si se pasa un `Data`, devuelve embeddings de nodos.
    - Si se pasa `(x, edge_index)`, también funciona.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_or_data: torch.Tensor | Data,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  # type: ignore[override]
        """Calcula embeddings de nodos.

        Args:
            x_or_data: objeto `Data` o tensor de features de nodos.
            edge_index: matriz de aristas si no se pasa un `Data`.
        Returns:
            Tensor de forma (num_nodes, out_channels).
        """
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
