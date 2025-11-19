"""Utilidades para generar grafos dummy compatibles con la API esperada.

Se usan para desarrollar y probar la parte secuencial y de recomendación
sin depender de tener todos los datos reales listos.
"""
from __future__ import annotations

from typing import List

import torch
from torch_geometric.data import Data


def generate_dummy_graph(num_players: int = 50, feature_dim: int = 16) -> Data:
    """Genera un grafo simple con features gaussianas y un grafo tipo anillo.

    Args:
        num_players: número de nodos/jugadores.
        feature_dim: dimensión de las features por nodo.
    """
    x = torch.randn(num_players, feature_dim)

    # Grafo en anillo sencillo
    src = torch.arange(num_players, dtype=torch.long)
    dst = (src + 1) % num_players
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)

    data = Data(x=x, edge_index=edge_index)
    data.player_ids = torch.arange(num_players)
    return data


def generate_dummy_temporal_graphs(
    num_seasons: int = 4,
    num_players: int = 50,
    feature_dim: int = 16,
) -> List[Data]:
    """Genera una lista de grafos para simular varias temporadas.

    La conectividad se mantiene fija (anillo), pero las features cambian
    ligeramente entre temporadas para simular evolución.
    """
    base_graph = generate_dummy_graph(num_players=num_players, feature_dim=feature_dim)
    graphs: List[Data] = []

    for t in range(num_seasons):
        # Clonar estructura
        g_t = Data(x=base_graph.x.clone(), edge_index=base_graph.edge_index.clone())
        g_t.player_ids = base_graph.player_ids.clone()

        # Añadir pequeña deriva temporal a las features
        drift = 0.1 * (t + 1) * torch.randn_like(g_t.x)
        g_t.x = g_t.x + drift
        g_t.season_index = t
        graphs.append(g_t)

    return graphs
