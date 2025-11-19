
"""Lógica para recomendar jugadores a un equipo.

Esta versión trabaja sobre modelos dummy o reales y asume que la
"calidad" del jugador viene dada por la salida del modelo secuencial.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
from torch import nn
from torch_geometric.data import Data

from bielsia.utils.sequence_utils import build_player_sequences_from_graph_list


def recommend_players_for_team(
    graphs: List[Data],
    gnn_model: nn.Module,
    seq_model: nn.Module,
    candidate_indices: Iterable[int],
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    """Devuelve los mejores jugadores entre los candidatos.

    Por ahora, el "score" es simplemente la salida del modelo secuencial
    por jugador en el escenario actual (sin modificar el grafo por equipo).

    Args:
        graphs: lista de grafos (por temporada).
        gnn_model: modelo GNN.
        seq_model: modelo secuencial entrenado.
        candidate_indices: iterable de índices de jugadores candidatos.
        top_k: número de jugadores a recomendar.

    Returns:
        Lista de tuplas (player_index, score) ordenada por score descendente.
    """
    device = next(seq_model.parameters()).device
    gnn_model = gnn_model.to(device)
    seq_model = seq_model.to(device)

    graphs = [g.to(device) for g in graphs]
    seq_model.eval()

    with torch.no_grad():
        sequences = build_player_sequences_from_graph_list(graphs, gnn_model)
        # (N, T, d_emb)
        preds = seq_model(sequences)  # (N, output_dim)

    preds = preds.squeeze(-1).cpu()

    scored = [(idx, float(preds[idx].item())) for idx in candidate_indices]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
