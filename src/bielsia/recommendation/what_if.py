"""Simulaciones "what if" sobre la carrera de un jugador.

Trabaja sobre grafos (reales o dummy), un modelo GNN y un modelo
secuencial ya entrenado para comparar escenarios alternativos.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

import torch
from torch import nn
from torch_geometric.data import Data

from bielsia.utils.sequence_utils import build_player_sequences_from_graph_list


def simulate_player_career_what_if(
    graphs: List[Data],
    gnn_model: nn.Module,
    seq_model: nn.Module,
    player_index: int,
    scenarios: Dict[str, List[Data]],
) -> Dict[str, float]:
    """Compara el score de carrera de un jugador en distintos escenarios.

    Args:
        graphs: lista de grafos del escenario real.
        gnn_model: modelo GNN entrenado o dummy.
        seq_model: modelo secuencial entrenado.
        player_index: índice del jugador en las matrices.
        scenarios: diccionario nombre -> lista de grafos modificados.

    Returns:
        Diccionario nombre -> score escalar para el jugador.
    """
    device = next(seq_model.parameters()).device
    gnn_model = gnn_model.to(device)
    seq_model = seq_model.to(device)

    seq_model.eval()
    scores: Dict[str, float] = {}

    with torch.no_grad():
        # Escenario real
        real_graphs = [g.to(device) for g in graphs]
        real_sequences = build_player_sequences_from_graph_list(real_graphs, gnn_model)
        real_seq = real_sequences[player_index : player_index + 1]  # (1, T, d_emb)
        real_score = seq_model(real_seq).item()
        scores["real"] = real_score

        # Escenarios alternativos
        for name, sc_graphs in scenarios.items():
            alt_graphs = [g.to(device) for g in sc_graphs]
            alt_sequences = build_player_sequences_from_graph_list(alt_graphs, gnn_model)
            alt_seq = alt_sequences[player_index : player_index + 1]
            alt_score = seq_model(alt_seq).item()
            scores[name] = alt_score

    return scores
