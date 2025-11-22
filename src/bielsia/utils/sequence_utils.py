"""Utilidades para construir secuencias temporales a partir de grafos.

La idea es tomar una lista de grafos (por temporada) y un modelo GNN
compatible (GCN/GAT/Graphormer o Dummy) y devolver un tensor con las
secuencias de embeddings por jugador listo para un modelo secuencial.
"""
from __future__ import annotations

from typing import List

import torch
from torch import nn
from torch_geometric.data import Data


def build_player_sequences_from_graph_list(
    graphs: List[Data],
    gnn_model: nn.Module,
    return_player_index: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """Construye secuencias de embeddings por jugador alineando IDs.

    Identifica la unión de todos los jugadores en los grafos y crea
    secuencias alineadas, rellenando con ceros (padding) cuando un
    jugador no está presente en una temporada.

    Args:
        graphs: lista de objetos Data, uno por paso temporal (temporada).
        gnn_model: modelo GNN que acepta `data` y devuelve embeddings de nodos
            de forma (num_players, emb_dim).
        return_player_index: si es True, devuelve también el dict {player: idx}.

    Returns:
        Tensor de forma (total_unique_players, num_steps, emb_dim).
        Si return_player_index es True, devuelve (sequences, player_to_idx).
    """
    if not graphs:
        raise ValueError("La lista de grafos no puede estar vacía.")

    # 1. Identificar todos los jugadores únicos y asignarles un índice global
    all_players = set()
    for g in graphs:
        if not hasattr(g, "player_ids"):
             raise ValueError("Cada grafo debe tener el atributo 'player_ids'.")
        all_players.update(g.player_ids)
    
    sorted_players = sorted(list(all_players))
    player_to_idx = {p: i for i, p in enumerate(sorted_players)}
    num_total_players = len(sorted_players)
    
    # 2. Generar embeddings por temporada
    with torch.no_grad():
        emb_list_per_season = []
        for g in graphs:
            emb = gnn_model(g) # (N_season, d_emb)
            emb_list_per_season.append(emb)

    # 3. Alinear embeddings en un tensor global (T, N_total, d_emb)
    num_steps = len(graphs)
    emb_dim = emb_list_per_season[0].shape[1]
    
    # Inicializamos con ceros (padding value)
    aligned_embeddings = torch.zeros((num_steps, num_total_players, emb_dim), device=graphs[0].x.device)
    
    for t, (g, emb) in enumerate(zip(graphs, emb_list_per_season)):
        # Mapear índices locales (temporada) a globales
        local_ids = g.player_ids
        for local_idx, player_name in enumerate(local_ids):
            if player_name in player_to_idx:
                global_idx = player_to_idx[player_name]
                aligned_embeddings[t, global_idx, :] = emb[local_idx, :]
                
    # 4. Reordenar a (N_total, T, d_emb)
    sequences = aligned_embeddings.transpose(0, 1)
    
    if return_player_index:
        return sequences, player_to_idx
    return sequences
