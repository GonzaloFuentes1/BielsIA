"""
Módulo para orquestar la construcción del dataset completo para BielsIA.

Este script es el punto de entrada para el pipeline de procesamiento de datos,
el cual transforma los datos crudos/intermedios en el grafo procesado final
que será consumido por los modelos.
"""
from pathlib import Path
import logging
from typing import Dict, List

import pandas as pd
import torch
from torch_geometric.data import Data

from .graph_construction import _load_player_stats_for_season, _create_nodes_and_features, _create_edges, build_player_graph

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _build_single_season_graph(interim_dir: Path, season: int) -> Data:
    """Construye el grafo de jugadores para una temporada concreta.

    Esta función NO guarda a disco. Devuelve un objeto ``Data`` con:
    - ``x``: features de jugadores para la temporada ``season``.
    - ``edge_index``, ``edge_attr``: aristas y atributos simples.
    - ``player_ids``: lista de nombres de jugadores en el mismo orden que ``x``.
    - ``season``: año de la temporada.
    """

    df, player_map = _load_player_stats_for_season(interim_dir, season)
    if not player_map:
        raise RuntimeError(f"No se pudieron cargar stats para la temporada {season}.")

    x, team_to_idx, pos_to_idx = _create_nodes_and_features(df)
    if x.numel() == 0:
        raise RuntimeError(f"No se pudieron crear features de nodos para la temporada {season}.")

    edge_index, edge_attr, pos_labels = _create_edges(df, player_map, team_to_idx, pos_to_idx)

    player_ids: List[str] = [row.player for row in df.itertuples()]

    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    g.player_ids = player_ids
    g.season = season
    g.pos_labels = pos_labels
    return g


def _compute_y_vector_from_next_season(
    df_next: pd.DataFrame,
    player_index: Dict[str, int],
) -> torch.Tensor:
    """Construye el vector Y aplanado (attack, creation, defense, involvement).

    Para simplificar, usamos columnas per‑90 y contadores básicos de la tabla
    ``standard`` de la temporada siguiente. El resultado es un tensor de
    forma ``(num_players, d_y)`` alineado con ``player_index``.
    """

    # Definición simple de componentes (todas concatenadas y aplanadas)
    y_attack_cols = [
        "per_90_minutes_gls",
        "per_90_minutes_xg",
        "per_90_minutes_npxg",
    ]
    y_creation_cols = [
        "per_90_minutes_ast",
        "per_90_minutes_xag",
        "per_90_minutes_xg+xag",
    ]
    # Mejorado con stats de defense.csv
    y_defense_cols = [
        "tackles_tkl",
        "int_",
        "blocks_blocks",
        "clr_"
    ]
    y_involvement_cols = [
        "playing_time_90s",
    ]

    all_cols = y_attack_cols + y_creation_cols + y_defense_cols + y_involvement_cols

    # Aseguramos que las columnas existen; si no, las creamos con ceros
    for c in all_cols:
        if c not in df_next.columns:
            df_next[c] = 0.0
            
    # Limpieza de NaNs en las columnas objetivo
    df_next[all_cols] = df_next[all_cols].fillna(0.0)
    import numpy as np
    df_next[all_cols] = df_next[all_cols].replace([np.inf, -np.inf], 0.0)

    num_players = len(player_index)
    d_y = len(all_cols)
    y = torch.zeros((num_players, d_y), dtype=torch.float)

    # Creamos un mapa rápido de player -> fila en df_next
    df_next = df_next.set_index("player", drop=False)

    for name, idx in player_index.items():
        if name in df_next.index:
            row = df_next.loc[name]
            values = [float(row[c]) for c in all_cols]
            y[idx] = torch.tensor(values, dtype=torch.float)
        else:
            # Jugador no presente en la temporada siguiente: dejamos Y en ceros
            continue

    return y


def build_temporal_player_graphs(
    interim_dir: Path,
    seasons: List[int],
    processed_dir: Path = None,
) -> List[Data]:
    """Construye una lista de grafos por temporada con X_t e Y_{t+1}.

    - Por cada temporada ``t`` en ``seasons`` se construye un grafo ``Data_t``.
    - ``Data_t.x`` son las features de la temporada ``t``.
    - ``Data_t.y`` es el vector multi‑target de la temporada ``t+1``
      (attack, creation, defense, involvement) **aplanado**.
    - Para la última temporada sin temporada siguiente, ``y`` será todo ceros.
    - NORMALIZACIÓN: Si se proporciona ``processed_dir``, se calcula la media y std
      de todos los targets de entrenamiento y se normaliza Y. Se guarda el scaler.
    """

    if not seasons:
        raise ValueError("'seasons' no puede estar vacío.")

    seasons_sorted = sorted(seasons)

    # 1) Construimos grafos X_t y guardamos también los DataFrames
    graphs: List[Data] = []
    dfs: Dict[int, pd.DataFrame] = {}
    player_indexes: Dict[int, Dict[str, int]] = {}

    for season in seasons_sorted:
        df, player_map = _load_player_stats_for_season(interim_dir, season)
        if not player_map:
            logging.warning("Saltando temporada %s por falta de datos", season)
            continue

        dfs[season] = df
        player_indexes[season] = {name: idx for name, idx in player_map.items()}

        x, team_to_idx, pos_to_idx = _create_nodes_and_features(df)
        edge_index, edge_attr, pos_labels = _create_edges(df, player_map, team_to_idx, pos_to_idx)

        g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        g.player_ids = [row.player for row in df.itertuples()]
        g.season = season
        g.pos_labels = pos_labels
        # Guardamos el número de jugadores reales para saber dónde cortar/enmascarar
        g.num_real_players = len(player_map)
        graphs.append(g)

    # 2) Asignamos Y_{t+1} a cada grafo t
    season_to_graph: Dict[int, Data] = {g.season: g for g in graphs}

    for i, season in enumerate(seasons_sorted):
        g_t = season_to_graph.get(season)
        if g_t is None:
            continue

        # Si existe temporada siguiente, calculamos Y_{t+1}; si no, zeros
        next_season = seasons_sorted[i + 1] if i + 1 < len(seasons_sorted) else None
        
        # Inicializamos Y con ceros para TODOS los nodos (Players + Teams + Pos)
        num_total_nodes = g_t.x.size(0)
        # Determinamos dimensión de Y (hardcoded por ahora en _compute... o dinámica)
        # Llamamos a _compute para obtener la parte de jugadores
        
        y_players = None
        if next_season is not None and next_season in dfs:
            df_next = dfs[next_season]
            player_index = player_indexes[season]
            y_players = _compute_y_vector_from_next_season(df_next, player_index)
        
        if y_players is not None:
            dim_y = y_players.size(1)
            # Creamos Y completo
            y_full = torch.zeros((num_total_nodes, dim_y), dtype=torch.float)
            # Asignamos la parte de jugadores (asumiendo que son los primeros N nodos)
            num_players = y_players.size(0)
            y_full[:num_players, :] = y_players
            g_t.y = y_full
            
            # Máscara para entrenar solo sobre jugadores
            mask = torch.zeros(num_total_nodes, dtype=torch.bool)
            mask[:num_players] = True
            g_t.train_mask = mask
        else:
            # Caso última temporada o sin datos futuros
            # Definimos una dimensión dummy o basada en el anterior si es posible
            # Por seguridad, si no hay Y, ponemos dimensión 0 o consistente?
            # Mejor consistente con el resto si es posible, pero aquí no sabemos dim_y sin calcularlo.
            # Calculamos un y_dummy para saber la dimensión
            dummy_y = _compute_y_vector_from_next_season(dfs[season], player_indexes[season])
            dim_y = dummy_y.size(1)
            g_t.y = torch.zeros((num_total_nodes, dim_y), dtype=torch.float)
            g_t.train_mask = torch.zeros(num_total_nodes, dtype=torch.bool) # Nada que entrenar

    # Devolvemos la lista en orden temporal
    graphs_sorted = [season_to_graph[s] for s in seasons_sorted if s in season_to_graph]

    # --- 3) NORMALIZACIÓN DE TARGETS (Y) ---
    if processed_dir:
        logging.info("Calculando normalización de targets (Y)...")
        all_y_list = []
        for g in graphs_sorted:
            if hasattr(g, 'train_mask') and g.train_mask.any():
                # Solo usamos los targets válidos para calcular mean/std
                valid_y = g.y[g.train_mask]
                # Filtrar filas que sean todo ceros? No, 0 es un valor válido (ej. 0 goles).
                # Pero si el jugador no jugó, sus stats son 0.
                # Si incluimos muchos ceros, la media baja y la std baja.
                # Esto es correcto si queremos predecir 0 para los que no juegan.
                all_y_list.append(valid_y)
        
        if all_y_list:
            all_y_tensor = torch.cat(all_y_list, dim=0)
            mean = all_y_tensor.mean(dim=0)
            std = all_y_tensor.std(dim=0)
            
            # Evitar división por cero
            std[std < 1e-6] = 1.0
            
            logging.info(f"Target Mean: {mean[:5]}...")
            logging.info(f"Target Std: {std[:5]}...")
            
            # Aplicar normalización a TODOS los grafos
            for g in graphs_sorted:
                if hasattr(g, 'y') and g.y is not None:
                    # Normalizamos todo Y. Los nodos masked (Teams/Pos) tienen 0.
                    # (0 - mean) / std -> tendrán un valor negativo fijo.
                    # Como no entrenamos sobre ellos (train_mask=False), no importa.
                    g.y = (g.y - mean) / std
            
            # Guardar parámetros del scaler
            scaler_path = processed_dir / "y_scaler_params.pt"
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'mean': mean, 'std': std}, scaler_path)
            logging.info(f"Scaler de targets guardado en: {scaler_path}")
        else:
            logging.warning("No se encontraron targets válidos para calcular normalización.")

    return graphs_sorted


def create_dataset(interim_dir: Path, processed_dir: Path, seasons: list):
    """
    Orquesta la creación del dataset completo, incluyendo la construcción del grafo.

    Args:
        interim_dir (Path): Directorio donde se encuentran los datos intermedios (CSV).
        processed_dir (Path): Directorio donde se guardará el grafo procesado.
        seasons (list): Lista de temporadas a incluir en el grafo.
    """
    logging.info("Iniciando la construcción del dataset...")
    
    try:
        # Mantenemos la funcionalidad existente de construir un grafo single‑season
        build_player_graph(
            interim_dir=interim_dir,
            processed_dir=processed_dir,
            seasons=seasons,
        )
        logging.info("Construcción del dataset (single‑season) completada exitosamente.")
    except Exception as e:
        logging.error(f"Falló la construcción del dataset: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    # Ejemplo de uso como script
    ROOT_DIR = Path(__file__).resolve().parents[3]
    INTERIM_DATA_DIR = ROOT_DIR / "data" / "interim"
    PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
    
    # Definir las temporadas a procesar
    TARGET_SEASONS = [2022, 2023, 2024]

    create_dataset(
        interim_dir=INTERIM_DATA_DIR,
        processed_dir=PROCESSED_DATA_DIR,
        seasons=TARGET_SEASONS
    )
