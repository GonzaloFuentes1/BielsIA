"""
Módulo para la construcción de los grafos de jugadores.

Este script contiene la lógica principal para:
1. Cargar los datos intermedios (CSV de stats).
2. Parsear y limpiar los datos.
3. Construir un grafo de PyTorch Geometric con nodos (jugadores) y aristas (relaciones).
4. Guardar el objeto de grafo procesado.

Se puede ejecutar directamente para construir el grafo de prueba.
"""
import glob
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from tqdm import tqdm

import networkx as nx
import matplotlib.pyplot as plt
# Intentamos importar to_networkx, pero puede haber fallos de compatibilidad
try:
    from torch_geometric.utils import to_networkx
    _TO_NETWORKX_AVAILABLE = True
except ImportError:
    logging.warning("No se pudo importar 'to_networkx' de torch_geometric.utils. La visualización se hará manualmente.")
    _TO_NETWORKX_AVAILABLE = False
# ---------------------------------------------


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _load_team_stats(interim_dir: Path, season: int, league: str) -> pd.DataFrame:
    """
    Carga y procesa las estadísticas de equipo para agregar contexto.
    Retorna un DataFrame con index=team y columnas de contexto.
    """
    # logging.info(f"Cargando estadísticas de equipo para la temporada {season} ({league})...")
    
    # 1. Possession
    poss_path = interim_dir / "team_season_stats" / f"{league}_{season}_possession.csv"
    # 2. Standard (Goals For)
    std_path = interim_dir / "team_season_stats" / f"{league}_{season}_standard.csv"
    # 3. Defense (Tackles/Int as proxy for defensive work)
    def_path = interim_dir / "team_season_stats" / f"{league}_{season}_defense.csv"
    
    team_stats = pd.DataFrame()
    
    try:
        # Possession
        if poss_path.exists():
            df_poss = pd.read_csv(poss_path)
            if 'poss_' in df_poss.columns:
                team_stats = df_poss[['team', 'poss_']].copy()
                team_stats = team_stats.rename(columns={'poss_': 'team_possession'})
        
        # Goals For
        if std_path.exists():
            df_std = pd.read_csv(std_path)
            if 'performance_gls' in df_std.columns:
                df_std = df_std[['team', 'performance_gls']].rename(columns={'performance_gls': 'team_goals_for'})
                if team_stats.empty:
                    team_stats = df_std
                else:
                    team_stats = pd.merge(team_stats, df_std, on='team', how='outer')

        # Defense
        if def_path.exists():
            df_def = pd.read_csv(def_path)
            # Usamos tackles + intercepciones como proxy de actividad defensiva
            if 'tackles_tkl' in df_def.columns:
                df_def = df_def[['team', 'tackles_tkl']].rename(columns={'tackles_tkl': 'team_tackles'})
                if team_stats.empty:
                    team_stats = df_def
                else:
                    team_stats = pd.merge(team_stats, df_def, on='team', how='outer')
                    
    except Exception as e:
        logging.error(f"Error cargando team stats para {league}: {e}")
        return pd.DataFrame()
        
    return team_stats


def _load_shot_events(interim_dir: Path, season: int, league: str) -> pd.DataFrame:
    """
    Carga y agrega los eventos de tiro para obtener métricas avanzadas por jugador.
    """
    # logging.info(f"Cargando eventos de tiro para la temporada {season} ({league})...")
    
    shots_dir = interim_dir / "match_data" / "shot_events"
    
    file_pattern = str(shots_dir / f"{league}_{season}_*.csv")
    all_files = glob.glob(file_pattern)
    
    if not all_files:
        # logging.warning(f"No se encontraron archivos de tiros para el patrón: {file_pattern}")
        return pd.DataFrame()
    
    # Leer y concatenar (puede ser pesado, optimizar si es necesario)
    df_list = []
    for f in tqdm(all_files, desc=f"Leyendo shot events {league}", leave=False):
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception:
            continue
            
    if not df_list:
        return pd.DataFrame()
        
    full_shots = pd.concat(df_list, ignore_index=True)
    
    # Agregación por jugador
    # Métricas:
    # 1. Distancia promedio (avg_shot_distance)
    # 2. Diversidad de partes del cuerpo (body_part_diversity) -> Count unique? No, mejor % de tiros con pie malo/cabeza.
    #    Simplificación: % de tiros que NO son con "Right Foot" (asumiendo diestros mayoría, o simplemente entropía).
    #    Mejor: Count distinct body_parts.
    # 3. Dependencia de asistencia: % de tiros con sca_1_event == 'Pass (Live)'
    
    # Limpieza básica
    full_shots['distance_'] = pd.to_numeric(full_shots['distance_'], errors='coerce')
    
    # GroupBy
    # Nota: Las columnas en los CSV de shot_events tienen un guion bajo al final (e.g., player_, team_)
    grouped = full_shots.groupby(['player_', 'team_'])
    
    agg_df = grouped.agg(
        shots_count=('index', 'count'),
        avg_shot_distance=('distance_', 'mean'),
        # Calculamos npxg promedio por tiro si existe la columna
        npxg_per_shot=('xg_', 'mean') # Usamos xg_ como proxy si npxg no está explícito en shot events
    ).reset_index()
    
    # Renombrar columnas para que coincidan con el resto del pipeline
    agg_df = agg_df.rename(columns={'player_': 'player', 'team_': 'team'})

    # Calcular % de tiros asistidos (aproximación)
    # sca_1_event contiene el evento previo.
    if 'sca_1_event' in full_shots.columns:
        assisted_shots = full_shots[full_shots['sca_1_event'].isin(['Pass (Live)', 'Pass (Dead)'])].groupby(['player_', 'team_']).size()
        agg_df = agg_df.set_index(['player', 'team'])
        # El índice de assisted_shots es (player_, team_), necesitamos alinearlo
        assisted_shots.index.names = ['player', 'team']
        
        agg_df['assisted_shots'] = assisted_shots
        agg_df['assisted_shots'] = agg_df['assisted_shots'].fillna(0)
        agg_df['sca_dependency'] = agg_df['assisted_shots'] / agg_df['shots_count']
        agg_df = agg_df.reset_index()
        agg_df = agg_df.drop(columns=['assisted_shots'])
    else:
        agg_df['sca_dependency'] = 0.0

    return agg_df[['player', 'team', 'avg_shot_distance', 'npxg_per_shot', 'sca_dependency']]


def _check_position_similarity(pos_str_a: str, pos_str_b: str) -> bool:
    """
    Verifica si dos jugadores comparten al menos una posición.
    Ej: "DF,FW" y "MF,FW" retorna True.
    """
    if pd.isna(pos_str_a) or pd.isna(pos_str_b):
        return False
    
    pos_set_a = set(pos_str_a.split(','))
    pos_set_b = set(pos_str_b.split(','))
    
    return not pos_set_a.isdisjoint(pos_set_b)


def _load_player_stats_for_season(interim_dir: Path, season: int) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Carga y procesa las estadísticas de jugadores para **una** temporada.
    
    Fusiona 'standard', 'defense', 'passing', 'possession' y 'misc' para tener un set de features completo.
    Soporta múltiples ligas.
    """

    logging.info(f"Cargando estadísticas de jugadores para la temporada: {season}")

    # Detectar ligas disponibles
    pattern = str(interim_dir / "player_season_stats" / f"*_{season}_standard.csv")
    files = glob.glob(pattern)
    leagues = []
    for f in files:
        filename = Path(f).name
        # filename format: {League}_{Season}_standard.csv
        suffix = f"_{season}_standard.csv"
        if filename.endswith(suffix):
            league = filename[:-len(suffix)]
            leagues.append(league)
            
    if not leagues:
        logging.error(f"No se encontraron datos para la temporada {season}")
        return pd.DataFrame(), {}
        
    logging.info(f"Ligas encontradas para {season}: {leagues}")
    
    all_league_dfs = []

    for league in leagues:
        # 1. Cargar Standard Stats (Base)
        std_path = interim_dir / "player_season_stats" / f"{league}_{season}_standard.csv"
        try:
            df_std = pd.read_csv(std_path)
        except FileNotFoundError:
            logging.error(f"Archivo no encontrado: {std_path}")
            continue

        # Lista de archivos adicionales a fusionar
        additional_files = [
            ("defense", ['tackles_tkl', 'tackles_tklw', 'int_', 'clr_', 'blocks_blocks']),
            ("passing", ['total_cmp%', 'total_prgdist', 'kp_', 'ppa_', 'prgp_']),
            ("possession", ['touches_att_pen', 'take-ons_succ', 'carries_prgdist', 'carries_prgc', 'receiving_prgr']),
            ("misc", ['performance_recov', 'aerial_duels_won%'])
        ]

        merge_keys = ['player', 'team']
        player_stats_df = df_std

        for suffix, cols_to_keep in additional_files:
            file_path = interim_dir / "player_season_stats" / f"{league}_{season}_{suffix}.csv"
            try:
                df_add = pd.read_csv(file_path)
                
                # Validar claves
                if not all(k in df_add.columns for k in merge_keys):
                    # logging.warning(f"Claves de fusión faltantes en {suffix} ({league}). Saltando.")
                    continue

                # Seleccionar columnas
                if cols_to_keep:
                    # Filtrar solo las que existen
                    valid_cols = [c for c in cols_to_keep if c in df_add.columns]
                    cols_selection = merge_keys + valid_cols
                    df_subset = df_add[cols_selection]
                else:
                    df_subset = df_add

                # Merge
                player_stats_df = pd.merge(player_stats_df, df_subset, on=merge_keys, how='left')
                
                # Rellenar NaNs de las nuevas columnas con 0
                new_cols = [c for c in df_subset.columns if c not in merge_keys]
                player_stats_df[new_cols] = player_stats_df[new_cols].fillna(0)
                
            except FileNotFoundError:
                # logging.warning(f"Archivo {suffix} no encontrado: {file_path}. Saltando.")
                continue

        # --- NUEVO: Cargar y fusionar Team Stats ---
        team_stats_df = _load_team_stats(interim_dir, season, league)
        if not team_stats_df.empty:
            player_stats_df = pd.merge(player_stats_df, team_stats_df, on='team', how='left')
            # Rellenar NaNs en team stats (por si acaso)
            team_cols = [c for c in team_stats_df.columns if c != 'team']
            player_stats_df[team_cols] = player_stats_df[team_cols].fillna(0)
        
        # --- NUEVO: Cargar y fusionar Shot Events ---
        shot_events_df = _load_shot_events(interim_dir, season, league)
        if not shot_events_df.empty:
            # Merge por player y team
            player_stats_df = pd.merge(player_stats_df, shot_events_df, on=['player', 'team'], how='left')
            shot_cols = [c for c in shot_events_df.columns if c not in ['player', 'team']]
            player_stats_df[shot_cols] = player_stats_df[shot_cols].fillna(0)

        # Añadir columna de liga para trazabilidad (opcional, pero útil)
        player_stats_df['league'] = league
        
        all_league_dfs.append(player_stats_df)

    if not all_league_dfs:
        return pd.DataFrame(), {}

    # Concatenar todas las ligas
    full_player_stats_df = pd.concat(all_league_dfs, ignore_index=True)

    # --- NUEVO: Features Derivadas ---
    # 1. Role Starter %
    if 'playing_time_starts' in full_player_stats_df.columns and 'playing_time_mp' in full_player_stats_df.columns:
        full_player_stats_df['role_starter_pct'] = full_player_stats_df['playing_time_starts'] / full_player_stats_df['playing_time_mp'].replace(0, 1)
    
    # 2. Player Goals Ratio (vs Team Goals)
    if 'performance_gls' in full_player_stats_df.columns and 'team_goals_for' in full_player_stats_df.columns:
        full_player_stats_df['player_goals_ratio'] = full_player_stats_df['performance_gls'] / full_player_stats_df['team_goals_for'].replace(0, 1)

    # --- MAPEO DE COLUMNAS ---
    COLUMN_MAP = {
        'player': 'player', 
        'team': 'team',     
        'pos': 'pos_',
        'age': 'age_',
    }
    
    rename_map = {v: k for k, v in COLUMN_MAP.items() if v in full_player_stats_df.columns}
    full_player_stats_df = full_player_stats_df.rename(columns=rename_map)
    
    # Columnas que *esperamos* tener después del renombre
    expected_cols_for_dropna = ['team', 'pos', 'player']
    
    missing_cols_after_rename = [col for col in expected_cols_for_dropna if col not in full_player_stats_df.columns]
    if missing_cols_after_rename:
        logging.error(f"Columnas esperadas no se encontraron después de renombrar: {missing_cols_after_rename}")
        return pd.DataFrame(), {}

    full_player_stats_df = full_player_stats_df.dropna(subset=expected_cols_for_dropna)
    full_player_stats_df = full_player_stats_df.drop_duplicates(subset=['player'], keep='first')
    full_player_stats_df = full_player_stats_df.reset_index(drop=True)

    player_map = {row.player: row.Index for row in full_player_stats_df.itertuples()}
    
    logging.info(f"Cargados {len(player_map)} jugadores únicos de la temporada {season} (Ligas: {len(leagues)}).")
    
    return full_player_stats_df, player_map


def _create_nodes_and_features(player_stats_df: pd.DataFrame, team_stats_df: pd.DataFrame = None) -> Tuple[torch.Tensor, Dict[str, int], Dict[str, int]]:
    """
    Convierte el DataFrame de stats de jugadores en el tensor de features de nodos.
    Implementa un enfoque de HIPERGRAFO (Bipartito):
    - Nodos 0..N-1: Jugadores
    - Nodos N..N+T-1: Equipos (Virtual Nodes)
    - Nodos N+T..N+T+P-1: Posiciones (Virtual Nodes)
    
    Returns:
        x (Tensor): Features concatenadas [Players; Teams; Positions]
        team_to_idx (Dict): Mapeo de nombre de equipo a índice de nodo virtual.
        pos_to_idx (Dict): Mapeo de nombre de posición a índice de nodo virtual.
    """
    logging.info("Creando tensor de features para los nodos (jugadores + hipernodos)...")
    
    # --- 1. Features de Jugadores ---
    feature_cols = [
        'age', 
        'playing_time_mp', 'playing_time_starts', 'playing_time_min', 'playing_time_90s',
        'performance_gls', 'performance_ast', 'performance_g+a', 'performance_g-pk',
        'performance_pk', 'performance_pkatt', 'performance_crdy', 'performance_crdr',
        'expected_xg', 'expected_npxg', 'expected_xag', 'expected_npxg+xag',
        'progression_prgc', 'progression_prgp', 'progression_prgr',
        'per_90_minutes_gls', 'per_90_minutes_ast', 'per_90_minutes_g+a',
        'per_90_minutes_g-pk', 'per_90_minutes_g+a-pk', 'per_90_minutes_xg',
        'per_90_minutes_xag', 'per_90_minutes_xg+xag', 'per_90_minutes_npxg',
        'per_90_minutes_npxg+xag',
        # Defense
        'tackles_tkl', 'tackles_tklw', 'int_', 'clr_', 'blocks_blocks',
        # Passing
        'total_cmp%', 'total_prgdist', 'kp_', 'ppa_', 'prgp_',
        # Possession
        'touches_att_pen', 'take-ons_succ', 'carries_prgdist', 'carries_prgc', 'receiving_prgr',
        # Misc
        'performance_recov', 'aerial_duels_won%',
        # --- NUEVAS FEATURES (Sección 3) ---
        # Team Context
        'team_possession', 'team_goals_for', 'team_tackles',
        # Shot Events (Advanced)
        'avg_shot_distance', 'npxg_per_shot', 'sca_dependency',
        # Derived / Role
        'role_starter_pct', 'player_goals_ratio'
    ]
    
    # existent_feature_cols = [col for col in feature_cols if col in player_stats_df.columns]
    # logging.info(f"Se usarán {len(existent_feature_cols)} columnas como features de nodo.")
    
    # if not existent_feature_cols:
    #     logging.error("No se encontró ninguna columna de features en el DataFrame.")
    #     return torch.empty(0), {}, {}

    # features_df = player_stats_df[existent_feature_cols].copy()
    
    # --- CAMBIO: Forzar todas las columnas, rellenando con 0 si faltan ---
    # Esto asegura consistencia dimensional entre temporadas (ej. si falta shot data)
    for col in feature_cols:
        if col not in player_stats_df.columns:
            player_stats_df[col] = 0.0
            
    features_df = player_stats_df[feature_cols].copy()
    logging.info(f"Usando {len(feature_cols)} features fijas para consistencia.")

    # --- LIMPIEZA ESPECÍFICA DE COLUMNAS ---
    # Limpiar columna 'age' si contiene strings tipo "28-045"
    if 'age' in features_df.columns:
        # Convertir a string, tomar la parte antes del guion, y convertir a float
        features_df['age'] = features_df['age'].astype(str).apply(lambda x: x.split('-')[0] if '-' in x else x)
        features_df['age'] = pd.to_numeric(features_df['age'], errors='coerce').fillna(0)

    # Limpieza de NaNs e Infinitos antes de escalar
    features_df = features_df.fillna(0)
    # Reemplazar inf/-inf con 0 (puede ocurrir por divisiones por cero)
    import numpy as np
    features_df = features_df.replace([np.inf, -np.inf], 0)

    # --- 2. Features de Equipos (Virtual Nodes) ---
    unique_teams = sorted(player_stats_df['team'].unique())
    num_players = len(player_stats_df)
    team_to_idx = {team: num_players + i for i, team in enumerate(unique_teams)}

    # --- NUEVO: Features Reales de Equipos (Mean Aggregation) ---
    # Agregamos la columna 'team' temporalmente para agrupar
    features_df_with_team = features_df.copy()
    features_df_with_team['team_temp_id'] = player_stats_df['team'].values
    
    # Group by team and calculate mean (Centroide del equipo)
    team_agg_df = features_df_with_team.groupby('team_temp_id').mean()
    
    # Ensure alignment with unique_teams
    team_agg_df = team_agg_df.reindex(unique_teams).fillna(0)
    
    # Escalar features de jugadores
    scaler = StandardScaler()
    player_features = scaler.fit_transform(features_df)
    num_features = player_features.shape[1]
    
    # Escalar features de equipos con el MISMO scaler para compartir espacio latente
    team_features_numpy = scaler.transform(team_agg_df)
    team_features = torch.tensor(team_features_numpy, dtype=torch.float)
    
    # --- 3. Features de Posiciones (Virtual Nodes) ---
    # Desglosamos posiciones compuestas "DF,MF" -> "DF", "MF"
    all_pos = set()
    for p in player_stats_df['pos']:
        if pd.notna(p):
            for sub_p in p.split(','):
                all_pos.add(sub_p.strip())
    unique_pos = sorted(list(all_pos))
    
    pos_start_idx = num_players + len(unique_teams)
    pos_to_idx = {pos: pos_start_idx + i for i, pos in enumerate(unique_pos)}
    
    pos_features = torch.zeros((len(unique_pos), num_features), dtype=torch.float)
    
    # --- 4. Concatenación ---
    all_features = torch.cat([
        torch.tensor(player_features, dtype=torch.float),
        team_features,
        pos_features
    ], dim=0)
    
    logging.info(f"Tensor de features creado. Shape: {all_features.shape} (Players={len(player_features)}, Teams={len(team_features)}, Pos={len(pos_features)})")
    
    return all_features, team_to_idx, pos_to_idx


def _create_edges(player_stats_df: pd.DataFrame, player_map: Dict[str, int], team_to_idx: Dict[str, int], pos_to_idx: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Crea las aristas del HIPERGRAFO.
    En lugar de conectar todos con todos (clique), conectamos:
    - Jugador <-> Equipo (Hiperarista de Equipo)
    - Jugador <-> Posición (Hiperarista de Posición)
    
    Esto reduce drásticamente el número de aristas y crea una estructura más limpia.
    """
    logging.info("Creando aristas de hipergrafo (Jugador -> Equipo, Jugador -> Posición)...")
    
    edge_list = []
    attr_list = [] # 0: Team Link, 1: Pos Link
    
    # Iteramos sobre cada jugador para conectarlo a sus hipernodos
    for row in tqdm(player_stats_df.itertuples(), total=len(player_stats_df), desc="Conectando hipernodos"):
        p_idx = row.Index # Índice del jugador (0..N-1)
        
        # 1. Conexión a Equipo
        if row.team in team_to_idx:
            t_idx = team_to_idx[row.team]
            # Bidireccional
            edge_list.append([p_idx, t_idx])
            edge_list.append([t_idx, p_idx])
            attr_list.append([1.0, 0.0]) # Feature de arista: [EsEquipo, EsPosicion]
            attr_list.append([1.0, 0.0])
            
        # 2. Conexión a Posición(es)
        if pd.notna(row.pos):
            for sub_p in row.pos.split(','):
                sub_p = sub_p.strip()
                if sub_p in pos_to_idx:
                    pos_node_idx = pos_to_idx[sub_p]
                    # Bidireccional
                    edge_list.append([p_idx, pos_node_idx])
                    edge_list.append([pos_node_idx, p_idx])
                    attr_list.append([0.0, 1.0])
                    attr_list.append([0.0, 1.0])

    if not edge_list:
        logging.warning("No se crearon aristas.")
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0), []

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(attr_list, dtype=torch.float)
    
    # Etiquetas para visualización (solo jugadores por ahora)
    pos_labels = [row.pos for row in player_stats_df.itertuples()]
    
    logging.info(f"Aristas creadas. edge_index shape: {edge_index.shape}")
    
    return edge_index, edge_attr, pos_labels


def _assemble_and_save_graph(
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    player_map: Dict[str, int],
    processed_dir: Path,
    seasons: List[int],
    pos_labels: List[str] = None,
    team_names: List[str] = None,
    pos_names: List[str] = None
):
    """Ensambla el objeto Data de PyG y lo guarda en disco."""
    logging.info("Ensamblando y guardando el objeto de grafo final...")
    
    graph = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    
    graph.player_map = player_map
    graph.seasons = seasons
    if pos_labels:
        graph.pos_labels = pos_labels
    if team_names:
        graph.team_names = team_names
    if pos_names:
        graph.pos_names = pos_names

    try:
        graph.validate()
        logging.info("Validación del grafo de PyG exitosa.")
    except Exception as e:
        logging.error(f"La validación del grafo de PyG falló: {e}")
        raise

    season_str = f"{seasons[0]}_{seasons[-1]}"
    output_path = processed_dir / "graphs" / f"player_graph_{season_str}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(graph, output_path)
    logging.info(f"Grafo guardado exitosamente en: {output_path}")


def build_player_graph(interim_dir: Path, processed_dir: Path, seasons: List[int]):
    """Orquesta la construcción del grafo de jugadores para MULTIPLES temporadas."""
    logging.info(f"Iniciando la construcción de grafos para las temporadas: {seasons}")

    if not seasons:
        logging.error("La lista 'seasons' no puede estar vacía.")
        return

    for season in seasons:
        logging.info(f"--- Procesando Temporada {season} ---")
        player_stats_df, player_map = _load_player_stats_for_season(interim_dir, season)
        
        if not player_map:
            logging.error(f"No se pudieron cargar las estadísticas para {season}. Saltando.")
            continue

        node_features, team_to_idx, pos_to_idx = _create_nodes_and_features(player_stats_df)
        if node_features.shape[0] == 0:
            logging.error(f"No se pudieron crear features para {season}. Saltando.")
            continue

        edge_index, edge_attr, pos_labels = _create_edges(player_stats_df, player_map, team_to_idx, pos_to_idx)

        # Extraer nombres de equipos y posiciones
        sorted_teams = sorted(team_to_idx.items(), key=lambda item: item[1])
        team_names = [item[0] for item in sorted_teams]

        sorted_pos = sorted(pos_to_idx.items(), key=lambda item: item[1])
        pos_names = [item[0] for item in sorted_pos]

        # Guardar grafo individual por temporada
        # Nota: _assemble_and_save_graph usa seasons[0] y seasons[-1] para el nombre.
        # Si pasamos una lista de un solo elemento, el nombre será "2020_2020".
        _assemble_and_save_graph(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            player_map=player_map,
            processed_dir=processed_dir,
            seasons=[season], # Pasamos solo la temporada actual
            pos_labels=pos_labels,
            team_names=team_names,
            pos_names=pos_names
        )
    
    logging.info("Construcción de todos los grafos finalizada.")


# --- Bloque para ejecutar el script ---
if __name__ == "__main__":
    import shutil

    # --- 1. Definición de Rutas ---
    SCRIPT_PATH = Path(__file__).resolve()
    BASE_DIR = SCRIPT_PATH.parent.parent.parent.parent
    
    DATA_DIR = BASE_DIR / "data" / "interim"
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
    REPORTS_DIR = BASE_DIR / "reports"

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Directorio Raíz (BASE_DIR): {BASE_DIR}")
    logging.info(f"Directorio de Datos (DATA_DIR): {DATA_DIR}")
    logging.info(f"Directorio Procesado (PROCESSED_DIR): {PROCESSED_DATA_DIR}")
    logging.info(f"Directorio de Reportes (REPORTS_DIR): {REPORTS_DIR}")
    
    # Limpiar grafos antiguos
    graphs_dir = PROCESSED_DATA_DIR / "graphs"
    if graphs_dir.exists():
        logging.info(f"Limpiando directorio de grafos antiguos: {graphs_dir}")
        shutil.rmtree(graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    seasons_to_build = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    
    # --- 2. Construcción del Grafo ---
    build_player_graph(
        interim_dir=DATA_DIR,
        processed_dir=PROCESSED_DATA_DIR,
        seasons=seasons_to_build
    )
    
    # --- 3. Carga, Estadísticas y Visualización (Iterativo) ---
    logging.info("--- Verificación y Visualización de Grafos Generados ---")
    
    for season in seasons_to_build:
        try:
            season_str = f"{season}_{season}"
            graph_path = PROCESSED_DATA_DIR / "graphs" / f"player_graph_{season_str}.pt"
            
            if not graph_path.exists():
                logging.warning(f"Grafo no encontrado para {season}: {graph_path}")
                continue

            # Fix for PyTorch 2.6+ security warning/error
            try:
                loaded_graph = torch.load(graph_path, weights_only=False)
            except TypeError:
                # Fallback for older PyTorch versions that don't support weights_only
                loaded_graph = torch.load(graph_path)
                
            logging.info(f"Grafo cargado exitosamente: {graph_path}")
            
            # --- Estadísticas ---
            print(f"\n--- Estadísticas del Grafo {season} ---")
            print(f"Total de nodos (vértices): {loaded_graph.num_nodes}")
            print(f"Total de aristas (bidireccional): {loaded_graph.num_edges}")
            print(f"Shape de features de nodo (x): {loaded_graph.x.shape}")
            print(f"Shape de features de arista (edge_attr): {loaded_graph.edge_attr.shape}")
            print(f"El grafo tiene {loaded_graph.num_edges / 2:.0f} conexiones únicas.")

            # --- Visualización ---
            logging.info(f"--- Generando visualización para {season}... ---")
            
            g_nx = None
            # Re-check availability inside loop just in case
            if _TO_NETWORKX_AVAILABLE:
                try:
                    g_nx = to_networkx(loaded_graph, to_undirected=True)
                except Exception as e:
                    logging.warning(f"Fallo al convertir con 'to_networkx': {e}. Intentando construcción manual.")
            
            if g_nx is None:
                g_nx = nx.Graph()
                g_nx.add_nodes_from(range(loaded_graph.num_nodes))
                edges_to_add = loaded_graph.edge_index.t().tolist()
                g_nx.add_edges_from(edges_to_add)

            plt.figure(figsize=(24, 24))
            
            # --- COLOREADO DE NODOS Y ARISTAS (HIPERGRAFO AVANZADO) ---
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors
            
            pos_color_map = {
                'GK': '#F39C12', # Orange
                'DF': '#3498DB', # Blue
                'MF': '#2ECC71', # Green
                'FW': '#E74C3C'  # Red
            }
            default_player_color = '#BDC3C7' # Grey

            num_players = len(loaded_graph.player_map)
            total_nodes = loaded_graph.num_nodes
            
            team_names = getattr(loaded_graph, 'team_names', [])
            num_teams = len(team_names)
            
            team_cmap = cm.get_cmap('tab20', num_teams if num_teams > 0 else 20)
            team_colors = [mcolors.to_hex(team_cmap(i)) for i in range(num_teams)]
            
            pos_labels = getattr(loaded_graph, 'pos_labels', [])
            pos_names = getattr(loaded_graph, 'pos_names', [])
            
            player_nodes = []
            player_colors = []
            team_nodes = []
            team_node_colors = []
            pos_nodes = []
            pos_node_colors = []
            
            for i in range(total_nodes):
                if i < num_players:
                    player_nodes.append(i)
                    if i < len(pos_labels):
                        p_pos = pos_labels[i].split(',')[0]
                        color = pos_color_map.get(p_pos, default_player_color)
                    else:
                        color = default_player_color
                    player_colors.append(color)
                elif i < num_players + num_teams:
                    team_nodes.append(i)
                    team_idx = i - num_players
                    if team_idx < len(team_colors):
                        team_node_colors.append(team_colors[team_idx])
                    else:
                        team_node_colors.append('#8E44AD')
                else:
                    pos_nodes.append(i)
                    pos_idx = i - num_players - num_teams
                    if pos_idx < len(pos_names):
                        p_name = pos_names[pos_idx]
                        color = pos_color_map.get(p_name, '#2C3E50')
                    else:
                        color = '#2C3E50'
                    pos_node_colors.append(color)

            pos = nx.spring_layout(g_nx, k=0.12, iterations=200, seed=42)
            
            nx.draw_networkx_nodes(g_nx, pos, nodelist=player_nodes, node_color=player_colors, node_shape='o', node_size=80, alpha=0.85, linewidths=0.5, edgecolors='white', label='Jugadores')
            nx.draw_networkx_nodes(g_nx, pos, nodelist=team_nodes, node_color=team_node_colors, node_shape='s', node_size=500, alpha=0.95, linewidths=2, edgecolors='black', label='Equipos')
            nx.draw_networkx_nodes(g_nx, pos, nodelist=pos_nodes, node_color=pos_node_colors, node_shape='D', node_size=400, alpha=0.95, linewidths=2, edgecolors='white', label='Posiciones')

            edge_colors_list = []
            for u, v in g_nx.edges():
                target = max(u, v)
                if target in team_nodes:
                    t_idx = target - num_players
                    if t_idx < len(team_colors):
                        edge_colors_list.append(team_colors[t_idx])
                    else:
                        edge_colors_list.append('#BDC3C7')
                elif target in pos_nodes:
                    edge_colors_list.append('#D7DBDD') 
                else:
                    edge_colors_list.append('#BDC3C7')

            nx.draw_networkx_edges(g_nx, pos, edge_color=edge_colors_list, width=0.8, alpha=0.4)
            
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='GK', markerfacecolor=pos_color_map['GK'], markersize=10),
                Line2D([0], [0], marker='o', color='w', label='DF', markerfacecolor=pos_color_map['DF'], markersize=10),
                Line2D([0], [0], marker='o', color='w', label='MF', markerfacecolor=pos_color_map['MF'], markersize=10),
                Line2D([0], [0], marker='o', color='w', label='FW', markerfacecolor=pos_color_map['FW'], markersize=10),
                Line2D([0], [0], marker='s', color='w', label='Equipos', markerfacecolor='grey', markersize=12, markeredgecolor='black'),
                Line2D([0], [0], marker='D', color='w', label='Posiciones', markerfacecolor='black', markersize=12),
            ]
            plt.legend(handles=legend_elements, loc='upper right', title="Tipos de Nodo")
            
            # Guardar en reports/figures
            figures_dir = REPORTS_DIR / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            output_viz_path = figures_dir / f"hypergraph_visualization_{season_str}.png"
            
            plt.title(f"Grafo de Jugadores (Hipergrafo) - Temporada {season}\nNodos: {total_nodes}, Aristas: {loaded_graph.num_edges}", fontsize=16)
            plt.axis('off')
            plt.savefig(output_viz_path, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Visualización guardada: {output_viz_path}")

        except Exception as e:
            logging.error(f"Error procesando visualización para {season}: {e}")