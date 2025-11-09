"""
Módulo para la construcción de los grafos de jugadores.

Este script contiene la lógica principal para:
1. Cargar los datos intermedios (CSV de stats).
2. Parsear y limpiar los datos.
3. Construir un grafo de PyTorch Geometric con nodos (jugadores) y aristas (relaciones).
4. Guardar el objeto de grafo procesado.

Se puede ejecutar directamente para construir el grafo de prueba.
"""
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


def _load_player_stats(interim_dir: Path, seasons: List[int]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Carga, concatena y procesa las estadísticas de jugadores.
    Para esta versión simple, carga solo la *primera* temporada de la lista.
    """
    logging.info(f"Cargando estadísticas de jugadores para las temporadas: {seasons}")
    
    if not seasons:
        logging.error("La lista 'seasons' no puede estar vacía.")
        return pd.DataFrame(), {}

    target_season = seasons[0]
    if len(seasons) > 1:
        logging.warning(f"Se proporcionaron múltiples temporadas. Usando solo la primera: {target_season}")

    file_path = interim_dir / f"ENG-Premier League_{target_season}_standard.csv"
    
    try:
        player_stats_df = pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(f"Archivo no encontrado: {file_path}")
        return pd.DataFrame(), {}

    logging.info(f"Columnas encontradas en el CSV: {list(player_stats_df.columns)}")
    
    # --- MAPEO DE COLUMNAS ---
    # Edita esto según lo que veas en el log de "Columnas encontradas".
    # Sintaxis: { 'nombre_que_espera_el_script': 'nombre_REAL_en_el_CSV' }
    COLUMN_MAP = {
        'player': 'player', 
        'team': 'team',     
        'pos': 'pos_',      # ¡Ajustado a 'pos_' según el último error!
        'age': 'age_',      # Agregado 'age' para asegurar su uso en features
    }
    
    rename_map = {v: k for k, v in COLUMN_MAP.items() if v in player_stats_df.columns}
    
    if len(rename_map) != len(COLUMN_MAP):
        missing_cols_actual = set(COLUMN_MAP.values()) - set(player_stats_df.columns)
        missing_cols_expected = set(COLUMN_MAP.keys()) - set(rename_map.values())
        logging.warning(f"No se encontraron todas las columnas del MAPA en el CSV. Faltan en el CSV: {missing_cols_actual}. Faltan para el script: {missing_cols_expected}")
        
    player_stats_df = player_stats_df.rename(columns=rename_map)
    
    # Columnas que *esperamos* tener después del renombre
    expected_cols_for_dropna = ['team', 'pos', 'player']
    
    missing_cols_after_rename = [col for col in expected_cols_for_dropna if col not in player_stats_df.columns]
    if missing_cols_after_rename:
        logging.error(f"Columnas esperadas no se encontraron después de renombrar: {missing_cols_after_rename}")
        logging.error("Por favor, corrige el 'COLUMN_MAP' en la función _load_player_stats.")
        return pd.DataFrame(), {}

    player_stats_df = player_stats_df.dropna(subset=expected_cols_for_dropna)
    player_stats_df = player_stats_df.drop_duplicates(subset=['player'], keep='first')
    player_stats_df = player_stats_df.reset_index(drop=True)

    player_map = {row.player: row.Index for row in player_stats_df.itertuples()}
    
    logging.info(f"Cargados {len(player_map)} jugadores únicos de la temporada {target_season}.")
    
    return player_stats_df, player_map


def _create_nodes_and_features(player_stats_df: pd.DataFrame) -> torch.Tensor:
    """Convierte el DataFrame de stats de jugadores en el tensor de features de nodos."""
    logging.info("Creando tensor de features para los nodos (jugadores)...")
    
    feature_cols = [
        'age', # 'age' ya está renombrado si se usó el COLUMN_MAP
        'playing_time_mp', 'playing_time_starts', 'playing_time_min', 'playing_time_90s',
        'performance_gls', 'performance_ast', 'performance_g+a', 'performance_g-pk',
        'performance_pk', 'performance_pkatt', 'performance_crdy', 'performance_crdr', # crdr en vez de crdrex
        'expected_xg', 'expected_npxg', 'expected_xag', 'expected_npxg+xag',
        'progression_prgc', 'progression_prgp', 'progression_prgr', # prgp en vez de prgpp
        'per_90_minutes_gls', 'per_90_minutes_ast', 'per_90_minutes_g+a',
        'per_90_minutes_g-pk', 'per_90_minutes_g+a-pk', 'per_90_minutes_xg',
        'per_90_minutes_xag', 'per_90_minutes_xg+xag', 'per_90_minutes_npxg',
        'per_90_minutes_npxg+xag'
    ]
    
    # Asegúrate de que las columnas tengan los nombres correctos (sin guiones bajos)
    # según tu lista original y el mapeo.
    # También, 'performance_crdr' y 'progression_prgp' son los nombres correctos del CSV
    # de acuerdo a la lista que entregaste en el primer mensaje.
    
    existent_feature_cols = [col for col in feature_cols if col in player_stats_df.columns]
    logging.info(f"Se usarán {len(existent_feature_cols)} columnas como features de nodo.")
    
    if not existent_feature_cols:
        logging.error("No se encontró ninguna columna de features en el DataFrame.")
        return torch.empty(0)

    features_df = player_stats_df[existent_feature_cols]
    features_df = features_df.fillna(0)
    
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features_df)
    
    logging.info(f"Tensor de features creado con shape: {features_normalized.shape}")
    return torch.tensor(features_normalized, dtype=torch.float)


def _create_edges(player_stats_df: pd.DataFrame, player_map: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Crea las aristas del grafo y sus atributos.
    Aristas basadas en:
    1. Mismo equipo.
    2. Posición similar.
    """
    logging.info("Creando aristas (mismo equipo, misma posición)...")
    
    num_players = len(player_stats_df)
    edge_list = []
    attr_list = []
    
    player_data = list(player_stats_df.itertuples())
    
    for i in tqdm(range(num_players), desc="Calculando aristas"):
        for j in range(i + 1, num_players):
            player_i = player_data[i]
            player_j = player_data[j]
            
            is_team_link = (player_i.team == player_j.team)
            is_pos_link = _check_position_similarity(player_i.pos, player_j.pos)
            
            if is_team_link or is_pos_link:
                edge_list.append([player_i.Index, player_j.Index])
                edge_list.append([player_j.Index, player_i.Index])
                
                attr = [1.0 if is_team_link else 0.0, 1.0 if is_pos_link else 0.0]
                attr_list.append(attr)
                attr_list.append(attr)
                        
    if not edge_list:
        logging.warning("No se crearon aristas.")
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(attr_list, dtype=torch.float)
    
    logging.info(f"Aristas creadas. edge_index shape: {edge_index.shape}, edge_attr shape: {edge_attr.shape}")
    
    return edge_index, edge_attr


def _assemble_and_save_graph(
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    player_map: Dict[str, int],
    processed_dir: Path,
    seasons: List[int]
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
    """
    Orquesta la construcción completa del grafo de jugadores.
    """
    logging.info("Iniciando la construcción del grafo de jugadores...")

    player_stats_df, player_map = _load_player_stats(interim_dir, seasons)
    if not player_map:
        logging.error("No se pudieron cargar las estadísticas de jugadores. Abortando.")
        return

    node_features = _create_nodes_and_features(player_stats_df)
    if node_features.shape[0] == 0:
        logging.error("No se pudieron crear los features de los nodos. Abortando.")
        return

    edge_index, edge_attr = _create_edges(player_stats_df, player_map)

    _assemble_and_save_graph(
        node_features=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        player_map=player_map,
        processed_dir=processed_dir,
        seasons=seasons
    )
    logging.info("Construcción del grafo finalizada.")


# --- Bloque para ejecutar el script ---
if __name__ == "__main__":
    
    # --- 1. Definición de Rutas ---
    SCRIPT_PATH = Path(__file__).resolve()
    BASE_DIR = SCRIPT_PATH.parent.parent.parent.parent
    
    DATA_DIR = BASE_DIR / "data" 
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
    REPORTS_DIR = BASE_DIR / "reports"

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Directorio Raíz (BASE_DIR): {BASE_DIR}")
    logging.info(f"Directorio de Datos (DATA_DIR): {DATA_DIR}")
    logging.info(f"Directorio Procesado (PROCESSED_DIR): {PROCESSED_DATA_DIR}")
    logging.info(f"Directorio de Reportes (REPORTS_DIR): {REPORTS_DIR}")
    
    seasons_to_build = [2023]
    
    # --- 2. Construcción del Grafo ---
    build_player_graph(
        interim_dir=DATA_DIR,
        processed_dir=PROCESSED_DATA_DIR,
        seasons=seasons_to_build
    )
    
    # --- 3. Carga, Estadísticas y Visualización ---
    logging.info("--- Verificación y Visualización del Grafo ---")
    
    try:
        season_str = f"{seasons_to_build[0]}_{seasons_to_build[-1]}"
        graph_path = PROCESSED_DATA_DIR / "graphs" / f"player_graph_{season_str}.pt"
        
        loaded_graph = torch.load(graph_path)
        logging.info(f"Grafo cargado exitosamente: {graph_path}")
        
        # --- Estadísticas (Usando print para que destaquen) ---
        print("\n--- Estadísticas del Grafo Generado ---")
        print(f"Total de nodos (vértices): {loaded_graph.num_nodes}")
        print(f"Total de aristas (bidireccional): {loaded_graph.num_edges}")
        print(f"Shape de features de nodo (x): {loaded_graph.x.shape}")
        print(f"Shape de features de arista (edge_attr): {loaded_graph.edge_attr.shape}")
        print(f"El grafo tiene {loaded_graph.num_edges / 2:.0f} conexiones únicas.")

        # --- Visualización ---
        logging.info("--- Generando visualización del grafo (puede tardar)... ---")
        logging.warning("ADVERTENCIA: Si el grafo es muy denso (>500 nodos), la imagen será un 'hairball' (ovillo).")
        logging.info("La imagen se guardará como un archivo PNG en el directorio 'reports/'.")
        
        g_nx = None
        if _TO_NETWORKX_AVAILABLE:
            try:
                # Intenta la conversión directa
                g_nx = to_networkx(loaded_graph, to_undirected=True)
                logging.info("Grafo convertido a NetworkX usando 'to_networkx'.")
            except Exception as e:
                logging.warning(f"Fallo al convertir con 'to_networkx': {e}. Intentando construcción manual.")
                _TO_NETWORKX_AVAILABLE = False # Desactivar para futuros intentos en la misma ejecución
        
        if not _TO_NETWORKX_AVAILABLE: # Si falló o no estaba disponible
            # Construcción manual del grafo NetworkX
            g_nx = nx.Graph()
            g_nx.add_nodes_from(range(loaded_graph.num_nodes))
            # Añadir aristas. edge_index es de forma [2, num_edges]
            # Convertimos a una lista de tuplas (source, target)
            edges_to_add = loaded_graph.edge_index.t().tolist()
            g_nx.add_edges_from(edges_to_add)
            logging.info("Grafo NetworkX construido manualmente.")

        plt.figure(figsize=(20, 20))
        
        # Usar spring_layout para que "intente" desenredar los nodos
        # 'k' ajusta la distancia, 'iterations' ajusta el tiempo de cálculo
        # 'seed' asegura que la posición sea la misma cada vez (para comparar)
        pos = nx.spring_layout(g_nx, k=0.15, iterations=50, seed=42)
        
        nx.draw(g_nx, pos=pos, node_size=25, with_labels=False, alpha=0.7, edge_color="#DDDDDD")
        
        output_viz_path = REPORTS_DIR / f"graph_visualization_{season_str}.png"
        plt.savefig(output_viz_path)
        plt.close() # Cierra la figura para liberar memoria
        
        logging.info(f"Visualización del grafo guardada exitosamente en: {output_viz_path}")

    except FileNotFoundError:
        logging.error(f"No se pudo cargar el grafo guardado. ¿Se ejecutó la construcción correctamente?")
    except Exception as e:
        logging.error(f"Ocurrió un error durante la visualización: {e}")