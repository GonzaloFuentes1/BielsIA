"""
Módulo para la construcción de los grafos de jugadores.

Este script contiene la lógica principal para:
1. Cargar los datos intermedios (CSV de stats, lineups, eventos).
2. Parsear y limpiar los datos.
3. Construir un grafo de PyTorch Geometric con nodos (jugadores) y aristas (interacciones).
4. Guardar el objeto de grafo procesado.
"""
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _load_player_stats(interim_dir: Path, seasons: List[int]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Carga, concatena y procesa las estadísticas de jugadores de varias temporadas."""
    logging.info(f"Cargando estadísticas de jugadores para las temporadas: {seasons}")
    # (Implementación pendiente)
    # - Leer y concatenar los archivos player_stats_*.csv
    # - Limpiar datos, eliminar duplicados.
    # - Crear un mapeo único de player_id -> node_idx
    # - Seleccionar las columnas que serán los features de los nodos.
    # - Devolver el DataFrame de features y el mapeo de jugadores.
    print("TODO: Implementar _load_player_stats")
    return pd.DataFrame(), {}


def _create_nodes_and_features(player_stats_df: pd.DataFrame) -> torch.Tensor:
    """Convierte el DataFrame de stats de jugadores en el tensor de features de nodos."""
    logging.info("Creando tensor de features para los nodos (jugadores)...")
    # (Implementación pendiente)
    # - Normalizar las características (ej: StandardScaler).
    # - Convertir el DataFrame de pandas a un tensor de PyTorch.
    print("TODO: Implementar _create_nodes_and_features")
    return torch.empty(0)


def _create_edges(player_map: Dict[str, int], interim_dir: Path, seasons: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Crea las aristas del grafo y sus atributos a partir de los datos de partidos."""
    logging.info("Creando aristas a partir de los datos de partidos...")
    # (Implementación pendiente)
    # - Iterar sobre cada partido de cada temporada.
    # - Leer lineup, events, shot_events.
    # - Crear aristas de equipo (co-ocurrencia).
    # - Crear aristas de colaboración (asistencias, creación de jugadas).
    # - Crear aristas adversariales.
    # - Asignar atributos a cada arista (ej: tipo de interacción, peso).
    # - Devolver edge_index y edge_attr.
    print("TODO: Implementar _create_edges")
    return torch.empty((2, 0), dtype=torch.long), torch.empty(0)


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
    
    # Añadir metadatos útiles al objeto
    graph.player_map = player_map
    graph.seasons = seasons

    # Validar el grafo
    if not graph.validate():
        raise ValueError("La validación del grafo de PyG falló.")

    # Guardar
    output_path = processed_dir / "graphs" / f"player_graph_{seasons[0]}_{seasons[-1]}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, output_path)
    logging.info(f"Grafo guardado exitosamente en: {output_path}")


def build_player_graph(interim_dir: Path, processed_dir: Path, seasons: List[int]):
    """
    Orquesta la construcción completa del grafo de jugadores.

    Args:
        interim_dir (Path): Directorio con datos intermedios (CSV).
        processed_dir (Path): Directorio para guardar el grafo final.
        seasons (List[int]): Lista de temporadas a incluir.
    """
    logging.info("Iniciando la construcción del grafo de jugadores...")

    # Paso 1: Cargar datos de jugadores y crear mapeo a nodos
    player_stats_df, player_map = _load_player_stats(interim_dir, seasons)
    if not player_map:
        logging.error("No se pudieron cargar las estadísticas de jugadores. Abortando.")
        return

    # Paso 2: Crear features de nodos
    node_features = _create_nodes_and_features(player_stats_df)

    # Paso 3: Crear aristas y sus features
    edge_index, edge_attr = _create_edges(player_map, interim_dir, seasons)

    # Paso 4: Ensamblar y guardar el grafo
    _assemble_and_save_graph(
        node_features=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        player_map=player_map,
        processed_dir=processed_dir,
        seasons=seasons
    )
    logging.info("Construcción del grafo finalizada.")

