"""
Módulo para orquestar la construcción del dataset completo para BielsIA.

Este script es el punto de entrada para el pipeline de procesamiento de datos,
el cual transforma los datos crudos/intermedios en el grafo procesado final
que será consumido por los modelos.
"""
from pathlib import Path
import logging

from .graph_construction import build_player_graph

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
        build_player_graph(
            interim_dir=interim_dir,
            processed_dir=processed_dir,
            seasons=seasons
        )
        logging.info("Construcción del dataset completada exitosamente.")
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
