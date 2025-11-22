
import sys
from pathlib import Path
import logging

# Añadir src al path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bielsia.data.graph_construction import build_player_graph

def main():
    logging.basicConfig(level=logging.INFO)
    
    root_dir = Path(__file__).resolve().parents[1]
    interim_dir = root_dir / "data" / "interim"
    processed_dir = root_dir / "data" / "processed"
    
    # Construir solo 2025
    build_player_graph(interim_dir, processed_dir, [2025])

if __name__ == "__main__":
    main()
