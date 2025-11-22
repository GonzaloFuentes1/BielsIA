"""Gestión centralizada de rutas del proyecto."""

from pathlib import Path

# Directorio raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
