"""
Script de chequeo para la etapa de descarga de datos granulares por partido.
"""
import logging
from pathlib import Path
import sys
import argparse

# Añadir el directorio 'src' al path para poder importar 'bielsia'
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / 'src'))

from bielsia.scraping.fbref_scraper import run_smoke_test

def main():
    """Función principal del script de chequeo."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Smoke test para el scraper de datos de partido.")
    parser.add_argument("--league", type=str, default="ENG-Premier League", help="ID de la liga en soccerdata.")
    parser.add_argument("--season", type=int, default=2023, help="Año final de la temporada (ej: 2023 para 22-23).")
    parser.add_argument("--matches", type=int, default=2, help="Número de partidos a descargar.")
    args = parser.parse_args()

    interim_dir = ROOT_DIR / "data" / "interim"

    logging.info("======================================================")
    logging.info("INICIANDO SMOKE TEST DE DESCARGA DE DATOS POR PARTIDO")
    logging.info(f"Liga: {args.league}, Temporada: {args.season}, Partidos: {args.matches}")
    logging.info("======================================================")

    try:
        run_smoke_test(
            interim_dir=interim_dir,
            league=args.league,
            season=args.season,
            num_matches=args.matches
        )
        logging.info("SMOKE TEST completado con éxito.")
        logging.info(f"Verifica los archivos generados en las subcarpetas de: {interim_dir}")
    except Exception as e:
        logging.exception("El SMOKE TEST ha fallado de forma inesperada.")
        sys.exit(1)

if __name__ == "__main__":
    main()
