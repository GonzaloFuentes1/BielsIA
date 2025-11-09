"""
Punto de entrada principal para la interfaz de línea de comandos (CLI) de BielsIA.

Permite orquestar las diferentes etapas del pipeline desde la terminal.
"""
import argparse

def main():
    """Función principal de la CLI."""
    parser = argparse.ArgumentParser(description="BielsIA CLI")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["fetch-data", "make-dataset", "train-gnn", "backtesting", "demo"],
        help="Modo de ejecución del pipeline."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="",
        help="Ruta al archivo de configuración YAML."
    )

    args = parser.parse_args()

    print(f"Ejecutando en modo: {args.mode}")
    if args.config_path:
        print(f"Usando config: {args.config_path}")

    # Aquí iría la lógica para llamar a las funciones correspondientes
    if args.mode == "demo":
        print("Ejecutando pipeline de demostración...")
        # from bielsia.data.graph_construction import build_player_graph
        # build_player_graph(None, None) # Llamada de ejemplo
    else:
        print("Modo no implementado todavía.")

if __name__ == "__main__":
    main()
