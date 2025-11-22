
import argparse
import torch
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bielsia.models.temporal import TemporalTransformerGNN
from bielsia.data.dataset_builder import build_temporal_player_graphs
from bielsia.training.train import _build_global_player_map, load_config
from bielsia.data.graph_construction import _load_player_stats_for_season

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description="Evaluar modelo BielsIA para un año específico")
    parser.add_argument("--year", type=int, required=True, help="Año objetivo a evaluar (ej. 2024)")
    parser.add_argument("--config", type=str, default=None, help="Ruta al archivo de configuración (opcional)")
    parser.add_argument("--model_path", type=str, default=None, help="Ruta al modelo .pth (opcional)")
    args = parser.parse_args()
    
    target_year = args.year
    input_year = target_year - 1
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    root_dir = Path(__file__).resolve().parents[1]
    interim_dir = root_dir / "data" / "interim"
    reports_dir = root_dir / "reports"
    figures_dir = reports_dir / "figures"
    tables_dir = reports_dir / "tables"
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Evaluando predicciones para el año {target_year} (usando input {input_year})")

    # 1. Configuración
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = root_dir / "configs" / "transformer_graphormer.yaml"
        
    config = load_config(str(config_path))
    device = get_device()
    
    # 2. Cargar Datos
    # Necesitamos cargar hasta target_year para tener el target en el grafo de input_year
    # Si target_year es 2024, cargamos hasta 2024.
    # graphs[input_year].y contendrá los datos de target_year.
    
    # Rango de temporadas: desde 2016 hasta target_year
    seasons = list(range(2016, target_year + 1))
    logger.info(f"Cargando grafos para temporadas: {seasons}")
    
    graphs = build_temporal_player_graphs(interim_dir, seasons)
    
    if not graphs:
        logger.error("No se pudieron cargar los grafos.")
        return

    # 3. Construir Mapa Global
    global_player_map = _build_global_player_map(graphs)
    num_global_players = len(global_player_map)
    logger.info(f"Total jugadores globales: {num_global_players}")

    # 4. Cargar Modelo
    first_graph = graphs[0]
    in_channels = first_graph.x.shape[1]
    out_channels = 11 # Default
    for g in graphs:
        if hasattr(g, 'y') and g.y is not None and g.y.numel() > 0:
            out_channels = g.y.shape[1]
            break
            
    logger.info(f"Dimensiones: In={in_channels}, Out={out_channels}")

    model = TemporalTransformerGNN(
        in_channels=in_channels,
        gnn_hidden=config['model']['gnn_hidden'],
        gnn_out=config['model']['gnn_out'],
        transformer_hidden=config['model']['transformer_hidden'],
        out_channels=out_channels,
        num_global_players=num_global_players,
        gnn_type=config['model'].get('gnn_type', 'hybrid'),
        num_layers=config['model']['transformer_layers'],
        heads=config['model']['heads'],
        dropout=config['model']['dropout']
    )
    
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = root_dir / "models" / "transformer_graphormer_best.pth"
        if not model_path.exists():
            model_path = root_dir / "models" / "saved" / "transformer_graphormer_best.pth"
        
    if not model_path.exists():
        logger.error(f"No se encontró el modelo en {model_path}")
        return

    logger.info(f"Cargando pesos desde {model_path}")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    except Exception as e:
        logger.warning(f"Carga estricta falló ({e}). Intentando strict=False...")
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    model.to(device)
    model.eval()
    
    graphs = [g.to(device) for g in graphs]

    # 5. Inferencia
    logger.info("Ejecutando inferencia...")
    with torch.no_grad():
        predictions, active_masks = model(graphs, global_player_map)
    
    # --- NUEVO: Cargar Scaler y Desnormalizar ---
    processed_dir = root_dir / "data" / "processed"
    scaler_path = processed_dir / "y_scaler_params.pt"
    
    mean_y = None
    std_y = None
    
    if scaler_path.exists():
        logger.info(f"Cargando scaler desde {scaler_path}")
        scaler_params = torch.load(scaler_path, map_location='cpu')
        mean_y = scaler_params['mean'].numpy()
        std_y = scaler_params['std'].numpy()
    else:
        logger.warning("No se encontró scaler. Las predicciones se asumirán en escala original (esto puede ser incorrecto si se entrenó con normalización).")

    # 6. Evaluación
    if input_year not in seasons:
        logger.error(f"El año de input {input_year} no está en las temporadas cargadas.")
        return
        
    input_idx = seasons.index(input_year)
    
    # Predicción hecha en input_year para target_year
    pred_target = predictions[input_idx] 
    
    # Target real (almacenado en el grafo de input_year como 'y')
    graph_input = graphs[input_idx]
    
    if not hasattr(graph_input, 'y') or graph_input.y is None:
        logger.error(f"El grafo de {input_year} no tiene target 'y' (datos de {target_year}).")
        return

    local_y = graph_input.y
    local_pids = graph_input.player_ids
    
    # Cargar DF del target_year para nombres y equipos
    df_target, _ = _load_player_stats_for_season(interim_dir, target_year)
    players_target_set = set(df_target['player'].unique())
    player_team_target = df_target.set_index('player')['team'].to_dict()
    
    y_preds = []
    y_trues = []
    names_list = []
    teams_list = []
    
    for loc_idx, pid in enumerate(local_pids):
        if pid in global_player_map and pid in players_target_set:
            glob_idx = global_player_map[pid]
            
            # Verificar si el target es válido (no todo ceros si esperamos datos)
            # Aunque dataset_builder pone ceros si no está, ya filtramos con players_target_set
            
            y_preds.append(pred_target[glob_idx].cpu().numpy())
            y_trues.append(local_y[loc_idx].cpu().numpy())
            
            names_list.append(pid)
            teams_list.append(player_team_target.get(pid, "Unknown"))

    y_preds = np.array(y_preds)
    y_trues = np.array(y_trues)
    
    # Desnormalizar Predicciones si existe scaler
    if mean_y is not None and std_y is not None:
        logger.info("Desnormalizando predicciones...")
        # Asegurar dimensiones
        if len(mean_y) == y_preds.shape[1]:
            y_preds = y_preds * std_y + mean_y
        else:
            logger.warning(f"Dimensión del scaler ({len(mean_y)}) no coincide con predicciones ({y_preds.shape[1]}). No se desnormaliza.")
    
    logger.info(f"Jugadores evaluados (presentes en {input_year} y {target_year}): {len(y_preds)}")
    
    if len(y_preds) == 0:
        logger.error("No hay jugadores para evaluar.")
        return

    # 7. Métricas
    mse = np.mean((y_preds - y_trues)**2)
    mae = np.mean(np.abs(y_preds - y_trues))
    
    from scipy.stats import spearmanr
    corrs = []
    for i in range(y_preds.shape[1]):
        if np.std(y_preds[:, i]) > 0 and np.std(y_trues[:, i]) > 0:
            corr, _ = spearmanr(y_preds[:, i], y_trues[:, i])
            corrs.append(corr)
        else:
            corrs.append(0.0)
    avg_corr = np.mean(corrs)

    logger.info(f"Global MSE {target_year}: {mse:.4f}")
    logger.info(f"Global MAE {target_year}: {mae:.4f}")
    logger.info(f"Global Spearman Corr {target_year}: {avg_corr:.4f}")
    
    # 8. Guardar Resultados
    df_results = pd.DataFrame({
        'player': names_list,
        'team': teams_list,
        'mse': np.mean((y_preds - y_trues)**2, axis=1),
        'mae': np.mean(np.abs(y_preds - y_trues), axis=1)
    })
    
    team_metrics = df_results.groupby('team').agg({
        'mse': 'mean',
        'mae': 'mean',
        'player': 'count'
    }).sort_values('mae')
    
    print(f"\n--- Métricas por Equipo {target_year} (Top 5 Mejor MAE) ---")
    print(team_metrics.head(5))
    
    csv_path = tables_dir / f"team_metrics_{target_year}.csv"
    team_metrics.to_csv(csv_path)
    logger.info(f"Tabla guardada en {csv_path}")
    
    # Gráfico
    plt.figure(figsize=(15, 8))
    sns.barplot(data=team_metrics.reset_index(), x='team', y='mae')
    plt.xticks(rotation=90)
    plt.title(f"Error MAE Promedio por Equipo ({target_year})")
    plt.tight_layout()
    plt.savefig(figures_dir / f"global_team_error_{target_year}.png")
    
    # Predicciones completas
    dim_names = [
        "Goals", "xG", "npxG", 
        "Assists", "xAG", "xG+xAG", 
        "Tackles", "Interceptions", "Blocks", "Clearances", 
        "Minutes"
    ]
    
    pred_df = pd.DataFrame(y_preds, columns=[f"Pred_{c}" for c in dim_names])
    true_df = pd.DataFrame(y_trues, columns=[f"True_{c}" for c in dim_names])
    
    full_results = pd.concat([df_results[['player', 'team']], pred_df, true_df], axis=1)
    full_results.to_csv(tables_dir / f"predictions_{target_year}_full.csv", index=False)

if __name__ == "__main__":
    main()
