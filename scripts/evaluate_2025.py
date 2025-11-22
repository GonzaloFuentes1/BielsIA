
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
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    root_dir = Path(__file__).resolve().parents[1]
    interim_dir = root_dir / "data" / "interim"
    reports_dir = root_dir / "reports"
    figures_dir = reports_dir / "figures"
    tables_dir = reports_dir / "tables"
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Configuración
    config_path = root_dir / "configs" / "transformer_graphormer.yaml"
    config = load_config(str(config_path))
    device = get_device()
    
    # 2. Cargar Datos (2016-2025)
    # Necesitamos hasta 2025 para tener el target de 2024 (que es 2025)
    seasons = list(range(2016, 2026)) 
    logger.info(f"Cargando grafos para temporadas: {seasons}")
    
    graphs = build_temporal_player_graphs(interim_dir, seasons)
    
    # 3. Construir Mapa Global
    global_player_map = _build_global_player_map(graphs)
    num_global_players = len(global_player_map)
    logger.info(f"Total jugadores globales (2016-2025): {num_global_players}")

    # 4. Cargar Modelo
    # Usamos la configuración del entrenamiento, pero actualizamos num_global_players
    # para acomodar el nuevo rango (el modelo es agnóstico a esto en sus pesos)
    
    # Determinar dimensiones
    first_graph = graphs[0]
    in_channels = first_graph.x.shape[1]
    # El target y tiene dimensión 11 (hardcoded en dataset_builder o inferido)
    # Buscamos un grafo con y válido para saber out_channels
    out_channels = 11
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
        num_global_players=num_global_players, # Nuevo tamaño
        gnn_type=config['model'].get('gnn_type', 'hybrid'),
        num_layers=config['model']['transformer_layers'],
        heads=config['model']['heads'],
        dropout=config['model']['dropout']
    )
    
    # Cargar pesos
    # El config dice save_dir: models, así que está en root/models
    model_path = root_dir / "models" / "transformer_graphormer_best.pth"
    if not model_path.exists():
        # Fallback a models/saved por si acaso
        model_path = root_dir / "models" / "saved" / "transformer_graphormer_best.pth"
        
    if not model_path.exists():
        logger.error(f"No se encontró el modelo en {model_path}")
        return

    logger.info(f"Cargando pesos desde {model_path}")
    # strict=False por si acaso el pos_encoder tiene tamaño diferente (si max_len cambió)
    # Pero pos_encoder se inicializa con tamaño fijo (20) en __init__, así que debería coincidir.
    # Sin embargo, num_global_players cambió, pero no afecta pesos.
    try:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    except Exception as e:
        logger.warning(f"Carga estricta falló ({e}). Intentando strict=False...")
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    model.to(device)
    model.eval()
    
    # Mover grafos a dispositivo
    graphs = [g.to(device) for g in graphs]

    # 5. Inferencia
    logger.info("Ejecutando inferencia...")
    with torch.no_grad():
        predictions, active_masks = model(graphs, global_player_map)
    
    # 6. Evaluación 2025
    # El grafo de 2024 (índice -2 en la lista 2016-2025) contiene en su 'y' los valores de 2025.
    # La predicción correspondiente es predictions[-2].
    
    target_idx = seasons.index(2024)
    pred_2025 = predictions[target_idx] # (Num_Global_Players, Out)
    mask_2025 = active_masks[target_idx] # (Num_Global_Players)
    
    # El target real está en graphs[target_idx].y
    # Pero graphs[target_idx].y está alineado a los nodos LOCALES de 2024.
    # Necesitamos mapearlo a GLOBAL para comparar con pred_2025.
    
    graph_2024 = graphs[target_idx]
    local_pids = graph_2024.player_ids
    
    # Construir tensores alineados para comparación
    y_true_global = torch.zeros_like(pred_2025)
    has_target_mask = torch.zeros_like(mask_2025) # Jugadores que estaban en 2024 Y tienen target (están en 2025)
    
    # Mapear local 2024 -> global
    # graph_2024.y tiene shape (Num_Local_Nodes, Out). Los primeros N son jugadores.
    local_y = graph_2024.y
    
    for loc_idx, pid in enumerate(local_pids):
        if pid in global_player_map:
            glob_idx = global_player_map[pid]
            # Verificar si el target no es todo ceros (indicando que el jugador existe en 2025)
            # _compute_y_vector_from_next_season pone ceros si no está.
            # Pero un jugador podría tener stats 0 reales.
            # Mejor: dataset_builder pone ceros si no está.
            # Asumiremos que si sum(abs(y)) > 0 existe, o confiamos en la intersección.
            # dataset_builder: "Jugador no presente en la temporada siguiente: dejamos Y en ceros"
            # Y también: "g_t.train_mask" marca los jugadores válidos (los primeros N).
            # Pero no marca "presente en siguiente".
            
            # Vamos a cargar DF 2025 para saber quién está realmente.
            pass

    # Cargar DF 2025 para filtrar jugadores válidos y obtener equipos
    df_2025, _ = _load_player_stats_for_season(interim_dir, 2025)
    players_2025_set = set(df_2025['player'].unique())
    
    # Mapa de equipos 2025
    player_team_2025 = df_2025.set_index('player')['team'].to_dict()
    
    valid_indices = []
    teams_list = []
    names_list = []
    
    y_preds = []
    y_trues = []
    
    for loc_idx, pid in enumerate(local_pids):
        if pid in global_player_map and pid in players_2025_set:
            glob_idx = global_player_map[pid]
            
            # Guardar datos para métricas
            y_preds.append(pred_2025[glob_idx].cpu().numpy())
            y_trues.append(local_y[loc_idx].cpu().numpy())
            
            names_list.append(pid)
            teams_list.append(player_team_2025.get(pid, "Unknown"))

    y_preds = np.array(y_preds)
    y_trues = np.array(y_trues)
    
    logger.info(f"Jugadores evaluados (presentes en 2024 y 2025): {len(y_preds)}")
    
    # 7. Calcular Métricas Globales
    mse = np.mean((y_preds - y_trues)**2)
    mae = np.mean(np.abs(y_preds - y_trues))
    
    # Calcular correlación (Spearman) para ver si el ranking es bueno pese a la escala
    from scipy.stats import spearmanr
    corrs = []
    for i in range(y_preds.shape[1]):
        # Evitar NaNs si la desviación es 0
        if np.std(y_preds[:, i]) > 0 and np.std(y_trues[:, i]) > 0:
            corr, _ = spearmanr(y_preds[:, i], y_trues[:, i])
            corrs.append(corr)
        else:
            corrs.append(0.0)
    avg_corr = np.mean(corrs)

    logger.info(f"Global MSE 2025: {mse:.4f}")
    logger.info(f"Global MAE 2025: {mae:.4f}")
    logger.info(f"Global Spearman Corr 2025: {avg_corr:.4f}")
    
    # Imprimir ejemplos de escala
    logger.info(f"Ejemplo Pred (Jugador 0): {y_preds[0]}")
    logger.info(f"Ejemplo True (Jugador 0): {y_trues[0]}")
    
    # 8. Análisis por Equipo
    df_results = pd.DataFrame({
        'player': names_list,
        'team': teams_list,
        'mse': np.mean((y_preds - y_trues)**2, axis=1),
        'mae': np.mean(np.abs(y_preds - y_trues), axis=1)
    })
    
    # Agrupar por equipo
    team_metrics = df_results.groupby('team').agg({
        'mse': 'mean',
        'mae': 'mean',
        'player': 'count'
    }).sort_values('mae')
    
    print("\n--- Métricas por Equipo (Top 5 Mejor MAE) ---")
    print(team_metrics.head(5))
    print("\n--- Métricas por Equipo (Top 5 Peor MAE) ---")
    print(team_metrics.tail(5))
    
    # Guardar tabla
    team_metrics.to_csv(tables_dir / "team_metrics_2025.csv")
    
    # Seleccionar equipos para detalle
    # Popular: Manchester City (si existe)
    # No popular: Uno con pocos jugadores o bajo rendimiento, o al azar del bottom.
    
    popular_team = "Manchester City"
    # Buscar un equipo "no popular" (ej. recién ascendido o tabla baja)
    # Tomamos el que tenga mayor error MAE como "interesante" o simplemente uno pequeño.
    # Usaremos el último de la lista ordenada por MAE (peor error) o uno con pocos jugadores.
    unpopular_team = team_metrics.index[-1] 
    
    logger.info(f"Comparando: {popular_team} vs {unpopular_team}")
    
    teams_to_plot = [popular_team, unpopular_team]
    
    # Gráfico de error por jugador en estos equipos
    df_plot = df_results[df_results['team'].isin(teams_to_plot)].copy()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x='player', y='mae', hue='team')
    plt.xticks(rotation=90)
    plt.title(f"Error MAE por Jugador: {popular_team} vs {unpopular_team} (2025)")
    plt.tight_layout()
    plt.savefig(figures_dir / "error_comparison_teams_2025.png")
    logger.info(f"Gráfico guardado en {figures_dir / 'error_comparison_teams_2025.png'}")

    # Gráfico Global de Error por Equipo
    plt.figure(figsize=(15, 8))
    sns.barplot(data=team_metrics.reset_index(), x='team', y='mae')
    plt.xticks(rotation=90)
    plt.title("Error MAE Promedio por Equipo (2025)")
    plt.tight_layout()
    plt.savefig(figures_dir / "global_team_error_2025.png")
    
    # Guardar predicciones completas
    # Crear DF con columnas de dimensiones
    dim_names = [
        "Goals", "xG", "npxG", 
        "Assists", "xAG", "xG+xAG", 
        "Tackles", "Interceptions", "Blocks", "Clearances", 
        "Minutes"
    ]
    
    # Expandir y_preds y y_trues
    pred_df = pd.DataFrame(y_preds, columns=[f"Pred_{c}" for c in dim_names])
    true_df = pd.DataFrame(y_trues, columns=[f"True_{c}" for c in dim_names])
    
    full_results = pd.concat([df_results[['player', 'team']], pred_df, true_df], axis=1)
    full_results.to_csv(tables_dir / "predictions_2025_full.csv", index=False)
    logger.info(f"Predicciones completas guardadas en {tables_dir / 'predictions_2025_full.csv'}")

if __name__ == "__main__":
    main()
