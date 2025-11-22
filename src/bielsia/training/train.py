"""
Script unificado de entrenamiento para modelos temporales (LSTM/Transformer).
Uso: python src/bielsia/training/train.py --config configs/train_config.yaml
"""
import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split

from bielsia.data.dataset_builder import build_temporal_player_graphs
from bielsia.models.temporal import TemporalLSTMGNN, TemporalTransformerGNN
from torch_geometric.data import Data

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def _build_global_player_map(graphs: List[Data]) -> Dict[str, int]:
    """Crea un mapeo único de todos los jugadores en todas las temporadas."""
    unique_players = set()
    for g in graphs:
        if hasattr(g, 'player_ids'):
            unique_players.update(g.player_ids)
    
    sorted_players = sorted(list(unique_players))
    return {pid: i for i, pid in enumerate(sorted_players)}

def get_device(device_config: str) -> torch.device:
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)

def train(config_path: str):
    # 1. Cargar Configuración
    config = load_config(config_path)
    print(f"Cargando configuración desde {config_path}")
    print(f"Experimento: {config['experiment']['name']}")
    
    device = get_device(config['training']['device'])
    print(f"Usando dispositivo: {device}")

    # 2. Cargar Datos
    root = Path(__file__).resolve().parents[3]
    interim_path = root / config['data']['interim_path']
    # Asumimos que processed_path está en data/processed si no se especifica
    processed_path = root / "data" / "processed"
    seasons = config['data']['seasons']
    
    print(f"Cargando grafos para temporadas: {seasons}")
    # Pasamos processed_path para que se calcule y guarde la normalización de targets
    graphs = build_temporal_player_graphs(interim_path, seasons, processed_dir=processed_path)
    
    if not graphs:
        print("Error: No se cargaron grafos.")
        return

    # 3. Construir Mapa Global y Split Train/Test
    global_player_map = _build_global_player_map(graphs)
    num_global_players = len(global_player_map)
    all_player_ids = list(global_player_map.keys())
    
    # Split de jugadores (10% test)
    train_ids, test_ids = train_test_split(all_player_ids, test_size=0.1, random_state=42)
    
    # Crear máscaras globales
    train_mask_global = torch.zeros(num_global_players, dtype=torch.bool, device=device)
    test_mask_global = torch.zeros(num_global_players, dtype=torch.bool, device=device)
    
    for pid in train_ids:
        train_mask_global[global_player_map[pid]] = True
    for pid in test_ids:
        test_mask_global[global_player_map[pid]] = True
        
    print(f"Total jugadores: {num_global_players}. Train: {len(train_ids)}, Test: {len(test_ids)}")

    # 4. Configurar Modelo
    first_graph = graphs[0]
    in_channels = first_graph.x.shape[1] if config['model']['in_channels'] == "auto" else config['model']['in_channels']
    
    # Determinar out_channels
    out_channels = 11 # Default
    if config['model']['out_channels'] == "auto":
        for g in graphs:
            if hasattr(g, 'y') and g.y is not None and g.y.numel() > 0:
                out_channels = g.y.shape[1]
                break
    else:
        out_channels = config['model']['out_channels']
            
    print(f"Dimensiones: In={in_channels}, Out={out_channels}")

    model_type = config['model']['type']
    gnn_type = config['model'].get('gnn_type', 'hybrid')
    print(f"Modelo: {model_type}, Encoder GNN: {gnn_type}")

    if model_type == "lstm":
        model = TemporalLSTMGNN(
            in_channels=in_channels,
            gnn_hidden=config['model']['gnn_hidden'],
            gnn_out=config['model']['gnn_out'],
            lstm_hidden=config['model']['lstm_hidden'],
            out_channels=out_channels,
            num_global_players=num_global_players,
            gnn_type=gnn_type,
            heads=config['model']['heads'],
            dropout=config['model']['dropout']
        )
    elif model_type == "transformer":
        model = TemporalTransformerGNN(
            in_channels=in_channels,
            gnn_hidden=config['model']['gnn_hidden'],
            gnn_out=config['model']['gnn_out'],
            transformer_hidden=config['model']['transformer_hidden'],
            out_channels=out_channels,
            num_global_players=num_global_players,
            gnn_type=gnn_type,
            num_layers=config['model']['transformer_layers'],
            heads=config['model']['heads'],
            dropout=config['model']['dropout']
        )
    else:
        raise ValueError(f"Tipo de modelo desconocido: {model_type}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    criterion = nn.MSELoss()

    # Mover grafos a dispositivo
    graphs = [g.to(device) for g in graphs]
    
    # Verificación de NaNs en datos de entrada
    print("Verificando integridad de datos...")
    for i, g in enumerate(graphs):
        if torch.isnan(g.x).any() or torch.isinf(g.x).any():
            print(f"ADVERTENCIA: NaNs/Infs detectados en features (x) del grafo {i} (Temporada {g.season})")
            g.x = torch.nan_to_num(g.x, nan=0.0, posinf=0.0, neginf=0.0)
            
        if hasattr(g, 'y') and g.y is not None:
            if torch.isnan(g.y).any() or torch.isinf(g.y).any():
                print(f"ADVERTENCIA: NaNs/Infs detectados en targets (y) del grafo {i} (Temporada {g.season})")
                g.y = torch.nan_to_num(g.y, nan=0.0, posinf=0.0, neginf=0.0)

    # 5. Bucle de Entrenamiento
    epochs = config['training']['epochs']
    log_interval = config['training']['log_interval']
    
    print(f"Iniciando entrenamiento ({model_type}) por {epochs} épocas...")
    model.train()
    
    best_test_loss = float('inf')
    best_epoch = -1
    patience = config['training'].get('patience', float('inf'))
    patience_counter = 0
    history = []
    
    save_dir = root / config['training']['save_dir']
    save_dir.mkdir(exist_ok=True, parents=True)
    
    save_name_base = Path(config['training']['save_name']).stem
    best_model_path = save_dir / f"{save_name_base}_best.pth"
    last_model_path = save_dir / f"{save_name_base}_last.pth"
    history_path = save_dir / f"{save_name_base}_history.json"

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward
        predictions, active_masks = model(graphs, global_player_map)
        
        total_train_loss = 0
        total_test_loss = 0
        valid_train_steps = 0
        valid_test_steps = 0
        
        # Calcular Loss
        for t in range(len(graphs)):
            graph = graphs[t]
            if not hasattr(graph, 'y') or graph.y is None or graph.y.numel() == 0:
                continue
                
            # Construir targets globales para este paso
            current_targets = torch.zeros((num_global_players, out_channels), device=device)
            target_mask = torch.zeros(num_global_players, dtype=torch.bool, device=device)
            
            if hasattr(graph, 'player_ids'):
                local_pids = graph.player_ids
                
                indices_global = []
                indices_local = []
                
                for loc_idx, pid in enumerate(local_pids):
                    if pid in global_player_map:
                        # Filtrar por train_mask si existe
                        if hasattr(graph, 'train_mask') and not graph.train_mask[loc_idx]:
                            continue
                        
                        indices_global.append(global_player_map[pid])
                        indices_local.append(loc_idx)
                
                if indices_local:
                    idx_g = torch.tensor(indices_global, device=device)
                    idx_l = torch.tensor(indices_local, device=device)
                    current_targets[idx_g] = graph.y[idx_l]
                    target_mask[idx_g] = True
            
            if target_mask.sum() > 0:
                # Intersección de máscaras: (Presente en T) AND (Es Train/Test)
                step_train_mask = target_mask & train_mask_global
                step_test_mask = target_mask & test_mask_global
                
                if step_train_mask.sum() > 0:
                    step_loss = criterion(predictions[t][step_train_mask], current_targets[step_train_mask])
                    total_train_loss += step_loss
                    valid_train_steps += 1
                
                if step_test_mask.sum() > 0:
                    # Loss de test (sin backward)
                    with torch.no_grad():
                        step_loss_test = criterion(predictions[t][step_test_mask], current_targets[step_test_mask])
                        total_test_loss += step_loss_test
                        valid_test_steps += 1
        
        if valid_train_steps > 0:
            final_train_loss = total_train_loss / valid_train_steps
            final_train_loss.backward()
            
            # Gradient Clipping para evitar explosión de gradientes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Calcular métricas
            train_loss_val = final_train_loss.item()
            test_loss_val = (total_test_loss / valid_test_steps).item() if valid_test_steps > 0 else 0.0
            
            # Guardar historial
            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss_val,
                'test_loss': test_loss_val
            })
            
            # Check Best Model
            if test_loss_val < best_test_loss:
                best_test_loss = test_loss_val
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping activado en época {epoch+1}. Sin mejora por {patience} épocas.")
                break
            
            if (epoch + 1) % log_interval == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss_val:.6f} - Test Loss: {test_loss_val:.6f} (Best: {best_test_loss:.6f} at Ep {best_epoch})")

    # 6. Guardar Modelo Final e Historial
    torch.save(model.state_dict(), last_model_path)
    
    import json
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
        
    print(f"Entrenamiento finalizado.")
    print(f"Mejor modelo guardado en: {best_model_path} (Loss: {best_test_loss:.6f})")
    print(f"Último modelo guardado en: {last_model_path}")
    print(f"Historial guardado en: {history_path}")

    # 7. Evaluación Final Detallada (Mejor Modelo)
    print("\nIniciando evaluación detallada del mejor modelo...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        predictions, _ = model(graphs, global_player_map)
        
        for t in range(len(graphs)):
            graph = graphs[t]
            if not hasattr(graph, 'y') or graph.y is None or graph.y.numel() == 0:
                continue
                
            # Construir targets globales para este paso (Misma lógica que training)
            current_targets = torch.zeros((num_global_players, out_channels), device=device)
            target_mask = torch.zeros(num_global_players, dtype=torch.bool, device=device)
            
            if hasattr(graph, 'player_ids'):
                local_pids = graph.player_ids
                indices_global = []
                indices_local = []
                for loc_idx, pid in enumerate(local_pids):
                    if pid in global_player_map:
                        if hasattr(graph, 'train_mask') and not graph.train_mask[loc_idx]:
                            continue
                        indices_global.append(global_player_map[pid])
                        indices_local.append(loc_idx)
                
                if indices_local:
                    idx_g = torch.tensor(indices_global, device=device)
                    idx_l = torch.tensor(indices_local, device=device)
                    current_targets[idx_g] = graph.y[idx_l]
                    target_mask[idx_g] = True
            
            # Filtrar solo Test Set
            step_test_mask = target_mask & test_mask_global
            
            if step_test_mask.sum() > 0:
                all_preds.append(predictions[t][step_test_mask])
                all_targets.append(current_targets[step_test_mask])

    if all_preds:
        all_preds_tensor = torch.cat(all_preds, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)
        
        # Calcular métricas por dimensión
        mse_per_dim = torch.mean((all_preds_tensor - all_targets_tensor) ** 2, dim=0)
        mae_per_dim = torch.mean(torch.abs(all_preds_tensor - all_targets_tensor), dim=0)
        
        metrics = {
            "model_name": config['experiment']['name'],
            "test_mse": torch.mean(mse_per_dim).item(),
            "test_mae": torch.mean(mae_per_dim).item(),
            "mse_per_dim": mse_per_dim.tolist(),
            "mae_per_dim": mae_per_dim.tolist()
        }
        
        metrics_path = save_dir / f"{save_name_base}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        print(f"Métricas detalladas guardadas en: {metrics_path}")
        print(f"Test MSE: {metrics['test_mse']:.6f}, Test MAE: {metrics['test_mae']:.6f}")
    else:
        print("Advertencia: No se encontraron muestras de test para evaluación.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelo BielsIA")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Ruta al archivo de configuración")
    args = parser.parse_args()
    
    train(args.config)
