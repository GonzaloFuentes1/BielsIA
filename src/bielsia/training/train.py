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
from collections import Counter
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
    
    # Estratificación por cantidad de temporadas jugadas
    player_counts = {pid: 0 for pid in all_player_ids}
    for g in graphs:
        if hasattr(g, 'player_ids'):
            for pid in g.player_ids:
                if pid in player_counts:
                    player_counts[pid] += 1
    
    stratify_labels = [player_counts[pid] for pid in all_player_ids]
    
    # Selección de Estrategia de Split
    split_strategy = config['training'].get('split_strategy', 'random') # random | temporal
    
    if split_strategy == 'temporal':
        print("Usando Split Temporal: Train en t < T-1, Val en T-1 (último año con target). Todos los jugadores en ambos sets.")
        # En split temporal, usamos todos los jugadores para train y test, 
        # pero la separación se hace en el bucle de épocas (por índice de tiempo).
        train_ids = all_player_ids
        test_ids = all_player_ids
    else:
        # Verificar si es posible estratificar
        counts = Counter(stratify_labels)
        if any(c < 2 for c in counts.values()):
            print("Advertencia: Clases de estratificación muy pequeñas. Usando random split.")
            train_ids, test_ids = train_test_split(all_player_ids, test_size=0.2, random_state=123)
        else:
            print("Usando split estratificado por temporadas jugadas.")
            train_ids, test_ids = train_test_split(all_player_ids, test_size=0.2, random_state=123, stratify=stratify_labels)
    
    # Guardar split para evaluación consistente
    save_dir = root / config['training']['save_dir']
    save_dir.mkdir(exist_ok=True, parents=True)
    save_name_base = Path(config['training']['save_name']).stem
    split_save_path = save_dir / f"{save_name_base}_split.json"
    
    import json
    with open(split_save_path, 'w') as f:
        json.dump({'train': train_ids, 'test': test_ids}, f)
    print(f"Split guardado en: {split_save_path}")
    
    # Crear máscaras globales
    train_mask_global = torch.zeros(num_global_players, dtype=torch.bool, device=device)
    test_mask_global = torch.zeros(num_global_players, dtype=torch.bool, device=device)
    
    for pid in train_ids:
        train_mask_global[global_player_map[pid]] = True
    for pid in test_ids:
        test_mask_global[global_player_map[pid]] = True
        
    print(f"Total jugadores: {num_global_players}. Train: {len(train_ids)}, Test: {len(test_ids)}")

    # --- DIAGNÓSTICO DE DISTRIBUCIÓN ---
    print("\n--- DIAGNÓSTICO DE DISTRIBUCIÓN TRAIN vs TEST ---")
    train_targets_list = []
    test_targets_list = []
    
    for g in graphs:
        if not hasattr(g, 'y') or g.y is None: continue
        if not hasattr(g, 'player_ids'): continue
        
        # Extraer targets
        local_pids = g.player_ids
        for loc_idx, pid in enumerate(local_pids):
            if pid in global_player_map:
                # Verificar si es un target válido (train_mask del grafo)
                if hasattr(g, 'train_mask') and not g.train_mask[loc_idx]:
                    continue
                    
                target_val = g.y[loc_idx]
                if pid in train_ids:
                    train_targets_list.append(target_val)
                elif pid in test_ids:
                    test_targets_list.append(target_val)
                    
    if train_targets_list and test_targets_list:
        train_t = torch.stack(train_targets_list)
        test_t = torch.stack(test_targets_list)
        
        print(f"Train Targets: Mean={train_t.mean():.4f}, Std={train_t.std():.4f}")
        print(f"Test Targets:  Mean={test_t.mean():.4f}, Std={test_t.std():.4f}")
        
        mse_baseline_train = (train_t ** 2).mean().item()
        mse_baseline_test = (test_t ** 2).mean().item()
        print(f"MSE Baseline (Prediciendo 0): Train={mse_baseline_train:.4f}, Test={mse_baseline_test:.4f}")
    else:
        print("No se pudieron extraer targets para diagnóstico.")
    print("---------------------------------------------------\n")

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
        # Configuración de cabezas (con fallback para compatibilidad)
        transformer_heads = config['model'].get('transformer_heads', config['model'].get('heads', 4))
        gnn_heads = config['model'].get('gnn_heads', 4)
        
        model = TemporalTransformerGNN(
            in_channels=in_channels,
            gnn_hidden=config['model']['gnn_hidden'],
            gnn_out=config['model']['gnn_out'],
            transformer_hidden=config['model']['transformer_hidden'],
            out_channels=out_channels,
            num_global_players=num_global_players,
            gnn_type=gnn_type,
            num_layers=config['model']['transformer_layers'],
            transformer_heads=transformer_heads,
            gnn_heads=gnn_heads,
            dropout=config['model']['dropout']
        )
    else:
        raise ValueError(f"Tipo de modelo desconocido: {model_type}")

    model = model.to(device)
    
    # Optimizador con Weight Decay (L2 Regularization)
    weight_decay = config['training'].get('weight_decay', 0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'], weight_decay=weight_decay)
    
    # Scheduler para reducir LR si se estanca
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config['training'].get('scheduler_patience', 50)
    )
    
    # Usar HuberLoss para ser más robusto a outliers (común en fútbol)
    criterion = nn.HuberLoss(delta=1.0)

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
    
    # Identificar el último índice con target válido para validación temporal
    # Buscamos el último grafo que tenga train_mask con al menos un valor True
    valid_indices_with_target = []
    for i, g in enumerate(graphs):
        if hasattr(g, 'y') and g.y is not None and g.y.numel() > 0:
            if hasattr(g, 'train_mask') and g.train_mask.sum() > 0:
                valid_indices_with_target.append(i)
    
    last_target_idx = valid_indices_with_target[-1] if valid_indices_with_target else -1
    
    if last_target_idx == 0 and split_strategy == 'temporal':
        print("ADVERTENCIA CRÍTICA: Solo hay un año con targets válidos. No habrá datos para entrenamiento (solo validación).")
        print("Revisa que 'seasons' incluya suficientes años y que los datos 'interim' existan.")
    
    if split_strategy == 'temporal' and last_target_idx != -1:
        train_years = [graphs[i].season for i in range(last_target_idx)]
        val_year = graphs[last_target_idx].season
        print(f"  [INFO] Estrategia Temporal:")
        print(f"  -> Entrenamiento (Inputs): {train_years}")
        print(f"  -> Validación (Input): {val_year} (Target: {val_year + 1})")

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
                # Lógica de Split
                if split_strategy == 'temporal':
                    # Temporal: Train en t < last_target_idx, Val en t == last_target_idx
                    is_val_step = (t == last_target_idx)
                    
                    if not is_val_step:
                        # Train Step
                        step_loss = criterion(predictions[t][target_mask], current_targets[target_mask])
                        total_train_loss += step_loss
                        valid_train_steps += 1
                    else:
                        # Val Step (se maneja abajo en el bloque de test/val, o aquí si queremos simplificar)
                        # Para consistencia con la estructura actual, lo ignoramos aquí y dejamos que el bloque de test lo capture
                        # PERO el bloque de test actual usa test_mask_global.
                        # En split temporal, test_mask_global incluye a TODOS.
                        # Así que necesitamos una bandera para saber si este paso es train o test.
                        pass

                else:
                    # Random Split (Player-wise)
                    # Intersección de máscaras: (Presente en T) AND (Es Train/Test)
                    step_train_mask = target_mask & train_mask_global
                    
                    if step_train_mask.sum() > 0:
                        step_loss = criterion(predictions[t][step_train_mask], current_targets[step_train_mask])
                        total_train_loss += step_loss
                        valid_train_steps += 1
                
                # El bloque de test original estaba aquí dentro del loop para random split.
                # Lo hemos movido fuera en la versión anterior, pero aquí hay un fragmento residual en mi mente.
                # En el código actual del archivo, el bloque de test está FUERA del loop de train steps.
                # Así que aquí solo acumulamos train loss.

        # Calcular Loss de Test CORRECTAMENTE (fuera del bucle de pasos, con model.eval())
        model.eval()
        with torch.no_grad():
            val_predictions, _ = model(graphs, global_player_map)
            
            total_test_loss = 0
            valid_test_steps = 0
            
            for t in range(len(graphs)):
                graph = graphs[t]
                if not hasattr(graph, 'y') or graph.y is None: continue
                
                # Reconstruir targets (copiar lógica de arriba o encapsular en función)
                current_targets = torch.zeros((num_global_players, out_channels), device=device)
                target_mask = torch.zeros(num_global_players, dtype=torch.bool, device=device)
                
                if hasattr(graph, 'player_ids'):
                    local_pids = graph.player_ids
                    indices_global = []
                    indices_local = []
                    for loc_idx, pid in enumerate(local_pids):
                        if pid in global_player_map:
                            if hasattr(graph, 'train_mask') and not graph.train_mask[loc_idx]: continue
                            indices_global.append(global_player_map[pid])
                            indices_local.append(loc_idx)
                    if indices_local:
                        idx_g = torch.tensor(indices_global, device=device)
                        idx_l = torch.tensor(indices_local, device=device)
                        current_targets[idx_g] = graph.y[idx_l]
                        target_mask[idx_g] = True
                
                if split_strategy == 'temporal':
                    # Temporal: Solo validamos en el último paso
                    if t == last_target_idx:
                        # Validar en TODOS los jugadores presentes (target_mask)
                        if target_mask.sum() > 0:
                            step_loss_test = criterion(val_predictions[t][target_mask], current_targets[target_mask])
                            total_test_loss += step_loss_test
                            valid_test_steps += 1
                else:
                    # Random Split: Validamos en test_mask_global en TODOS los pasos
                    step_test_mask = target_mask & test_mask_global
                    if step_test_mask.sum() > 0:
                        step_loss_test = criterion(val_predictions[t][step_test_mask], current_targets[step_test_mask])
                        total_test_loss += step_loss_test
                        valid_test_steps += 1
            
            test_loss_val = (total_test_loss / valid_test_steps).item() if valid_test_steps > 0 else 0.0

        # Volver a modo train
        model.train()

        if valid_train_steps > 0:
            final_train_loss = total_train_loss / valid_train_steps
            final_train_loss.backward()
            
            # Gradient Clipping para evitar explosión de gradientes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Calcular métricas
            train_loss_val = final_train_loss.item()
            # test_loss_val YA CALCULADO ARRIBA
            
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
            
            # Step del Scheduler
            scheduler.step(test_loss_val)

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
    
    # Cargar Scaler para denormalizar
    scaler_path = processed_path / "y_scaler_params.pt"
    mean_y, std_y = None, None
    if scaler_path.exists():
        scaler_data = torch.load(scaler_path)
        mean_y = scaler_data['mean'].to(device)
        std_y = scaler_data['std'].to(device)
        print("Scaler cargado. Se calcularán métricas en escala original también.")

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
            if split_strategy == 'temporal':
                # En temporal, solo evaluamos el último paso
                if t == last_target_idx:
                    step_test_mask = target_mask # Todos los jugadores
                else:
                    step_test_mask = torch.zeros_like(target_mask)
            else:
                step_test_mask = target_mask & test_mask_global
            
            if step_test_mask.sum() > 0:
                all_preds.append(predictions[t][step_test_mask])
                all_targets.append(current_targets[step_test_mask])

    if all_preds:
        all_preds_tensor = torch.cat(all_preds, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)
        
        # Métricas Normalizadas (Lo que ve el Loss)
        mse_per_dim = torch.mean((all_preds_tensor - all_targets_tensor) ** 2, dim=0)
        mae_per_dim = torch.mean(torch.abs(all_preds_tensor - all_targets_tensor), dim=0)
        
        metrics = {
            "model_name": config['experiment']['name'],
            "test_mse_norm": torch.mean(mse_per_dim).item(),
            "test_mae_norm": torch.mean(mae_per_dim).item(),
            "mae_per_dim_norm": mae_per_dim.tolist()
        }
        
        # Métricas Denormalizadas (Interpretables)
        if mean_y is not None and std_y is not None:
            # Denormalize: y_orig = y_norm * std + mean
            preds_orig = all_preds_tensor * std_y + mean_y
            targets_orig = all_targets_tensor * std_y + mean_y
            
            mae_per_dim_orig = torch.mean(torch.abs(preds_orig - targets_orig), dim=0)
            metrics["mae_per_dim_orig"] = mae_per_dim_orig.tolist()
            
            print(f"Test MAE (Norm): {metrics['test_mae_norm']:.6f}")
            print(f"Test MAE (Orig - Goals/90): {mae_per_dim_orig[0]:.4f}") # Asumiendo Goals es idx 0
        
        metrics_path = save_dir / f"{save_name_base}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        print(f"Métricas detalladas guardadas en: {metrics_path}")
    else:
        print("Advertencia: No se encontraron muestras de test para evaluación.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelo BielsIA")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Ruta al archivo de configuración")
    args = parser.parse_args()
    
    train(args.config)
