import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bielsia.data.dataset_builder import build_temporal_player_graphs
from bielsia.models.temporal import create_gnn_encoder
from torch_geometric.data import Data

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaselineGNN(nn.Module):
    """
    Modelo Baseline: GNN Encoder -> MLP Head.
    Predice t+1 usando solo información de t.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, gnn_type="graphormer", heads=4, dropout=0.2):
        super().__init__()
        
        # 1. Encoder GNN (Reutilizamos la factoría del proyecto para consistencia)
        self.encoder = create_gnn_encoder(
            type=gnn_type,
            in_channels=in_channels,
            hidden=hidden_channels,
            out=hidden_channels, # El encoder saca hidden dim
            heads=heads,
            dropout=dropout
        )
        
        # 2. MLP Head para predicción
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index):
        # x: Features en tiempo t
        embedding = self.encoder(x, edge_index)
        # Predicción para tiempo t+1
        out = self.head(embedding)
        return out

def prepare_pairs(graphs: List[Data], seasons: List[int]) -> List[Dict]:
    """
    Construye pares (Input_Graph_t, Target_Tensor_t+1, Mask).
    Solo incluye jugadores presentes en ambos años.
    """
    pairs = []
    
    # Iteramos hasta el penúltimo para tener t y t+1
    for i in range(len(graphs) - 1):
        season_t = seasons[i]
        season_next = seasons[i+1]
        
        g_t = graphs[i]
        g_next = graphs[i+1]
        
        if not hasattr(g_t, 'player_ids') or not hasattr(g_next, 'player_ids'):
            print(f"Skipping {season_t}->{season_next}: Missing player_ids")
            continue
            
        # Mapear IDs de t+1 a sus índices
        next_player_map = {pid: idx for idx, pid in enumerate(g_next.player_ids)}
        
        # Construir targets alineados con g_t
        num_nodes_t = g_t.x.shape[0]
        
        # Usamos g_next.y si existe (targets reales), sino g_next.x
        if hasattr(g_next, 'y') and g_next.y is not None:
            target_source = g_next.y
            target_dim = g_next.y.shape[1]
        else:
            # Fallback: predecir las features del siguiente grafo
            target_source = g_next.x
            target_dim = g_next.x.shape[1]

        targets = torch.zeros((num_nodes_t, target_dim))
        mask = torch.zeros(num_nodes_t, dtype=torch.bool)
        
        count_matched = 0
        for idx_t, pid in enumerate(g_t.player_ids):
            if pid in next_player_map:
                idx_next = next_player_map[pid]
                targets[idx_t] = target_source[idx_next]
                mask[idx_t] = True
                count_matched += 1
        
        print(f"Par {season_t} -> {season_next}: {count_matched} jugadores coincidentes.")
        
        pairs.append({
            'season_input': season_t,
            'season_target': season_next,
            'graph_input': g_t,
            'targets': targets,
            'mask': mask
        })
        
    return pairs

def train_baseline(config_name, gnn_type="graphormer"):
    root_dir = Path(__file__).resolve().parents[1]
    interim_dir = root_dir / "data" / "interim"
    models_dir = root_dir / "models" / "baseline"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Definir Split Temporal Estricto
    # Train: 2016->17, 17->18, 18->19, 19->20
    train_seasons_start = [2016, 2017, 2018, 2019]
    # Val: 2020->21
    val_seasons_start = [2020]
    # Test: 2021->22, 22->23, 23->24
    test_seasons_start = [2021, 2022, 2023]
    
    # Cargamos hasta 2024 para poder hacer el par 2023->2024
    all_seasons = list(range(2016, 2025)) 
    
    print(f"Cargando grafos {all_seasons}...")
    graphs = build_temporal_player_graphs(interim_dir, all_seasons)
    
    # Preparar todos los pares posibles
    all_pairs = prepare_pairs(graphs, all_seasons)
    
    train_pairs = [p for p in all_pairs if p['season_input'] in train_seasons_start]
    val_pairs = [p for p in all_pairs if p['season_input'] in val_seasons_start]
    test_pairs = [p for p in all_pairs if p['season_input'] in test_seasons_start]
    
    print(f"Split: {len(train_pairs)} Train, {len(val_pairs)} Val, {len(test_pairs)} Test pairs")
    
    if not test_pairs:
        print("Error: No se generaron pares de test. Verifica si existen datos de 2024.")
        return

    # Configuración Modelo
    device = get_device()
    first_g = graphs[0]
    in_channels = first_g.x.shape[1]
    out_channels = all_pairs[0]['targets'].shape[1]
    
    model = BaselineGNN(
        in_channels=in_channels,
        hidden_channels=256,
        out_channels=out_channels,
        gnn_type=gnn_type,
        heads=4,
        dropout=0.2
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # Loop Entrenamiento
    best_val_loss = float('inf')
    patience = 500
    counter = 0
    
    print(f"Iniciando entrenamiento Baseline ({gnn_type})...")
    
    for epoch in range(3000):
        model.train()
        total_loss = 0
        steps = 0
        
        for pair in train_pairs:
            g = pair['graph_input'].to(device)
            y = pair['targets'].to(device)
            mask = pair['mask'].to(device)
            
            if mask.sum() == 0: continue
            
            optimizer.zero_grad()
            out = model(g.x, g.edge_index)
            
            loss = criterion(out[mask], y[mask])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
        avg_train_loss = total_loss / steps if steps > 0 else 0
        
        # Validación
        model.eval()
        val_loss = 0
        val_steps = 0
        with torch.no_grad():
            for pair in val_pairs:
                g = pair['graph_input'].to(device)
                y = pair['targets'].to(device)
                mask = pair['mask'].to(device)
                
                if mask.sum() == 0: continue
                
                out = model(g.x, g.edge_index)
                loss = criterion(out[mask], y[mask])
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), models_dir / f"baseline_{gnn_type}_best.pth")
            counter = 0
        else:
            counter += 1
            
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f}")
            
        if counter >= patience:
            print("Early stopping.")
            break
            
    # Evaluación Final en Test
    print("\nEvaluando en Test (incluyendo 2023->2024)...")
    model.load_state_dict(torch.load(models_dir / f"baseline_{gnn_type}_best.pth"))
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for pair in test_pairs:
            g = pair['graph_input'].to(device)
            y = pair['targets'].to(device)
            mask = pair['mask'].to(device)
            
            out = model(g.x, g.edge_index)
            
            # Calcular métricas solo en masked
            pred = out[mask].cpu().numpy()
            true = y[mask].cpu().numpy()
            
            mae = np.mean(np.abs(pred - true))
            mse = np.mean((pred - true)**2)
            
            print(f"Test Pair {pair['season_input']}->{pair['season_target']}: MAE={mae:.4f}, MSE={mse:.4f}")
            
            results.append({
                'input_year': pair['season_input'],
                'target_year': pair['season_target'],
                'mae': mae,
                'mse': mse
            })
            
            # Guardar predicciones detalladas para comparación
            valid_indices = torch.where(mask)[0].cpu().numpy()
            player_ids = np.array(pair['graph_input'].player_ids)[valid_indices]
            
            df_pred = pd.DataFrame(pred, columns=[f"Pred_{i}" for i in range(pred.shape[1])])
            df_true = pd.DataFrame(true, columns=[f"True_{i}" for i in range(true.shape[1])])
            df_meta = pd.DataFrame({'player': player_ids, 'year_target': pair['season_target']})
            
            full_df = pd.concat([df_meta, df_pred, df_true], axis=1)
            full_df.to_csv(root_dir / "reports" / "tables" / f"baseline_{gnn_type}_preds_{pair['season_target']}.csv", index=False)

    # Guardar resumen
    pd.DataFrame(results).to_csv(root_dir / "reports" / "tables" / f"baseline_{gnn_type}_summary.csv", index=False)
    print("Baseline finalizado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn_type", type=str, default="graphormer", help="gcn, gat, graphormer")
    args = parser.parse_args()
    
    train_baseline("baseline", args.gnn_type)
