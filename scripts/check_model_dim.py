
import torch
import sys
from pathlib import Path

def check_model():
    root_dir = Path(__file__).resolve().parents[1]
    model_path = root_dir / "models" / "transformer_graphormer_best.pth"
    
    if not model_path.exists():
        print("Model not found.")
        return

    state_dict = torch.load(model_path, map_location='cpu')
    
    # Buscar la última capa lineal del predictor
    # El nombre suele ser predictor.3.weight o similar (dependiendo de la definición en temporal.py)
    # En TemporalTransformerGNN:
    # self.predictor = nn.Sequential(
    #     nn.Linear(transformer_hidden, transformer_hidden // 2),
    #     nn.ReLU(),
    #     nn.Dropout(dropout),
    #     nn.Linear(transformer_hidden // 2, out_channels)
    # )
    # So keys: predictor.0.weight, predictor.3.weight
    
    for key, val in state_dict.items():
        if "predictor" in key and "weight" in key:
            print(f"{key}: {val.shape}")

if __name__ == "__main__":
    check_model()
