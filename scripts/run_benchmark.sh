#!/bin/bash
# Script para ejecutar todos los experimentos de benchmark
export PYTHONPATH="$PWD/src"

echo "--------------------------------------------------"
echo "Generando grafos y visualizaciones (2016-2024)..."
echo "--------------------------------------------------"
python src/bielsia/data/graph_construction.py

configs=(
    "configs/lstm_gat.yaml"
    "configs/lstm_gcn.yaml"
    "configs/lstm_graphormer.yaml"
    "configs/transformer_gat.yaml"
    "configs/transformer_gcn.yaml"
    "configs/transformer_graphormer.yaml"
)

best_mae=10000
best_model=""
best_config=""

for config in "${configs[@]}"; do
    echo "--------------------------------------------------"
    echo "Ejecutando entrenamiento con: $config"
    echo "--------------------------------------------------"
    python src/bielsia/training/train.py --config "$config"
    
    # Extract MAE and Model Path using Python
    read_metrics_cmd="
import yaml, json, sys
from pathlib import Path
try:
    with open('$config') as f: c = yaml.safe_load(f)
    save_dir = Path(c['training']['save_dir'])
    save_name = Path(c['training']['save_name']).stem
    metrics_path = save_dir / f'{save_name}_metrics.json'
    if metrics_path.exists():
        with open(metrics_path) as f: m = json.load(f)
        print(f\"{m['test_mae']} {save_dir}/{save_name}_best.pth\")
    else:
        print('ERROR_NO_FILE')
except Exception as e:
    print(f'ERROR_{e}')
"
    result=$(python -c "$read_metrics_cmd")
    
    if [[ "$result" == ERROR* ]]; then
        echo "Error leyendo métricas para $config: $result"
    else
        mae=$(echo $result | cut -d' ' -f1)
        model_path=$(echo $result | cut -d' ' -f2)
        
        echo "Modelo: $config - MAE: $mae"
        
        # Compare float using python
        is_better=$(python -c "print(1 if $mae < $best_mae else 0)")
        if [ "$is_better" -eq 1 ]; then
            best_mae=$mae
            best_model=$model_path
            best_config=$config
            echo "-> Nuevo mejor modelo!"
        fi
    fi
done

echo "--------------------------------------------------"
echo "Entrenamiento finalizado."
echo "Mejor Modelo: $best_model"
echo "Mejor Config: $best_config"
echo "Mejor MAE (Test 2023): $best_mae"
echo "--------------------------------------------------"

if [ -n "$best_model" ]; then
    echo "Evaluando el mejor modelo en 2024..."
    python scripts/evaluate_year.py --year 2024 --config "$best_config" --model_path "$best_model"
else
    echo "No se encontró ningún modelo válido."
fi

echo "Benchmark finalizado."
