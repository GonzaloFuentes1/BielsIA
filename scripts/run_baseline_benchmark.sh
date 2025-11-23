#!/bin/bash

# Script para ejecutar el baseline con diferentes encoders
export PYTHONPATH="$PWD/src"

echo "========================================================"
echo "INICIANDO BASELINE BENCHMARK (GNN Simple t -> t+1)"
echo "========================================================"

# Encoders a probar
ENCODERS=("graphormer" "gat" "gcn")

for encoder in "${ENCODERS[@]}"; do
    echo ""
    echo "--------------------------------------------------------"
    echo "Entrenando Baseline con Encoder: $encoder"
    echo "--------------------------------------------------------"
    
    python scripts/run_baseline.py --gnn_type "$encoder"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Falló el baseline de $encoder"
    else
        echo "Baseline $encoder completado."
    fi
done

echo ""
echo "========================================================"
echo "BASELINE BENCHMARK FINALIZADO"
echo "========================================================"
