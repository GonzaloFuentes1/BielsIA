#!/usr/bin/env bash
# Script para entrenar un modelo GNN
CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
  echo "Error: Debes proporcionar la ruta a un archivo de configuración."
  echo "Uso: $0 experiments/configs/gcn_baseline.yaml"
  exit 1
fi
echo "Entrenando modelo GNN con config: $CONFIG_FILE (placeholder)..."
# python src/bielsia/cli.py --mode train-gnn --config-path $CONFIG_FILE
sleep 1
echo "Entrenamiento completado."
