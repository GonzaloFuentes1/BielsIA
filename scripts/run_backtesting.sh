#!/usr/bin/env bash
# Script para ejecutar el backtesting
CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
  echo "Error: Debes proporcionar la ruta a un archivo de configuración."
  echo "Uso: $0 experiments/configs/backtesting_example.yaml"
  exit 1
fi
echo "Ejecutando backtesting con config: $CONFIG_FILE (placeholder)..."
# python src/bielsia/cli.py --mode backtesting --config-path $CONFIG_FILE
sleep 1
echo "Backtesting completado."
