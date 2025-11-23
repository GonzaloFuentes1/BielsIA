#!/bin/bash

# Directorio base
BASE_DIR="/workspace1/gonzalo.fuentes/BielsIA"
CONFIGS_DIR="$BASE_DIR/configs"
REPORTS_DIR="$BASE_DIR/reports"
MODELS_DIR="$BASE_DIR/models"

# Activar entorno virtual explícitamente
source /workspace1/gonzalo.fuentes/BielsIA/.venv/bin/activate

cd "$BASE_DIR"

# Lista de todos los modelos
MODELS=(
    "lstm_gat"
    "lstm_gcn"
    "lstm_graphormer"
    "transformer_gat"
    "transformer_gcn"
    "transformer_graphormer"
)

echo "========================================================"
echo "EJECUTANDO INFERENCIA FINAL PARA TODOS LOS MODELOS"
echo "========================================================"

for model_name in "${MODELS[@]}"; do
    config_file="$CONFIGS_DIR/${model_name}.yaml"
    
    echo ""
    echo "--------------------------------------------------------"
    echo "Procesando Modelo: $model_name"
    echo "--------------------------------------------------------"

    # Identificar el mejor modelo guardado
    best_model_path="$MODELS_DIR/${model_name}_best.pth"
    
    if [ ! -f "$best_model_path" ]; then
        echo "ERROR: No se encontró el modelo entrenado en $best_model_path"
        continue
    fi
    
    echo "Modelo encontrado: $best_model_path"

    # 1. Evaluación 2024 (Out-of-Sample / Test)
    echo "[1/2] Evaluando 2024..."
    python3 scripts/evaluate_year.py --year 2024 --config "$config_file" --model_path "$best_model_path"

    # 2. Análisis Comprensivo y Guardado de Resultados
    echo "[2/2] Generando Reportes..."
    python3 scripts/comprehensive_analysis.py --model_name "$model_name"

    # 3. Mover resultados a carpeta específica del modelo
    MODEL_RESULT_DIR="$REPORTS_DIR/benchmark/$model_name"
    mkdir -p "$MODEL_RESULT_DIR"
    
    # Mover tablas
    mv "$REPORTS_DIR/tables/predictions_2024_${model_name}.csv" "$MODEL_RESULT_DIR/" 2>/dev/null
    mv "$REPORTS_DIR/tables/error_analysis_2024.csv" "$MODEL_RESULT_DIR/" 2>/dev/null
    mv "$REPORTS_DIR/tables/team_metrics_2024_${model_name}.csv" "$MODEL_RESULT_DIR/" 2>/dev/null
    mv "$REPORTS_DIR/tables/split_metrics_2024_${model_name}.csv" "$MODEL_RESULT_DIR/" 2>/dev/null

    # Mover figuras
    mv "$REPORTS_DIR/figures/"*${model_name}.png "$MODEL_RESULT_DIR/" 2>/dev/null
    # Mover visualizaciones de hipergrafos si se generaron
    mv "$REPORTS_DIR/figures/hypergraph_visualization_"*.png "$MODEL_RESULT_DIR/" 2>/dev/null

    echo "Resultados guardados en: $MODEL_RESULT_DIR"
done

echo ""
echo "========================================================"
echo "GENERANDO COMPARATIVA FINAL ENTRE MODELOS"
echo "========================================================"
python3 scripts/compare_models.py

echo ""
echo "========================================================"
echo "INFERENCIA FINAL COMPLETADA"
echo "========================================================"
