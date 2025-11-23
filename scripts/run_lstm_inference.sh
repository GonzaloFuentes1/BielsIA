#!/bin/bash

# Directorio base
BASE_DIR="/workspace1/gonzalo.fuentes/BielsIA"
CONFIGS_DIR="$BASE_DIR/configs"
REPORTS_DIR="$BASE_DIR/reports"
MODELS_DIR="$BASE_DIR/models"

# Crear directorios si no existen
mkdir -p "$REPORTS_DIR/benchmark"

# Activar entorno virtual
source /workspace1/gonzalo.fuentes/BielsIA/.venv/bin/activate

# Lista de configs a ejecutar (Solo LSTMs que ya entrenaron)
CONFIGS_LIST=(
    "$CONFIGS_DIR/lstm_gat.yaml"
    "$CONFIGS_DIR/lstm_gcn.yaml"
    "$CONFIGS_DIR/lstm_graphormer.yaml"
)

echo "========================================================"
echo "RECUPERANDO INFERENCIA PARA LSTMs"
echo "========================================================"

for config_file in "${CONFIGS_LIST[@]}"; do
    model_name=$(basename "$config_file" .yaml)
    echo ""
    echo "--------------------------------------------------------"
    echo "Procesando Modelo: $model_name"
    echo "--------------------------------------------------------"

    # Identificar el mejor modelo guardado
    best_model_path="$MODELS_DIR/${model_name}_best.pth"
    
    # Fallback si está en models/saved
    if [ ! -f "$best_model_path" ]; then
        best_model_path="$MODELS_DIR/saved/${model_name}_best.pth"
    fi

    if [ ! -f "$best_model_path" ]; then
        echo "ERROR: No se encontró el modelo entrenado en $best_model_path"
        continue
    fi
    
    echo "Modelo encontrado: $best_model_path"

    # 2. Evaluación 2024 (Out-of-Sample / Test)
    echo "[1/2] Evaluando 2024..."
    python3 "$BASE_DIR/scripts/evaluate_year.py" --year 2024 --config "$config_file" --model_path "$best_model_path"

    # 3. Análisis Comprensivo y Guardado de Resultados
    echo "[2/2] Generando Reportes..."
    python3 "$BASE_DIR/scripts/comprehensive_analysis.py" --model_name "$model_name"

    # 4. Mover resultados a carpeta específica del modelo
    MODEL_RESULT_DIR="$REPORTS_DIR/benchmark/$model_name"
    mkdir -p "$MODEL_RESULT_DIR"
    
    # Mover tablas
    mv "$REPORTS_DIR/tables/predictions_2024_${model_name}.csv" "$MODEL_RESULT_DIR/" 2>/dev/null
    mv "$REPORTS_DIR/tables/error_analysis_by_dimension.csv" "$MODEL_RESULT_DIR/" 2>/dev/null
    mv "$REPORTS_DIR/tables/team_metrics_2024_${model_name}.csv" "$MODEL_RESULT_DIR/" 2>/dev/null
    mv "$REPORTS_DIR/tables/split_metrics_2024_${model_name}.csv" "$MODEL_RESULT_DIR/" 2>/dev/null

    # Mover figuras
    mv "$REPORTS_DIR/figures/"*.png "$MODEL_RESULT_DIR/" 2>/dev/null

    echo "Resultados guardados en: $MODEL_RESULT_DIR"
done

echo ""
echo "========================================================"
echo "RECUPERACIÓN FINALIZADA"
echo "========================================================"
