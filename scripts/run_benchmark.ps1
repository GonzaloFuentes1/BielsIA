# Script para ejecutar todos los experimentos de benchmark
$env:PYTHONPATH = "$PWD/src"

Write-Host "--------------------------------------------------"
Write-Host "Generando grafos y visualizaciones (2016-2024)..."
Write-Host "--------------------------------------------------"
python src/bielsia/data/graph_construction.py

$configs = @(
    # "configs/lstm_gat.yaml",
    # "configs/lstm_gcn.yaml",
    # "configs/lstm_graphormer.yaml",
    # "configs/lstm_hybrid_gcn_graphormer.yaml",
    # "configs/lstm_hybrid_gat_graphormer.yaml",
    # "configs/lstm_hybrid_gat_gcn.yaml",
    # "configs/lstm_hybrid_graphormer_gcn.yaml",
    # "configs/lstm_hybrid_graphormer_gat.yaml",
    "configs/transformer_gat.yaml",
    "configs/transformer_gcn.yaml",
    "configs/transformer_graphormer.yaml"
    # "configs/transformer_hybrid_gcn_graphormer.yaml",
    # "configs/transformer_hybrid_gat_graphormer.yaml",
    # "configs/transformer_hybrid_gat_gcn.yaml",
    # "configs/transformer_hybrid_graphormer_gcn.yaml",
    # "configs/transformer_hybrid_graphormer_gat.yaml"
)

foreach ($config in $configs) {
    Write-Host "--------------------------------------------------"
    Write-Host "Ejecutando entrenamiento con: $config"
    Write-Host "--------------------------------------------------"
    python src/bielsia/training/train.py --config $config
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error ejecutando $config" -ForegroundColor Red
    }
}

Write-Host "--------------------------------------------------"
Write-Host "Generando reporte comparativo..."
Write-Host "--------------------------------------------------"
python src/bielsia/evaluation/compare.py

Write-Host "Benchmark finalizado."
