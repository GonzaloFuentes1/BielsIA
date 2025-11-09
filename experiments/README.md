# Experimentos de Modelado

Este directorio se utiliza para organizar y registrar los experimentos de entrenamiento de modelos.

-   `configs/`: Contiene los archivos de configuración YAML para cada experimento específico. Cada archivo define los hiperparámetros, el modelo a usar, el dataset, etc.
-   `runs/`: Directorio donde se guardarán los resultados de cada ejecución (logs, checkpoints del modelo, métricas, etc.). Esta carpeta está en el `.gitignore` para no versionar los artefactos de los experimentos.

Ejemplos de experimentos que se pueden definir:
-   `gat_baseline.yaml`: Un entrenamiento base con el modelo GAT.
-   `gcn_baseline.yaml`: Un entrenamiento base con el modelo GCN.
-   `graphormer_baseline.yaml`: Un entrenamiento base con el modelo Graphormer.
-   `backtesting_example.yaml`: Una configuración para un experimento de backtesting.
