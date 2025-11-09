# Scripts de Automatización

Este directorio contiene scripts de shell para automatizar tareas comunes del pipeline.

-   `download_fbref_data.sh`: Descarga o actualiza los datos crudos desde la fuente (ej: FbRef).
-   `make_dataset.sh`: Ejecuta el pipeline de procesamiento de datos para construir el dataset final y los grafos.
-   `train_gnn.sh`: Lanza un trabajo de entrenamiento para un modelo GNN, usando una configuración específica.
-   `run_backtesting.sh`: Ejecuta el pipeline de backtesting con un modelo y un conjunto de datos de fichajes.
