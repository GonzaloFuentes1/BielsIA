# Código Fuente de BielsIA

Este directorio contiene el código fuente principal del proyecto, empaquetado como un módulo Python instalable llamado `bielsia`.

La estructura interna sigue una organización por funcionalidad:

-   `config/`: Carga y gestión de configuraciones.
-   `scraping/`: Módulos para la descarga (`scraper`) y el parseo (`parser`) de datos de fuentes web como FbRef.
-   `data/`: Lógica para la construcción de datasets, la creación de grafos (`graph_construction`) y la división de datos (`splits`).
-   `models/`: Definiciones de las arquitecturas de los modelos GNN (GCN, GAT, Graphormer).
-   `training/`: Bucles de entrenamiento (`train_loop`) y scripts para orquestar el entrenamiento (`train_gnn`).
-   `evaluation/`: Métricas de evaluación, lógica para evaluar una temporada completa y el framework de backtesting.
-   `recommendation/`: Lógica para generar recomendaciones, ya sea para un equipo (`team_recommender`) o buscando equipos para un jugador (`player_recommender`).
-   `utils/`: Funciones y clases de utilidad general (logging, gestión de rutas, etc.).
-   `cli.py`: Punto de entrada para la interfaz de línea de comandos (CLI) del proyecto.

El objetivo es que todo el pipeline pueda ser orquestado mediante código importando estos módulos o a través de la CLI.
