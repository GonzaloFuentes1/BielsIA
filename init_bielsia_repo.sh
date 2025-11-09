#!/usr//bin/env bash

# ==============================================================================
# Script para inicializar el esqueleto del repositorio del proyecto BielsIA
#
# Uso:
#   1. Guardar este script como 'init_bielsia_repo.sh'
#   2. Darle permisos de ejecución: chmod +x init_bielsia_repo.sh
#   3. Ejecutarlo en una carpeta vacía: ./init_bielsia_repo.sh
#
# El script creará una carpeta 'bielsia/' con toda la estructura de
# directorios y archivos iniciales.
# ==============================================================================

set -e

# --- Crear directorio raíz y entrar en él ---
echo "Creando el directorio raíz del proyecto: bielsia/"
mkdir -p bielsia
cd bielsia

# --- Crear estructura de directorios principal ---
echo "Creando la estructura de directorios..."
mkdir -p configs
mkdir -p data/{raw,interim,processed/{graphs,features},sample}
mkdir -p notebooks
mkdir -p src/bielsia/{config,scraping,data,models,training,evaluation,recommendation,utils}
mkdir -p experiments/{configs,runs}
mkdir -p docs
mkdir -p tests
mkdir -p scripts
mkdir -p reports/{figures,tables}

# --- Crear archivos de configuración y raíz ---
echo "Creando archivos de configuración y raíz..."

# .gitignore
cat << 'EOF' > .gitignore
# Entornos virtuales y dependencias
.venv/
venv/
__pycache__/
*.pyc
*.pyo
.DS_Store

# Configuraciones de IDE
.idea/
.vscode/

# Datos generados (no versionar)
data/raw/
data/interim/
data/processed/

# Resultados de experimentos
experiments/runs/

# Archivos de build y distribución
build/
dist/
*.egg-info/
EOF

# pyproject.toml
cat << 'EOF' > pyproject.toml
[project]
name = "bielsia"
version = "0.1.0"
description = "Un sistema de recomendación de fichajes en fútbol basado en grafos y GNNs."
requires-python = ">=3.10"
dependencies = [
    "torch",
    "torch-geometric",
    "pandas",
    "numpy",
    "scikit-learn",
    "matplotlib",
    "pyyaml",
    "tqdm",
    "loguru"
]

[project.scripts]
bielsia = "bielsia.cli:main"

[project.optional-dependencies]
dev = [
    "pytest",
    "ruff",
    "ipykernel"
]
EOF

# Makefile
cat << 'EOF' > Makefile
.PHONY: setup test lint demo

# Asume que el entorno virtual se llamará .venv
VENV_PYTHON = ./.venv/bin/python

setup:
	@echo ">>> Creando entorno virtual en .venv..."
	python3 -m venv .venv
	@echo ">>> Instalando dependencias en modo editable..."
	$(VENV_PYTHON) -m pip install -e ".[dev]"
	@echo ">>> Setup completo. Activa el entorno con: source .venv/bin/activate"

test:
	@echo ">>> Ejecutando tests con pytest..."
	$(VENV_PYTHON) -m pytest tests/

lint:
	@echo ">>> Pasando linter (ruff)..."
	$(VENV_PYTHON) -m ruff check .

demo:
	@echo ">>> Ejecutando un pipeline de demostración..."
	$(VENV_PYTHON) -m bielsia.cli --mode demo --config-path configs/demo.yaml
EOF

# README.md principal
cat << 'EOF' > README.md
# BielsIA – Building Intelligent Ensembles by Linking Scouting Insights and Analytics

**BielsIA** es un proyecto universitario que desarrolla un sistema de recomendación de fichajes en el fútbol profesional. A diferencia de los métodos tradicionales que se centran en estadísticas individuales, este proyecto utiliza **redes neuronales de grafos (GNNs)** para modelar las complejas interacciones entre jugadores y equipos, buscando el "fit" contextual de un jugador en un nuevo entorno.

El sistema se basa en datos históricos de partidos (ej: Premier League desde 2020) para construir un grafo de jugadores. Los modelos GNN (como GCN, GAT y Graphormer) aprenden representaciones latentes de los jugadores que capturan tanto su habilidad individual como su sinergia con otros. Finalmente, el proyecto incluye un innovador módulo de **backtesting** que evalúa la calidad de las recomendaciones utilizando fichajes históricos reales, permitiendo validar si el modelo habría predicho fichajes exitosos y evitado los fallidos.

## Pipeline del Proyecto

El flujo de trabajo se organiza en las siguientes etapas clave:

1.  **Descarga y Limpieza de Datos**: Extracción de datos de fuentes públicas como FbRef, incluyendo estadísticas de jugadores, resultados de partidos y eventos de juego.
2.  **Construcción del Dataset y Grafo**: Procesamiento de los datos crudos para construir un grafo heterogéneo donde los nodos son jugadores y las aristas representan interacciones (compañeros de equipo, rivales, etc.).
3.  **Entrenamiento de Modelos GNN**: Entrenamiento de modelos como GCN, GAT y Graphormer sobre el grafo para aprender representaciones de jugadores que capturen su contexto y estilo.
4.  **Recomendación de Fichajes**: Uso de los embeddings aprendidos para generar recomendaciones de jugadores para un equipo específico, basándose en la afinidad y el potencial de sinergia.
5.  **Backtesting con Fichajes Reales**: Evaluación del sistema simulando escenarios de fichajes pasados y comparando las recomendaciones del modelo con el rendimiento real de los jugadores fichados.

## Estructura del Repositorio

-   `configs/`: Archivos de configuración YAML para experimentos y ejecuciones.
-   `data/`: Datos del proyecto, separados en `raw`, `interim`, `processed` y `sample`.
-   `notebooks/`: Jupyter Notebooks para exploración, prototipado y visualización.
-   `src/bielsia/`: Código fuente principal del proyecto, estructurado como un paquete Python instalable.
-   `experiments/`: Configuraciones y resultados de experimentos de modelado.
-   `docs/`: Documentación detallada del proyecto (diccionario de datos, arquitectura, etc.).
-   `tests/`: Tests unitarios y de integración para asegurar la calidad del código.
-   `scripts/`: Scripts de utilidad para automatizar tareas (descarga, entrenamiento, etc.).
-   `reports/`: Figuras, tablas y otros artefactos para el informe final.
EOF

# --- Archivos de configuración ---
echo "Creando archivos de configuración de ejemplo..."
touch configs/demo.yaml
touch configs/training_default.yaml

# --- READMEs y archivos de datos ---
echo "Creando estructura y READMEs para 'data'..."

cat << 'EOF' > data/README.md
# Directorio de Datos

Este directorio contiene todos los datos utilizados en el proyecto, organizados de la siguiente manera:

-   `raw/`: Datos brutos y sin procesar, tal como se descargan de las fuentes originales (ej: HTMLs de FbRef). Esta carpeta está en el `.gitignore`.
-   `interim/`: Datos intermedios que han sido transformados o limpiados, pero que aún no están en su forma final para el modelado. Esta carpeta está en el `.gitignore`.
-   `processed/`: Datos finales, listos para ser consumidos por los modelos. Incluye:
    -   `graphs/`: Grafos serializados (ej: en formato PyG).
    -   `features/`: Tablas de características de nodos/aristas.
-   `sample/`: Un pequeño subconjunto de datos para ejecutar el pipeline rápidamente en modo de prueba o demostración.
EOF

cat << 'EOF' > data/sample/README.md
# Datos de Muestra

Estos archivos CSV contienen un subconjunto muy pequeño de datos (jugadores y partidos) para permitir una ejecución rápida del pipeline en modo "demo".

Son útiles para:
- Probar la lógica de construcción de grafos sin esperar a procesar toda la base de datos.
- Ejecutar tests de integración.
- Depurar el código de modelado con un ciclo de feedback rápido.
EOF

cat << 'EOF' > data/sample/players_sample.csv
player_id,player_name,position,age
abcdef123,John Doe,Midfielder,25
ghjkl456,Peter Pan,Forward,22
EOF

cat << 'EOF' > data/sample/matches_sample.csv
match_id,home_team,away_team,home_score,away_score
12345,Team A,Team B,2,1
EOF

# --- READMEs y archivos de código fuente ---
echo "Creando estructura y READMEs para 'src'..."

# Módulos __init__.py
touch src/bielsia/__init__.py
touch src/bielsia/config/__init__.py
touch src/bielsia/scraping/__init__.py
touch src/bielsia/data/__init__.py
touch src/bielsia/models/__init__.py
touch src/bielsia/training/__init__.py
touch src/bielsia/evaluation/__init__.py
touch src/bielsia/recommendation/__init__.py
touch src/bielsia/utils/__init__.py

# README de src/bielsia
cat << 'EOF' > src/bielsia/README.md
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
EOF

# Archivos .py con stubs
cat << 'EOF' > src/bielsia/config/default_config.yaml
# Este es un archivo de configuración por defecto.
# Puede ser sobreescrito por los archivos en /configs o /experiments.
training:
  learning_rate: 0.001
  epochs: 100
  batch_size: 32

model:
  name: "gcn"
  hidden_dim: 128
  output_dim: 64
EOF

cat << 'EOF' > src/bielsia/scraping/fbref_scraper.py
"""Módulo para descargar datos de FbRef."""

def download_fbref_season_data(season: int, league: str):
    """
    Descarga los datos de una temporada y liga específicas desde FbRef.
    (Implementación pendiente)
    """
    print(f"Descargando datos para {league}, temporada {season}...")
    pass
EOF

cat << 'EOF' > src/bielsia/scraping/fbref_parser.py
"""Módulo para parsear HTMLs descargados de FbRef."""

def parse_player_stats_from_html(html_content: str):
    """
    Parsea las estadísticas de jugadores desde el contenido HTML.
    (Implementación pendiente)
    """
    print("Parseando estadísticas de jugadores...")
    pass
EOF

cat << 'EOF' > src/bielsia/data/dataset_builder.py
"""Módulo para construir el dataset a partir de datos crudos."""

def create_feature_tables(raw_data_path: str):
    """
    Crea tablas de características a partir de los datos crudos.
    (Implementación pendiente)
    """
    print("Creando tablas de características...")
    pass
EOF

cat << 'EOF' > src/bielsia/data/graph_construction.py
"""Módulo para la construcción de los grafos de jugadores."""

import pandas as pd

def build_player_graph(player_df: pd.DataFrame, match_df: pd.DataFrame):
    """
    Construye el grafo de jugadores y sus interacciones.
    (Implementación pendiente)
    """
    print("Construyendo el grafo de jugadores...")
    # Devuelve un objeto grafo dummy para los tests
    return "GraphObject"
EOF

cat << 'EOF' > src/bielsia/data/splits.py
"""Módulo para crear divisiones de datos (train/validation/test)."""

def create_temporal_split(graph, test_season: int):
    """
    Divide el grafo en conjuntos de entrenamiento, validación y test
    de forma temporal.
    (Implementación pendiente)
    """
    print(f"Creando split temporal con test en la temporada {test_season}...")
    pass
EOF

cat << 'EOF' > src/bielsia/models/gcn.py
"""Definición del modelo GCN (Graph Convolutional Network)."""

import torch.nn as nn

class GCN(nn.Module):
    """Implementación de un modelo GCN simple."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # Definición de capas pendiente
        print("Inicializando modelo GCN...")
        pass

    def forward(self, x, edge_index):
        # Lógica de forward pendiente
        pass
EOF

cat << 'EOF' > src/bielsia/models/gat.py
"""Definición del modelo GAT (Graph Attention Network)."""

import torch.nn as nn

class GAT(nn.Module):
    """Implementación de un modelo GAT simple."""
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super().__init__()
        # Definición de capas pendiente
        print("Inicializando modelo GAT...")
        pass

    def forward(self, x, edge_index):
        # Lógica de forward pendiente
        pass
EOF

cat << 'EOF' > src/bielsia/models/graphormer.py
"""Definición del modelo Graphormer (placeholder)."""

import torch.nn as nn

class Graphormer(nn.Module):
    """Placeholder para una futura implementación de Graphormer."""
    def __init__(self):
        super().__init__()
        print("Inicializando modelo Graphormer (placeholder)...")
        pass

    def forward(self, data):
        # Lógica de forward pendiente
        pass
EOF

cat << 'EOF' > src/bielsia/training/train_loop.py
"""Bucle de entrenamiento genérico."""

def train_one_epoch(model, dataloader, optimizer, criterion):
    """
    Ejecuta una época de entrenamiento.
    (Implementación pendiente)
    """
    pass
EOF

cat << 'EOF' > src/bielsia/training/train_gnn.py
"""Script principal para orquestar el entrenamiento de un modelo GNN."""

def train_gnn_model(config: dict):
    """
    Orquesta el proceso de entrenamiento completo basado en un config.
    (Implementación pendiente)
    """
    print(f"Iniciando entrenamiento del modelo {config.get('model', {}).get('name')}...")
    pass
EOF

cat << 'EOF' > src/bielsia/evaluation/metrics.py
"""Métricas de evaluación para el rendimiento del modelo."""

def calculate_hit_rate_at_k(recommendations, ground_truth, k: int):
    """
    Calcula la métrica Hit-Rate@K.
    (Implementación pendiente)
    """
    pass
EOF

cat << 'EOF' > src/bielsia/evaluation/eval_season.py
"""Evaluación del modelo en una temporada completa."""

def evaluate_model_on_season(model, graph, season: int):
    """
    Evalúa el rendimiento del modelo en una temporada específica.
    (Implementación pendiente)
    """
    pass
EOF

cat << 'EOF' > src/bielsia/evaluation/backtesting.py
"""Framework para el backtesting de fichajes históricos."""

def run_backtesting_experiment(model, historical_transfers_df):
    """
    Ejecuta un experimento de backtesting.
    (Implementación pendiente)
    """
    print("Ejecutando experimento de backtesting...")
    return "BacktestingResults"
EOF

cat << 'EOF' > src/bielsia/recommendation/team_recommender.py
"""Lógica para recomendar jugadores a un equipo."""

def recommend_players_for_team(model, graph, team_id: str, n_recommendations: int = 10):
    """
    Recomienda los 'n' mejores jugadores para un equipo dado.
    (Implementación pendiente)
    """
    print(f"Generando {n_recommendations} recomendaciones para el equipo {team_id}...")
    pass
EOF

cat << 'EOF' > src/bielsia/recommendation/player_recommender.py
"""Lógica para recomendar equipos para un jugador."""

def recommend_teams_for_player(model, graph, player_id: str, n_recommendations: int = 5):
    """
    Recomienda los 'n' mejores equipos para un jugador dado.
    (Implementación pendiente)
    """
    print(f"Generando {n_recommendations} recomendaciones de equipos para el jugador {player_id}...")
    pass
EOF

cat << 'EOF' > src/bielsia/utils/logging_utils.py
"""Utilidades para la configuración del logging."""

def setup_logging():
    """
    Configura el logger para el proyecto.
    (Implementación pendiente)
    """
    pass
EOF

cat << 'EOF' > src/bielsia/utils/seed.py
"""Utilidades para fijar la semilla aleatoria."""

import random
import numpy as np
import torch

def set_seed(seed_value: int):
    """Fija la semilla para reproducibilidad."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
EOF

cat << 'EOF' > src/bielsia/utils/paths.py
"""Gestión centralizada de rutas del proyecto."""

from pathlib import Path

# Directorio raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EOF

cat << 'EOF' > src/bielsia/cli.py
"""
Punto de entrada principal para la interfaz de línea de comandos (CLI) de BielsIA.

Permite orquestar las diferentes etapas del pipeline desde la terminal.
"""
import argparse

def main():
    """Función principal de la CLI."""
    parser = argparse.ArgumentParser(description="BielsIA CLI")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["fetch-data", "make-dataset", "train-gnn", "backtesting", "demo"],
        help="Modo de ejecución del pipeline."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="",
        help="Ruta al archivo de configuración YAML."
    )

    args = parser.parse_args()

    print(f"Ejecutando en modo: {args.mode}")
    if args.config_path:
        print(f"Usando config: {args.config_path}")

    # Aquí iría la lógica para llamar a las funciones correspondientes
    if args.mode == "demo":
        print("Ejecutando pipeline de demostración...")
        # from bielsia.data.graph_construction import build_player_graph
        # build_player_graph(None, None) # Llamada de ejemplo
    else:
        print("Modo no implementado todavía.")

if __name__ == "__main__":
    main()
EOF

# --- Notebooks (placeholders) ---
echo "Creando notebooks de ejemplo..."
# Estructura JSON mínima para un notebook vacío
EMPTY_NB_CONTENT='{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 2
}'
echo "$EMPTY_NB_CONTENT" > notebooks/01_exploracion_datos.ipynb
echo "$EMPTY_NB_CONTENT" > notebooks/02_construccion_grafos.ipynb
echo "$EMPTY_NB_CONTENT" > notebooks/03_modelado_y_recomendaciones.ipynb
echo "$EMPTY_NB_CONTENT" > notebooks/04_backtesting_fichajes.ipynb

cat << 'EOF' > notebooks/README.md
# Cuadernos de Jupyter

Este directorio contiene cuadernos de Jupyter para análisis exploratorio, prototipado de modelos y visualización de resultados.

-   `01_exploracion_datos.ipynb`: Análisis inicial de los datos crudos, visualización de distribuciones y estadísticas descriptivas.
-   `02_construccion_grafos.ipynb`: Prototipado y visualización de la construcción de grafos.
-   `03_modelado_y_recomendaciones.ipynb`: Experimentos interactivos con los modelos GNN y la generación de recomendaciones.
-   `04_backtesting_fichajes.ipynb`: Análisis y visualización de los resultados del backtesting.
EOF

# --- Experimentos ---
echo "Creando estructura para 'experiments'..."
cat << 'EOF' > experiments/README.md
# Experimentos de Modelado

Este directorio se utiliza para organizar y registrar los experimentos de entrenamiento de modelos.

-   `configs/`: Contiene los archivos de configuración YAML para cada experimento específico. Cada archivo define los hiperparámetros, el modelo a usar, el dataset, etc.
-   `runs/`: Directorio donde se guardarán los resultados de cada ejecución (logs, checkpoints del modelo, métricas, etc.). Esta carpeta está en el `.gitignore` para no versionar los artefactos de los experimentos.

Ejemplos de experimentos que se pueden definir:
-   `gat_baseline.yaml`: Un entrenamiento base con el modelo GAT.
-   `gcn_baseline.yaml`: Un entrenamiento base con el modelo GCN.
-   `graphormer_baseline.yaml`: Un entrenamiento base con el modelo Graphormer.
-   `backtesting_example.yaml`: Una configuración para un experimento de backtesting.
EOF
touch experiments/configs/gat_baseline.yaml
touch experiments/configs/gcn_baseline.yaml
touch experiments/configs/graphormer_baseline.yaml
touch experiments/configs/backtesting_example.yaml

# --- Documentación ---
echo "Creando estructura para 'docs'..."
cat << 'EOF' > docs/README.md
# Documentación del Proyecto

Este directorio contiene la documentación detallada del proyecto.

-   `project_overview.md`: Descripción en profundidad de la motivación, objetivos y arquitectura del sistema.
-   `data_dictionary.md`: Diccionario que describe cada una de las variables y características utilizadas en el proyecto.
-   `graphs_definition.md`: Explicación formal de la estructura de los grafos (tipos de nodos, tipos de aristas, atributos).
-   `backtesting_overview.md`: Descripción detallada de la metodología de backtesting implementada.
EOF
touch docs/project_overview.md
touch docs/data_dictionary.md
touch docs/graphs_definition.md
touch docs/backtesting_overview.md

# --- Tests ---
echo "Creando estructura y tests iniciales para 'tests'..."
cat << 'EOF' > tests/README.md
# Tests del Proyecto

Este directorio contiene los tests para asegurar la calidad y robustez del código.

-   `test_imports.py`: Un test simple para asegurar que todos los módulos principales se pueden importar sin errores de sintaxis o dependencias circulares.
-   `test_graph_construction.py`: Tests para la lógica de construcción de grafos.
-   `test_backtesting.py`: Tests para el framework de backtesting.

Para ejecutar los tests, utiliza el comando `make test` desde la raíz del proyecto.
EOF

cat << 'EOF' > tests/test_imports.py
"""Test para asegurar que los módulos principales son importables."""

import pytest

def test_project_imports():
    """Asegura que los módulos clave del proyecto se pueden importar."""
    try:
        import bielsia
        import bielsia.cli
        import bielsia.data.graph_construction
        import bielsia.models.gcn
        import bielsia.evaluation.backtesting
        from bielsia.utils import paths
    except ImportError as e:
        pytest.fail(f"Fallo al importar un módulo: {e}")
    assert True
EOF

cat << 'EOF' > tests/test_graph_construction.py
"""Tests para la construcción de grafos."""

import pytest
import pandas as pd
from bielsia.data.graph_construction import build_player_graph

def test_build_player_graph_runs():
    """
    Test simple para asegurar que la función de construcción de grafos
    se ejecuta sin errores con datos dummy.
    """
    # Datos dummy
    players = pd.DataFrame([{"player_id": "p1", "name": "A"}])
    matches = pd.DataFrame([{"match_id": "m1"}])

    try:
        result = build_player_graph(players, matches)
        # El stub devuelve un string, comprobamos eso
        assert isinstance(result, str)
    except Exception as e:
        pytest.fail(f"build_player_graph falló con datos dummy: {e}")
EOF

cat << 'EOF' > tests/test_backtesting.py
"""Tests para el framework de backtesting."""

import pytest
import pandas as pd
from bielsia.evaluation.backtesting import run_backtesting_experiment

def test_run_backtesting_experiment_runs():
    """
    Test simple para asegurar que la función de backtesting se ejecuta
    sin errores con datos dummy.
    """
    # Datos dummy
    transfers = pd.DataFrame([{"player_id": "p1", "from_team": "t1", "to_team": "t2"}])
    model_dummy = "DummyModel" # Simula un modelo entrenado

    try:
        result = run_backtesting_experiment(model_dummy, transfers)
        assert isinstance(result, str)
    except Exception as e:
        pytest.fail(f"run_backtesting_experiment falló con datos dummy: {e}")
EOF

# --- Scripts ---
echo "Creando estructura y scripts para 'scripts'..."
cat << 'EOF' > scripts/README.md
# Scripts de Automatización

Este directorio contiene scripts de shell para automatizar tareas comunes del pipeline.

-   `download_fbref_data.sh`: Descarga o actualiza los datos crudos desde la fuente (ej: FbRef).
-   `make_dataset.sh`: Ejecuta el pipeline de procesamiento de datos para construir el dataset final y los grafos.
-   `train_gnn.sh`: Lanza un trabajo de entrenamiento para un modelo GNN, usando una configuración específica.
-   `run_backtesting.sh`: Ejecuta el pipeline de backtesting con un modelo y un conjunto de datos de fichajes.
EOF

cat << 'EOF' > scripts/download_fbref_data.sh
#!/usr/bin/env bash
# Script para descargar datos de FbRef
echo "Descargando datos crudos (placeholder)..."
# python src/bielsia/cli.py --mode fetch-data
sleep 1
echo "Datos descargados."
EOF

cat << 'EOF' > scripts/make_dataset.sh
#!/usr/bin/env bash
# Script para construir el dataset y los grafos
echo "Construyendo dataset y grafos (placeholder)..."
# python src/bielsia/cli.py --mode make-dataset
sleep 1
echo "Dataset construido."
EOF

cat << 'EOF' > scripts/train_gnn.sh
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
EOF

cat << 'EOF' > scripts/run_backtesting.sh
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
EOF
chmod +x scripts/*.sh

# --- Reportes ---
echo "Creando estructura para 'reports'..."
cat << 'EOF' > reports/README.md
# Informes y Resultados

Este directorio almacena los artefactos generados para el informe final del proyecto.

-   `figures/`: Gráficos, diagramas y visualizaciones generadas durante el análisis y la evaluación de modelos (ej: curvas de pérdida, embeddings de jugadores, etc.).
-   `tables/`: Tablas con resultados numéricos (ej: comparación de métricas de modelos, resultados del backtesting).
EOF
touch reports/figures/.gitkeep
touch reports/tables/.gitkeep

echo ""
echo "======================================================"
echo "¡El esqueleto del repositorio BielsIA ha sido creado!"
echo "Directorio actual: $(pwd)"
echo "Para empezar, puedes probar los siguientes comandos:"
echo "  make setup"
echo "  source .venv/bin/activate"
echo "  make test"
echo "  make demo"
echo "======================================================"
