# Guía de Entrenamiento de Modelos Temporales

Esta guía explica cómo configurar y ejecutar el entrenamiento de modelos de Deep Learning sobre grafos temporales en BielsIA.

## Estructura del Proyecto de Entrenamiento

El sistema de entrenamiento se ha unificado para soportar múltiples arquitecturas (LSTM, Transformer) y configuraciones mediante archivos YAML.

### Archivos Clave

*   **`src/bielsia/training/train.py`**: Script principal de entrenamiento. Carga la configuración, construye el dataset y ejecuta el bucle de entrenamiento.
*   **`configs/train_config.yaml`**: Archivo de configuración central. Define hiperparámetros, rutas de datos y arquitectura del modelo.
*   **`src/bielsia/models/temporal.py`**: Definición de las arquitecturas de modelos (`TemporalLSTMGNN`, `TemporalTransformerGNN`).

## Configuración (`configs/train_config.yaml`)

Puedes modificar este archivo para ajustar el experimento sin tocar el código.

```yaml
data:
  seasons: [2020, 2021, 2022, 2023, 2024] # Temporadas a incluir
  interim_path: "data/interim"

model:
  type: "transformer" # "lstm" o "transformer"
  
  # Hiperparámetros GNN (Encoder por temporada)
  gnn_hidden: 64
  gnn_out: 64
  heads: 2
  dropout: 0.2
  
  # Hiperparámetros Transformer
  transformer_hidden: 128
  transformer_layers: 2
  
  # Hiperparámetros LSTM
  lstm_hidden: 128

training:
  epochs: 500
  lr: 0.001
  device: "auto" # "cuda", "cpu" o "auto"
  save_dir: "models"
  save_name: "temporal_model_v1.pth"
```

## Ejecución del Entrenamiento

Para entrenar el modelo, simplemente ejecuta el siguiente comando desde la raíz del proyecto:

```bash
python src/bielsia/training/train.py --config configs/train_config.yaml
```

El script realizará lo siguiente automáticamente:
1.  Cargará los grafos de las temporadas especificadas.
2.  Construirá un **Mapa Global de Jugadores** para manejar jugadores que entran y salen de la liga.
3.  Inicializará el modelo seleccionado (Transformer o LSTM).
4.  Entrenará durante el número de épocas definido.
5.  Guardará el modelo entrenado en la carpeta `models/`.

## Manejo de Jugadores Faltantes (Missing Data)

Una duda común es cómo se manejan los jugadores que no están presentes en todas las temporadas (por ejemplo, fichajes nuevos o jugadores que se retiran).

El sistema utiliza una estrategia de **Alineación Global y Enmascaramiento**:

1.  **Mapa Global**: Se crea un índice único de todos los jugadores que han aparecido en al menos una temporada del periodo de entrenamiento.
2.  **Padding**: Si un jugador no está presente en la temporada $t$, su vector de características (embedding) para ese paso temporal se rellena con ceros.
3.  **Masking (Enmascaramiento)**:
    *   **En el Transformer**: Se utiliza una `src_key_padding_mask` en el mecanismo de atención. Esto asegura que el modelo **ignore** completamente los pasos temporales donde el jugador no existía (vectores de ceros), evitando que "contaminen" la representación aprendida.
    *   **En la Loss**: La función de pérdida solo se calcula sobre los jugadores que están presentes y tienen datos válidos en la temporada objetivo.

Esto permite que el modelo aprenda de la historia completa de un veterano y, al mismo tiempo, pueda hacer predicciones para un jugador que acaba de llegar, basándose solo en sus datos disponibles.
