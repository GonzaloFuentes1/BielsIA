# Directorio de Datos

Este directorio contiene todos los datos utilizados en el proyecto, organizados de la siguiente manera:

-   `raw/`: Datos brutos y sin procesar, tal como se descargan de las fuentes originales (ej: HTMLs de FbRef). Esta carpeta está en el `.gitignore`.
-   `interim/`: Datos intermedios que han sido transformados o limpiados, pero que aún no están en su forma final para el modelado. Esta carpeta está en el `.gitignore`.
-   `processed/`: Datos finales, listos para ser consumidos por los modelos. Incluye:
    -   `graphs/`: Grafos serializados (ej: en formato PyG).
    -   `features/`: Tablas de características de nodos/aristas.
-   `sample/`: Un pequeño subconjunto de datos para ejecutar el pipeline rápidamente en modo de prueba o demostración.
