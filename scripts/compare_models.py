import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Configuración
BASE_DIR = Path("/workspace1/gonzalo.fuentes/BielsIA")
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "reports" / "comparison"

# Nombres de las dimensiones (Orden definido en dataset_builder.py)
DIMENSION_NAMES = [
    "Goals/90", "xG/90", "npxG/90", 
    "Assists/90", "xAG/90", "xG+xAG/90",
    "Tackles", "Interceptions", "Blocks", "Clearances",
    "Minutes Played"
]

def main():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # Buscar archivos de métricas JSON
    metric_files = list(MODELS_DIR.glob("*_metrics.json"))
    
    if not metric_files:
        print(f"No se encontraron archivos de métricas (*_metrics.json) en {MODELS_DIR}")
        return

    print(f"Encontrados {len(metric_files)} archivos de métricas.")

    summary_data = []
    dimension_data_norm = []
    dimension_data_orig = []

    for m_file in metric_files:
        try:
            with open(m_file, 'r') as f:
                metrics = json.load(f)
            
            model_name = metrics.get('model_name', m_file.stem.replace('_metrics', ''))
            
            # Usamos MAE Normalizado para el ranking global (comparable entre modelos)
            test_mae_norm = metrics.get('test_mae_norm', metrics.get('test_mae', 0))
            
            summary_data.append({
                'Model': model_name,
                'Test_MAE_Norm': test_mae_norm
            })
            
            # Desglose por dimensión (Normalizado)
            mae_per_dim_norm = metrics.get('mae_per_dim_norm', metrics.get('mae_per_dim', []))
            if len(mae_per_dim_norm) == len(DIMENSION_NAMES):
                for i, val in enumerate(mae_per_dim_norm):
                    dimension_data_norm.append({
                        'Model': model_name,
                        'Dimensión': DIMENSION_NAMES[i],
                        'MAE_Norm': val
                    })
            
            # Desglose por dimensión (Original - Interpretable)
            mae_per_dim_orig = metrics.get('mae_per_dim_orig', [])
            if len(mae_per_dim_orig) == len(DIMENSION_NAMES):
                for i, val in enumerate(mae_per_dim_orig):
                    dimension_data_orig.append({
                        'Model': model_name,
                        'Dimensión': DIMENSION_NAMES[i],
                        'MAE_Original': val
                    })

        except Exception as e:
            print(f"Error leyendo {m_file}: {e}")

    if not summary_data:
        print("No se pudieron extraer datos para comparar.")
        return

    # --- TABLA RESUMEN ---
    df_summary = pd.DataFrame(summary_data).sort_values('Test_MAE_Norm')
    print("\n" + "="*60)
    print("RANKING DE MODELOS (Menor Error Normalizado 2024)")
    print("="*60)
    print(df_summary.round(4).to_string(index=False))
    
    ranking_path = OUTPUT_DIR / "model_ranking.csv"
    df_summary.to_csv(ranking_path, index=False)
    
    # Guardar tabla de errores interpretables si existe
    if dimension_data_orig:
        df_orig = pd.DataFrame(dimension_data_orig)
        # Pivotar para tener Modelos en columnas y Dimensiones en filas
        df_orig_pivot = df_orig.pivot(index='Dimensión', columns='Model', values='MAE_Original')
        orig_path = OUTPUT_DIR / "interpretable_errors_2024.csv"
        df_orig_pivot.to_csv(orig_path)
        print(f"\nTabla de errores interpretables (unidades reales) guardada en: {orig_path}")

    # --- GRÁFICOS ---
    sns.set_theme(style="whitegrid")
    
    # 1. Comparativa Global (MAE Normalizado)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_summary, x='Model', y='Test_MAE_Norm', palette='viridis')
    plt.title("Error Global Normalizado (MAE) en Test 2024", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("MAE Normalizado (Desviaciones Estándar)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "global_ranking_2024.png")
    plt.close()

    # 2. Comparativa por Dimensión (Normalizado - Para comparar barras)
    if dimension_data_norm:
        df_dims = pd.DataFrame(dimension_data_norm)
        
        # Graficar TODAS las dimensiones
        plt.figure(figsize=(18, 8)) # Aumentamos el ancho para que quepan todas
        sns.barplot(data=df_dims, x='Dimensión', y='MAE_Norm', hue='Model', palette="rocket")
        plt.title("Error Normalizado por Dimensión (Test 2024) - Todas las Variables", fontsize=14)
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
        plt.ylabel("MAE (Std Dev)")
        plt.xticks(rotation=45, ha='right') # Rotamos etiquetas para mejor lectura
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "dimension_comparison_2024.png")
        plt.close()

    print(f"Gráficos comparativos generados en: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
