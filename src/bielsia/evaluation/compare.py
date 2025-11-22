import json
import pandas as pd
from pathlib import Path
import argparse
import sys
import matplotlib.pyplot as plt
import seaborn as sns

def compare_models(models_dir: str, output_file: str):
    path = Path(models_dir)
    if not path.exists():
        print(f"El directorio {models_dir} no existe.")
        return

    metric_files = list(path.rglob("*_metrics.json"))
    
    if not metric_files:
        print(f"No se encontraron archivos _metrics.json en {models_dir}")
        return

    # Definir nombres de las dimensiones (Hardcoded basado en dataset_builder.py)
    dim_names = [
        "Goals", "xG", "npxG", 
        "Assists", "xAG", "xG+xAG", 
        "Tackles", "Interceptions", "Blocks", "Clearances", 
        "Minutes"
    ]

    results = []
    for mf in metric_files:
        try:
            with open(mf, 'r') as f:
                data = json.load(f)
            
            row = {
                "Model": data.get("model_name", mf.stem.replace("_metrics", "")),
                "Test MSE": data.get("test_mse", 0),
                "Test MAE": data.get("test_mae", 0)
            }
            
            # Add per-dim metrics with names
            if "mae_per_dim" in data:
                for i, val in enumerate(data["mae_per_dim"]):
                    col_name = dim_names[i] if i < len(dim_names) else f"Dim_{i}"
                    row[col_name] = val
                    
            results.append(row)
        except Exception as e:
            print(f"Error leyendo {mf}: {e}")
            
    if not results:
        print("No se pudieron extraer resultados.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values("Test MSE")
    
    # Reordenar columnas: Model, Test MSE, Test MAE, luego las dimensiones nombradas
    base_cols = ["Model", "Test MSE", "Test MAE"]
    metric_cols = [c for c in df.columns if c not in base_cols]
    
    # Ordenar metric_cols según el orden de dim_names si es posible
    def sort_key(col):
        if col in dim_names:
            return dim_names.index(col)
        return 999
        
    metric_cols.sort(key=sort_key)
    
    df = df[base_cols + metric_cols]

    print("\n" + "="*50)
    print("COMPARATIVA DE MODELOS (Ordenado por MSE)")
    print("="*50)
    print(df.to_string(index=False))
    print("="*50)
    
    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nTabla guardada en {out_path}")
        
        # Generar gráficos
        plot_comparison(df, out_path.parent.parent / "figures")

def plot_comparison(df: pd.DataFrame, output_dir: Path):
    """Genera gráficos comparativos de los modelos."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Bar Plot MSE
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Model", y="Test MSE", palette="viridis")
    plt.title("Comparación de MSE en Test")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "model_mse_comparison.png")
    plt.close()

    # 2. Bar Plot MAE
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Model", y="Test MAE", palette="magma")
    plt.title("Comparación de MAE en Test")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "model_mae_comparison.png")
    plt.close()
    
    print(f"Gráficos guardados en {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", default="models", help="Directorio con archivos _metrics.json")
    parser.add_argument("--output", default="reports/tables/model_comparison.csv", help="Archivo de salida CSV")
    args = parser.parse_args()
    
    compare_models(args.models_dir, args.output)
