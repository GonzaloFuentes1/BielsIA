import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import torch
from pathlib import Path
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

import argparse

# Configuración
BASE_DIR = Path("/workspace1/gonzalo.fuentes/BielsIA")
REPORTS_DIR = BASE_DIR / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Dimensiones (Variables de Y)
DIMENSIONS = [
    "Goals", "xG", "npxG", 
    "Assists", "xAG", "xG+xAG", 
    "Tackles", "Interceptions", "Blocks", "Clearances", 
    "Minutes"
]

def load_data(model_name="full"):
    """Carga los datos de 2024 (Eval) y el Scaler."""
    print(f"Cargando datos para modelo: {model_name}...")
    try:
        suffix = "full" if model_name == "full" else model_name
        
        # 1. Cargar Predicciones (Ya están en escala original)
        # Intentar cargar desde la carpeta de benchmark específica del modelo
        benchmark_path = REPORTS_DIR / "benchmark" / model_name / f"predictions_2024_{suffix}.csv"
        
        if benchmark_path.exists():
             df_2024 = pd.read_csv(benchmark_path)
             print(f"Cargado desde: {benchmark_path}")
        else:
            # Fallback a la carpeta tables general (comportamiento anterior)
            fallback_path = TABLES_DIR / f"predictions_2024_{suffix}.csv"
            print(f"No encontrado en benchmark, intentando: {fallback_path}")
            df_2024 = pd.read_csv(fallback_path)
        
        # 2. Cargar Scaler para calcular métricas normalizadas
        scaler_path = PROCESSED_DIR / "y_scaler_params.pt"
        scaler = None
        if scaler_path.exists():
            scaler = torch.load(scaler_path, map_location='cpu')
            print("Scaler cargado exitosamente.")
        else:
            print("Advertencia: No se encontró scaler. No se podrán calcular métricas normalizadas.")

        return df_2024, scaler
    except FileNotFoundError as e:
        print(f"Error cargando archivos: {e}")
        return None, None

def calculate_metrics(df, scaler=None):
    """Calcula métricas en escala Original y Normalizada."""
    metrics = []
    
    # Preparar scaler numpy si existe
    mean_y, std_y = None, None
    if scaler:
        mean_y = scaler['mean'].numpy()
        std_y = scaler['std'].numpy()

    for i, dim in enumerate(DIMENSIONS):
        pred_col = f"Pred_{dim}"
        true_col = f"True_{dim}"
        
        if true_col not in df.columns:
            continue
            
        y_true_orig = df[true_col].values
        y_pred_orig = df[pred_col].values
        
        # --- 1. Métricas Originales (Denormalized) ---
        mae_orig = np.mean(np.abs(y_pred_orig - y_true_orig))
        r2 = r2_score(y_true_orig, y_pred_orig)
        
        # --- 2. Métricas Normalizadas (Standardized) ---
        mae_norm = np.nan
        if mean_y is not None and std_y is not None:
            # Normalizar: (x - mean) / std
            # Nota: Usamos el mean/std correspondiente a la dimensión i
            m = mean_y[i]
            s = std_y[i]
            
            y_true_norm = (y_true_orig - m) / s
            y_pred_norm = (y_pred_orig - m) / s
            
            mae_norm = np.mean(np.abs(y_pred_norm - y_true_norm))
        
        metrics.append({
            "Dimensión": dim,
            "MAE_Original": mae_orig,
            "MAE_Normalizado": mae_norm,
            "R2": r2,
            "Mean_True": np.mean(y_true_orig)
        })
        
    return pd.DataFrame(metrics)

def plot_errors(metrics_df, model_name):
    """Genera gráficos de error Original y Normalizado."""
    print("Generando gráficos de error...")
    
    # 1. Error Normalizado (Comparativo entre dimensiones)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=metrics_df, x='Dimensión', y='MAE_Normalizado', palette='viridis')
    plt.title(f'Error Normalizado (Desviaciones Estándar) - {model_name} 2024', fontsize=14)
    plt.ylabel('MAE (Std Dev)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"error_normalized_2024_{model_name}.png")
    plt.close()

    # 2. Error Original (Interpretación directa)
    # Dividimos en Ofensivo/Defensivo por escalas muy diferentes (ej. Minutos vs Goles)
    offensive_vars = ["Goals", "xG", "npxG", "Assists", "xAG", "xG+xAG"]
    defensive_vars = ["Tackles", "Interceptions", "Blocks", "Clearances"]
    minutes_var = ["Minutes"]

    # Ofensivo
    df_off = metrics_df[metrics_df['Dimensión'].isin(offensive_vars)]
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_off, x='Dimensión', y='MAE_Original', palette='rocket')
    plt.title(f'Error Original (Por 90 min) - Ofensivas - {model_name} 2024', fontsize=14)
    plt.ylabel('MAE (Unidades Reales)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"error_original_offensive_2024_{model_name}.png")
    plt.close()
    
    # Defensivo
    df_def = metrics_df[metrics_df['Dimensión'].isin(defensive_vars)]
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_def, x='Dimensión', y='MAE_Original', palette='mako')
    plt.title(f'Error Original (Total Temporada) - Defensivas - {model_name} 2024', fontsize=14)
    plt.ylabel('MAE (Unidades Reales)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"error_original_defensive_2024_{model_name}.png")
    plt.close()

    print("Gráficos guardados en reports/figures/")

def plot_distributions_by_year(df, year_label, model_name, color_true='blue', color_pred='orange'):
    """Genera gráficos de distribución (KDE) para Real vs Predicho."""
    print(f"Generando distribuciones para {year_label}...")
    
    dist_dir = FIGURES_DIR / "distributions"
    dist_dir.mkdir(exist_ok=True)

    # Configurar grid de subplots
    n_cols = 3
    n_rows = (len(DIMENSIONS) + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 4 * n_rows))
    
    for i, dim in enumerate(DIMENSIONS):
        plt.subplot(n_rows, n_cols, i + 1)
        
        true_col = f"True_{dim}"
        pred_col = f"Pred_{dim}"
        
        if true_col in df.columns and pred_col in df.columns:
            sns.kdeplot(df[true_col], color=color_true, label='Real', fill=True, alpha=0.3)
            sns.kdeplot(df[pred_col], color=color_pred, label='Predicho', fill=True, alpha=0.3)
            
            plt.title(f"{dim}")
            plt.xlabel("Valor")
            plt.ylabel("Densidad")
            if i == 0: # Solo leyenda en el primero para no saturar
                plt.legend()
    
    plt.suptitle(f"Distribución Real vs Predicha ({year_label}) - {model_name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(dist_dir / f"distribution_{year_label}_{model_name}.png")
    plt.close()
    print(f"Gráfico de distribuciones guardado en {dist_dir}")

def main():
    parser = argparse.ArgumentParser(description="Análisis comprensivo de resultados BielsIA")
    parser.add_argument("--model_name", type=str, default="full", help="Nombre del modelo")
    args = parser.parse_args()

    # Limpiar gráficos antiguos de 2023 si existen
    for f in FIGURES_DIR.glob("*2023*.png"):
        f.unlink()
    
    df_2024, scaler = load_data(args.model_name)
    
    if df_2024 is None: return

    # 1. Calcular Métricas
    metrics_df = calculate_metrics(df_2024, scaler)
    
    # Guardar CSV
    output_csv_path = TABLES_DIR / "error_analysis_2024.csv"
    metrics_df.to_csv(output_csv_path, index=False)
    print(f"\n[INFO] Tabla de análisis guardada en: {output_csv_path}")
    
    # 2. Reporte de Texto
    print("\n" + "="*80)
    print(f"REPORTE DE ERROR 2024: {args.model_name}")
    print("="*80)
    print(metrics_df[['Dimensión', 'MAE_Original', 'MAE_Normalizado', 'R2']].round(4).to_string(index=False))
    
    # 3. Generar Gráficos
    plot_errors(metrics_df, args.model_name)
    plot_distributions_by_year(df_2024, "2024", args.model_name, color_true='blue', color_pred='orange')

if __name__ == "__main__":
    main()
