
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

# Configuración
BASE_DIR = Path("/workspace1/gonzalo.fuentes/BielsIA")
REPORTS_DIR = BASE_DIR / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = BASE_DIR / "models"

# Dimensiones (Variables de Y)
DIMENSIONS = [
    "Goals", "xG", "npxG", 
    "Assists", "xAG", "xG+xAG", 
    "Tackles", "Interceptions", "Blocks", "Clearances", 
    "Minutes"
]

def load_data():
    """Carga los datos de 2023 (Test) y 2024 (Eval)."""
    print("Cargando datos...")
    try:
        df_2023 = pd.read_csv(TABLES_DIR / "predictions_2023_full.csv")
        df_2024 = pd.read_csv(TABLES_DIR / "predictions_2024_full.csv")
        
        # Cargar métricas de entrenamiento para referencia
        with open(MODELS_DIR / "transformer_graphormer_metrics.json") as f:
            train_metrics = json.load(f)
            
        return df_2023, df_2024, train_metrics
    except FileNotFoundError as e:
        print(f"Error cargando archivos: {e}")
        return None, None, None

def calculate_metrics(df, year_label):
    """Calcula métricas detalladas por dimensión."""
    metrics = []
    
    for dim in DIMENSIONS:
        pred_col = f"Pred_{dim}"
        true_col = f"True_{dim}"
        
        if true_col not in df.columns:
            continue
            
        y_true = df[true_col]
        y_pred = df[pred_col]
        
        # Métricas
        mae = (y_pred - y_true).abs().mean()
        mse = ((y_pred - y_true) ** 2).mean()
        r2 = r2_score(y_true, y_pred)
        corr, _ = spearmanr(y_pred, y_true)
        
        # Sesgo (Bias): Mean Signed Error
        bias = (y_pred - y_true).mean()
        
        # Baseline (Promedio)
        baseline_mae = (y_true - y_true.mean()).abs().mean()
        improvement = ((baseline_mae - mae) / baseline_mae) * 100 if baseline_mae > 0 else 0
        
        metrics.append({
            "Dimensión": dim,
            "Año": year_label,
            "MAE": mae,
            "R2": r2,
            "Corr": corr,
            "Bias": bias,
            "Mejora_vs_Promedio_%": improvement
        })
        
    return pd.DataFrame(metrics)

def analyze_teams_players(df, year):
    """Analiza mejores/peores equipos y jugadores."""
    # Calcular error promedio por jugador (normalizado por dimensión para no sesgar por volumen)
    # Usamos error relativo simple o MAE directo si no. Usaremos MAE directo por consistencia.
    errors = []
    for dim in DIMENSIONS:
        if f"True_{dim}" in df.columns:
            errors.append((df[f"Pred_{dim}"] - df[f"True_{dim}"]).abs())
    
    df['avg_error'] = pd.concat(errors, axis=1).mean(axis=1)
    
    # Equipos
    team_metrics = df.groupby('team')['avg_error'].agg(['mean', 'count']).sort_values('mean')
    best_teams = team_metrics.head(5)
    worst_teams = team_metrics.tail(5)
    
    # Jugadores
    best_players = df[['player', 'team', 'avg_error']].sort_values('avg_error').head(5)
    worst_players = df[['player', 'team', 'avg_error']].sort_values('avg_error').tail(5)
    
    return best_teams, worst_teams, best_players, worst_players

def plot_all_distributions(df_2023, df_2024):
    """Genera una grilla de gráficos KDE para TODAS las dimensiones."""
    print("Generando gráficos de distribución...")
    
    # Configuración de la grilla (4 filas x 3 columnas para 11 variables)
    n_cols = 3
    n_rows = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
    fig.suptitle('Distribución Predicción vs Realidad: 2023 (Test) vs 2024 (Eval)', fontsize=20)
    
    axes = axes.flatten()
    
    for i, dim in enumerate(DIMENSIONS):
        ax = axes[i]
        
        # Datos 2024 (Prioridad visual)
        if f"True_{dim}" in df_2024.columns:
            sns.kdeplot(df_2024[f"True_{dim}"], ax=ax, color='blue', fill=True, alpha=0.2, label='Real 2024')
            sns.kdeplot(df_2024[f"Pred_{dim}"], ax=ax, color='orange', fill=True, alpha=0.2, label='Pred 2024')
            
            # Datos 2023 (Línea punteada para referencia de estabilidad)
            sns.kdeplot(df_2023[f"True_{dim}"], ax=ax, color='green', linestyle="--", label='Real 2023')
            
            ax.set_title(f"{dim}", fontsize=14)
            ax.set_xlabel("")
            ax.legend(fontsize='small')
            
            # Anotación de MAE 2024
            mae = (df_2024[f"Pred_{dim}"] - df_2024[f"True_{dim}"]).abs().mean()
            ax.text(0.95, 0.95, f'MAE 24: {mae:.2f}', transform=ax.transAxes, ha='right', va='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Ocultar ejes vacíos si sobran
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_path = FIGURES_DIR / "full_distributions_comparison.png"
    plt.savefig(save_path)
    print(f"Gráfico guardado en: {save_path}")

def main():
    df_2023, df_2024, _ = load_data()
    
    if df_2023 is None: return

    # 1. Calcular Métricas
    metrics_23 = calculate_metrics(df_2023, "2023")
    metrics_24 = calculate_metrics(df_2024, "2024")
    
    # Unir para comparación
    comparison = pd.merge(metrics_23, metrics_24, on="Dimensión", suffixes=('_23', '_24'))
    
    # Calcular degradación
    comparison['Crecimiento_Error_%'] = ((comparison['MAE_24'] - comparison['MAE_23']) / comparison['MAE_23']) * 100
    
    # 2. Reporte de Texto
    print("\n" + "="*100)
    print("REPORTE UNIFICADO DE ANÁLISIS: BIELSIA (Transformer-Graphormer)")
    print("="*100)
    
    print("\n1. COMPARATIVA DE RENDIMIENTO POR VARIABLE (2023 vs 2024)")
    print("-" * 100)
    cols_show = ['Dimensión', 'MAE_23', 'MAE_24', 'Crecimiento_Error_%', 'R2_24', 'Corr_24', 'Bias_24']
    print(comparison[cols_show].round(4).to_string(index=False))
    
    print("\n2. DIAGNÓSTICO DE CALIDAD (2024)")
    print("-" * 100)
    print("Variables Fiables (R2 > 0.1 & Crecimiento < 30%):")
    reliable = comparison[(comparison['R2_24'] > 0.1) & (comparison['Crecimiento_Error_%'] < 30)]
    print(reliable['Dimensión'].tolist() if not reliable.empty else "Ninguna")
    
    print("\nVariables Rotas (Crecimiento > 100%):")
    broken = comparison[comparison['Crecimiento_Error_%'] > 100]
    print(broken['Dimensión'].tolist() if not broken.empty else "Ninguna")

    # 3. Análisis de Equipos/Jugadores 2024
    best_t, worst_t, best_p, worst_p = analyze_teams_players(df_2024, 2024)
    
    print("\n3. ANÁLISIS DE CONTEXTO 2024")
    print("-" * 100)
    print("Top 5 Equipos Más Predecibles (Menor Error Promedio):")
    print(best_t[['mean', 'count']].to_string())
    print("\nTop 5 Equipos Más Caóticos:")
    print(worst_t[['mean', 'count']].to_string())
    
    # 4. Generar Gráficos
    plot_all_distributions(df_2023, df_2024)

if __name__ == "__main__":
    main()
