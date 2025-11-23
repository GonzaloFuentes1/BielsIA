import pandas as pd
import numpy as np
from pathlib import Path

# Configuración
BASE_DIR = Path("/workspace1/gonzalo.fuentes/BielsIA")
PREDICTIONS_FILE = BASE_DIR / "reports/benchmark/transformer_graphormer/predictions_2024_transformer_graphormer.csv"

KNOWN_TEAMS = [
    'Arsenal', 'Atlético Madrid', 'Barcelona', 'Bayern Munich', 'Betis', 
    'Chelsea', 'Dortmund', 'Inter', 'Juventus', 'Leverkusen', 'Liverpool', 
    'Manchester City', 'Manchester Utd', 'Milan', 'Napoli', 'Newcastle Utd', 
    'RB Leipzig', 'Real Madrid', 'Roma', 'Sevilla', 'Tottenham', 'Valencia', 'West Ham'
]

def main():
    if not PREDICTIONS_FILE.exists():
        print(f"Error: No se encontró el archivo {PREDICTIONS_FILE}")
        return

    df = pd.read_csv(PREDICTIONS_FILE)
    
    # Identificar columnas de dimensiones
    # Asumimos que las columnas son Pred_[Dim] y True_[Dim]
    pred_cols = [c for c in df.columns if c.startswith('Pred_')]
    dimensions = [c.replace('Pred_', '') for c in pred_cols]
    
    print(f"Analizando {len(df)} jugadores en {len(dimensions)} dimensiones.")
    
    # Calcular error normalizado por dimensión para evitar que 'Minutes' domine
    # Error_Norm = |Pred - True| / Std(True_Population)
    
    player_errors = []
    
    # Pre-calcular desviaciones estándar de las columnas reales para normalizar
    std_devs = {}
    for dim in dimensions:
        true_col = f"True_{dim}"
        if true_col in df.columns:
            std_devs[dim] = df[true_col].std()
            if std_devs[dim] == 0: std_devs[dim] = 1.0 # Evitar división por cero
            
    for idx, row in df.iterrows():
        total_norm_error = 0
        count = 0
        diffs = {}
        
        for dim in dimensions:
            true_col = f"True_{dim}"
            pred_col = f"Pred_{dim}"
            
            if true_col in df.columns:
                true_val = row[true_col]
                pred_val = row[pred_col]
                
                # Error absoluto
                abs_err = abs(pred_val - true_val)
                
                # Normalizar por la desviación estándar de la población
                norm_err = abs_err / std_devs[dim]
                
                total_norm_error += norm_err
                count += 1

                # Store raw difference
                diffs[dim] = pred_val - true_val
        
        avg_error = total_norm_error / count if count > 0 else 0
        
        player_data = {
            'player': row['player'],
            'team': row['team'],
            'error_score': avg_error
        }
        player_data.update(diffs)
        player_errors.append(player_data)
    
    # Crear DataFrame de errores
    df_errors = pd.DataFrame(player_errors)
    
    # Filtrar por equipos conocidos
    df_errors = df_errors[df_errors['team'].isin(KNOWN_TEAMS)]
    print(f"\nFiltrando por {len(KNOWN_TEAMS)} equipos conocidos. Jugadores restantes: {len(df_errors)}")

    # Ordenar
    df_sorted = df_errors.sort_values('error_score')
    
    def print_list(df_subset, title):
        print("\n" + "="*80)
        print(title)
        print("="*80)
        for i, row in df_subset.iterrows():
            print(f"{i+1}. {row['player']} ({row['team']}) - Error Score: {row['error_score']:.4f}")
            # Print diffs in a readable way
            diff_strs = []
            for dim in dimensions:
                if dim in row:
                    val = row[dim]
                    diff_strs.append(f"{dim}: {val:+.2f}")
            print("   " + ", ".join(diff_strs))
            print("-" * 40)

    # Top 20 Mejores
    print_list(df_sorted.head(50).reset_index(drop=True), "TOP 20 MEJORES PREDICCIONES (Menor Error)")
    
    # Top 20 Peores
    print_list(df_sorted.tail(50).iloc[::-1].reset_index(drop=True), "TOP 20 PEORES PREDICCIONES (Mayor Error)")

if __name__ == "__main__":
    main()
