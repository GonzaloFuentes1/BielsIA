"""
Módulo para parsear, estandarizar y validar los DataFrames de jugadores.

Este módulo toma los datos descargados por soccerdata y aplica transformaciones
y validaciones para asegurar que cumplen con el estándar interno del proyecto.
También genera archivos de muestra.

NOTA: Este parser está diseñado para los datos agregados por temporada.
Se necesitará una lógica de parsing adicional para los nuevos datos por partido.
"""
import logging
from pathlib import Path
import pandas as pd

# Columnas mínimas requeridas después de la estandarización
INTERNAL_MIN_COLS = ['player', 'team', 'pos', 'min_total']

def standardize_player_stats_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza los nombres de las columnas de un DataFrame de soccerdata.

    - Aplana el MultiIndex si existe.
    - Limpia y convierte a minúsculas los nombres de columna.
    - Renombra columnas clave a un estándar interno.

    Args:
        df (pd.DataFrame): DataFrame con columnas originales de soccerdata.

    Returns:
        pd.DataFrame: DataFrame con columnas estandarizadas.
    """
    df_std = df.copy()

    # Aplanar MultiIndex si es necesario
    if isinstance(df_std.columns, pd.MultiIndex):
        logging.info("Aplanando columnas MultiIndex...")
        df_std.columns = ['_'.join(col).strip() for col in df_std.columns.values]

    # Convertir el índice (que contiene player y team) en columnas
    df_std.reset_index(inplace=True)

    logging.debug(f"Columnas antes de la limpieza: {list(df_std.columns)}")

    # Limpiar y convertir a minúsculas
    df_std.columns = [col.lower().strip() for col in df_std.columns]
    
    logging.debug(f"Columnas después de pasar a minúsculas: {list(df_std.columns)}")

    # Mapa para renombrar desde las columnas aplanadas de soccerdata a nuestro estándar
    rename_map = {
        'pos_': 'pos',
        'playing time_min': 'min_total',
        # 'player' y 'team' ya deberían ser correctos tras reset_index
    }
    df_std.rename(columns=rename_map, inplace=True)
    
    logging.info(f"Columnas finales después del renombrado: {list(df_std.columns)}")

    return df_std

def validate_player_stats_df(df: pd.DataFrame) -> None:
    """
    Valida que un DataFrame estandarizado cumpla con los requisitos mínimos.

    Args:
        df (pd.DataFrame): DataFrame estandarizado.

    Raises:
        ValueError: Si alguna de las validaciones falla.
    """
    if df.empty:
        raise ValueError("El DataFrame está vacío después de la estandarización.")

    missing_cols = set(INTERNAL_MIN_COLS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas esenciales en el DataFrame estandarizado: {missing_cols}")

    for col in INTERNAL_MIN_COLS:
        if df[col].isnull().all():
            raise ValueError(f"La columna esencial '{col}' está completamente vacía (null).")
    
    logging.info("Validación del DataFrame estandarizado superada con éxito.")

def save_processed_data(df: pd.DataFrame, processed_dir: Path, league: str, season: int) -> Path:
    """
    Guarda el DataFrame final estandarizado en la carpeta 'processed'.

    Args:
        df (pd.DataFrame): DataFrame estandarizado.
        processed_dir (Path): Directorio para datos procesados.
        league (str): ID de la liga.
        season (int): Año final de la temporada.

    Returns:
        Path: Ruta al archivo guardado.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    season_str = f"{season-1}-{season}"
    
    file_path = processed_dir / f"fbref_{league}_{season_str}_players_std.csv"
    df.to_csv(file_path, index=False)
    logging.info(f"Datos estandarizados guardados en: {file_path}")
    return file_path

def create_sample_from_df(df: pd.DataFrame, sample_dir: Path, league: str, season: int, n_rows: int = 200) -> Path:
    """
    Crea y guarda un archivo de muestra a partir de un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame del cual tomar la muestra.
        sample_dir (Path): Directorio para los datos de muestra.
        league (str): ID de la liga.
        season (int): Año final de la temporada.
        n_rows (int): Número de filas para la muestra.

    Returns:
        Path: Ruta al archivo de muestra guardado.
    """
    sample_dir.mkdir(parents=True, exist_ok=True)
    season_str = f"{season-1}-{season}"
    
    sample_size = min(n_rows, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    sample_path = sample_dir / f"fbref_{league}_{season_str}_players_sample.csv"
    df_sample.to_csv(sample_path, index=False)
    logging.info(f"Archivo de muestra ({sample_size} filas) guardado en: {sample_path}")
    return sample_path

def run_parser_smoke_test(raw_df: pd.DataFrame, processed_dir: Path, sample_dir: Path, league: str, season: int) -> dict:
    """
    Ejecuta una prueba de parsing y estandarización.

    Args:
        raw_df (pd.DataFrame): DataFrame crudo de soccerdata.
        processed_dir (Path): Directorio para datos procesados.
        sample_dir (Path): Directorio para datos de muestra.
        league (str): ID de la liga.
        season (int): Año final de la temporada.

    Returns:
        dict: Un resumen del proceso.
    """
    logging.info(f"--- Iniciando SMOKE TEST de parsing para {league} temp. {season} ---")
    summary = {"success": False}
    try:
        std_df = standardize_player_stats_columns(raw_df)
        validate_player_stats_df(std_df)
        
        proc_path = save_processed_data(std_df, processed_dir, league, season)
        sample_path = create_sample_from_df(std_df, sample_dir, league, season)
        
        summary.update({
            "n_rows": len(std_df),
            "n_cols": len(std_df.columns),
            "proc_path": str(proc_path),
            "sample_path": str(sample_path),
            "success": True
        })
        logging.info("--- SMOKE TEST de parsing finalizado con éxito ---")
        
    except ValueError as e:
        logging.error(f"SMOKE TEST DE PARSING FALLIDO: {e}")
        summary["error"] = str(e)
        
    return summary