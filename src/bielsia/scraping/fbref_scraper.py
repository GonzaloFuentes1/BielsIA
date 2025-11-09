"""
Módulo wrapper sobre soccerdata para descargar datos de jugadores desde FBref.

Este módulo abstrae la lógica de scraping para que el resto del proyecto
BielsIA no dependa directamente de la implementación de soccerdata. Proporciona
funciones para descargas de prueba y de múltiples temporadas.
"""
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

import pandas as pd
import soccerdata as sd
from tqdm import tqdm

# Configuración de logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configurar encoding UTF-8 solo para StreamHandler estándar
for handler in logging.root.handlers:
    if isinstance(handler, logging.StreamHandler) and hasattr(handler, 'stream'):
        if hasattr(handler.stream, 'reconfigure'):
            try:
                handler.stream.reconfigure(encoding='utf-8')
            except Exception:
                pass  # Ignorar si no se puede reconfigurar

# --- CONFIGURACIÓN OPTIMIZADA DE DELAYS ---
PLAYER_SEASON_STATS_TYPES = ['standard', 'shooting', 'passing', 'passing_types', 
                               'goal_shot_creation', 'defense', 'possession', 
                               'playing_time', 'misc']
TEAM_SEASON_STATS_TYPES = ['standard', 'shooting', 'passing', 'passing_types',
                            'goal_shot_creation', 'defense', 'possession', 'misc']
MATCH_DATA_TYPES = ['lineup', 'shot_events']
PLAYER_MATCH_STATS_TYPES = ['summary', 'passing', 'passing_types', 'defense', 
                              'possession', 'misc']
DELAY_BETWEEN_STAT_TYPES = 60
DELAY_BETWEEN_DATA_TYPES = 5
DELAY_IN_BATCH_MATCH = 10  # Reducido a 10s con paralelismo
DELAY_BETWEEN_BATCHES = 30
BATCH_SIZE = 20
MAX_RETRIES = 2
MAX_WORKERS = 8  # Número de procesos paralelos


def download_season_schedule(league: str, season: int) -> pd.DataFrame:
    """Descarga el calendario de partidos para una liga y temporada."""
    logging.info(f"Descargando calendario para {league}, temporada {season}...")
    try:
        fbref = sd.FBref(leagues=league, seasons=season)
        schedule_df = fbref.read_schedule()
        if schedule_df.empty:
            logging.warning(f"No se encontró calendario para {league} {season}.")
            return pd.DataFrame()
        logging.info(f"Calendario descargado: {len(schedule_df)} partidos.")
        return schedule_df
    except Exception as e:
        logging.error(f"Error al descargar calendario: {e}")
        return pd.DataFrame()


def download_player_season_stats(league: str, season: int, stat_type: str) -> pd.DataFrame:
    """Descarga estadísticas agregadas de jugadores para una temporada."""
    logging.info(f"Descargando player stats ('{stat_type}') para {league}, temporada {season}...")
    try:
        fbref = sd.FBref(leagues=league, seasons=season)
        stats_df = fbref.read_player_season_stats(stat_type=stat_type)
        if stats_df.empty:
            logging.warning(f"No se encontraron stats '{stat_type}' para {league} {season}.")
            return pd.DataFrame()
        logging.info(f"Stats '{stat_type}' descargadas: {len(stats_df)} registros.")
        return stats_df
    except Exception as e:
        logging.error(f"Error al descargar player stats '{stat_type}': {e}")
        return pd.DataFrame()


def match_data_exists(interim_dir: Path, league: str, season: int, match_id: str) -> bool:
    """Verifica si todos los datos de un partido ya fueron descargados."""
    for data_type in MATCH_DATA_TYPES:
        file_path = interim_dir / "match_data" / data_type / f"{league}_{season}_{match_id}.csv"
        if not file_exists(file_path):
            return False
    return True


def download_single_match_worker(args: Tuple[str, int, str, Path, float]) -> Tuple[str, bool, Dict[str, pd.DataFrame]]:
    """
    Worker function para descargar un partido en un proceso separado.
    Retorna: (match_id, success, match_data)
    """
    league, season, match_id, interim_dir, base_delay = args
    
    # Verificar si ya existe
    if match_data_exists(interim_dir, league, season, match_id):
        return (match_id, True, {})
    
    match_data = {}
    fbref_instance = sd.FBref(leagues=league, seasons=season)
    
    for i, data_type in enumerate(MATCH_DATA_TYPES):
        for attempt in range(MAX_RETRIES):
            try:
                if data_type == 'lineup':
                    df = fbref_instance.read_lineup(match_id=match_id)
                elif data_type == 'shot_events':
                    df = fbref_instance.read_shot_events(match_id=match_id)
                
                if not df.empty:
                    match_data[data_type] = df
                    break
                    
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait_time = 30
                    logging.warning(f"[{match_id}] Error en '{data_type}': {str(e)[:50]}... Reintentando en {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"[{match_id}] Fallo en '{data_type}'")
        
        # Delay entre tipos de datos del mismo partido
        if i < len(MATCH_DATA_TYPES) - 1 and match_data.get(data_type) is not None:
            time.sleep(DELAY_BETWEEN_DATA_TYPES)
    
    # Guardar inmediatamente si se descargó algo
    if match_data:
        for data_type, df in match_data.items():
            data_path = interim_dir / "match_data" / data_type / f"{league}_{season}_{match_id}.csv"
            save_data(df, data_path)
    
    return (match_id, bool(match_data), match_data)


def download_batch_matches_parallel(league: str, season: int, match_ids: List[str], 
                                    interim_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Descarga datos para múltiples partidos usando multiprocesamiento.
    """
    all_matches_data = {}
    
    # Preparar argumentos para cada worker
    tasks = []
    for i, match_id in enumerate(match_ids):
        # Escalonar los delays iniciales para evitar requests simultáneos
        base_delay = i * (DELAY_IN_BATCH_MATCH / MAX_WORKERS)
        tasks.append((league, season, match_id, interim_dir, base_delay))
    
    # Ejecutar en paralelo
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Añadir delay entre submissions para rate limiting
        futures = []
        for task in tasks:
            future = executor.submit(download_single_match_worker, task)
            futures.append(future)
            time.sleep(DELAY_IN_BATCH_MATCH / MAX_WORKERS)  # Delay escalonado
        
        # Recolectar resultados con barra de progreso
        for future in tqdm(as_completed(futures), total=len(futures), desc="Batch", leave=False):
            try:
                match_id, success, match_data = future.result()
                if success and match_data:
                    all_matches_data[match_id] = match_data
            except Exception as e:
                logging.error(f"Error en worker: {e}")
    
    return all_matches_data


def download_batch_matches(league: str, season: int, match_ids: List[str], 
                           interim_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Wrapper que decide si usar descarga paralela o secuencial.
    """
    if len(match_ids) >= 5:  # Usar paralelo solo si hay suficientes partidos
        return download_batch_matches_parallel(league, season, match_ids, interim_dir)
    else:
        # Fallback a descarga secuencial para batches pequeños
        all_matches_data = {}
        for match_id in tqdm(match_ids, desc="Batch", leave=False):
            if match_data_exists(interim_dir, league, season, match_id):
                continue
            
            match_data = download_match_data_safe(league, season, match_id)
            if match_data:
                all_matches_data[match_id] = match_data
                for data_type, df in match_data.items():
                    data_path = interim_dir / "match_data" / data_type / f"{league}_{season}_{match_id}.csv"
                    save_data(df, data_path)
                time.sleep(DELAY_IN_BATCH_MATCH)
        
        return all_matches_data


def download_match_data_safe(league: str, season: int, match_id: str) -> Dict[str, pd.DataFrame]:
    """
    Descarga datos de un partido con reintentos (versión secuencial).
    """
    match_data = {}
    fbref_instance = sd.FBref(leagues=league, seasons=season)
    
    for i, data_type in enumerate(MATCH_DATA_TYPES):
        for attempt in range(MAX_RETRIES):
            try:
                if data_type == 'lineup':
                    df = fbref_instance.read_lineup(match_id=match_id)
                elif data_type == 'shot_events':
                    df = fbref_instance.read_shot_events(match_id=match_id)
                
                if not df.empty:
                    match_data[data_type] = df
                    break
                    
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait_time = 30
                    logging.warning(f"Error en '{data_type}': {str(e)[:50]}... Reintentando en {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Fallo en '{data_type}' para {match_id}")
        
        if i < len(MATCH_DATA_TYPES) - 1 and match_data.get(data_type) is not None:
            time.sleep(DELAY_BETWEEN_DATA_TYPES)
    
    return match_data


def save_data(df: pd.DataFrame, file_path: Path):
    """Guarda un DataFrame en formato CSV, manejando MultiIndex."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if df.empty:
        logging.warning(f"Intentando guardar un DataFrame vacío en {file_path}. Saltando.")
        return

    df_to_save = df.copy()
    if isinstance(df_to_save.columns, pd.MultiIndex):
        df_to_save.columns = ['_'.join(map(str, col)).strip().lower().replace(' ', '_') 
                               for col in df_to_save.columns.values]
    
    df_to_save.reset_index(inplace=True)
    
    df_to_save.to_csv(file_path, index=False)
    logging.debug(f"Guardado: {file_path}")


def file_exists(file_path: Path) -> bool:
    """Verifica si un archivo ya existe y tiene contenido."""
    if file_path.exists():
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                return True
        except Exception:
            pass
    return False


def get_downloaded_matches(interim_dir: Path, league: str, season: int) -> set:
    """Obtiene el set de match_ids que ya fueron descargados."""
    downloaded = set()
    lineup_dir = interim_dir / "match_data" / "lineup"
    
    if not lineup_dir.exists():
        return downloaded
    
    pattern = f"{league}_{season}_*.csv"
    for file in lineup_dir.glob(pattern):
        # Extraer match_id del nombre: ENG-Premier League_2023_abc123.csv
        match_id = file.stem.split('_', 2)[-1]  # Toma la última parte después de los primeros 2 '_'
        downloaded.add(match_id)
    
    return downloaded


def run_smoke_test(interim_dir: Path, league: str, season: int, num_matches: int = 2):
    """Ejecuta una prueba de descarga limitada para N partidos y stats de temporada."""
    logging.info(f"=== SMOKE TEST: {league} temp. {season} ({num_matches} partidos) ===")
    
    # 1. Descargar stats de temporada (más rápido)
    for stat_type in PLAYER_SEASON_STATS_TYPES:
        stats_df = download_player_season_stats(league, season, stat_type)
        if not stats_df.empty:
            file_path = interim_dir / "player_stats" / f"{league}_{season}_{stat_type}.csv"
            save_data(stats_df, file_path)
        time.sleep(DELAY_BETWEEN_STAT_TYPES)

    # 2. Descargar datos de partidos
    schedule_df = download_season_schedule(league, season)
    if schedule_df.empty:
        logging.error("No se pudo descargar el calendario.")
        return

    save_data(schedule_df, interim_dir / f"schedule_{league}_{season}.csv")
    match_ids = schedule_df['game_id'].head(num_matches).tolist()
    
    for match_id in match_ids:
        match_data = download_match_data_safe(league, season, match_id)
        for data_type, df in match_data.items():
            data_path = interim_dir / data_type / f"{league}_{season}_{match_id}.csv"
            save_data(df, data_path)
        time.sleep(30)  # Incrementado de 5 a 30 segundos entre partidos
            
    logging.info("=== SMOKE TEST completado ===")


def run_full_download(interim_dir: Path, league: str, seasons: List[int] = None):
    """
    Descarga completa de TODOS los datos disponibles con procesamiento paralelo.
    """
    if seasons is None:
        seasons = [2023]
    
    logging.info(f"=== DESCARGA COMPLETA: {league} ({len(seasons)} temporadas) ===")
    logging.info(f"Procesamiento paralelo: {MAX_WORKERS} workers")
    logging.info(f"Tipos de datos: Player season stats ({len(PLAYER_SEASON_STATS_TYPES)}), "
                 f"Team season stats ({len(TEAM_SEASON_STATS_TYPES)}), "
                 f"Match data ({len(MATCH_DATA_TYPES)}), "
                 f"Player match stats ({len(PLAYER_MATCH_STATS_TYPES)})")

    for season in seasons:
        logging.info(f"\n{'='*60}")
        logging.info(f"TEMPORADA {season-1}-{season}")
        logging.info(f"{'='*60}")
        
        # ==== 1. STATS DE JUGADORES POR TEMPORADA ====
        logging.info("\n[1/6] Descargando stats de jugadores por temporada...")
        for stat_type in PLAYER_SEASON_STATS_TYPES:
            file_path = interim_dir / "player_season_stats" / f"{league}_{season}_{stat_type}.csv"
            
            if file_exists(file_path):
                logging.info(f"  - {stat_type}: [EXISTE] Saltando...")
                continue
            
            try:
                fbref = sd.FBref(leagues=league, seasons=season)
                stats_df = fbref.read_player_season_stats(stat_type=stat_type)
                if not stats_df.empty:
                    save_data(stats_df, file_path)
                    logging.info(f"  - {stat_type}: {len(stats_df)} registros")
                time.sleep(DELAY_BETWEEN_STAT_TYPES)
            except Exception as e:
                logging.error(f"  - Error en {stat_type}: {e}")

        # ==== 2. STATS DE EQUIPOS POR TEMPORADA ====
        logging.info("\n[2/6] Descargando stats de equipos por temporada...")
        for stat_type in TEAM_SEASON_STATS_TYPES:
            file_path = interim_dir / "team_season_stats" / f"{league}_{season}_{stat_type}.csv"
            
            if file_exists(file_path):
                logging.info(f"  - {stat_type}: [EXISTE] Saltando...")
                continue
            
            try:
                fbref = sd.FBref(leagues=league, seasons=season)
                stats_df = fbref.read_team_season_stats(stat_type=stat_type)
                if not stats_df.empty:
                    save_data(stats_df, file_path)
                    logging.info(f"  - {stat_type}: {len(stats_df)} equipos")
                time.sleep(DELAY_BETWEEN_STAT_TYPES)
            except Exception as e:
                logging.error(f"  - Error en {stat_type}: {e}")

        # ==== 3. CALENDARIO ====
        logging.info("\n[3/6] Descargando calendario de partidos...")
        schedule_path = interim_dir / "schedules" / f"{league}_{season}_schedule.csv"
        
        if file_exists(schedule_path):
            logging.info(f"  - Calendario: [EXISTE] Cargando desde archivo...")
            schedule_df = pd.read_csv(schedule_path)
        else:
            fbref = sd.FBref(leagues=league, seasons=season)
            schedule_df = fbref.read_schedule(force_cache=False)
            if schedule_df.empty:
                logging.warning(f"No se pudo descargar calendario. Saltando temporada {season}.")
                continue
            save_data(schedule_df.reset_index(), schedule_path)
        
        match_ids = schedule_df['game_id'].dropna().tolist()
        total_matches = len(match_ids)
        logging.info(f"  - Calendario: {total_matches} partidos")

        # ==== 4. DATOS DE PARTIDO (LINEUP + SHOTS) - PARALELO ====
        logging.info(f"\n[4/6] Descargando datos de partido (lineup + shots) - MODO PARALELO...")
        
        downloaded_matches = get_downloaded_matches(interim_dir, league, season)
        pending_matches = [mid for mid in match_ids if mid not in downloaded_matches]
        
        logging.info(f"  Total: {total_matches} partidos")
        logging.info(f"  Ya descargados: {len(downloaded_matches)} partidos")
        logging.info(f"  Pendientes: {len(pending_matches)} partidos")
        
        if not pending_matches:
            logging.info("  [CHECKPOINT] Todos los partidos ya fueron descargados. Saltando paso 4.")
        else:
            num_batches = (len(pending_matches) + BATCH_SIZE - 1) // BATCH_SIZE
            logging.info(f"  Procesando en {num_batches} batches con {MAX_WORKERS} workers...")
            
            successful = 0
            for batch_num in range(0, len(pending_matches), BATCH_SIZE):
                batch_match_ids = pending_matches[batch_num:batch_num + BATCH_SIZE]
                
                logging.info(f"\n  [BATCH {batch_num // BATCH_SIZE + 1}/{num_batches}] "
                           f"Partidos {batch_num + 1}-{min(batch_num + BATCH_SIZE, len(pending_matches))}")
                
                batch_data = download_batch_matches(league, season, batch_match_ids, interim_dir)
                
                successful += len(batch_data)
                
                logging.info(f"  [OK] {len(batch_data)}/{len(batch_match_ids)} partidos nuevos")
                
                if batch_num + BATCH_SIZE < len(pending_matches):
                    logging.info(f"  [PAUSA] {DELAY_BETWEEN_BATCHES}s...")
                    time.sleep(DELAY_BETWEEN_BATCHES)
            
            logging.info(f"  Total nuevos descargados: {successful}/{len(pending_matches)} partidos")

        # ==== 5. STATS DE JUGADORES POR PARTIDO ====
        logging.info(f"\n[5/6] Descargando stats de jugadores por partido...")
        for stat_type in PLAYER_MATCH_STATS_TYPES:
            file_path = interim_dir / "player_match_stats" / f"{league}_{season}_{stat_type}.csv"
            
            if file_exists(file_path):
                logging.info(f"  - {stat_type}: [EXISTE] Saltando...")
                continue
            
            try:
                fbref = sd.FBref(leagues=league, seasons=season)
                stats_df = fbref.read_player_match_stats(stat_type=stat_type, force_cache=False)
                if not stats_df.empty:
                    save_data(stats_df.reset_index(), file_path)
                    logging.info(f"  - {stat_type}: {len(stats_df)} registros")
                time.sleep(DELAY_BETWEEN_STAT_TYPES)
            except Exception as e:
                logging.error(f"  - Error en {stat_type}: {e}")

        # ==== 6. EVENTOS DE PARTIDO ====
        logging.info(f"\n[6/6] Descargando eventos de partido...")
        file_path = interim_dir / "match_events" / f"{league}_{season}_events.csv"
        
        if file_exists(file_path):
            logging.info(f"  - Eventos: [EXISTE] Saltando...")
        else:
            try:
                fbref = sd.FBref(leagues=league, seasons=season)
                events_df = fbref.read_events(force_cache=False)
                if not events_df.empty:
                    save_data(events_df.reset_index(), file_path)
                    logging.info(f"  - Eventos: {len(events_df)} registros")
            except Exception as e:
                logging.error(f"  - Error descargando eventos: {e}")
        
        logging.info(f"\n[COMPLETADO] Temporada {season} - Descarga completa finalizada")
    
    logging.info("\n=== DESCARGA MASIVA FINALIZADA ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descargador de datos FBref para BielsIA")
    parser.add_argument(
        "mode",
        choices=["smoke-test", "full-download"],
        help="Modo: 'smoke-test' (prueba rápida) o 'full-download' (descarga completa)"
    )
    parser.add_argument("--league", type=str, default="ENG-Premier League", help="Liga a descargar")
    parser.add_argument("--seasons", type=str, default="2023", help="Temporadas (separadas por coma)")
    parser.add_argument("--season", type=int, default=2024, help="Temporada para smoke-test")
    parser.add_argument("--matches", type=int, default=2, help="Partidos en smoke-test")
    
    args = parser.parse_args()

    ROOT_DIR = Path(__file__).resolve().parents[3]
    INTERIM_DATA_DIR = ROOT_DIR / "data" / "interim"

    if args.mode == "smoke-test":
        run_smoke_test(INTERIM_DATA_DIR, args.league, args.season, args.matches)
    else:
        seasons_list = [int(s.strip()) for s in args.seasons.split(',')]
        run_full_download(INTERIM_DATA_DIR, args.league, seasons=seasons_list)