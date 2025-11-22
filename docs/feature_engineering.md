# Definición de Features (X) y Targets (Y)

Este documento detalla las variables utilizadas actualmente en el modelo y propone nuevas métricas basadas en los datos disponibles.

## 1. Features de Entrada (X) - Estado Actual
Estas son las características que describen a cada jugador (nodo) en cada temporada.

### Fuente: `player_season_stats`
Se fusionan los archivos `standard`, `defense`, `passing`, `possession`, `misc`.

#### Generales
- `age`: Edad.
- `playing_time_mp`: Partidos jugados.
- `playing_time_starts`: Partidos como titular.
- `playing_time_min`: Minutos jugados.
- `playing_time_90s`: 90s jugados.

#### Rendimiento Ofensivo (Standard)
- `performance_gls`: Goles totales.
- `performance_ast`: Asistencias totales.
- `performance_g+a`: Goles + Asistencias.
- `performance_pk`: Penales marcados.
- `expected_xg`: Goles esperados (xG).
- `expected_npxg`: xG sin penales.
- `expected_xag`: Asistencias esperadas (xAG).

#### Métricas por 90 minutos (Standard)
- `per_90_minutes_gls`, `per_90_minutes_ast`, `per_90_minutes_g+a`
- `per_90_minutes_xg`, `per_90_minutes_xag`, `per_90_minutes_npxg`

#### Progresión (Standard)
- `progression_prgc`: Conducciones progresivas.
- `progression_prgp`: Pases progresivos recibidos.
- `progression_prgr`: Pases progresivos realizados (revisar nombre exacto en CSV).

#### Defensivo (Defense)
- `tackles_tkl`: Tackles totales.
- `tackles_tklw`: Tackles ganados.
- `int_`: Intercepciones.
- `clr_`: Despejes.
- `blocks_blocks`: Bloqueos.

#### Pases (Passing)
- `total_cmp%`: Porcentaje de pases completados.
- `total_prgdist`: Distancia progresiva de pases.
- `kp_`: Pases clave.
- `ppa_`: Pases al área penal.
- `prgp_`: Pases progresivos.

#### Posesión (Possession)
- `touches_att_pen`: Toques en área rival.
- `take-ons_succ`: Regates exitosos.
- `carries_prgdist`: Distancia progresiva conducida.
- `carries_prgc`: Conducciones progresivas (count).
- `receiving_prgr`: Pases progresivos recibidos.

#### Misceláneos (Misc)
- `performance_recov`: Recuperaciones de balón.
- `aerial_duels_won%`: % Duelos aéreos ganados.

#### Contexto de Equipo (Team Context) - [NUEVO]
Features agregadas al nodo del jugador para darle contexto sobre el estilo de su equipo.
- `team_possession`: Posesión promedio del equipo.
- `team_goals_for`: Goles a favor del equipo.
- `team_tackles`: Volumen defensivo del equipo.

#### Métricas Avanzadas de Tiro (Shot Events) - [NUEVO]
- `avg_shot_distance`: Distancia media de tiro.
- `npxg_per_shot`: Calidad media de tiro (xG/tiro).
- `sca_dependency`: % de tiros que provienen de asistencia (vs jugada individual).

#### Métricas Derivadas (Derived) - [NUEVO]
- `role_starter_pct`: % de partidos como titular (Starts / MP).
- `player_goals_ratio`: % de goles del equipo anotados por el jugador.

---

## 2. Target de Predicción (Y) - Estado Actual
Lo que el modelo intenta predecir para la **siguiente temporada**.

Actualmente es un vector multidimensional que incluye:

#### Ataque
- `per_90_minutes_gls` (Goles/90)
- `per_90_minutes_xg` (xG/90)
- `per_90_minutes_npxg` (npxG/90)

#### Creación
- `per_90_minutes_ast` (Asistencias/90)
- `per_90_minutes_xag` (xAG/90)
- `per_90_minutes_xg+xag` (Contribución total esperada/90)

#### Defensa
- `tackles_tkl` (Tackles totales)
- `int_` (Intercepciones)
- `blocks_blocks` (Bloqueos)
- `clr_` (Despejes)

#### Participación
- `playing_time_90s` (Tiempo de juego, proxy de relevancia en el equipo)

> **Nota sobre Normalización**: Aunque estas variables tienen escalas muy diferentes (ej. Goles ~0.3 vs Minutos ~20), el modelo aplica internamente una normalización (`StandardScaler`) durante el entrenamiento para aprender todas las dimensiones con igual peso. Las predicciones finales se desnormalizan para ser interpretables en sus unidades originales.

---

## 3. Nuevas Features Implementadas (Sección 3)

### B. Datos de Equipo (`team_season_stats`) - [IMPLEMENTADO]
Se ha agregado contexto del equipo al nodo del jugador para aislar el talento individual del contexto colectivo.
*   **Dominio del Equipo**: `team_possession` (% Posesión promedio).
*   **Potencia Ofensiva**: `team_goals_for` (Goles a favor).
*   **Intensidad Defensiva**: `team_tackles` (Tackles totales del equipo).

### C. Datos de Eventos de Tiro (`match_data/shot_events`) - [IMPLEMENTADO]
Se generan métricas avanzadas agregando los eventos de tiro de toda la temporada:
*   **Selección de Tiro**: `avg_shot_distance` (Promedio de la columna `distance_`).
*   **Calidad de Oportunidades**: `npxg_per_shot` (xG promedio por tiro).
*   **Dependencia de Asistencia**: `sca_dependency` (% de tiros asistidos).

## 4. Arquitectura del Grafo (Hipergrafo) - [NUEVO]

Se ha migrado de un grafo de jugadores conectados por equipo (clique) a una arquitectura de **Hipergrafo Bipartito**.

### Nodos
El tensor de features `x` contiene tres tipos de nodos concatenados:
1.  **Jugadores**: Nodos 0 a N-1. Contienen todas las features estadísticas.
2.  **Equipos (Virtual Nodes)**: Nodos N a N+T-1. Representan a los clubes.
    *   **Actualización**: Ya no son vectores cero. Se calculan como el **centroide (promedio)** de las features de todos sus jugadores en esa temporada. Esto permite que el nodo "Man City" tenga features altas en posesión y goles, y "Sheffield" tenga features altas en defensa/tackles.
3.  **Posiciones (Virtual Nodes)**: Nodos N+T a N+T+P-1. Representan las posiciones (GK, DF, MF, FW).

### Aristas (Hyperconnections)
No existen aristas directas entre jugadores. La conectividad es indirecta a través de los nodos virtuales:
*   **Jugador <-> Equipo**: Cada jugador se conecta a su equipo.
*   **Jugador <-> Posición**: Cada jugador se conecta a su(s) posición(es).

Esta estructura reduce la complejidad de $O(N^2)$ (en cliques de equipo) a $O(N)$, y permite al modelo aprender representaciones explícitas para equipos y posiciones.

## 5. Propuestas Futuras

### A. Ingeniería de Features Históricas (Rolling Windows)
Podemos crear features que resuman el pasado del jugador, no solo la temporada actual.
*   **Promedio móvil (3 años)**: Promedio de Goles/90 en los últimos 3 años.
*   **Tendencia**: (Stats Año T) - (Stats Año T-1). ¿Está mejorando o empeorando?
*   **Consistencia**: Desviación estándar de sus métricas clave en los últimos años.

### D. Datos de Alineaciones (`match_data/lineup`)
*   **Rol en el Equipo**: % de partidos como titular vs suplente.
*   **Minutos en Partidos Clave**: Filtrar partidos contra el "Big 6" y calcular minutos jugados.

### E. Datos de Partido (`player_match_stats`)
Tenemos datos partido a partido. Podríamos calcular métricas de consistencia intra-temporada.
*   **Varianza de rendimiento**: Desviación estándar de sus ratings o stats partido a partido. ¿Es regular o muy irregular?
*   **Rendimiento bajo presión**: Stats en partidos con marcador ajustado (empate o diferencia de 1 gol).

### F. Nuevas Variables para Y (Target)
*   **Valor de Mercado**: Si tuviéramos datos de Transfermarkt, sería el target ideal.
*   **Disponibilidad**: Predecir `playing_time_90s` como proxy de salud física y confianza del entrenador.

### G. Ajustes Técnicos
*   **Normalización de Y**: [IMPLEMENTADO] Se aplica `StandardScaler` (media 0, std 1) a los targets durante el entrenamiento para evitar que variables de gran volumen (Minutos, Tackles) dominen la función de pérdida sobre variables pequeñas (Goles, xG). Se desnormaliza para la evaluación.
