"""Tests para la construcción de grafos."""

import pytest
import pandas as pd
from bielsia.data.graph_construction import build_player_graph

def test_build_player_graph_runs():
    """
    Test simple para asegurar que la función de construcción de grafos
    se ejecuta sin errores con datos dummy.
    """
    # Datos dummy
    players = pd.DataFrame([{"player_id": "p1", "name": "A"}])
    matches = pd.DataFrame([{"match_id": "m1"}])

    try:
        result = build_player_graph(players, matches)
        # El stub devuelve un string, comprobamos eso
        assert isinstance(result, str)
    except Exception as e:
        pytest.fail(f"build_player_graph falló con datos dummy: {e}")
