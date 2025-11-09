"""Tests para el framework de backtesting."""

import pytest
import pandas as pd
from bielsia.evaluation.backtesting import run_backtesting_experiment

def test_run_backtesting_experiment_runs():
    """
    Test simple para asegurar que la función de backtesting se ejecuta
    sin errores con datos dummy.
    """
    # Datos dummy
    transfers = pd.DataFrame([{"player_id": "p1", "from_team": "t1", "to_team": "t2"}])
    model_dummy = "DummyModel" # Simula un modelo entrenado

    try:
        result = run_backtesting_experiment(model_dummy, transfers)
        assert isinstance(result, str)
    except Exception as e:
        pytest.fail(f"run_backtesting_experiment falló con datos dummy: {e}")
