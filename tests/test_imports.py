"""Test para asegurar que los módulos principales son importables."""

import pytest

def test_project_imports():
    """Asegura que los módulos clave del proyecto se pueden importar."""
    try:
        import bielsia
        import bielsia.cli
        import bielsia.data.graph_construction
        import bielsia.models.gcn
        import bielsia.evaluation.backtesting
        from bielsia.utils import paths
    except ImportError as e:
        pytest.fail(f"Fallo al importar un módulo: {e}")
    assert True
