.PHONY: setup test lint demo

# Asume que el entorno virtual se llamará .venv
VENV_PYTHON = ./.venv/bin/python

setup:
	@echo ">>> Creando entorno virtual en .venv..."
	python3 -m venv .venv
	@echo ">>> Instalando dependencias en modo editable..."
	$(VENV_PYTHON) -m pip install -e ".[dev]"
	@echo ">>> Setup completo. Activa el entorno con: source .venv/bin/activate"

test:
	@echo ">>> Ejecutando tests con pytest..."
	$(VENV_PYTHON) -m pytest tests/

lint:
	@echo ">>> Pasando linter (ruff)..."
	$(VENV_PYTHON) -m ruff check .

demo:
	@echo ">>> Ejecutando un pipeline de demostración..."
	$(VENV_PYTHON) -m bielsia.cli --mode demo --config-path configs/demo.yaml
