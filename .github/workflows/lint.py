name: Lint & Code Quality

on:
  push:
    branches: [ main ]
    paths:
      - '**.pyx'
      - '**.py'
      - '.github/workflows/lint.yml'
  pull_request:
    branches: [ main ]
    paths:
      - '**.pyx'
      - '**.py'
      - '.github/workflows/lint.yml'

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      
      - name: Set up Python
        uses: actions/setup-python@v6
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install black isort flake8 mypy pylint
      
      - name: Run Black formatter check
        run: black --check --diff .
        continue-on-error: true
      
      - name: Run isort import sorter check
        run: isort --check-only --diff .
        continue-on-error: true
      
      - name: Run Flake8 linter
        run: flake8 . --count --statistics --show-source
        continue-on-error: true
      
      - name: Run MyPy type checker
        run: mypy . --ignore-missing-imports || true
        continue-on-error: true
      
      - name: Run Pylint
        run: pylint **/*.py --exit-zero
        continue-on-error: true
