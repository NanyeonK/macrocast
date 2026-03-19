#!/bin/bash
# macrocast project initialization script
# Run: bash init_project.sh [project_path]

PROJECT_DIR="${1:-$HOME/projects/macrocast}"

echo "=== Initializing macrocast project at $PROJECT_DIR ==="

# Create directory structure
mkdir -p "$PROJECT_DIR"/{macrocast/{data,pipeline,evaluation,utils},macrocastR/{R,man,tests/testthat},tests/{data,pipeline,evaluation},examples,docs,paper}

cd "$PROJECT_DIR"

# ─── Python package __init__.py files ───
cat > macrocast/__init__.py << 'EOF'
"""macrocast: Decomposing ML Forecast Gains in Macroeconomic Forecasting."""

__version__ = "0.1.0"
EOF

for subdir in data pipeline evaluation utils; do
    touch "macrocast/${subdir}/__init__.py"
done

# ─── Test __init__.py files ───
for subdir in "" data pipeline evaluation; do
    touch "tests/${subdir}/__init__.py" 2>/dev/null || true
done

# ─── pyproject.toml ───
cat > pyproject.toml << 'TOML'
[project]
name = "macrocast"
version = "0.1.0"
description = "Decomposing ML Forecast Gains in Macroeconomic Forecasting"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [
    {name = "Chan"},
]
keywords = ["forecasting", "macroeconomics", "machine-learning", "FRED"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
]

dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "scikit-learn>=1.3",
    "statsmodels>=0.14",
    "scipy>=1.11",
    "requests>=2.31",
    "openpyxl>=3.1",
    "pyyaml>=6.0",
    "joblib>=1.3",
    "tqdm>=4.65",
]

[project.optional-dependencies]
ml = [
    "lightgbm>=4.0",
    "torch>=2.0",
]
viz = [
    "matplotlib>=3.7",
    "seaborn>=0.13",
]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.4",
    "mypy>=1.5",
    "pre-commit>=3.4",
]
docs = [
    "mkdocs-material>=9.4",
    "mkdocstrings[python]>=0.23",
]
all = ["macrocast[ml,viz,dev,docs]"]

[project.scripts]
macrocast = "macrocast.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
target-version = "py310"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "SIM"]
ignore = ["E501"]

[tool.ruff.lint.isort]
known-first-party = ["macrocast"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
TOML

# ─── .gitignore ───
cat > .gitignore << 'GI'
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/
*.egg

# Virtual environments
.venv/
venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# macrocast cache
.macrocast/

# Data (downloaded, not committed)
data_cache/
*.csv.gz

# R
macrocastR/.Rproj.user
macrocastR/src/*.o
macrocastR/src/*.so

# OS
.DS_Store
Thumbs.db

# Secrets
.env
.env.local

# Claude Code local settings
.claude/settings.local.json
GI

# ─── README.md ───
cat > README.md << 'README'
# macrocast

Decomposing ML Forecast Gains in Macroeconomic Forecasting.

An open-source Python (+ R) framework for systematic evaluation of machine learning methods in macroeconomic forecasting, with built-in support for the FRED-MD, FRED-QD, and FRED-SD database ecosystem.

## Installation

```bash
pip install macrocast
# or with all extras
pip install macrocast[all]
```

## Quick Start

```python
import macrocast as mc

# Load and transform FRED-MD
md = mc.load_fred_md()
md_t = md.transform()

# Run decomposition experiment
from macrocast.pipeline import ForecastExperiment
exp = ForecastExperiment(data=md_t, target="INDPRO", horizons=[1, 6, 12])
results = exp.run()

# Analyze
from macrocast.evaluation import decompose
decompose(results).summary()
```

## Citation

If you use macrocast in your research, please cite:

```bibtex
@article{macrocast2026,
  title={macrocast: An Open-Source Framework for Decomposing Machine Learning Gains in Macroeconomic Forecasting},
  author={...},
  journal={International Journal of Forecasting},
  year={2026}
}
```

## License

MIT
README

# ─── Copy CLAUDE.md and .claude/ from setup ───
echo ""
echo "=== Directory structure created ==="
echo ""
echo "Next steps:"
echo "  1. cd $PROJECT_DIR"
echo "  2. Copy CLAUDE.md and .claude/ directory to project root"
echo "  3. git init && git add -A && git commit -m 'Initial project structure'"
echo "  4. uv venv && uv sync"
echo "  5. claude  (start Claude Code)"
echo ""
echo "Useful Claude Code commands:"
echo "  /build data/fred_md     - implement FRED-MD module"
echo "  /test fred_md           - run specific tests"
echo "  /status                 - check project progress"
echo "  /validate-data fred_md  - test data download"
