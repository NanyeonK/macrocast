# Replication Registry

Papers replicated with macrocast, in priority order. Each entry links to
a smoke test (`tests/replication/`) and a demonstration notebook (`examples/`).

---

## Tier 0 — Coulombe Core

| ID | Authors | Year | Journal | Status | Test | Notebook |
|----|---------|------|---------|--------|------|----------|
| C1 | Coulombe, Leroux, Stevanovic, Surprenant | 2021 | IJF | Done | test_clss2021.py | clss2021_replication.ipynb |
| C2 | Coulombe, Leroux, Stevanovic, Surprenant | 2022 | IJF | Planned | test_clss2022.py | clss2022_replication.ipynb |
| C3 | Coulombe, Goulet-Coulombe, Kichian | 2024 | — | Planned | test_coulombe2024.py | coulombe2024_replication.ipynb |

## Tier A — Immediately Replicable

| ID | Authors | Year | Journal | Status | Test | Notebook |
|----|---------|------|---------|--------|------|----------|
| A1 | Medeiros, Vasconcelos, Veiga, Zilberman | 2021 | JBES | Planned | test_medeiros2021.py | medeiros2021_replication.ipynb |
| A2 | Naghi, O'Neill, Zaharieva | 2024 | JAE | Planned | test_naghi2024.py | naghi2024_replication.ipynb |
| A3 | Bae | 2024 | IJF | Planned | test_bae2024.py | bae2024_replication.ipynb |
| A4 | Chu, Qureshi | 2023 | CompEcon | Planned | test_chu2023.py | chu2023_replication.ipynb |
| A5 | Smeekes, Wijler | 2018 | IJF | Planned | test_smeekes2018.py | smeekes2018_replication.ipynb |
| A6 | Stock, Watson | 2002 | JBES | Planned | test_stock2002.py | stock2002_replication.ipynb |
| A7 | Bai, Ng | 2009 | JAE | Planned | test_baing2009.py | baing2009_replication.ipynb |

## Tier B — Partial Replication (Extensions Required)

| ID | Authors | Year | Journal | Status | Gap |
|----|---------|------|---------|--------|-----|
| B1 | Hauzenberger, Huber, Klieber | 2023 | IJF | Planned | Real-time vintages + PyTorch autoencoder |
| B2 | Eraslan, Schroeder | 2023 | IJF | Planned | TVP/SV component needs macrocastR |
| B3 | Gruber, Kastner | 2025 | IJF | Planned | macrocastR BVAR extension needed |

---

## Notes

- **Status** values: `Planned`, `In progress`, `Done`, `Partial`
- Smoke tests use synthetic data where possible for CI speed; notebooks use real FRED-MD/QD data
- Tier B notebooks replicate the linear/factor components only until R extensions are built
