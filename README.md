# Ultrasound-Speckle-Suppression-with-Structure-Preservation-and-Auto-Tuning
Ultrasound enhancement lab for CAMUS NII/GT: log-domain speckle suppression, structure-aware detail preservation, and automatic grid search to balance CNR, ENL, and edge ratio. Includes reproducible evaluation, visualization panels, best-parameter export, and deployment-friendly modular Python code.
**Ultrasound image enhancement with speckle suppression + structure preservation + auto grid search**

## Why this repo
This project targets a practical trade-off in ultrasound enhancement:
- suppress speckle noise (ENL↑)
- preserve anatomy boundaries (Edge ratio ~ 1.0)
- keep lesion/region contrast (CNR non-decreasing)

## Features
- NII / NII.GZ + GT loading (SimpleITK)
- Multiplicative speckle handling in log-domain
- Lee filtering + structure-guided detail reinjection
- Automatic grid search for balanced parameters
- Case-wise metrics: CNR, ENL, Edge ratio
- Export: compare panels + CSV + best params

## Quick Start
```bash
pip install -r requirements.txt
python scripts/run_grid_search.py
