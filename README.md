## PFAS_BDE.py – Bond Dissociation Energy Analysis Script

This script performs bond dissociation energy (BDE) analysis for per- and polyfluoroalkyl substances (PFAS) and related molecules using quantum chemistry data from a curated PFAS database.
It returns a grid image of all available BDE fragments and parent molecules corresponding to a selected quantum method (e.g., M06-2X, B3LYP, etc.) from the database.

---

## Features
- Filters molecules by PFAS type (protonated or deprotonated)
- Selects quantum method and BDE cleavage type
- Works with both gas-phase and water-phase datasets
- Dynamically determines valid methods from the dataset
- Optionally outputs an image grid of molecular fragments
- Logs execution and input validation steps

---
## Script Usage

You can run the script from the command line with:

python PFAS_BDE.py -ph -p protonated -m M06-2X -t homolytic -o bde_grid.png


Arguments:

| Flag | Description                                  | Required |
| ---- | -------------------------------------------- | -------- |
| `-ph` | phase: `gas` or `water`                    | No (gas)    |
| `-p` | PFAS type: `protonated` or `deprotonated`    | Yes    |
| `-m` | Quantum method name (must be in the dataset) | Yes    |
| `-t` | BDE type: `homolytic` or `heterolytic`       | No (homolytic)     |
| `-o` | Output image file name                       | No  (Out)   |

-------------------
## Requirements

- Python 3.7+
- pandas
- RDKit
- argparse
- PIL / matplotlib (for image output)

Install dependencies via pip:

```bash
pip install pandas rdkit pillow matplotlib
```
Linked Database

The script automatically pulls data from the following hosted repository:
https://github.com/mmarciesky/PFAS_Database

