## PFAS_BDE.py – Bond Dissociation Energy Analysis Script

This script performs bond dissociation energy (BDE) analysis for per- and polyfluoroalkyl substances (PFAS) and PFAS like molecules found in the PFAS database.
It returns in image of all available BDE and molecules found using a particular QM method from the data base. 
---

## Features
- Filters molecules by PFAS type (protonated or deprotonated)
- Selects quantum method and BDE cleavage type
- Works with both gas-phase and water-phase datasets
- Dynamically determines valid methods from the dataset
- Optionally outputs an image grid of molecular fragments
- Logs execution and input validation steps

---

## Requirements

- Python 3.7+
- pandas
- RDKit
- argparse
- PIL / matplotlib (for image output)

Install dependencies via pip:

```bash
pip install pandas rdkit pillow matplotlib
