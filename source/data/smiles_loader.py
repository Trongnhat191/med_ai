"""Load antibiotic SMILES CSV into a name → SMILES dict."""
from pathlib import Path
import pandas as pd


def load_smiles_dict(csv_path: Path) -> dict:
    """
    Read the antibiotic SMILES CSV and return a dict mapping
    antibiotic name (str) → SMILES string (str).
    """
    df = pd.read_csv(csv_path)
    return {
        str(row["antibiotic"]).strip(): str(row["smiles"]).strip()
        for _, row in df.iterrows()
    }
