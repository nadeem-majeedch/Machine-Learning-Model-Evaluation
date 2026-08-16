"""Assemble ML_Model_Evaluation_Simple.ipynb from the simple content modules."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

from nb_simple_1 import CELLS_S1
from nb_simple_2 import CELLS_S2

IMPORTS_CODE = r'''# --- Setup: imports and reproducibility ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report,
                             roc_curve, roc_auc_score,
                             mean_absolute_error, mean_squared_error, r2_score)

%matplotlib inline
np.random.seed(42)   # reproducibility: the same notebook run twice gives the same output

print("Imports OK")'''


def build():
    cells = []
    cells.append(new_markdown_cell(CELLS_S1[0][1]))          # title
    cells.append(new_code_cell(IMPORTS_CODE))                 # imports
    for block in CELLS_S1[1:] + CELLS_S2:
        kind, text = block
        cells.append(new_code_cell(text) if kind == "code" else new_markdown_cell(text))

    nb = new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    }

    with open("ML_Model_Evaluation_Simple.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    n_md = sum(1 for c in cells if c.cell_type == "markdown")
    n_code = sum(1 for c in cells if c.cell_type == "code")
    print(f"Wrote ML_Model_Evaluation_Simple.ipynb")
    print(f"  total cells : {len(cells)}  ({n_md} markdown, {n_code} code)")
    return nb


if __name__ == "__main__":
    build()
