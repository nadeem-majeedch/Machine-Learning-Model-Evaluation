"""Assemble ML_Model_Evaluation.ipynb from the content modules."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

from nb_content_1 import CELLS_1
from nb_content_2 import CELLS_2
from nb_content_3 import CELLS_3
from nb_content_4 import CELLS_4

IMPORTS_CODE = r'''# --- Setup: imports and reproducibility ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report,
                             roc_curve, roc_auc_score,
                             precision_recall_curve, average_precision_score,
                             mean_absolute_error, mean_squared_error, r2_score)

%matplotlib inline
import sklearn

np.random.seed(42)   # reproducibility: the same notebook run twice gives the same output

print("Imports OK")
print("NumPy:", np.__version__, "| Pandas:", pd.__version__, "| scikit-learn:", sklearn.__version__)'''


def build():
    cells = []
    # 1) Title cell (markdown)
    cells.append(new_markdown_cell(CELLS_1[0][1]))
    # 2) Imports (code) — everything else depends on this
    cells.append(new_code_cell(IMPORTS_CODE))
    # 3) Everything else, in order
    for block in CELLS_1[1:] + CELLS_2 + CELLS_3 + CELLS_4:
        kind, text = block
        cells.append(new_code_cell(text) if kind == "code" else new_markdown_cell(text))

    nb = new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    }

    with open("ML_Model_Evaluation.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    n_md = sum(1 for c in cells if c.cell_type == "markdown")
    n_code = sum(1 for c in cells if c.cell_type == "code")
    print(f"Wrote ML_Model_Evaluation.ipynb")
    print(f"  total cells : {len(cells)}  ({n_md} markdown, {n_code} code)")
    return nb


if __name__ == "__main__":
    build()
