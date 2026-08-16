CELLS_S1 = []

CELLS_S1.append(("md", r"""# 📊 Evaluating Machine Learning Models — Simple Edition

**The beginner-friendly version of the full course.**

- This notebook teaches the **essential** evaluation skills: the confusion matrix,
  accuracy, precision, recall, F1, ROC/AUC, and how to pick the right metric for a
  real problem.
- Explanations are short and use plain language, with small worked examples and simple
  code.
- **Want to go deeper?** The full notebook `ML_Model_Evaluation.ipynb` covers every
  topic here in depth, and adds probability calibration, cost-sensitive decision rules,
  data leakage, time-series evaluation, cross-validation details, and interactive
  widgets.

> **The one idea behind this entire notebook:** do not ask only *which model has the
> highest score*. Ask which metric fits the problem, what each type of error costs, and
> what happens when the model is wrong.
"""))

CELLS_S1.append(("md", r"""## How This Notebook Fits with the Full Course

This is the **Simple edition** — the essential evaluation curriculum in plain language:
17 short sections, about 1–2 hours, no advanced topics and no widgets. The full course,
`ML_Model_Evaluation.ipynb`, covers every topic here in depth — and adds much more.

### Recommended reading paths

| Path | What you do | Best for |
|---|---|---|
| **Beginner** | This notebook, 1 → 2 → … → 17 straight through | First contact, quick revision |
| **Hybrid** | Start here; jump to the full course via the “go deeper” cards at the end of every section | Learning while exploring |
| **Deep** | The full course, 1 → 2 → … → 29 in order | The complete treatment |

### Topics that exist only in the full course

- **Section 2** — Classification vs Regression
- **Section 12** — ROC-AUC vs Precision–Recall
- **Section 13** — Probability Calibration
- **Section 16** — Multiclass Evaluation
- **Section 18** — Data Leakage — When Your Model Cheats
- **Section 19** — Time-Series Evaluation — Never Shuffle the Clock
- **Section 25** — Common Mistakes
- **Section 26** — Beginner vs Expert Summary

If a topic here interests you, the “go deeper” card at the end of its section names the
matching section of the full course.
"""))

CELLS_S1.append(("md", r"""## Learning Objectives

After working through this notebook you should be able to:

- Read a confusion matrix and explain TP, TN, FP, FN in plain words
- Compute accuracy, precision, recall, specificity and F1 by hand **and** with
  scikit-learn
- Explain why accuracy can be misleading when classes are imbalanced
- Explain the difference between a false positive and a false negative — and which one
  is more expensive in a given business scenario
- Explain what a classification threshold is and how changing it trades precision
  against recall
- Read an ROC curve and interpret AUC
- Explain how regression models are evaluated (MAE, RMSE, R²)
- Choose a sensible metric **and** threshold for a simple real-world problem

**Before you start:** install the libraries with
`pip install numpy pandas matplotlib scikit-learn`, then run the cell below.
"""))

CELLS_S1.append(("md", r"""<a id="toc"></a>
## Table of Contents

1. [Introduction to Model Evaluation](#sec1)
2. [The Confusion Matrix](#sec2)
3. [Accuracy](#sec3)
4. [Precision](#sec4)
5. [Recall (Sensitivity)](#sec5)
6. [Specificity](#sec6)
7. [F1 Score](#sec7)
8. [The Precision–Recall Trade-off](#sec8)
9. [ROC Curve and AUC](#sec9)
10. [The Classification Report](#sec10)
11. [Imbalanced Datasets](#sec11)
12. [Cross-Validation](#sec12)
13. [Regression Metrics](#sec13)
14. [Mini Case Study: Loan Default Prediction](#sec14)
15. [Cheat Sheet](#sec15)
16. [Practice Exercises](#sec16)
17. [Final Quiz](#sec17)

---
"""))

CELLS_S1.append(("code", r"""# --- Setup: imports and reproducibility ---
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

print("Imports OK")"""))

CELLS_S1.append(("md", r"""<a id="sec1"></a>
## 1. Introduction to Model Evaluation

### What is model evaluation?

**Model evaluation** means measuring how good a model's predictions are on data it has
**not seen before**. That last part is the whole point: a model that memorizes its
training data looks perfect in the lab and fails in the real world.

### Why evaluate at all?

- **To choose between models** — which of several candidates should we deploy?
- **To detect problems** — is the model overfitting (memorizing instead of learning)?
- **To set expectations** — how often will the model be wrong, and what does that cost?

### Training vs test data

We split the data into a **training set** (the model learns from it) and a **test set**
(we measure the model on it). **Never judge a model on data it trained on** — it has
already seen the answers.

- **Overfitting:** the model memorizes the training data (including its noise) and
  performs worse on new data. *Symptom: great training score, worse test score.*
- **Generalization:** the model performs well on **new** data. This is the actual goal.

### The three questions of evaluation

| Question | Example |
|---|---|
| **Technical:** which model has the highest score? | "Model A beats Model B by 2% accuracy." |
| **Data-science:** which metric matters for this problem? | "We care about recall here, not accuracy." |
| **Real-world:** what happens when the model is wrong? | "A missed disease costs lives; a false alarm costs money." |

Professional evaluation asks all three — and the last one usually decides the metric.

The next cell shows why we never trust training accuracy.
"""))

CELLS_S1.append(("code", r"""# --- Why we evaluate on NEW data: a quick demonstration ---
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                           n_redundant=0, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

m = RandomForestClassifier(n_estimators=50, random_state=1).fit(X_train, y_train)

print("Training accuracy:", round(accuracy_score(y_train, m.predict(X_train)), 3))
print("Test accuracy:    ", round(accuracy_score(y_test, m.predict(X_test)), 3))
print()
print("The model memorized the training data (perfect score there)")
print("but is worse on data it has never seen. Only the test score counts.")"""))

CELLS_S1.append(("md", r"""📖 **What to read next — go deeper:** see **Section 1: Introduction to Model
Evaluation** in `ML_Model_Evaluation.ipynb` — the core teaching principle (business
impact), the three questions of evaluation, and overfitting vs generalization in full
depth.
"""))

CELLS_S1.append(("md", r"""<a id="sec2"></a>
## 2. The Confusion Matrix

For a **binary** problem (yes/no, positive/negative), we can compare the model's
predictions with the truth and count four kinds of results:

|  | **Predicted: NO** | **Predicted: YES** |
|---|---|---|
| **Actually: YES** | False Negative (FN) | True Positive (TP) |
| **Actually: NO** | True Negative (TN) | False Positive (FP) |

In plain words:

- **TP (True Positive):** the model said *yes* and was right.
- **TN (True Negative):** the model said *no* and was right.
- **FP (False Positive):** the model said *yes* but was wrong — a **false alarm**.
- **FN (False Negative):** the model said *no* but was wrong — a **missed positive**.

### Example: disease detection

A test predicts whether a patient is sick.

- **FP:** a healthy person is told they are sick. → Stress, extra tests, wasted money.
- **FN:** a sick person is told they are healthy. → The disease goes untreated.

**Which error is more dangerous?** Usually the FN — the sick patient may never get
treatment. But it depends on the situation, and this "which error is worse?" question
is the most important one in evaluation. Everything else in this notebook builds on it.

The next cell builds a small confusion matrix we will reuse for the rest of the
notebook: 200 patients, 40 sick and 160 healthy.
"""))

CELLS_S1.append(("code", r"""# --- A small example we will reuse: 200 patients ---
# Truth: 40 sick (1), 160 healthy (0).
# The model makes 12 false alarms and misses 8 sick patients.
y_true = np.array([1] * 40 + [0] * 160)
y_pred = np.array([1] * 32 + [0] * 8 + [1] * 12 + [0] * 148)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
print("Confusion matrix (rows = truth, columns = prediction):")
print(cm)
print()
print(f"TP = {tp}  (sick and detected)")
print(f"TN = {tn}  (healthy and correctly cleared)")
print(f"FP = {fp}  (healthy, but flagged as sick — false alarm)")
print(f"FN = {fn}  (sick, but told they are healthy — missed!)")"""))

CELLS_S1.append(("code", r"""# --- Visualise the confusion matrix ---
fig, ax = plt.subplots(figsize=(5.5, 4.2))
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted NO", "Predicted YES"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Actually NO", "Actually YES"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=18)
ax.set_title("Confusion matrix — disease detection")
plt.show()"""))

CELLS_S1.append(("md", r"""📖 **What to read next — go deeper:** see **Section 3: The Confusion Matrix** in
`ML_Model_Evaluation.ipynb` — a deeper walk through TP/TN/FP/FN, more examples, and the
error-cost framing that drives the whole course.
"""))

CELLS_S1.append(("md", r"""<a id="sec3"></a>
## 3. Accuracy

### Definition

**Accuracy** answers: *how often is the model right overall?*

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

### Worked example (our 200 patients)

$$\text{Accuracy} = \frac{32 + 148}{200} = \frac{180}{200} = 0.90$$

The model is right 90% of the time. Sounds great… until we look closer.

### The trap: accuracy lies on imbalanced data

Suppose a dataset has **99% healthy** people and **1% sick**. A lazy model predicts
*"healthy"* for everyone. It is wrong about every sick person, yet:

$$\text{Accuracy} = \frac{0 + 198}{200} = 0.99$$

**99% accuracy — and it detected zero sick patients.** Is this model useful? For
disease detection: **absolutely not**. The same trap appears in fraud detection and
intrusion detection, where the interesting cases are rare.

> **Never judge an imbalanced problem by accuracy alone.**
"""))

CELLS_S1.append(("code", r"""# --- Why accuracy can be misleading ---
# 99% healthy, 1% sick. The "model" predicts healthy for everyone.
y_true_rare = np.array([1] * 10 + [0] * 990)
y_pred_all0 = np.zeros(1000, dtype=int)

acc = accuracy_score(y_true_rare, y_pred_all0)
rec = recall_score(y_true_rare, y_pred_all0)
print(f"Accuracy = {acc:.3f}   (looks great!)")
print(f"Recall   = {rec:.3f}   (found ZERO of the 10 sick people)")
print()
print("The model is useless for the problem, even though accuracy is 99%.")"""))

CELLS_S1.append(("md", r"""📖 **What to read next — go deeper:** see **Section 4: Accuracy** in
`ML_Model_Evaluation.ipynb` — the accuracy trap in depth, more worked examples, and when
accuracy is — and is not — the metric you actually need.
"""))

CELLS_S1.append(("md", r"""<a id="sec4"></a>
## 4. Precision

### Definition

**Precision** answers: *of all the times the model said YES, how often was it right?*

$$\text{Precision} = \frac{TP}{TP + FP}$$

### Worked example (our 200 patients)

$$\text{Precision} = \frac{32}{32 + 12} = \frac{32}{44} \approx 0.727$$

Of 44 positive predictions, about 73% were correct; the other 27% were false alarms.

### When precision matters: false alarms are expensive

- **Spam filter:** FP = a real email lands in spam. Customers miss important mail.
- **Bank fraud alerts:** FP = a legitimate customer is blocked or interrogated. They get
  frustrated and take their business elsewhere.
- **Recommender systems:** FP = recommending something the user does not want. Cheap per
  event, but annoying at scale.

**High precision means fewer false alarms.** If FP costs you money or trust, precision
is the metric to watch.

### Python

```python
precision_score(y_true, y_pred)   # 0.727 in our example
```
"""))

CELLS_S1.append(("code", r"""# --- Precision: by hand and with scikit-learn ---
tp, fp, fn, tn = 32, 12, 8, 148
prec_hand = tp / (tp + fp)
prec_sk   = precision_score(y_true, y_pred)

print(f"By hand:       precision = {tp} / ({tp} + {fp}) = {prec_hand:.3f}")
print(f"scikit-learn:  precision = {prec_sk:.3f}")
print()
print("Meaning: when the model flags someone as sick, it is right about 73% of the time.")"""))

CELLS_S1.append(("md", r"""📖 **What to read next — go deeper:** see **Section 5: Precision** in
`ML_Model_Evaluation.ipynb` — more false-alarm business cases, additional worked
examples, and how precision behaves as the threshold changes.
"""))

CELLS_S1.append(("md", r"""<a id="sec5"></a>
## 5. Recall (Sensitivity)

### Definition

**Recall** answers: *of all the truly positive cases, how many did the model find?*

$$\text{Recall} = \frac{TP}{TP + FN}$$

### Worked example (our 200 patients)

$$\text{Recall} = \frac{32}{32 + 8} = \frac{32}{40} = 0.80$$

The model found 80% of the sick patients; it missed 20% of them.

### When recall matters: missed positives are expensive

- **Disease screening:** FN = a sick person is sent home untreated.
- **Security intrusion detection:** FN = an attack goes unnoticed.
- **Fraud detection:** FN = a fraudulent transaction is approved.

**High recall means few missed positives.** If missing a positive case is the dangerous
error, recall is the metric to watch.

### Python

```python
recall_score(y_true, y_pred)   # 0.80 in our example
```
"""))

CELLS_S1.append(("code", r"""# --- Recall: by hand and with scikit-learn ---
tp, fp, fn, tn = 32, 12, 8, 148
rec_hand = tp / (tp + fn)
rec_sk   = recall_score(y_true, y_pred)

print(f"By hand:       recall = {tp} / ({tp} + {fn}) = {rec_hand:.3f}")
print(f"scikit-learn:  recall = {rec_sk:.3f}")
print()
print("Meaning: the model catches 80% of the sick patients (and misses 20%).")"""))

CELLS_S1.append(("md", r"""📖 **What to read next — go deeper:** see **Section 6: Recall / Sensitivity** in
`ML_Model_Evaluation.ipynb` — more missed-positive scenarios, the recall–specificity
relationship, and how recall is used in multi-class settings.
"""))

CELLS_S1.append(("md", r"""<a id="sec6"></a>
## 6. Specificity

### Definition

**Specificity** answers: *of all the truly negative cases, how many did the model
correctly identify?*

$$\text{Specificity} = \frac{TN}{TN + FP}$$

### Worked example (our 200 patients)

$$\text{Specificity} = \frac{148}{148 + 12} = \frac{148}{160} = 0.925$$

The model correctly cleared 92.5% of the healthy patients — only 7.5% got false alarms.

### Precision vs specificity — same idea, different question

| Metric | Question |
|---|---|
| **Precision** | Of the model's *YES* answers, how many are right? (denominator: TP + FP) |
| **Specificity** | Of the *true NO*s, how many did the model recognize? (denominator: TN + FP) |

Precision asks "can I trust a positive?"; specificity asks "how well do I avoid crying
wolf?" When false alarms are expensive (security alerts, spam), both matter — but
specificity looks at the negative class directly.
"""))

CELLS_S1.append(("code", r"""# --- Specificity: by hand ---
tp, fp, fn, tn = 32, 12, 8, 148
spec = tn / (tn + fp)

print(f"Specificity = {tn} / ({tn} + {fp}) = {spec:.3f}")
print("The model correctly clears 92.5% of healthy patients.")"""))

CELLS_S1.append(("md", r"""📖 **What to read next — go deeper:** see **Section 7: Specificity** in
`ML_Model_Evaluation.ipynb` — how specificity and precision see the same errors from
different angles, with more examples and the trade-off discussion.
"""))

CELLS_S1.append(("md", r"""<a id="sec7"></a>
## 7. F1 Score

### Why another metric?

Precision and recall pull in opposite directions. A model that says YES too often gets
high recall but low precision; a cautious model gets high precision but low recall. The
**F1 score** combines both into one number — it is the **harmonic mean**:

$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

The harmonic mean is harsh when one number is small: F1 is only high when **both**
precision and recall are high.

### Examples

| Precision | Recall | F1 | Comment |
|---|---|---|---|
| 0.90 | 0.20 | 0.33 | High precision, terrible recall → low F1 |
| 0.50 | 0.60 | 0.55 | Both mediocre → mediocre F1 |
| 0.80 | 0.85 | 0.82 | Both high → high F1 |

### A warning

F1 treats precision and recall as equally important. If the real problem is much more
sensitive to one of them, the model with the highest F1 is **not** automatically the
best business solution. F1 is a convenient summary — not a substitute for thinking
about error costs.
"""))

CELLS_S1.append(("code", r"""# --- F1: by hand and with scikit-learn ---
prec, rec = 0.727, 0.800
f1_hand = 2 * prec * rec / (prec + rec)
f1_sk   = f1_score(y_true, y_pred)

print(f"By hand:       F1 = {f1_hand:.3f}")
print(f"scikit-learn:  F1 = {f1_sk:.3f}")
print()
print("Example of the warning: for disease screening, missing a patient (FN)")
print("is far worse than a false alarm, so recall matters more than precision —")
print("even if a different threshold gives a higher F1.")"""))

CELLS_S1.append(("md", r"""📖 **What to read next — go deeper:** see **Section 8: F1 Score** in
`ML_Model_Evaluation.ipynb` — the harmonic mean explained, Fβ (weighted F1), and the
cases where F1 quietly misleads you.
"""))

CELLS_S1.append(("md", r"""<a id="sec8"></a>
## 8. The Precision–Recall Trade-off

### What is a threshold?

Most classifiers do not say "YES" or "NO" directly — they output a **probability**, e.g.
"60% chance this transaction is fraud". We then choose a **threshold**: predict fraud if
the probability is above it.

- **Low threshold** (e.g., 0.1): more things get flagged → **higher recall** (fewer
  misses) but **lower precision** (more false alarms).
- **High threshold** (e.g., 0.9): fewer things get flagged → **higher precision** (fewer
  false alarms) but **lower recall** (more misses).

The next cell shows this trade-off on a small fraud-like dataset.
"""))

CELLS_S1.append(("code", r"""# --- How the threshold trades precision against recall ---
X_f, y_f = make_classification(n_samples=1500, n_features=6, n_informative=5,
                               n_redundant=0, weights=[0.95, 0.05], random_state=42)
Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(X_f, y_f, test_size=0.3,
                                              random_state=42, stratify=y_f)

model = LogisticRegression(max_iter=1000).fit(Xf_tr, yf_tr)
probs = model.predict_proba(Xf_te)[:, 1]

thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
print("threshold | precision | recall | flagged cases")
for t in thresholds:
    preds = (probs >= t).astype(int)
    p = precision_score(yf_te, preds)
    r = recall_score(yf_te, preds)
    print(f"   {t:.1f}    |   {p:.3f}   |  {r:.3f}  |  {(preds == 1).sum()}")"""))

CELLS_S1.append(("code", r"""# --- Visualise the trade-off ---
grid = np.arange(0.05, 1.0, 0.02)
prec_grid, rec_grid = [], []
for t in grid:
    preds = (probs >= t).astype(int)
    prec_grid.append(precision_score(yf_te, preds))
    rec_grid.append(recall_score(yf_te, preds))

plt.figure(figsize=(8, 4.5))
plt.plot(grid, prec_grid, label="Precision", color="steelblue")
plt.plot(grid, rec_grid, label="Recall", color="coral")
plt.axvline(0.5, color="gray", linestyle="--", label="Default threshold (0.5)")
plt.xlabel("Classification threshold")
plt.ylabel("Score")
plt.title("Precision and recall vs threshold (fraud detection)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()"""))

CELLS_S1.append(("md", r"""**Read the plot from left to right:**

- Left (low threshold): recall is high, precision is low — we catch most fraud, but we
  also flag many innocent customers.
- Right (high threshold): precision is high, recall is low — few false alarms, but more
  fraud slips through.

**The business decision:** in fraud detection, a low threshold means more fraud caught
*but* more legitimate customers blocked or questioned. A high threshold means fewer
annoyed customers *but* more fraud approved. There is no universally "right" threshold —
it depends on which error costs more. **Choosing a threshold is a decision, not a
default.**
"""))

CELLS_S1.append(("md", r"""📖 **What to read next — go deeper:** see **Section 9: The Precision–Recall
Trade-off** in `ML_Model_Evaluation.ipynb` — threshold selection in depth, cost-based
threshold choice, and the precision–recall curve.
"""))

CELLS_S1.append(("md", r"""<a id="sec9"></a>
## 9. ROC Curve and AUC

### What the ROC curve shows

The **ROC curve** plots the trade-off between two rates as the threshold changes:

- **TPR (True Positive Rate)** = recall = TP / (TP + FN)
- **FPR (False Positive Rate)** = FP / (FP + TN) — the share of negatives wrongly
  flagged.

Every possible threshold is one point on the curve. The **diagonal** is what a random
guesser achieves. A good model arches above the diagonal.

### AUC

**AUC** is the area under the ROC curve — a single number summarizing the whole curve:

- **AUC = 1.0:** perfect separation.
- **AUC = 0.5:** no better than random guessing.
- **AUC < 0.5:** worse than random (something is wrong).

AUC answers: *if I pick a random positive and a random negative, how often does the
model give the positive the higher score?* It measures **ranking**, not accuracy.
"""))

CELLS_S1.append(("code", r"""# --- ROC curve and AUC on the fraud dataset ---
fpr, tpr, _ = roc_curve(yf_te, probs)
auc = roc_auc_score(yf_te, probs)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC = 0.5)")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR = recall)")
plt.title("ROC curve — fraud detection")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print(f"AUC = {auc:.3f}  -> the model ranks fraud above normal transactions")
print("most of the time. AUC does NOT tell you which threshold to deploy.")"""))

CELLS_S1.append(("md", r"""**A quick caveat:** on strongly imbalanced data (e.g., 1% fraud), ROC curves can look
optimistically good even when the model finds almost nothing — because the false
positive rate only counts the huge negative class. For rare events, a precision–recall
curve is often more informative. We meet this again in [Section 11](#sec11), and the
full notebook (`ML_Model_Evaluation.ipynb`) treats ROC vs precision–recall in depth.
"""))
