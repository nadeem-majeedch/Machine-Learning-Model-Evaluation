# Content part 1: Title, TOC, Learning Objectives, Core Principle, Sections 1–7
# Each item is a tuple: ("md", text) or ("code", source)

CELLS_1 = []

CELLS_1.append(("md", r"""# Evaluating Machine Learning Models

### A Complete Teaching Notebook — From Beginner to Expert

**Audience:** Data Science & AI students (undergraduate, postgraduate, and self-study)

**Libraries used:** NumPy · Pandas · Matplotlib · scikit-learn

**How to use this notebook**

1. Read the markdown explanations *before* running the code.
2. Run the code cells **top to bottom** — every code cell is self-contained where practical.
3. After every code cell, read the *"What did we learn?"* note and answer the *"What would this mean in a real-world application?"* question for yourself.

> ### ⭐ The one idea that matters above all others
>
> **Model evaluation is not about finding the model with the highest score.**
> It is about finding the model — and the decision threshold — that best satisfies
> the objectives and constraints of the real-world problem.

[Next: Table of Contents ➡](#toc)"""))

CELLS_1.append(("md", r"""<a id="toc"></a>
## Table of Contents

**Foundations**

- [How the Two Notebooks Fit Together](#roadmap)
- [Learning Objectives](#objectives)
- [The Core Teaching Principle: Business Impact](#core)
- [1. Introduction to Model Evaluation](#sec1)
- [2. Classification vs Regression](#sec2)
- [3. Confusion Matrix](#sec3)
- [4. Accuracy](#sec4)
- [5. Precision](#sec5)
- [6. Recall / Sensitivity](#sec6)
- [7. Specificity](#sec7)

**Classification Metrics in Depth**

- [8. F1 Score](#sec8)
- [9. Precision–Recall Trade-off](#sec9)
- [10. ROC Curve](#sec10)
- [11. AUC](#sec11)
- [12. ROC-AUC vs Precision–Recall](#sec12)
- [13. Probability Calibration](#sec13)
- [14. Classification Report](#sec14)
- [15. Imbalanced Datasets](#sec15)
- [16. Multiclass Evaluation](#sec16)

**Model Selection & Regression**

- [17. Cross-Validation](#sec17)
- [18. Data Leakage](#sec18)
- [19. Time-Series Evaluation](#sec19)
- [20. Regression Metrics](#sec20)

**Putting It All Together**

- [21. End-to-End Case Study: Loan Default Prediction](#sec21)
- [22. Business Decision Case Study](#sec22)
- [23. Cost-Sensitive Evaluation: From Cost Matrix to Decision Rule](#sec23)
- [24. Choosing the Right Metric](#sec24)
- [25. Common Mistakes](#sec25)
- [26. Beginner vs Expert Summary](#sec26)
- [27. Final Cheat Sheet](#sec27)

**Practice & Assessment**

- [28. Practice Exercises](#sec28)
- [Exercise Solutions](#exercise-solutions)
- [29. Final Quiz](#sec29)
- [Quiz Answers](#quiz-answers)
- [Final Decision-Making Framework](#framework)

---
*Navigation: every section ends with [⬅ Previous] / [🏠 Table of Contents] / [Next ➡] links.*"""))

CELLS_1.append(("md", r"""<a id="roadmap"></a>
## How This Notebook Fits with the Simple Edition

This is the **full course** — 29 sections, from beginner to expert. A **Simple edition**
also exists: `ML_Model_Evaluation_Simple.ipynb` covers the same core curriculum in plain
language — 17 short sections, about 1–2 hours, no advanced topics and no widgets.

### Recommended reading paths

| Path | What you do | Best for |
|---|---|---|
| **Beginner** | Simple 1 → 2 → … → 17 straight through | First contact, quick revision, non-experts |
| **Hybrid** | Start in Simple; jump here via the “go deeper” cards at the end of every section | Learning the essentials while exploring depth |
| **Deep** | This notebook, 1 → 2 → … → 29 in order | The complete treatment |

### Deep-only topics — you meet these only here

| Section | Topic |
|---|---|
| 2 | Classification vs Regression |
| 12 | ROC-AUC vs Precision–Recall |
| 13 | Probability Calibration |
| 16 | Multiclass Evaluation |
| 18 | Data Leakage — When Your Model Cheats |
| 19 | Time-Series Evaluation — Never Shuffle the Clock |
| 25 | Common Mistakes When Evaluating Machine Learning Models |
| 26 | Beginner vs Expert Summary |

Every Simple section has a counterpart in this notebook: the 📖 / 📗 cards at the end of
each section link the two notebooks in both directions.

[⬅ Previous](#toc) · [🏠 Table of Contents](#toc) · [Next ➡](#objectives)
"""))

CELLS_1.append(("md", r"""<a id="objectives"></a>
## Learning Objectives

By the end of this notebook you should be able to **explain**, **calculate**, and — most importantly — **justify** the following:

**Concepts**

- Why machine learning models need evaluation
- Training vs validation vs test data
- What a prediction means
- Classification vs regression evaluation
- The Confusion Matrix: TP, TN, FP, FN
- Accuracy, Precision, Recall / Sensitivity, Specificity, F1 Score
- ROC Curve, AUC, Precision–Recall Curve, PR-AUC
- Probability calibration and reliability diagrams
- Classification threshold and probability predictions
- The precision–recall trade-off
- Classification reports and macro / micro / weighted averaging
- Imbalanced datasets and why accuracy can be misleading
- Cross-validation and model stability
- Regression metrics: MAE, MSE, RMSE, R²
- Data leakage and model generalization
- Time-series evaluation: why shuffling leaks, and temporal (forward-chaining) validation
- The business cost of FP and FN
- Cost-sensitive evaluation: deriving the optimal threshold from a cost matrix
- How to choose an appropriate metric and threshold for a given problem

**The most important objective**

> *Model evaluation is not simply about calculating numbers. It is about determining
> whether a model is useful for the problem it is intended to solve.*

**Skills you will practice**

- Computing every metric by hand **and** with scikit-learn
- Interpreting results in a business context
- Choosing metrics, thresholds, and models based on error costs

[⬅ Previous](#roadmap) · [🏠 Table of Contents](#toc) · [Next ➡](#core)"""))

CELLS_1.append(("md", r"""<a id="core"></a>
## The Core Teaching Principle: Business Impact

Throughout this notebook we will repeat one pattern until it becomes automatic.

### The decision chain

```
    Business Problem
          ↓
     Cost of Errors
          ↓
    Important Error Type
          ↓
    Appropriate Metric
          ↓
    Model Evaluation
          ↓
    Threshold Selection
          ↓
    Business Decision
```

Every model makes mistakes. The question is never *"does the model make mistakes?"* —
it always does. The question is **which mistakes are expensive, and how expensive?**

### The three questions of a professional Data Scientist

| Question | Example |
|---|---|
| **Technical:** Which model has the highest metric? | "Model A has accuracy 0.97, Model B has 0.94." |
| **Data Science:** Which metric matters for this problem? | "A missed cancer is worse than a false alarm, so recall matters more than precision here." |
| **Real-World:** What are the consequences if the model is wrong? | "Each missed fraud costs €5,000; each false alert costs €50 in review time." |

A beginner asks the first question. A professional asks all three — and answers them in order.

### The chain that connects everything

```
Model → Prediction → Error → Cost → Metric → Threshold → Decision → Business Impact
```

Whenever you see a metric in this notebook, ask yourself:
**Which link of this chain does this metric represent, and which link does it hide?**

[⬅ Previous](#objectives) · [🏠 Table of Contents](#toc) · [Next ➡](#sec1)"""))

CELLS_1.append(("md", r"""<a id="sec1"></a>
## 1. Introduction to Model Evaluation

### What is model evaluation?

Model evaluation is the process of **measuring how well a model's predictions match reality**
on data the model has **never seen during training**.

Think of an exam. A student who memorizes the textbook word-for-word may score 100% on
homework (data they have already seen) but fail the real exam (new questions). A student
who *understands* the subject may score slightly lower on homework but pass the exam.
We evaluate the second student more highly, because the goal is to perform well on
**new** problems, not to repeat old ones.

Machine learning models work the same way. A model that memorized the training data looks
brilliant during training but collapses in production.

### Why is evaluation necessary?

1. **To choose between models** — which of several candidate models should we deploy?
2. **To detect problems** — is the model overfitting? Underfitting? Leaking information?
3. **To justify deployment** — will this model actually help the business?
4. **To set expectations** — how often will the model be wrong, and what does that cost?
5. **To monitor over time** — does performance stay acceptable after deployment?

### Training, Validation, and Test data

| Dataset | Purpose |
|---|---|
| **Training** | The model *learns* from this data (adjusts its parameters). |
| **Validation** | Used *during development* to compare models and tune hyperparameters. |
| **Test** | Used **once, at the very end**, to estimate how the model performs on truly new data. |

The test set is like the final exam: you may not peek at it while studying, or the grade
becomes meaningless.

### Overfitting, Underfitting, and Generalization

- **Overfitting:** the model memorizes the training data (including its noise) and performs
  poorly on new data. *Symptoms: very high training score, lower test score.*
- **Underfitting:** the model is too simple to capture the pattern at all.
  *Symptoms: low training score AND low test score.*
- **Generalization:** the model performs well on **new** data. This is the actual goal.

### Data leakage

Data leakage happens when information from the test set (or from the future) "leaks" into
the training process. Common causes: scaling or imputing missing values **before** splitting
the data, using features that are only known after the event you are predicting, or
duplicated rows that appear in both training and test sets. Leakage produces unrealistically
good evaluation results that collapse in production. We build two concrete leakage
demonstrations in [Section 18](#sec18) — one with a target-derived feature and one with
preprocessing fitted before the split — and show how a proper pipeline fixes both.

### Why testing on the training data is wrong

A model's training accuracy measures **memorization**, not **learning**. The whole point of
evaluation is to answer: *"will this model work on data it has never seen?"* — and you can
only answer that with data the model has never seen.

> **High accuracy ≠ a good model.** A model can achieve 100% training accuracy by
> memorizing, or 99% accuracy by predicting the majority class for everything. Neither is
> useful. Evaluation must always measure **usefulness**, not just numbers.

Let's demonstrate the difference between memorization and generalization with real code.
"""))

CELLS_1.append(("code", r"""# --- Demo: memorization vs generalization ---
# Build a synthetic binary classification dataset with some label noise.
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, flip_y=0.05, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# A DEEP tree: no depth limit -> it is free to memorize the training data
deep_tree = DecisionTreeClassifier(random_state=42)
deep_tree.fit(X_train, y_train)

# A SHALLOW tree: limited depth -> it must learn a simple, general pattern
shallow_tree = DecisionTreeClassifier(max_depth=2, random_state=42)
shallow_tree.fit(X_train, y_train)

for name, model in [("Deep tree (no depth limit)", deep_tree),
                    ("Shallow tree (max_depth=2)", shallow_tree)]:
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))
    print(f"{name:28s}  training accuracy = {train_acc:.3f}   test accuracy = {test_acc:.3f}")"""))

CELLS_1.append(("md", r"""**What did we learn from this output?**

The deep tree achieves (near-)perfect **training** accuracy but clearly lower **test**
accuracy: it memorized the training data, including its noise. The shallow tree has a
lower training accuracy but a **higher test accuracy**: it learned a general pattern.

**What would this mean in a real-world application?**

If we selected the model by training accuracy alone (a classic beginner mistake), we would
deploy the deep tree — and discover in production that it performs worse than the model we
rejected. Evaluation on a held-out test set is not an optional extra; it is the only honest
measure of usefulness.

---

#### Beginner Perspective

*Always keep a separate test set that the model never sees during training, and report the
test performance — never just the training performance.*

#### Expert Perspective

*Training/test score gaps diagnose overfitting, but a single split is noisy. Experts use
cross-validation (Section 17), watch for data leakage at every preprocessing step, and ask
whether the evaluation data matches the deployment distribution (a model evaluated on
clean data can fail on messy production data).*

---

[🏠 Table of Contents](#toc) · [Next ➡](#sec2)"""))

CELLS_1.append(("md", r"""📗 **Back to basics:** see **Section 1: Introduction to Model
Evaluation** in `ML_Model_Evaluation_Simple.ipynb` — the same ideas in plainer language:
what evaluation is, why it matters, and the training/test split.
"""))

CELLS_1.append(("md", r"""<a id="sec2"></a>
## 2. Classification vs Regression

Before we can evaluate a model, we must know **what kind of prediction it makes**,
because the meaning of "wrong" depends on the type of problem.

### Classification: predicting a *category*

The target is a discrete label.

- **Binary classification** — two classes. *Example: will this customer churn? (yes / no)*
- **Multiclass classification** — more than two unordered classes.
  *Example: which digit is in this image? (0–9)*
- **Multilabel classification** — each item can belong to several classes at once.
  *Example: which tags apply to this photo? (sunset, beach, people)*

In classification, a wrong prediction is **either right or wrong** — there is no
"almost right". (There are *degrees of wrongness* in the business sense — confusing a
benign tumor with a malignant one is worse than confusing two benign subtypes — but the
prediction itself is categorical.)

### Regression: predicting a *number*

The target is a continuous quantity.

*Example: what will this house sell for? What will this customer spend next month?*

In regression, an error has a **magnitude**: predicting 300,000 when the true price is
250,000 is an error of 50,000 — and predicting 150,000 is a *larger* error.

### Why do we need different metrics?

- In classification, errors are **counts** (TP, FP, FN, TN) — every prediction falls into
  one of four boxes, and metrics are built from those counts.
- In regression, errors are **differences between numbers** — metrics are built from the
  size and distribution of those differences.

A metric designed for one problem is meaningless for the other: "precision" makes no sense
for a house-price prediction, and "R²" makes no sense for a spam filter. Choosing the
right metric family is the first evaluation decision you make.

The plot below shows the two problem types side by side: classification produces discrete
groups, regression produces a continuous relationship.
"""))

CELLS_1.append(("code", r"""# --- Classification vs Regression: two different kinds of prediction ---
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# Left: binary classification (discrete labels)
Xc, yc = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0,
                             n_clusters_per_class=1, random_state=7)
axes[0].scatter(Xc[yc == 0, 0], Xc[yc == 0, 1], c="steelblue", s=22, alpha=0.7, label="Class 0")
axes[0].scatter(Xc[yc == 1, 0], Xc[yc == 1, 1], c="coral",    s=22, alpha=0.7, label="Class 1")
axes[0].set_title("Classification: discrete labels")
axes[0].set_xlabel("Feature 1"); axes[0].set_ylabel("Feature 2"); axes[0].legend()

# Right: regression (continuous values)
Xr, yr = make_regression(n_samples=200, n_features=1, noise=20, random_state=7)
axes[1].scatter(Xr, yr, c="seagreen", s=22, alpha=0.7)
axes[1].set_title("Regression: continuous values")
axes[1].set_xlabel("Feature"); axes[1].set_ylabel("Target")

plt.tight_layout()
plt.show()"""))

CELLS_1.append(("md", r"""**What did we learn from this output?**

The left plot shows a classification problem: each point belongs to a discrete class, and
a prediction is right or wrong. The right plot shows a regression problem: there is a
continuous relationship, and predictions can be *close* or *far* from the true value.

**What would this mean in a real-world application?**

If you evaluate a regression model with classification metrics (or vice versa), the numbers
will be meaningless — you would be measuring the wrong thing. The choice of the metric
family must always follow the type of problem.

---

#### Beginner Perspective

*Classification = categories, regression = numbers. Use count-based metrics for one and
distance-based metrics for the other.*

#### Expert Perspective

*Even within one problem type, the "right" metric depends on the cost structure: in
regression, a huge error on one house may matter far more than many small errors (that is
what MSE captures and MAE does not) — the metric encodes the business's tolerance for
large errors.*

---

[⬅ Previous](#sec1) · [🏠 Table of Contents](#toc) · [Next ➡](#sec3)"""))

CELLS_1.append(("md", r"""📗 **Back to basics:** this topic has no separate section in the Simple
notebook — the essential version lives in **Section 1: Introduction to Model
Evaluation** (`ML_Model_Evaluation_Simple.ipynb`), which covers evaluating a model on
unseen data.
"""))

CELLS_1.append(("md", r"""<a id="sec3"></a>
## 3. The Confusion Matrix

The confusion matrix is the **foundation** of classification evaluation. Every important
classification metric is built from its four numbers, so it is worth understanding deeply.

For a **binary** classifier there are exactly four outcomes, depending on whether the
*actual* class was positive and whether the *prediction* was positive:

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | ✅ **TP** — True Positive | ❌ **FN** — False Negative |
| **Actual Negative** | ❌ **FP** — False Positive | ✅ **TN** — True Negative |

### Simple definitions

| Symbol | Name | Meaning in words |
|---|---|---|
| **TP** | True Positive | Correctly predicted positive ("caught it, and it was real") |
| **TN** | True Negative | Correctly predicted negative ("correctly left it alone") |
| **FP** | False Positive | Predicted positive, but it was actually negative ("false alarm") |
| **FN** | False Negative | Predicted negative, but it was actually positive ("missed it") |

### A real-life example: disease detection

Imagine a test that screens 200 patients for a disease. 80 patients are actually sick
(positive) and 120 are healthy (negative).

- The test correctly identifies **70** sick patients → **TP = 70**
- The test misses **10** sick patients (tells them they are healthy) → **FN = 10**
- The test wrongly alarms **15** healthy patients → **FP = 15**
- The test correctly clears **105** healthy patients → **TN = 105**

### Business meaning of errors — the crucial part

| Error | In this scenario | Consequence |
|---|---|---|
| **FP** | Healthy person incorrectly told they are sick | Unnecessary worry, extra tests, possibly unnecessary treatment |
| **FN** | Sick person incorrectly told they are healthy | Disease goes untreated; the patient may get worse — and may even spread it |

**Which error is potentially more dangerous?**

Most medical experts would say **FN is more dangerous here**: a missed disease can
progress, cause serious harm, and in the worst case lead to death. An FP causes distress
and wasted resources, but the patient is still healthy and can recover from a wrong scare.

**Which metric should therefore receive greater attention?**

Because FN is the dangerous error, we should pay attention to **recall / sensitivity**
(the ability to find actual sick patients). This is a *judgment*, not a formula — and the
judgment is driven by the business (here: medical) consequences, not by the numbers
themselves. (For a *screening* test where follow-up is cheap, we may even accept many FP
to catch almost every sick person — the threshold can be tuned to make FN very unlikely.)

Let's build the confusion matrix in Python.
"""))

CELLS_1.append(("code", r"""# --- Confusion Matrix in Python ---
import numpy as np
from sklearn.metrics import confusion_matrix

# 200 patients: first 80 are actually sick (1), next 120 are healthy (0)
y_true = np.array([1] * 80 + [0] * 120)

# Model predictions: 70 sick detected (TP), 10 sick missed (FN),
# 15 healthy alarmed (FP), 105 healthy cleared (TN)
y_pred = np.array([1] * 70 + [0] * 10 + [1] * 15 + [0] * 105)

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix (rows = ACTUAL, columns = PREDICTED):")
print(cm)
print()
print("sklearn layout:  [[TN, FP],")
print("                  [FN, TP]]")
print()

# Extract the four numbers explicitly
tn, fp, fn, tp = cm.ravel()
print(f"TP = {tp}   (sick patients correctly detected)")
print(f"TN = {tn}   (healthy patients correctly cleared)")
print(f"FP = {fp}   (healthy patients wrongly alarmed)")
print(f"FN = {fn}   (sick patients wrongly told they are healthy)")"""))

CELLS_1.append(("code", r"""# --- Confusion Matrix as a heatmap ---
fig, ax = plt.subplots(figsize=(5.2, 4.2))
cell_labels = [["TN = " + str(tn), "FP = " + str(fp)],
               ["FN = " + str(fn), "TP = " + str(tp)]]
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted Negative", "Predicted Positive"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Negative", "Actual Positive"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, cell_labels[i][j], ha="center", va="center", fontsize=13)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix — Disease Detection")
plt.show()"""))

CELLS_1.append(("md", r"""**What did we learn from this output?**

The matrix shows the *whole story* in four numbers: 70 sick patients caught, 10 missed,
15 false alarms, 105 correctly cleared. No single percentage can replace this picture —
this is why experts always look at the confusion matrix before judging a model.

**What would this mean in a real-world application?**

If you were a hospital administrator, the FN = 10 row alone would tell you that 10 sick
people are going home untreated. Depending on the disease, you might decide this is
unacceptable — even if the model "looks good" in other respects — and push for a lower
threshold that catches more sick people, accepting more false alarms in exchange.

---

#### Beginner Perspective

*Memorize the four boxes: TP and TN are correct; FP is a false alarm; FN is a miss. Read
the matrix with actual classes on rows and predicted classes on columns (scikit-learn's
default).*

#### Expert Perspective

*Experts attach costs to each cell: cost(FP) × FP + cost(FN) × FN + ... becomes an
objective function, and the confusion matrix becomes the input to a **business
optimization**, not just a report. They also check whether FN and FP are evenly spread
across subgroups — a model can be excellent overall and systematically wrong for a
particular group of customers.*

---

[⬅ Previous](#sec2) · [🏠 Table of Contents](#toc) · [Next ➡](#sec4)"""))

CELLS_1.append(("md", r"""📗 **Back to basics:** see **Section 2: The Confusion Matrix** in
`ML_Model_Evaluation_Simple.ipynb` — TP/TN/FP/FN in plain words, with the
disease-detection example.
"""))

CELLS_1.append(("md", r"""<a id="sec4"></a>
## 4. Accuracy

### A. Simple Definition

Accuracy answers one question:

> **"How often is the model right overall?"**

It is the fraction of all predictions that are correct.

### B. Intuition

Imagine a teacher marking 200 exam answers. Accuracy is the fraction of answers the model
got right out of everything it was asked. Intuitive, simple, and the first metric everyone
learns — which is exactly why its limitations (below) are so easy to miss.

### C. Formula

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{\text{correct predictions}}{\text{all predictions}}$$

### D. Meaning of Each Component

| Component | Meaning |
|---|---|
| TP + TN | All predictions that were correct |
| TP + TN + FP + FN | All predictions (the whole dataset) |

### E. Small Numerical Example

Using our earlier confusion matrix: TP = 80, TN = 90, FP = 10, FN = 20.

$$\text{Accuracy} = \frac{80 + 90}{80 + 90 + 10 + 20} = \frac{170}{200} = 0.85$$

The model is correct 85% of the time.
"""))

CELLS_1.append(("code", r"""# --- Accuracy: by hand and with scikit-learn ---
from sklearn.metrics import accuracy_score

# Example values: TP=80, TN=90, FP=10, FN=20
TP, TN, FP, FN = 80, 90, 10, 20
acc_manual = (TP + TN) / (TP + TN + FP + FN)
print(f"By hand: Accuracy = ({TP} + {TN}) / ({TP} + {TN} + {FP} + {FN}) = {acc_manual:.3f}")

# The same numbers as label/prediction arrays
# 100 actual positives (80 TP + 20 FN) and 100 actual negatives (90 TN + 10 FP)
y_true = np.array([1] * 100 + [0] * 100)
y_pred = np.array([1] * 80 + [0] * 20 + [1] * 10 + [0] * 90)

print(f"With sklearn: Accuracy = {accuracy_score(y_true, y_pred):.3f}")"""))

CELLS_1.append(("md", r"""### F. Real-Life Beginner Scenario — Student pass/fail

A model predicts whether students pass (1) or fail (0) a course. TP = student predicted
to pass and passed; TN = student predicted to fail and failed; FP = predicted to pass but
failed; FN = predicted to fail but passed. Accuracy tells you the fraction of students the
model judged correctly — but it does **not** tell you whether the model's *mistakes* are
the expensive kind (e.g., wrongly failing a student who would have passed).

### G. Real-Life Expert Scenario — Rare disease screening

A hospital screens for a disease that only **1%** of patients have. Consider a "model"
that simply predicts **"not diseased" for every single patient** — it never looks at the
data. What is its accuracy?

- Of 10,000 patients, 9,900 are healthy → all correctly predicted "not diseased" → TN = 9,900
- 100 are sick → all wrongly predicted "not diseased" → FN = 100, TP = 0, FP = 0

$$\text{Accuracy} = \frac{0 + 9900}{10000} = 0.99$$

**A useless model scores 99% accuracy.** It detected zero sick patients. This is the
classic trap of imbalanced data: when one class dominates, accuracy is easily gamed by
always predicting the majority class.

### H. FP/FN Business Impact

| Error | In this scenario | Consequence |
|---|---|---|
| FP | Healthy person flagged as sick | Anxiety, unnecessary tests, wasted money |
| FN | Sick person cleared as healthy | **Disease progresses untreated** — potentially fatal |

Here FN is dramatically more expensive, yet accuracy treats FP and FN **identically**:
both are simply "wrong". Accuracy has no concept of *which* error matters more.

### I. Metric Selection

Accuracy is appropriate when:

- the classes are reasonably balanced, **and**
- the costs of FP and FN are roughly equal, **and**
- overall correctness is genuinely what the business cares about.

Accuracy is **misleading** when the data is imbalanced or when one error type is much more
expensive than the other — which, in real business problems, is almost always the case.

### J. Threshold Consideration

Accuracy is reported at a fixed threshold (usually 0.5). Moving the threshold changes FP
and FN counts and therefore changes accuracy. A threshold that maximizes accuracy may not
minimize business cost — cost depends on the *types* of errors, not just their total.

### K. When to Use It

Balanced datasets, similar error costs, and problems where "is the prediction correct?"
is a fair question (e.g., image classification with roughly equal class sizes).

### L. When NOT to Use It

- Imbalanced datasets (fraud, disease, intrusion, rare events)
- When FP and FN have very different costs
- When you care about finding a specific class (accuracy hides misses)

### M. Advantages and Limitations

| ✅ Advantages | ❌ Limitations |
|---|---|
| Simple, intuitive, easy to explain to non-technical stakeholders | Treats all errors as equally bad |
| Single number that summarizes overall correctness | Misleading on imbalanced data |
| Easy to compute | Hides how errors are distributed between classes |

Now let's see the trap with real numbers.
"""))

CELLS_1.append(("code", r"""# --- Why accuracy can be completely misleading ---
# 10,000 credit-card transactions: 100 fraudulent (1%), 9,900 legitimate (0%)
y_true = np.array([1] * 100 + [0] * 9900)

# A "model" that ALWAYS predicts "not fraudulent"
y_pred = np.zeros_like(y_true)

from sklearn.metrics import recall_score

acc = accuracy_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
print(f"Accuracy = {acc:.4f}   <-- looks amazing!")
print(f"Recall   = {rec:.4f}   <-- we found ZERO of the 100 frauds")
print()
print("Is this model actually useful?  NO.")
print("It would let every single fraudulent transaction through.")"""))

CELLS_1.append(("md", r"""**What did we learn from this output?**

99% accuracy, zero fraud detected. The model is technically "accurate" and commercially
worthless — worse than worthless, because it creates a false sense of security.

**What would this mean in a real-world application?**

The same trap applies to **fraud detection**, **disease detection**, and **intrusion
detection**: these are all rare-event problems. In all of them, the minority class is the
one that matters, and accuracy actively hides how badly the model handles it.

> **Never evaluate an imbalanced classification problem using accuracy alone.**

(We will see the tools for these problems — precision, recall, F1, PR-AUC — in the next
sections.)

---

#### Beginner Perspective

*Accuracy = "how often is the model right overall". If one class is rare, accuracy can
look great while the model is useless. Always check the confusion matrix too.*

#### Expert Perspective

*Experts replace accuracy with a **cost-weighted objective** whenever FP and FN have
different costs, and they choose evaluation metrics from the business loss function —
not from habit. On imbalanced data, accuracy is rarely even reported.*

---

[⬅ Previous](#sec3) · [🏠 Table of Contents](#toc) · [Next ➡](#sec5)"""))

CELLS_1.append(("md", r"""📗 **Back to basics:** see **Section 3: Accuracy** in
`ML_Model_Evaluation_Simple.ipynb` — the accuracy trap and why accuracy alone is never
enough.
"""))

CELLS_1.append(("md", r"""<a id="sec5"></a>
## 5. Precision

### A. Simple Definition

Precision answers:

> **"When the model says 'positive', how often should I trust it?"**

### B. Intuition

Precision looks only at the **positive predictions the model made** and asks how many of
them were right. High precision means **few false alarms**: when the model raises its hand
and says "positive!", it is almost always correct.

### C. Formula

$$\text{Precision} = \frac{TP}{TP + FP}$$

### D. Meaning of Each Component

| Component | Meaning |
|---|---|
| TP | Actual positives correctly predicted as positive |
| FP | Actual negatives wrongly predicted as positive |
| TP + FP | **All predicted positives** — everything the model flagged |

Precision is computed **only over predicted positives**. Missed positives (FN) do not
appear in this formula — precision never tells you what the model *missed*.

### E. Small Numerical Example

A spam filter labels 55 emails as spam. Of these, 50 really are spam and 5 are legitimate
emails that were wrongly filtered:

TP = 50, FP = 5

$$\text{Precision} = \frac{50}{50 + 5} = \frac{50}{55} \approx 0.909$$

**91% of the emails flagged as spam really were spam.**

### F. Real-Life Beginner Scenario — Spam Email

- FP = a legitimate email (e.g., your boss's message) lands in the spam folder.
- FN = a real spam email lands in your inbox.

High precision means the spam folder almost never contains real mail — which matters,
because **you may never see an email that was wrongly filtered**.

### G. Real-Life Expert Scenario — Bank fraud alerts

A bank's fraud-detection system flags transactions for review. Every alert either blocks a
customer's card or is investigated by a human analyst.

- FP = a legitimate customer's transaction is flagged as fraud.
- FN = real fraud is not flagged.

### H. FP/FN Business Impact

| Error | In this scenario | Consequence |
|---|---|---|
| **FP** | Legitimate customer blocked / investigated | Customer frustration, blocked purchases, lost sales, wasted analyst time, reputational damage |
| **FN** | Real fraud not flagged | Direct financial loss from the fraud itself |

**Which error is more expensive?** It depends. If blocking customers causes churn and each
investigation costs analyst hours, FP is very expensive. If fraud losses are large and
investigations are cheap, FN is worse. Precision is the metric that measures **the cost of
FP directly** — every false alarm reduces precision.

### I. Metric Selection

**Use precision when false positives are particularly costly**, because precision is the
direct measure of the false-alarm rate among predictions: *"of all the positives I act on,
how many will be wrong?"* For example, if every fraud alert triggers a manual
investigation, low precision means analysts waste most of their time on innocent customers.

### J. Threshold Consideration

Raising the classification threshold (requiring higher confidence before saying
"positive") usually **raises precision**: the model flags fewer cases, and the ones it
flags tend to be the most confident — hence more likely to be genuinely positive. The
cost: you will also *miss* more true positives (lower recall). Lowering the threshold has
the opposite effect.

### K. When to Use It

When the cost of an FP is high: spam filtering, expensive manual review queues, medical
treatment decisions (don't treat healthy people unnecessarily), automated blocking
systems.

### L. When NOT to Use It

When a missed positive is the dominant risk. A spam filter with 100% precision but 5%
recall would catch almost no spam — the model would be useless. Precision alone can
look great while the model misses almost everything.

### M. Advantages and Limitations

| ✅ Advantages | ❌ Limitations |
|---|---|
| Directly measures the false-alarm burden | Ignores missed positives (FN) completely |
| Meaningful for imbalanced data | Can be inflated by predicting only a few, very confident positives |
| Easy to explain to business stakeholders | Says nothing about how many positives were found |

Now the Python calculation.
"""))

CELLS_1.append(("code", r"""# --- Precision in Python ---
from sklearn.metrics import precision_score

# Spam detection: 60 real spam emails, 140 legitimate emails
# Predictions: 50 TP, 10 FN, 5 FP, 135 TN
y_true = np.array([1] * 60 + [0] * 140)
y_pred = np.array([1] * 50 + [0] * 10 + [1] * 5 + [0] * 135)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
print("Confusion matrix:\n", cm)
print(f"\nTP={tp}, FP={fp}, FN={fn}, TN={tn}")

# Manual calculation
precision_manual = tp / (tp + fp)
print(f"\nBy hand:  Precision = {tp} / ({tp} + {fp}) = {precision_manual:.3f}")

# scikit-learn
print(f"sklearn:  Precision = {precision_score(y_true, y_pred):.3f}")"""))

CELLS_1.append(("md", r"""**What did we learn from this output?**

The spam filter's precision is 0.909: 91% of the emails it puts in the spam folder really
are spam. The remaining ~9% (5 emails) are legitimate messages wrongly filtered.

**What would this mean in a real-world application?**

If those 5 wrongly-filtered emails contain important business mail, 9% of your flagged
messages are real. Depending on how much a lost email costs, you might demand higher
precision (fewer false alarms) — even if that means some spam slips through. The
"acceptable" precision is a business decision, not a mathematical one.

---

#### Beginner Perspective

*Precision = "of everything the model called positive, how much was right?" High precision
= few false alarms.*

#### Expert Perspective

*Precision must be interpreted relative to the operating threshold, the prevalence of the
positive class, the downstream intervention cost, and the false-positive burden. On rare
events, even a "low" precision (e.g., 0.05) can be a great business result if the
intervention is cheap — precision is never judged in the abstract, only against the cost
of acting on a false alarm.*

---

[⬅ Previous](#sec4) · [🏠 Table of Contents](#toc) · [Next ➡](#sec6)"""))

CELLS_1.append(("md", r"""📗 **Back to basics:** see **Section 4: Precision** in
`ML_Model_Evaluation_Simple.ipynb` — the worked example and the false-alarm business
cases.
"""))

CELLS_1.append(("md", r"""<a id="sec6"></a>
## 6. Recall / Sensitivity

### A. Simple Definition

Recall (also called **sensitivity** or the **true positive rate**) answers:

> **"Of all the actual positives, how many did the model find?"**

### B. Intuition

Recall looks at the **actual positive cases** and asks how many the model managed to
catch. High recall means the model **rarely misses a positive** — good when missing one is
dangerous.

### C. Formula

$$\text{Recall} = \frac{TP}{TP + FN}$$

### D. Meaning of Each Component

| Component | Meaning |
|---|---|
| TP | Actual positives correctly found |
| FN | Actual positives **missed** |
| TP + FN | All actual positives |

Recall is computed **only over actual positives**. False alarms (FP) do not appear in the
formula — recall never tells you how many false alarms the model raises.

### E. Small Numerical Example

A hospital test screens 40 sick patients and correctly identifies 32 of them:

TP = 32, FN = 8

$$\text{Recall} = \frac{32}{32 + 8} = \frac{32}{40} = 0.80$$

The test finds **80% of the sick patients** — and misses 20%.

### F. Real-Life Beginner Scenario — Disease detection

- FN = a sick patient is told they are healthy.
- FP = a healthy patient is told they are sick.

Recall measures the first kind of failure directly: **how many sick patients are being
sent home untreated?**

### G. Real-Life Expert Scenario — Security intrusion detection

A security system monitors a network for intrusions.

- FN = an actual intrusion is **not** detected — an attacker operates unnoticed.
- FP = normal traffic is flagged as an attack (annoying, but often just an alert).

### H. FP/FN Business Impact

| Error | In this scenario | Consequence |
|---|---|---|
| FP | Normal traffic flagged as attack | Analyst time wasted on false alarms |
| FN | Real intrusion missed | **The attacker may steal data, money, or credentials — potentially for months** |

**Which error is more expensive?** For intrusion detection, FN is usually far more
expensive: the *cost of a missed breach* dwarfs the cost of investigating a few false
alarms. Recall is the metric that directly measures the FN problem: *"of all real
attacks, how many did we catch?"*

### I. Metric Selection

**Use recall when false negatives are particularly costly**, because recall is the direct
measure of missed positives: *"of all the positives that exist, how many did we find?"*
If missing a case can be catastrophic (untreated disease, undetected fraud, missed
breach), recall deserves the most attention.

### J. Threshold Consideration

Lowering the threshold makes the model predict "positive" more easily → it finds more true
positives (higher recall) but also raises more false alarms (lower precision). You can
trade recall against precision by moving the threshold — but you cannot maximize both at
once.

### K. When to Use It

Screening for rare but serious events: disease screening, fraud detection, intrusion
detection, safety systems. Whenever the question "did we miss any?" is the critical one.

### L. When NOT to Use It

When false positives are the expensive error and acting on every positive is costly.
A model with 100% recall can achieve it by labeling *everything* positive — useless if
every positive triggers an expensive action.

### M. Advantages and Limitations

| ✅ Advantages | ❌ Limitations |
|---|---|
| Directly measures the missed-positive problem | Ignores false alarms completely |
| Meaningful for imbalanced data | Can be trivially maxed out by predicting everything positive |
| Easy to explain: "how many did we catch?" | Says nothing about precision of the positive predictions |

Now the Python calculation.
"""))

CELLS_1.append(("code", r"""# --- Recall / Sensitivity in Python ---
from sklearn.metrics import recall_score

# Disease detection: 40 sick patients, 160 healthy patients
# Predictions: 32 TP, 8 FN, 4 FP, 156 TN
y_true = np.array([1] * 40 + [0] * 160)
y_pred = np.array([1] * 32 + [0] * 8 + [1] * 4 + [0] * 156)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
print("Confusion matrix:\n", cm)
print(f"\nTP={tp}, FP={fp}, FN={fn}, TN={tn}")

# Manual calculation
recall_manual = tp / (tp + fn)
print(f"\nBy hand:  Recall = {tp} / ({tp} + {fn}) = {recall_manual:.3f}")

# scikit-learn
print(f"sklearn:  Recall = {recall_score(y_true, y_pred):.3f}")

# A model that predicts EVERYTHING as positive has recall = 1.0 but precision near 0
y_pred_all_positive = np.ones_like(y_true)
print(f"\nIf we predicted everything positive: recall = {recall_score(y_true, y_pred_all_positive):.2f}, "
      f"precision = {precision_score(y_true, y_pred_all_positive):.3f}")"""))

CELLS_1.append(("md", r"""**What did we learn from this output?**

The test finds 80% of sick patients — 8 out of 40 are missed. And notice the warning at
the bottom: recall can be pushed to 1.0 by simply calling *everything* positive. A model
with recall 1.0 is not automatically good — it may just be shouting "positive!" at
everything, which is why precision and recall must always be examined together.

**What would this mean in a real-world application?**

For the hospital, recall = 0.80 means: *out of every 5 sick patients, 1 goes home
untreated.* If the disease is dangerous, the hospital may refuse to accept that — and
lower the threshold so recall rises, accepting more false alarms (higher FP) as the price
of catching more real cases. The trade-off is a business decision.

---

#### Beginner Perspective

*Recall = "of all the real positives, how many did we catch?" High recall = few misses.
Remember: a model that always says "positive" has perfect recall but is useless.*

#### Expert Perspective

*Experts treat recall as the *cost of missing the positive class*, and they always read it
together with precision and prevalence. For rare events, "high recall" can still mean very
few positives found in absolute terms; experts convert recall into expected
*number of caught cases* and expected *cost of missed cases* to make it concrete for the
business.*

---

[⬅ Previous](#sec5) · [🏠 Table of Contents](#toc) · [Next ➡](#sec7)"""))

CELLS_1.append(("md", r"""📗 **Back to basics:** see **Section 5: Recall (Sensitivity)** in
`ML_Model_Evaluation_Simple.ipynb` — the worked example and the missed-positive
business cases.
"""))

CELLS_1.append(("md", r"""<a id="sec7"></a>
## 7. Specificity

### A. Simple Definition

Specificity answers:

> **"Of all the actual negatives, how many did the model correctly identify?"**

### B. Intuition

While recall measures how well the model finds **positives**, specificity measures how well
it **leaves negatives alone**. High specificity means the model rarely raises a false
alarm — actual negatives are rarely wrongly flagged.

### C. Formula

$$\text{Specificity} = \frac{TN}{TN + FP}$$

### D. Meaning of Each Component

| Component | Meaning |
|---|---|
| TN | Actual negatives correctly identified as negative |
| FP | Actual negatives wrongly flagged as positive |
| TN + FP | All actual negatives |

### E. Small Numerical Example

Using the same disease-test numbers as the recall example (TN = 156, FP = 4):

$$\text{Specificity} = \frac{156}{156 + 4} = \frac{156}{160} = 0.975$$

The test correctly clears **97.5% of healthy patients** — only 2.5% of healthy people get
a false alarm.

### F. Real-Life Beginner Scenario

A medical test screening healthy people for a disease. Specificity answers:
*"if you are healthy, how likely is the test to tell you so?"* Low specificity means many
healthy people get frightening false-positive results.

### G. Real-Life Expert Scenario

An automated **quality-control** system in a factory inspects products.

- FP = a good product is rejected (wasted product, lost money).
- FN = a defective product ships to the customer (bad reputation, returns, liability).

### H. FP/FN Business Impact

| Error | In this scenario | Consequence |
|---|---|---|
| FP | Good product rejected | Direct loss of product value + wasted inspection time |
| FN | Defective product shipped | Customer complaints, returns, brand damage |

**Which error is more expensive?** If products are expensive to produce and defects are
cheap to replace, FP is the expensive error — and **specificity** is the metric that
directly measures it (how often good products are wrongly rejected).

### Precision vs Specificity — an important distinction

Both involve FP, but they ask **different questions from different viewpoints**:

| Metric | Question | Conditioned on |
|---|---|---|
| **Precision** | Of all *predicted* positives, how many were right? | The model's positive predictions |
| **Specificity** | Of all *actual* negatives, how many were correctly left alone? | The actual negatives |

Precision asks "are my alarms trustworthy?" Specificity asks "how often do I wrongly
alarm the people who are actually fine?" Both drop when FP rises — precision drops
because FP inflates the denominator of predicted positives; specificity drops because FP
inflates the numerator of wrongly-flagged negatives.

### I. Metric Selection

**Use specificity when reducing false alarms on actual negatives is the priority** —
e.g., when you cannot afford to harass healthy people, reject good products, or trigger
expensive follow-up procedures on innocent cases. In medicine, a screening test with low
specificity floods the system with patients who need unnecessary confirmatory tests.

### J. Threshold Consideration

Raising the threshold makes the model more conservative about predicting "positive", which
usually **increases specificity** (fewer negatives are wrongly flagged) at the cost of
**decreasing recall** (more positives are missed). Specificity and recall trade off
directly against each other through the threshold.

### K. When to Use It

Quality control, medical screening (to size the follow-up burden), fraud alerting (to
measure how many innocent customers get bothered), and any setting where "don't bother the
negatives" is a business requirement.

### L. When NOT to Use It

When the dominant risk is missing positives. A test with 100% specificity could achieve it
by predicting "negative" for everything — and detect zero sick patients. Specificity alone
says nothing about finding the positive class.

### M. Advantages and Limitations

| ✅ Advantages | ❌ Limitations |
|---|---|
| Directly measures the false-alarm burden on actual negatives | Ignores missed positives (FN) completely |
| Easy to explain: "how often do healthy people get left alone?" | Trivially maximized by predicting everything negative |
| Useful for sizing downstream costs (tests, investigations) | Must be paired with recall for a complete picture |

Now the Python calculation.
"""))

CELLS_1.append(("code", r"""# --- Specificity in Python ---
# scikit-learn has no specificity_score function; we compute it from the confusion matrix

# Same disease-test example: 32 TP, 8 FN, 4 FP, 156 TN
y_true = np.array([1] * 40 + [0] * 160)
y_pred = np.array([1] * 32 + [0] * 8 + [1] * 4 + [0] * 156)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

specificity = tn / (tn + fp)
print(f"Specificity = TN / (TN + FP) = {tn} / ({tn} + {fp}) = {specificity:.3f}")

# Compare with recall and precision on the same model
recall = tp / (tp + fn)
precision = tp / (tp + fp)
print(f"Recall    = TP / (TP + FN) = {recall:.3f}   (finding sick patients)")
print(f"Precision = TP / (TP + FP) = {precision:.3f}   (trust in positive predictions)")
print(f"Specificity = {specificity:.3f}              (leaving healthy patients alone)")"""))

CELLS_1.append(("md", r"""**What did we learn from this output?**

The same model scores 0.800 recall, 0.889 precision, and 0.975 specificity. These three
numbers answer three different business questions: *do we find the sick?* (recall), *can
we trust our alarms?* (precision), and *do we avoid alarming healthy people?*
(specificity). None of them alone is "the" answer.

**What would this mean in a real-world application?**

If this test were deployed nationwide and 1,000,000 healthy people were screened,
specificity 0.975 means **25,000 healthy people** would receive a false-positive result —
each needing a follow-up test. A hospital administrator evaluating this test would budget
for that follow-up load, or demand a higher-specificity test (accepting lower recall).
The metric that matters is dictated by the operational consequences.

---

#### Beginner Perspective

*Specificity = "how often are actual negatives correctly left alone?" It is recall's
mirror image, measured on the negative class.*

#### Expert Perspective

*Experts pair specificity with recall to fully describe the two error types, and use
specificity to estimate operational load (number of false alarms × cost per alarm). In
multi-class and multi-threshold settings they generalize it via ROC analysis — the false
positive rate (FPR = 1 − specificity) is the x-axis of the ROC curve, next section.*

---

[⬅ Previous](#sec6) · [🏠 Table of Contents](#toc) · [Next ➡](#sec8)"""))
