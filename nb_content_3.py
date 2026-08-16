# Content part 3: Sections 16–22
# Each item is a tuple: ("md", text) or ("code", source)

CELLS_3 = []

CELLS_3.append(("md", r"""📗 **Back to basics:** see **Section 11: Imbalanced Datasets** in
`ML_Model_Evaluation_Simple.ipynb` — the 99% / 1% example and why accuracy is useless
there.
"""))

CELLS_3.append(("md", r"""<a id="sec16"></a>
## 16. Multiclass Evaluation

Everything so far was **binary** (two classes). Real problems often have three or more
classes — digits 0–9, disease types A/B/C, sentiment positive/neutral/negative. The
confusion matrix generalizes naturally: it becomes a **C × C** table where the diagonal is
"correct" and everything off-diagonal is some kind of confusion between classes.

### Per-class metrics

For each class, we can define a **one-vs-rest** view: treat that class as "positive" and
everything else as "negative". Then precision, recall, and F1 can be computed for
*each* class:

- **Precision for class k** = correct predictions of class k / all predictions of class k
- **Recall for class k** = correct predictions of class k / all actual members of class k

### Averaging strategies

| Average | How it works | Effect |
|---|---|---|
| **Macro** | Average each class's score **equally** | Small classes count as much as big ones — fair, but can be dragged down by a rare hard class |
| **Micro** | Pool all predictions into one big confusion matrix, then compute | For multiclass, **equals accuracy**; dominated by the biggest class |
| **Weighted** | Average each class's score, **weighted by support** | Bigger classes dominate — closer to "how it will feel on average" |

### Why per-class business priorities matter

Averaging hides class-specific stakes. In a medical diagnostic system, "confusing a
benign tumor with a malignant one" and "confusing two benign subtypes" are both errors in
the same cell count — but one is life-threatening and the other is harmless. **Always
inspect the per-class rows for the classes whose errors are expensive**, and never let a
good-looking average hide a dangerous class.
"""))

CELLS_3.append(("code", r"""# --- Multiclass evaluation: 3 classes ---
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# Synthetic 3-class problem (Classes A, B, C)
X, y = make_classification(n_samples=900, n_features=10, n_informative=8, n_redundant=0,
                           n_classes=3, n_clusters_per_class=1, random_state=15)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=15, stratify=y)

clf = RandomForestClassifier(n_estimators=100, random_state=15).fit(X_train, y_train)
y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix (rows = actual, columns = predicted):")
print(cm)

# Heatmap
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Predicted A", "Predicted B", "Predicted C"])
ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["Actual A", "Actual B", "Actual C"])
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Multiclass Confusion Matrix")
plt.show()

print(classification_report(y_test, y_pred, target_names=["Class A", "Class B", "Class C"]))"""))

CELLS_3.append(("code", r"""# --- Macro, micro, and weighted averages computed by hand ---
cm = confusion_matrix(y_test, y_pred)

# Per-class precision and recall from the matrix
per_class_precision = cm.diagonal() / cm.sum(axis=0)   # correct / all predicted as k
per_class_recall    = cm.diagonal() / cm.sum(axis=1)   # correct / all actual k
per_class_f1 = 2 * per_class_precision * per_class_recall / (per_class_precision + per_class_recall)

print("Per-class metrics:")
for i, name in enumerate(["Class A", "Class B", "Class C"]):
    print(f"  {name}: precision = {per_class_precision[i]:.3f}, "
          f"recall = {per_class_recall[i]:.3f}, f1 = {per_class_f1[i]:.3f}")

macro_f1 = per_class_f1.mean()
print(f"\nMacro F1 (unweighted mean of per-class F1): {macro_f1:.3f}")

supports = cm.sum(axis=1)
weighted_f1 = (per_class_f1 * supports).sum() / supports.sum()
print(f"Weighted F1 (weighted by support): {weighted_f1:.3f}")

# Micro F1 for multiclass == accuracy (pool everything into one table)
print(f"Micro F1 (pooled) == accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Compare with scikit-learn's values
from sklearn.metrics import f1_score
print(f"sklearn macro F1    = {f1_score(y_test, y_pred, average='macro'):.3f}")
print(f"sklearn weighted F1 = {f1_score(y_test, y_pred, average='weighted'):.3f}")
print(f"sklearn micro F1    = {f1_score(y_test, y_pred, average='micro'):.3f}")"""))

CELLS_3.append(("md", r"""**What did we learn from this output?**

The confusion matrix shows exactly which classes get confused with which. The per-class
metrics reveal that some classes are handled better than others, and the three averaging
strategies give three different summaries: macro treats every class equally, weighted
favors larger classes, micro equals accuracy.

**What would this mean in a real-world application?**

If Class C were a dangerous disease and Class A a harmless condition, an excellent
overall F1 would be irrelevant if the recall for Class C were low — patients with the
dangerous disease would be misclassified. **Business priorities are per-class.** Adjust
the model, the threshold (per class, if the model outputs per-class probabilities), or
the cost matrix so the expensive confusions are minimized — even if the macro average
suffers slightly.

---

#### Beginner Perspective

*For multiclass: check the diagonal of the confusion matrix, read per-class precision /
recall / F1, and know the difference between macro (classes equal) and weighted (classes
weighted by size) averages.*

#### Expert Perspective

*Experts define a *cost matrix* C[i][j] = cost of predicting class j when the truth is
class i, minimize expected cost rather than error rate, and for ordinal or hierarchical
problems choose confusion-aware metrics (e.g., they might penalize "far" confusions more
than "near" ones). They also verify per-class support is large enough for stable
estimates.*

---

[⬅ Previous](#sec15) · [🏠 Table of Contents](#toc) · [Next ➡](#sec17)"""))

CELLS_3.append(("md", r"""📗 **Back to basics:** only here in depth — the nearest basic material is
**Section 10: The Classification Report** in `ML_Model_Evaluation_Simple.ipynb`, which
shows the per-class rows that multiclass evaluation builds on.
"""))

CELLS_3.append(("md", r"""<a id="sec17"></a>
## 17. Cross-Validation

### Why is one train/test split not enough?

A single split is a **single roll of the dice**. The test set may happen to be easy or
hard by chance; a model selected on one lucky split can be much worse on the next one.
The numbers you compute are estimates with noise, and you should know how much noise.

### K-Fold Cross-Validation

The data is split into **K equal parts (folds)**. The model is trained K times — each
time on K−1 folds and evaluated on the remaining fold. Every sample is used for testing
exactly once. The final score is the **mean over the K folds**, and the spread (standard
deviation) tells you how stable the model is.

### Stratified K-Fold

Plain K-Fold can accidentally produce folds with very different class proportions — a
disaster for imbalanced data. **Stratified** K-Fold shuffles so that **each fold keeps
the same class proportions as the full dataset**. For classification, always prefer the
stratified version.

### Why this matters for model selection

- Cross-validation gives you a **distribution of performance**, not one number.
- Model A: mean 0.90 ± 0.01. Model B: mean 0.89 ± 0.03. A single lucky split might pick
  B; the honest comparison picks A as both better and more stable.
- You can also see overfitting risk: if training-fold scores are far above
  validation-fold scores, the model is memorizing.
"""))

CELLS_3.append(("code", r"""# --- K-Fold vs Stratified K-Fold ---
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression

# Imbalanced data (10% positive) — the right kind of problem to show the difference
X, y = make_classification(n_samples=1000, n_features=10, n_informative=6,
                           weights=[0.9, 0.1], random_state=11)

model = LogisticRegression(max_iter=1000, random_state=11)

# Plain K-Fold: folds may end up with very different class proportions
kf = KFold(n_splits=5, shuffle=True, random_state=11)
acc_plain = cross_val_score(model, X, y, cv=kf, scoring="accuracy")

# Stratified K-Fold: each fold preserves the 90/10 split
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
acc_strat = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
f1_strat = cross_val_score(model, X, y, cv=skf, scoring="f1")

print("KFold accuracy:         ", np.round(acc_plain, 3), " mean =", round(acc_plain.mean(), 3))
print("StratifiedKFold acc:    ", np.round(acc_strat, 3), " mean =", round(acc_strat.mean(), 3))
print("StratifiedKFold F1:     ", np.round(f1_strat, 3),  " mean =", round(f1_strat.mean(), 3))
print()
print("Report the score as mean ± std, e.g.  F1 = "
      f"{f1_strat.mean():.3f} ± {f1_strat.std():.3f}")"""))

CELLS_3.append(("code", r"""# --- How noisy is a single train/test split? ---
# Train the SAME model on 20 different random splits and look at the spread.
accs = []
for seed in range(20):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)
    m = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    accs.append(accuracy_score(y_test, m.predict(X_test)))

print(f"20 random splits of the same model:")
print(f"  min  = {min(accs):.3f}")
print(f"  mean = {np.mean(accs):.3f}")
print(f"  max  = {max(accs):.3f}")
print(f"  std  = {np.std(accs):.3f}")
print()
print("A single split could report 0.90 or 0.95 for the SAME model —")
print("which is why one split is not enough to trust.")"""))

CELLS_3.append(("md", r"""**What did we learn from this output?**

Cross-validation gives a mean **and** a spread: here the model's F1 ranges from about
0.39 to 0.74 across the five folds (0.54 ± 0.12). The 20-split experiment shows a single
train/test split of the *same* model can swing by several points — selecting a model on
one lucky split is gambling.

**What would this mean in a real-world application?**

Before approving a model for deployment, a bank should see not just "F1 = 0.54" but
"F1 = 0.54 ± 0.12 over stratified folds" — and should worry when the spread is this
wide. A model whose score varies wildly across folds is fragile: it may perform far worse
on next month's real data. Also note we scored with
**F1** here — on this imbalanced problem, accuracy across folds hides what we care about.
The metric *inside* cross-validation must be the business-relevant metric.

---

#### Beginner Perspective

*One split can lie. K-fold cross-validation averages performance over K train/test
repetitions and shows stability. For classification, use StratifiedKFold.*

#### Expert Perspective

*Experts choose the CV scheme to match the data structure (grouped data needs GroupKFold;
time series needs forward-chaining ([Section 19](#sec19)); rare positives need
stratification) and choose the
scoring function inside CV to match the business objective. They also watch for
preprocessing leakage *inside* the folds — scaling and feature selection must be fitted on
the training fold only (Section 18).*

---

[⬅ Previous](#sec16) · [🏠 Table of Contents](#toc) · [Next ➡](#sec18)"""))

CELLS_3.append(("md", r"""📗 **Back to basics:** see **Section 12: Cross-Validation** in
`ML_Model_Evaluation_Simple.ipynb` — why one split is not enough, with a Stratified
K-Fold demo.
"""))

CELLS_3.append(("md", r"""<a id="sec18"></a>
## 18. Data Leakage — When Your Model Cheats

### What is data leakage?

**Data leakage** happens when information that would NOT be available at prediction time
sneaks into the training process — most dangerously, information from the **test set** or
from the **future**. The model then "cheats": it looks brilliant during evaluation because
the answers were in the room the whole time.

This is the deepest kind of evaluation mistake, because **every metric lies at once** —
accuracy, precision, recall, AUC, everything. A leaked model can score 99% and still be
completely useless in production.

> A leaked model is a genius in the lab and a fraud in the field.

### Two classic flavors of leakage

**1. Target-derived features.** A feature is built from the thing you are trying to
predict. Example: predicting loan default using the feature "customer asked for a
repayment extension" when that flag is only recorded *after* the default decision was
made. The feature contains the answer.

**2. Preprocessing before splitting.** A scaler, imputer, or feature selector is fitted on
the **full dataset** — including the rows that will later be used as the test set. The
test set silently "votes" on the model's parameters, so the test score is optimistic.

### The leak detector's checklist

Before trusting any evaluation, ask three questions:

1. **Would this feature exist at prediction time?** If it is only known after the event
   you are predicting, it is leaking.
2. **Was anything fitted on the full dataset?** Scaling, imputation, feature selection,
   target encoding — any fit must happen inside the training folds only.
3. **Does any feature suspiciously encode the answer?** A feature that correlates with the
   target at near-perfect level on a hard problem is usually a leak, not a discovery.

Let us make both flavors concrete with two demonstrations.
"""))

CELLS_3.append(("code", r"""# --- Demo 1: a target-derived feature (leakage flavor #1) ---
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# A realistic imbalanced problem (20% positive class)
X, y = make_classification(n_samples=2000, n_features=8, n_informative=6,
                           weights=[0.8, 0.2], random_state=31)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=31, stratify=y)

# The LEAKY feature: it is derived from the outcome y (only 5% noise added).
# Imagine a feature like "was flagged in a later audit" — it is only known AFTER
# the prediction would have been made, so it can never exist in production.
rng = np.random.RandomState(0)
leak_train = (y_train + rng.binomial(1, 0.05, len(y_train))) > 0
leak_test  = (y_test  + rng.binomial(1, 0.05, len(y_test)))  > 0

clf_leak   = RandomForestClassifier(n_estimators=100, random_state=31)
clf_leak.fit(np.column_stack([X_train, leak_train.astype(float)]), y_train)

clf_honest = RandomForestClassifier(n_estimators=100, random_state=31)
clf_honest.fit(X_train, y_train)

acc_l = accuracy_score(y_test, clf_leak.predict(np.column_stack([X_test, leak_test.astype(float)])))
acc_h = accuracy_score(y_test, clf_honest.predict(X_test))
auc_l = roc_auc_score(y_test, clf_leak.predict_proba(np.column_stack([X_test, leak_test.astype(float)]))[:, 1])
auc_h = roc_auc_score(y_test, clf_honest.predict_proba(X_test)[:, 1])

print("LEAKY  model (with the impossible feature):")
print(f"  accuracy = {acc_l:.3f}   ROC-AUC = {auc_l:.3f}")
print("HONEST  model (only features available at prediction time):")
print(f"  accuracy = {acc_h:.3f}   ROC-AUC = {auc_h:.3f}")
print()
print("Same data, same model family, same amount of real signal —")
print("the only difference is one impossible feature.")"""))

CELLS_3.append(("md", r"""**What did we learn from this output?**

One leaked feature (out of nine) lifts ROC-AUC from **0.974 to 0.999** — near-perfect
"performance" on a genuinely hard problem. The honest model is the same algorithm on the
same data without the cheat; that is the real level of performance.

**What would this mean in a real-world application?**

A bank's evaluation committee would see AUC 0.996, approve the model for deployment, and
then watch it fail on new customers — because in production the impossible feature does
not exist yet. Worse, the failure is invisible in the metrics: the test score was
inflated by the leak, so there was no honest evaluation to catch it. Notice also how
leakage corrupts the FP/FN accounting: the leaked model's confusion matrix claims almost
no errors, but the *real* model (once deployed) will produce far more FP and FN than the
numbers promised — every business decision built on those numbers is built on fantasy.

The single most reliable symptom of leakage is **suspiciously good performance**: on a
hard, noisy problem, ask "why is this model SO much better than everything else?" If the
answer is not a genuine new source of signal, look for the leak.
"""))

CELLS_3.append(("md", r"""### Demo 2: preprocessing fitted before the split (leakage flavor #2)

The second flavor is subtler because no single feature looks wrong. Here we use **feature
selection on the whole dataset before cross-validation** — and, to make the cheat
impossible to miss, we use **pure noise**: 40 random Gaussian features and coin-flip
labels. There is no signal at all.

- The **honest pipeline** selects features *inside* each training fold (via `Pipeline`),
  so the test fold never influences the choice of features. Its CV score should hover
  around 0.5 — chance.
- The **leaky pipeline** selects the "best" features on the **full dataset** first, then
  cross-validates. Feature selection gets to peek at every fold, including the ones that
  are later used as test folds.
"""))

CELLS_3.append(("code", r"""# --- Demo 2: feature selection before the split (leakage flavor #2) ---
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

# Pure noise: 200 rows x 40 features, random labels. There is NO signal to learn.
rng = np.random.RandomState(7)
X = rng.randn(200, 40)
y = rng.binomial(1, 0.5, 200)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

# LEAKY: SelectKBest is fitted on the FULL dataset (including future test folds).
selector = SelectKBest(f_classif, k=5).fit(X, y)
X_leaky = selector.transform(X)
leak_scores = cross_val_score(LogisticRegression(max_iter=2000), X_leaky, y,
                              cv=cv, scoring="roc_auc")

# HONEST: the same selection happens INSIDE each fold via a Pipeline, so each
# test fold is completely untouched during feature selection.
pipe = Pipeline([("select", SelectKBest(f_classif, k=5)),
                 ("clf", LogisticRegression(max_iter=2000))])
honest_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")

print(f"LEAKY  pipeline (selection on full data):  AUC = {leak_scores.mean():.3f} ± {leak_scores.std():.3f}")
print(f"HONEST pipeline (selection inside folds):  AUC = {honest_scores.mean():.3f} ± {honest_scores.std():.3f}")
print()
print("On pure noise the honest pipeline is at chance (~0.5) — there is nothing to learn.")
print("The leaky pipeline reports a phantom 0.66: the 'signal' is the test set voting")
print("through the feature selector.")"""))

CELLS_3.append(("md", r"""**What did we learn from this output?**

The honest pipeline on pure noise sits at chance (**0.528 ± 0.052** — a coin flip, as it
should be). The leaky pipeline reports **0.659 ± 0.050** — a phantom "signal" created
entirely by letting the test folds participate in feature selection. There is literally
nothing to predict, yet the evaluation claims a working model.

**What would this mean in a real-world application?**

This is how teams ship models that "worked in the lab" and fail on real data: a small,
noisy dataset, a few hundred features, feature selection applied to the whole dataset
"to clean the data up" — and the evaluation says 0.66 while reality is 0.5. The fix is
mechanical: **put every preprocessing step inside the cross-validation folds**, which is
exactly what `Pipeline` does. If you fit scalers, imputers, encoders, or selectors
manually, you must re-fit them inside each training fold — never on the combined data.

### The fix: golden rules for leak-free evaluation

1. **Everything that touches the data must be fitted inside the training folds.**
   Scaling, imputation, feature selection, encoding, class balancing — all of it. A
   `Pipeline` makes this automatic and is the safest habit.
2. **Every feature must exist at prediction time.** If you would not know the value when
   making the real-world prediction, it must not be a feature.
3. **No future information.** For time series, never shuffle: use temporal splits where
   training data always ends before validation data begins (we demonstrate this in
   [Section 19](#sec19)).
4. **Beware target encoding done on the full dataset** — a classic way for the label to
   leak into the features.
5. **Watch for duplicated rows** between train and test (or between folds) — identical
   rows in both sets leak the answer directly.
6. **Treat suspiciously high scores as a red flag, not a triumph.** On hard problems,
   ask where the signal could possibly come from.
7. **Sanity-check with an ablation:** remove the suspicious feature and see if the score
   collapses. If it does, you found a leak, not a discovery.

---

#### Beginner Perspective

*Never fit preprocessing on the full dataset. Use `Pipeline` so every step is fitted
inside the folds, and never use a feature that would not exist at prediction time.*

#### Expert Perspective

*Experts design the validation scheme around the data-generating process: temporal
splits for time series, grouped splits when rows share entities, and leak audits where
feature importance and ablations are checked against domain knowledge. They also know
that leakage has a cost dimension: every inflated metric misprices FP and FN — the
deployed model will make far more expensive errors than the evaluation promised, which
is why honest evaluation is a business requirement, not an academic nicety.*

---

[⬅ Previous](#sec17) · [🏠 Table of Contents](#toc) · [Next ➡](#sec19)"""))

CELLS_3.append(("md", r"""📗 **Back to basics:** only here in depth — the seed idea (“judge a model only
on data it has never seen”) is **Section 1: Introduction to Model Evaluation** in
`ML_Model_Evaluation_Simple.ipynb`.
"""))

CELLS_3.append(("md", r"""<a id="sec19"></a>
## 19. Time-Series Evaluation — Never Shuffle the Clock

### Why is time special?

Most of this notebook assumes rows are **independent**: one customer, one image, one
transaction. Time series break that assumption — each row's value depends on the rows
before it. Today's sales, temperature, CPU load, or stock price is correlated with
yesterday's.

That one fact changes how evaluation must be done. If you split a time series with a
**random shuffle**, the test set is no longer "the future" — it is a random sample from
the same period as the training data. A test row at time *t* has its previous value
(at *t*−1) sitting inside the training set, so the model can **interpolate** instead of
**forecast**. The evaluation then measures something easy, not something real: it is a
quiet form of the data leakage from [Section 18](#sec18), produced by the split itself.

### The three canonical schemes

1. **Single temporal cut.** Train on the past (e.g., first 80%), test on the future
   (last 20%). Simple, honest, one number — but only one number.
2. **Forward-chaining (expanding window).** Train on an ever-growing past and validate
   on the next block: fold 1 trains on months 1–2, validates on month 3; fold 2 trains
   on months 1–3, validates on month 4; and so on. This is `TimeSeriesSplit` and it
   mimics deployment exactly — at each step you only know the past.
3. **Sliding window.** Same idea, but training uses a fixed-size recent window (e.g.,
   the last 3 months only). The model adapts faster when behavior drifts, at the cost
   of using less data.

The next cell shows what happens when you ignore all of this.
"""))

CELLS_3.append(("code", r"""# --- Why shuffling a time series leaks the future into training ---
from sklearn.linear_model import LinearRegression

# A smooth, slowly drifting series (sine + trend + noise): a plausible demand or
# temperature signal where "the future looks like the past".
rng = np.random.RandomState(0)
n = 500
t = np.arange(n)
y = np.sin(t / 12.0) + 0.02 * t + rng.randn(n) * 0.3

# Feature: the previous value. Target: the next value. (A 1-step-ahead forecast.)
X = y[:-1].reshape(-1, 1)
target = y[1:]

# (1) HONEST: train on the past, test on the future
cut = int(0.8 * len(X))
m = LinearRegression().fit(X[:cut], target[:cut])
r2_honest = r2_score(target[cut:], m.predict(X[cut:]))

# (2) SHUFFLED: random split — test rows are interleaved with training rows, so
#     each test point's "previous value" is almost certainly in the training set.
X_tr, X_te, y_tr, y_te = train_test_split(X, target, test_size=0.2, random_state=0)
m2 = LinearRegression().fit(X_tr, y_tr)
r2_shuf = r2_score(y_te, m2.predict(X_te))

print(f"HONEST   (train on past, test on future):  R² = {r2_honest:.3f}")
print(f"SHUFFLED (random split, future leaks in):  R² = {r2_shuf:.3f}")
print()
print("Same model, same data, same number of rows.")
print("Shuffling only changed WHICH rows are test rows — and it let the model")
print("interpolate instead of forecast.")"""))

CELLS_3.append(("md", r"""**What did we learn from this output?**

The exact same `LinearRegression` scores **R² = 0.979** when the rows are shuffled but
only **R² = 0.813** on a proper temporal split. The shuffle did not add information — it
*moved the test set into the training period*, so every test point's lagged feature had
an almost-identical neighbor in training. The model looks dramatically better than it
will ever be in practice.

**What would this mean in a real-world application?**

Deployment is always a forecast: when the model runs, only the past exists. A demand-
forecasting model evaluated on shuffled data promises accuracy it cannot deliver, so
inventory and staffing budgets built on that number will be wrong. The same leak
inflates classification metrics too — a fraud or churn model scored on shuffled months
overstates precision and recall, and the FP/FN counts the business sees in production
will be worse than the evaluation promised. If the rows have a time axis, the split
must respect it.
"""))

CELLS_3.append(("code", r"""# --- Forward-chaining: TimeSeriesSplit and its fold plot ---
from sklearn.model_selection import TimeSeriesSplit

# TimeSeriesSplit: each fold trains on the PAST and validates on the NEXT block.
# The training window grows with every fold (expanding window / forward-chaining).
tss = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(LinearRegression(), X, target, cv=tss, scoring="r2")

print("TimeSeriesSplit R² per fold:", np.round(scores, 3))
print(f"mean = {scores.mean():.3f} ± {scores.std():.3f}")
print("Note the first fold: it trains on only 84 rows, so it scores worst —")
print("report the per-fold spread, not just the mean.")

# Visualize the folds: every validation block lies strictly AFTER its training block
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(t, y, lw=1, color="slategray", label="Series")
for k, (tr, te) in enumerate(tss.split(X)):
    ax.axvspan(t[min(te)], t[max(te)], alpha=0.25, color=plt.cm.viridis(k / tss.n_splits))
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.set_title("TimeSeriesSplit: each validation block lies strictly after its training block")
ax.legend(fontsize=8)
plt.show()"""))

CELLS_3.append(("md", r"""### Try it yourself — the temporal cut-point explorer

Drag the slider to choose where the **temporal cut** falls (how much of the history
trains the model). Watch two numbers at once:

- **Honest R²** — train on the past, test on the future. This is the number the deployed
  model will actually deliver.
- **Shuffled R²** — the same model, same data, random split. It stays almost perfectly
  flat, no matter where you cut.

Experiments to try:

1. Drag from 50% to 80%: the honest R² wobbles while the shuffled one barely moves.
   Which number depends on time? Only the honest one.
2. Push to 90–95%: the honest estimate collapses — the model must forecast far into the
   future and is judged on only 25–50 points. The shuffled line does not care, because
   it never looks at time at all.
3. Ask yourself: if a team reported the shuffled number to the budget committee, which
   line of the plot is the lie?

*This widget needs `ipywidgets`. If it is not installed, the cell below shows a static
snapshot instead of live sliders — install it with `pip install ipywidgets`, restart the
kernel, and re-run the cell.*
"""))

CELLS_3.append(("code", r"""# --- Interactive: temporal cut-point explorer ---
# Self-contained: rebuilds the series and both models here, so it works after a restart.
rng = np.random.RandomState(0)
n_ts = 500
t_ts = np.arange(n_ts)
y_ts = np.sin(t_ts / 12.0) + 0.02 * t_ts + rng.randn(n_ts) * 0.3
X_ts = y_ts[:-1].reshape(-1, 1)
target_ts = y_ts[1:]


def honest_r2(cut_frac):
    # Train on the past, test on the future, at the given cut fraction.
    cut = int(cut_frac * len(X_ts))
    m = LinearRegression().fit(X_ts[:cut], target_ts[:cut])
    return r2_score(target_ts[cut:], m.predict(X_ts[cut:])), cut


def shuffled_r2(test_frac, seed=0):
    # Same model, same data, random split of the same size.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_ts, target_ts, test_size=test_frac, random_state=seed)
    m = LinearRegression().fit(X_tr, y_tr)
    return r2_score(y_te, m.predict(X_te))


try:
    import ipywidgets as widgets
    from IPython.display import display

    def explore(cut_frac):
        r2_h, cut = honest_r2(cut_frac)
        r2_s = shuffled_r2(1 - cut_frac)
        m = LinearRegression().fit(X_ts[:cut], target_ts[:cut])
        y_forecast = m.predict(X_ts[cut:])
        test_t = np.arange(cut, len(X_ts))

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))

        # Left: the series, the cut, and the honest forecast of the future
        axes[0].plot(t_ts, y_ts, lw=1, color="slategray", label="Series")
        axes[0].axvline(cut, color="crimson", linestyle="--", lw=1.5,
                        label=f"Cut at t = {cut}")
        axes[0].plot(test_t, y_forecast, lw=1.8, color="seagreen", label="Honest forecast")
        axes[0].set_xlabel("Time"); axes[0].set_ylabel("Value")
        axes[0].set_title(f"Train on the past ({cut} rows), forecast the future ({len(X_ts)-cut} rows)")
        axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

        # Right: honest vs shuffled at the current cut
        bars = axes[1].bar(["Honest\n(temporal cut)", "Shuffled\n(random split)"],
                           [r2_h, r2_s], color=["steelblue", "coral"])
        for bar, v in zip(bars, [r2_h, r2_s]):
            axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}",
                         ha="center")
        axes[1].set_ylim(0, 1.08)
        axes[1].set_ylabel("R²")
        axes[1].set_title("Same model, same data — only the split changes")
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()

        print(f"Cut = {cut_frac:.0%} of history for training ({cut} rows)  |  test = {len(X_ts)-cut} future rows")
        print(f"Honest R²   = {r2_h:.3f}   (the number deployment will actually deliver)")
        print(f"Shuffled R² = {r2_s:.3f}   (the number shuffling pretends you have)")
        print("The shuffled estimate is flat across cuts: it never depends on time — and it lies.")

    slider_cut = widgets.FloatSlider(value=0.80, min=0.50, max=0.95, step=0.01,
                                     description="Cut point:")
    out = widgets.interactive_output(explore, {"cut_frac": slider_cut})
    with out:
        explore(0.80)   # start at the section's default 80/20 split
    display(widgets.VBox([slider_cut, out]))
    print("✅ Drag the cut point: honest R² moves with the cut, the shuffled one stays flat.")

except ImportError:
    print("ipywidgets is not installed, so the interactive slider is unavailable.")
    print("Install it with:  pip install ipywidgets   (then restart the kernel and re-run this cell).")
    print("Until then, use the table above: honest R² moves with the cut; shuffled R² stays ~0.98.")"""))

CELLS_3.append(("md", r"""**What did we learn from this output?**

The folds show the honest spread: R² ranges from **0.55 to 0.87** (mean 0.76), and the
first fold — with the least training data — is the worst. A single number would hide
this. The plot is the key picture: **training always ends before validation begins**,
which is exactly how the model will be used in production.

### How to score a model on temporally ordered data

1. **Never shuffle.** Random splits are the leak.
2. **Use `TimeSeriesSplit` (or a manual temporal cut) and `cross_val_score`.** Each test
   fold is the future of its training fold.
3. **Choose expanding vs sliding window deliberately.** Expanding uses all history and
   suits stable series; sliding windows adapt to drift (e.g., fashion, prices) but use
   less data. Test both and compare the per-fold means.
4. **Keep every preprocessing step inside the folds** — the `Pipeline` habit from
   [Section 18](#sec18) applies to time series exactly as to i.i.d. data.
5. **Report the per-fold spread**, especially the early folds, which show how the model
   behaves when data is scarce.
6. **For grouped time series** (many stores, many sensors, each with its own timeline),
   use a grouped temporal split so a single entity does not appear in both train and
   test folds.

### Case study: sales across 20 stores (why the entity axis matters)

We now have **two axes of leakage**: *time* (the future) and *entity* (the store). A
single-series model only has the time axis, but a **panel** of many stores has both.
Each of the 20 stores below has its own baseline level, a shared seasonal pattern, and a
gentle upward trend — a typical retail sales setup.

The model predicts sales from the day, the season, and a **store indicator** (one column
per store), exactly like a model that "remembers" each store's typical level. Now run
the same model through three validation protocols:

- **A — Plain shuffle:** random 80/20 split. Leaks time *and* store.
- **B — Plain temporal cut** (the `TimeSeriesSplit` idea from above, applied to the
  pooled panel): train on days < 320, test on days ≥ 320. Time is respected, but the
  test stores were **already in training** — the model can recognize them.
- **C — Grouped temporal:** train on 16 stores' full history, test on 4 **brand-new
  stores** the model has never seen. Leaks neither time nor store.

All three test sets have the same size (1,600 rows), so the R² numbers are comparable.
"""))

CELLS_3.append(("code", r"""# --- Panel case study: three validation protocols on 20 stores ---
# Each store has its own level, plus shared seasonality and trend.
rng = np.random.RandomState(3)
n_stores, n_days = 20, 400
period = 60
store_effect = rng.normal(0, 3.0, n_stores)
day = np.arange(n_days)

rows = []
for s in range(n_stores):
    sales = (store_effect[s] + 0.02 * day
             + 2.0 * np.sin(2 * np.pi * day / period) + rng.normal(0, 0.5, n_days))
    rows.append(pd.DataFrame({"store": f"S{s:02d}", "day": day,
                              "sin": np.sin(2 * np.pi * day / period),
                              "cos": np.cos(2 * np.pi * day / period),
                              "sales": sales}))
df = pd.concat(rows, ignore_index=True)


def build_features(d, store_cols):
    # One indicator column per store; columns for unseen stores are zeroed.
    dummies = pd.get_dummies(d["store"]).reindex(columns=store_cols, fill_value=0)
    return pd.concat([d[["day", "sin", "cos"]], dummies], axis=1)


all_stores = sorted(df["store"].unique())
train_stores = all_stores[:16]
test_stores = all_stores[16:]

# A. Plain shuffle — leaks time AND store
X = build_features(df, all_stores)
y = df["sales"].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
r2_A = r2_score(y_te, LinearRegression().fit(X_tr, y_tr).predict(X_te))

# B. Plain temporal cut — time respected, but the test stores WERE in training
b_tr = df[df["day"] < 320]
b_te = df[df["day"] >= 320]
mB = LinearRegression().fit(build_features(b_tr, all_stores), b_tr["sales"].values)
r2_B = r2_score(b_te["sales"].values, mB.predict(build_features(b_te, all_stores)))

# C. Grouped temporal — test stores are brand new (never seen in training)
c_tr = df[df["store"].isin(train_stores)]
c_te = df[df["store"].isin(test_stores)]
mC = LinearRegression().fit(build_features(c_tr, train_stores), c_tr["sales"].values)
r2_C = r2_score(c_te["sales"].values, mC.predict(build_features(c_te, train_stores)))

print(f"A. Plain shuffle         R² = {r2_A:.3f}   (leaks time + store)")
print(f"B. Plain temporal cut   R² = {r2_B:.3f}   (leaks store only)")
print(f"C. Grouped temporal     R² = {r2_C:.3f}   (leaks neither)")
print()
print("A and B both look great. Only C answers the real question:")
print("how well does this model work for stores it has never seen?")

# The mechanism, in pictures: true sales vs each protocol's predictions, per test store
fig = plt.figure(figsize=(13.5, 6.5))
gs = fig.add_gridspec(2, 4)
ax_bar = fig.add_subplot(gs[0, :])
bars = ax_bar.bar(["A. Plain shuffle", "B. Plain temporal", "C. Grouped temporal"],
                  [r2_A, r2_B, r2_C], color=["coral", "orange", "steelblue"])
for bar, v in zip(bars, [r2_A, r2_B, r2_C]):
    ax_bar.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
ax_bar.set_ylim(0, 1.08)
ax_bar.set_ylabel("R² on test data")
ax_bar.set_title("Same model, same 20-store panel — only the validation protocol changes")
ax_bar.grid(axis="y", alpha=0.3)

for i, s in enumerate(test_stores):
    ax = fig.add_subplot(gs[1, i])
    sub = df[df["store"] == s]
    ax.plot(sub["day"], sub["sales"], lw=1, color="slategray", label="True sales")
    ax.plot(sub["day"], mB.predict(build_features(sub, all_stores)),
            lw=1.5, color="coral", ls="--", label="Pooled model (knows this store)")
    ax.plot(sub["day"], mC.predict(build_features(sub, train_stores)),
            lw=1.5, color="steelblue", label="Grouped model (new store)")
    ax.set_title(f"Store {s}")
    ax.set_xlabel("Day")
    if i == 0:
        ax.set_ylabel("Sales")
        ax.legend(fontsize=7)
plt.tight_layout()
plt.show()"""))

CELLS_3.append(("md", r"""**What did we learn from this output?**

Fixing the time axis alone changes almost nothing: **R² = 0.983 shuffled vs 0.977 on a
plain temporal cut**. The reason is visible in the four store panels — the pooled model
"recognizes" each test store because that store's past was in training, so it snaps to
the store's level. Only the **grouped temporal split (R² = 0.268)** removes the store
from the picture: the grouped model has to predict each new store from the global
pattern alone, which is what deployment to a new store actually requires.

**What would this mean in a real-world application?**

A retail chain evaluating "will this sales model work?" gets two very different answers
depending on the protocol: ~0.98 for stores already known (and already being served by
the model) versus ~0.27 for new locations. Reporting the pooled number to an expansion
committee would promise new-store accuracy the model cannot deliver. The right protocol
must match the deployment question: existing stores, new stores, or both. A bonus
diagnostic: a vanilla pooled `TimeSeriesSplit` on this panel produces wildly unstable
folds (some ≈ 0.9, some ≈ 0.3) because the row-order split randomly lands test blocks
on known or unknown stores — instability across folds is itself a red flag that the
split structure does not match the data structure.

### Try it yourself — which stores are "new"?

Pick which **4 stores are held out** as new, and the grouped R² is recomputed for
exactly your selection. The right-hand histogram shows where your choice falls among
**all 4,845 possible 4-store hold-out sets** (computed analytically, so it is instant).

Experiments to try:

1. The defaults (S16–S19) reproduce the case study: R² ≈ 0.27. Where does that sit in
   the distribution?
2. Find an **easy** hold-out: pick stores whose levels sit near the middle of the left
   chart (e.g., S04, S06, S08, S14) and watch R² climb toward 0.95.
3. Find a **hard** hold-out: pick stores with extreme levels (e.g., S00, S11, S13,
   S18) and watch R² go **negative** — worse than predicting the average.
4. Look at the spread of the histogram. One hold-out set is one draw: the "honest
   number" is really a distribution. And the leaky protocols (A ≈ 0.98, B ≈ 0.98) sit
   entirely outside it — no honest evaluation comes close to them.

*This widget needs `ipywidgets`. If it is not installed, the cell below shows a static
snapshot instead of live dropdowns — install it with `pip install ipywidgets`, restart
the kernel, and re-run the cell.*
"""))

CELLS_3.append(("code", r"""# --- Interactive: which 4 stores are "new"? ---
# Self-contained: rebuilds the panel, then enumerates ALL 4,845 possible 4-store
# hold-out sets analytically, so the histogram is exact and instant.
from itertools import combinations

rng = np.random.RandomState(3)
n_stores, n_days = 20, 400
period = 60
store_effect = rng.normal(0, 3.0, n_stores)
day = np.arange(n_days)
rows = []
for s in range(n_stores):
    sales = (store_effect[s] + 0.02 * day
             + 2.0 * np.sin(2 * np.pi * day / period) + rng.normal(0, 0.5, n_days))
    rows.append(pd.DataFrame({"store": f"S{s:02d}", "day": day,
                              "sin": np.sin(2 * np.pi * day / period),
                              "cos": np.cos(2 * np.pi * day / period),
                              "sales": sales}))
df = pd.concat(rows, ignore_index=True)
all_stores = sorted(df["store"].unique())


def build_features(d, store_cols):
    dummies = pd.get_dummies(d["store"]).reindex(columns=store_cols, fill_value=0)
    return pd.concat([d[["day", "sin", "cos"]], dummies], axis=1)


def grouped_r2(held):
    # Exact same code path as the case study above (brute-force fit).
    held = list(dict.fromkeys(held))
    tr = df[~df["store"].isin(held)]
    te = df[df["store"].isin(held)]
    cols = [s for s in all_stores if s not in held]
    m = LinearRegression().fit(build_features(tr, cols), tr["sales"].values)
    return r2_score(te["sales"].values, m.predict(build_features(te, cols)))


# Analytic shortcut for the full enumeration: the day/seasonal coefficients are the
# same for EVERY hold-out set (store effects are independent of the shared pattern),
# so only the store-level intercept changes between combos. Precompute per-store
# residual sums once, then each of the 4,845 combos costs O(20) instead of a model fit.
beta = LinearRegression().fit(
    pd.concat([df[["day", "sin", "cos"]],
               pd.get_dummies(df["store"]).reindex(columns=all_stores, fill_value=0)], axis=1),
    df["sales"].values).coef_[:3]

S1, S2, Sy, Sy2, ns = {}, {}, {}, {}, {}
for s in all_stores:
    sub = df[df["store"] == s]
    e = sub["sales"].values - sub[["day", "sin", "cos"]].to_numpy() @ beta
    S1[s] = e.sum(); S2[s] = (e ** 2).sum()
    Sy[s] = sub["sales"].sum(); Sy2[s] = (sub["sales"] ** 2).sum(); ns[s] = len(sub)


def analytic_r2(combo):
    train = [s for s in all_stores if s not in combo]
    b = sum(S1[s] for s in train) / sum(ns[s] for s in train)      # combo intercept
    rss = sum(S2[s] - 2 * b * S1[s] + ns[s] * b * b for s in combo)
    nt = sum(ns[s] for s in combo)
    mu = sum(Sy[s] for s in combo) / nt
    tss = sum(Sy2[s] for s in combo) - nt * mu * mu
    return 1 - rss / tss


all_r2 = np.array([analytic_r2(c) for c in combinations(all_stores, 4)])
store_means = df.groupby("store")["sales"].mean()
print(f"Enumerated all {len(all_r2)} hold-out sets: grouped R² ranges "
      f"{all_r2.min():.2f} to {all_r2.max():.2f} (median {np.median(all_r2):.2f})")


try:
    import ipywidgets as widgets
    from IPython.display import display

    def explore(s1, s2, s3, s4):
        held = list(dict.fromkeys([s1, s2, s3, s4]))
        r2_cur = grouped_r2(held)
        pct = (all_r2 < r2_cur).mean()

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))

        # Left: store levels, held-out stores highlighted
        colors = ["coral" if s in held else "steelblue" for s in all_stores]
        axes[0].bar(all_stores, store_means.values, color=colors)
        axes[0].set_xlabel("Store"); axes[0].set_ylabel("Mean sales (level)")
        axes[0].set_title("Store levels — held-out stores in coral")
        axes[0].tick_params(axis="x", rotation=90, labelsize=7)
        axes[0].grid(axis="y", alpha=0.3)

        # Right: the honest R² distribution over ALL hold-out sets
        axes[1].hist(all_r2, bins=40, color="steelblue", alpha=0.65)
        axes[1].axvline(r2_cur, color="coral", lw=2,
                        label=f"Your selection: R² = {r2_cur:.3f}")
        axes[1].set_xlabel("Grouped R² (4 stores held out)")
        axes[1].set_ylabel("Hold-out sets (of 4,845)")
        axes[1].set_title("The honest number is a distribution")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        print(f"Held-out stores: {held}  ->  grouped R² = {r2_cur:.3f}")
        print(f"This is better than {pct:.0%} of all 4,845 possible hold-out sets.")
        print("Reference: plain shuffle R² = 0.983 and pooled temporal R² = 0.977 —")
        print("both sit far above the entire honest range.")

    dropdowns = [widgets.Dropdown(options=all_stores, value=all_stores[i],
                                  description=f"New store {k}:",
                                  layout=widgets.Layout(width="150px"))
                 for k, i in enumerate([16, 17, 18, 19])]
    out = widgets.interactive_output(explore,
                                     {"s1": dropdowns[0], "s2": dropdowns[1],
                                      "s3": dropdowns[2], "s4": dropdowns[3]})
    with out:
        explore(*[d.value for d in dropdowns])   # initial view = the case study's S16–S19
    display(widgets.VBox([widgets.HBox(dropdowns), out]))
    print("✅ Change which stores are 'new' and watch the honest R² move.")

except ImportError:
    print("ipywidgets is not installed, so the interactive dropdowns are unavailable.")
    print("Install it with:  pip install ipywidgets   (then restart the kernel and re-run this cell).")
    print("Until then: the case study above shows one draw from this distribution (S16–S19 -> R² = 0.27).")"""))

CELLS_3.append(("md", r"""### The evaluation protocol decision tree

Everything in Section 19 reduces to two questions:

1. **What is the structure of the rows?** Independent, one timeline, or many entities × time?
2. **What will deployment look like?** New random rows, the next month of the same series,
   or brand-new entities?

```
Start: what do your rows look like?
│
├─ INDEPENDENT rows (no time axis, no shared entity)
│    └→ K-Fold / Stratified K-Fold — shuffling is harmless (Section 17)
│         └ rows share an entity (same patient, same user)?
│              └→ GroupKFold — one entity must not straddle folds
│
├─ ONE time series (a single timeline)
│    └→ TimeSeriesSplit (expanding window) or a sliding window
│         └ NEVER shuffle; training must end before validation begins
│
└─ PANEL (many entities × time)
     ├─ deployment to KNOWN entities (same stores next month)
     │    └→ temporal split — train on the past of the same entities
     │
     └─ deployment to NEW entities (expansion to new stores)
          └→ GROUPED temporal split — hold out whole entities
               └ both axes must be respected: the case study above
```

### The one-line version

| Your data | Your deployment question | Use | Never use |
|---|---|---|---|
| i.i.d. rows | new random rows | K-Fold / Stratified K-Fold | — |
| one time series | the next month(s) | TimeSeriesSplit / sliding window | shuffled splits |
| panel, known entities | same stores next month | temporal cut (pooled) | shuffled splits |
| panel, new entities | expansion to new stores | grouped temporal hold-out | shuffled OR plain `TimeSeriesSplit` |

**The rule that covers all four rows:** choose the split so that the test set contains
exactly what the model will not know at prediction time — new random rows, the future,
new entities, or all three. If a split lets the model see any of them during training,
the evaluation is a lie, no matter which metric you compute on it.
"""))

CELLS_3.append(("md", r"""---

#### Beginner Perspective

*Time matters. Train on the past, test on the future, and never shuffle the clock.
`TimeSeriesSplit` does the first two for you.*

#### Expert Perspective

*Experts match the validation scheme to the data-generating process: temporal
forward-chaining for single series, grouped temporal splits for panel data, and
embedding- or date-based splits when leakage can creep through lags and rolling
features. They also treat horizon carefully — a 1-step-ahead and a 30-step-ahead model
need different validation, and every metric must be computed at the same horizon the
business actually forecasts.*

---

[⬅ Previous](#sec18) · [🏠 Table of Contents](#toc) · [Next ➡](#sec20)"""))

CELLS_3.append(("md", r"""📗 **Back to basics:** only here in depth — the essential idea (“never evaluate
on data the model could have seen”) is **Section 1: Introduction to Model Evaluation**
in `ML_Model_Evaluation_Simple.ipynb`.
"""))

CELLS_3.append(("md", r"""<a id="sec20"></a>
## 20. Regression Metrics

Regression predictions are numbers, so errors are **differences between predicted and
true values**. Let $y_i$ be the true value of sample $i$ and $\hat{y}_i$ the prediction;
$n$ is the number of samples.

### MAE — Mean Absolute Error

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

- **Measures:** the average absolute error, in the **same units as the target**.
- **Intuition:** "on average, how far off are the predictions, ignoring direction?"
- **When to use:** when every error counts equally and you want an easily explainable
  number.
- **Limitation:** gives no extra penalty to large errors.

### MSE — Mean Squared Error

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- **Measures:** the average of *squared* errors.
- **Intuition:** large errors are penalized **quadratically** — one big mistake hurts far
  more than many small ones.
- **When to use:** when large errors are disproportionately bad (e.g., one grossly wrong
  price estimate).
- **Limitation:** units are squared, so the number is hard to interpret directly; also
  sensitive to outliers.

### RMSE — Root Mean Squared Error

$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

- **Measures:** the same as MSE but back in the **original units**.
- **Intuition:** an "average-size" error with a heavy penalty on large errors — think of
  it as a standard deviation of the errors.
- **When to use:** when you want MSE's big-error penalty *and* interpretable units.
- **Limitation:** still outlier-sensitive; always larger than or equal to MAE (the gap
  grows with the size of the largest errors).

### R² — Coefficient of Determination

$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

- **Measures:** the proportion of variance in the target explained by the model, compared
  with a baseline that always predicts the mean $\bar{y}$.
- **Intuition:** R² = 1 is perfect; R² = 0 means "no better than predicting the mean";
  R² can be negative (worse than the mean baseline).
- **When to use:** to communicate "how much of the variation does the model capture?"
- **Limitation:** R² rises mechanically with more features and can look impressive while
  errors are still large in absolute terms.

### Business interpretation — why units matter

If a house-price model has MAE = 50,000, it means: *on average, the predicted price is
about 50,000 currency units away from the true price.* Whether that is good depends on the
application — for a €100,000 house it is terrible; for a €5,000,000 house it might be
fine. **The meaning of an error always depends on the application** — the same reasoning
we used for classification costs applies to regression errors too.
"""))

CELLS_3.append(("code", r"""# --- Regression metrics: house price prediction ---
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Synthetic house-price data
rng = np.random.RandomState(42)
n = 300
area = rng.uniform(800, 3500, n)
bedrooms = rng.randint(1, 6, n).astype(float)
price = 40000 + 140 * area + 8000 * bedrooms + rng.normal(0, 20000, n)

df = pd.DataFrame({"area": area, "bedrooms": bedrooms, "price": price})

Xr = df[["area", "bedrooms"]]
yr = df["price"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.3, random_state=42)
lr = LinearRegression().fit(Xr_train, yr_train)
pred = lr.predict(Xr_test)

mae  = mean_absolute_error(yr_test, pred)
mse  = mean_squared_error(yr_test, pred)
rmse = np.sqrt(mse)
r2   = r2_score(yr_test, pred)

print(f"MAE  = {mae:,.0f}   (average absolute error in currency units)")
print(f"MSE  = {mse:,.0f}   (average squared error — penalizes big errors)")
print(f"RMSE = {rmse:,.0f}   (same as MSE but back in currency units)")
print(f"R²   = {r2:.3f}   (fraction of price variance explained)")

# Compare the error sizes with the scale of the target
print(f"\nTypical house price: ~{yr_test.mean():,.0f}")
print(f"RMSE is about {rmse / yr_test.mean():.1%} of the average price")"""))

CELLS_3.append(("md", r"""**What did we learn from this output?**

MAE and RMSE describe the *size* of the errors in currency units; R² describes the
*fraction of variance explained*. Note RMSE > MAE — the big errors are pulling the squared
average up. The last line is the business translation: "the typical prediction error is
about X% of the average house price."

**What would this mean in a real-world application?**

If the model is used to price 1,000 houses a month and MAE = 40,000, the company should
know that its *average* price quote is wrong by 40,000 — before deciding whether that
error is acceptable relative to profit margins and competition. If occasional very wrong
quotes cause lawsuits or lost deals, RMSE (with its big-error penalty) is the metric to
watch rather than MAE.

| Metric | Best for | Main limitation |
|---|---|---|
| MAE | Explainable, unit-level average error | Ignores error size distribution |
| MSE | Penalizing large errors | Hard to interpret (squared units) |
| RMSE | Penalizing large errors + interpretable units | Outlier-sensitive |
| R² | "How much variance is explained?" | Can hide large absolute errors |

---

#### Beginner Perspective

*MAE = average error in original units. RMSE = like MAE but punishes big mistakes.
R² = how much of the variation the model explains.*

#### Expert Perspective

*Experts choose the regression loss to match business asymmetry (a quote 20% too high and
one 20% too low rarely cost the same; use asymmetric or quantile losses), and they prefer
unit-scale metrics (MAE/RMSE) for business reporting while using R² only as a relative
goodness-of-fit measure, never in isolation.*

---

[⬅ Previous](#sec19) · [🏠 Table of Contents](#toc) · [Next ➡](#sec21)"""))

CELLS_3.append(("md", r"""📗 **Back to basics:** see **Section 13: Regression Metrics** in
`ML_Model_Evaluation_Simple.ipynb` — MAE, MSE, RMSE and R² on a small house-price
example.
"""))

CELLS_3.append(("md", r"""<a id="sec21"></a>
## 21. End-to-End Case Study: Loan Default Prediction

Everything so far comes together in one complete, realistic workflow. We will:

1. Generate a small synthetic loan dataset
2. Explore it and understand the class distribution
3. Define the **business problem** and the cost of each error
4. Split, train three simple models, and get predictions + probabilities
5. Evaluate with the full toolkit: confusion matrix, accuracy, precision, recall,
   specificity, F1, ROC, AUC, PR curve, classification report
6. Analyze thresholds and compare models
7. Recommend a model **and** a threshold **and** explain the business consequences

### Step 1–2. Create and explore the dataset
"""))

CELLS_3.append(("code", r"""# --- Step 1: generate a small synthetic loan dataset ---
rng = np.random.RandomState(2024)
n = 1200

income          = rng.normal(60000, 25000, n).clip(15000, 250000)
credit_score    = rng.normal(680, 60, n).clip(400, 850)
debt_to_income  = rng.uniform(0.05, 0.75, n)
loan_amount     = rng.uniform(5000, 60000, n)
employment_years = rng.uniform(0, 20, n).round(1)

# Default probability: higher with debt-to-income and loan size,
# lower with credit score, income, and employment stability.
# (Features are standardized so the coefficients are comparable in size.)
logit = (-3.4
         - 2.0 * (credit_score - 680) / 60
         + 2.0 * (debt_to_income - 0.4) / 0.2
         + 1.0 * (loan_amount - 32500) / 16000
         - 0.75 * (income - 60000) / 24000
         - 0.5 * (employment_years - 10) / 5)
p_default = 1 / (1 + np.exp(-logit))
default = (rng.rand(n) < p_default).astype(int)

df = pd.DataFrame({
    "income": income.round(0),
    "credit_score": credit_score.round(0),
    "debt_to_income": debt_to_income.round(3),
    "loan_amount": loan_amount.round(0),
    "employment_years": employment_years,
    "default": default,
})

print(df.head())
print("\nClass distribution:")
print(df["default"].value_counts(normalize=True).round(3))
print(f"Default rate: {df['default'].mean():.1%}")"""))

CELLS_3.append(("code", r"""# --- Step 2: explore the data ---
print(df.describe().round(2))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

# Left: class distribution
counts = df["default"].value_counts().sort_index()
bars = axes[0].bar(["No default (0)", "Default (1)"], counts.values,
                   color=["steelblue", "coral"])
for bar, v in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, v, str(v), ha="center", va="bottom")
axes[0].set_title("Loan dataset — class distribution")
axes[0].set_ylabel("Number of applicants")

# Right: mean feature values per class
features = ["income", "credit_score", "debt_to_income", "loan_amount", "employment_years"]
means = df.groupby("default")[features].mean().T
means.columns = ["No default", "Default"]
means.plot(kind="bar", ax=axes[1], color=["steelblue", "coral"])
axes[1].set_title("Mean feature values by class")
axes[1].set_ylabel("Mean value")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()"""))

CELLS_3.append(("md", r"""### Step 3. The business problem and the cost of errors

A bank must decide, for each loan applicant, whether to **approve** or **reject** the
loan. The model predicts the probability that the applicant **will default** — that is
the "positive" class (1 = will default, 0 = will repay). The two errors are not
symmetric:

| Error | What it means | Business consequence |
|---|---|---|
| **FN** | Model says "no default", but the applicant **defaults** → loan approved | The bank **loses the loaned money** (principal). With loan amounts up to 60,000 and losses possibly much larger, this is the expensive error per incident. |
| **FP** | Model says "will default", but the applicant **would have repaid** → loan rejected | The bank loses the **expected profit** (interest) and the customer may go to a competitor. Also an ethical/customer-experience concern, but the per-incident loss is smaller than a default. |

**Which error is more expensive?** In this case study we will assume a **default costs
about 10× more than a rejected good customer**:

- Cost of FN (an approved loan that defaults) = **10,000** currency units
- Cost of FP (a rejected good customer) = **1,000** currency units

> These numbers are assumptions — in a real project they would come from the bank's
> financial analysis. Changing them changes the whole conclusion, which is exactly the
> point: **the metric and threshold follow the costs, not the other way around.**

### Step 4. Split the data and train the models
"""))

CELLS_3.append(("code", r"""# --- Step 4: split and train three simple models ---
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

X = df[features]
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

print(f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")
print(f"Default rate in test set: {y_test.mean():.1%}")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
}

fitted = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    fitted[name] = model
    print(f"Trained: {name}")"""))

CELLS_3.append(("code", r"""# --- Step 5-6: predictions, confusion matrix, and all core metrics at threshold 0.5 ---
rows = []
for name, model in fitted.items():
    y_pred = model.predict(X_test)                    # predictions (labels)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    rows.append({
        "Model": name,
        "Accuracy":   accuracy_score(y_test, y_pred),
        "Precision":  precision_score(y_test, y_pred, zero_division=0),
        "Recall":     recall_score(y_test, y_pred, zero_division=0),
        "Specificity": specificity,
        "F1":         f1_score(y_test, y_pred, zero_division=0),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    })

results = pd.DataFrame(rows).set_index("Model")
results.round(3)"""))

CELLS_3.append(("code", r"""# --- Confusion matrices for all three models ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, model) in zip(axes, fitted.items()):
    cm = confusion_matrix(y_test, model.predict(X_test))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred. No", "Pred. Yes"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual No", "Actual Yes"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)
    ax.set_title(name)
plt.tight_layout()
plt.show()"""))

CELLS_3.append(("code", r"""# --- Step 7-8: ROC curves, AUC, PR curves, PR-AUC for all models ---
from sklearn.metrics import precision_recall_curve, average_precision_score

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))

for name, model in fitted.items():
    probs = model.predict_proba(X_test)[:, 1]          # probabilities of default

    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_val = roc_auc_score(y_test, probs)
    axes[0].plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc_val:.3f})")

    precision, recall, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    axes[1].plot(recall, precision, linewidth=2, label=f"{name} (PR-AUC = {ap:.3f})")

axes[0].plot([0, 1], [0, 1], "k--", label="Random")
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curves — loan default models")
axes[0].legend(loc="lower right"); axes[0].grid(alpha=0.3)

axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision–Recall Curves — loan default models")
axes[1].legend(loc="upper right"); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()"""))

CELLS_3.append(("code", r"""# --- Classification report (Logistic Regression as the example) ---
y_pred_lr = fitted["Logistic Regression"].predict(X_test)
print(classification_report(y_test, y_pred_lr, target_names=["No default", "Default"]))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_lr))"""))

CELLS_3.append(("md", r"""### Step 9. Threshold analysis — choosing the operating point

At the default 0.5 threshold, all three models look reasonable. But the bank does not have
to use 0.5 — it should choose the threshold that **minimizes total business cost** given
its assumed error costs (FP = 10,000, FN = 1,000). Let's sweep the threshold for each
model and compute the total cost at every point.
"""))

CELLS_3.append(("code", r"""# --- Step 10-11: cost-based threshold analysis ---
COST_FP = 1000    # a rejected good customer: lost profit
COST_FN = 10000   # an approved loan that defaults: the bank loses the loaned money

def total_cost(y_true, y_prob, threshold, cost_fp, cost_fn):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp * cost_fp + fn * cost_fn

thresholds = np.arange(0.05, 0.96, 0.05)

rows = []
for name, model in fitted.items():
    probs = model.predict_proba(X_test)[:, 1]
    costs = [total_cost(y_test, probs, t, COST_FP, COST_FN) for t in thresholds]
    best_idx = int(np.argmin(costs))
    rows.append({
        "Model": name,
        "Best threshold": thresholds[best_idx],
        "Min total cost": int(costs[best_idx]),
        "Cost at 0.5": int(total_cost(y_test, probs, 0.5, COST_FP, COST_FN)),
    })

pd.DataFrame(rows).set_index("Model").round(0)"""))

CELLS_3.append(("code", r"""# --- Visualize total cost vs threshold for every model ---
plt.figure(figsize=(9, 5))
for name, model in fitted.items():
    probs = model.predict_proba(X_test)[:, 1]
    costs = [total_cost(y_test, probs, t, COST_FP, COST_FN) for t in thresholds]
    plt.plot(thresholds, costs, "o-", label=name)

plt.xlabel("Classification threshold")
plt.ylabel(f"Total cost = FP × {COST_FP} + FN × {COST_FN}")
plt.title("Business Cost vs Classification Threshold")
plt.legend()
plt.grid(alpha=0.3)
plt.show()"""))

CELLS_3.append(("md", r"""### Step 10b. Interactive threshold explorer — drag the sliders! 🎛️

This is the payoff of everything we just did. **Move the threshold slider** and watch
three things update live:

1. the **confusion matrix** (how many defaults slip through vs good customers rejected),
2. the **total business cost**, and
3. the **cost-optimal threshold** (gray dashed line) — which moves when you change the
   error costs with the two cost sliders.

Try these experiments:

1. Swap the costs: set **Cost FP = 10,000** and **Cost FN = 1,000** (a rejected good
   customer becomes the expensive error). Watch the cost-optimal threshold jump **above**
   0.5 — the bank now approves conservatively.
2. Set **Cost FN = 20,000** and **Cost FP = 0**. The model should now approve almost
   everything — missing a default is ruinous and false alarms are free.
3. Find the threshold that maximizes *accuracy* — then compare its business cost with
   the cost-optimal threshold. Are they the same point? (Almost never!)

> This is the core lesson of the notebook made visible: **the right threshold is the one
> that minimizes business cost — not 0.5, and not the point that maximizes accuracy or
> F1.**

*Technical note:* the sliders need `ipywidgets` (already included in most Jupyter
installations; otherwise `pip install ipywidgets` and restart the kernel). If the sliders
appear but do not respond, re-run this cell with a running kernel. If `ipywidgets` is
missing, the cell prints a clear message instead of failing.
"""))

CELLS_3.append(("code", r"""# --- Step 10b: INTERACTIVE threshold explorer (drag the sliders!) ---
# Uses the Logistic Regression model trained in Step 4. If you restart the kernel and
# run only this cell, run Section 21 from Step 1 first so that `fitted` and `y_test` exist.
if "fitted" not in globals() or "y_test" not in globals():
    raise RuntimeError("Run the earlier cells of Section 21 first (Steps 1-4) — this cell needs the loan model.")

# The probability scores of the recommended (Logistic Regression) model
probs = fitted["Logistic Regression"].predict_proba(X_test)[:, 1]
thresholds = np.arange(0.05, 0.96, 0.05)


def total_cost(y_true, y_prob, threshold, cost_fp, cost_fn):
    # Business cost of a threshold given the cost of each error type
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp * cost_fp + fn * cost_fn


try:
    import ipywidgets as widgets
    from IPython.display import display

    def explore(threshold, cost_fp, cost_fn):
        # Redraw the confusion matrix, the cost curve, and the numbers for one threshold
        y_pred = (probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        cost = fp * cost_fp + fn * cost_fn

        # Where is the cost-optimal threshold for the CURRENT costs?
        sweep = [total_cost(y_test, probs, t, cost_fp, cost_fn) for t in thresholds]
        best_idx = int(np.argmin(sweep))
        best_t = thresholds[best_idx]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

        # Left: confusion matrix at the current threshold
        cm = np.array([[tn, fp], [fn, tp]])
        axes[0].imshow(cm, cmap="Blues")
        axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["Pred. No", "Pred. Yes"])
        axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["Actual No", "Actual Yes"])
        for i in range(2):
            for j in range(2):
                axes[0].text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
        axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
        axes[0].set_title(f"Confusion matrix at threshold {threshold:.2f}")

        # Right: full cost curve + current point + cost-optimal threshold
        axes[1].plot(thresholds, sweep, "o-", color="steelblue", label="Total cost")
        axes[1].axvline(threshold, color="coral", linestyle="--", alpha=0.8)
        axes[1].plot(threshold, cost, "o", color="coral", markersize=9, label="Current point")
        axes[1].axvline(best_t, color="gray", linestyle=":", alpha=0.9)
        axes[1].annotate(f"cost-optimal = {best_t:.2f}", (best_t, sweep[best_idx]),
                         textcoords="offset points", xytext=(8, -12), fontsize=9, color="gray")
        axes[1].set_xlabel("Classification threshold")
        axes[1].set_ylabel(f"Cost = FP×{cost_fp:,} + FN×{cost_fn:,}")
        axes[1].set_title("Total business cost vs threshold")
        axes[1].legend(loc="upper right", fontsize=8)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Numeric readout
        print(f"threshold = {threshold:.2f}  |  FP = {fp}   FN = {fn}   TP = {tp}   TN = {tn}")
        print(f"Precision = {tp / (tp + fp):.3f}   Recall = {tp / (tp + fn):.3f}")
        print(f"Total cost = {fp:,} × {cost_fp:,} + {fn:,} × {cost_fn:,} = {cost:,.0f} currency units")

    slider_t = widgets.FloatSlider(value=0.10, min=0.0, max=1.0, step=0.01,
                                   description="Threshold:", readout_format=".2f")
    slider_fp = widgets.IntSlider(value=1000, min=0, max=20000, step=500, description="Cost FP:")
    slider_fn = widgets.IntSlider(value=10000, min=0, max=20000, step=500, description="Cost FN:")

    out = widgets.interactive_output(explore,
                                     {"threshold": slider_t, "cost_fp": slider_fp, "cost_fn": slider_fn})
    with out:
        explore(0.10, 1000, 10000)   # embed a static snapshot of the initial view
    display(widgets.VBox([widgets.HBox([slider_t, slider_fp, slider_fn]), out]))
    print("✅ The sliders are live! Drag the threshold (and the costs) and watch everything update.")

except ImportError:
    print("ipywidgets is not installed, so the interactive slider is unavailable.")
    print("Install it with:  pip install ipywidgets   (then restart the kernel and re-run this cell).")
    print("Until then, use the static cost curve above: pick a threshold and read its cost off the curve.")"""))

CELLS_3.append(("code", r"""# --- Step 12-13: the final recommendation ---
best_model, best_thresh, best_cost = None, None, np.inf
for name, model in fitted.items():
    probs = model.predict_proba(X_test)[:, 1]
    costs = [total_cost(y_test, probs, t, COST_FP, COST_FN) for t in thresholds]
    idx = int(np.argmin(costs))
    if costs[idx] < best_cost:
        best_model, best_thresh, best_cost = name, thresholds[idx], costs[idx]

print(f"Recommended model:    {best_model}")
print(f"Recommended threshold: {best_thresh:.2f}")
print(f"Minimum expected cost: {best_cost:,.0f} currency units (on the test set)")
print()

probs = fitted[best_model].predict_proba(X_test)[:, 1]
y_pred_best = (probs >= best_thresh).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best).ravel()

print(f"Confusion matrix at the recommended threshold ({best_thresh:.2f}):")
print(f"  TP = {tp}  (defaults we caught and rejected)")
print(f"  FN = {fn}  (defaults we approved — each costs {COST_FN:,})")
print(f"  FP = {fp}  (good customers we rejected — each costs {COST_FP:,})")
print(f"  TN = {tn}  (good customers correctly approved)")
print(f"\n  Cost breakdown: {fp} × {COST_FP:,} + {fn} × {COST_FN:,} = {best_cost:,.0f}")"""))

CELLS_3.append(("md", r"""### Step 14. Discussion and business consequences

**What did we learn from this workflow?**

1. **Metrics are chosen after the costs, not before.** We defined the cost of each error
   first (FN = approved default = 10,000; FP = rejected good customer = 1,000), and
   *then* picked metrics and a threshold. The chosen threshold (0.10) is well **below**
   the default 0.5 — because an approved default costs 10× more than a rejected good
   customer, the bank approves loans more liberally to avoid missing defaulters. At the
   default 0.5 threshold the cost was about 208,000 — almost double the 114,000 at the
   optimum. **The default threshold is a convention, not a decision.** (The formal
   rule behind this choice is derived in Section 23 — Cost-Sensitive Evaluation.)
2. **Accuracy at 0.5 was not the answer.** The best business threshold is the one that
   minimizes cost, which is a different objective than maximizing accuracy or F1.
3. **The confusion matrix at the recommended threshold is the real deliverable** — it says
   exactly how many defaults slip through (FN) and how many good customers are rejected
   (FP), in business units the bank understands.
4. **Models are compared on the business objective.** Random Forest and Logistic
   Regression came out with different balances; the recommendation is the model +
   threshold pair with the lowest expected cost.
5. **The cost numbers are assumptions.** If the bank decided a rejected good customer
   actually costs 20,000 (e.g., reputational damage), the optimum would move to a
   *higher* threshold with higher precision — fewer good customers rejected, at the cost
   of more approved defaults. **The analysis stays the same; the answer changes.**

**Would this result be good enough for the real world?** That is a question for the bank's
risk committee, informed by this analysis. What we can say confidently: of all the
thresholds and models we tried, this combination minimizes the bank's assumed error cost
on the test data. The final "yes or no" is a business decision built on this technical
foundation — exactly how professional evaluation works.

---

[⬅ Previous](#sec20) · [🏠 Table of Contents](#toc) · [Next ➡](#sec22)"""))

CELLS_3.append(("md", r"""📗 **Back to basics:** see **Section 14: Mini Case Study: Loan Default
Prediction** in `ML_Model_Evaluation_Simple.ipynb` — the same bank story, simplified,
with the cost-of-errors table and threshold sweep.
"""))

CELLS_3.append(("md", r"""<a id="sec22"></a>
## 22. Business Decision Case Study

### Two models, two different profiles

Two teams propose models for the same classification problem:

| Metric | Model A | Model B |
|---|---|---|
| Accuracy | 97% | 94% |
| Precision | 95% | 80% |
| Recall | 60% | 90% |

**Which model should we deploy?**

Do not answer yet. The correct answer depends on:

- the business problem,
- the cost of FP,
- the cost of FN,
- risk tolerance,
- operational capacity,
- customer impact.

Model A is "more accurate" and more precise. Model B finds far more positives. Let's
translate both into confusion matrices and then into **money** under two different
business scenarios.
"""))

CELLS_3.append(("code", r"""# --- Step 1: translate precision/recall into confusion matrices ---
def implied_cm(precision, recall, n_pos=100, n_neg=900):
    # Approximate confusion matrix (TP, FP, FN, TN) consistent with P and R
    tp = round(recall * n_pos)
    fn = n_pos - tp
    fp = round(tp / precision - tp) if precision > 0 else 0
    tn = n_neg - fp
    return tp, fp, fn, tn

cm_a = implied_cm(0.95, 0.60)     # Model A: high precision, lower recall
cm_b = implied_cm(0.80, 0.90)     # Model B: lower precision, high recall

pd.DataFrame({
    "Model":    ["A", "B"],
    "Precision": [0.95, 0.80],
    "Recall":    [0.60, 0.90],
    "TP":  [cm_a[0], cm_b[0]],
    "FP":  [cm_a[1], cm_b[1]],
    "FN":  [cm_a[2], cm_b[2]],
    "TN":  [cm_a[3], cm_b[3]],
})

print("Model A: TP =", cm_a[0], "| FP =", cm_a[1], "| FN =", cm_a[2], "| TN =", cm_a[3])
print("Model B: TP =", cm_b[0], "| FP =", cm_b[1], "| FN =", cm_b[2], "| TN =", cm_b[3])"""))

CELLS_3.append(("code", r"""# --- Step 2: compute the business cost of each model under two scenarios ---
def business_cost(cm, cost_fp, cost_fn):
    tp, fp, fn, tn = cm
    return fp * cost_fp + fn * cost_fn

scenarios = [
    ("A: Fraud detection — missed fraud is expensive (FN)",
     50, 5000),          # cost per FP (blocked customer), cost per FN (missed fraud)
    ("B: Manual investigation — false alarms are expensive (FP)",
     500, 100),          # cost per FP (analyst hours), cost per FN (missed review)
]

print(f"{'Scenario':55s} {'Cost A':>10s} {'Cost B':>10s} {'Cheaper':>8s}")
for name, cost_fp, cost_fn in scenarios:
    ca = business_cost(cm_a, cost_fp, cost_fn)
    cb = business_cost(cm_b, cost_fp, cost_fn)
    winner = "Model B" if cb < ca else "Model A"
    print(f"{name:55s} {ca:>10,.0f} {cb:>10,.0f} {winner:>8s}")"""))

CELLS_3.append(("md", r"""**What did we learn from this output?**

- **Scenario A (fraud detection):** missing fraud (FN) costs 5,000 each, blocking a
  legitimate customer (FP) costs only 50. Model B's extra found frauds save far more than
  its extra false alarms cost → **Model B wins**.
- **Scenario B (manual investigation):** each false alarm costs 500 (analyst hours),
  each missed case only 100. Model A's precision saves the investigation budget → **Model
  A wins**.

**The same two models, the same metrics — different answers, because the costs changed.**

### The lesson

> **"The best model depends on the decision context."**

- A model with **higher accuracy** can be the *worse* business choice.
- A model with **lower accuracy but higher recall** can be the better choice when FN is
  expensive.
- A model with **higher precision** can be the better choice when FP is expensive.
- There is **no universally best model or metric** — only models and metrics that fit
  (or fail to fit) a specific business problem, its costs, and its risk tolerance.

This is why the same evaluation toolkit is used in every serious ML project — and why the
*business conversation* around costs happens before the final model choice, not after.

---
"""))

CELLS_3.append(("md", r"""### Interactive: which model wins as the costs change? 🎛️

Model A (high precision) and Model B (high recall) are the two candidates from this
section. The widget below lets you **drag the cost of each error** and watch the winner
flip in real time:

- **Left panel:** the total cost of each model at the current costs — the cheaper model
  is highlighted in green.
- **Right panel:** a **decision map** over the whole cost space. Every point is one
  possible (cost of FP, cost of FN) pair, and the color shows which model wins there.
  The black dashed line is the **tie line** — along it, the two models cost exactly the
  same. The black dot is your current position.

Experiments to try:

1. Start at the **fraud scenario** (cost FP = 50, cost FN = 5,000): Model B wins —
   catching frauds (FN) is what matters.
2. Move to the **manual-investigation scenario** (cost FP = 500, cost FN = 100): watch
   the winner flip to Model A — false alarms are now the expensive error.
3. Slide the dot along the black dashed tie line: the models are tied. Notice the line
   passes through the origin — **only the ratio cost(FN)/cost(FP) matters** (here the
   models tie at cost(FN) ≈ 0.63 × cost(FP)). Doubling both costs changes nothing.

> **The lesson, made visible:** the "best" model is not a property of the models — it is
> a property of the *costs*. Change the costs and the winner changes with them.
"""))

CELLS_3.append(("code", r"""# --- Interactive: which model wins as the costs change? ---
# Uses the two candidate models from this section (Model A: high precision,
# Model B: high recall). Recomputes their confusion matrices if run standalone.

def implied_cm(precision, recall, n_pos=100, n_neg=900):
    # Approximate confusion matrix (TP, FP, FN, TN) consistent with P and R
    tp = round(recall * n_pos)
    fn = n_pos - tp
    fp = round(tp / precision - tp) if precision > 0 else 0
    tn = n_neg - fp
    return tp, fp, fn, tn

if "cm_a" not in globals() or "cm_b" not in globals():
    cm_a = implied_cm(0.95, 0.60)     # Model A: high precision, lower recall
    cm_b = implied_cm(0.80, 0.90)     # Model B: lower precision, high recall

fp_a, fn_a = cm_a[1], cm_a[2]         # Model A: 3 FP, 40 FN (per 1,000 applicants)
fp_b, fn_b = cm_b[1], cm_b[2]         # Model B: 22 FP, 10 FN

# Tie condition: fp_a*c_fp + fn_a*c_fn == fp_b*c_fp + fn_b*c_fn
#   =>  cost(FN) = break_even * cost(FP), with break_even = (fp_b - fp_a)/(fn_a - fn_b)
break_even = (fp_b - fp_a) / (fn_a - fn_b)


try:
    import ipywidgets as widgets
    from IPython.display import display

    def compare(cost_fp, cost_fn):
        cost_a = fp_a * cost_fp + fn_a * cost_fn
        cost_b = fp_b * cost_fp + fn_b * cost_fn
        winner = "Model B" if cost_b < cost_a else "Model A"

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

        # Left: total cost of each model at the current costs
        bars = axes[0].bar(["Model A\n(high precision)", "Model B\n(high recall)"],
                           [cost_a, cost_b])
        bars[0].set_color("seagreen" if cost_a <= cost_b else "silver")
        bars[1].set_color("seagreen" if cost_b < cost_a else "silver")
        for bar, v in zip(bars, [cost_a, cost_b]):
            axes[0].text(bar.get_x() + bar.get_width() / 2, v, f"{v:,.0f}",
                         ha="center", va="bottom")
        axes[0].set_ylabel("Total cost (currency units)")
        axes[0].set_title(f"Total cost of each model — {winner} wins")
        axes[0].grid(axis="y", alpha=0.3)

        # Right: decision map over the whole (cost FP, cost FN) space
        fp_grid = np.linspace(0, 1000, 200)
        fn_grid = np.linspace(0, 10000, 200)
        FPG, FNG = np.meshgrid(fp_grid, fn_grid)
        winner_map = (fp_a * FPG + fn_a * FNG > fp_b * FPG + fn_b * FNG).astype(int)  # 1 = B wins
        axes[1].imshow(winner_map, origin="lower", aspect="auto",
                       extent=[0, 1000, 0, 10000], cmap="coolwarm", alpha=0.55)
        axes[1].plot(fp_grid, break_even * fp_grid, "k--", linewidth=1.6,
                     label=f"Tie line: cost(FN) = {break_even:.2f} × cost(FP)")
        axes[1].plot(cost_fp, cost_fn, "o", color="black", markersize=10, label="Current costs")
        axes[1].text(180, 8800, "Model B wins\n(FN expensive)", color="darkred", fontsize=10, ha="center")
        axes[1].text(620, 250, "Model A wins\n(FP expensive)", color="darkblue", fontsize=10, ha="center")
        axes[1].set_xlabel("Cost of FP (false alarm)")
        axes[1].set_ylabel("Cost of FN (miss)")
        axes[1].set_title("Which model wins for every (cost FP, cost FN) pair?")
        axes[1].legend(fontsize=8, loc="lower right")

        plt.tight_layout()
        plt.show()

        ratio = cost_fn / cost_fp if cost_fp > 0 else float("inf")
        print(f"Model A cost = {cost_a:,.0f}   |   Model B cost = {cost_b:,.0f}")
        print(f"{winner} is cheaper by {abs(cost_a - cost_b):,.0f} currency units")
        print(f"Current ratio cost(FN)/cost(FP) = {ratio:.2f}   (models tie at {break_even:.2f})")

    slider_fp = widgets.IntSlider(value=50, min=0, max=1000, step=10, description="Cost FP:")
    slider_fn = widgets.IntSlider(value=5000, min=0, max=10000, step=100, description="Cost FN:")

    out = widgets.interactive_output(compare, {"cost_fp": slider_fp, "cost_fn": slider_fn})
    with out:
        compare(50, 5000)   # start at the fraud scenario from this section
    display(widgets.VBox([widgets.HBox([slider_fp, slider_fn]), out]))
    print("✅ Drag the cost sliders and watch the winner flip!")

except ImportError:
    print("ipywidgets is not installed, so the interactive slider is unavailable.")
    print("Install it with:  pip install ipywidgets   (then restart the kernel and re-run this cell).")
    print("Until then, use the scenario table above: Model B wins the fraud scenario, Model A wins the investigation scenario.")"""))

CELLS_3.append(("md", r"""#### Beginner Perspective

*Never pick a model by accuracy alone. Ask: what does each type of error cost? The model
that minimizes total cost wins — even if it "looks" worse on paper.*

#### Expert Perspective

*Experts estimate a full cost matrix per prediction, sometimes including operational
capacity constraints (an investigation queue that overflows) and customer-impact terms,
then optimize the threshold and model jointly. They also re-estimate costs periodically:
as business conditions change, the "best" model can change with them.*

---

[⬅ Previous](#sec21) · [🏠 Table of Contents](#toc) · [Next ➡](#sec23)"""))

CELLS_3.append(("md", r"""📗 **Back to basics:** see **Section 14: Mini Case Study: Loan Default
Prediction** in `ML_Model_Evaluation_Simple.ipynb` — the business framing (“the
threshold is a decision, not a default”) in one worked example.
"""))

CELLS_3.append(("md", r"""<a id="sec23"></a>
## 23. Cost-Sensitive Evaluation: From Cost Matrix to Decision Rule

### Why another section on costs?

We have used costs throughout: the loan case study (Section 21) swept thresholds to
minimize `FP × 1,000 + FN × 10,000`, and the business-decision case study (Section 22)
changed its winner when the costs changed. This section formalizes what we were doing —
**how to turn a cost matrix into the optimal decision rule** — and shows that the loan
case study's chosen threshold of 0.10 was not a judgment call at all: it is the output of
a formula.

### A. The cost matrix

Every outcome of a binary classifier has a price. The usual convention assigns cost 0 to
correct predictions and focuses on the two errors:

|  | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | 0 | **C_FP** |
| **Actual Positive** | **C_FN** | 0 |

- **C_FP** = cost of a false positive (a false alarm)
- **C_FN** = cost of a false negative (a miss)

In the loan case study (Section 21): C_FP = 1,000 (good customer wrongly rejected),
C_FN = 10,000 (loan approved that defaults). In a screening test: C_FP = cost of an
unnecessary follow-up, C_FN = cost of a missed disease.

### B. Expected cost per decision

For any threshold, the expected cost per decision on a dataset is the average:

$$\text{E[cost]} = \frac{C_{FP} \cdot FP + C_{FN} \cdot FN}{n}$$

where n is the number of decisions. This is exactly the quantity the loan case study
minimized — expressed per decision rather than as a total.

### C. The optimal decision rule — the derivation

For a single sample with estimated probability p of being positive:

- If we **predict positive**, we pay C_FP whenever the sample is actually negative:
  expected cost = (1 − p) · C_FP
- If we **predict negative**, we pay C_FN whenever the sample is actually positive:
  expected cost = p · C_FN

The optimal (Bayes) decision predicts positive exactly when its expected cost is lower:

$$(1 - p) \cdot C_{FP} < p \cdot C_{FN}$$

Rearranging:

$$C_{FP} - p \cdot C_{FP} < p \cdot C_{FN} \quad\Longrightarrow\quad C_{FP} < p \cdot (C_{FP} + C_{FN})$$

$$p > \frac{C_{FP}}{C_{FP} + C_{FN}}$$

**The optimal threshold is**

$$\boxed{t^* = \frac{C_{FP}}{C_{FP} + C_{FN}}}$$

This is the single most useful formula in cost-sensitive classification:

| Costs | t* | Business meaning |
|---|---|---|
| C_FP = C_FN | 0.50 | The default 0.5 threshold is optimal **only** when errors cost the same |
| C_FN ≫ C_FP (misses are the disaster) | ≈ 0 | Flag/act on almost everything — avoid misses at any cost |
| C_FP ≫ C_FN (false alarms are the disaster) | ≈ 1 | Act only on near-certainty |
"""))

CELLS_3.append(("code", r"""# --- From cost matrix to optimal threshold: the derivation in action ---
# Self-contained synthetic rare-event problem (5% positive class)
X, y = make_classification(n_samples=6000, n_features=6, n_informative=4, n_redundant=0,
                           weights=[0.95, 0.05], random_state=26)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=26, stratify=y)
model = LogisticRegression(max_iter=1000, random_state=26).fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]

C_FP, C_FN = 200, 2000        # a false alarm costs 200; a miss costs 2,000

t_star = C_FP / (C_FP + C_FN)
print(f"Cost matrix:  C_FP = {C_FP},  C_FN = {C_FN}")
print(f"Theoretical optimal threshold:  t* = {C_FP} / ({C_FP} + {C_FN}) = {t_star:.3f}")


def expected_cost(y_true, y_prob, t, c_fp, c_fn):
    # Expected cost per decision at threshold t
    preds = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return (fp * c_fp + fn * c_fn) / len(y_true)


thresholds = np.arange(0.01, 1.00, 0.01)
costs = [expected_cost(y_test, probs, t, C_FP, C_FN) for t in thresholds]
best_t = thresholds[int(np.argmin(costs))]
print(f"Empirical sweep optimum:            t = {best_t:.3f}  (expected cost {min(costs):.1f}/decision)")
print(f"Expected cost at the theoretical t*:      {expected_cost(y_test, probs, t_star, C_FP, C_FN):.1f}/decision")
print(f"Default threshold 0.5:              expected cost {expected_cost(y_test, probs, 0.5, C_FP, C_FN):.1f}/decision")
print(f"Baseline 'predict negative always':  expected cost {expected_cost(y_test, probs, 1.01, C_FP, C_FN):.1f}/decision")
print(f"Baseline 'predict positive always':  expected cost {expected_cost(y_test, probs, -0.01, C_FP, C_FN):.1f}/decision")

plt.figure(figsize=(8, 4.5))
plt.plot(thresholds, costs, color="steelblue", label="Expected cost vs threshold")
plt.axvline(t_star, color="seagreen", linestyle="--", label=f"t* = {t_star:.2f} (theoretical)")
plt.axvline(0.5, color="coral", linestyle=":", alpha=0.7, label="Default 0.5")
plt.xlabel("Classification threshold")
plt.ylabel("Expected cost per decision")
plt.title("Expected cost curve — the optimum is not 0.5")
plt.legend()
plt.grid(alpha=0.3)
plt.show()"""))

CELLS_3.append(("md", r"""### D. The loan case study, explained by the formula

The loan case study (Section 21) used C_FP = 1,000 and C_FN = 10,000:

$$t^* = \frac{1000}{1000 + 10000} = \frac{1000}{11000} \approx 0.09$$

Its empirical threshold sweep found the minimum at **0.10** — the nearest grid point to
0.09. The "business judgment" was the formula all along.

### E. Expected business value — does the model pay for itself?

A model is worth deploying only if it reduces expected cost (or raises expected profit)
compared with a sensible baseline:

- **Baseline "do nothing"** (predict negative for everyone): expected cost =
  C_FN × (share of actual positives)
- **Baseline "act on everything"** (predict positive for everyone): expected cost =
  C_FP × (share of actual negatives)
- **The model at t***: expected cost from its confusion matrix

$$\text{Net value of the model} = \text{Cost(baseline)} - \text{Cost(model at } t^*)$$

If the net value is positive, the model pays for itself. If not, the cheaper baseline
beats the model no matter how impressive the AUC is.

### F. Caveats — where the formula needs care

1. **The probabilities must be well calibrated** (Section 13): the rule treats p as a
   true probability. Miscalibrated probabilities make t* wrong, no matter how good the
   ranking.
2. **t* does not depend on prevalence, but the expected cost does.** The same costs give
   the same threshold whether the positive class is 1% or 50%; what changes is how much
   money the model saves.
3. **Costs can differ per group.** If a false alarm costs more for premium customers,
   the optimal threshold differs per segment. One threshold for everyone is a business
   choice, not a mathematical necessity.
4. **Operational constraints can override the formula.** If the fraud team can only
   review 100 alerts a day (Section 22), the pure cost-minimizing threshold may flood the
   queue — capacity becomes part of the cost structure.
5. **The cost matrix generalizes.** Correct predictions can have values too (e.g.,
   profit on a correctly approved loan); the decision rule extends the same way:
   predict positive when its *net expected value* beats the alternative.
"""))

CELLS_3.append(("code", r"""# --- The loan case study, explained by the formula ---
# Rebuild the loan model (identical generator and seed to Section 21)
rng = np.random.RandomState(2024)
n = 1200
income = rng.normal(60000, 25000, n).clip(15000, 250000)
credit_score = rng.normal(680, 60, n).clip(400, 850)
debt_to_income = rng.uniform(0.05, 0.75, n)
loan_amount = rng.uniform(5000, 60000, n)
employment_years = rng.uniform(0, 20, n).round(1)

logit = (-3.4 - 2.0 * (credit_score - 680) / 60 + 2.0 * (debt_to_income - 0.4) / 0.2
         + 1.0 * (loan_amount - 32500) / 16000 - 0.75 * (income - 60000) / 24000
         - 0.5 * (employment_years - 10) / 5)
p_default = 1 / (1 + np.exp(-logit))
default = (rng.rand(n) < p_default).astype(int)

df = pd.DataFrame({
    "income": income.round(0), "credit_score": credit_score.round(0),
    "debt_to_income": debt_to_income.round(3), "loan_amount": loan_amount.round(0),
    "employment_years": employment_years, "default": default,
})
features = ["income", "credit_score", "debt_to_income", "loan_amount", "employment_years"]
X, y = df[features], df["default"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
fitted = {"Logistic Regression": LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)}

C_FP_loan, C_FN_loan = 1000, 10000     # rejected good customer vs approved default
t_star_loan = C_FP_loan / (C_FP_loan + C_FN_loan)
print(f"Loan costs: C_FP = {C_FP_loan}, C_FN = {C_FN_loan}")
print(f"Theoretical optimal threshold: t* = {C_FP_loan}/{C_FP_loan + C_FN_loan} = {t_star_loan:.4f}")

probs_loan = fitted["Logistic Regression"].predict_proba(X_test)[:, 1]
costs_loan = [expected_cost(y_test, probs_loan, t, C_FP_loan, C_FN_loan) for t in thresholds]
best_idx = int(np.argmin(costs_loan))
print(f"Empirical sweep optimum (the 0.10 found in Section 21): t = {thresholds[best_idx]:.2f}, "
      f"expected cost {costs_loan[best_idx]:.0f}/decision")
print(f"Expected cost at the default 0.5: "
      f"{expected_cost(y_test, probs_loan, 0.5, C_FP_loan, C_FN_loan):.0f}/decision")

# Net value of the model vs the 'do nothing' baseline (approve every loan)
baseline_cost = y_test.sum() * C_FN_loan / len(y_test)
print(f"\nBaseline 'approve every loan' (no model): {baseline_cost:.0f}/decision")
print(f"Model at t*: {costs_loan[best_idx]:.0f}/decision -> net savings "
      f"{baseline_cost - costs_loan[best_idx]:.0f}/decision") """))

CELLS_3.append(("md", r"""### Interactive: set the cost matrix and watch the threshold move 🎛️

Drag the two cost sliders and watch the **optimal threshold** (green line) move to
t* = C_FP/(C_FP + C_FN). The right panel compares the expected cost per decision at t*,
at the default 0.5, and for the "no model" baseline. Notice:

- Setting **C_FN ≫ C_FP** (misses expensive) pushes t* toward 0 — the bank approves
  almost everything to avoid missing a default.
- Setting **C_FP ≫ C_FN** pushes t* toward 1 — the bank acts only on near-certainty.
- The default slider values (1,000 / 10,000) reproduce the loan case study's optimum
  from Section 21.
"""))

CELLS_3.append(("code", r"""# --- Interactive: set the cost matrix and watch the optimal threshold move ---
# Uses the loan model rebuilt above (same generator as Section 21).
if "fitted" not in globals() or "y_test" not in globals():
    raise RuntimeError("Run the cells of this section first — the loan model is needed.")

try:
    import ipywidgets as widgets
    from IPython.display import display

    def cost_view(c_fp, c_fn):
        if c_fp + c_fn == 0:
            print("Set at least one cost above 0.")
            return
        t_star = c_fp / (c_fp + c_fn)
        cost_at_star = expected_cost(y_test, probs_loan, t_star, c_fp, c_fn)
        cost_at_05 = expected_cost(y_test, probs_loan, 0.5, c_fp, c_fn)
        baseline = y_test.sum() * c_fn / len(y_test)   # 'do nothing' baseline

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))

        # Left: expected cost curve with the theoretical optimum marked
        costs_all = [expected_cost(y_test, probs_loan, t, c_fp, c_fn) for t in thresholds]
        axes[0].plot(thresholds, costs_all, color="steelblue", label="Expected cost")
        axes[0].axvline(t_star, color="seagreen", linestyle="--", linewidth=2,
                        label=f"t* = {t_star:.2f}")
        axes[0].axvline(0.5, color="coral", linestyle=":", alpha=0.7, label="Default 0.5")
        axes[0].set_xlabel("Classification threshold")
        axes[0].set_ylabel(f"Expected cost (FP×{c_fp:,} + FN×{c_fn:,})")
        axes[0].set_title("Expected cost per decision vs threshold")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        # Right: cost at t* vs default 0.5 vs the no-model baseline
        axes[1].bar(["At t*", "At 0.5", "No model"],
                    [cost_at_star, cost_at_05, baseline],
                    color=["seagreen", "coral", "silver"])
        axes[1].set_ylabel("Expected cost per decision")
        axes[1].set_title("Where to operate?")
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()

        print(f"C_FP = {c_fp:,}, C_FN = {c_fn:,}  ->  t* = {t_star:.3f}")
        print(f"Expected cost at t*: {cost_at_star:,.1f}  |  at 0.5: {cost_at_05:,.1f}  "
              f"|  no model: {baseline:,.1f}  (per decision)")
        print(f"Net savings vs no model: {baseline - cost_at_star:,.1f} per decision")

    slider_fp = widgets.IntSlider(value=1000, min=0, max=5000, step=100, description="Cost FP:")
    slider_fn = widgets.IntSlider(value=10000, min=0, max=20000, step=500, description="Cost FN:")

    out = widgets.interactive_output(cost_view, {"c_fp": slider_fp, "c_fn": slider_fn})
    with out:
        cost_view(1000, 10000)   # reproduce the loan case study's optimum
    display(widgets.VBox([widgets.HBox([slider_fp, slider_fn]), out]))
    print("✅ Drag the cost sliders and watch the optimal threshold move.")

except ImportError:
    print("ipywidgets is not installed, so the interactive slider is unavailable.")
    print("Install it with:  pip install ipywidgets   (then restart the kernel and re-run this cell).")
    print("Until then, the static cost curve above shows the same relationship.")"""))

CELLS_3.append(("md", r"""**What did we learn from these outputs?**

In the first demo, the theoretical rule t* = 200/(200 + 2000) ≈ 0.09 predicts exactly
where the empirical cost curve bottoms out — while the default 0.5 threshold costs far
more per decision, and both "predict everything negative" and "predict everything
positive" baselines are worse than the model at t*.

In the loan demo, the formula gives t* = 1000/11000 ≈ 0.09 — the neighborhood of the
0.10 threshold the loan case study found empirically, and the reason that threshold
minimized cost. The expected-cost-per-decision view also shows the model's net savings
against the "approve every loan" baseline: the business value of the model, in currency
per decision.

**What would this mean in a real-world application?**

A bank that knows its cost matrix can compute the optimal decision rule in one line —
*before* training anything — and can approve a model for deployment by asking "does it
save more than it costs?" rather than "what is its F1?" The interactive lab above makes
this immediate: move the cost sliders and the formula gives you the new operating point
instantly.

---

#### Beginner Perspective

*Given C_FP (cost of a false alarm) and C_FN (cost of a miss), the best threshold is
t* = C_FP/(C_FP + C_FN) — not 0.5. The threshold is a business formula, not a default.*

#### Expert Perspective

*Experts treat the threshold as the output of expected-cost minimization under a cost
matrix and remember it is only optimal when probabilities are well calibrated (Section
13). They generalize to per-group cost matrices, operational constraints, and full value
matrices with profits on correct decisions — and they evaluate models on net business
value against a no-model baseline, not on isolated metrics.*

---

[⬅ Previous](#sec22) · [🏠 Table of Contents](#toc) · [Next ➡](#sec24)"""))

CELLS_3.append(("md", r"""📗 **Back to basics:** see **Section 14: Mini Case Study: Loan Default
Prediction** in `ML_Model_Evaluation_Simple.ipynb` — the cost table (FN = 10,000 vs
FP = 1,000) and the cost-vs-threshold curve.
"""))

CELLS_3.append(("md", r"""<a id="sec24"></a>
## 24. Choosing the Right Metric — A Selection Guide

### The guide

| Scenario | FP means | FN means | More costly error | Emphasized metric | Threshold note |
|---|---|---|---|---|---|
| **Spam detection** | Legitimate email filtered | Spam in the inbox | FP often (missed real mail) | **Precision** | Raise threshold to be more conservative |
| **Disease screening** | Healthy person alarmed | Sick person missed | FN (dangerous) | **Recall / Sensitivity** | Lower threshold to catch more cases |
| **Fraud detection** | Legitimate customer blocked | Fraud not caught | Depends (usually FN at scale; FP if reviews are expensive) | **Recall** or **PR-AUC**; F1 if balanced | Tune by cost of FP vs FN |
| **Intrusion detection** | Normal traffic flagged | Real attack missed | FN (breach) | **Recall** | Lower threshold; accept more alerts |
| **Loan approval** | Loan given, defaults | Good customer rejected | Depends on portfolio; FP (loss of principal) often heavier | **Precision + cost-based threshold** | Raise threshold if defaults are costly |
| **Customer churn** | Offer to loyal customer | Churner gets no offer | FN (lost revenue) | **Recall** | Lower threshold; watch offer budget |
| **Recommendation system** | Bad recommendation shown | Good item never shown | FP (user annoyance) | **Precision** (top-k) | Rank-based evaluation, not threshold |
| **Image classification** | Wrong class label | Wrong class label | Class-dependent | **Per-class metrics**, macro F1 | Per-class thresholds if needed |
| **Imbalanced classification** | False alarm on majority | Missed rare event | Almost always FN | **PR-AUC, Recall, F1** | Never accuracy alone |

### The general decision framework

```
    What are we predicting?
            ↓
      What does FP mean?
            ↓
      What does FN mean?
            ↓
    Which error is more expensive?
            ↓
    Which metric reflects that concern?
            ↓
     What threshold should we use?
            ↓
   Is the model useful in practice?
```

### The three questions, again

For **every** evaluation you will ever do:

1. **Technical:** Which model has the highest metric? *(easiest to answer)*
2. **Data Science:** Which metric matters for this problem? *(harder — requires the next
   question)*
3. **Real-World:** What are the consequences if the model is wrong? *(the question that
   actually decides everything)*

If you can only remember one thing from this notebook, remember that metrics are **not
scores to maximize** — they are **lenses for looking at the consequences of errors**.

---

[⬅ Previous](#sec23) · [🏠 Table of Contents](#toc) · [Next ➡](#sec25)"""))
