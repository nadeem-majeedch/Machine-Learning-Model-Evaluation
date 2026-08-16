# Content part 2: Sections 8–15
# Each item is a tuple: ("md", text) or ("code", source)

CELLS_2 = []

CELLS_2.append(("md", r"""📗 **Back to basics:** see **Section 6: Specificity** in
`ML_Model_Evaluation_Simple.ipynb` — the same formula with a simpler example.
"""))

CELLS_2.append(("md", r"""<a id="sec8"></a>
## 8. F1 Score

### A. Simple Definition

The F1 score is the **harmonic mean of precision and recall** — a single number that
summarizes both.

### B. Intuition

Precision alone can be high while the model misses everything; recall alone can be high
while the model cries wolf. F1 punishes models that are extreme in either direction: it is
only high when **both** precision and recall are decent. Think of it as the "balanced
score" for the positive class.

### C. Formula

$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Why the harmonic mean and not the arithmetic mean?

The harmonic mean is much more sensitive to small values. If precision = 0.9 and
recall = 0.1:

- Arithmetic mean = (0.9 + 0.1) / 2 = **0.50**
- Harmonic mean = 2·(0.9·0.1)/(0.9+0.1) = 0.18/1.0 = **0.18**

The harmonic mean correctly tells you: *this model is bad*, because it misses 90% of the
positives — a fact the arithmetic mean would hide. F1 can never exceed the smaller of
precision and recall, which is exactly the property we want.

### D. Meaning of Each Component

| Component | Meaning |
|---|---|
| Precision | Trust in positive predictions (FP control) |
| Recall | Ability to find positives (FN control) |
| F1 | Balance between the two |

### E. Small Numerical Examples — three situations

| Situation | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| High precision, low recall | 30 | 2 | 60 | 0.938 | 0.333 | **0.49** |
| Low precision, high recall | 85 | 80 | 5 | 0.515 | 0.944 | **0.67** |
| Balanced | 60 | 20 | 20 | 0.750 | 0.750 | **0.75** |

F1 is highest when the two are balanced — but "highest F1" is not automatically the best
model for the business, as we will see below.
"""))

CELLS_2.append(("code", r"""# --- F1 Score: the three situations ---
def report(name, tp, tn, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print(f"{name:34s} P = {precision:.3f}  R = {recall:.3f}  F1 = {f1:.3f}")

report("High precision, low recall",  tp=30, tn=40, fp=2,  fn=60)
report("Low precision, high recall",  tp=85, tn=10, fp=80, fn=5)
report("Balanced",                    tp=60, tn=40, fp=20, fn=20)

# scikit-learn on the balanced example
from sklearn.metrics import f1_score
y_true = np.array([1] * 80 + [0] * 60)          # 80 actual positives, 60 actual negatives
y_pred = np.array([1] * 60 + [0] * 20 + [1] * 20 + [0] * 40)  # 60 TP, 20 FN, 20 FP, 40 TN
print(f"\nsklearn F1 (balanced example) = {f1_score(y_true, y_pred):.3f}")"""))

CELLS_2.append(("md", r"""### F. Real-Life Beginner Scenario

A teacher's model predicts which students will pass an exam. Precision says "of the
students I predicted would pass, how many really did?" Recall says "of the students who
really passed, how many did I predict?" F1 combines both into one "how good am I at this
prediction task overall?" number.

### G. Real-Life Expert Scenario — Customer churn

A telecom company predicts which customers will churn, so it can send retention offers.

- FP = offer sent to a customer who was not going to churn → wasted offer (cost).
- FN = churning customer gets no offer → **customer lost (revenue loss)**.

### H. FP/FN Business Impact

| Error | In this scenario | Consequence |
|---|---|---|
| FP | Offer to a loyal customer | Wasted money, possible irritation ("why are you begging me to stay?") |
| FN | No offer to a churning customer | Lost recurring revenue — usually far more expensive than the offer |

### I. Metric Selection

**Use F1 when you want a single number that balances precision and recall** — e.g., when
FP and FN costs are *roughly similar* or unknown, or when you are comparing models across
many thresholds and need one summary. It is the default "balanced" score for the positive
class.

### J. Threshold Consideration

F1 is usually reported at one threshold (often 0.5). Because precision and recall move in
opposite directions with the threshold, F1 has a **maximum at some intermediate
threshold** — the "balance point". If the business cares more about one error than the
other, the best threshold is *not* the F1-maximizing one.

### K. When to Use It

Imbalanced classification with roughly symmetric error costs; model selection over many
candidates; summarizing performance in a single number for non-technical audiences.

### L. When NOT to Use It

When FP and FN costs are very different. F1 treats both errors equally, so it cannot
represent a problem where one error costs 100× the other. **Never use F1 as the only
metric when error costs are asymmetric.**

### M. Advantages and Limitations

| ✅ Advantages | ❌ Limitations |
|---|---|
| Single balanced number for precision + recall | Treats FP and FN as equally costly |
| Works well on imbalanced data | Hides the individual precision/recall values |
| Easy to compare across models | Maximizing F1 ≠ minimizing business cost |

### The crucial question: is the model with the highest F1 always the best model?

**No.** Consider a medical screening test:

- Model X: F1 = 0.80 (precision 0.95, recall 0.69)
- Model Y: F1 = 0.78 (precision 0.65, recall 0.97)

Model X has the higher F1, but it *misses 31% of sick patients*. Model Y finds 97% of
them at the price of more false alarms. If the disease is dangerous, **Model Y is the
better business solution despite the lower F1** — because in this problem FN is the
expensive error, and F1 does not know that. This is why we always translate metrics back
into error costs.

---

#### Beginner Perspective

*F1 = harmonic mean of precision and recall. It is high only when both are decent. But
"highest F1" does not automatically mean "best for the business".*

#### Expert Perspective

*Experts generalize F1 with the Fβ score, which weights recall β times more than
precision (F2 favors recall, F0.5 favors precision), and they treat even that as a proxy:
the truly correct objective is the expected cost of errors — cost(FP)·FP + cost(FN)·FN —
which may have its optimum at a completely different threshold than F1's.*

---

[⬅ Previous](#sec7) · [🏠 Table of Contents](#toc) · [Next ➡](#sec9)"""))

CELLS_2.append(("md", r"""📗 **Back to basics:** see **Section 7: F1 Score** in
`ML_Model_Evaluation_Simple.ipynb` — the harmonic-mean intuition with a small table of
examples and the same warning.
"""))

CELLS_2.append(("md", r"""<a id="sec9"></a>
## 9. The Precision–Recall Trade-off

### What is a classification threshold?

Most real classifiers do not output a final label directly. They output a **probability**
(a number between 0 and 1) — "how confident am I that this is positive?" — and we convert
that probability into a label by comparing it to a **threshold**:

$$\text{prediction} = \begin{cases} \text{positive}, & \text{if } p \geq \text{threshold} \\ \text{negative}, & \text{otherwise} \end{cases}$$

The default threshold is usually 0.5, but it is **not sacred**. It is a decision knob.

### What happens when we move the threshold?

- **Lower threshold** → the model predicts "positive" more easily → it finds **more true
  positives (higher recall)** but also raises **more false alarms (lower precision)**.
- **Higher threshold** → the model predicts "positive" only when very confident → **fewer
  false alarms (higher precision)** but **more missed positives (lower recall)**.

Precision and recall trade off against each other. You cannot have both at their maximum
simultaneously — the only question is *where* to balance them, and that is a business
decision.

Let's see it with real probabilities.
"""))

CELLS_2.append(("code", r"""# --- Threshold sweep: precision and recall as functions of the threshold ---
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix

# Synthetic dataset: 20% positive class
X, y = make_classification(n_samples=500, n_features=5, n_informative=4, n_redundant=0,
                           weights=[0.8, 0.2], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]   # predicted probabilities of the positive class

thresholds = np.arange(0.05, 1.00, 0.05)
precisions, recalls = [], []
for t in thresholds:
    preds = (probs >= t).astype(int)
    precisions.append(precision_score(y_test, preds, zero_division=0))
    recalls.append(recall_score(y_test, preds, zero_division=0))

plt.figure(figsize=(8.5, 4.5))
plt.plot(thresholds, precisions, "o-", label="Precision", color="steelblue")
plt.plot(thresholds, recalls, "s-", label="Recall", color="coral")
plt.xlabel("Classification threshold")
plt.ylabel("Score")
plt.title("Precision and Recall vs Classification Threshold")
plt.legend()
plt.grid(alpha=0.3)
plt.show()"""))

CELLS_2.append(("code", r"""# --- The same sweep, but looking at the actual errors (FP and FN) ---
print(f"{'Threshold':>9s} {'Precision':>9s} {'Recall':>9s} {'FP':>4s} {'FN':>4s} {'Flagged':>7s}")
for t in [0.15, 0.30, 0.50, 0.70, 0.85]:
    preds = (probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    p = precision_score(y_test, preds, zero_division=0)
    r = recall_score(y_test, preds, zero_division=0)
    print(f"{t:>9.2f} {p:>9.3f} {r:>9.3f} {fp:>4d} {fn:>4d} {tp + fp:>7d}")"""))

CELLS_2.append(("md", r"""**What did we learn from this output?**

As the threshold rises, precision climbs and recall falls — the classic trade-off. The
error table shows *why*: at threshold 0.15 the model flags many cases (high recall, many
FP); at 0.85 it flags few (high precision, many FN). The 0.5 default is just one point on
this curve, and often not the best one for the business.

**What would this mean in a real-world application — fraud detection?**

- **Low threshold** → more fraud detected, but more legitimate customers blocked and more
  alerts for analysts to investigate.
- **High threshold** → fewer false alarms, but more fraud slips through undetected.

If blocking a legitimate customer is cheap to undo but a missed fraud is expensive, you
operate at a low threshold and accept the false alarms. If every alert costs an hour of an
analyst's time, you operate at a high threshold and accept the missed frauds. **The
"right" threshold is a business decision, not a technical default.**

---

#### Beginner Perspective

*Threshold down → more positives → higher recall, lower precision. Threshold up → fewer
positives → higher precision, lower recall. You cannot have both at once.*

#### Expert Perspective

*Experts treat the threshold as an *operating point* chosen to optimize an explicit cost
function (Section 21 shows this in practice), and they evaluate the whole trade-off curve
(ROC / PR, Sections 10–12) rather than a single threshold. They also account for
prevalence drift: a threshold tuned on last year's data may be wrong this year if the
positive rate changes.*

---

[⬅ Previous](#sec8) · [🏠 Table of Contents](#toc) · [Next ➡](#sec10)"""))

CELLS_2.append(("md", r"""📗 **Back to basics:** see **Section 8: The Precision–Recall Trade-off** in
`ML_Model_Evaluation_Simple.ipynb` — thresholds and the trade-off on a small fraud
dataset, with the same business framing.
"""))

CELLS_2.append(("md", r"""<a id="sec10"></a>
## 10. The ROC Curve

### What is the ROC curve?

The **Receiver Operating Characteristic** curve plots, for **every possible threshold**,
two quantities against each other:

- **True Positive Rate (TPR)** = recall = TP / (TP + FN) — how many positives we catch
- **False Positive Rate (FPR)** = FP / (FP + TN) = 1 − specificity — how many negatives we
  wrongly flag

$$\text{TPR} = \frac{TP}{TP + FN} \qquad \text{FPR} = \frac{FP}{FP + TN}$$

### Reading the curve

- Each point on the curve = one threshold.
- Moving **up-left** along the curve = a more conservative (higher) threshold: fewer false
  alarms, but also fewer positives found.
- Moving **down-right** = a more permissive (lower) threshold: more positives found, but
  more false alarms.
- The **diagonal** line (from bottom-left to top-right) = a completely **random
  classifier**: for any TPR you can achieve, FPR is just as high. A useful model sits
  clearly above the diagonal.

### The business interpretation

Increasing sensitivity (recall) almost always increases the false positive rate. The ROC
curve shows this trade-off explicitly for *all* thresholds at once — which is why ROC
analysis must always be connected back to the actual cost of FP and FN. A point that is
"perfect" on the curve may be the worst point for the business if it floods your
operations with false alarms.

Let's build one.
"""))

CELLS_2.append(("code", r"""# --- ROC curve ---
from sklearn.metrics import roc_curve

# Fresh model so this cell runs standalone
X, y = make_classification(n_samples=500, n_features=5, n_informative=4, n_redundant=0,
                           weights=[0.8, 0.2], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
model = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, probs)

plt.figure(figsize=(6.5, 5.5))
plt.plot(fpr, tpr, "b-", linewidth=2, label="Logistic Regression")
plt.plot([0, 1], [0, 1], "k--", label="Random classifier (diagonal)")

# Mark a few operating points (thresholds) on the curve
for t in [0.2, 0.5, 0.8]:
    idx = int(np.argmin(np.abs(thresholds - t)))
    plt.plot(fpr[idx], tpr[idx], "ro", markersize=7)
    plt.annotate(f"threshold = {t}", (fpr[idx], tpr[idx]),
                 textcoords="offset points", xytext=(8, -10), fontsize=9)

plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()"""))

CELLS_2.append(("md", r"""**What did we learn from this output?**

The curve starts at (0, 0) — a threshold so high that nothing is flagged (no positives
found, no false alarms) — and ends at (1, 1) — a threshold so low that everything is
flagged (all positives found, all negatives flagged too). The red dots show three
specific operating thresholds: 0.8 is conservative (high up-left), 0.2 is aggressive
(low down-right). The whole curve is the collection of *every* possible trade-off.

**What would this mean in a real-world application?**

A bank choosing where to operate picks a *point on the curve*, not the whole curve. If
they choose the 0.2 point, they catch more fraud but must investigate many more false
alarms. The curve tells you *what trade-offs exist*; only the cost analysis tells you
*which one to pick*.

---

#### Beginner Perspective

*ROC plots recall (TPR) against the false alarm rate (FPR) for every possible threshold.
Above the diagonal = better than random. Farther up-left = better discrimination.*

#### Expert Perspective

*ROC curves are threshold-independent summaries of ranking ability, and are robust to
class imbalance in a specific sense (they do not change when the prevalence changes).
But on heavily imbalanced data they can look optimistic, because the FPR denominator (TN)
is huge — see the ROC vs PR comparison in Section 12.*

---

[⬅ Previous](#sec9) · [🏠 Table of Contents](#toc) · [Next ➡](#sec11)"""))

CELLS_2.append(("md", r"""📗 **Back to basics:** see **Section 9: ROC Curve and AUC** in
`ML_Model_Evaluation_Simple.ipynb` — the ROC curve and AUC on the same fraud dataset,
without the deeper math.
"""))

CELLS_2.append(("md", r"""<a id="sec11"></a>
## 11. AUC — Area Under the ROC Curve

### A. Simple Definition

**AUC** (Area Under the ROC Curve) is a single number summarizing the whole ROC curve:
the area beneath it, ranging from 0 to 1.

### B. Intuition — the rank interpretation

AUC answers: *"if I pick one random positive and one random negative, how often does the
model give the positive the higher score?"* It measures how well the model **separates**
positive from negative cases in its ranking — its **discrimination** ability — across all
thresholds at once.

### C. Interpretation of values

| AUC | Meaning |
|---|---|
| **1.0** | Perfect ranking: every positive is scored above every negative |
| **0.5** | Random ranking: no better than a coin flip |
| **< 0.5** | Worse than random — you could flip the predictions and do better |
| **0.7–0.8** | Usually considered "acceptable" discrimination |
| **0.9+** | Usually considered "excellent" discrimination |

### D. Advantages

- A single number, easy to compare models.
- Threshold-independent: describes the model's ranking ability without committing to a
  specific operating point.
- Widely used and understood.

### E. Limitations — the part students often miss

1. **AUC does not tell you which threshold to deploy.** Two models with AUC 0.9 can
   require completely different thresholds to achieve the same recall; the curve does not
   name your operating point.
2. **On imbalanced data, AUC can look optimistically high** (see Section 12).
3. **AUC hides where errors happen** — two very different curves can have the same area
   (one good at low FPR, the other at high FPR), and those are different business
   situations.
4. **AUC is a ranking measure, not a calibration measure.** A model can rank perfectly
   while its probabilities are systematically wrong — which matters when the *probability*
   itself drives a decision (e.g., loan pricing). (We evaluate this in Section 13.)

> **Key message:** AUC summarizes *ranking/discrimination* performance across thresholds,
> but it does NOT automatically tell us which threshold is appropriate for deployment.
"""))

CELLS_2.append(("code", r"""# --- Computing AUC in Python ---
from sklearn.metrics import roc_auc_score

# Reuse the model from the ROC section
fpr, tpr, _ = roc_curve(y_test, probs)
auc_value = roc_auc_score(y_test, probs)
print(f"ROC-AUC = {auc_value:.3f}")

# Sanity-check the extremes with synthetic scores
rng = np.random.RandomState(0)
y = np.array([1] * 500 + [0] * 500)

# Perfect separation: every positive scored above every negative -> AUC = 1.0
perfect_scores = np.concatenate([np.ones(500) * 0.99, np.ones(500) * 0.01])

# Random scores: shuffle the perfect scores so there is NO relationship with the
# label -> AUC should come out very close to 0.5
random_scores = rng.permutation(perfect_scores)

# Reversed scores: perfect in the WRONG direction -> AUC = 0.0
reversed_scores = np.concatenate([np.ones(500) * 0.01, np.ones(500) * 0.99])

print(f"AUC with perfect ranking      = {roc_auc_score(y, perfect_scores):.3f}")
print(f"AUC with random scores        = {roc_auc_score(y, random_scores):.3f}")
print(f"AUC with reversed ranking     = {roc_auc_score(y, reversed_scores):.3f}")"""))

CELLS_2.append(("md", r"""**What did we learn from this output?**

The real model scores an AUC of about 0.9x — meaning a random positive is scored above a
random negative about 90% of the time. The extremes behave as expected: perfect ranking
→ 1.0, random → ~0.5, reversed → 0.0. Note that a "reversed" model (AUC < 0.5) is not
useless — it is a signal that you should invert your predictions or debug your labels.

**What would this mean in a real-world application?**

If a fraud model has AUC 0.9, it ranks frauds above legitimate transactions very
reliably — but the bank still has to pick a threshold to decide *which transactions to
block*. AUC describes the model; the threshold, the costs, and the operational capacity
describe the decision.

---

#### Beginner Perspective

*AUC = "how well does the model separate the two classes across all thresholds?" 0.5 is
random, 1.0 is perfect. Higher is better — but it is not the whole story.*

#### Expert Perspective

*Experts remember AUC is a *relative ranking* measure: it cannot be compared across
datasets with different difficulty or prevalence, it rewards no absolute probability
accuracy, and on rare-event problems PR-AUC (next section) is usually the more honest
summary. They also compare models on the region of the ROC curve where the business will
actually operate, not on the full area.*

---

[⬅ Previous](#sec10) · [🏠 Table of Contents](#toc) · [Next ➡](#sec12)"""))

CELLS_2.append(("md", r"""📗 **Back to basics:** see **Section 9: ROC Curve and AUC** in
`ML_Model_Evaluation_Simple.ipynb` — AUC as “the model ranks positives above
negatives”, in one short paragraph.
"""))

CELLS_2.append(("md", r"""<a id="sec12"></a>
## 12. ROC-AUC vs Precision–Recall

### Why can't we just use ROC-AUC for everything?

The ROC curve plots TPR against FPR. FPR = FP / (FP + TN) — its denominator includes all
actual negatives, which on imbalanced data is a **huge** number. A model can therefore
achieve a very low FPR (and a high ROC-AUC) even while producing *many* false positives in
absolute terms, simply because the negative class is enormous. ROC-AUC looks at the
problem from the majority class's perspective.

The **Precision–Recall (PR) curve** plots precision against recall and focuses directly
on the **minority (positive) class**:

- **Precision** = TP / (TP + FP) — how many flagged cases are real
- **Recall** = TP / (TP + FN) — how many real positives are found

On imbalanced data, the PR curve is more informative about *how useful the model is for
finding the rare event*, because it punishes false positives directly.

### The comparison table

| Metric | What it tells us | Strength | Limitation | Useful scenario |
|---|---|---|---|---|
| **ROC-AUC** | Overall ranking quality across all thresholds | Threshold-independent, prevalence-invariant, standard | Optimistic on imbalanced data; hides operating point | Balanced data, model comparison, ranking ability |
| **Precision–Recall curve** | Precision at every recall level | Directly measures usefulness for the rare positive class | Sensitive to prevalence (can't compare across datasets) | Rare-event detection: fraud, disease, intrusion |
| **PR-AUC** | Single summary of the PR curve | Captures the FP burden on the minority-class problem | Harder to explain; needs prevalence context | Comparing models on imbalanced data |

### The business example

A fraud team wants to know: *"if we act on the top 10% of suspicious transactions, how
many will be real frauds?"* — that is a precision question, read off the PR curve. The ROC
curve cannot answer it directly, because its axes are about *rates* within each class, not
about the actual composition of the flagged set.

Let's demonstrate with two synthetic models that have **almost identical ROC-AUC but very
different PR-AUC**.
"""))

CELLS_2.append(("code", r"""# --- ROC-AUC vs PR-AUC on an imbalanced problem ---
from sklearn.metrics import precision_recall_curve, average_precision_score

rng = np.random.RandomState(7)
n_pos, n_neg = 200, 3800                      # 5% positive class (rare event)
y = np.array([1] * n_pos + [0] * n_neg)

# Model X: decent separation, but 10% of negatives form a high-scoring "borderline"
# cluster that produces many false alarms -> good ranking, bad precision
scores_x = np.concatenate([
    rng.normal(1.0, 0.6, n_pos),                     # positives
    rng.normal(0.2, 0.3, int(0.90 * n_neg)),         # most negatives, low scores
    rng.normal(2.0, 0.7, n_neg - int(0.90 * n_neg))  # a cluster of false-alarm-prone negatives
])

# Model Y: same kind of separation, but negatives spread far below the positives
scores_y = np.concatenate([
    rng.normal(1.6, 0.9, n_pos),
    rng.normal(0.3, 0.95, n_neg)
])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for scores, color, name in [(scores_x, "steelblue", "Model X"),
                            (scores_y, "coral", "Model Y")]:
    # ROC curve + AUC
    fpr, tpr, _ = roc_curve(y, scores)
    auc_roc = roc_auc_score(y, scores)
    axes[0].plot(fpr, tpr, color=color, linewidth=2,
                 label=f"{name} (ROC-AUC = {auc_roc:.3f})")

    # PR curve + PR-AUC
    precision, recall, _ = precision_recall_curve(y, scores)
    auc_pr = average_precision_score(y, scores)
    axes[1].plot(recall, precision, color=color, linewidth=2,
                 label=f"{name} (PR-AUC = {auc_pr:.3f})")

axes[0].plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.5)")
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curves — look almost the same")
axes[0].legend(loc="lower right"); axes[0].grid(alpha=0.3)

# A random classifier's PR baseline on imbalanced data is the prevalence (0.05)
axes[1].axhline(0.05, color="k", linestyle="--", label="Random baseline (prevalence 0.05)")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision–Recall Curves — clearly different")
axes[1].legend(loc="upper right"); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

roc_x = roc_auc_score(y, scores_x); roc_y = roc_auc_score(y, scores_y)
ap_x = average_precision_score(y, scores_x); ap_y = average_precision_score(y, scores_y)
print(f"Model X: ROC-AUC = {roc_x:.3f}   PR-AUC = {ap_x:.3f}")
print(f"Model Y: ROC-AUC = {roc_y:.3f}   PR-AUC = {ap_y:.3f}")"""))

CELLS_2.append(("md", r"""**What did we learn from this output?**

The two ROC curves are nearly indistinguishable (ROC-AUC ≈ 0.80 vs ≈ 0.82). Judging by
ROC-AUC alone, you might call the models equivalent. But the PR curves tell a different
story: Model Y's PR-AUC is almost **twice** Model X's (≈ 0.25 vs ≈ 0.14). Model X looks
fine by ROC, yet in practice it would flood the fraud team with false positives.

**What would this mean in a real-world application?**

A fraud detection team that compares these models on ROC-AUC might deploy Model X and then
discover in production that most of its alerts are false alarms. Comparing on PR-AUC (or
on precision at the operating recall) reveals the truth: Model Y is the better tool for
finding the rare frauds. **On imbalanced problems, prefer PR curves for decision-making.**

> Rule of thumb: use **ROC-AUC** for overall ranking quality and balanced data; use
> **PR-AUC / PR curves** when the positive class is rare and false positives are costly.

---
"""))

CELLS_2.append(("md", r"""### Interactive explorer: one model, two curves, one operating point 🎛️

The curves above are two views of the *same* model. The widget below fits a fresh
logistic-regression model on a synthetic **fraud-detection dataset (5% fraud)** and lets
you drag the threshold. Watch how **one operating point** moves along *both* curves at
once:

- **Left panel:** precision, recall, and F1 as functions of the threshold.
- **Middle panel:** the ROC curve — the operating point shows the recall (TPR) you get
  at the current false-positive rate.
- **Right panel:** the PR curve — the operating point shows the precision you get at the
  current recall, compared with the random baseline (the fraud rate, 0.05).

Experiments to try:

1. Start at threshold **0.9** (very strict) and drag down to **0.1**. Watch recall climb
   along the ROC curve while precision collapses on the PR curve.
2. Find where **F1 is highest** — is that the point you would actually deploy for fraud
   detection? (Think about the cost of each false alarm!)
3. Notice that the ROC curve stays high even when precision is terrible — the ROC view
   makes this model look better than the PR view does. Which curve do you trust when
   only 5% of transactions are fraud?
"""))

CELLS_2.append(("code", r"""# --- Interactive: threshold sweep on the ROC and PR curves (fraud detection) ---
# Self-contained cell: builds a small synthetic fraud dataset + logistic regression here,
# so it works even if run on its own after a kernel restart.
from sklearn.linear_model import LogisticRegression

X_f, y_f = make_classification(n_samples=1500, n_features=6, n_informative=5, n_redundant=0,
                               weights=[0.95, 0.05], flip_y=0.02, random_state=42)
X_f_train, X_f_test, y_f_train, y_f_test = train_test_split(
    X_f, y_f, test_size=0.3, random_state=42, stratify=y_f)

fraud_model = LogisticRegression(max_iter=1000, random_state=42).fit(X_f_train, y_f_train)
fraud_probs = fraud_model.predict_proba(X_f_test)[:, 1]


# One threshold -> (precision, recall, f1, fpr, and the four confusion-matrix counts)
def curve_point(y_true, y_prob, t):
    preds = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return prec, rec, f1, fpr, tn, fp, fn, tp


# Precompute the three curves on a fine grid of thresholds (0.01 ... 0.99)
grid = np.arange(0.01, 1.00, 0.01)
prec_grid, rec_grid, f1_grid, fpr_grid = [], [], [], []
for t in grid:
    prec, rec, f1, fpr, *_ = curve_point(y_f_test, fraud_probs, t)
    prec_grid.append(prec); rec_grid.append(rec)
    f1_grid.append(f1);   fpr_grid.append(fpr)


try:
    import ipywidgets as widgets
    from IPython.display import display

    def sweep(threshold):
        # Exact values at the current slider position
        prec, rec, f1, fpr, tn, fp, fn, tp = curve_point(y_f_test, fraud_probs, threshold)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

        # Left: precision / recall / F1 vs threshold, with the current point marked
        axes[0].plot(grid, prec_grid, label="Precision", color="steelblue")
        axes[0].plot(grid, rec_grid, label="Recall", color="coral")
        axes[0].plot(grid, f1_grid, label="F1", color="seagreen")
        axes[0].axvline(threshold, color="gray", linestyle="--", alpha=0.8)
        axes[0].plot(threshold, prec, "o", color="steelblue")
        axes[0].plot(threshold, rec, "o", color="coral")
        axes[0].plot(threshold, f1, "o", color="seagreen")
        axes[0].set_xlabel("Classification threshold"); axes[0].set_ylabel("Score")
        axes[0].set_title("Precision, Recall, F1 vs threshold")
        axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

        # Middle: ROC curve with the operating point (FPR, TPR)
        axes[1].plot(fpr_grid, rec_grid, color="steelblue", label="ROC curve")
        axes[1].plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random (AUC = 0.5)")
        axes[1].plot(fpr, rec, "o", color="coral", markersize=10)
        axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate (Recall)")
        axes[1].set_title("ROC curve — operating point")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

        # Right: PR curve with the operating point (Recall, Precision) + random baseline
        axes[2].plot(rec_grid, prec_grid, color="steelblue", label="PR curve")
        axes[2].axhline(0.05, color="k", linestyle="--", alpha=0.6, label="Random (prevalence 0.05)")
        axes[2].plot(rec, prec, "o", color="coral", markersize=10)
        axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision")
        axes[2].set_title("Precision–Recall curve — operating point")
        axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        print(f"threshold = {threshold:.2f}  |  TP = {tp}  FP = {fp}  FN = {fn}  TN = {tn}")
        print(f"Precision = {prec:.3f}   Recall = {rec:.3f}   F1 = {f1:.3f}")

    slider = widgets.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.01,
                                 description="Threshold:", readout_format=".2f")
    out = widgets.interactive_output(sweep, {"threshold": slider})
    with out:
        sweep(0.5)   # embed a static snapshot of the initial view
    display(widgets.VBox([slider, out]))
    print("✅ Drag the threshold and watch the operating point move along both curves.")

except ImportError:
    print("ipywidgets is not installed, so the interactive slider is unavailable.")
    print("Install it with:  pip install ipywidgets   (then restart the kernel and re-run this cell).")
    print("Until then, use the static curves in Section 9 and this section: pick a threshold and trace it on both curves.")"""))

CELLS_2.append(("md", r"""#### Beginner Perspective

*On imbalanced data, ROC can look great while the model is actually poor at finding the
rare class. PR curves focus on the class you care about — check both, trust PR for
rare-event problems.*

#### Expert Perspective

*Experts treat ROC-AUC and PR-AUC as answers to different questions (ranking vs
usefulness for the minority class) and know PR-AUC is prevalence-dependent, so it cannot
be compared across datasets with different base rates. They often skip the summary
entirely and evaluate precision at the *specific* recall (or business cost) they intend
to operate at.*

---

[⬅ Previous](#sec11) · [🏠 Table of Contents](#toc) · [Next ➡](#sec13)"""))

CELLS_2.append(("md", r"""📗 **Back to basics:** only here in depth — the nearest basic material is
**Section 9: ROC Curve and AUC** in `ML_Model_Evaluation_Simple.ipynb`, whose caveat
explains why a precision–recall curve is more informative for rare events.
"""))

CELLS_2.append(("md", r"""<a id="sec13"></a>
## 13. Probability Calibration

### A. Simple Definition

Calibration answers one question:

> **"When the model says '70%', is it right 70% of the time?"**

A predicted probability of 0.70 should mean what it says: if we collect all the cases
where the model predicted 0.70, about 70% of them should actually be positive.

### B. Intuition — the weather forecast

A weather app says "30% chance of rain". If, over a year, it rains on about 3 out of
every 10 days that the app rated 30%, the forecast is **well calibrated**. If it only
rains on 1 of those days, the app is **overconfident** — the number 0.30 did not mean
what it claimed. The same logic applies to fraud, disease, and default probabilities.

### C. Ranking vs calibration — two different things

- **Ranking** (what AUC measures, Section 11): does the model put the positives above
  the negatives? Only the *order* of the scores matters.
- **Calibration** (this section): do the *absolute values* of the scores match reality?

**A model can rank perfectly and still be badly miscalibrated.** Imagine a model whose
scores are all inflated: true probabilities of 0.1, 0.4, 0.6, 0.9 come out as 0.5, 0.8,
0.9, 0.99. The order is preserved — the AUC is identical to a perfectly calibrated
model's — but every number overstates the true risk. If the business *uses the number*
to price a loan, set an insurance premium, or decide how urgently to act, miscalibration
means systematically wrong decisions.

### D. Why do models become miscalibrated?

1. Most training objectives optimize ranking or accuracy, not the truth of the
   probabilities themselves.
2. Different model families have very different natural calibration:
   - **Logistic regression** is usually well calibrated — it is explicitly a
     probabilistic model.
   - **Random forests / decision trees** tend to be **overconfident**: their
     probabilities bunch up near 0 and 1 (leaves become "pure", so the model claims 90%
     or 100% where reality is closer to 70%).
   - **Deep neural networks** are often overconfident too (well known from
     adversarial-example research).
3. Class imbalance, regularization, and class weights distort probabilities further.

### E. How to check: the reliability diagram

A **reliability diagram** (also called a calibration curve) is built like this:

1. Collect all predicted probabilities.
2. Split them into **bins** (e.g., deciles: 0–0.1, 0.1–0.2, ..., 0.9–1.0).
3. For each bin, compute the **mean predicted probability** and the **observed
   frequency** of positives.
4. Plot observed frequency (y-axis) against mean predicted probability (x-axis).

- **On the diagonal** → perfectly calibrated.
- **Below the diagonal** → overconfident: the model says higher than reality.
- **Above the diagonal** → underconfident: the model says lower than reality.

Let's see the difference with real models.
"""))

CELLS_2.append(("code", r"""# --- Calibration: two models with similar AUC, very different probabilities ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve

# Synthetic data generated from a KNOWN linear model, so we know the true
# probabilities and can check whether each model's numbers are truthful.
rng = np.random.RandomState(21)
n, d = 3000, 5
X = rng.normal(0, 1, (n, d))
w_true = np.array([3.0, 2.25, 1.5, 0.75, 0.45])
p_true = 1 / (1 + np.exp(-(X @ w_true)))      # the true probabilities (linear logit)
y = (rng.rand(n) < p_true).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=21, stratify=y)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=21),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=21),
}
fitted_cal, probs_cal = {}, {}
for name, model in models.items():
    model.fit(X_train, y_train)
    fitted_cal[name] = model
    probs_cal[name] = model.predict_proba(X_test)[:, 1]

print(f"{'Model':22s} {'AUC (ranking)':>14s} {'Brier (probabilities)':>24s}")
for name in models:
    auc = roc_auc_score(y_test, probs_cal[name])
    brier = np.mean((probs_cal[name] - y_test) ** 2)   # the Brier score, computed by hand
    print(f"{name:22s} {auc:>14.3f} {brier:>24.3f}")

# Reliability diagram: observed frequency vs mean predicted probability, per bin
fig, ax = plt.subplots(figsize=(6.5, 6))
for name, color in [("Logistic Regression", "steelblue"), ("Random Forest", "coral")]:
    frac_pos, mean_pred = calibration_curve(y_test, probs_cal[name], n_bins=10)
    ax.plot(mean_pred, frac_pos, "o-", color=color, label=name)
ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed positive frequency")
ax.set_title("Reliability diagram — Logistic Regression vs Random Forest")
ax.legend()
ax.grid(alpha=0.3)
plt.show()"""))

CELLS_2.append(("md", r"""### F. Quantitative measures

**Brier score** — the mean squared error of the probabilities:

$$\text{Brier} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i)^2$$

- $p_i$ = predicted probability, $y_i$ = true label (0 or 1).
- Lower is better: 0 = perfect, 0.25 = always predicting 0.5, 1 = always wrong.
- It rewards both good calibration *and* confident, correct predictions.

**Expected Calibration Error (ECE)** — the average gap between predicted and observed
probability, weighted by bin size:

$$\text{ECE} = \sum_{k=1}^{K} \frac{|B_k|}{n} \cdot |\text{observed}_k - \text{predicted}_k|$$

- $K$ = number of bins, $|B_k|$ = number of samples in bin $k$.
- Lower is better; 0 = perfectly calibrated on average.

### G. Business impact

| Situation | What miscalibration does |
|---|---|
| Bank prices a loan from P(default) | Underpriced risk → losses accumulate silently |
| Insurance premium set from P(claim) | Wrong premiums → adverse selection |
| Medical triage from P(disease) | A patient with true 60% risk is treated as if it were 20% |
| Any expected-value calculation | Every downstream number is off by the same systematic amount |

Calibration matters less when the business only uses the **order** of the scores (e.g.,
"investigate the top 10% of transactions") — that is a ranking use. It matters *most*
when the probability value itself drives a decision or a price.

### H. When calibration matters — and when it does not

**It matters when:** the probability is used as a number — pricing, risk and
expected-value calculations, thresholds chosen as "approve if p ≥ 0.5" (with a
miscalibrated model that threshold means nothing, which connects directly to the
threshold discussion in Section 9), medical and safety decisions.

**It matters less when:** the task is pure ranking (recommendations, search, "which 10%
should we review?") and when comparing models with AUC, which deliberately ignores
absolute values.

### I. Can we fix it?

Yes — **recalibration** fits a second model on top of the raw probabilities, using data
the original model never saw: **Platt scaling** (a logistic regression on the
probabilities) or **isotonic regression** (a monotone, non-parametric fit). scikit-learn
provides `CalibratedClassifierCV`. The next cell shows the fix in action — and the
data-leakage trap that comes with it (Section 18).
"""))

CELLS_2.append(("code", r"""# --- ECE by hand, and fixing calibration with Platt scaling ---
from sklearn.calibration import CalibratedClassifierCV

def ece(y_true, y_prob, n_bins=10):
    # Expected Calibration Error, computed bin by bin
    bins = np.linspace(0, 1, n_bins + 1)
    total, n = 0.0, len(y_true)
    for i in range(n_bins):
        mask = (y_prob > bins[i]) & (y_prob <= bins[i + 1])
        if mask.sum() == 0:
            continue
        pred = y_prob[mask].mean()      # mean predicted probability in the bin
        obs = y_true[mask].mean()       # observed positive frequency in the bin
        total += (mask.sum() / n) * abs(obs - pred)
    return total

print("Expected Calibration Error (by hand, 10 bins):")
for name in models:
    print(f"  {name:22s} ECE = {ece(y_test, probs_cal[name]):.3f}")

# Recalibrate the Random Forest with Platt scaling (sigmoid), fit by cross-validation
# on the TRAINING data only — never on the test set (that would be data leakage).
cal_rf = CalibratedClassifierCV(RandomForestClassifier(n_estimators=200, random_state=21),
                                method="sigmoid", cv=3)
cal_rf.fit(X_train, y_train)
probs_cal_rf = cal_rf.predict_proba(X_test)[:, 1]

p_raw = probs_cal["Random Forest"]
print("\nRandom Forest          raw:  AUC = %.3f  Brier = %.3f  ECE = %.3f"
      % (roc_auc_score(y_test, p_raw), np.mean((p_raw - y_test) ** 2), ece(y_test, p_raw)))
print("Random Forest  recalibrated:  AUC = %.3f  Brier = %.3f  ECE = %.3f"
      % (roc_auc_score(y_test, probs_cal_rf), np.mean((probs_cal_rf - y_test) ** 2),
         ece(y_test, probs_cal_rf)))

# The fix, shown on the reliability diagram
fig, ax = plt.subplots(figsize=(6.5, 6))
frac_pos, mean_pred = calibration_curve(y_test, p_raw, n_bins=10)
ax.plot(mean_pred, frac_pos, "o-", color="coral", label="Random Forest (raw)")
frac_pos, mean_pred = calibration_curve(y_test, probs_cal_rf, n_bins=10)
ax.plot(mean_pred, frac_pos, "o-", color="seagreen", label="Random Forest (Platt-scaled)")
ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed positive frequency")
ax.set_title("Recalibration with Platt scaling")
ax.legend()
ax.grid(alpha=0.3)
plt.show()"""))

CELLS_2.append(("md", r"""### Interactive calibration lab: watch overconfidence happen 🎛️

This widget turns the section's ideas into knobs you can drag:

- **Overconfidence slider (k):** distorts the Logistic Regression's well-calibrated
  probabilities. k = 1 leaves them honest; k > 1 pushes them toward 0 and 1
  (overconfident); k < 1 flattens them toward 0.5 (underconfident).
- **Platt scaling toggle:** refits a sigmoid calibration on the distorted
  probabilities (training data only) and shows the curve pulled back toward the
  diagonal.

Watch the three things that matter:

1. **AUC does not move** when you change k — the distortion changes the *values* but
   never the *order*, so ranking (Section 11) is untouched while calibration collapses.
   This is the ranking-vs-calibration distinction made visible.
2. **Brier and ECE climb** as k moves away from 1 — the probabilities become
   systematically wrong, in both directions: over-confidence and under-confidence.
3. **Toggle Platt scaling on** (try k = 3) and watch the curve return toward the
   diagonal while AUC stays the same — you fixed the numbers without touching the
   ranking. (At k = 1 the model is already honest, so there is nothing to fix — and the
   unnecessary correction even hurts slightly.)
"""))

CELLS_2.append(("code", r"""# --- Interactive calibration lab: overconfidence slider + Platt scaling toggle ---
# Self-contained: rebuilds the same known-truth dataset and Logistic Regression as the
# rest of this section, so the cell also works after a kernel restart.
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

rng = np.random.RandomState(21)
n, d = 3000, 5
X = rng.normal(0, 1, (n, d))
w_true = np.array([3.0, 2.25, 1.5, 0.75, 0.45])
p_true = 1 / (1 + np.exp(-(X @ w_true)))
y = (rng.rand(n) < p_true).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=21, stratify=y)

lr = LogisticRegression(max_iter=1000, random_state=21).fit(X_train, y_train)
p_tr = lr.predict_proba(X_train)[:, 1]
p_te = lr.predict_proba(X_test)[:, 1]


def ece(y_true, y_prob, n_bins=10):
    # Expected Calibration Error, computed bin by bin
    bins = np.linspace(0, 1, n_bins + 1)
    total, n = 0.0, len(y_true)
    for i in range(n_bins):
        mask = (y_prob > bins[i]) & (y_prob <= bins[i + 1])
        if mask.sum() == 0:
            continue
        total += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return total


def distort(p, k):
    # Monotone "overconfidence" transform: k = 1 keeps p unchanged, k > 1 pushes
    # probabilities toward 0 and 1, k < 1 pulls them toward 0.5. The ranking never
    # changes, so AUC is unaffected by construction.
    logit_p = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return 1 / (1 + np.exp(-k * logit_p))


try:
    import ipywidgets as widgets
    from IPython.display import display

    def view(k, platt_on):
        dist_train = distort(p_tr, k)
        dist_test = distort(p_te, k)

        if platt_on:
            # Platt scaling: fit a sigmoid on the (distorted) TRAINING probabilities
            platt = LogisticRegression(max_iter=1000, random_state=0)
            platt.fit(dist_train.reshape(-1, 1), y_train)
            fixed_test = platt.predict_proba(dist_test.reshape(-1, 1))[:, 1]
        else:
            fixed_test = None

        auc_val = roc_auc_score(y_test, dist_test)          # unchanged by k, by construction
        brier = np.mean((dist_test - y_test) ** 2)
        ece_val = ece(y_test, dist_test)

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

        # Left: reliability diagram
        if platt_on:
            frac, pred = calibration_curve(y_test, dist_test, n_bins=10)
            axes[0].plot(pred, frac, "o-", color="coral", alpha=0.6,
                         label=f"Distorted (k = {k:.1f})")
            frac, pred = calibration_curve(y_test, fixed_test, n_bins=10)
            axes[0].plot(pred, frac, "o-", color="seagreen", label="After Platt scaling")
        else:
            frac, pred = calibration_curve(y_test, dist_test, n_bins=10)
            axes[0].plot(pred, frac, "o-", color="steelblue", label=f"Distorted (k = {k:.1f})")
        axes[0].plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        axes[0].set_xlabel("Mean predicted probability")
        axes[0].set_ylabel("Observed positive frequency")
        axes[0].set_title("Reliability diagram")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        # Right: Brier and ECE bars (with and without the fix)
        if platt_on:
            brier_fix = np.mean((fixed_test - y_test) ** 2)
            ece_fix = ece(y_test, fixed_test)
            x = np.arange(2)
            w = 0.35
            axes[1].bar(x - w / 2, [brier, ece_val], w, color="coral", label="Distorted")
            axes[1].bar(x + w / 2, [brier_fix, ece_fix], w, color="seagreen", label="Platt-fixed")
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(["Brier", "ECE"])
            axes[1].legend(fontsize=8)
        else:
            axes[1].bar(["Brier", "ECE"], [brier, ece_val], color="steelblue")
        axes[1].set_title("Calibration errors")
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()

        print(f"k = {k:.1f}  |  AUC = {auc_val:.3f} (unchanged by k)  |  Brier = {brier:.3f}  ECE = {ece_val:.3f}")
        if platt_on:
            print(f"After Platt scaling:  Brier = {brier_fix:.3f}  ECE = {ece_fix:.3f}")

    slider_k = widgets.FloatSlider(value=1.0, min=0.3, max=5.0, step=0.1,
                                   description="Overconfidence k:", readout_format=".1f")
    toggle = widgets.ToggleButton(value=False, description="Apply Platt scaling")

    out = widgets.interactive_output(view, {"k": slider_k, "platt_on": toggle})
    with out:
        view(1.0, False)   # embed a static snapshot of the initial view
    display(widgets.VBox([slider_k, toggle, out]))
    print("✅ Move the slider and toggle Platt scaling — AUC stays fixed while Brier and ECE move.")

except ImportError:
    print("ipywidgets is not installed, so the interactive lab is unavailable.")
    print("Install it with:  pip install ipywidgets   (then restart the kernel and re-run this cell).")
    print("Until then, the static experiments in the two code cells above demonstrate the same ideas.")"""))

CELLS_2.append(("md", r"""**What did we learn from these outputs — and from playing with the interactive lab above?**

The two models have very similar AUC — they rank almost identically. But their Brier
scores and ECE values are very different, and the reliability diagram shows why: the
Random Forest's curve bulges away from the diagonal (it says "90%" where reality is
closer to "70%"), while the Logistic Regression hugs it. After Platt scaling, the Random
Forest's curve moves back toward the diagonal: ECE drops noticeably and Brier improves
a little, while AUC stays essentially unchanged — calibration improved without losing
discrimination.

**What would this mean in a real-world application?**

A bank that used the raw Random Forest's probabilities to price loans would
systematically underprice risk: it believes defaults are rarer than they are. Its AUC
looked fine — the ranking was good — but the *numbers* were wrong, and the numbers are
exactly what the pricing formula uses. Recalibration fixes the numbers. And note the
critical detail: the recalibration was fit with cross-validation on the *training* data
— fitting it on the test set would have been data leakage (Section 18), and the
improvement would have been an illusion.

---

#### Beginner Perspective

*A high AUC does not mean the probabilities are meaningful. If you will use the
probability number itself (pricing, risk, triage), check calibration with a reliability
diagram and a Brier score.*

#### Expert Perspective

*Experts separate three properties of probabilities: calibration (absolute accuracy),
sharpness (concentration near 0 and 1), and discrimination (ranking). The Brier score
bundles calibration and sharpness; ECE isolates calibration. They also check calibration
per subgroup (customer segment, region, model version) — a model can be well calibrated
overall yet badly calibrated for the very group a decision affects. And they remember
that recalibration must always be fit on held-out data, ideally inside cross-validation.*

---

[⬅ Previous](#sec12) · [🏠 Table of Contents](#toc) · [Next ➡](#sec14)"""))

CELLS_2.append(("md", r"""📗 **Back to basics:** only here in depth — the nearest basic material is
**Section 8: The Precision–Recall Trade-off** in `ML_Model_Evaluation_Simple.ipynb`,
which introduces probabilities and thresholds.
"""))

CELLS_2.append(("md", r"""<a id="sec14"></a>
## 14. The Classification Report

The classification report is a compact table that prints precision, recall, F1, and
support **for every class**, plus aggregated averages.

### Reading the report — line by line

For a binary problem, scikit-learn prints one row per class and three extra rows:
**macro avg**, **weighted avg** (and for multiclass, **micro avg** appears in the
`accuracy` line).

| Row | Meaning |
|---|---|
| `precision` | Precision for this class, treating it as "positive" |
| `recall` | Recall for this class, treating it as "positive" |
| `f1-score` | F1 for this class |
| `support` | Number of **actual** samples of this class in the test set |
| `macro avg` | **Unweighted** average of the per-class scores — every class counts equally, regardless of size |
| `weighted avg` | Average of the per-class scores **weighted by support** — bigger classes dominate |
| `micro avg` | Pool all predictions into one big confusion matrix (for multiclass this equals accuracy) |

**Why do the averages matter?** On imbalanced data, the *weighted* average can look great
simply because the majority class is easy, while the *macro* average honestly reflects
that the minority class is handled badly. Always compare the two — a big gap between them
is a red flag.

> **A classification report should not be read mechanically.** Interpret the numbers
> according to the application. If the minority class row shows recall 0.10, no average
> in the world makes that acceptable for a disease-screening problem.
"""))

CELLS_2.append(("code", r"""# --- Classification report ---
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# Small imbalanced binary problem (self-contained cell)
X, y = make_classification(n_samples=800, n_features=8, n_informative=5, n_redundant=0,
                           weights=[0.9, 0.1], random_state=13)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=13, stratify=y)

clf = RandomForestClassifier(n_estimators=100, random_state=13).fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=["Negative (0)", "Positive (1)"]))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))"""))

CELLS_2.append(("md", r"""**What did we learn from this output?**

Each row answers a question about one class: for the Positive class, precision says "of
the positives I predicted, how many were right?", recall says "of the actual positives,
how many did I find?". The `support` column tells you how many samples each row is based
on (here ~24 positives — a small number, so these metrics are noisy). The macro average
treats both classes equally; the weighted average is dragged toward the (larger) negative
class.

**What would this mean in a real-world application?**

If this were a disease-screening model, the row that matters is **Positive**:
*"out of every 100 sick patients, how many do we send home?"* A report that looks great in
the `weighted avg` row can still be unacceptable for the minority class — which is often
exactly the class the business cares about. This is why professionals read the per-class
rows, not just the averages.

---

#### Beginner Perspective

*Read each class row separately. Macro = every class equal; weighted = bigger classes
count more. Big gap between them = the model ignores the small class.*

#### Expert Perspective

*Experts read the report *per operating decision*: they check the class they will act on,
look at `support` to judge how reliable the numbers are, and treat macro/weighted as
diagnostics for class-imbalance blindness rather than as final scores. For multiclass
problems with different error costs per class (Section 16), they build a cost matrix
rather than averaging rows.*

---

[⬅ Previous](#sec12) · [🏠 Table of Contents](#toc) · [Next ➡](#sec15)"""))

CELLS_2.append(("md", r"""📗 **Back to basics:** see **Section 10: The Classification Report** in
`ML_Model_Evaluation_Simple.ipynb` — the same report on the same fraud model, read line
by line.
"""))

CELLS_2.append(("md", r"""<a id="sec15"></a>
## 15. Imbalanced Datasets

### What is class imbalance?

A dataset is imbalanced when one class is much more common than the other. Real-world
examples: 0.1–2% of credit-card transactions are fraud; 0.5–5% of patients have a given
rare disease; a tiny fraction of network connections are attacks. The rare class is the
**minority class** — and it is almost always the class that matters.

### Why does accuracy fail here?

Because the majority class dominates the accuracy numerator. A model that predicts the
majority class for everything can reach 98–99% accuracy while doing **nothing**. The
accuracy number looks impressive; the model is useless. (We demonstrated this in
Section 4.)

### What should we use instead?

The tools for imbalanced problems are exactly the metrics we have built:

- **Precision** — when the model flags something, how often is it right? (FP control)
- **Recall** — how many of the rare positives do we find? (FN control)
- **F1** — balance of the two
- **ROC-AUC** — overall ranking quality (but optimistic, see Section 12)
- **PR-AUC** — usefulness for finding the rare class (preferred)

Let's build an imbalanced dataset, fit a real model, and compare it honestly against the
naive "always negative" baseline.
"""))

CELLS_2.append(("code", r"""# --- Imbalanced dataset: real model vs "always negative" baseline ---
from sklearn.ensemble import RandomForestClassifier

# 2% positive class
X, y = make_classification(n_samples=2000, n_features=8, n_informative=5, n_redundant=0,
                           weights=[0.98, 0.02], random_state=14)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=14, stratify=y)

print("Class distribution in the test set:")
print(pd.Series(y_test).value_counts().sort_index(), "\n")

dummy_pred = np.zeros_like(y_test)   # the naive baseline: always predict majority class
rf = RandomForestClassifier(n_estimators=100, random_state=14).fit(X_train, y_train)
rf_pred = rf.predict(X_test)

summary = []
for name, pred in [("Always-negative (dummy)", dummy_pred), ("Random Forest", rf_pred)]:
    summary.append({
        "Accuracy":  accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall":    recall_score(y_test, pred, zero_division=0),
        "F1":        f1_score(y_test, pred, zero_division=0),
        "Frauds found": int((pred == 1).sum()),
    })

pd.DataFrame(summary, index=["Always-negative (dummy)", "Random Forest"]).round(3)"""))

CELLS_2.append(("code", r"""# --- Class distribution chart ---
counts = pd.Series(y).value_counts().sort_index()
plt.figure(figsize=(6, 4))
bars = plt.bar(["Majority class (0)", "Minority class (1)"], counts.values,
               color=["steelblue", "coral"])
for bar, v in zip(bars, counts.values):
    plt.text(bar.get_x() + bar.get_width() / 2, v, f"{v}  ({v / len(y):.1%})",
             ha="center", va="bottom")
plt.title("Class Distribution — Highly Imbalanced Dataset")
plt.ylabel("Number of samples")
plt.ylim(0, counts.max() * 1.15)
plt.show()"""))

CELLS_2.append(("md", r"""**What did we learn from this output?**

The dummy model scores ~98% accuracy — yet catches **zero** frauds. The Random Forest has
lower accuracy but actually finds some frauds. If you compared these models on accuracy
alone, you would pick the *worse* one. This is the accuracy trap in action.

**What would this mean in a real-world application?**

For fraud, disease, or intrusion detection, the meaningful question is not "how often is
the model right overall?" but "how many of the rare, expensive events does it catch, and
at what cost in false alarms?" Those are precision/recall/PR questions.

### The key question — which model is better?

Consider two models for an imbalanced problem:

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| **A** | 98% | 95% | 15% |
| **B** | 95% | 60% | 80% |

**Which model is better?** Model A is "more accurate"; Model B finds far more positives.

**There is no answer without knowing the cost of FP and FN.**

- If a missed positive costs €10,000 and a false alarm costs €5 → **Model B** (find them
  all; the false alarms are cheap).
- If a false alarm costs €1,000 and a missed positive costs €10 → **Model A** (only act on
  high-confidence signals).
- If costs are unknown → compute F1 for each and use that as a starting balance — but
  remember F1 itself assumes symmetric costs.

This single idea — *the better model depends on error costs* — is the heart of this
notebook. The next sections (case studies, Sections 21–22) will make it concrete.

---

#### Beginner Perspective

*Imbalance breaks accuracy. Use precision, recall, F1, and PR curves. Always compare your
model against a "predict the majority class" baseline.*

#### Expert Perspective

*Experts do not just pick a metric — they estimate the actual costs of each error, choose
a threshold that minimizes expected cost, and evaluate with cross-validation (Section 17)
to make sure the imbalance and the metric behave stably across folds. They may also use
class weights or resampling, but they judge *those* techniques by the same cost-based
metrics, never by accuracy.*

---

[⬅ Previous](#sec14) · [🏠 Table of Contents](#toc) · [Next ➡](#sec16)"""))
