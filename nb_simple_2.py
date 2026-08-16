CELLS_S2 = []

CELLS_S2.append(("md", r"""📖 **What to read next — go deeper:** see **Sections 10–12: The ROC Curve · AUC ·
ROC-AUC vs Precision–Recall** in `ML_Model_Evaluation.ipynb` — the ROC curve in depth,
what AUC really measures, and why precision–recall beats ROC on imbalanced data.
"""))

CELLS_S2.append(("md", r"""<a id="sec10"></a>
## 10. The Classification Report

scikit-learn's `classification_report` prints precision, recall and F1 for **each
class**, plus averages. Let's look at it on the fraud model from [Section 8](#sec8).
"""))

CELLS_S2.append(("code", r"""# --- Classification report for the fraud model ---
from sklearn.metrics import classification_report

preds = (probs >= 0.5).astype(int)
print(classification_report(yf_te, preds, target_names=["Normal (0)", "Fraud (1)"]))"""))

CELLS_S2.append(("md", r"""**How to read it:**

- Each row is one class. *Fraud (1)* is usually the interesting one.
- **precision / recall / f1-score:** same definitions as before, per class.
- **support:** how many real examples of that class were in the test set (used to trust
  the numbers — a row with support 5 is unreliable).
- **macro avg:** the plain average of the per-class scores — each class counts equally.
- **weighted avg:** the average weighted by support — big classes dominate.

**Do not read the report mechanically.** For fraud detection, the *Fraud* row is what
matters; the *Normal* row is almost perfect simply because normal transactions are easy.
The numbers only mean something in the context of the business question.
"""))

CELLS_S2.append(("md", r"""📖 **What to read next — go deeper:** see **Section 14: The Classification Report** in
`ML_Model_Evaluation.ipynb` — reading the report line by line, macro vs weighted
averages, and how the report fits into multiclass evaluation.
"""))

CELLS_S2.append(("md", r"""<a id="sec11"></a>
## 11. Imbalanced Datasets

**Class imbalance** means one class is much rarer than the other — 1% fraud, 1% sick,
0.5% intrusions. The interesting class is the rare one, and **accuracy is useless**
(remember [Section 3](#sec3)).

The next cell compares two models on a 99% / 1% dataset:
"""))

CELLS_S2.append(("code", r"""# --- Imbalanced data: the baseline that predicts "no" for everyone ---
X_i, y_i = make_classification(n_samples=2000, n_features=8, n_informative=4,
                               n_redundant=0, weights=[0.99, 0.01], random_state=5)
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(
    X_i, y_i, test_size=0.3, random_state=5, stratify=y_i)

# Baseline: never flag anything
base_pred = np.zeros(len(yi_te), dtype=int)

# A real model
lr = LogisticRegression(max_iter=1000).fit(Xi_tr, yi_tr)
lr_pred = lr.predict(Xi_te)

for name, p in [("Always says NO (baseline)", base_pred), ("Logistic regression", lr_pred)]:
    print(f"{name}:")
    print(f"  accuracy = {accuracy_score(yi_te, p):.3f}   "
          f"recall = {recall_score(yi_te, p):.3f}   "
          f"F1 = {f1_score(yi_te, p):.3f}")
print()
print("Both score ~99% accuracy. Only recall and F1 reveal which one finds fraud.")"""))

CELLS_S2.append(("md", r"""### Two models, which one is better?

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| **A** | 97% | 95% | 60% |
| **B** | 94% | 80% | 90% |

Model A has the higher accuracy. Model B finds far more of the rare cases.

**Which is better? There is no answer without the costs.**

- If the rare event is **fraud** and an FN is an approved fraud (expensive) → **Model B**
  — the higher recall is worth more than the accuracy difference.
- If the rare event is **a disease** and FP means invasive follow-up tests → maybe
  **Model A**, because precision (fewer false alarms) matters more.

The "best" model depends on the business problem, not on a leaderboard.
"""))

CELLS_S2.append(("md", r"""📖 **What to read next — go deeper:** see **Section 15: Imbalanced Datasets** in
`ML_Model_Evaluation.ipynb` — the full toolkit for rare events: resampling, class
weights, threshold tuning, and choosing between models for imbalance.
"""))

CELLS_S2.append(("md", r"""<a id="sec12"></a>
## 12. Cross-Validation

### Why one split is not enough

A single train/test split is one roll of the dice — the test set may happen to be easy
or hard by luck. **Cross-validation** repeats the experiment: the data is cut into
**K folds**, the model trains K times, and each fold is used for testing once. The
result is a mean **and** a spread.

**Stratified K-Fold** keeps the class proportions the same in every fold — important for
imbalanced data. We use the fraud dataset from [Section 8](#sec8) here.
"""))

CELLS_S2.append(("code", r"""# --- Cross-validation on the fraud dataset ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(LogisticRegression(max_iter=1000), X_f, y_f,
                         cv=skf, scoring="f1")
print(f"F1 over 5 folds: {np.round(scores, 3)}")
print(f"mean = {scores.mean():.3f}  ±  {scores.std():.3f}")
print()
print("The folds swing from about 0.38 to 0.62: one lucky split could report")
print("either end. Report the mean AND the spread — a fragile model is a risk.")"""))

CELLS_S2.append(("md", r"""📖 **What to read next — go deeper:** see **Section 17: Cross-Validation** in
`ML_Model_Evaluation.ipynb` — K-Fold and Stratified K-Fold in depth, common CV
mistakes, and the bias–variance trade-off of CV estimates.
"""))

CELLS_S2.append(("md", r"""<a id="sec13"></a>
## 13. Regression Metrics

Regression predicts **numbers** (house prices, temperatures), so errors are
*differences* between predictions and true values. Four standard metrics:

| Metric | Formula | What it answers |
|---|---|---|
| **MAE** | $$\frac{1}{n}\sum |y_i - \hat{y}_i|$$ | Average error in the original units. Easy to explain. |
| **MSE** | $$\frac{1}{n}\sum (y_i - \hat{y}_i)^2$$ | Average **squared** error — punishes big mistakes. Units are squared. |
| **RMSE** | $$\sqrt{MSE}$$ | Like MSE, but back in original units. |
| **R²** | $$1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$ | Fraction of variance explained. 1 = perfect, 0 = no better than predicting the mean. |

**Business meaning:** if MAE = 50,000 for house prices, the average prediction is off
by about 50,000 currency units. Whether that is acceptable depends on the application —
50,000 is small for a luxury house and huge for a used car.
"""))

CELLS_S2.append(("code", r"""# --- Regression metrics on a small house-price example ---
X_h, y_h = make_regression(n_samples=300, n_features=4, noise=30, random_state=7)
y_h = y_h + 200000   # shift to look like prices

Xh_tr, Xh_te, yh_tr, yh_te = train_test_split(X_h, y_h, test_size=0.3, random_state=7)
reg = LinearRegression().fit(Xh_tr, yh_tr)
pred = reg.predict(Xh_te)

mae = mean_absolute_error(yh_te, pred)
mse = mean_squared_error(yh_te, pred)
rmse = np.sqrt(mse)
r2 = r2_score(yh_te, pred)

print(f"MAE  = {mae:,.0f}   (average error, in currency)")
print(f"MSE  = {mse:,.0f}")
print(f"RMSE = {rmse:,.0f}")
print(f"R²   = {r2:.3f}   (fraction of price variance explained)")
print()
print(f"On average, the prediction is off by about {mae:,.0f} units.")"""))

CELLS_S2.append(("md", r"""📖 **What to read next — go deeper:** see **Section 20: Regression Metrics** in
`ML_Model_Evaluation.ipynb` — MAE / MSE / RMSE / R² in depth, plus MAPE, and how to
choose among regression metrics for your problem.
"""))

CELLS_S2.append(("md", r"""<a id="sec14"></a>
## 14. Mini Case Study: Loan Default Prediction

Let's put everything together on a realistic problem. A bank must decide whether to
**approve** or **reject** a loan. The model predicts the probability that the applicant
**will default** — that is the "positive" class.

### The business problem and the cost of errors

| Error | What it means | Business consequence |
|---|---|---|
| **FN** | Model says "no default" but the applicant defaults → loan approved | The bank **loses the loaned money**. Expensive per incident. |
| **FP** | Model says "will default" but the applicant would have repaid → loan rejected | The bank loses the **interest profit** and the customer may go elsewhere. Cheaper per incident. |

For this exercise we assume a default costs **10×** more than rejecting a good
customer: **FN = 10,000**, **FP = 1,000** currency units. (In a real project these
numbers would come from the bank's own financial analysis — and changing them changes
the answer, which is exactly the point.)
"""))

CELLS_S2.append(("code", r"""# --- Step 1: generate a small synthetic loan dataset ---
rng = np.random.RandomState(2024)
n = 1200

income           = rng.normal(60000, 25000, n).clip(15000, 250000)
credit_score     = rng.normal(680, 60, n).clip(400, 850)
debt_to_income   = rng.uniform(0.05, 0.75, n)
loan_amount      = rng.uniform(5000, 60000, n)
employment_years = rng.uniform(0, 20, n).round(1)

# Default probability: higher with debt and loan size, lower with credit score, etc.
logit = (-3.4 - 2.0 * (credit_score - 680) / 60
         + 2.0 * (debt_to_income - 0.4) / 0.2
         + 1.0 * (loan_amount - 32500) / 16000
         - 0.75 * (income - 60000) / 24000
         - 0.5 * (employment_years - 10) / 5)
p_default = 1 / (1 + np.exp(-logit))
default = (rng.rand(n) < p_default).astype(int)

df = pd.DataFrame({"income": income.round(0), "credit_score": credit_score.round(0),
                   "debt_to_income": debt_to_income.round(3),
                   "loan_amount": loan_amount.round(0),
                   "employment_years": employment_years, "default": default})
print("Class distribution:")
print(df["default"].value_counts(normalize=True).round(3))
print(f"Default rate: {df['default'].mean():.1%}")

# --- Step 2: train a model and evaluate at the default threshold 0.5 ---
features = ["income", "credit_score", "debt_to_income", "loan_amount", "employment_years"]
X = df[features]
y = df["default"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
print()
print("Confusion matrix at threshold 0.5:")
print(f"  TN={tn}  FP={fp}")
print(f"  FN={fn}  TP={tp}")
print(f"  accuracy={accuracy_score(y_test, preds):.3f}  "
      f"precision={precision_score(y_test, preds):.3f}  "
      f"recall={recall_score(y_test, preds):.3f}")"""))

CELLS_S2.append(("code", r"""# --- Step 3: which threshold should the bank actually use? ---
COST_FP = 1000    # rejected good customer (lost profit)
COST_FN = 10000   # approved loan that defaults (lost money)

def total_cost(y_true, probs, t):
    preds = (probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return fp * COST_FP + fn * COST_FN, fp, fn

thresholds = [0.05, 0.10, 0.15, 0.30, 0.50, 0.70, 0.90]
print("threshold | FP | FN | total cost")
best = (float("inf"), None)
for t in thresholds:
    cost, fp, fn = total_cost(y_test, probs, t)
    print(f"   {t:.2f}    | {fp:3d} | {fn:3d} | {cost:9,.0f}")
    if cost < best[0]:
        best = (cost, t)

print()
print(f"Cheapest threshold in this sweep: {best[1]:.2f}  (total cost {best[0]:,.0f})")
print("Compare with the default 0.5 in the table above. The difference")
print("is the price of following a convention instead of the business costs.")"""))

CELLS_S2.append(("code", r"""# --- Step 4: ROC curve and the cost curve ---
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))

fpr, tpr, _ = roc_curve(y_test, probs)
axes[0].plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y_test, probs):.3f}")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate (recall)")
axes[0].set_title("ROC curve — loan model")
axes[0].legend()
axes[0].grid(alpha=0.3)

ts = np.arange(0.02, 0.95, 0.02)
costs = [total_cost(y_test, probs, t)[0] for t in ts]
best_t = ts[int(np.argmin(costs))]
axes[1].plot(ts, costs, color="coral")
axes[1].axvline(0.5, color="gray", linestyle="--", label="Default threshold (0.5)")
axes[1].axvline(best_t, color="seagreen", linestyle="--",
                label=f"Cost-optimal threshold ({best_t:.2f})")
axes[1].set_xlabel("Classification threshold")
axes[1].set_ylabel("Total cost (FP × 1,000 + FN × 10,000)")
axes[1].set_title("Business cost vs threshold")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
print(f"Cost at default 0.5: {total_cost(y_test, probs, 0.5)[0]:,.0f}")
print(f"Cost at optimal    : {total_cost(y_test, probs, best_t)[0]:,.0f}  (threshold {best_t:.2f})")"""))

CELLS_S2.append(("md", r"""### What did we learn?

- The model is good (AUC ≈ 0.92), but **AUC does not pick a threshold**.
- The default threshold **0.5 is just a convention**. On this problem the bank saves a
  large amount of money by lowering the threshold — rejecting more borderline
  applicants — because an approved default (FN) is so much more expensive than a
  rejected good customer (FP).
- The metric to report is the one that matches the question: here the business cares
  about the **cost of FN vs FP**, so we evaluated the cost, not just accuracy or F1.

**The takeaway of the whole notebook:** a model is only as useful as the decisions it
enables. Understand the problem → know what each error costs → pick the metric → pick
the threshold. The full `ML_Model_Evaluation.ipynb` shows a much longer version of this
case study, with three models, a full threshold analysis and an interactive explorer.
"""))

CELLS_S2.append(("md", r"""📖 **What to read next — go deeper:** see **Sections 21–23: Loan Default Case Study ·
Business Decisions · Cost-Sensitive Evaluation** in `ML_Model_Evaluation.ipynb` — the
full case study with three models, a complete threshold analysis, and the cost-matrix
decision rule.
"""))

CELLS_S2.append(("md", r"""<a id="sec15"></a>
## 15. Cheat Sheet

| Metric | Formula | Measures | Best used when | Business question |
|---|---|---|---|---|
| **Accuracy** | (TP+TN)/(all) | Overall correctness | Balanced classes | How often is the model right overall? |
| **Precision** | TP/(TP+FP) | Trust in positive predictions | False alarms are expensive | When the model says YES, how often should I believe it? |
| **Recall** | TP/(TP+FN) | Finding the positives | Missed positives are expensive | How many real positives can I find? |
| **Specificity** | TN/(TN+FP) | Correctly clearing negatives | Avoiding false alarms matters | How well do I avoid crying wolf? |
| **F1** | 2PR/(P+R) | Balance of precision & recall | One summary number needed | Can I balance precision and recall? |
| **ROC-AUC** | Area under ROC | Ranking quality across thresholds | Comparing models overall | How well does the model rank positives above negatives? |
| **MAE** | mean(|y−ŷ|) | Average error (regression) | Errors in original units | How far off is the average prediction? |
| **RMSE** | sqrt(MSE) | Error, penalizing big mistakes | Big errors are worse | How badly do big mistakes hurt? |
| **R²** | 1 − SS_res/SS_tot | Variance explained | Model comparison | How much of the variation does the model explain? |

**Rule of thumb:** start from the business problem, not from the metric list.
"""))

CELLS_S2.append(("md", r"""📖 **What to read next — go deeper:** see **Section 27: Final Cheat Sheet** in
`ML_Model_Evaluation.ipynb` — the complete one-page summary of every metric, formula,
and business question covered in the course.
"""))

CELLS_S2.append(("md", r"""<a id="sec16"></a>
## 16. Practice Exercises

Try these before looking at the solutions below.

### Beginner

1. A model gives TN = 85, FP = 15, FN = 25, TP = 75. Write the confusion matrix and
   compute accuracy by hand.
2. From the same matrix, compute precision and explain in one sentence what it means.
3. Compute recall and specificity from the same matrix.

### Intermediate

4. A rare-disease dataset has 1% sick people. A model predicts "healthy" for everyone.
   Its accuracy is 99%. Why is it useless? Which metric reveals this?
5. Model A: precision 0.9, recall 0.2. Model B: precision 0.5, recall 0.6. Compute F1
   for both. Which has the higher F1? Does that make it the better model?
6. A logistic regression flags 40 cases at threshold 0.5 and 12 cases at 0.7. Without
   seeing the data, what happened to precision, recall, FP and FN?

### Expert

7. A bank must decide between two models. FN (approved default) costs 10× an FP
   (rejected good customer). Model A: recall 0.55. Model B: recall 0.88 but precision
   0.60. Which would you choose and why? What information is still missing?
8. On a dataset with 0.5% fraud, a model reports AUC = 0.95 but precision = 0.03 at
   the threshold it uses. Explain how both can be true, and which number you would
   trust for the deployment decision.
9. Design the evaluation plan for a spam filter where a real email in the spam folder
   costs much more than spam in the inbox. Which metric should be maximized, and what
   does that imply about the threshold?
"""))

CELLS_S2.append(("md", r"""<a id="exercise-solutions"></a>
## Exercise Solutions

1. Confusion matrix (rows = truth, cols = prediction): [[85, 15], [25, 75]].
   Accuracy = (75 + 85) / 200 = 0.80.
2. Precision = 75 / (75 + 15) = 0.833. *Of all predicted positives, 83% were correct.*
3. Recall = 75 / (75 + 25) = 0.75. Specificity = 85 / (85 + 15) = 0.85.
4. Recall = 0 — it finds none of the sick people. Accuracy hides this on imbalanced
   data; recall (and F1, precision) reveal it.
5. F1(A) = 2·0.9·0.2/1.1 ≈ 0.33. F1(B) = 2·0.5·0.6/1.1 ≈ 0.55. B has the higher F1 —
   but if precision is what the problem needs, A may still be the better choice.
6. Raising the threshold means fewer positives: FP falls, recall falls, precision
   usually rises, FN rises. Whether total cost improves depends on the error costs.
7. Likely B if FN is 10× more expensive: catching more defaults saves more than the
   extra false alarms cost — but the missing information is the *exact* FP/FN counts
   and the actual costs, so the decision should be made with numbers, not intuition.
8. AUC measures ranking across all thresholds; precision is measured at one threshold.
   At 0.5 the model may flag many cases (high recall, low precision). AUC 0.95 can hide
   that precision at the operating point is 0.03 — for deployment, precision/recall at
   the chosen threshold matters.
9. Maximize precision (fewer real emails wrongly filtered) — or equivalently choose a
   high threshold. Since FP (real mail lost) is the expensive error, a cautious
   classifier is right for the business.

[⬆ Back to Table of Contents](#toc)
"""))

CELLS_S2.append(("md", r"""📖 **What to read next — go deeper:** see **Section 28: Practice Exercises** in
`ML_Model_Evaluation.ipynb` — 15 graded exercises with full solutions covering every
topic in the course.
"""))

CELLS_S2.append(("md", r"""<a id="sec17"></a>
## 17. Final Quiz

### Section A — Multiple choice

1. Accuracy is misleading when…
   a) classes are balanced
   b) the dataset is imbalanced
   c) the model is linear
   d) precision is high

2. Precision answers…
   a) how many positives did the model find?
   b) of all predicted positives, how many were correct?
   c) of all true negatives, how many were identified?
   d) how often is the model right overall?

3. Recall = …
   a) TP / (TP + FP)
   b) TP / (TP + FN)
   c) TN / (TN + FP)
   d) TN / (TN + FN)

4. Lowering the classification threshold usually…
   a) raises precision and lowers recall
   b) raises recall and lowers precision
   c) leaves both unchanged
   d) always lowers accuracy

5. AUC = 0.5 means…
   a) the model is perfect
   b) the model ranks no better than random
   c) accuracy is 50%
   d) the model is overfitting

### Section B — True / False

6. A 99% accurate model can still be useless for fraud detection.
7. The threshold 0.5 is always the best choice.
8. The model with the highest F1 is always the best business solution.
9. High recall means few false negatives.
10. MAE is measured in the same units as the target variable.

### Section C — Short answer

11. Give one example where precision matters more than recall.
12. Give one example where recall matters more than precision.
13. Why can a 99% accurate classifier still be useless?
14. In a confusion matrix, what is a false positive, in plain words?
15. Which error is more expensive in disease screening: FP or FN? Why?
"""))

CELLS_S2.append(("md", r"""📖 **What to read next — go deeper:** see **Section 29: Final Quiz** in
`ML_Model_Evaluation.ipynb` — a 25-question quiz with answers, plus the final
workflow that ties the whole course together.
"""))

CELLS_S2.append(("md", r"""<a id="quiz-answers"></a>
## Quiz Answers

1. **b** — accuracy is misleading when the dataset is imbalanced.
2. **b** — precision asks: of all predicted positives, how many were correct?
3. **b** — recall = TP / (TP + FN).
4. **b** — lowering the threshold flags more things: recall rises, precision falls.
5. **b** — AUC = 0.5 is no better than random ranking.
6. **True** — it may catch zero of the rare cases (see Section 3).
7. **False** — the right threshold follows the costs of FP and FN (Section 8 and 14).
8. **False** — F1 balances P and R, but the business may value one far more (Section 7).
9. **True** — recall = TP/(TP+FN); high recall means few missed positives.
10. **True** — MAE is an average of |y − ŷ|, in the target's units (Section 13).
11. Spam filtering (a real email lost is expensive); fraud alerts when false alarms
    annoy customers; medical tests with expensive or invasive follow-ups.
12. Disease screening (a missed patient is dangerous); security intrusion detection;
    fraud detection when approved fraud is very costly.
13. On imbalanced data a model can score 99% by predicting the majority class for
    everyone — while finding none of the rare, important cases.
14. The model said YES but the truth was NO — a false alarm.
15. Usually FN — a sick patient who is told they are healthy may not get treated. (The
    answer depends on context, which is exactly why costs must be discussed.)

[⬆ Back to Table of Contents](#toc)
"""))

CELLS_S2.append(("md", r"""## Final Message

You now know how to evaluate a machine learning model properly:

1. **Understand the problem** — what are we predicting, and what does each error cost?
2. **Look at the confusion matrix** — the four numbers tell the real story.
3. **Pick the metric for the problem** — accuracy for balanced problems, precision
   when false alarms hurt, recall when misses hurt, F1 for a summary, AUC for ranking.
4. **Choose the threshold deliberately** — it is a business decision, not a default.

> **Model evaluation is not about finding the model with the highest score. It is about
> finding the model — and the decision threshold — that best fits the real problem.**

**Want to go deeper?** Open `ML_Model_Evaluation.ipynb` for the full course: every
topic here in depth, plus probability calibration, cost-sensitive decision rules, data
leakage, time-series evaluation, cross-validation, multiclass evaluation, and
interactive widgets.
"""))
