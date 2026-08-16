# Content part 4: Sections 23–27, Exercise Solutions, Quiz Answers, Final Framework
# Each item is a tuple: ("md", text) or ("code", source)

CELLS_4 = []

CELLS_4.append(("md", r"""📗 **Back to basics:** see **Section 15: Cheat Sheet** in
`ML_Model_Evaluation_Simple.ipynb` — the “Best used when” column is the same selection
guide in one table.
"""))

CELLS_4.append(("md", r"""<a id="sec25"></a>
## 25. Common Mistakes When Evaluating Machine Learning Models

Every item below is a real, common failure mode in industry. For each one: **why is this
a problem?**

| # | Mistake | Why this is a problem |
|---|---|---|
| 1 | **Using training accuracy as final performance** | Training accuracy measures memorization, not generalization. A model that memorized the data looks perfect during training and fails on new data (Section 1). |
| 2 | **Relying only on accuracy** | Accuracy treats all errors as equal and is trivially inflated on imbalanced data. It hides both the false-alarm burden and the miss rate (Section 4). |
| 3 | **Ignoring class imbalance** | When one class dominates, "high" accuracy is easy to fake. The minority class — often the one that matters — is silently neglected (Section 15). |
| 4 | **Ignoring false positives** | Every FP has a cost: wasted investigations, blocked customers, unnecessary treatment. A model judged only by recall or accuracy can be bankrupting the business with false alarms. |
| 5 | **Ignoring false negatives** | Every FN is a missed case: an undetected fraud, an untreated disease, an unnoticed attack. Models that look "precise" can be missing almost everything that matters. |
| 6 | **Using the test set repeatedly** | The test set exists to be used **once**. If you tune on it, you leak it into your decisions and it stops measuring new data — your reported performance becomes optimistic. |
| 7 | **Data leakage** | Information from the future/test leaks into training (e.g., scaling before splitting, target-derived features). Results look fantastic and collapse in production. Two concrete demonstrations and the fix: [Section 18](#sec18). |
| 8 | **Selecting a model based on one metric only** | One number hides the rest of the story. A model selected by F1 alone may have unacceptable per-class behavior, calibration, or cost performance. |
| 9 | **Ignoring business requirements** | A technically excellent model that does not serve the business objective (or cannot be operated) is not a good model. Evaluation must start from the problem, not from the metric. |
| 10 | **Ignoring the cost of errors** | If FP costs 100× FN, a "balanced" metric like F1 is the wrong objective. Cost structure must drive metric and threshold choice. |
| 11 | **Misinterpreting ROC-AUC** | AUC is a threshold-independent *ranking* summary. It does not pick an operating threshold, and on imbalanced data it can look optimistically high (Section 12). |
| 12 | **Not checking the confusion matrix** | Metrics summarize; the confusion matrix tells the actual story. Without it you cannot see *which* errors happen, in what quantities, for which classes. |
| 13 | **Using inappropriate thresholds** | 0.5 is a default, not a decision. The threshold must be chosen from the costs of FP and FN — the same model can be great at one threshold and terrible at another. |
| 14 | **Assuming higher accuracy always means a better model** | Accuracy is one lens among many. A model with lower accuracy can be far more valuable if it finds the expensive cases (Sections 21–22). |
| 15 | **Assuming F1 is always the best metric** | F1 assumes symmetric error costs. When FP and FN cost different amounts, F1 points you to the wrong model and the wrong threshold. |
| 16 | **Selecting a threshold without considering business consequences** | The threshold decides how many false alarms and misses you *buy*. Choosing it in isolation from operations (queue capacity, customer impact) leads to unmanageable or useless deployments. |

**The common thread:** every one of these mistakes comes from treating evaluation as a
mechanical scoring exercise instead of a decision process anchored in the business
problem.

---

[⬅ Previous](#sec24) · [🏠 Table of Contents](#toc) · [Next ➡](#sec26)"""))

CELLS_4.append(("md", r"""📗 **Back to basics:** only as a formal list here — the nearest basic material is
**Section 15: Cheat Sheet** in `ML_Model_Evaluation_Simple.ipynb`, whose “Best used
when” column encodes the same lessons.
"""))

CELLS_4.append(("md", r"""<a id="sec26"></a>
## 26. Beginner vs Expert Summary

| Concept | Beginner Perspective | Expert Perspective |
|---|---|---|
| **Accuracy** | "How often is the model right overall?" | "Only meaningful if classes are balanced and errors cost the same; often replaced by a cost-weighted objective." |
| **Precision** | "Of predicted positives, how many were correct?" | "Must be interpreted at the operating threshold, against prevalence, intervention cost, and FP burden." |
| **Recall** | "Of actual positives, how many did we find?" | "Is the cost of the FN; convert to expected caught cases and missed-case cost for the business." |
| **Specificity** | "How often are actual negatives left alone?" | "Drives operational load (false alarms × cost); equals 1 − FPR, the ROC x-axis." |
| **F1** | "Balances precision and recall." | "Assumes symmetric costs; use Fβ or a real cost function when errors are asymmetric." |
| **ROC-AUC** | "How well the model separates classes." | "A prevalence-invariant ranking summary; does not select a threshold and can mislead on imbalanced data." |
| **PR-AUC** | "Good for rare positives." | "Prevalence-dependent — compare only within datasets; evaluate precision at the operating recall when possible." |
| **Calibration** | "Does 70% mean 70%? Check the reliability diagram." | "Separate calibration from sharpness and discrimination; Brier bundles calibration + sharpness while ECE isolates calibration; check per subgroup." |
| **Classification report** | "Per-class precision/recall/F1 plus averages." | "Read per-class rows for the classes whose errors are expensive; check support; macro-vs-weighted gap reveals imbalance blindness." |
| **Cross-validation** | "K-fold averages performance over splits." | "Choose the CV scheme to match data structure; fit preprocessing inside folds; report mean ± std; score with the business metric." |
| **Threshold** | "Lower = more positives, higher = fewer." | "An operating point chosen to optimize expected cost — for well-calibrated probabilities with costs C_FP and C_FN the optimum is t* = C_FP/(C_FP + C_FN), not a default of 0.5." |
| **The best model** | "The one with the best score." | "The model + threshold pair that best satisfies the objectives and constraints of the real-world problem." |

---

[⬅ Previous](#sec25) · [🏠 Table of Contents](#toc) · [Next ➡](#sec27)"""))

CELLS_4.append(("md", r"""📗 **Back to basics:** the whole `ML_Model_Evaluation_Simple.ipynb` is the
beginner version of this material — start there, then come back here for the expert
perspective.
"""))

CELLS_4.append(("md", r"""<a id="sec27"></a>
## 27. Final Cheat Sheet

| Metric | Formula | Measures | Best used when | Main problem | Business question |
|---|---|---|---|---|---|
| **Accuracy** | (TP+TN) / all | Overall correctness | Balanced classes, equal error costs | Misleading on imbalanced data | "How often is the model correct overall?" |
| **Precision** | TP / (TP+FP) | Trust in positive predictions | FP is expensive | Ignores missed positives | "When the model says positive, how often should I trust it?" |
| **Recall / Sensitivity** | TP / (TP+FN) | Fraction of positives found | FN is expensive | Ignores false alarms | "How many actual positives can I find?" |
| **Specificity** | TN / (TN+FP) | Fraction of negatives left alone | False alarms on negatives are costly | Ignores missed positives | "How well can I avoid incorrectly flagging negatives?" |
| **F1** | 2·P·R / (P+R) | Balance of precision & recall | Symmetric error costs | Treats FP and FN as equal | "Can I balance precision and recall?" |
| **ROC-AUC** | Area under ROC | Ranking quality across thresholds | Balanced data, model comparison | Doesn't pick a threshold; optimistic when imbalanced | "How well does the model rank positives above negatives?" |
| **PR-AUC** | Area under PR curve | Usefulness for the rare class | Imbalanced / rare-event problems | Prevalence-dependent; harder to explain | "How useful is the model at finding rare positives?" |
| **Brier score** | mean(p − y)² | Truthfulness of the probabilities themselves | Probabilities drive prices or decisions | Harder to explain to non-experts | "How truthful are the probability numbers?" |
| **MAE** | mean\|y−ŷ\| | Average error (units) | Explainable, symmetric errors | Ignores error-size distribution | "On average, how far off are the predictions?" |
| **MSE** | mean(y−ŷ)² | Squared errors | Large errors are disproportionately bad | Squared units, outlier-sensitive | "How much do large errors hurt?" |
| **RMSE** | √MSE | Error size, big-error penalty | Same as MSE + interpretable units | Outlier-sensitive | "What is a typical error, penalizing big mistakes?" |
| **R²** | 1 − SS_res/SS_tot | Variance explained vs mean baseline | Relative goodness of fit | Can hide large absolute errors | "How much of the variation does the model explain?" |

---

[⬅ Previous](#sec26) · [🏠 Table of Contents](#toc) · [Next ➡](#sec28)"""))

CELLS_4.append(("md", r"""📗 **Back to basics:** see **Section 15: Cheat Sheet** in
`ML_Model_Evaluation_Simple.ipynb` — the essential table with every formula and
business question.
"""))

CELLS_4.append(("md", r"""<a id="sec28"></a>
## 28. Practice Exercises

Solve these **before** looking at the solutions at the end of the notebook.
The exercises deliberately ask *why*, not only *what*.

### Beginner

1. A confusion matrix has TN = 85, FP = 15, FN = 25, TP = 75. Identify which cell is
   which and write the matrix in scikit-learn layout.
2. From the same confusion matrix, compute accuracy by hand (show your working).
3. From the same confusion matrix, compute precision by hand and explain in one sentence
   what it means.
4. From the same confusion matrix, compute recall and specificity by hand.
5. A model on a rare-disease dataset (1% positive) predicts "no disease" for everyone.
   Its accuracy is 99%. Is this model useful? Explain with recall.

### Intermediate

6. Two models for the same problem: Model A (accuracy 0.95, precision 0.93, recall 0.45)
   and Model B (accuracy 0.92, precision 0.85, recall 0.88). Which would you choose for
   cancer screening? Justify using error costs.
7. A dataset has 99% negative class. A model scores 98% accuracy and 0.05 recall. Which
   two or three metrics should you report instead of accuracy, and why?
8. Interpret this classification report row: `Class 1  precision 0.62  recall 0.18
   f1-score 0.28  support 231`. What does each number mean, and what should the team do?
9. A logistic regression outputs probabilities. At threshold 0.5 it flags 40 cases; at
   0.7 it flags 12. Without seeing the data, what happened to precision, recall, FP and
   FN, and which way does the business cost move if FN is the expensive error?
10. Compute F1 for a model with precision 0.9 and recall 0.2, and for one with precision
    0.5 and recall 0.6. Which has the higher F1? Does that make it the better model?

### Advanced / Expert

11. A bank must approve loans. A default costs €8,000; a rejected good customer costs
    €500. Design the evaluation: which metrics, which threshold strategy, and how you
    would decide between two models with different precision/recall profiles.
12. On a fraud dataset with 0.5% positives, Model X has ROC-AUC 0.97 and PR-AUC 0.10;
    Model Y has ROC-AUC 0.93 and PR-AUC 0.45. Which do you trust for deployment and why?
13. Using the threshold table below (from a model's test set), find the threshold that
    minimizes cost if FP = €200 and FN = €1,000. Show your cost calculation per row.

    | Threshold | FP | FN |
    |---|---|---|
    | 0.10 | 400 | 5 |
    | 0.30 | 120 | 20 |
    | 0.50 | 40 | 45 |
    | 0.70 | 10 | 80 |
    | 0.90 | 2 | 140 |

14. Two models are compared with a single 80/20 train/test split: Model A scores 0.92,
    Model B 0.90. Explain why this comparison is unreliable, and design a better
    comparison for an imbalanced dataset.
15. You discover your feature engineering used the *full* dataset's statistics (e.g.,
    `StandardScaler().fit(X_all)` before splitting). What is this problem called, why is
    it serious, and what is the correct procedure?

[Jump to Exercise Solutions ⬇](#exercise-solutions)

---

[⬅ Previous](#sec27) · [🏠 Table of Contents](#toc) · [Next ➡](#exercise-solutions)"""))

CELLS_4.append(("md", r"""📗 **Back to basics:** see **Section 16: Practice Exercises** in
`ML_Model_Evaluation_Simple.ipynb` — nine graded exercises with solutions, if the full
set feels like a lot.
"""))

CELLS_4.append(("md", r"""<a id="exercise-solutions"></a>
## Exercise Solutions

### Beginner

1. scikit-learn layout (rows = actual, columns = predicted):
   `[[TN=85, FP=15], [FN=25, TP=75]]` — i.e., top-left TN, top-right FP, bottom-left FN,
   bottom-right TP.
2. Accuracy = (TP + TN) / (TP + TN + FP + FN) = (75 + 85) / (75 + 85 + 15 + 25) = 160/200 = **0.80**.
3. Precision = TP / (TP + FP) = 75 / (75 + 15) = 75/90 ≈ **0.833**. Meaning: of everything
   the model predicted positive, about 83% were actually positive.
4. Recall = TP / (TP + FN) = 75 / (75 + 25) = **0.75**. Specificity = TN / (TN + FP) =
   85 / (85 + 15) = **0.85**.
5. **No.** Recall = TP / (TP + FN) = 0 / (0 + 100) = **0** — the model finds zero sick
   patients. The 99% accuracy is just the majority class being easy. In disease screening,
   this model is worse than useless.

### Intermediate

6. For **cancer screening**, FN is the dangerous error (a missed patient). Model B has far
   higher recall (0.88 vs 0.45), so it finds roughly twice as many real cases. Choose
   **Model B** despite its lower accuracy and precision, and tune the threshold to the
   acceptable false-alarm load.
7. Report **recall, precision, and PR-AUC** (plus F1 for balance). Accuracy hides the fact
   that the model finds almost none of the 1% positives; the PR curve shows how useful the
   model is for the rare class directly.
8. `support 231` = there are 231 actual Class-1 samples; precision 0.62 = 62% of
   predictions of Class 1 were correct; recall 0.18 = only 18% of actual Class-1 samples
   were found; F1 = 0.28 balances these. The team should investigate why so many Class-1
   cases are missed (class imbalance, threshold, features, or data quality) before
   trusting the model.
9. Raising the threshold to 0.7 flags fewer cases: precision usually **rises**, recall
   **falls**, FP **falls**, FN **rises**. If FN is the expensive error, total business
   cost likely **rises** — the more permissive threshold (0.5, or lower) was better.
10. F1 for (0.9, 0.2): 2·0.9·0.2/(0.9+0.2) = 0.36/1.1 ≈ **0.327**. F1 for (0.5, 0.6):
    2·0.5·0.6/(0.5+0.6) = 0.6/1.1 ≈ **0.545**. The second has higher F1 — but "better"
    depends on which error costs more. If a missed positive is catastrophic (low recall
    unacceptable), the first model's profile may still be the right choice.

### Advanced / Expert

11. Design: (a) define the costs — FP (default) €8,000, FN (rejected good customer) €500;
    (b) since FP ≈ 16× FN, emphasize **precision** and choose a **high threshold**;
    (c) sweep thresholds to minimize total cost = FP×8000 + FN×500; (d) compare candidate
    models on this cost, plus ROC/PR curves and cross-validation stability; (e) present
    the confusion matrix at the optimal threshold with the cost breakdown to the risk
    committee.
12. Trust **Model Y**. On 0.5% positives, ROC-AUC is inflated and insensitive to the
    false-alarm burden, while PR-AUC directly reflects how useful the model is at finding
    frauds. Y's PR-AUC (0.45) is 4.5× X's (0.10) — X would flood operations with false
    positives despite its impressive ROC-AUC.
13. Cost per row = FP×200 + FN×1000:
    - 0.10: 400·200 + 5·1000 = 80,000 + 5,000 = 85,000
    - 0.30: 120·200 + 20·1000 = 24,000 + 20,000 = 44,000
    - 0.50: 40·200 + 45·1000 = 8,000 + 45,000 = **53,000**
    - 0.70: 10·200 + 80·1000 = 2,000 + 80,000 = 82,000
    - 0.90: 2·200 + 140·1000 = 400 + 140,000 = 140,400
    → Minimum at **0.30** (44,000).
14. A single split can be lucky or unlucky; the 0.02 gap may be noise. Use **stratified
    K-fold cross-validation** (5–10 folds), report mean ± std for a business-relevant
    metric (e.g., F1 or PR-AUC), and check that Model A is better *and* more stable.
15. This is **data leakage**: the scaler saw test-set statistics, so test scores are
    optimistically biased and the model's real-world performance will be worse. Correct
    procedure: split **first**, then fit the scaler **only on the training split** and
    transform the test split with it (inside each CV fold as well).

---

[⬅ Previous](#sec28) · [🏠 Table of Contents](#toc) · [Next ➡](#sec29)"""))

CELLS_4.append(("md", r"""<a id="sec29"></a>
## 29. Final Quiz

### Section A — Multiple choice

1. Which statement about accuracy is **true**?
   a) Accuracy is always the most important metric.
   b) Accuracy treats FP and FN as equally bad.
   c) Accuracy is reliable on imbalanced datasets.
   d) Accuracy ignores TN.

2. Precision is the metric to emphasize when…
   a) missed positives are catastrophic
   b) false positives are expensive
   c) classes are perfectly balanced
   d) you never look at the confusion matrix

3. On an imbalanced dataset, which pair is most informative?
   a) Accuracy and specificity
   b) Precision–Recall curve and PR-AUC
   c) MSE and RMSE
   d) Training and validation accuracy

4. ROC-AUC = 0.5 means…
   a) the model is perfect
   b) the model ranks no better than random
   c) the model is overfitting
   d) the model has 50% accuracy

5. Specificity is defined as…
   a) TP / (TP + FN)
   b) TP / (TP + FP)
   c) TN / (TN + FP)
   d) TN / (TN + FN)

### Section B — True / False

6. True or False: A model can have 99% accuracy and still be useless for fraud detection.
7. True or False: The default classification threshold of 0.5 is always the best threshold
   for deployment.
8. True or False: Higher F1 always means a better business solution.
9. True or False: Using the test set multiple times to tune a model invalidates the test
   set as a measure of generalization.
10. True or False: On imbalanced data, ROC-AUC can look optimistically high while PR-AUC
    reveals poor performance on the minority class.

### Section C — Short conceptual questions

11. In one sentence each: what does FP mean and what does FN mean in **fraud detection**?
12. Give one example where **precision** matters more than recall, and one where
    **recall** matters more than precision.
13. Why can a 99% accurate classifier still be useless? (Use the words: minority class,
    majority class, recall.)
14. What is the difference between macro and weighted averaging in a classification
    report?
15. What is data leakage, and why does it make evaluation results meaningless?

### Section D — Numerical questions

16. TP = 40, TN = 300, FP = 10, FN = 50. Compute accuracy, precision, recall, and
    specificity.
17. A model outputs these probabilities for 5 samples: [0.9, 0.7, 0.55, 0.4, 0.2] with
    true labels [1, 1, 0, 1, 0]. At threshold 0.5, build the confusion matrix. What
    happens to the predictions if the threshold is raised to 0.6?
18. Compute F1 for precision 0.8 and recall 0.6.
19. A house-price model has MAE = 12,000 and RMSE = 25,000. What does each number mean,
    and which would you report to a CEO and why?
20. R² = 0.90 for a pricing model. Does this guarantee the model is good for the
    business? Explain.

### Section E — Scenario / business decision questions

21. Model A has higher accuracy but significantly lower recall than Model B. Which would
    you choose for disease screening, and why?
22. Why might a fraud detection system prefer recall over accuracy?
23. A bank's manual-investigation team can handle only 100 alerts per day. Precision is
    currently 0.30. What happens if they deploy a model with recall 0.95 and precision
    0.10? What trade-off should they make?
24. You must evaluate a model that will decide whether to send expensive retention offers
    to customers predicted to churn. FP costs €20 per offer; FN costs €500 in lost
    revenue. Which metric should drive the threshold, and should the threshold be higher
    or lower than 0.5?
25. A team reports only ROC-AUC (0.96) for a rare-event model. What three questions
    should you ask before approving deployment?

[Jump to Quiz Answers ⬇](#quiz-answers)

---

[⬅ Previous](#exercise-solutions) · [🏠 Table of Contents](#toc) · [Next ➡](#quiz-answers)"""))

CELLS_4.append(("md", r"""📗 **Back to basics:** see **Section 17: Final Quiz** in
`ML_Model_Evaluation_Simple.ipynb` — a 15-question check before you attempt the full
quiz.
"""))

CELLS_4.append(("md", r"""<a id="quiz-answers"></a>
## Quiz Answers

### Section A — Multiple choice

1. **b** — accuracy counts FP and FN both as one wrong prediction, giving them equal
   weight.
2. **b** — precision is the direct measure of the false-alarm burden among positive
   predictions.
3. **b** — the PR curve and PR-AUC focus on the rare positive class, where accuracy and
   even ROC-AUC mislead.
4. **b** — AUC 0.5 means a random positive is scored above a random negative only half the
   time: no better than chance.
5. **c** — Specificity = TN / (TN + FP): of all actual negatives, how many are correctly
   left alone.

### Section B — True / False

6. **True.** 99% accuracy is achievable by predicting the majority class and finding zero
   frauds (recall = 0).
7. **False.** 0.5 is a default; the right threshold minimizes the business's cost of FP
   and FN.
8. **False.** F1 assumes symmetric error costs; when one error is much more expensive, the
   highest-F1 model may be the wrong choice.
9. **True.** Each use of the test set leaks information into model selection; eventually
   you are fitting to the test set and it no longer measures generalization.
10. **True.** The huge negative class inflates ROC-AUC; PR-AUC directly reflects the
    false-positive burden on the rare class.

### Section C — Short conceptual questions

11. **FP** = a legitimate transaction is flagged as fraudulent (customer blocked,
    investigation). **FN** = a fraudulent transaction is not flagged (money lost).
12. **Precision example:** spam filtering or expensive manual review — a false alarm costs
    real money or trust. **Recall example:** disease screening or intrusion detection —
    a miss can be catastrophic.
13. If the minority class (e.g., frauds) is 1% of the data, a model predicting the
    majority class for everything gets 99% accuracy but recall = 0: it finds none of the
    cases the business actually cares about.
14. Macro averages each class equally regardless of size; weighted averages each class by
    its support (number of samples). A big gap between them means the small class is
    handled much worse than the large one.
15. Data leakage is when information unavailable at prediction time (e.g., test-set
    statistics or future data) reaches the training process. It makes scores unrealistically
    good, so the evaluation no longer estimates real-world performance. See the two
    demonstrations in [Section 18](#sec18).

### Section D — Numerical questions

16. Accuracy = (40+300)/(40+300+10+50) = 340/400 = **0.85**.
    Precision = 40/(40+10) = **0.80**. Recall = 40/(40+50) = **0.444**.
    Specificity = 300/(300+10) ≈ **0.968**.
17. At 0.5: predictions [1,1,1,0,0], truth [1,1,0,1,0] → TP=2 (samples 1,2), FP=1 (sample
    3), FN=1 (sample 4), TN=1 (sample 5). At 0.6: sample 3 (0.55) becomes 0 and sample 4
    (0.4) stays 0 → TP=2, FP=0, FN=2, TN=1. Higher threshold: fewer FP, more FN.
18. F1 = 2·(0.8·0.6)/(0.8+0.6) = 0.96/1.4 ≈ **0.686**.
19. MAE = 12,000: the average prediction is off by €12,000. RMSE = 25,000: the
    big-error-penalized typical error is €25,000. Report **MAE to the CEO** (interpretable
    in business units); keep RMSE internally to watch large errors.
20. **No.** R² says the model explains 90% of price *variation*, but errors can still be
    large in absolute terms (e.g., MAE 40,000) or concentrated on the most profitable
    customers. Evaluate absolute error, cost of errors, and stability before deployment.

### Section E — Scenario / business decision questions

21. **Model B** for disease screening. FN is the dangerous error: a missed patient. Higher
    recall means fewer sick people sent home, even if accuracy and precision are lower.
    The false alarms (FP) are the acceptable price of catching the disease.
22. Because the cost of a missed fraud (FN) — stolen money — is far larger than the cost
    of a false alert, and because accuracy is dominated by the huge number of legitimate
    transactions and hides the miss rate entirely.
23. With precision 0.10, only 1 in 10 alerts is real fraud; if the team can handle 100
    alerts/day, they only investigate ~10 real cases — the model's high recall is useless
    operationally. They should raise the threshold to improve precision (fewer, more
    accurate alerts) until the daily alert volume fits the team's capacity, accepting
    lower recall.
24. FN (€500) costs 25× FP (€20), so **recall** should drive the threshold, and the
    threshold should be **lower than 0.5** to catch more churners — the offers are cheap
    compared to the lost revenue. (But still bound the total offer budget.)
25. Ask: (1) What is PR-AUC and what does the PR curve look like — is the model actually
    useful for the rare class? (2) At what threshold will we operate, and what are FP/FN
    counts and costs there? (3) Was evaluation done with proper splits/cross-validation
    without leakage, on data matching the deployment population?

---

[⬅ Previous](#sec29) · [🏠 Table of Contents](#toc) · [Next ➡](#framework)"""))

CELLS_4.append(("md", r"""<a id="framework"></a>
## How Should I Evaluate a Machine Learning Model? — The Final Workflow

Use this checklist on every project, in this order.

### Step 1 — Understand the business/problem objective
What decision will this model support? Approve a loan? Block a transaction? Diagnose a
patient? The decision defines everything that follows.

### Step 2 — Understand the data distribution
Class balance? Prevalence? Which class is rare, and which class matters?

### Step 3 — Identify what FP means
What actually happens when the model cries wolf? Who is affected, and what does it cost?

### Step 4 — Identify what FN means
What actually happens when the model misses a positive? What does that cost?

### Step 5 — Determine which error is more costly
Compare cost(FP) and cost(FN) in real units — not "which feels worse" but "which loses
more money / harms more people".

### Step 6 — Select appropriate evaluation metrics
Pick the metrics that reflect the expensive error: precision if FP dominates, recall if FN
dominates, F1 or cost-weighted objectives for balance, PR-AUC for rare events.

### Step 7 — Evaluate multiple metrics rather than relying on one number
One number hides the story. Report the metric set that answers the business questions.

### Step 8 — Examine the confusion matrix
Look at the actual counts: how many FP and FN, for which classes. Metrics summarize; the
matrix shows what really happens.

### Step 9 — Evaluate different classification thresholds
Sweep thresholds; watch precision, recall, FP, FN, and cost at each point.

### Step 10 — Consider ROC-AUC and/or PR-AUC where appropriate
ROC-AUC for overall ranking (balanced data); PR curves/PR-AUC for rare-event problems.

### Step 11 — Consider cross-validation and model stability
Report mean ± std over stratified folds; make sure the choice is not a lucky split.

### Step 12 — Evaluate the model from both technical and business perspectives
Is it technically sound (stable, no leakage, correct metrics) **and** operationally
feasible (fits capacity, acceptable error profile, explainable)?

### Step 13 — Choose an appropriate operating threshold
Optimize the threshold on the business cost function, not on a default of 0.5.

### Step 14 — Only then decide whether the model is suitable for deployment
The decision is the conclusion of the whole chain — never a number taken in isolation.

---

> ### The key message
>
> **Model evaluation is not about finding the model with the highest score.**
> It is about finding the model and decision threshold that best satisfy the objectives
> and constraints of the real-world problem.

---

[⬅ Previous](#quiz-answers) · [🏠 Table of Contents](#toc)"""))

CELLS_4.append(("md", r"""<a id="final-message"></a>
## Congratulations — you have finished the notebook! 🎉

**What you can now do:**

- Compute every core evaluation metric by hand and with scikit-learn
- Read a confusion matrix and a classification report properly
- Explain why accuracy fails on imbalanced data
- Choose between precision, recall, F1, ROC-AUC, and PR-AUC for a given problem
- Evaluate models with cross-validation and spot data leakage
- Turn FP/FN costs into a threshold and a deployment decision

**The three questions to carry into every project:**

1. *Technical:* Which model has the highest metric?
2. *Data Science:* Which metric matters for this problem?
3. *Real-World:* What are the consequences if the model is wrong?

**Suggested next steps**

- Re-run Section 21's case study with *different* values for `COST_FP` and `COST_FN`
  and watch the recommended threshold change — that is the whole lesson made visible.
- Repeat the exercises on your own datasets: for every metric you compute, write down the
  FP and FN counts, the cost of each, and the threshold you would deploy.
- Read scikit-learn's documentation for the functions used here
  (`sklearn.metrics`, `sklearn.model_selection`) to deepen your command of the tools.

Thank you for working through this notebook. Evaluate well, and always ask *what the
errors cost*. 🚀

---

[⬅ Previous](#framework) · [🏠 Table of Contents](#toc)"""))
