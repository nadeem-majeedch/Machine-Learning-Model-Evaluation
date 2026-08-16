# 📊 Evaluating Machine Learning Models

A complete teaching course on **model evaluation** for Data Science & AI students —
two Jupyter notebooks that take learners from the very basics (confusion matrix,
accuracy) to expert-level understanding (cost-sensitive decision rules, calibration,
data leakage, time-series validation), plus printable study materials.

> **The one idea behind the whole course:** do not ask only *which model has the
> highest score*. Ask which metric fits the problem, what each type of error costs,
> and what happens when the model is wrong.

---

## 📓 The two notebooks

| Notebook | Audience | Coverage | Effort |
|---|---|---|---|
| [`ML_Model_Evaluation_Simple.ipynb`](ML_Model_Evaluation_Simple.ipynb) | Beginners, quick revision, non-experts | 17 short sections, plain language, worked examples | ≈ 1–2 hours |
| [`ML_Model_Evaluation.ipynb`](ML_Model_Evaluation.ipynb) | Students who want the full treatment | 29 sections, from beginner to expert, 7 interactive widgets | full course |

The two notebooks **link in both directions**: every Simple section ends with a
*"go deeper"* card naming its counterpart in the full course, and every full-course
section ends with a *"back to basics"* card pointing back to the Simple edition.
Both open with a *"How This Notebook Fits"* roadmap showing the recommended reading
paths (beginner / hybrid / deep).

### Simple edition — section list

1. Introduction to Model Evaluation
2. The Confusion Matrix
3. Accuracy
4. Precision
5. Recall (Sensitivity)
6. Specificity
7. F1 Score
8. The Precision–Recall Trade-off
9. ROC Curve and AUC
10. The Classification Report
11. Imbalanced Datasets
12. Cross-Validation
13. Regression Metrics
14. Mini Case Study: Loan Default Prediction
15. Cheat Sheet
16. Practice Exercises (9, with solutions)
17. Final Quiz (15 questions, with answers)

### Full course — section list

**Foundations**

1. Introduction to Model Evaluation
2. Classification vs Regression
3. The Confusion Matrix
4. Accuracy
5. Precision
6. Recall / Sensitivity
7. Specificity

**Classification metrics in depth**

8. F1 Score
9. The Precision–Recall Trade-off
10. The ROC Curve
11. AUC — Area Under the ROC Curve
12. ROC-AUC vs Precision–Recall
13. Probability Calibration
14. The Classification Report
15. Imbalanced Datasets
16. Multiclass Evaluation

**Model selection & regression**

17. Cross-Validation
18. Data Leakage — When Your Model Cheats
19. Time-Series Evaluation — Never Shuffle the Clock
20. Regression Metrics

**Putting it all together**

21. End-to-End Case Study: Loan Default Prediction
22. Business Decision Case Study
23. Cost-Sensitive Evaluation: From Cost Matrix to Decision Rule
24. Choosing the Right Metric — A Selection Guide
25. Common Mistakes When Evaluating Machine Learning Models
26. Beginner vs Expert Summary
27. Final Cheat Sheet

**Practice & assessment**

28. Practice Exercises (15, with solutions)
29. Final Quiz (25 questions, with answers)

### 🎛️ Interactive widgets (full course only)

Seven `ipywidgets` explorations, all pre-executed with embedded state (no kernel
needed to view the rendered results):

- **Threshold explorer** (loan case study) — drag a slider and watch FP/FN counts and
  total business cost update live
- **Threshold sweep on ROC & PR curves** (imbalanced fraud data) — live precision,
  recall, F1, and the operating point on both curves
- **Which model wins as the costs change?** — a high-precision vs high-recall model,
  with error-cost sliders
- **Calibration lab** — overconfidence slider + Platt-scaling toggle, with reliability
  diagram, Brier score, and ECE updating live
- **Temporal cut-point explorer** — choose the time split and watch honest R² wobble
  while the shuffled baseline stays flat
- **"Which 4 stores are new?"** — pick the held-out stores in the grouped time-series
  case study and see grouped R² (and where your choice sits in the full distribution)
- **Cost-matrix explorer** — set FP/FN costs and watch the optimal threshold move

## 🖨️ Printable study materials

| File | What it is |
|---|---|
| [`ML_Model_Evaluation_Cheat_Sheet.pdf`](ML_Model_Evaluation_Cheat_Sheet.pdf) | One-page A4 cheat sheet: confusion matrix, decision framework, 9 metrics with formulas, FP vs FN cost guide, golden rules |
| [`ML_Model_Evaluation_Study_Map.pdf`](ML_Model_Evaluation_Study_Map.pdf) | One-page A4 study map: all 17 Simple sections mapped to their deep counterparts, deep-only topics, and recommended reading order |

(PNG previews of both are included for quick viewing.)

## 🚀 Getting started

Requires Python 3.9+ with Jupyter.

```bash
pip install numpy pandas matplotlib scikit-learn ipywidgets

# then open a notebook:
jupyter notebook ML_Model_Evaluation_Simple.ipynb
```

Run the cells **top to bottom** — every notebook was executed end-to-end and
committed with its outputs, so you can browse the rendered results immediately.

## 🌐 GitHub Pages

This repository ships with a [GitHub Actions workflow](.github/workflows/pages.yml) that
automatically publishes the course as a static website. On every push to `main`/`master`
it:

1. renders both notebooks to self-contained HTML (`jupyter nbconvert --to html
   --embed-images` — figures are inlined, and the interactive widgets are embedded as
   static state, so they display without a running kernel),
2. copies the cheat-sheet and study-map PDFs/PNGs alongside them,
3. generates a landing page (`build_index.py`) and deploys everything to GitHub Pages.

**To enable it:** push the repo to GitHub, then go to *Settings → Pages* and set the
source to **“GitHub Actions”**. The site will be live at
`https://<username>.github.io/<repository>/` after the first run — no branch or folder
configuration needed.

## 🗂️ Repository layout

```
ML_Model_Evaluation.ipynb            # the full course (deliverable)
ML_Model_Evaluation_Simple.ipynb     # the beginner edition (deliverable)
ML_Model_Evaluation_Cheat_Sheet.pdf / .png
ML_Model_Evaluation_Study_Map.pdf / .png
.github/workflows/pages.yml          # GitHub Pages deployment workflow
build_index.py                       # generates the Pages landing page (index.html)
build_notebook.py                    # rebuilds the full notebook
build_simple.py                      # rebuilds the Simple notebook
nb_content_1..4.py                   # content modules for the full notebook
nb_simple_1.py, nb_simple_2.py       # content modules for the Simple notebook
make_cheatsheet.py                   # regenerates the cheat-sheet PDF/PNG
make_studymap.py                     # regenerates the study-map PDF/PNG
```

The `.ipynb` files are the source of truth for readers; the `nb_*.py` + `build_*.py`
files exist so the notebooks can be regenerated and re-executed reproducibly
(`python build_notebook.py` then execute, e.g. `jupyter nbconvert --execute --inplace`).

## 📚 Teaching philosophy

- Every metric is taught as: **plain definition → formula → hand-worked example →
  "which error is more expensive?" → when to use it**.
- FP/FN costs and business consequences are threaded through every section, not
  tacked on at the end.
- Accuracy alone is actively debunked (the 99%-accuracy/zero-recall trap), and the
  threshold `0.5` is framed as *a decision, not a default*.
- Exercises and quizzes are graded (beginner / intermediate / expert) with full
  solutions.

## 📄 License

This project is licensed under the [MIT License](LICENSE) — see `LICENSE` for the
full text. You are free to use, modify, and distribute the notebooks and materials
for teaching or any other purpose, provided the copyright notice is retained.

---

*Built with Python 3 and Jupyter Notebooks.*
### Engr. Dr. Muhammad Nadeem Majeed 
(Stay Blessed Always) 