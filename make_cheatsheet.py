"""Generate a one-page printable cheat sheet (PDF + PNG preview).

Source of truth: ML_Model_Evaluation_Simple.ipynb (cheat-sheet table + FP/FN framework).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

W, H = 8.27, 11.69          # A4 portrait, inches
ML = 0.45                    # left margin
CW = W - 2 * ML              # content width

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

DARK = "#1f4e79"
GRAY = "#555555"
LINE = "#b9cbe0"
FILL = "#f2f6fb"


def box(x, y, w, h, fill=FILL):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fill, ec=LINE, lw=1.0))


def txt(x, y, s, size=8, weight="normal", color="black", ha="left", va="center"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va)


def header(x, y, s):
    txt(x, y, s, size=10.5, weight="bold", color=DARK)


def grid_line(x0, y0, x1, y1, color=LINE, lw=0.8):
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw)


# ---------------- Title ----------------
txt(ML, 11.50, "Machine Learning Model Evaluation", size=17, weight="bold", color=DARK)
txt(ML, 11.26, "One-Page Cheat Sheet — Simple Edition", size=11, weight="bold", color=GRAY)
ax.plot([ML, W - ML], [11.02, 11.02], color=DARK, lw=2)
txt(ML, 10.76, "from ML_Model_Evaluation_Simple.ipynb   ·   full course: ML_Model_Evaluation.ipynb",
    size=8, color=GRAY)

# ---------------- 1. Confusion matrix ----------------
box(ML, 9.42, CW, 1.24)
header(ML + 0.14, 10.48, "1 · The Confusion Matrix")

gx = ML + 0.14
rowl = 1.05           # left label column width
cw = 1.18             # cell column width
gy_top = 10.12
rh = 0.335

# header cells
txt(gx + rowl + cw / 2, gy_top + 0.17, "Predicted: NO", size=8, weight="bold")
txt(gx + rowl + cw + cw / 2, gy_top + 0.17, "Predicted: YES", size=8, weight="bold")
txt(gx + rowl / 2, gy_top - rh + 0.17, "Actually: YES", size=8, weight="bold")
txt(gx + rowl / 2, gy_top - 2 * rh + 0.17, "Actually: NO", size=8, weight="bold")

cells = [(gy_top + 0.17, "FN", "TP"),
         (gy_top - rh + 0.17, "TN", "FP")]
for (cy, left, right) in cells:
    txt(gx + rowl + cw / 2, cy, left, size=9, weight="bold")
    txt(gx + rowl + cw + cw / 2, cy, right, size=9, weight="bold")

# grid borders: 4 vertical lines (left, label|cells, mid, right) and 3 horizontal
for vx in [gx, gx + rowl, gx + rowl + cw, gx + rowl + 2 * cw]:
    grid_line(vx, gy_top - 2 * rh, vx, gy_top)
for hy in [gy_top, gy_top - rh, gy_top - 2 * rh]:
    grid_line(gx, hy, gx + rowl + 2 * cw, hy)

# definitions on the right
dx0, dy0, dw = 4.05, 9.98, 1.85
defs = [("TP", "said YES and was right  — hit", "seagreen"),
        ("TN", "said NO and was right", "seagreen"),
        ("FP", "said YES but wrong  — false alarm", "crimson"),
        ("FN", "said NO but wrong  — miss", "crimson")]
for i, (name, d, color) in enumerate(defs):
    col = i % 2
    row = i // 2
    x = dx0 + col * dw
    y = dy0 - row * 0.30
    txt(x, y, name, size=8.5, weight="bold", color=color)
    txt(x + 0.32, y, d, size=7.5)

# ---------------- 2. Decision framework ----------------
box(ML, 8.52, CW, 0.80)
header(ML + 0.14, 9.18, "2 · The Decision Framework")
txt(ML + 0.14, 8.84, "Business Problem  →  Cost of Errors  →  Important Error Type  →  Appropriate Metric",
    size=8.2)
txt(ML + 0.14, 8.62, "Model Evaluation  →  Threshold Selection  →  Business Decision",
    size=8.2)

# ---------------- 3. Metrics table ----------------
box(ML, 4.28, CW, 4.14)
header(ML + 0.14, 8.26, "3 · Metrics at a Glance")

cols = [0.55, 1.70, 3.60, 5.35]     # x positions (relative to ML + 0.15)
colw = [1.10, 1.85, 1.70, 2.55]     # widths
hdr = ["Metric", "Formula", "Measures", "Business question"]
for c, (x, w, h) in enumerate(zip(cols, colw, hdr)):
    txt(ML + 0.15 + x, 7.94, h, size=8, weight="bold", color=DARK)
grid_line(ML + 0.15, 7.78, ML + 0.15 + sum(colw), 7.78, lw=1.0)

rows = [
    ("Accuracy",   "(TP+TN)/(TP+TN+FP+FN)",   "Overall correctness",        "How often is the model right overall?"),
    ("Precision",  "TP/(TP+FP)",              "Trust in positive predictions", "When it says YES, how often is it right?"),
    ("Recall",     "TP/(TP+FN)",              "Finding the positives",       "How many real positives does it find?"),
    ("Specificity","TN/(TN+FP)",              "Correctly clearing negatives","How well does it avoid false alarms?"),
    ("F1 Score",   "2·P·R/(P+R)",             "Balance of P and R",          "Are both precision and recall decent?"),
    ("ROC-AUC",    "area under ROC",          "Ranking quality",             "How well does it rank pos vs neg?"),
    ("MAE",        "mean(|y−ŷ|)",             "Average error (regression)",  "How far off is the average prediction?"),
    ("RMSE",       "√MSE",                    "Error, big errors penalized", "How bad are the big mistakes?"),
    ("R²",         "1 − SSres/SStot",         "Variance explained",          "How much variation does it explain?"),
]
ry = 7.52
for (m, f, me, q) in rows:
    txt(ML + 0.15 + cols[0], ry, m, size=7.6, weight="bold")
    txt(ML + 0.15 + cols[1], ry, f, size=7.6)
    txt(ML + 0.15 + cols[2], ry, me, size=7.6)
    txt(ML + 0.15 + cols[3], ry, q, size=7.6)
    ry -= 0.355

# ---------------- 4. FP vs FN ----------------
box(ML, 2.60, CW, 1.56)
header(ML + 0.14, 4.02, "4 · FP vs FN — which error is more expensive?")

fp_x = ML + 0.18
fn_x = ML + 3.95
txt(fp_x, 3.76, "FP — false alarm:", size=8.5, weight="bold", color="crimson")
txt(fp_x + 1.30, 3.76, "model says YES, truth is NO", size=7.5)
txt(fn_x, 3.76, "FN — miss:", size=8.5, weight="bold", color="crimson")
txt(fn_x + 1.10, 3.76, "model says NO, truth is YES", size=7.5)

fp_lines = ["  spam: real email filtered out",
            "  fraud: legitimate customer blocked",
            "  cost: annoyance, lost trust, wasted checks",
            "  FP expensive → maximise PRECISION (raise threshold)"]
fn_lines = ["  disease: sick patient sent home",
            "  fraud: fraudulent payment approved",
            "  cost: untreated disease, lost money, breaches",
            "  FN expensive → maximise RECALL (lower threshold)"]
y = 3.50
for line in fp_lines:
    txt(fp_x, y, line, size=7.5)
    y -= 0.22
y = 3.50
for line in fn_lines:
    txt(fn_x, y, line, size=7.5)
    y -= 0.22

# ---------------- 5. Golden rules ----------------
box(ML, 1.02, CW, 1.42)
header(ML + 0.14, 2.30, "5 · Golden Rules")
rules = [
    "• Never judge an imbalanced problem by accuracy alone — read precision, recall and F1.",
    "• The threshold 0.5 is a convention, not a decision — set it from the costs of FP and FN.",
    "• Always read the confusion matrix — a single number can hide the whole story.",
    "• The highest metric is not automatically the best model — the business question decides.",
    "• Judge the model only on data it has never seen.",
]
ry = 2.06
for r in rules:
    txt(ML + 0.18, ry, r, size=7.8)
    ry -= 0.215

# ---------------- Footer ----------------
txt(W / 2, 0.42, "From ML_Model_Evaluation_Simple.ipynb   ·   full course: ML_Model_Evaluation.ipynb",
    size=7.5, color=GRAY, ha="center")
txt(W / 2, 0.24, "numpy · pandas · matplotlib · scikit-learn   —   no deep-learning libraries",
    size=7, color=GRAY, ha="center")

# ---------------- overflow check ----------------
renderer = fig.canvas.get_renderer()
bad = []
for t in ax.texts:
    bb = t.get_window_extent(renderer=renderer)
    # convert pixel bbox to figure fraction
    inv = fig.transFigure.inverted()
    (x0, y0) = inv.transform((bb.x0, bb.y0))
    (x1, y1) = inv.transform((bb.x1, bb.y1))
    if x0 < -0.01 or x1 > 1.01 or y0 < -0.01 or y1 > 1.01:
        bad.append((t.get_text()[:40], round(x0, 2), round(x1, 2), round(y0, 2), round(y1, 2)))
if bad:
    print("OVERFLOW WARNINGS:")
    for b in bad:
        print("  ", b)
else:
    print("No text overflow — layout fits the page.")

fig.savefig("ML_Model_Evaluation_Cheat_Sheet.pdf", format="pdf")
fig.savefig("ML_Model_Evaluation_Cheat_Sheet.png", format="png", dpi=150)
print("Saved ML_Model_Evaluation_Cheat_Sheet.pdf and .png")
