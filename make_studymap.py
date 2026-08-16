"""Generate a one-page printable study map (PDF + PNG preview).

Maps the 17 Simple-notebook sections to their ML_Model_Evaluation.ipynb
counterparts and shows the recommended reading order. Designed to print
alongside ML_Model_Evaluation_Cheat_Sheet.pdf (same A4 layout, same palette).
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
AMBER_FILL = "#fbf6ec"
AMBER_LINE = "#d9c9a3"


def box(x, y, w, h, fill=FILL, ec=LINE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fill, ec=ec, lw=1.0))


def txt(x, y, s, size=8, weight="normal", color="black", ha="left", va="center"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va)


def header(x, y, s):
    txt(x, y, s, size=10.5, weight="bold", color=DARK)


def wrap(s, maxchars):
    """Wrap s into at most 2 lines by word boundaries; truncate with … if needed."""
    words = s.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= maxchars:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    if len(lines) > 2:
        lines = lines[:2]
        lines[-1] = lines[-1][: maxchars - 1].rstrip() + "…"
    return lines


def grid_line(x0, y0, x1, y1, color=LINE, lw=0.7):
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw)


# ---------------- Title ----------------
txt(ML, 11.54, "Machine Learning Model Evaluation — Study Map", size=17,
    weight="bold", color=DARK)
txt(ML, 11.30, "One page, two notebooks: where every topic lives, and in which order to read it",
    size=10.5, weight="bold", color=GRAY)
ax.plot([ML, W - ML], [11.06, 11.06], color=DARK, lw=2)
txt(ML, 10.80, "ML_Model_Evaluation_Simple.ipynb (beginner)  ⇄  ML_Model_Evaluation.ipynb (full course)",
    size=8, color=GRAY)

# ---------------- How to use ----------------
box(ML, 9.98, CW, 0.74)
header(ML + 0.14, 10.56, "How to use this map")
txt(ML + 0.14, 10.30, "The 17 Simple sections are the core curriculum, in order. Each row shows where the same topic is treated in depth —",
    size=7.6)
txt(ML + 0.14, 10.12, "jump there whenever you want more, or follow the ‘go deeper’ / ‘back to basics’ cards at the end of every section.",
    size=7.6)

# ---------------- Main table ----------------
rows = [
    ("1",  "Introduction to Model Evaluation",
     "1 · Introduction to Model Evaluation (+ Core Teaching Principle)",
     "the three questions of evaluation, overfitting vs generalization"),
    ("2",  "The Confusion Matrix", "3 · The Confusion Matrix",
     "deeper error-cost framing, more examples"),
    ("3",  "Accuracy", "4 · Accuracy",
     "the accuracy trap in depth, when accuracy is the wrong metric"),
    ("4",  "Precision", "5 · Precision",
     "more false-alarm cases, precision vs threshold"),
    ("5",  "Recall (Sensitivity)", "6 · Recall / Sensitivity",
     "more missed-positive scenarios, multiclass recall"),
    ("6",  "Specificity", "7 · Specificity",
     "precision vs specificity, the negative class in depth"),
    ("7",  "F1 Score", "8 · F1 Score",
     "Fβ / weighted F1, when F1 misleads"),
    ("8",  "The Precision–Recall Trade-off", "9 · The Precision–Recall Trade-off",
     "cost-based threshold choice, the PR curve"),
    ("9",  "ROC Curve and AUC", "10 · The ROC Curve · 11 · AUC · 12 · ROC-AUC vs PR",
     "what AUC really measures, PR vs ROC on imbalance"),
    ("10", "The Classification Report", "14 · The Classification Report",
     "macro vs weighted averages, multiclass use"),
    ("11", "Imbalanced Datasets", "15 · Imbalanced Datasets",
     "resampling, class weights, threshold tuning"),
    ("12", "Cross-Validation", "17 · Cross-Validation",
     "K-Fold in depth, common CV mistakes"),
    ("13", "Regression Metrics", "20 · Regression Metrics",
     "MAPE, choosing among regression metrics"),
    ("14", "Mini Case Study: Loan Default",
     "21 · End-to-End Case Study · 22 · Business Decision · 23 · Cost-Sensitive",
     "three models, the cost-matrix decision rule"),
    ("15", "Cheat Sheet", "24 · Choosing the Right Metric · 27 · Final Cheat Sheet",
     "the selection guide and the complete summary"),
    ("16", "Practice Exercises", "28 · Practice Exercises",
     "15 exercises with full solutions"),
    ("17", "Final Quiz", "29 · Final Quiz",
     "25 questions plus the final workflow"),
]

table_top = 9.90
header_y = 9.64
row_h = 0.36
first_row_y = header_y - 0.66          # 8.98
# bottom of the lowest text, then give the box a little padding
last_row_bottom = first_row_y - 16 * row_h - 0.11
# box bottom sits below the lowest text (smaller y = lower on page)
table_bot = last_row_bottom - 0.16

box(ML, table_bot, CW, table_top - table_bot)
header(ML + 0.14, header_y, "The 17 Simple sections → where each is covered in depth")

colx = [0.60, 0.95, 2.80, 5.25]                  # absolute x (ML + 0.15 + offset)
colw = [0.35, 1.85, 2.45, 2.72]

hdr_txt = ["#", "Simple section", "Deep counterpart", "What the deep version adds"]
for c, (x, w, h) in enumerate(zip(colx, colw, hdr_txt)):
    txt(x + 0.03, header_y - 0.26, h, size=7.6, weight="bold", color=DARK)
grid_line(ML + 0.15, header_y - 0.40, ML + 0.15 + sum(colw), header_y - 0.40, lw=1.0)

y = first_row_y
for i, (num, title, deep, adds) in enumerate(rows):
    lines_t = wrap(title, 32)
    lines_d = wrap(deep, 44)
    lines_a = wrap(adds, 48)
    # draw each column, aligned to the first line of the tallest cell
    nlines = max(len(lines_t), len(lines_d), len(lines_a))
    base = y + 0.055 * (nlines - 1)
    txt(colx[0] + 0.03, base, num, size=7.6, weight="bold", color=DARK)
    for j, ln in enumerate(lines_t):
        txt(colx[1] + 0.03, base - j * 0.115, ln, size=7.0, weight="bold")
    for j, ln in enumerate(lines_d):
        txt(colx[2] + 0.03, base - j * 0.115, ln, size=7.0)
    for j, ln in enumerate(lines_a):
        txt(colx[3] + 0.03, base - j * 0.115, ln, size=6.6, color=GRAY)
    y -= row_h
    if i < len(rows) - 1:
        grid_line(ML + 0.15, y + row_h - 0.015, ML + 0.15 + sum(colw),
                  y + row_h - 0.015, lw=0.4)

# ---------------- Deep-only topics ----------------
deep_only = [
    ("2",  "Classification vs Regression", "framing only; light in Simple"),
    ("12", "ROC-AUC vs Precision–Recall", "why PR beats ROC for rare events"),
    ("13", "Probability Calibration", "rank ≠ calibration; reliability diagrams"),
    ("16", "Multiclass Evaluation", "per-class metrics, averaging"),
    ("18", "Data Leakage — When Your Model Cheats", "leaked features inflate metrics"),
    ("19", "Time-Series Evaluation", "forward-chaining, grouped splits"),
    ("25", "Common Mistakes", "the pitfalls, catalogued"),
    ("26", "Beginner vs Expert Summary", "the same ideas at two levels"),
]
deep_top = table_bot - 0.10
deep_h = 1.18
yellow_bot = deep_top - deep_h
box(ML, yellow_bot, CW, deep_h, fill=AMBER_FILL, ec=AMBER_LINE)
header(ML + 0.14, deep_top - 0.14, "Deep-only topics — not in the Simple notebook")
txt(ML + 0.14, deep_top - 0.36, "Meet these only in the full course; they extend the ideas above (reading order: follow the numbered sections).",
    size=7.2, color=GRAY)
col1 = ML + 0.18
col2 = ML + 3.78
for i, (num, title, note) in enumerate(deep_only):
    x = col1 if i < 4 else col2
    txt(x, deep_top - 0.56 - (i % 4) * 0.16, f"{num} · {title} — {note}", size=6.8)

# ---------------- Reading order ----------------
ro_top = yellow_bot - 0.06
ro_h = 0.56
box(ML, ro_top - ro_h, CW, ro_h)
header(ML + 0.14, ro_top - 0.10, "Recommended reading order")
txt(ML + 0.14, ro_top - 0.34, "Beginner:  Simple 1 → 2 → … → 17 straight through  (≈ 1–2 hours, no prerequisites)",
    size=7.4)
txt(ML + 0.14, ro_top - 0.50, "Deep:  full course 1 → 2 → … → 29 in order   ·   Hybrid:  start Simple, jump to depth via the cards",
    size=7.4)

# ---------------- Footer ----------------
txt(W / 2, 0.42, "Study map · print alongside ML_Model_Evaluation_Cheat_Sheet.pdf   ·   notebooks: ML_Model_Evaluation_Simple.ipynb & ML_Model_Evaluation.ipynb",
    size=7.5, color=GRAY, ha="center")
txt(W / 2, 0.24, "numpy · pandas · matplotlib · scikit-learn   —   no deep-learning libraries",
    size=7, color=GRAY, ha="center")

# ---------------- overflow check ----------------
renderer = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
bad = []
for t in ax.texts:
    bb = t.get_window_extent(renderer=renderer)
    (x0, y0) = inv.transform((bb.x0, bb.y0))
    (x1, y1) = inv.transform((bb.x1, bb.y1))
    if x0 < -0.01 or x1 > 1.01 or y0 < -0.01 or y1 > 1.01:
        bad.append((t.get_text()[:40], round(x0, 2), round(x1, 2),
                    round(y0, 2), round(y1, 2)))
if bad:
    print("OVERFLOW WARNINGS:")
    for b in bad:
        print("  ", b)
else:
    print("No text overflow — layout fits the page.")

fig.savefig("ML_Model_Evaluation_Study_Map.pdf", format="pdf")
fig.savefig("ML_Model_Evaluation_Study_Map.png", format="png", dpi=150)
print("Saved ML_Model_Evaluation_Study_Map.pdf and .png")
