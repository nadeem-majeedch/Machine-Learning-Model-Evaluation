"""Generate the GitHub Pages landing page (_site/index.html).

Called by .github/workflows/pages.yml after the notebooks are rendered to HTML.
Also usable locally: python build_index.py  (expects the rendered .html files
next to the source notebooks, and writes into _site/).
"""
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "_site"

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evaluating Machine Learning Models — Course Site</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
         Roboto, "Helvetica Neue", Arial, sans-serif; color: #222; background: #f4f7fb; }
  header { background: #1f4e79; color: #fff; padding: 52px 24px; text-align: center; }
  header h1 { margin: 0 0 10px; font-size: 2.1rem; }
  header p { margin: 0 auto; max-width: 720px; font-size: 1.05rem; opacity: .92; }
  header .idea { margin-top: 18px; font-style: italic; opacity: .85; }
  main { max-width: 960px; margin: 0 auto; padding: 32px 20px 48px; }
  h2 { color: #1f4e79; border-bottom: 2px solid #b9cbe0; padding-bottom: 6px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 18px; margin: 18px 0 34px; }
  .card { background: #fff; border: 1px solid #d7e1ee; border-radius: 8px;
          padding: 20px; box-shadow: 0 1px 3px rgba(31,78,121,.08); }
  .card h3 { margin: 0 0 8px; color: #1f4e79; }
  .card .meta { color: #666; font-size: .85rem; margin-bottom: 10px; }
  .card p { margin: 0 0 14px; font-size: .93rem; line-height: 1.5; }
  .btn { display: inline-block; background: #1f4e79; color: #fff; text-decoration: none;
         padding: 9px 16px; border-radius: 6px; font-size: .9rem; margin-right: 8px; }
  .btn.secondary { background: #fff; color: #1f4e79; border: 1px solid #1f4e79; }
  .assets a { color: #1f4e79; }
  .assets li { margin: 6px 0; }
  footer { background: #1f4e79; color: #dbe6f2; text-align: center;
           padding: 18px; font-size: .85rem; }
  footer a { color: #fff; }
</style>
</head>
<body>
<header>
  <h1>Evaluating Machine Learning Models</h1>
  <p>A complete teaching course for Data Science &amp; AI students — from the
     confusion matrix to cost-sensitive decision rules, calibration, and
     time-series validation.</p>
  <p class="idea">“Do not ask only which model has the highest score. Ask which
     metric fits the problem, what each error costs, and what happens when the
     model is wrong.”</p>
</header>

<main>
  <h2>📓 The notebooks</h2>
  <div class="grid">
    <div class="card">
      <h3>Simple Edition</h3>
      <div class="meta">17 sections · ≈ 1–2 hours · no prerequisites</div>
      <p>The essential curriculum in plain language: confusion matrix, accuracy,
         precision, recall, F1, ROC/AUC, and picking the right metric — with
         worked examples and a loan case study.</p>
      <a class="btn" href="ML_Model_Evaluation_Simple.html">Read the notebook</a>
    </div>
    <div class="card">
      <h3>Full Course</h3>
      <div class="meta">29 sections · beginner to expert · 7 interactive widgets</div>
      <p>The complete treatment: probability calibration, cost-sensitive decision
         rules, data leakage, time-series evaluation, cross-validation, multiclass
         evaluation, and a full loan case study.</p>
      <a class="btn" href="ML_Model_Evaluation.html">Read the notebook</a>
    </div>
  </div>

  <h2>🖨️ Printable study materials</h2>
  <ul class="assets">
    <li><a href="ML_Model_Evaluation_Cheat_Sheet.pdf">Cheat Sheet (PDF)</a> — one-page
        summary: confusion matrix, metrics table, FP vs FN cost guide.</li>
    <li><a href="ML_Model_Evaluation_Cheat_Sheet.png">Cheat Sheet (PNG preview)</a></li>
    <li><a href="ML_Model_Evaluation_Study_Map.pdf">Study Map (PDF)</a> — every Simple
        section mapped to its deep counterpart, plus reading order.</li>
    <li><a href="ML_Model_Evaluation_Study_Map.png">Study Map (PNG preview)</a></li>
  </ul>
</main>

<footer>
  Licensed under the <a href="https://opensource.org/licenses/MIT">MIT License</a> ·
  Built with NumPy · Pandas · Matplotlib · scikit-learn
</footer>
</body>
</html>
"""


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "index.html").write_text(PAGE, encoding="utf-8")
    print(f"Wrote {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
