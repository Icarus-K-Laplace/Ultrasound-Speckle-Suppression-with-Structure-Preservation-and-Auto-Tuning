import os
import pandas as pd
from datetime import datetime

def make_markdown_report(results_dir="results"):
    grid_csv = os.path.join(results_dir, "grid_results.csv")
    best_csv = os.path.join(results_dir, "best_metrics.csv")
    best_param_csv = os.path.join(results_dir, "best_params.csv")
    img_dir = os.path.join(results_dir, "best_compare")

    lines = []
    lines.append(f"# Ultrasound Enhancement Report\n")
    lines.append(f"Generated: {datetime.now()}\n")

    if os.path.exists(best_param_csv):
        bp = pd.read_csv(best_param_csv)
        lines.append("## Best Parameters\n")
        lines.append(bp.to_markdown(index=False))
        lines.append("\n")

    if os.path.exists(best_csv):
        bm = pd.read_csv(best_csv)
        lines.append("## Best Case Metrics\n")
        lines.append(bm.to_markdown(index=False))
        lines.append("\n")
        mean_vals = bm.mean(numeric_only=True)
        lines.append("## Mean Metrics\n")
        lines.append(mean_vals.to_frame("value").to_markdown())
        lines.append("\n")

    if os.path.exists(grid_csv):
        gs = pd.read_csv(grid_csv).head(10)
        lines.append("## Top-10 Grid Results\n")
        lines.append(gs.to_markdown(index=False))
        lines.append("\n")

    if os.path.exists(img_dir):
        lines.append("## Visual Results\n")
        for fn in sorted(os.listdir(img_dir)):
            if fn.lower().endswith(".png"):
                lines.append(f"### {fn}\n")
                lines.append(f"![{fn}]({os.path.join('best_compare', fn).replace('\\\\','/')})\n")

    out_md = os.path.join(results_dir, "REPORT.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report generated: {out_md}")

if __name__ == "__main__":
    make_markdown_report("results")
