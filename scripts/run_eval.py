import os
import cv2
import pandas as pd

from src.ultraenhance.io_utils import read_nii, to_2d_slice, norm_to_uint8
from src.ultraenhance.enhance import enhance_ultrasound_u8
from src.ultraenhance.metrics import compute_metrics_with_gt

def run_eval(cfg):
    pairs = cfg["pairs"]
    out_dir = cfg.get("out_dir", "results")
    params = cfg["best_params"]

    os.makedirs(out_dir, exist_ok=True)
    rows = []

    for i, pair in enumerate(pairs, 1):
        if not (os.path.exists(pair["nii"]) and os.path.exists(pair["gt"])):
            print(f"[SKIP] missing file: {pair}")
            continue

        img = to_2d_slice(read_nii(pair["nii"]))
        gt  = to_2d_slice(read_nii(pair["gt"]))

        img_u8 = norm_to_uint8(img)
        out_u8, coh_u8, edge_u8 = enhance_ultrasound_u8(img_u8, params)
        m = compute_metrics_with_gt(img_u8, out_u8, gt)
        rows.append({"case": os.path.basename(pair["nii"]), **m})

        gt_mask = (gt > 0).astype("uint8") * 255
        overlay = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours((gt_mask > 0).astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0,255,0), 1)

        panel = cv2.hconcat([
            cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(out_u8, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(coh_u8, cv2.COLOR_GRAY2BGR),
            overlay
        ])
        cv2.imwrite(os.path.join(out_dir, f"eval_compare_{i:02d}.png"), panel)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, "eval_metrics.csv"), index=False, encoding="utf-8-sig")
        print(df)
    else:
        print("No valid cases.")

if __name__ == "__main__":
    import json
    with open("configs/default.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    run_eval(cfg)
