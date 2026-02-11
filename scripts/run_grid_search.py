import os
import json
import cv2
import pandas as pd

from src.ultraenhance.io_utils import read_nii, to_2d_slice, norm_to_uint8
from src.ultraenhance.enhance import enhance_ultrasound_u8
from src.ultraenhance.metrics import compute_metrics_with_gt
from src.ultraenhance.optimize import make_param_grid, score_from_means

CONFIG = "configs/default.json"

def main():
    with open(CONFIG, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    pairs = cfg["pairs"]
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    valid_pairs = []
    for p in pairs:
        if os.path.exists(p["nii"]) and os.path.exists(p["gt"]):
            valid_pairs.append(p)

    if not valid_pairs:
        print("No valid pairs found.")
        return

    all_records = []
    best_score = -1e9
    best_params = None
    best_case_df = None

    for idx, pset in enumerate(make_param_grid(), 1):
        case_rows = []
        for pair in valid_pairs:
            img = to_2d_slice(read_nii(pair["nii"]))
            gt  = to_2d_slice(read_nii(pair["gt"]))

            img_u8 = norm_to_uint8(img)
            out_u8, coh_u8, edge_u8 = enhance_ultrasound_u8(img_u8, pset)

            m = compute_metrics_with_gt(img_u8, out_u8, gt)
            case_rows.append({"case": os.path.basename(pair["nii"]), **m})

        case_df = pd.DataFrame(case_rows)
        mean_m = case_df.mean(numeric_only=True).to_dict()
        score = score_from_means(mean_m)

        rec = {"idx": idx, **pset, "score": score, **mean_m}
        all_records.append(rec)

        if score > best_score:
            best_score = score
            best_params = pset
            best_case_df = case_df.copy()

    grid_df = pd.DataFrame(all_records).sort_values("score", ascending=False)
    grid_df.to_csv(os.path.join(out_dir, "grid_results.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([best_params]).to_csv(os.path.join(out_dir, "best_params.csv"), index=False, encoding="utf-8-sig")
    best_case_df.to_csv(os.path.join(out_dir, "best_metrics.csv"), index=False, encoding="utf-8-sig")

    # save compare panels with best params
    best_img_dir = os.path.join(out_dir, "best_compare")
    os.makedirs(best_img_dir, exist_ok=True)
    for i, pair in enumerate(valid_pairs, 1):
        img = to_2d_slice(read_nii(pair["nii"]))
        gt  = to_2d_slice(read_nii(pair["gt"]))
        img_u8 = norm_to_uint8(img)
        out_u8, coh_u8, edge_u8 = enhance_ultrasound_u8(img_u8, best_params)

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
        cv2.imwrite(os.path.join(best_img_dir, f"best_compare_{i:02d}.png"), panel)

    print("Done.")
    print("Best params:", best_params)
    print("Best score:", best_score)

if __name__ == "__main__":
    main()
