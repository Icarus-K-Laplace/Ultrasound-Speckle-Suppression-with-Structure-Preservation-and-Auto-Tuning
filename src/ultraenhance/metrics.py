import cv2
import numpy as np

def compute_metrics_with_gt(before_u8, after_u8, gt_slice):
    roi = (gt_slice > 0).astype(np.uint8)
    bg = (roi == 0)

    if roi.sum() < 10 or bg.sum() < 10:
        return {k: np.nan for k in [
            "CNR_before","CNR_after","ENL_before","ENL_after",
            "Edge_before","Edge_after","Edge_ratio"
        ]}

    b = before_u8.astype(np.float32)
    a = after_u8.astype(np.float32)

    def cnr(x):
        rv, bv = x[roi > 0], x[bg]
        return float(abs(rv.mean()-bv.mean()) / np.sqrt(rv.var()+bv.var()+1e-6))

    def enl(x):
        rv = x[roi > 0]
        return float((rv.mean()**2) / (rv.var()+1e-6))

    def edge(x):
        gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.sqrt(gx**2 + gy**2).mean())

    eb, ea = edge(b), edge(a)
    return {
        "CNR_before": cnr(b),
        "CNR_after": cnr(a),
        "ENL_before": enl(b),
        "ENL_after": enl(a),
        "Edge_before": eb,
        "Edge_after": ea,
        "Edge_ratio": float(ea/(eb+1e-6))
    }
