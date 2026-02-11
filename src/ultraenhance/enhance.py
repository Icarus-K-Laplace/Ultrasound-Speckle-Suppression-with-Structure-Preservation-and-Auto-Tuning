import cv2
import numpy as np

def _to_float01(img_u8):
    return img_u8.astype(np.float32) / 255.0

def _to_u8(img01):
    return np.clip(img01 * 255.0, 0, 255).astype(np.uint8)

def lee_filter(log_img, win=7, noise_percentile=20):
    mean = cv2.boxFilter(log_img, -1, (win, win), normalize=True)
    mean2 = cv2.boxFilter(log_img * log_img, -1, (win, win), normalize=True)
    var = np.maximum(mean2 - mean * mean, 0.0)
    noise_var = max(float(np.percentile(var, noise_percentile)), 1e-6)
    w = var / (var + noise_var + 1e-6)
    return mean + w * (log_img - mean)

def coherence_and_grad(img01, win=7):
    gx = cv2.Sobel(img01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img01, cv2.CV_32F, 0, 1, ksize=3)

    jxx = cv2.boxFilter(gx * gx, -1, (win, win), normalize=True)
    jyy = cv2.boxFilter(gy * gy, -1, (win, win), normalize=True)
    jxy = cv2.boxFilter(gx * gy, -1, (win, win), normalize=True)

    num = (jxx - jyy) ** 2 + 4.0 * (jxy ** 2)
    den = (jxx + jyy) ** 2 + 1e-6
    coh = np.clip(num / den, 0.0, 1.0)

    grad = np.sqrt(gx**2 + gy**2)
    grad = grad / (grad.max() + 1e-6)
    return coh, grad

def enhance_ultrasound_u8(img_u8, p):
    """
    p keys:
      lee_win, noise_percentile, edge_mix_coh, edge_mix_grad,
      alpha_base, alpha_edge, contrast_gain, post_sigma
    """
    img01 = _to_float01(img_u8)

    # multiplicative speckle -> log-domain
    log_img = np.log(img01 + 1e-6)
    log_dn = lee_filter(
        log_img,
        win=int(p["lee_win"]),
        noise_percentile=float(p["noise_percentile"])
    )

    dn01 = np.exp(log_dn)
    dn01 = np.clip(dn01, 0.0, 1.0)

    coh, grad = coherence_and_grad(img01, win=7)
    edge_w = np.clip(
        p["edge_mix_coh"] * coh + p["edge_mix_grad"] * grad, 0.0, 1.0
    )

    blur1 = cv2.GaussianBlur(img01, (0, 0), sigmaX=1.0)
    blur2 = cv2.GaussianBlur(img01, (0, 0), sigmaX=2.0)
    detail = 0.7 * (img01 - blur1) + 0.3 * (blur1 - blur2)

    out01 = dn01 + p["alpha_base"] * detail + p["alpha_edge"] * edge_w * detail

    local_mean = cv2.boxFilter(out01, -1, (7, 7), normalize=True)
    contrast = out01 - local_mean
    out01 = out01 + p["contrast_gain"] * edge_w * contrast

    if float(p["post_sigma"]) > 0:
        out01 = cv2.GaussianBlur(out01, (0, 0), sigmaX=float(p["post_sigma"]))

    out01 = np.clip(out01, 0.0, 1.0)
    return _to_u8(out01), _to_u8(coh), _to_u8(edge_w)
