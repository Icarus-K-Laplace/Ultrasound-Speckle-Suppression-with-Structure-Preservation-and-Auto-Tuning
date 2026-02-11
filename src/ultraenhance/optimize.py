from itertools import product

def make_param_grid():
    grid = {
        "lee_win": [5,7,9],
        "noise_percentile": [15,20,25],
        "edge_mix_coh": [0.7,0.8],
        "edge_mix_grad": [0.3,0.2],
        "alpha_base": [0.20,0.30,0.40],
        "alpha_edge": [0.15,0.22,0.30],
        "contrast_gain": [0.00,0.04,0.08],
        "post_sigma": [0.0,0.4,0.6]
    }
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in product(*vals):
        p = dict(zip(keys, combo))
        if p["alpha_base"] + p["alpha_edge"] <= 0.75:
            yield p

def score_from_means(m):
    d_cnr = m["CNR_after"] - m["CNR_before"]
    d_enl = m["ENL_after"] - m["ENL_before"]
    edge_penalty = abs(m["Edge_ratio"] - 1.0)

    score = 2.2*d_cnr + 1.8*d_enl - 1.2*edge_penalty
    if d_cnr >= 0: score += 0.15
    else: score -= 0.15
    if d_enl >= 0: score += 0.15
    else: score -= 0.10
    if 0.9 <= m["Edge_ratio"] <= 1.1: score += 0.15
    else: score -= 0.10
    return float(score)
