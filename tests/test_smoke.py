import numpy as np
from src.ultraenhance.enhance import enhance_ultrasound_u8

def test_enhance_smoke():
    img = (np.random.rand(256, 256) * 255).astype(np.uint8)
    params = {
        "lee_win": 7,
        "noise_percentile": 20,
        "edge_mix_coh": 0.8,
        "edge_mix_grad": 0.2,
        "alpha_base": 0.3,
        "alpha_edge": 0.22,
        "contrast_gain": 0.04,
        "post_sigma": 0.4
    }
    out, coh, edge = enhance_ultrasound_u8(img, params)
    assert out.shape == img.shape
    assert coh.shape == img.shape
    assert edge.shape == img.shape
    assert out.dtype == np.uint8
