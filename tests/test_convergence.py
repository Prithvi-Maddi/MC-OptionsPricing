import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.mc_numpy import price as mc_numpy
from src.black_scholes import price as bs_price

def test_se_scales_like_inverse_sqrt_n():
    S0, K, r, sigma, T = 100.0, 100.0, 0.02, 0.20, 1.0
    steps = 252
    
    N1 = 50_000
    N2 = 200_000

    res1 = mc_numpy(S0, K, r, sigma, T, steps, N1, call=True)
    res2 = mc_numpy(S0, K, r, sigma, T, steps, N2, call=True)

    ratio = res1["se"] / max(res2["se"], 1e-12)
    assert 1.6 < ratio < 2.5, f"Expected ~2x SE reduction, got ratio = {ratio:.2f}"

def test_mc_is_close_to_bs_for_large_n():
    S0, K, r, sigma, T = 100.0, 100.0, 0.02, 0.20, 1.0
    steps = 252
    N = 500_000

    res = mc_numpy(S0, K, r, sigma, T, steps, N, call=True, seed=7)
    bs = bs_price(S0, K, r, sigma, T, call=True)
    rel_err = abs(res["mc_price"] - bs) / bs
    assert rel_err < 0.01, f"Expected MC price to be within 1% of BS price at N = {N}, got rel_err = {rel_err:.2%}"

    
    