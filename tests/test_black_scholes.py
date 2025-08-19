import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.black_scholes import price as bs_price

def test_black_scholes():
    S0, K, r, sigma, T = 100.0, 100.0, 0.02, 0.20, 1.0
    c = bs_price(S0, K, r, sigma, T, call=True)
    p = bs_price(S0, K, r, sigma, T, call=False)
    assert 0 < c < S0
    assert 0 < p < K
