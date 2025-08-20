import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.mc_numpy import price_terminal as mc_plain
from src.mc_numpy import price_antithetic_terminal as mc_anti
from src.mc_numpy import price_control_variate_terminal as mc_cv

def test_variance_reduction_reduces_se():
    S0=K=100.0; r=0.02; sigma=0.20; T=1.0; N=50_000; seed=7
    p = mc_plain(S0,K,r,sigma,T,N,True,seed)
    a = mc_anti (S0,K,r,sigma,T,N,True,seed)
    c = mc_cv   (S0,K,r,sigma,T,N,True,seed)
    # Antithetic should not be worse; CV should usually be strictly better
    assert a["se"] <= p["se"] * 1.05   # allow tiny noise
    assert c["se"] <  p["se"] * 0.8    # expect a noticeable drop