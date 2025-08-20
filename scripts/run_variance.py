import time, csv, sys, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.black_scholes import price as bs_price
from src.mc_numpy import (
    price_terminal as mc_plain,      
    price_antithetic_terminal as mc_anti,
    price_control_variate_terminal as mc_cv
)

def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    res = fn(*args, **kwargs)
    res["elapsed_s"] = time.perf_counter() - t0
    return res

def min_paths_for_target(method, S0,K,r,sigma,T, target_rel=0.005, seed=123, grid=None):
    if grid is None:
        grid = [5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000]
    bs = bs_price(S0,K,r,sigma,T,True)
    best = None
    for n in grid:
        res = timed(method, S0,K,r,sigma,T,n,True,seed)
        rel = abs(res["mc_price"] - bs) / bs
        if best is None or rel < best[0]:
            best = (rel, n, res)
        if rel <= target_rel:
            return (rel, n, res, bs)
    return (*best, bs)  # didn’t hit target

def main():
    S0=K=100.0; r=0.02; sigma=0.20; T=1.0
    target = 0.005  # 0.5% relative error target
    out = ROOT / "results"
    out.mkdir(exist_ok=True, parents=True)
    csv_path = out / "variance_test.csv"

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method","n_paths","price","se","time_s","rel_err_vs_bs","hit_target"])
        for name, fn in [
            ("plain_terminal", mc_plain),
            ("antithetic_terminal", mc_anti),
            ("control_variate_terminal", mc_cv),
        ]:
            rel, n, res, bs = min_paths_for_target(fn, S0,K,r,sigma,T, target_rel=target)
            hit = rel <= target
            print(f"{name:22s}  n={n:7d}  price={res['mc_price']:.6f}  SE={res['se']:.6f}  time={res['elapsed_s']:.3f}s  rel_err={rel:.3%}  target_hit={hit}")
            w.writerow([name, n, res["mc_price"], res["se"], res["elapsed_s"], rel, hit])

    print(f"\nSaved → {csv_path}")

if __name__ == "__main__":
    main()