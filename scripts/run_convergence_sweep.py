import csv, time, sys, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.mc_numpy import price as mc_numpy
from src.black_scholes import price as bs_price

def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    res = fn(*args, **kwargs)
    res["elapsed_s"] = time.perf_counter() - start
    return res

def main():
    S0, K, r, sigma, T = 100.0, 100.0, 0.02, 0.20, 1.0
    steps = 252
    call = True
    path_grid = [10_000, 20_000, 50_000, 100_000, 200_000, 500_000]

    outdir = ROOT / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "convergence_sweep.csv"

    bs = bs_price(S0, K, r, sigma, T, call=call)
    print(f"BS(call) = {bs:.6f}")

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_paths","mc_price","se","ci_low","ci_high","time_s","rel_err"])
        for n in path_grid:
            res = timed(mc_numpy, S0, K, r, sigma, T, steps, n, call, 123)
            rel_err = abs(res["mc_price"] - bs) / bs
            w.writerow([n, res["mc_price"], res["se"], res["ci_low"], res["ci_high"], res["elapsed_s"], rel_err])
            print(f"N={n:>7}  price={res['mc_price']:.6f}  SE={res['se']:.6f}  time={res['elapsed_s']:.3f}s  rel_err={rel_err:.2%}")

    print(f"\nSaved → {csv_path}")

if __name__ == "__main__":
    main()