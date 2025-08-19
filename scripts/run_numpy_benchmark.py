import time, csv, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.black_scholes import price as bs_price
from src.mc_naive import price as mc_naive
from src.mc_numpy import price as mc_numpy

def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    res = fn(*args, **kwargs)
    res["elapsed_s"] = time.perf_counter() - start
    return res

def main():
    S0, K, r, sigma, T = 100.0, 100.0, 0.02, 0.20, 1.0
    steps = 252
    call = True
    grid = [20_000, 50_000, 100_000]
    
    outdir = ROOT / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "numpy_benchmark.csv"
    bs = bs_price(S0, K, r, sigma, T, call=call)
    print(f"Black–Scholes (call): {bs:.6f}\n")
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method","n_paths","steps","price","se","ci_low","ci_high","time_s","rel_err","speedup_vs_naive"])
        for n_paths in grid:
            res_naive = timed(mc_naive, S0,K,r,sigma,T,steps,n_paths,call,123)
            res_np = timed(mc_numpy, S0,K,r,sigma,T,steps,n_paths,call,123)

            rel_err_naive = abs(res_naive["mc_price"] - bs) / bs
            rel_err_np = abs(res_np["mc_price"] - bs) / bs
            speedup = res_naive["elapsed_s"]/res_np["elapsed_s"]

            print(f"n={n_paths:>7}")
            print(f"  naive → {res_naive['mc_price']:.4f}, SE={res_naive['se']:.4f}, t={res_naive['elapsed_s']:.3f}s, rel_err={rel_err_naive:.3%}")
            print(f"  numpy → {res_np['mc_price']:.4f}, SE={res_np['se']:.4f}, t={res_np['elapsed_s']:.3f}s, rel_err={rel_err_np:.3%}, speedup≈{speedup:.1f}x\n")

            w.writerow(["naive", n_paths, steps, res_naive["mc_price"], res_naive["se"],
                        res_naive["ci_low"], res_naive["ci_high"], res_naive["elapsed_s"], rel_err_naive, ""])
            w.writerow(["numpy", n_paths, steps, res_np["mc_price"], res_np["se"],
                        res_np["ci_low"], res_np["ci_high"], res_np["elapsed_s"], rel_err_np, speedup])

    print(f"\nSaved results to {csv_path}")

if __name__ == "__main__":
    main()