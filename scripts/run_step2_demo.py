import csv, json, os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.black_scholes import price as bs_price
from src.mc_naive import price as mc_price

def main():
    # Demo parameters
    S0, K, r, sigma, T = 100.0, 100.0, 0.02, 0.20, 1.0
    steps = 252
    n_paths = 20000
    call = True

    bs = bs_price(S0, K, r, sigma, T, call=call)
    res = mc_price(S0, K, r, sigma, T, steps, n_paths, call=call, seed=123)

    outdir = ROOT / "results"
    outdir.mkdir(parents=True, exist_ok=True)

    summary = (
        f"Parameters:\n"
        f"  S0={S0}, K={K}, r={r}, sigma={sigma}, T={T}, steps={steps}, n_paths={n_paths}\n\n"
        f"Black–Scholes (call): {bs:.6f}\n"
        f"Monte Carlo (naive): {res['mc_price']:.6f}\n"
        f"SE: {res['se']:.6f}\n"
        f"95% CI: ({res['ci_low']:.6f}, {res['ci_high']:.6f})\n"
        f"Runtime (s): {res['elapsed_s']:.3f}\n"
    )
    (outdir / "step2_summary.txt").write_text(summary)

    with (outdir / "step2_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["S0","K","r","sigma","T","steps","n_paths","call","bs_price","mc_price","se","ci_low","ci_high","elapsed_s"])
        w.writerow([S0,K,r,sigma,T,steps,n_paths,call,bs,res["mc_price"],res["se"],res["ci_low"],res["ci_high"],res["elapsed_s"]])

    (outdir / "step2_summary.json").write_text(json.dumps({
        "params": {"S0":S0,"K":K,"r":r,"sigma":sigma,"T":T,"steps":steps,"n_paths":n_paths,"call":call},
        "black_scholes": bs,
        "monte_carlo": res
    }, indent=2))

    print("Saved results to", outdir)


if __name__ == "__main__":
    main()