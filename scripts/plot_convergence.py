import csv, sys, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import matplotlib.pyplot as plt

def main():
    csv_path = ROOT / "results" / "convergence_sweep.csv"
    rows = []
    with csv_path.open() as f:
        next(f)
        for line in f:
            n_paths, mc_price, se, ci_low, ci_high, time_s, rel_err = line.strip().split(",")
            rows.append({
                "n_paths": int(n_paths),
                "mc_price": float(mc_price),
                "se": float(se),
                "rel": float(rel_err),
                "t": float(time_s),
            })
    rows.sort(key=lambda r: r["n_paths"])
    N = [r["n_paths"] for r in rows]
    SE = [r["se"] for r in rows]
    REL = [r["rel"] for r in rows]
    T = [r["t"] for r in rows]

    # SE vs N (log-log to see slope ~ -1/2)
    plt.figure()
    plt.loglog(N, SE, marker="o")
    plt.xlabel("Number of paths (N)")
    plt.ylabel("Standard error (SE)")
    plt.title("SE vs N (expect slope ≈ -1/2)")
    plt.grid(True, which="both", ls=":")
    out1 = ROOT / "results" / "se_vs_n.png"
    plt.savefig(out1, dpi=160, bbox_inches="tight")

    # Relative error vs N
    plt.figure()
    plt.loglog(N, REL, marker="o")
    plt.xlabel("Number of paths (N)")
    plt.ylabel("Relative error vs BS")
    plt.title("Relative error vs N")
    plt.grid(True, which="both", ls=":")
    out2 = ROOT / "results" / "relerr_vs_n.png"
    plt.savefig(out2, dpi=160, bbox_inches="tight")

    # Time vs N (linear plot)
    plt.figure()
    plt.plot(N, T, marker="o")
    plt.xlabel("Number of paths (N)")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime vs N (expect ~linear)")
    plt.grid(True, ls=":")
    out3 = ROOT / "results" / "time_vs_n.png"
    plt.savefig(out3, dpi=160, bbox_inches="tight")

    print("Saved plots:", out1, out2, out3)

if __name__ == "__main__":
    main()