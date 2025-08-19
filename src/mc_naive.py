import math, random, statistics, time
from .gbm import simulate_terminal

def price( S0: float, K: float, r:float, sigma:float, T:float, steps: int, n_paths: int, call: bool=True, seed: int | None = 123):
    if seed is not None:
        random.seed(seed)

    disc = math.exp(-r * T)
    payoffs = []
    t0 = time.perf_counter()
    for i in range(n_paths):
        ST = simulate_terminal(S0, r, sigma, T, steps)
        payoff = max(ST - K, 0) if call else max(K-ST, 0.0)
        payoffs.append(disc * payoff)
    elapsed = time.perf_counter() - t0

    mean = statistics.fmean(payoffs)
    std = statistics.pstdev(payoffs) if n_paths > 1 else 0.0
    se = std / math.sqrt(max(1, n_paths))
    ci_low, ci_high = mean - 1.96 * se, mean + 1.96*se

    return {"mc_price": mean, "se": se, "ci_low": ci_low, "ci_high": ci_high, "elapsed_s": elapsed, "n_paths": n_paths, "steps": steps, "call": call}

