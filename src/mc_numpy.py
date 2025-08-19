import numpy as np
import math

def price(S0: float, K: float, r:float, sigma:float, T:float, steps: int, n_paths: int, call: bool=True, seed: int | None = 123):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    dt = T / steps
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * math.sqrt(dt)

    Z = rng.standard_normal(size=(n_paths, steps))

    log_increments = drift + vol * Z
    log_paths = np.cumsum(log_increments, axis=1)
    ST = S0 * np.exp(log_paths[:, -1])
    if call:
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)

    discounted = math.exp(-r * T) * payoffs
    mc_price = float(np.mean(discounted))
    se = float(np.std(discounted, ddof=1) / math.sqrt(n_paths))
    ci_low, ci_high = mc_price - 1.96 * se, mc_price + 1.96 * se

    return {"mc_price": mc_price, "se": se, "ci_low": ci_low, "ci_high": ci_high, "elapsed_s": None, "n_paths": n_paths, "steps": steps, "call": call}