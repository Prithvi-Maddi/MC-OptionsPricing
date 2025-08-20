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

#Antithetic variates using terminal distribution, returns a dict
def price_antithetic_terminal(S0, K, r, sigma, T, n_paths, call=True, seed=None):
    rng = np.random.default_rng(seed)
    m = n_paths // 2
    Z = rng.standard_normal(m)
    drift = (r - 0.5 * sigma**2)
    volT = sigma * math.sqrt(T)

    ST_pos = S0 * np.exp(drift + volT * Z)
    ST_neg = S0 * np.exp(drift + volT * (-Z))

    if call:
        payoff_pos = np.maximum(ST_pos - K, 0.0)
        payoff_neg = np.maximum(ST_neg - K, 0.0)
    else:
        payoff_pos = np.maximum(K - ST_pos, 0.0)
        payoff_neg = np.maximum(K - ST_neg, 0.0)

    payoff_pair_avg = 0.5 * (payoff_pos + payoff_neg)
    discounted = math.exp(-r * T) * payoff_pair_avg

    mc_price = float(np.mean(discounted))
    se = float(np.std(discounted, ddof=1) / np.sqrt(m))   
    return {
        "mc_price": mc_price,
        "se": se,
        "ci_low": mc_price - 1.96*se,
        "ci_high": mc_price + 1.96*se,
        "elapsed_s": None,
        "n_paths": int(2*m), "steps": 1, "call": call
    }
 
 #   Plain Monte Carlo using the closed-form distribution of S_T.
def price_terminal(S0, K, r, sigma, T, n_paths, call=True, seed=None):
   
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    drift = (r - 0.5 * sigma**2) * T
    volT = sigma * math.sqrt(T)
    ST = S0 * np.exp(drift + volT * Z)

    if call:
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    disc = math.exp(-r * T)
    Y = disc * payoff

    mc_price = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n_paths))
    return {
        "mc_price": mc_price,
        "se": se,
        "ci_low": mc_price - 1.96*se,
        "ci_high": mc_price + 1.96*se,
        "elapsed_s": None,
        "n_paths": n_paths, "steps": 1, "call": call
    }

# Control Variate with X = discounted S_T. Uses terminal distribution for Euro Options
def price_control_variate_terminal(S0, K, r, sigma, T, n_paths, call=True, seed=None):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    drift = (r - 0.5 * sigma**2) * T
    volT = sigma * math.sqrt(T)

    ST = S0 * np.exp(drift + volT * Z)

    if call:
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    disc = math.exp(-r * T)
    Y = disc * payoff           # target
    X = disc * ST               # control, E[X] = S0

    # Estimate optimal c = Cov(Y,X)/Var(X)
    Xc = X - np.mean(X)
    Yc = Y - np.mean(Y)
    varX = np.dot(Xc, Xc) / (n_paths - 1)
    covYX = np.dot(Xc, Yc) / (n_paths - 1)
    c_opt = covYX / max(varX, 1e-18)

    Y_tilde = Y - c_opt * (X - S0)
    mc_price = float(np.mean(Y_tilde))
    se = float(np.std(Y_tilde, ddof=1) / np.sqrt(n_paths))
    return {
        "mc_price": mc_price,
        "se": se,
        "ci_low": mc_price - 1.96*se,
        "ci_high": mc_price + 1.96*se,
        "elapsed_s": None,
        "n_paths": n_paths, "steps": 1, "call": call,
        "c_opt": float(c_opt)
    }

