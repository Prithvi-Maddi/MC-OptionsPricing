import math

#Standard Normal CDF using an error function without external libs
def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

#Black-Scholes pricing for a European call/put option without dividends
def price(S:float,K:float, r:float, sigma:float, T:float, call: bool = True) -> float:
    if T<=0 or sigma <= 0:
        intrinsic = max(S - K, 0.0) if call else max(K-S, 0.0)
        return math.exp(-r*T) * intrinsic if T> 0 else intrinsic
    
    d1 = (math.log(S/K) + (r+ 0.5*sigma*sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)

    if call:
        return S * normal_cdf(d1) - K * math.exp(-r *T) * normal_cdf(d2)
    else:
        return K * math.exp(-r * T) * normal_cdf(-d2) - S * normal_cdf(-d1)
    