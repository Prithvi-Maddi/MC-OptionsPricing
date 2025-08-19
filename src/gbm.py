import math, random

#Generate N(0,1) from two uniforms via Box-Muller
def box_muller() -> float:
    u1 = max(1e-12, random.random()) 
    u2 = random.random()

    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0*math.pi * u2)

#Simulate a GBM Path (risk-nuetral) and return S_T
def simulate_terminal(S0: float, r:float, sigma: float, T: float, steps: int) -> float:
    dt = T/steps
    drift = (r - 0.5*sigma*sigma) *dt
    vol = sigma * math.sqrt(dt)
    S = S0
    for i in range(steps):
        Z = box_muller()
        S *= math.exp(drift + vol*Z)
    return S
