import numpy as np
from scipy.optimize import root_scalar

def f_Q(Q, 
        P, E, W, C_m,       # Observables
        alpha, beta, gamma, # Parameters
        kappa, D_s,         # Other constants
        r, delta, X):
    """
    Implicit function f(Q) = LHS(Q) - RHS(Q).

    We want f(Q) = 0 for the equation:
      exp( [kappa / U(Q)] * [ B(Q) - beta*P - gamma*E ] ) 
        = W * [ (P - C_m)* (kappa / U(Q)) * beta - 1 ]
    """

    # 1) U(Q):
    UQ = (Q * (1 + r)) / ((r + delta) * (1 + alpha * X)) - beta * P - gamma * E

    # Prevent division by zero or invalid U(Q)
    if UQ <= 0:
        return np.inf  # Push the solver away from invalid regions

    # 2) B(Q):
    BQ = D_s * np.exp(alpha * (Q - X)) * (1 + r) / (r + delta)

    # 3) Left-hand side (LHS):
    #    LHS = exp( [kappa / UQ]*[ BQ - beta*P - gamma*E ] )
    arg = (kappa / UQ) * (BQ - beta * P - gamma * E)
    LHS = np.exp(arg)

    # 4) Right-hand side (RHS):
    #    bracket = (P - C_m)* (kappa / UQ)* beta - 1
    bracket = (P - C_m) * (kappa / UQ) * beta - 1.0
    RHS = W * bracket
    err = LHS - RHS
    print("err", err)
    return err

def solve_for_Q_with_bounds(P, E, W, C_m, 
                            alpha, beta, gamma, 
                            kappa, D_s, 
                            r, delta, X,
                            Q_min= 1e-6, Q_max= 1e7):
    """
    Solve the implicit equation f_Q(Q) = 0 for Q, enforcing Q > 0.
    """

    # Define a lambda wrapper for f_Q
    func = lambda Q: f_Q(Q, P, E, W, C_m, alpha, beta, gamma,
                         kappa, D_s, r, delta, X)

    # Check the function values at the bounds
    f_min = func(Q_min)
    f_max = func(Q_max)

    print(f"f(Q_min={Q_min}) = {f_min}")
    print(f"f(Q_max={Q_max}) = {f_max}")
    quit()

    # Use root_scalar with bounds [Q_min, Q_max]
    result = root_scalar(func, bracket=[Q_min, Q_max], method='brentq')

    if not result.converged:
        raise RuntimeError(f"Root finding failed: {result.flag}")

    return result.root

# ---------------------------------------------------------------------------
# Example usage:
if __name__ == "__main__":
    # Example parameter values (YOU must supply real values)
    P     = 40000    # Price
    E     = 9000     # Emissions or energy usage
    J = 10
    C_m   = 15000     # Marginal cost
    alpha = 1     # Sensitivity param
    beta  = 0.5     # Price sensitivity in burn-in utility
    gamma = 1     # Emissions sensitivity in burn-in utility
    kappa = 10     # Scaling factor in exponent
    D_s   = 1600    # Scale factor for B(Q)
    r     = 0.00247    # Interest rate
    delta = 0.01    # Depreciation
    e = 0.26599820413049985 
    c = 0.11238312722172052 
    omega = 1.0

    X     = (beta*e + gamma*c)/omega

    W     =   J*kappa   # W_{s,t} or similar

    # Solve
    Q_star = solve_for_Q_with_bounds(P, E, W, C_m,
                         alpha, beta, gamma, kappa,
                         D_s, r, delta, X)
    
    print(f"Solved Q = {Q_star:.6f}")
