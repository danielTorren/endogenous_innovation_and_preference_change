import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def calculate_alpha(D_plus, D_minus, X_plus):
    """
    Calculates alpha using the provided formula:
    alpha = (D^+ - D^-)/(D^- X^+)
    """
    alpha = (D_plus - D_minus) / (D_minus * X_plus)
    return alpha


def calculate_delta(beta_avg, P_avg, gamma_avg, E_avg, L):
    """
    Calculates delta assuming nu=1 using the formula:
    delta = 1 - (1 / ((beta_avg * P_avg - gamma_avg * E_avg)))^(1 / L)
    """

    delta = 1 - (1 / (beta_avg * P_avg + gamma_avg * E_avg)) ** (1 / L)
    return delta

def calculate_nu(beta_avg, P_avg, gamma_avg, E_avg, L, delta):

    nu = 1 /((1- delta) ** L * (beta_avg * P_avg + gamma_avg * E_avg))
    return nu

def calculate_average_quality_plus(alpha, X, r, delta, nu, beta_plus, W_plus, P_minus, C_minus, y_minus, E_minus):
    """
    Calculates average quality (Q_bar) using the derived formula:
    Q_bar = (alpha X + 1)(r + delta) / (factor) * sqrt(...)
    factor = 0.4 * (1 + r)
    """
    factor = 0.4 * (1 + r)
    term1 = nu * beta_plus * W_plus * (P_minus - C_minus)
    term2 = nu * W_plus * (beta_plus * C_minus + y_minus * E_minus)
    sqrt_term = np.sqrt((term1 ** 2 - term2 ** 2) / W_plus)
    
    Q_bar = ((alpha * X + 1) * (r + delta)) / factor * sqrt_term
    return Q_bar

def calculate_average_quality_minus(alpha, X, r, delta, nu, beta_minus, W_minus, P_plus, C_plus, y_plus, E_plus):
    """
    Calculates average quality (Q_bar) using the derived formula:
    Q_bar = (alpha X + 1)(r + delta) / (factor) * sqrt(...)
    factor = 0.4 * (1 + r)
    """
    factor = 1.6 * (1 + r)
    term1 = nu * beta_minus * W_minus * (P_plus - C_plus)
    term2 = nu * W_minus * (beta_minus * C_plus + y_plus * E_plus)
    sqrt_term = np.sqrt((term1 ** 2 - term2 ** 2) / W_minus)
    
    Q_bar = ((alpha * X + 1) * (r + delta)) / factor * sqrt_term
    return Q_bar


def calc_X(beta,c,gamma,e, omega):
    return (beta*c + gamma*e)/omega

# Adjust the system to solve iteratively for Q_bar and derive the other variables
def solve_for_Q_bar(Q_bar_guess, S, J, P_minus, P_plus, C_minus, C_plus, beta_plus, beta_minus, nu, gamma_minus, gamma_plus, E_minus, E_plus, r, delta, alpha, X_minus, X_plus):
    # Calculate B_plus and B_minus based on Q_bar
    B_plus = (1.6 * Q_bar_guess * (1 + r)) / ((alpha * X_minus + 1) * (r + delta))
    B_minus = (0.4 * Q_bar_guess * (1 + r)) / ((alpha * X_plus + 1) * (r + delta))
    
    # Calculate W_plus and W_minus
    W_plus = S*J*(B_plus / P_minus) ** 2
    W_minus = J*(B_minus / P_plus) ** 2
    
    # Calculate P_minus and P_plus based on the derived W values
    P_minus_calc = (nu * beta_plus * C_minus * W_plus +
                    np.sqrt(W_plus * ((B_minus)**2 + W_plus * nu**2 +
                                      (beta_plus * C_minus + gamma_minus * E_minus)**2))) / (beta_plus * W_plus)
    P_plus_calc = (nu * beta_minus * C_plus * W_minus +
                   np.sqrt(W_minus * ((B_plus)**2 + nu**2 * W_minus *
                                      (beta_minus * C_plus + gamma_plus * E_plus)**2))) / (beta_minus * W_minus)
    
    #print("Q_bar_guess, B_plus, B_minus, W_plus, W_minus,P_plus_calc, P_minus_calc",Q_bar_guess, B_plus, B_minus, W_plus, W_minus,P_plus_calc, P_minus_calc)
    # Error between calculated and given prices
    error = ((P_minus_calc - P_minus) / P_minus) ** 2 + ((P_plus_calc - P_plus) / P_plus) ** 2

    return error

# Adjust the system to solve iteratively for Q_bar and derive the other variables
def solve_for_Q_bar_P_minus(Q_bar_guess, S, J, P_minus, P_plus, C_minus, C_plus, beta_plus, beta_minus, nu, gamma_minus, gamma_plus, E_minus, E_plus, r, delta, alpha, X_minus, X_plus):
    # Calculate B_plus and B_minus based on Q_bar
    B_plus = (1.6 * Q_bar_guess * (1 + r)) / ((alpha * X_minus + 1) * (r + delta))
    B_minus = (0.4 * Q_bar_guess * (1 + r)) / ((alpha * X_plus + 1) * (r + delta))
    
    # Calculate W_plus and W_minus
    W_plus = S*J*(B_plus / P_minus) ** 2
    
    # Calculate P_minus and P_plus based on the derived W values
    P_minus_calc = (nu * beta_plus * C_minus * W_plus +
                    np.sqrt(W_plus * ((B_minus)**2 + W_plus * nu**2 +
                                      (beta_plus * C_minus + gamma_minus * E_minus)**2))) / (beta_plus * W_plus)
    
    #print("Q_bar_guess, B_plus, B_minus, W_plus, W_minus,P_plus_calc, P_minus_calc",Q_bar_guess, B_plus, B_minus, W_plus, W_minus,P_plus_calc, P_minus_calc)
    # Error between calculated and given prices
    error = ((P_minus_calc - P_minus) / P_minus) ** 2

    return error

# Adjust the system to solve iteratively for Q_bar and derive the other variables
def solve_for_Q_bar_P_plus(Q_bar_guess, S, J, P_minus, P_plus, C_minus, C_plus, beta_plus, beta_minus, nu, gamma_minus, gamma_plus, E_minus, E_plus, r, delta, alpha, X_minus, X_plus):
    # Calculate B_plus and B_minus based on Q_bar
    B_plus = (1.6 * Q_bar_guess * (1 + r)) / ((alpha * X_minus + 1) * (r + delta))
    B_minus = (0.4 * Q_bar_guess * (1 + r)) / ((alpha * X_plus + 1) * (r + delta))
    
    # Calculate W_plus and W_minus
    W_minus = J*(B_minus / P_plus) ** 2
    
    # Calculate P_minus and P_plus based on the derived W values
    P_plus_calc = (nu * beta_minus * C_plus * W_minus +
                   np.sqrt(W_minus * ((B_plus)**2 + nu**2 * W_minus *
                                      (beta_minus * C_plus + gamma_plus * E_plus)**2))) / (beta_minus * W_minus)
    
    #print("Q_bar_guess, B_plus, B_minus, W_plus, W_minus,P_plus_calc, P_minus_calc",Q_bar_guess, B_plus, B_minus, W_plus, W_minus,P_plus_calc, P_minus_calc)
    # Error between calculated and given prices
    error = ((P_plus_calc - P_plus) / P_plus) ** 2

    return error

def calculate_Q(W, P, C, k, beta, gamma, E, alpha, X, r, delta, L):
    """
    Calculates Q based on the given formula:
    Q = [(ln(W(P-C)k * beta - 1) + beta*P + gamma*E) * (alpha*X + 1) * (r + delta)] 
        / [(1 - delta)^L * (1 + r)]
    """
    try:
        term1 = np.log(W * (P - C) * k * beta - 1)
        term2 = beta * P + gamma * E
        numerator = (term1 + term2) * (alpha * X + 1) * (r + delta)
        denominator = (1 - delta)**L * (1 + r)
        Q = numerator / denominator
        return Q
    except Exception as e:
        print(f"Error in calculate_Q: {e}")
        return np.nan
    

def main():
    # Example values
    D_plus = 30000
    D_minus = 1#100

    #ICE
    beta_plus = 1.0
    beta_minus = 0.25404965019667486
    gamma_minus = 1.323175261441015e-08
    gamma_plus = 3.9605791486638497
    c_minus = 0.05623261991186667
    c_plus = 0.16853363453157436
    e_minus = 0.26599820413049985#THEY ARE THE SAME
    e_plus = 0.26599820413049985
    omega_minus= 0.7#0.5 + 0.2*1
    omega_plus= 1.3#0.5 + 0.8*1

    r = 0.00417

    #nu = 1
    cost = 10000

    E = 6000
    delta = 0.001
    L = 0

    W_min, W_max = 1, 5
    W_points =  1000
    n_points = 50
    P_values = [20000, 50000, 100000]
    X_plus = calc_X(beta_plus,c_plus,gamma_plus,e_plus, omega_minus)
    alpha = calculate_alpha(D_plus, D_minus, X_plus)
    print("alpha", alpha)
    beta = (beta_plus + beta_minus)/2
    c= (c_plus + c_minus)/2
    gamma= (gamma_plus + gamma_minus)/2
    e = (e_plus + e_minus)/2
    omega= (omega_plus + omega_minus)/2
    X = calc_X(beta,c,gamma,e, omega)

    # Generate W and n values
    W_vals = np.logspace(W_min, W_max, W_points)
    k_vals = np.arange(1, n_points + 1)

    # Compute Q for each P value
    Q_grids = []

    for P in P_values:
        Q_grid = np.zeros((len(W_vals), len(k_vals)))
        for i, W in enumerate(W_vals):
            for j, k in enumerate(k_vals):
                Q_grid[i, j] = calculate_Q(W, P, cost, k, beta, gamma, E, alpha, X, r, delta, L)
        Q_grids.append(Q_grid)
        #print(Q_grid)

    # Plot heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    titles = [f"P = {P}" for P in P_values]

    for i, ax in enumerate(axes):
        c = ax.contourf(k_vals, W_vals, Q_grids[i], levels=50, cmap='viridis')
        fig.colorbar(c, ax=ax, label='Q')
        ax.set_title(titles[i])
        ax.set_xlabel("n")
        ax.set_ylabel("W")
        #ax.set_yscale("log")

    plt.suptitle("Heatmaps of Q for Different P Values", fontsize=16)
    plt.show()
if __name__ == "__main__":
    main()
    
