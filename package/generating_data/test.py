import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw

def plot_terms(C, beta, B_vit, gamma_i, E_at, beta_i, tau_4, W, k_values):
    """
    Plots the three terms of the equation P = C + 1 / (k * beta) +
    Lambert function term as a function of k.

    Parameters:
        C: Constant term
        beta: Parameter beta
        B_vit: Parameter B_{v,i,t}
        gamma_i: Parameter gamma_i
        E_at: Parameter E_{a,t}
        beta_i: Parameter beta_i
        tau_4: Parameter tau_4
        W: Scaling factor in the Lambert function
        k_values: Array of k values to evaluate the terms at
    """
    # Compute terms
    term1 = C + 1 / (k_values * beta)
    argument = np.exp(k_values * (B_vit - gamma_i * E_at - beta_i * (C - tau_4)) - 1) / W
    term2 = lambertw(argument).real / (k_values * beta)
    P = term1 + term2  # Full equation
    
    # Plot terms
    plt.figure(figsize=(10, 6))
    
    # Plot each term
    plt.plot(k_values, term1, label=r"$C + \frac{1}{k\beta}$", linestyle="--", linewidth=2)
    plt.plot(k_values, term2, label=r"$\frac{\mathbb{W}(\cdots)}{k\beta}$", linestyle=":", linewidth=2)
    plt.plot(k_values, P, label=r"$P = C + \frac{1}{k\beta} + \frac{\mathbb{W}(\cdots)}{k\beta}$", linewidth=2)
    
    # Add labels and legend
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("$P$", fontsize=14)
    plt.title("Plot of Individual Terms and $P$ as a Function of $k$", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

# Example usage
C = 2
beta = 0.5
B_vit = 1.5
gamma_i = 0.8
E_at = 0.6
beta_i = 0.3
tau_4 = 1.0
W = 0.1
k_values = np.linspace(0.1, 100, 500)  # Avoid zero to prevent division by zero

plot_terms(C, beta, B_vit, gamma_i, E_at, beta_i, tau_4, W, k_values)
