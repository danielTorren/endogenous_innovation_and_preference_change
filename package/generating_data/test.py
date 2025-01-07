import numpy as np
import matplotlib.pyplot as plt

# Calculate u* based on the provided formulas
def u_star(L, Q, delta, beta, c, omega, gamma, e, d_max, alpha):
    term1 = Q * (1 - delta) ** L
    term2 = ((beta * c / omega) + (gamma * e / omega)) * d_max / alpha
    term3 = (1 / (1 - delta) ** L) * np.log(((beta * c / omega) + (gamma * e / omega)) * d_max / (alpha * Q * (1 - delta) ** L)) - 1
    return term1 + term2 * term3

# Calculate lifetime utility B
def lifetime_utility(L, Q, delta, beta, c, omega, gamma, e, d_max, alpha, r):
    u_L = u_star(L, Q, delta, beta, c, omega, gamma, e, d_max, alpha)
    g = (u_star(L + 1, Q, delta, beta, c, omega, gamma, e, d_max, alpha) / u_L) - 1
    return (u_L * (1 + r)) / (r - g)

# Define the profit function
def profit(P, B, U, beta, gamma, E, C, I, k):
    denominator = (B / (beta * P + gamma * E) + U) ** k
    return (P - C) * I * (B / (beta * P + gamma * E)) / denominator

# Define the analytical derivative of the profit function
def profit_derivative(P, B, U, beta, gamma, E, C, I, k):
    term1 = I * B / (beta * P + gamma * E) / (B / (beta * P + gamma * E) + U) ** k
    term2 = -(P - C) * I * B * beta / (beta * P + gamma * E) ** 2 / (B / (beta * P + gamma * E) + U) ** k
    term3 = (P - C) * I * beta * k * B ** 2 / (beta * P + gamma * E) ** 3 / (B / (beta * P + gamma * E) + U) ** (k + 1)
    return term1 + term2 + term3

def optimal_price(U, beta, gamma, E, C, k, B):

    # Coefficients for the quadratic formula
    a = U * beta * (B - 1)
    b = (1 - k) * B**2 + (U * (gamma * E - C * beta) - 1) * B - U * gamma * E
    c = C * (k - 1) * B**2 + U * gamma * E * C * B + gamma * E / beta

    # Discriminant
    discriminant = b**2 - 4 * a * c

    # Check for valid solutions
    if discriminant < 0:
        raise ValueError("The quadratic equation has no real solutions.")

    # Calculate the two solutions for P*
    P1 = (-b + np.sqrt(discriminant)) / (2 * a)
    P2 = (-b - np.sqrt(discriminant)) / (2 * a)

    return P1, P2

if __name__ == "__main__":
    # Input constants
    U = 1000000
    beta = 0.5
    gamma = 0.5
    E = 6000
    C = 10000
    k = 10
    I = 10000
    Q = 100
    delta = 0.001
    alpha = 0.5
    d_max = 10000
    L = 10
    r = 0.00417
    c = 0.12  # Gas price
    e = 0.26599820413049985  # Gasoline emissions per kWh
    omega = 1.03  # km/kWh
    
    # Define the price range
    P = np.linspace(10000, 100000, 500)

    # Calculate B
    B = lifetime_utility(L, Q, delta, beta, c, omega, gamma, e, d_max, alpha, r)

    # Calculate profit and its numerical derivative
    profits = profit(P, B, U, beta, gamma, E, C, I, k)
    numerical_derivative = np.gradient(profits, P)

    # Calculate the analytical derivative
    analytical_derivative = profit_derivative(P, B, U, beta, gamma, E, C, I, k)

    # Plot the results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    # Profit curve
    ax1.plot(P, profits, label="Profit Function")
    ax1.set_title("Profit Function and Its Derivatives")
    ax1.set_xlabel("Price (P)")
    ax1.set_ylabel("Profit")
    ax1.legend()
    ax1.grid()

    # Derivatives comparison
    ax2.plot(P, numerical_derivative, label="Numerical Derivative", linestyle="dashed")
    ax2.plot(P, analytical_derivative, label="Analytical Derivative")
    ax2.set_xlabel("Price (P)")
    ax2.set_ylabel("Derivative of Profit")
    ax2.legend()
    ax2.grid()




    # Calculate optimal price
    B = lifetime_utility(L, Q, delta, beta, c, omega, gamma, e, d_max, alpha, r)
    P1, P2 = optimal_price(U, beta, gamma, E, C, k, B)

    # Display the results
    print(f"The two possible optimal prices are: P1 = {P1:.2f}, P2 = {P2:.2f}")
    plt.tight_layout()
    plt.show()
