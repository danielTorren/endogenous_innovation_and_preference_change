import numpy as np
import matplotlib.pyplot as plt

def compute_quality(P_star, beta_i, gamma_i, E, alpha, r, delta, c_z_t, e_z_t, omega_a_t, eta, nu_z_i_t):
    """
    Computes the quality Q and distance based on the given parameters and updated definition of D.
    Parameters:
        P_star (float): Price of the car.
        beta_i (float or array): Price insensitivity parameter.
        gamma_i (float or array): Environmental concern parameter.
        E (float): Production emissions (kgCO2).
        alpha (float): Elasticity parameter.
        r (float): Interest rate.
        delta (float): Depreciation rate.
        c_z_t (float): Cost (price) at time t.
        e_z_t (float): Emissions at time t.
        omega_a_t (float): Weight or scaling factor at time t.
        eta (float): Constant parameter.
        nu_z_i_t (float): Another constant parameter.
    Returns:
        Q (array): The computed quality Q.
        distance (array): The computed distance driven.
    """
    # Compute D using the new definition
    D = (beta_i * c_z_t / omega_a_t) + (gamma_i * e_z_t / omega_a_t) + eta * nu_z_i_t

    # Compute the numerator and denominator for Q
    numerator = (P_star * beta_i + gamma_i * E) * (r + (1 - delta) / (1 - alpha))
    denominator = ((alpha / D) ** (1 / (1 - alpha))) * D * ((1 / alpha) - 1)

    # Compute Q
    Q = (numerator / denominator) ** (1 - alpha)

    # Compute distance
    distance = (alpha * Q / D) ** (1 / (1 - alpha))

    return Q, distance

if __name__ == "__main__":
    # Generate beta_i and gamma_i values between 0 and 1
    #FREE VARIABLES
    beta_i_values = np.linspace(0.01, 1, 100)  # Avoid zero to prevent division by zero
    gamma_i_values = np.linspace(0.01, 1, 100)
    eta = 0.5  # Constant

    #4 degrees of freedom: Q, beta, gamma and eta
    #Paramterise: price of the car, distance driven
    #leaves us with 2 parameters that we can set. 

    # Create a meshgrid
    beta_i_grid, gamma_i_grid = np.meshgrid(beta_i_values, gamma_i_values)

    # Fixed parameters
    prices = [5000, 100000]  # Two prices for comparison
    E = 9000
    alpha = 0.3
    r = 0.02
    delta = 0.0005
    c_z_t = 0.089951  # Gas price
    e_z_t = 0.26599820413049985  # Gasoline emissions per kWh
    nu_z_i_t = 0.0355  # Another constant parameter, time spent per kilometer
    omega_a_t = 1  # km/kWh

    

    # Initialize storage for results
    Q_results = []
    distance_results = []

    # Compute Q and distance for each price
    for P_star in prices:
        Q_values = np.zeros(beta_i_grid.shape)
        distance_values = np.zeros(beta_i_grid.shape)

        for i in range(beta_i_grid.shape[0]):
            for j in range(beta_i_grid.shape[1]):
                beta_i = beta_i_grid[i, j]
                gamma_i = gamma_i_grid[i, j]
                Q, distance = compute_quality(P_star, beta_i, gamma_i, E, alpha, r, delta,
                                              c_z_t, e_z_t, omega_a_t, eta, nu_z_i_t)
                Q_values[i, j] = Q
                distance_values[i, j] = distance

        Q_results.append(Q_values)
        distance_results.append(distance_values)

    # Plotting the results using subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"eta = {eta}")
    for row, P_star in enumerate(prices):
        # Quality Q Plot
        ax1 = axes[row, 0]
        contour1 = ax1.contourf(beta_i_grid, gamma_i_grid, Q_results[row], levels=50, cmap='plasma')
        fig.colorbar(contour1, ax=ax1, label='Quality (Q)')
        ax1.set_xlabel('Beta_i (Price Insensitivity)')
        ax1.set_ylabel('Gamma_i (Environmental Concern)')
        ax1.set_title(f'Quality (Q) (P* = ${P_star})')

        # Distance Driven Plot
        ax2 = axes[row, 1]
        contour2 = ax2.contourf(beta_i_grid, gamma_i_grid, distance_results[row], levels=50, cmap='viridis')
        fig.colorbar(contour2, ax=ax2,  label='Distance Driven (km)')
        ax2.set_xlabel('Beta_i (Price Insensitivity)')
        ax2.set_ylabel('Gamma_i (Environmental Concern)')
        ax2.set_title(f'Distance Driven (P* = ${P_star})')

    fig.savefig(f"results/quality_distnace.png", dpi=600, format="png")


    plt.tight_layout()
    plt.show()

    
