import numpy as np
import matplotlib.pyplot as plt

    #(P_star, beta_i, gamma_i, E, alpha, r, delta, c_t, e_t, omega_a_t, eta, nu_i_t)
def compute_quality(P_star, beta_i, gamma_i, E, alpha, r, delta, c_t, e_t, omega_a_t, eta, nu_i_t):
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
        c_t (float): Cost (price) at time t.
        e_t (float): Emissions at time t.
        omega_a_t (float): Weight or scaling factor at time t.
        eta (float): Constant parameter.
        nu_i_t (float): Another constant parameter.
    Returns:
        Q (array): The computed quality Q.
        distance (array): The computed distance driven.
    """
    # Compute D using the new definition
    D = (beta_i * c_t / omega_a_t) + (gamma_i * e_t / omega_a_t) + eta * nu_i_t

    # Compute the numerator and denominator for Q
    #numerator = (P_star * beta_i + gamma_i * E) * (r + (1 - delta) / (1 - alpha))
    #denominator = ((alpha / D) ** (1 / (1 - alpha))) * D * ((1 / alpha) - 1)

    # Compute Q
    #Q = (numerator / denominator) ** (1 - alpha)

    Q = 2.75
    # Compute distance
    distance = (alpha * Q / D) ** (1 / (1 - alpha))

    return Q, distance

def calc_eta_Q(beta_mean, gamma_mean, alpha, c_t, e_t, omega_a_t, d_mean, d_max,nu_i_t, decay):

    eta = d_mean**(1-alpha)*((beta_mean * c_t / omega_a_t) + (gamma_mean * e_t / omega_a_t))/(nu_i_t*(decay*d_max**(1-alpha)-d_mean**(1-alpha)))
    Q = (d_max**(1-alpha)/alpha)*eta*nu_i_t
    return eta, Q

def plots_Q_nu():
    # Fixed parameters
    willingnessToPay_eurostonne = 199#euros/tonne, https://link.springer.com/article/10.1007/s10640-020-00411-6
    exchange_rate = 1.1383 #https://www.exchange-rates.org/exchange-rate-history/eur-usd-2020
    willingnessToPay_dollarKillongram = exchange_rate*willingnessToPay_eurostonne/1000
    #0.22652170000000002, this is the mean value and should represnt the median beta and median gamma
    beta_mean = 0.9
    gamma_mean = 0.1
    alpha = 0.95

    c_t = 0.089951  # Gas price
    e_t = 0.26599820413049985  # Gasoline emissions per kWh
    nu_i_t = 0.0355  # Another constant parameter, time spent per kilometer
    omega_a_t = 1  # km/kWh
    d_mean = 1400 
    d_max = 5000

    delta = 0.0005
    L = 120
    decay = 0.9#(1-delta)**L

    eta, Q = calc_eta_Q(beta_mean, gamma_mean, alpha, c_t, e_t, omega_a_t, d_mean, d_max,nu_i_t, decay )

    print("eta, Q",eta, Q)

    price_commponent = (beta_mean * c_t / omega_a_t) 
    emissions_component = (gamma_mean * e_t / omega_a_t)
    time_component = eta*nu_i_t

    print("commponents: ", price_commponent, emissions_component, time_component)
    


def plots_Q_d():
        # Generate beta_i and gamma_i values between 0 and 1
    #FREE VARIABLES
    beta_i_values = np.linspace(0, 1, 100)  # Avoid zero to prevent division by zero
    gamma_i_values = np.linspace(0, 0.4, 100)
    eta = 27#1  # Constant

    #4 degrees of freedom: Q, beta, gamma and eta
    #Paramterise: price of the car, distance driven
    #leaves us with 2 parameters that we can set. 

    # Create a meshgrid
    beta_i_grid, gamma_i_grid = np.meshgrid(beta_i_values, gamma_i_values)

    # Fixed parameters
    prices = [5000, 100000]  # Two prices for comparison
    E = 9000
    alpha = 0.95
    r = 0.02
    delta = 0.005
    c_t = 0.089951  # Gas price
    e_t = 0.26599820413049985  # Gasoline emissions per kWh
    nu_i_t = 0.0355  # Another constant parameter, time spent per kilometer
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
                                              c_t, e_t, omega_a_t, eta, nu_i_t)
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


def plot_distance():

    beta_2 = 0.9
    gamma_2 = 0.1
    alpha = 0.95
    c = 0.089951  # Gas price
    e= 0.26599820413049985  # Gasoline emissions per kWh
    nu = 0.0355  # Another constant parameter, time spent per kilometer
    omega = 1  # km/kWh
    delta_2 = 0.95#(1-delta)**L
    eta_fixed = 27
    Q = np.asarray([1.3,1.7,1.8])#make thequality range 1-2

    distance = ((alpha * Q * delta_2) / (beta_2 * omega * c + gamma_2 * omega * e + eta_fixed * nu)) ** (1 / (1 - alpha))

    print(distance)
                                                                                                             
if __name__ == "__main__":
    plot_distance()
    #plots_Q_nu()
    #plots_Q_d()

    
