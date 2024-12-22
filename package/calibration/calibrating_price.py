import numpy as np
import matplotlib.pyplot as plt


def compute_quality(beta_i, gamma_i, alpha, c, e, omega, distance, delta, L):

    Q = distance**(1-alpha)*(beta_i*c/omega + gamma_i*e/omega)/(alpha*(1-delta)**L)

    return Q

    
def calc_lifetime_utility(beta_i, gamma_i, alpha, c, e, omega, distance, delta, L, prod_emmisions, r, price_new, price_old):
    
    X = beta_i*c/omega + gamma_i*e/omega
    Q = distance**(1-alpha)*(X)/(alpha*(1-delta)**L)
    u = Q*(1-delta)**(L)*distance**(alpha) - distance*(X)
    
    diving_thing = (r + np.log(1+delta)/(1-alpha))
    #print(diving_thing)
    U = u/diving_thing - beta_i*(price_new - price_old) - gamma_i*prod_emmisions

    return U, u, Q

def plot_lifetime_utility_condition():
            # Generate beta_i and gamma_i values between 0 and 1
    #FREE VARIABLES

    beta_multipier = 1
    gamma_multiplier = 1
    beta_i_values =  beta_multipier*np.linspace(0, 1 , 100)  # Avoid zero to prevent division by zero
    gamma_i_values = gamma_multiplier*np.linspace(0, 1, 100)

    """
    MEDIAN VALUES: BETA = 0.5547978388069427, GAMMA  = 0.520750967535867
    """
    # Create a meshgrid
    beta_i_grid, gamma_i_grid = np.meshgrid(beta_i_values, gamma_i_values)

    # Fixed parameters
    alpha = 0.5
    c = 0.12  # Gas price
    e = 0.26599820413049985  # Gasoline emissions per kWh
    omega = 1.03  # km/kWh
    delta = 0.0001
    L = 0#NEW CAR
    prod_emmisions = 6000
    r = 0.00417
    distance = 1400
    mu = 0.5
    cost_list = [5000, 10000, 50000, 100000]

    # Initialize storage for results
    U_results = []
    u_results = []
    Q_results = []

    for cost in cost_list:
        price_new = cost
        price_old = price_new/(1+mu)
        U_values = np.zeros(beta_i_grid.shape)
        u_values = np.zeros(beta_i_grid.shape)
        Q_values = np.zeros(beta_i_grid.shape)
        for i in range(beta_i_grid.shape[0]):
            for j in range(beta_i_grid.shape[1]):
                beta_i = beta_i_grid[i, j]
                gamma_i = gamma_i_grid[i, j]
                U, u, Q = calc_lifetime_utility(beta_i, gamma_i, alpha, c, e, omega, distance, delta, L, prod_emmisions, r, price_new, price_old)
                U_values[i, j] = U
                u_values[i, j] = u
                Q_values[i, j] = Q

        U_results.append(U_values)
        u_results.append(u_values)
        Q_results.append(Q_values)

    # Plotting the results using subplots
    fig1, axes1 = plt.subplots(1, len(cost_list), figsize=(15, 6))

    for i, ax in enumerate(axes1.flat):
        contour1 = ax.contourf(beta_i_grid, gamma_i_grid, U_results[i], levels=50, cmap='plasma')
        fig1.colorbar(contour1, ax=ax, label=f'Utility U_{i}')
        ax.set_xlabel('Beta_i (Price Insensitivity)')
        ax.set_ylabel('Gamma_i (Environmental Concern)')
        ax.set_title(f'Cost = {cost_list[i]}')

    # Plotting the results using subplots
    fig2, axes2 = plt.subplots(1, len(cost_list), figsize=(15, 6))

    for i, ax in enumerate(axes2.flat):
        contour1 = ax.contourf(beta_i_grid, gamma_i_grid, u_results[i], levels=50, cmap='plasma')
        fig2.colorbar(contour1, ax=ax, label=f'Present Utility u_{i}')
        ax.set_xlabel('Beta_i (Price Insensitivity)')
        ax.set_ylabel('Gamma_i (Environmental Concern)')
        ax.set_title(f'Cost = {cost_list[i]}')


    # Plotting the results using subplots
    fig3, axes3 = plt.subplots(1, len(cost_list), figsize=(15, 6))

    for i, ax in enumerate(axes3.flat):
        contour1 = ax.contourf(beta_i_grid, gamma_i_grid, Q_results[i], levels=50, cmap='plasma')
        fig3.colorbar(contour1, ax=ax, label=f'Quality Q_{i}')
        ax.set_xlabel('Beta_i (Price Insensitivity)')
        ax.set_ylabel('Gamma_i (Environmental Concern)')
        ax.set_title(f'Cost = {cost_list[i]}')

    #fig.savefig(f"results/quality_distnace.png", dpi=600, format="png")


    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    #plots_Q_d()
    plot_lifetime_utility_condition()
    
