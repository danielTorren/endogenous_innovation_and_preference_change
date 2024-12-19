import numpy as np
import matplotlib.pyplot as plt


def compute_quality(beta_i, gamma_i, alpha, c, e, omega, distance, delta, L):

    Q = distance**(1-alpha)*(beta_i*c/omega + gamma_i*e/omega)/(alpha*(1-delta)**L)

    return Q

    
def plots_Q_d():
        # Generate beta_i and gamma_i values between 0 and 1
    #FREE VARIABLES
    beta_i_values = np.linspace(0.2, 1 , 100)  # Avoid zero to prevent division by zero
    gamma_i_values = np.linspace(0, 3, 100)

    """
    MEDIAN VALUES: BETA = 0.5547978388069427, GAMMA  = 0.520750967535867
    """
    # Create a meshgrid
    beta_i_grid, gamma_i_grid = np.meshgrid(beta_i_values, gamma_i_values)

    # Fixed parameters
    alpha = 0.5
    c_t = 0.12  # Gas price
    e_t = 0.26599820413049985  # Gasoline emissions per kWh
    omega_a_t = 1.03  # km/kWh
    delta = 10e-4
    L = 60

    # Initialize storage for results
    Q_results = []

    # Compute Q and distance for each price

    
    distance_list = [500, 1000, 1400, 5000]

    for distance in distance_list:
        Q_values = np.zeros(beta_i_grid.shape)
        for i in range(beta_i_grid.shape[0]):
            for j in range(beta_i_grid.shape[1]):
                beta_i = beta_i_grid[i, j]
                gamma_i = gamma_i_grid[i, j]
                Q= compute_quality(beta_i, gamma_i, alpha, c_t, e_t, omega_a_t, distance, delta, L)
                Q_values[i, j] = Q
        Q_results.append(Q_values)

    # Plotting the results using subplots
    fig, axes = plt.subplots(1, len(distance_list), figsize=(15, 6))

    for i, ax in enumerate(axes.flat):
        contour1 = ax.contourf(beta_i_grid, gamma_i_grid, Q_results[i], levels=50, cmap='plasma')
        fig.colorbar(contour1, ax=ax, label='Quality (Q)')
        ax.set_xlabel('Beta_i (Price Insensitivity)')
        ax.set_ylabel('Gamma_i (Environmental Concern)')
        ax.set_title(f'Distance km = {distance_list[i]}')

    fig.savefig(f"results/quality_distnace.png", dpi=600, format="png")


    plt.tight_layout()
    plt.show()


def plot_distance():

    beta_2 = 0.9
    gamma_2 = 0.1
    alpha = 0.5
    c = 0.089951  # Gas price
    e = 0.26599820413049985  # Gasoline emissions per kWh
    nu = 0.0355  # Another constant parameter, time spent per kilometer
    omega = 1  # km/kWh
    delta_2 = 0.95#(1-delta)**L
    eta_fixed = 27
    Q = np.asarray([1.3,1.7,1.8])#make thequality range 1-2

    distance = ((alpha * Q * delta_2) / (beta_2 * omega * c + gamma_2 * omega * e + eta_fixed * nu)) ** (1 / (1 - alpha))

    print(distance)
                                                                                                             
if __name__ == "__main__":
    plots_Q_d()

    
