import numpy as np
import matplotlib.pyplot as plt

# Parameters for the utility function
Q_v_t = 3000  # Scaling factor for utility
alpha = 0.5    # Rate of diminishing marginal utility
c_ICE_t = 0.12   # Cost per km   
beta_i = 0.5  # Cost parameter
gamma_i = 0.5 # Emission parameter
e_ICE_t =0.26599820413049985    # Emission per km
omega_v_t = 1.03 # Scaling factor for cost
d_max = 5000  # Maximum distance

# Define the utility function
def utility_function(d_i_t):
    cost = beta_i * c_ICE_t + gamma_i * e_ICE_t
    utility = Q_v_t * (1 - np.exp(-alpha * d_i_t / d_max)) - d_i_t * cost / omega_v_t
    return utility

# Generate a range of distances
distances = np.linspace(0, d_max, 500)
utilities = utility_function(distances)

# Define the first-order derivative of the utility function
def derivative_utility_function(d_i_t):
    cost = beta_i * c_ICE_t + gamma_i * e_ICE_t
    dU_dd = Q_v_t * (alpha / d_max) * np.exp(-alpha * d_i_t / d_max) - cost / omega_v_t
    return dU_dd


# Adjust the parameters to ensure the optimal distance is around 1400 km
from scipy.optimize import minimize_scalar

# Define a function to find the optimal distance
def find_optimal_distance(alpha_val):
    cost = beta_i * c_ICE_t + gamma_i * e_ICE_t

    # Utility function with alpha as a parameter
    def utility_function_alpha(d_i_t):
        return Q_v_t * (1 - np.exp(-alpha_val * d_i_t / d_max)) - d_i_t * cost / omega_v_t

    # Derivative of utility function
    def derivative_utility_function_alpha(d_i_t):
        return Q_v_t * (alpha_val / d_max) * np.exp(-alpha_val * d_i_t / d_max) - cost / omega_v_t

    # Find the optimal distance by solving the derivative equals zero
    res = minimize_scalar(lambda d: -utility_function_alpha(d), bounds=(0, d_max), method='bounded')
    return res.x

# Iteratively adjust alpha to target an optimal distance around 1400 km
target_distance = 1400
alpha_range = np.linspace(0.1, 1, 100)

optimal_alpha = None
for alpha_val in alpha_range:
    optimal_dist = find_optimal_distance(alpha_val)
    if abs(optimal_dist - target_distance) < 50:  # Allowable margin
        optimal_alpha = alpha_val
        break

# Update alpha with the found value
if optimal_alpha is not None:
    alpha = optimal_alpha

# Recalculate utility and derivative with the updated alpha
utilities = utility_function(distances)
derivatives = derivative_utility_function(distances)

# Plot the updated utility function and its derivative
fig2, axs2 = plt.subplots(2, 1, figsize=(10, 12))

# Plot the utility function
axs2[0].plot(distances, utilities, label='Utility Function', color='blue')
axs2[0].axvline(d_max, color='r', linestyle='--', label='$d_{max}$')
axs2[0].set_title('Adjusted Utility Function vs Distance Driven', fontsize=14)
axs2[0].set_xlabel('Distance Driven (km)', fontsize=12)
axs2[0].set_ylabel('Utility', fontsize=12)
axs2[0].legend()
axs2[0].grid(True)

# Plot the first-order derivative
axs2[1].plot(distances, derivatives, label='First-Order Derivative', color='green', linestyle='--')
axs2[1].axvline(d_max, color='r', linestyle='--', label='$d_{max}$')
axs2[1].set_title('First-Order Derivative of Adjusted Utility Function', fontsize=14)
axs2[1].set_xlabel('Distance Driven (km)', fontsize=12)
axs2[1].set_ylabel('Derivative Value', fontsize=12)
axs2[1].legend()
axs2[1].grid(True)

# Adjust layout and show the plots
plt.tight_layout()
plt.show()

# Output the optimal alpha and corresponding distance
optimal_alpha, find_optimal_distance(optimal_alpha)



