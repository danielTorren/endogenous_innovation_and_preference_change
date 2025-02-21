import numpy as np
from scipy.optimize import minimize
from scipy.stats import lognorm
import matplotlib.pyplot as plt

# Observed data
quintiles = [0.2, 0.4, 0.6, 0.8, 1.0]
mean_incomes = [16981, 47103, 80693, 127666, 295369]
shares = [0.0299, 0.0830, 0.1421, 0.2248, 0.5202]

# Better initial guess for mu and sigma
log_incomes = np.log(mean_incomes)
mu_guess = np.mean(log_incomes)
sigma_guess = np.std(log_incomes)
initial_guess = [mu_guess, sigma_guess]
print(f"Initial guess: mu = {mu_guess:.4f}, sigma = {sigma_guess:.4f}")

# Objective function to minimize
def objective(params):
    mu, sigma = params
    error = 0
    
    # Calculate theoretical mean incomes and shares
    for i in range(len(quintiles)):
        lower = 0 if i == 0 else quintiles[i-1]
        upper = quintiles[i]
        
        # Theoretical mean income for the quintile
        theoretical_mean = lognorm.mean(s=sigma, scale=np.exp(mu)) * (
            lognorm.cdf(upper, s=sigma, scale=np.exp(mu)) - 
            lognorm.cdf(lower, s=sigma, scale=np.exp(mu))
        )
        
        # Theoretical share for the quintile
        theoretical_share = lognorm.cdf(upper, s=sigma, scale=np.exp(mu)) - lognorm.cdf(lower, s=sigma, scale=np.exp(mu))
        
        # Add weighted errors (prioritize matching shares)
        error += 10 * (theoretical_share - shares[i])**2  # Weight shares more heavily
        error += (theoretical_mean - mean_incomes[i])**2  # Mean income error
    
    return error

# Bounds for mu and sigma (to avoid invalid values)
bounds = [(None, None), (1e-3, None)]  # sigma must be positive

# Minimize the objective function using L-BFGS-B
result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)

# Extract the optimized parameters
mu_opt, sigma_opt = result.x
print(f"Optimized mu: {mu_opt:.4f}, sigma: {sigma_opt:.4f}")

# Generate a population of 5000 agents with incomes based on the log-normal distribution
np.random.seed(42)  # For reproducibility
population_size = 5000
incomes = lognorm.rvs(s=sigma_opt, scale=np.exp(mu_opt), size=population_size)

# Sort incomes for analysis
incomes_sorted = np.sort(incomes)

# Calculate quintiles and top 5%
quintile_indices = [int(population_size * q) for q in quintiles]
top_5_index = int(population_size * 0.95)

# Calculate mean income and share for each quintile and top 5%
quintile_means = []
quintile_shares = []
for i in range(len(quintile_indices)):
    lower = 0 if i == 0 else quintile_indices[i-1]
    upper = quintile_indices[i]
    quintile_income = incomes_sorted[lower:upper]
    quintile_mean = np.mean(quintile_income)
    quintile_share = np.sum(quintile_income) / np.sum(incomes_sorted)
    quintile_means.append(quintile_mean)
    quintile_shares.append(quintile_share)

# Top 5%
top_5_income = incomes_sorted[top_5_index:]
top_5_mean = np.mean(top_5_income)
top_5_share = np.sum(top_5_income) / np.sum(incomes_sorted)

# Report results
print("\nMean income and share for each quintile:")
for i in range(len(quintile_means)):
    print(f"Quintile {i+1}: Mean Income = ${quintile_means[i]:.2f}, Share = {quintile_shares[i]*100:.2f}%")

print(f"\nTop 5%: Mean Income = ${top_5_mean:.2f}, Share = {top_5_share*100:.2f}%")

# Calculate GINI coefficient
def gini_coefficient(incomes):
    sorted_incomes = np.sort(incomes)
    n = len(sorted_incomes)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * sorted_incomes)) / (n * np.sum(sorted_incomes))

gini = gini_coefficient(incomes_sorted)
print(f"\nGINI Coefficient: {gini:.4f}")

# Plot cumulative income distribution and Lorenz curve
def lorenz_curve(incomes):
    sorted_incomes = np.sort(incomes)
    cumulative_income = np.cumsum(sorted_incomes)
    cumulative_income_share = cumulative_income / cumulative_income[-1]
    cumulative_population_share = np.arange(1, len(sorted_incomes) + 1) / len(sorted_incomes)
    return cumulative_population_share, cumulative_income_share

# Lorenz curve
lorenz_x, lorenz_y = lorenz_curve(incomes_sorted)

# Perfect equality line
perfect_equality_x = np.linspace(0, 1, 100)
perfect_equality_y = perfect_equality_x

# Plot
plt.figure(figsize=(12, 6))

# Cumulative income distribution
plt.subplot(1, 2, 1)
plt.plot(np.arange(1, population_size + 1) / population_size, np.cumsum(incomes_sorted) / np.sum(incomes_sorted), label="Cumulative Income Distribution")
plt.xlabel("Population Share")
plt.ylabel("Income Share")
plt.title("Cumulative Income Distribution")
plt.legend()

# Lorenz curve
plt.subplot(1, 2, 2)
plt.plot(lorenz_x, lorenz_y, label="Lorenz Curve")
plt.plot(perfect_equality_x, perfect_equality_y, label="Perfect Equality", linestyle="--")
plt.xlabel("Cumulative Population Share")
plt.ylabel("Cumulative Income Share")
plt.title("Lorenz Curve")
plt.legend()

plt.tight_layout()


optimize_mu = 11.225243392518447
optimised_sigma = 0.927
np.random.seed(42)  # For reproducibility
population_size = 500
incomes = lognorm.rvs(s=sigma_opt, scale=np.exp(mu_opt), size=population_size)
#print(incomes)
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(incomes, bins=30, alpha=0.5, label='Income Vec')

plt.show()