from struct import pack
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import poisson

# Input data for new cars (distance distributions)
bin_centers_new = np.array([
    335.2791667,
    1005.8375,
    1676.331279,
    2346.892946,
    3017.452946,
    5028.99
])
bin_counts_new = np.array([145, 231, 207, 62, 31, 29])

# Percentages for 8-year-old cars
percentages_10plusyr = np.array([
    0.413342054,
    0.294637018,
    0.184434271,
    0.038587312,
    0.030739045,
    0.038260301
])


# Assume a total count for the older car data
total_count_10plusyr = 3058  # Adjust this as needed
bin_counts_10plusyr = percentages_10plusyr * total_count_10plusyr

# Define the Poisson fitting function
def fit_function(k, lamb):
    """Poisson probability mass function scaled by total count."""
    return poisson.pmf(k, lamb) * np.sum(bin_counts_new)

# Fit Poisson to new car data
x_data_new = np.arange(len(bin_counts_new))
parameters_new, _ = curve_fit(fit_function, x_data_new, bin_counts_new)
lambda_new = parameters_new[0]

# Fit Poisson to 8-year-old car data
x_data_10plusyr = np.arange(len(bin_counts_10plusyr))
parameters_10plusyr, _ = curve_fit(fit_function, x_data_10plusyr, bin_counts_10plusyr)
lambda_10plusyr = parameters_10plusyr[0]

print(f"Fitted lambda (new cars): {lambda_new}")
print(f"Fitted lambda (10-plus-year-old cars): {lambda_10plusyr}")

# Calculate the monthly depreciation rate
months_old = 12*15#Assume 15 years
r = 1 - (lambda_10plusyr / lambda_new) ** (1 / months_old)
print(f"Monthly depreciation rate: {r:.6f}")

# Simulate the depreciation over months_old
months = np.arange(months_old + 1)  # 0 to months_old
lambda_over_time = lambda_new * (1 - r) ** months

# Generate simulated counts for 10-plus-year-old cars using the adjusted lambda
simulated_counts_10plusyr = poisson.pmf(x_data_new, lambda_over_time[-1]) * np.sum(bin_counts_new)

# Plot the depreciation of lambda over time
plt.plot(months, lambda_over_time, label="Lambda Over Time (Depreciation)", color='blue')
plt.xlabel("Months")
plt.ylabel("Lambda (Mean Distance)")
plt.title("Depreciation of Distance Driven Over Time")
plt.legend()
plt.show()

# Compare the simulated and actual 8-year-old car distributions
plt.bar(bin_centers_new, bin_counts_10plusyr, width=300, alpha=0.6, label="Actual 10 plus year Data", color='gray')
plt.bar(bin_centers_new, simulated_counts_10plusyr, width=300, alpha=0.6, label="Simulated 10 plus year Data", color='blue')
plt.xlabel("Distance Driven")
plt.ylabel("Counts")
plt.legend()
plt.title("Simulated vs Actual 8-Year-Old Car Distance Distribution")
plt.show()
