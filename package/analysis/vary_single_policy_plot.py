import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem, t
from package.resources.utility import load_object

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem, t

def calculate_mean_and_confidence(data):
    """
    Calculate the mean and 95% confidence interval for the given data.
    """
    mean = np.mean(data, axis=1)
    confidence_interval = sem(data, axis=1) * t.ppf((1 + 0.95) / 2., data.shape[1] - 1)
    return mean, confidence_interval

def plot_mean_and_ci(data_array, policy_list, output_names, bounds):
    """
    Plot the mean values and 95% confidence intervals for the outputs.
    The plots have columns for policies and rows for the three outputs.
    """
    num_policies = len(policy_list)
    num_outputs = len(output_names)
    repetitions = data_array.shape[1]

    fig, axes = plt.subplots(num_outputs, num_policies, figsize=(15, 10), sharex=True, sharey='row')

    for i, policy in enumerate(policy_list):
        min_val, max_val = bounds[policy]
        intensities = np.linspace(min_val, max_val, repetitions)
        policy_data = data_array[i]
        output_data =  np.transpose(policy_data, (2,0,1))
        for j, output_name in enumerate(output_names):
            data = output_data[j]
            mean, ci = calculate_mean_and_confidence(data)
            
            ax = axes[j, i] if num_outputs > 1 else axes[i]
            ax.plot(intensities, mean, label='Mean', marker='o')
            ax.fill_between(intensities, mean - ci, mean + ci, color='b', alpha=0.2, label='95% CI')
            
            ax.set_title(policy if j == 0 else "")
            ax.set_xlabel("Intensity" if j == num_outputs - 1 else "")
            ax.set_ylabel(output_name if i == 0 else "")
            ax.grid(True)
    
    fig.tight_layout()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()


def main(fileName, dpi=600):
    try:
        base_params = load_object(fileName + "/Data", "base_params")
        data_array = load_object(fileName + "/Data", "data_array")
        policy_list =  load_object( fileName + "/Data", "policy_list")
        bounds = load_object(fileName + "/Data", "bounds")
        
    except FileNotFoundError:
        print("Data files not found.")
        return
    
    print(f"Shape of data_array before reshape: {data_array.shape}")
        
    output_names = ["EV Uptake", "Emissions", "Policy Distortion"]
    plot_mean_and_ci(data_array, policy_list, output_names, bounds)

    plt.show()

if __name__ == "__main__":
    main("results/grid_search_policy_intensity_00_36_01__05_01_2025")

