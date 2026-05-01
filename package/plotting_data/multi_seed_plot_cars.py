import numpy as np
import matplotlib.pyplot as plt
from package.resources.utility import load_object
from matplotlib.lines import Line2D
import os

# Import the real-world vehicle data
from package.plotting_data.single_experiment_plot import info_real_cars


def extract_car_attributes_from_objects(cars_list, base_params):
    """
    Extract price, range, efficiency, and type from car objects.
    Removes duplicates by tracking unique_id.
    """
    fuel_tank_L = base_params["parameters_ICE"]["fuel_tank"]
    
    seen_ids = set()
    records = []
    
    for car in cars_list:
        uid = getattr(car, "unique_id", id(car))
        if uid in seen_ids:
            continue
        seen_ids.add(uid)
        
        attrs = car.attributes_fitness
        price = getattr(car, "price", car.ProdCost_t * 1.0)
        
        if car.transportType == 3:  # EV
            efficiency = attrs[1]  # km/kWh
            battery_kwh = attrs[3]  # kWh
            range_km = efficiency * battery_kwh
            vtype = "EV"
        else:  # ICE
            efficiency = attrs[1]  # km/L
            range_km = efficiency * fuel_tank_L
            vtype = "ICE"
        
        records.append({
            "price": price,
            "range_km": range_km,
            "efficiency": efficiency,
            "vtype": vtype
        })
    
    return records


def plot_multi_seed_2d_scatter(fileName, outputs, base_params, dpi=300):
    """
    Simple 2D scatter plot: Driving Range (km) vs Price (USD)
    Flattens all cars from all seeds and plots them together.
    """
    
    # Load real-world vehicle data
    MILES_PER_KM, KM_PER_MILE, MPGE_TO_KM_KWH, MPG_TO_KM_L, REAL_WORLD_VEHICLES = info_real_cars()
    
    # Get cars_on_sale (list of lists: one list per seed)
    cars_on_sale_per_seed = outputs.get("cars_on_sale", [])
    
    if not cars_on_sale_per_seed:
        print("No cars_on_sale found in outputs.")
        return None
    
    print(f"Processing {len(cars_on_sale_per_seed)} seeds...")
    
    # Flatten all cars from all seeds and extract attributes
    all_cars = []
    for seed_idx, seed_cars in enumerate(cars_on_sale_per_seed):
        car_attrs = extract_car_attributes_from_objects(seed_cars, base_params)
        all_cars.extend(car_attrs)
        print(f"  Seed {seed_idx}: {len(seed_cars)} raw -> {len(car_attrs)} unique cars")
    
    # Split by type
    sim_ev = [c for c in all_cars if c["vtype"] == "EV"]
    sim_ice = [c for c in all_cars if c["vtype"] == "ICE"]
    
    # Real-world vehicles
    real_ev = [v for v in REAL_WORLD_VEHICLES if v["type"] == "EV"]
    real_ice = [v for v in REAL_WORLD_VEHICLES if v["type"] in ["ICE", "PHEV"]]
    
    print(f"\nTotal unique cars across all seeds: {len(all_cars)}")
    print(f"  EVs: {len(sim_ev)}, ICEs: {len(sim_ice)}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors
    EV_COLOR = "#2E8B57"  # SeaGreen
    ICE_COLOR = "#4169E1"  # RoyalBlue
    
    # Plot simulated EVs (green circles)
    if sim_ev:
        ev_ranges = np.array([c["range_km"] for c in sim_ev])
        ev_prices = np.array([c["price"] for c in sim_ev])
        ax.scatter(ev_ranges, ev_prices, marker="o", s=30, alpha=0.4, 
                  c=EV_COLOR, edgecolors="darkgreen", linewidths=0.3,
                  label=f"Simulated EV (n={len(sim_ev)})", zorder=2)
    
    # Plot simulated ICEs (blue squares)
    if sim_ice:
        ice_ranges = np.array([c["range_km"] for c in sim_ice])
        ice_prices = np.array([c["price"] for c in sim_ice])
        ax.scatter(ice_ranges, ice_prices, marker="s", s=30, alpha=0.4,
                  c=ICE_COLOR, edgecolors="darkblue", linewidths=0.3,
                  label=f"Simulated ICE (n={len(sim_ice)})", zorder=2)
    
    # Plot real-world EVs (green diamonds)
    for v in real_ev:
        ax.scatter(v["range_km"], v["price_usd"], marker="D", s=200, 
                  c=EV_COLOR, edgecolors="darkgreen", linewidths=1.5, zorder=5)
        ax.annotate(v["label"], xy=(v["range_km"], v["price_usd"]),
                   xytext=(5, 5), textcoords="offset points",
                   fontsize=8, fontweight="bold", color=EV_COLOR,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                            edgecolor=EV_COLOR, alpha=0.8))
    
    # Plot real-world ICE (blue triangles)
    for v in real_ice:
        ax.scatter(v["range_km"], v["price_usd"], marker="^", s=200, 
                  c=ICE_COLOR, edgecolors="darkblue", linewidths=1.5, zorder=5)
        ax.annotate(v["label"], xy=(v["range_km"], v["price_usd"]),
                   xytext=(5, 5), textcoords="offset points",
                   fontsize=8, fontweight="bold", color=ICE_COLOR,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                            edgecolor=ICE_COLOR, alpha=0.8))
    
    # Legend
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=EV_COLOR,
               markeredgecolor="darkgreen", markersize=8, label=f"Simulated EV ({len(sim_ev)})"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=ICE_COLOR,
               markeredgecolor="darkblue", markersize=8, label=f"Simulated ICE ({len(sim_ice)})"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=EV_COLOR,
               markeredgecolor="darkgreen", markersize=8, label="Real-world EV"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=ICE_COLOR,
               markeredgecolor="darkblue", markersize=8, label="Real-world ICE"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10, framealpha=0.9)
    
    ax.set_xlabel("Driving Range (km)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Price (USD)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, linestyle="--")
    
    # Add some padding
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min, y_max + y_range * 0.05)
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(fileName, "Plots")
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(f"{save_path}/multi_seed_2d_scatter.png", dpi=dpi, bbox_inches="tight")
    print(f"\nSaved to {save_path}/multi_seed_2d_scatter.png")
    
    return fig, ax


def main(fileName):
    base_params = load_object(fileName + "/Data", "base_params")
    outputs = load_object(fileName + "/Data", "outputs")
    
    print(f"Loaded data from {fileName}")
    print(f"Keys in outputs: {list(outputs.keys())}")
    
    if "cars_on_sale" in outputs:
        cars_on_sale = outputs["cars_on_sale"]
        print(f"\nNumber of seeds: {len(cars_on_sale)}")
        print(f"Raw cars per seed: {[len(seed) for seed in cars_on_sale[:5]]}...")
        
        # Plot the 2D scatter
        plot_multi_seed_2d_scatter(fileName, outputs, base_params, dpi=200)
    else:
        print("\n⚠ No 'cars_on_sale' found in outputs.")
    
    plt.show()


if __name__ == "__main__":
    main(fileName="results/multi_seed_cars_15_53_16__30_04_2026")