import numpy as np
import matplotlib.pyplot as plt
from package.resources.utility import load_object
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
import os

# Import the real-world vehicle data
from package.plotting_data.single_experiment_plot import info_real_cars


def extract_car_attributes_from_objects(cars_list, base_params):
    """
    Extract price, range, efficiency, and type from car objects.
    Removes duplicates by tracking unique_id.

    Units: the model does ALL fuel accounting in kWh, not litres -- gas prices
    and gasoline emissions are converted to a per-kWh basis in
    calibration_data_inputs.py at 33.41 kWh per US gallon (AFDC gallon
    equivalent). So ICE efficiency is km/kWh and fuel_tank is kWh, exactly like
    the EV's km/kWh and battery kWh. Range is efficiency * capacity either way.
    """
    fuel_tank_kWh = base_params["parameters_ICE"]["fuel_tank"]

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
            efficiency = attrs[1]  # km/kWh (gasoline on a kWh basis, not km/L)
            range_km = efficiency * fuel_tank_kWh
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


def _density_levels(kde, sample_xy, mass_fractions):
    """
    Convert "contour enclosing X% of the simulated cars" into KDE density levels.

    Evaluates the KDE at the sample points themselves and takes the (1-X)
    quantile of those densities, so the level enclosing e.g. 50% of the mass is
    the density value that 50% of the cars sit above.
    """
    dens = kde(sample_xy)
    # Largest mass fraction -> lowest density level, so sorting the fractions
    # descending gives strictly increasing levels to pair the labels against
    pairs = [(float(np.quantile(dens, 1.0 - f)), f)
             for f in sorted(mass_fractions, reverse=True)]

    # contour/contourf require strictly increasing levels
    kept = []
    for level, frac in pairs:
        if not kept or level > kept[-1][0]:
            kept.append((level, frac))
    return kept


def plot_multi_seed_2d_contour(fileName, outputs, base_params, dpi=300,
                               mass_fractions=(0.95, 0.8, 0.5), grid_n=200):
    """
    Real-world vehicles on top of a 2D density contour map of the simulated cars.

    The simulated cloud (all unique cars on sale, pooled across seeds) is turned
    into a Gaussian KDE over (Driving Range, Price) and drawn as nested contours
    enclosing 95 / 80 / 50 % of the simulated cars. EV and ICE get their own
    density, so the two populations can be compared against their real-world
    counterparts separately.

    Note this is a SNAPSHOT of the final time step, not a time series -- see
    generate_multi_seed_cars in package/resources/run.py.
    """

    MILES_PER_KM, KM_PER_MILE, MPGE_TO_KM_KWH, MPG_TO_KM_L, REAL_WORLD_VEHICLES = info_real_cars()

    cars_on_sale_per_seed = outputs.get("cars_on_sale", [])
    if not cars_on_sale_per_seed:
        print("No cars_on_sale found in outputs.")
        return None

    all_cars = []
    for seed_cars in cars_on_sale_per_seed:
        all_cars.extend(extract_car_attributes_from_objects(seed_cars, base_params))

    sim_ev = [c for c in all_cars if c["vtype"] == "EV"]
    sim_ice = [c for c in all_cars if c["vtype"] == "ICE"]

    real_ev = [v for v in REAL_WORLD_VEHICLES if v["type"] == "EV"]
    real_ice = [v for v in REAL_WORLD_VEHICLES if v["type"] in ["ICE", "PHEV"]]

    EV_COLOR = "#2E8B57"   # SeaGreen  - matches the scatter plot
    ICE_COLOR = "#4169E1"  # RoyalBlue - matches the scatter plot

    fig, ax = plt.subplots(figsize=(12, 8))

    # Grid spans simulated and real data so nothing is clipped out of frame
    all_x = [c["range_km"] for c in all_cars] + [v["range_km"] for v in REAL_WORLD_VEHICLES]
    all_y = [c["price"] for c in all_cars] + [v["price_usd"] for v in REAL_WORLD_VEHICLES]
    x_pad = 0.08 * (max(all_x) - min(all_x))
    y_pad = 0.08 * (max(all_y) - min(all_y))
    xx, yy = np.meshgrid(
        np.linspace(min(all_x) - x_pad, max(all_x) + x_pad, grid_n),
        np.linspace(min(all_y) - y_pad, max(all_y) + y_pad, grid_n),
    )
    grid_xy = np.vstack([xx.ravel(), yy.ravel()])

    for sim, colour, cmap, name in (
        (sim_ice, ICE_COLOR, "Blues", "ICE"),
        (sim_ev, EV_COLOR, "Greens", "EV"),
    ):
        if len(sim) < 5:
            print(f"Skipping {name} density: only {len(sim)} cars.")
            continue

        sample_xy = np.vstack([
            np.array([c["range_km"] for c in sim]),
            np.array([c["price"] for c in sim]),
        ])
        try:
            kde = gaussian_kde(sample_xy)
        except np.linalg.LinAlgError:
            print(f"Skipping {name} density: degenerate covariance (no spread in the data).")
            continue

        zz = kde(grid_xy).reshape(xx.shape)
        level_pairs = _density_levels(kde, sample_xy, mass_fractions)
        levels = [lv for lv, _ in level_pairs]
        if len(levels) < 2 or zz.max() <= levels[-1]:
            print(f"Skipping {name} density: could not form distinct contour levels.")
            continue

        # Filled bands give the "where the mass is" read, lines give the boundary
        ax.contourf(xx, yy, zz, levels=levels + [zz.max()], cmap=cmap,
                    alpha=0.35, zorder=1)
        cs = ax.contour(xx, yy, zz, levels=levels, colors=colour,
                        linewidths=1.5, zorder=2)
        ax.clabel(cs, inline=True, fontsize=8,
                  fmt={lv: f"{int(round(f * 100))}%" for lv, f in level_pairs})

        print(f"{name}: {len(sim)} cars, contour levels {levels}")

    # Real-world vehicles on top, with a white ring so they read over the fills.
    # Labels are fanned out vertically within each type, cheapest first, so the
    # tightly-clustered real ICE models don't overprint each other.
    for group, marker, size, colour in (
        (real_ice, "^", 180, ICE_COLOR),
        (real_ev, "D", 160, EV_COLOR),
    ):
        for rank, v in enumerate(sorted(group, key=lambda c: c["price_usd"])):
            ax.scatter(v["range_km"], v["price_usd"], marker=marker, s=size,
                       c=colour, edgecolors="white", linewidths=1.8, zorder=5)
            ax.annotate(
                v["label"], xy=(v["range_km"], v["price_usd"]),
                xytext=(10, 8 + 13 * (rank % len(group))), textcoords="offset points",
                fontsize=8, fontweight="bold", color="#222222",
                arrowprops=dict(arrowstyle="-", color=colour, linewidth=0.7,
                                shrinkA=0, shrinkB=3),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=colour, alpha=0.85), zorder=6,
            )

    legend_handles = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor=EV_COLOR,
               markeredgecolor="white", markersize=9, label="Real-world EV"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=ICE_COLOR,
               markeredgecolor="white", markersize=10, label="Real-world ICE"),
        Line2D([0], [0], color=EV_COLOR, linewidth=1.5,
               label=f"Simulated EV density (n={len(sim_ev)})"),
        Line2D([0], [0], color=ICE_COLOR, linewidth=1.5,
               label=f"Simulated ICE density (n={len(sim_ice)})"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10, framealpha=0.9)

    ax.set_xlabel("Driving Range (km)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Price (USD)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Simulated cars on sale (contours enclose 50 / 80 / 95% of cars, pooled over "
        f"{len(cars_on_sale_per_seed)} seeds) vs real-world vehicles",
        fontsize=11
    )
    ax.grid(alpha=0.25, linestyle="--", zorder=0)

    plt.tight_layout()

    save_path = os.path.join(fileName, "Plots")
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(f"{save_path}/multi_seed_2d_contour.png", dpi=dpi, bbox_inches="tight")
    print(f"Saved to {save_path}/multi_seed_2d_contour.png")

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

        # Separate figure: real-world data over a density contour map of the sim
        plot_multi_seed_2d_contour(fileName, outputs, base_params, dpi=200)
    else:
        print("\n⚠ No 'cars_on_sale' found in outputs.")
    
    plt.show()


if __name__ == "__main__":
    main(fileName="results/multi_seed_cars_16_12_08__04_08_2026")