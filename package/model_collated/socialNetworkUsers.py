import numpy as np
import networkx as nx
import scipy.sparse as sp
from collections import defaultdict
from package.model_collated.personalCar import PersonalCar

class SocialNetworkUsers:
    def __init__(self, parameters_social_network, parameters_vehicle_user):
        # Initialize network and user parameters
        self.num_individuals = parameters_social_network["num_individuals"]
        self.id_generator = parameters_social_network["IDGenerator_firms"]
        self.second_hand_merchant = parameters_social_network["second_hand_merchant"]

        # Social preferences and other parameters
        self.parameters_vehicle_user = parameters_vehicle_user
        self.init_social_preferences(parameters_social_network)

        # Vehicles and adjacency matrix
        self.available_vehicles = parameters_social_network["init_vehicle_options"]
        self.adjacency_matrix, self.network = self.create_network(parameters_social_network)
        
        # Initialize user data and time series storage
        self.initialize_user_data()
        self.select_initial_vehicles()
        self.setup_time_series()

    def init_social_preferences(self, params):
        # Initialize preferences: environment, innovativeness, price sensitivity, origin, min distances
        np.random.seed(params["init_vals_environmental_seed"])
        self.gamma_vec = np.random.beta(params["a_environment"], params["b_environment"], size=self.num_individuals)

        np.random.seed(params["init_vals_innovative_seed"])
        self.chi_vec = np.random.beta(params["a_innovativeness"], params["b_innovativeness"], size=self.num_individuals)

        np.random.seed(params["init_vals_price_seed"])
        self.beta_vec = np.random.beta(params["a_price"], params["b_price"], size=self.num_individuals)

        np.random.seed(params["d_min_seed"])
        self.d_i_min_vec = np.random.uniform(size=self.num_individuals)

        # Set urban/rural origin randomly
        self.origin_vec = np.random.randint(2, size=self.num_individuals)

        #print("self.gamma_vec", self.gamma_vec)
        #print("self.chi_vec", self.chi_vec)
        #print(" self.d_i_min_vec",  self.d_i_min_vec.shape)
        #print("self.origin_vec", self.origin_vec.shape)
        #print("self.beta_vec", self.beta_vec)
        #quit()


    def create_network(self, params):
        # Generate Watts-Strogatz small-world network
        network = nx.watts_strogatz_graph(n=self.num_individuals, k=int(self.num_individuals * params["network_density"]), p=params["prob_rewire"], seed=params["network_structure_seed"])
        adjacency_matrix = nx.to_numpy_array(network)
        self.sparse_adjacency_matrix = sp.csr_matrix(adjacency_matrix)
        # Calculate the total number of neighbors for each user
        self.total_neighbors = np.array(self.sparse_adjacency_matrix.sum(axis=1)).flatten()
        return sp.csr_matrix(adjacency_matrix), network

    def initialize_user_data(self):
        # Initialize vehicle ownership, types, adoption flags, and vectors for computations
        self.consider_ev_vec = np.zeros(self.num_individuals)
        self.vehicle_type_vec = np.zeros(self.num_individuals)
        self.vehicle_id_vec = np.zeros(self.num_individuals)  # Unique vehicle identifiers
        self.vehicle_objs = np.empty(self.num_individuals, dtype=object)  # Stores the actual vehicle objects
        self.update_vehicle_data()

    def select_initial_vehicles(self):
        # Each user initially chooses a vehicle
        choices = self.sequential_select_vehicle()
        self.vehicle_id_vec, self.vehicle_type_vec = choices[:, 0], choices[:, 1]

    def sequential_select_vehicle(self):
        # Vectorized calculation of utilities for each user and each available vehicle
        utilities = np.array([[self.calculate_utility(vehicle, user_idx) for vehicle in self.available_vehicles]
                              for user_idx in range(self.num_individuals)])
        
        # Track availability for each vehicle; start with all vehicles available
        available_mask = np.ones(len(self.available_vehicles), dtype=bool)
        chosen_vehicle_ids = []
        chosen_vehicle_types = []

        for user_idx in range(self.num_individuals):
            # Mask unavailable second-hand vehicles in the utility array
            masked_utilities = np.where(available_mask, utilities[user_idx], -np.inf)
            
            # Calculate probabilities based on masked utilities for the user
            probabilities = self.softmax(masked_utilities)
            
            # Choose a vehicle based on probabilities
            choice = np.random.choice(len(self.available_vehicles), p=probabilities)
            chosen_vehicle = self.available_vehicles[choice]
            
            # Append the chosen vehicle's ID and type
            chosen_vehicle_ids.append(chosen_vehicle.id)
            chosen_vehicle_types.append(chosen_vehicle.transportType)
            
            # Update availability if the chosen vehicle is second-hand (can only be chosen once)
            if chosen_vehicle.owner_id == self.second_hand_merchant.id:
                available_mask[choice] = False  # Mark as unavailable for future users

        return np.stack((np.array(chosen_vehicle_ids), np.array(chosen_vehicle_types)), axis=-1)

    def calculate_utility(self, vehicle, user_idx):
        # Compute utility for a specific vehicle and user
        d_i_t = np.maximum(self.d_i_min_vec[user_idx], self.vectorized_optimal_distance(vehicle, user_idx))
        commuting_util = self.vectorized_commuting_utility(vehicle, d_i_t, user_idx)
        lifetime_util = commuting_util / (self.parameters_vehicle_user["r"] + (1 - vehicle.delta_z) / (1 - self.parameters_vehicle_user["alpha"]))
        return lifetime_util - self.beta_vec[user_idx] * vehicle.price - self.gamma_vec[user_idx] * vehicle.emissions

    def vectorized_optimal_distance(self, vehicle, user_idx):
        # Compute optimal travel distance based on user and vehicle parameters (vectorized)
        numerator = self.parameters_vehicle_user["alpha"] * vehicle.Quality_a_t * (1 - vehicle.delta_z) ** vehicle.L_a_t
        denominator = (self.beta_vec[user_idx] * vehicle.Eff_omega_a_t ** -1 * vehicle.fuel_cost_c_z +
                       self.gamma_vec[user_idx] * vehicle.Eff_omega_a_t ** -1 * vehicle.e_z_t +
                       self.parameters_vehicle_user["eta"] * vehicle.nu_z_i_t)
        return (numerator / denominator) ** (1 / (1 - self.parameters_vehicle_user["alpha"]))

    def vectorized_commuting_utility(self, vehicle, d_i_t, user_idx):
        # Compute commuting utility for a vector of distances
        cost_component = (self.beta_vec[user_idx] * (1 / vehicle.Eff_omega_a_t) * vehicle.fuel_cost_c_z +
                          self.gamma_vec[user_idx] * (1 / vehicle.Eff_omega_a_t) * vehicle.e_z_t +
                          self.parameters_vehicle_user["eta"] * vehicle.nu_z_i_t)
        return np.maximum(0, vehicle.Quality_a_t * (1 - vehicle.delta_z) ** vehicle.L_a_t * (d_i_t ** self.parameters_vehicle_user["alpha"]) - d_i_t * cost_component)

    def update_vehicle_data(self):
        # Update data based on choices of vehicles in the current time step
        # Create a sparse vector for EV adoption (where vehicle_type_vec == 3)
        self.ev_adoption_vec = sp.csr_matrix((self.vehicle_type_vec == 3).astype(int))

        # Use sparse matrix multiplication to calculate the number of EV-adopting neighbors
        ev_neighbors = self.sparse_adjacency_matrix.dot(self.ev_adoption_vec.T).toarray().flatten()

        # Calculate the proportion of neighbors with EVs
        proportion_ev_neighbors = np.divide(ev_neighbors, self.total_neighbors, where=self.total_neighbors != 0)

        # Determine whether each user considers buying an EV based on the chi threshold
        self.consider_ev_vec = (proportion_ev_neighbors >= self.chi_vec).astype(int)

    def update_vehicles_ownership(self):
        # Update the ownership status of vehicles based on decisions and transfers
        current_vehicle_ids = np.array([v.id if v else -1 for v in self.vehicle_objs])
        needs_transfer = current_vehicle_ids != self.vehicle_id_vec

        # Transfer vehicles to the second-hand merchant where needed
        for user_idx, transfer in enumerate(needs_transfer):
            if transfer:
                current_vehicle = self.vehicle_objs[user_idx]
                if current_vehicle:  # Transfer current vehicle if it exists
                    current_vehicle.owner_id = self.second_hand_merchant.id
                    self.second_hand_merchant.add_to_stock(current_vehicle)

                chosen_vehicle_id = self.vehicle_id_vec[user_idx]
                chosen_vehicle = next(v for v in self.available_vehicles if v.id == chosen_vehicle_id)

                if chosen_vehicle.owner_id == self.second_hand_merchant.id:  # Second-hand vehicle
                    chosen_vehicle.owner_id = user_idx
                    self.second_hand_merchant.remove_car(chosen_vehicle)
                    self.vehicle_objs[user_idx] = chosen_vehicle
                elif chosen_vehicle.transportType in [0, 1]:  # Public transport
                    self.vehicle_objs[user_idx] = chosen_vehicle
                else:  # New vehicle purchase
                    new_vehicle_id = self.id_generator.get_new_id()
                    new_personal_car = PersonalCar(
                        unique_id=new_vehicle_id,
                        firm=chosen_vehicle.firm,
                        owner_id=user_idx,
                        component_string=chosen_vehicle.component_string,
                        parameters=chosen_vehicle.parameters,
                        attributes_fitness=chosen_vehicle.attributes_fitness,
                        sale_price=chosen_vehicle.price
                    )
                    self.vehicle_objs[user_idx] = new_personal_car

    def setup_time_series(self):
        # Initialize time series tracking dictionaries
        self.time_series_data = {
            "driving_emissions": [],
            "production_emissions": [],
            "total_utility": [],
            "total_distance": [],
            "ev_adoption_rate": [],
            "consider_ev_rate": [],
            "vehicle_type_counts": defaultdict(list)
        }

    def save_time_series_data(self):
        # Calculate and store time series metrics (vectorized)
        driving_emissions = np.sum(self.vehicle_type_vec == 2)  # Example: ICE emissions
        production_emissions = np.sum(self.vehicle_type_vec == 3)  # Example: EV production emissions
        total_utility = np.sum([self.calculate_utility(vehicle, idx) for idx, vehicle in enumerate(self.available_vehicles)])

    ####################################################################################################################   
    def set_up_time_series_social_network(self):
        self.history_driving_emissions = []
        self.history_production_emissions = []
        self.history_total_utility = []
        self.history_total_distance_driven = []
        self.history_ev_adoption_rate = []
        self.history_urban_public_transport_users = []
        self.history_rural_public_transport_users = []
        self.history_consider_ev_rate = []
        self.history_ICE_users = []
        self.history_EV_users = []
        self.history_second_hand_users = []
        # New history attributes for vehicle attributes
        self.history_quality = []
        self.history_efficiency = []
        self.history_production_cost = []
        self.history_attributes_EV_cars_on_sale_all_firms = []
        self.history_attributes_ICE_cars_on_sale_all_firms = []

    def save_timeseries_data_social_network(self):
        #self.history_driving_emissions.append(self.total_driving_emissions)
        pass
        """
        self.history_driving_emissions.append(self.total_driving_emissions)
        self.history_production_emissions.append(self.total_production_emissions)
        self.history_total_utility.append(self.total_utility)
        self.history_total_distance_driven.append(self.total_distance_travelled)
        self.history_ev_adoption_rate.append(np.mean(self.ev_adoption_vec))
        self.history_consider_ev_rate.append(np.mean(self.consider_ev_vec))
        self.history_urban_public_transport_users.append(self.urban_public_transport_users)
        self.history_rural_public_transport_users.append(self.rural_public_transport_users)
        self.history_ICE_users.append(self.ICE_users)
        self.history_EV_users.append(self.EV_users)
        self.history_second_hand_users.append(self.second_hand_users)

        # New history saving for vehicle attributes
        quality_vals = [vehicle.Quality_a_t for vehicle in self.vehicles_chosen_list]
        efficiency_vals = [vehicle.Eff_omega_a_t for vehicle in self.vehicles_chosen_list]
        production_cost_vals = [vehicle.ProdCost_z_t for vehicle in self.vehicles_chosen_list]

        self.history_quality.append(quality_vals)
        self.history_efficiency.append(efficiency_vals)
        self.history_production_cost.append(production_cost_vals)
        data_ev = [[vehicle.Quality_a_t, vehicle.Eff_omega_a_t, vehicle.ProdCost_z_t]   for vehicle in self.cars_on_sale_all_firms if vehicle.transportType == 3]
        data_ice = [[vehicle.Quality_a_t ,vehicle.Eff_omega_a_t, vehicle.ProdCost_z_t]   for vehicle in self.cars_on_sale_all_firms if vehicle.transportType == 2]

        self.history_attributes_EV_cars_on_sale_all_firms.append(data_ev)
        self.history_attributes_ICE_cars_on_sale_all_firms.append(data_ice)
        """

    #########################################################################################################

    @staticmethod
    def softmax(values):
        # Utility function for softmax probability calculation
        exp_vals = np.exp(values - np.max(values))
        return exp_vals / exp_vals.sum()


    def next_step(self, carbon_price, vehicles_on_sale):
        # Update step with available vehicles, updating choices for each user
        self.available_vehicles = vehicles_on_sale
        choices = self.sequential_select_vehicle()
        self.vehicle_id_vec, self.vehicle_type_vec = choices[:, 0], choices[:, 1]
        self.update_vehicle_data()
        self.update_vehicles_ownership()  # Update ownership and transfer to second-hand merchant if necessary
        self.save_time_series_data()

        return self.consider_ev_vec, [vehicle for vehicle in self.available_vehicles if vehicle.id in self.vehicle_id_vec]

