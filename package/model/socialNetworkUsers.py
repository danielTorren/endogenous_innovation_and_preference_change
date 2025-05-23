# imports
import numpy as np
import networkx as nx
import numpy.typing as npt
import scipy.sparse as sp
import numpy as np
from package.model.personalCar import PersonalCar
from package.model.VehicleUser import VehicleUser
from package.model.carModel import CarModel

class Social_Network:
    def __init__(self, parameters_social_network: dict, parameters_vehicle_user: dict):
        """
        Initialize the Social_Network model with user parameters and policy settings.

        Args:
            parameters_social_network (dict): Parameters defining social, behavioral, and policy settings.
            parameters_vehicle_user (dict): 
        """
        self.t_social_network = 0

        self.policy_distortion = 0
        self.net_policy_distortion = 0
        
        self.rebate = parameters_social_network["rebate"]
        self.used_rebate = parameters_social_network["used_rebate"]

        self.rebate_calibration = parameters_social_network["rebate"]
        self.used_rebate_calibration = parameters_social_network["used_rebate"]

        self.prob_switch_car = parameters_social_network["prob_switch_car"]

        self.beta_vec = parameters_social_network["beta_vec"] 
        self.gamma_vec = parameters_social_network["gamma_vec"]
        self.chi_vec = parameters_social_network["chi_vec"]
        self.nu_vec = parameters_social_network["nu_vec"]

        self.d_vec = parameters_social_network["d_vec"]

        self.alpha = parameters_social_network["alpha"]
        self.zeta = parameters_social_network["zeta"]

        self.scrap_price = parameters_social_network["scrap_price"]

        self.beta_segment_vec = parameters_social_network["beta_segment_vals"] 
        self.gamma_segment_vec = parameters_social_network["gamma_segment_vals"] 

        self.history_prop_EV = []
        
        # Initialize parameters
        self.parameters_vehicle_user = parameters_vehicle_user
        self.init_initial_state(parameters_social_network)

        #measure effects on bottom and top percentiles
        self.beta_median = np.median(self.beta_vec )
        self.beta_rich = np.percentile(self.beta_vec, 90)
        self.num_poor = self.num_individuals*0.5
        self.num_rich = self.num_individuals*0.1

        self.gamma_median = np.median(self.gamma_vec)

        self.emissions_cumulative = 0
        self.emissions_cumulative_production = 0
        self.emissions_cumulative_driving = 0

        self.emissions_flow = 0
        self.utility_cumulative = 0

        self.init_network_settings(parameters_social_network)

        self.random_state = parameters_social_network["random_state"]
        self.seed_inputs = parameters_social_network["seed_inputs"]

        self.mu =  parameters_vehicle_user["mu"]
        self.r = parameters_vehicle_user["r"]
        self.kappa = parameters_vehicle_user["kappa"]

        # Generate a list of indices and shuffle them
        self.user_indices = np.arange(self.num_individuals)

        # Efficient user list creation with list comprehension
        self.vehicleUsers_list = [VehicleUser(user_id=i) for i in range(self.num_individuals)]

        # Create network and calculate initial emissions
        self.adjacency_matrix, self.network = self.create_network()
        self.network_density = nx.density(self.network)

        self.weighting_matrix = self._normlize_matrix(self.adjacency_matrix)#INTRODUCE HOMOPHILY INTO THE NETWORK BY ASSORTING BY BETA WITHING GROUPS

        #Assume nobody adopts EV at the start, THIS MAY BE AN ISSUE
        self.consider_ev_vec = np.zeros(self.num_individuals).astype(np.int8)

        self.current_vehicles = self.set_init_cars_selection(parameters_social_network)
        
        self.consider_ev_vec, self.ev_adoption_vec = self.calculate_ev_adoption(ev_type=3)#BASED ON CONSUMPTION PREVIOUS TIME STEP

    def init_initial_state(self, parameters_social_network):
        """
        Initialize key state variables related to users, policies, and system settings.

        Args:
            parameters_social_network (dict): Dictionary of parameters defining the structure and initial setup of the social network.
        """
        self.num_individuals = int(round(parameters_social_network["num_individuals"]))
        self.id_generator = parameters_social_network["IDGenerator_firms"]
        self.second_hand_merchant = parameters_social_network["second_hand_merchant"]
        self.burn_in_second_hand_market = self.second_hand_merchant.burn_in_second_hand_market
        self.save_timeseries_data_state = parameters_social_network["save_timeseries_data_state"]
        self.compression_factor_state = parameters_social_network["compression_factor_state"]
        self.carbon_price =  parameters_social_network["carbon_price"]

    def init_network_settings(self, parameters_social_network):
        """
        Initialize the network settings
        """

        self.prob_rewire = parameters_social_network["SW_prob_rewire"]
        self.SW_network_density_input = parameters_social_network["SW_network_density"]
        self.SW_prob_rewire = parameters_social_network["SW_prob_rewire"]
        self.SW_K = int(round((self.num_individuals - 1) * self.SW_network_density_input))

    def set_init_cars_selection(self, parameters_social_network):
        """
        Assign each user an initial car based on utility maximization, without allowing user choice.

        Args:
            parameters_social_network (dict): Contains the list of old cars and related attributes.

        Returns:
            list: Assigned list of vehicle objects corresponding to each user.
        """
        old_cars = parameters_social_network["old_cars"]

        # Extract properties using list comprehensions
        quality_a_t = np.array([vehicle.Quality_a_t for vehicle in old_cars])
        eff_omega_a_t = np.array([vehicle.Eff_omega_a_t for vehicle in old_cars])
        ProdCost_t = np.array([vehicle.ProdCost_t for vehicle in old_cars])
        production_emissions = np.array([vehicle.emissions for vehicle in old_cars])
        fuel_cost_c = np.array([vehicle.fuel_cost_c for vehicle in old_cars])
        e_t = np.array([vehicle.e_t for vehicle in old_cars])
        transport_type = np.array([vehicle.transportType for vehicle in old_cars])
        delta = np.array([vehicle.delta for vehicle in old_cars])
        rebate_vec = np.where(transport_type == 3, self.rebate_calibration + self.rebate, 0)
        B = np.array([vehicle.B for vehicle in old_cars])
        # Create the dictionary directly with NumPy arrays
        vehicle_dict_vecs = {
            "Quality_a_t": quality_a_t,
            "Eff_omega_a_t": eff_omega_a_t,
            "ProdCost_t": ProdCost_t,
            "production_emissions": production_emissions,
            "fuel_cost_c": fuel_cost_c,
            "e_t": e_t,
            "transportType": transport_type,
            "rebate": rebate_vec,
            "delta": delta,
            "B": B
        }

        # Calculate price difference, applying rebate only for transportType == 3 (included in rebate calculation)

        price_difference = 1.2*vehicle_dict_vecs["ProdCost_t"][:, np.newaxis]  # Apply rebate
        price_difference_T = price_difference.T
        U_a_i_t_matrix  = -price_difference_T - self.gamma_vec[:, np.newaxis]*vehicle_dict_vecs["production_emissions"] + self.beta_vec[:, np.newaxis]*vehicle_dict_vecs["Quality_a_t"]**self.alpha + self.nu_vec[:, np.newaxis]*(vehicle_dict_vecs["B"]*vehicle_dict_vecs["Eff_omega_a_t"])**self.zeta - self.d_vec[:, np.newaxis]*(((1+self.r)*(1-vehicle_dict_vecs["delta"])*(vehicle_dict_vecs["fuel_cost_c"] + self.gamma_vec[:, np.newaxis]*vehicle_dict_vecs["e_t"]))/(vehicle_dict_vecs["Eff_omega_a_t"]*(self.r - vehicle_dict_vecs["delta"] - self.r*vehicle_dict_vecs["delta"])))

        #U_a_i_t_matrix = self.beta_vec[:, np.newaxis]*vehicle_dict_vecs["Quality_a_t"]**self.alpha + self.nu_vec[:, np.newaxis]*(vehicle_dict_vecs["B"]*vehicle_dict_vecs["Eff_omega_a_t"]*(1-vehicle_dict_vecs["delta"])**vehicle_dict_vecs["L_a_t"])**self.zeta - self.d_vec[:, np.newaxis]*(((1+self.r)*(1-vehicle_dict_vecs["delta"])*(vehicle_dict_vecs["fuel_cost_c"] + self.gamma_vec[:, np.newaxis]*vehicle_dict_vecs["e_t"]))/(vehicle_dict_vecs["Eff_omega_a_t"]*((1-vehicle_dict_vecs["delta"])**vehicle_dict_vecs["L_a_t"])*(self.r - vehicle_dict_vecs["delta"] - self.r*vehicle_dict_vecs["delta"])))

        # Sort people by their maximum utility for any car
        people_indices = np.argsort(np.max(U_a_i_t_matrix, axis=1))[::-1]  # Descending order
        assigned_cars = set()
        
        # Initialize vehicle assignment
        user_vehicle_map = {}

        for person_idx in people_indices:
            # Find the car with the highest utility for this person
            car_utilities = U_a_i_t_matrix[person_idx, :]
            sorted_car_indices = np.argsort(car_utilities)[::-1]  # Descending order

            # Assign the first available car from the sorted list
            for car_idx in sorted_car_indices:
                if car_idx not in assigned_cars:
                    assigned_cars.add(car_idx)
                    user_vehicle_map[person_idx] = car_idx
                    break  # Move to the next person after assigning a car

        # Assign cars based on the computed mapping
        for i, (person_idx, car_idx) in enumerate(user_vehicle_map.items()):
            self.vehicleUsers_list[person_idx].vehicle = old_cars[car_idx]

        # Set the user ID of cars
        for individual in self.vehicleUsers_list:
            individual.vehicle.owner_id = individual.user_id

        current_cars = [user.vehicle for user in self.vehicleUsers_list]


        return current_cars  # Return the assigned cars
    
    def _normlize_matrix(self, matrix: sp.csr_matrix) -> sp.csr_matrix:
        """
        Normalize a sparse matrix row-wise.
        
        Args:
            matrix (sp.csr_matrix): Sparse matrix to normalize
            
        Returns:
            sp.csr_matrix: Row-normalized sparse matrix
        """
        row_sums = np.array(matrix.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        inv_row_sums = 1.0 / row_sums
        diagonal_matrix = sp.diags(inv_row_sums)
        norm_matrix = diagonal_matrix.dot(matrix)
        return norm_matrix

    def create_network(self) -> tuple[npt.NDArray, npt.NDArray, nx.Graph]:
        """
        Create a Watts-Strogatz small-world network to model user interactions.

        Returns:
            tuple: 
                adjacency_matrix (np.ndarray): Binary matrix indicating connections between users.
                network (nx.Graph): NetworkX graph object representing the social network.
        """

        network = nx.watts_strogatz_graph(n=self.num_individuals, k=self.SW_K, p=self.prob_rewire, seed=self.seed_inputs)#FIX THE NETWORK STRUCTURE

        adjacency_matrix = nx.to_numpy_array(network)
        self.sparse_adjacency_matrix = sp.csr_matrix(adjacency_matrix)
        # Get the non-zero indices of the adjacency matrix
        self.row_indices_sparse, self.col_indices_sparse = self.sparse_adjacency_matrix.nonzero()

        self.network_density = nx.density(network)
        # Calculate the total number of neighbors for each user
        self.total_neighbors = np.array(self.sparse_adjacency_matrix.sum(axis=1)).flatten()
        return adjacency_matrix, network

    def calculate_ev_adoption(self, ev_type=3):
        """
        Determine which users consider adopting electric vehicles (EVs) based on neighbor influence.

        Args:
            ev_type (int): Vehicle type identifier for EVs. Defaults to 3.

        Returns:
            tuple:
                consider_ev_vec (np.ndarray): Binary vector of users considering EVs.
                ev_adoption_vec (np.ndarray): Binary vector of users currently using EVs.
        """
        
        self.vehicle_type_vec = np.array([user.vehicle.transportType for user in self.vehicleUsers_list])  # Current vehicle types

        # Create a binary vec indicating EV users
        ev_adoption_vec = (self.vehicle_type_vec == ev_type).astype(int)

        # Calculate the number of EV-adopting neighbors using sparse matrix multiplication
        ev_neighbors = self.weighting_matrix.dot(ev_adoption_vec)

        consider_ev_vec = (ev_neighbors >= self.chi_vec).astype(np.int8)

        return consider_ev_vec, ev_adoption_vec

    def update_VehicleUsers(self):
        """
        Perform a full timestep update for all users:
        - Determines who considers switching vehicles.
        - Calculates utilities for keeping or switching.
        - Applies policy impacts and updates emissions.
        - Tracks all related counters and statistics.

        Returns:
            list: Updated list of current vehicles assigned to each user.
        """

        self.new_bought_vehicles = []#track list of new vehicles
        self.second_hand_bought = 0#track number of second hand bought
        user_vehicle_list = self.current_vehicles.copy()#assume most people keep their cars
        
        #########################################################
        #LIMIT CALCULATION FOR THOSE THAT DONT NEED TO SWTICH
        # 1) Determine which users can switch
        switch_draws = (self.random_state.rand(self.num_individuals) < self.prob_switch_car)
        switcher_indices = np.where(switch_draws)[0]  # e.g., [2, 5, 7, ...]
        num_switchers = len(switcher_indices)
        non_switcher_indices = np.where(~switch_draws)[0]  # e.g., [0, 1, 3, 4, 6, ...]

        if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
            self.emissions_flow = 0#MEASURIBNG THE FLOW
            self.zero_util_count = 0#tracking if are people actually choosing or beign forced to choose the same
            self.num_switchers = num_switchers
            self.drive_min_num = 0

        self.sub_beta_vec = self.beta_vec[switcher_indices]
        self.sub_gamma_vec = self.gamma_vec[switcher_indices]
        self.sub_d_vec = self.d_vec[switcher_indices]
        self.sub_nu_vec = self.nu_vec[switcher_indices]

        # Generate current utilities and vehicles
        #Calculate the optimal distance for all user with current car, NEEDS TO BE DONE ALWAYS AND FOR ALL USERS
        CV_vehicle_dict_vecs = self.gen_current_vehicle_dict_vecs(self.current_vehicles)

        # NEED THIS FOR SOME OF THE COUNTERS I THINK - CHECK THIS
        if self.second_hand_cars:
            index_current_cars_start = len(self.new_cars) + len(self.second_hand_cars)
        else:
            index_current_cars_start = len(self.new_cars)

        #NON-SWTICHERS
        if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
            self.prep_counters()
            __, full_CV_utility_vec = self.generate_utilities_current(CV_vehicle_dict_vecs, self.beta_vec, self.gamma_vec, self.d_vec, self.nu_vec)


        for i, person_index in enumerate(non_switcher_indices):
            user = self.vehicleUsers_list[person_index]
            user.vehicle.update_timer_L_a_t()# Update the age or timer of the chosen vehicle
            # Handle consequences of the choice
            driven_distance = self.d_vec[person_index]
            self.update_emisisons(user.vehicle, driven_distance)

            #NEEDED FOR OPTIMISATION, ELECTRICITY SUBSIDY
            if user.vehicle.transportType == 3:
                elec_sub = (self.electricity_price_subsidy_dollars*driven_distance)/user.vehicle.Eff_omega_a_t
                self.policy_distortion += elec_sub 
                self.net_policy_distortion -= elec_sub 
            else:
                #NEEDED FOR OPTIMISATION, add the carbon price paid 
                carbon_price_paid = (self.carbon_price*user.vehicle.e_t*driven_distance)/user.vehicle.Eff_omega_a_t
                self.policy_distortion += carbon_price_paid
                self.net_policy_distortion += carbon_price_paid

            if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
                self.keep_car += 1
                utility = full_CV_utility_vec[person_index]
                self.update_counters(person_index, user.vehicle, driven_distance, utility) 

                
        ##################################################################
        #SWITCHERS

        # 2) Shuffle only that subset of user indices
        shuffle_indices = self.random_state.permutation(switcher_indices)

        self.NC_vehicle_dict_vecs = self.gen_vehicle_dict_vecs_new_cars(self.new_cars)

        #THIS CAN BE DONE FOR THE SUBSET OF USERS
        CV_filtered_vechicles_dicts, CV_filtered_vehicles = self.filter_vehicle_dict_for_switchers(CV_vehicle_dict_vecs, self.current_vehicles, switcher_indices)
        utilities_current_matrix, __ = self.generate_utilities_current(CV_filtered_vechicles_dicts, self.sub_beta_vec, self.sub_gamma_vec, self.sub_d_vec, self.sub_nu_vec)

        self.second_hand_merchant_offer_price = self.calc_offer_prices_heursitic(self.NC_vehicle_dict_vecs, CV_filtered_vechicles_dicts, CV_filtered_vehicles)

        # pass those indices to generate_utilities
        utilities_buying_matrix_switchers, buying_vehicles_list = self.generate_utilities(self.sub_beta_vec, self.sub_gamma_vec, self.second_hand_merchant_offer_price, self.sub_d_vec, self.sub_nu_vec)

        # Preallocate the final utilities and distance matrices
        total_columns = utilities_buying_matrix_switchers.shape[1] + utilities_current_matrix.shape[1]#number of cars which is new+secodn hand + current in the switchers

        self.utilities_matrix_switchers = np.empty((num_switchers, total_columns))

        # Assign matrices directly
        self.utilities_matrix_switchers[:, :utilities_buying_matrix_switchers.shape[1]] = utilities_buying_matrix_switchers
        self.utilities_matrix_switchers[:, utilities_buying_matrix_switchers.shape[1]:] = utilities_current_matrix

        # Combine the list of vehicles
        available_and_current_vehicles_list = buying_vehicles_list + CV_filtered_vehicles# ITS CURRENT VEHICLES AND NOT FILTERED VEHCILES AS THE SHUFFLING INDEX DOENST ACCOUNT FOR THE FILTERING

        utilities_kappa = self.masking_options(self.utilities_matrix_switchers, available_and_current_vehicles_list, self.consider_ev_vec[switcher_indices])

        #########################################################################################################################
        # Create a mapping from global to reduced indices
        global_to_reduced = {global_idx: local_idx for local_idx, global_idx in enumerate(switcher_indices)}
        # Translate shuffle_indices to reduced indices
        shuffle_indices_reduced = [global_to_reduced[idx] for idx in shuffle_indices]

        #########################################################################################################################

        for i, reduced_index in enumerate(shuffle_indices_reduced):
            # Map the reduced index back to the global index
            global_index = switcher_indices[reduced_index]
            
            user = self.vehicleUsers_list[global_index]  # Use the global index to access the user
            vehicle_chosen, user_vehicle, vehicle_chosen_index, utilities_kappa = self.user_chooses(
                global_index, user, available_and_current_vehicles_list, utilities_kappa, reduced_index, index_current_cars_start 
            )
            user_vehicle_list[global_index] = user_vehicle  # Update using the global index

            driven_distance = self.d_vec[global_index]  # Use the reduced index for the matrix
            self.update_emisisons(vehicle_chosen, driven_distance)

            if user.vehicle.transportType == 3:
                elec_sub = (self.electricity_price_subsidy_dollars*driven_distance)/user.vehicle.Eff_omega_a_t
                self.policy_distortion += elec_sub
                self.net_policy_distortion -= elec_sub
            else:
                carbon_price_paid = (self.carbon_price*user.vehicle.e_t*driven_distance)/user.vehicle.Eff_omega_a_t
                self.policy_distortion += carbon_price_paid
                self.net_policy_distortion += carbon_price_paid
            
            utility = self.utilities_matrix_switchers[reduced_index][vehicle_chosen_index]
            self.utility_cumulative += utility
            
            if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
                
                self.update_counters(global_index, vehicle_chosen, driven_distance, utility)

        if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
            self.emissions_flow_history.append(self.emissions_flow)

        return user_vehicle_list

    def calc_offer_prices_heursitic(self, vehicle_dict_vecs_new_cars, vehicle_dict_vecs_current_cars, current_cars):
        """
        Estimate second-hand car offer prices using heuristic comparison to new cars.

        Args:
            vehicle_dict_vecs_new_cars (dict): Property arrays of new cars.
            vehicle_dict_vecs_current_cars (dict): Property arrays of second-hand cars.
            current_cars (list): List of current vehicle objects.

        Returns:
            np.ndarray: Offer prices for second-hand cars.
        """
        # Extract Quality, Efficiency, and Prices of first-hand cars
        first_hand_quality = vehicle_dict_vecs_new_cars["Quality_a_t"]
        first_hand_efficiency =  vehicle_dict_vecs_new_cars["Eff_omega_a_t"]
        first_hand_prices = vehicle_dict_vecs_new_cars["price"]
        first_hand_B = vehicle_dict_vecs_new_cars["B"]

        # Extract Quality, Efficiency, and Age of second-hand cars
        second_hand_quality = vehicle_dict_vecs_current_cars["Quality_a_t"]
        second_hand_efficiency = vehicle_dict_vecs_current_cars["Eff_omega_a_t"]
        second_hand_ages = vehicle_dict_vecs_current_cars["L_a_t"]
        second_hand_delta_P = vehicle_dict_vecs_current_cars["delta_P"]
        second_hand_B = vehicle_dict_vecs_current_cars["B"]
        
        first_hand_quality_max = np.max(first_hand_quality)
        first_hand_efficiency_max = np.max(first_hand_efficiency)
        first_hand_B_max = np.max(first_hand_B)

        normalized_first_hand_quality = first_hand_quality / first_hand_quality_max 
        normalized_first_hand_efficiency = first_hand_efficiency / first_hand_efficiency_max 
        normalized_first_hand_B = first_hand_B / first_hand_B_max

        normalized_second_hand_quality = second_hand_quality  / first_hand_quality_max 
        normalized_second_hand_efficiency = second_hand_efficiency / first_hand_efficiency_max
        normalized_second_hand_B = second_hand_B / first_hand_B_max

        # Compute proximity (Euclidean distance) for all second-hand cars to all first-hand cars
        diff_quality = normalized_second_hand_quality[:, np.newaxis] - normalized_first_hand_quality
        diff_efficiency = normalized_second_hand_efficiency[:, np.newaxis] - normalized_first_hand_efficiency
        diff_B = normalized_second_hand_B[:, np.newaxis] - normalized_first_hand_B

        distances = np.sqrt(diff_quality ** 2 + diff_efficiency ** 2 + diff_B ** 2)

        # Find the closest first-hand car for each second-hand car
        closest_idxs = np.argmin(distances, axis=1)

        # Get the prices of the closest first-hand cars
        #closest_prices = first_hand_prices[closest_idxs]
        closest_prices = np.maximum(first_hand_prices[closest_idxs] - (self.rebate_calibration + self.rebate),0)

        # Adjust prices based on car age and depreciation
        adjusted_prices = closest_prices * (1 - second_hand_delta_P) ** second_hand_ages

        # Calculate offer prices
        offer_prices = adjusted_prices / (1 + self.mu)

        # Ensure offer prices are not below the scrap price
        offer_prices = np.maximum(offer_prices, self.scrap_price)

        # Assign prices back to second-hand car objects
        for i, car in enumerate(current_cars):
            car.price_second_hand_merchant = adjusted_prices[i]
            car.cost_second_hand_merchant = offer_prices[i]

        return offer_prices

    def gen_mask(self, available_and_current_vehicles_list, consider_ev_vec):
        """
        Generate mask for valid EV options based on user consideration.

        Args:
            available_and_current_vehicles_list (list): All vehicle options.
            consider_ev_vec (np.ndarray): Vector indicating which users consider EVs.

        Returns:
            np.ndarray: Mask matrix of shape (users, vehicles).
        """
            
        # Generate individual masks based on vehicle type and user conditions
        # Create a boolean vector where True indicates that a vehicle is NOT an EV (non-EV)
        not_ev_vec = np.array([vehicle.transportType == 2 for vehicle in available_and_current_vehicles_list], dtype=bool)
        ones = np.ones(len(not_ev_vec), dtype=bool)
        consider_ev_all_vehicles = np.outer(consider_ev_vec == 1, ones)#DO CONSIDER EVS #THIS IS TOO GET THE CORRECT SHAPE ie PEOPLE BY CARS
        dont_consider_evs_on_ev = np.outer(consider_ev_vec == 0, not_ev_vec)#This part masks EVs (False for EVs) only for individuals who do not consider EVs

        # Create an outer product to apply the EV consideration across all cars
        ev_mask_matrix = (consider_ev_all_vehicles | dont_consider_evs_on_ev).astype(int)

        return ev_mask_matrix

    def masking_options(self, utilities_matrix, available_and_current_vehicles_list, consider_ev_vec):
        """
        Apply mask to utility matrix to prevent users from selecting vehicles they don't consider.

        Args:
            utilities_matrix (np.ndarray): Utility matrix before masking.
            available_and_current_vehicles_list (list): Vehicles available to users.
            consider_ev_vec (np.ndarray): Which users are considering EVs.

        Returns:
            np.ndarray: Masked utility matrix.
        """
        # Step 1: Generate the mask
        combined_mask = self.gen_mask(available_and_current_vehicles_list, consider_ev_vec)

        # Step 2: Apply mask in-place, setting masked-out values to -inf
        masked_utilities = np.where(combined_mask, utilities_matrix, -np.inf)

        # Step 3: Identify valid rows (at least one non -inf value)
        valid_rows = np.any(combined_mask, axis=1)

        # Step 4: Compute row-wise max only for valid rows for numerical stability
        row_max_utilities = np.max(masked_utilities[valid_rows], axis=1, keepdims=True)

        # Further mask to remove -np.inf values
        filtered_values = masked_utilities[valid_rows]  # Subset based on valid_rows mask
        filtered_values = filtered_values[np.isfinite(filtered_values)]  # Keeps only finite values

        exp_input = self.kappa * (masked_utilities[valid_rows] - row_max_utilities)

        # Step 6: Exponentiate, directly filling only valid entries
        utilities_kappa = np.zeros_like(utilities_matrix)
        utilities_kappa[valid_rows] = np.exp(exp_input)

        return utilities_kappa

    def user_chooses(self, person_index, user, available_and_current_vehicles_list, utilities_kappa, reduced_person_index, index_current_cars_start ):
        """
        Let a user choose a vehicle based on masked and exponentiated utility values.

        Args:
            person_index (int): Global user index.
            user (VehicleUser): User object.
            available_and_current_vehicles_list (list): All available vehicles.
            utilities_kappa (np.ndarray): Masked and exponentiated utility matrix.
            reduced_person_index (int): Row index in the utility matrix.
            index_current_cars_start (int): Starting index for current vehicles.

        Returns:
            tuple: (chosen vehicle, assigned vehicle, index of chosen vehicle, updated utilities matrix)
        """
        # Select individual-specific utilities
        individual_specific_util_kappa = utilities_kappa[reduced_person_index]  
        
        #check for nans and set them to 0
        if np.isnan(individual_specific_util_kappa).any():
            individual_specific_util_kappa = np.nan_to_num(individual_specific_util_kappa)#Set all the nans to 0

        #SWICHING_CLAUSE
        if not np.any(individual_specific_util_kappa):#NO car option all zero, THIS SHOULD ONLY REALLY BE TRIGGERED RIGHT AT THE START
            #keep current car
            choice_index = index_current_cars_start + reduced_person_index
            if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
                self.zero_util_count += 1
        else:#at leat 1 non zero probability
            # Calculate the probability of choosing each vehicle              
            sum_U_kappa = np.sum(individual_specific_util_kappa)
            probability_choose = individual_specific_util_kappa / sum_U_kappa
            choice_index = self.random_state.choice(len(available_and_current_vehicles_list), p=probability_choose)

        # Record the chosen vehicle
        vehicle_chosen = available_and_current_vehicles_list[choice_index]

        # Handle consequences of the choice
        if user.user_id != vehicle_chosen.owner_id:  # New vehicle, not currently owned
            # Transfer the user's current vehicle to the second-hand merchant, if any
            if isinstance(user.vehicle, PersonalCar):#YOU SELL YOUR CAR?
                if (user.vehicle.init_car) or (user.vehicle.cost_second_hand_merchant == self.scrap_price) or (self.t_social_network <= self.burn_in_second_hand_market):#ITS AN INITAL CAR WE DOTN WANT TO ALLOW THSOE TO BE SOLD 
                    user.vehicle.owner_id = -99#send to shadow realm
                    user.vehicle = None
                else:
                    if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
                        self.second_hand_merchant.spent += user.vehicle.cost_second_hand_merchant
                        self.second_hand_merchant_price_paid.append(user.vehicle.cost_second_hand_merchant)
                    
                    user.vehicle.owner_id = self.second_hand_merchant.id
                    self.second_hand_merchant.add_to_stock(user.vehicle)
                    user.vehicle = None
                    
            if vehicle_chosen.owner_id == self.second_hand_merchant.id:# Buy a second-hand car
                #USED ADOPTION SUBSIDY OPTIMIZATION
                if vehicle_chosen.transportType == 3:
                    adopt_sub = np.minimum(vehicle_chosen.price, self.used_rebate)  
                    self.policy_distortion += adopt_sub
                    self.net_policy_distortion -= adopt_sub    

                #SET THE UTILITY TO 0 of that second hand car
                utilities_kappa[:, choice_index] = 0#THIS STOPS OTHER INDIVIDUALS FROM BUYING SECOND HAND CAR THAT YOU BOUGHT, VERY IMPORANT LINE
                
                vehicle_chosen.owner_id = user.user_id
                vehicle_chosen.scenario = "current_car"
                user.vehicle = vehicle_chosen
                self.second_hand_merchant.remove_car(vehicle_chosen)#REmove it last in case of issue of removing and the obeject disappearing
                self.second_hand_merchant.income += user.vehicle.price

                if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
                    self.car_prices_sold_second_hand.append(user.vehicle.price)
                    self.buy_second_hand_car+= 1
                    self.second_hand_bought += 1
            elif isinstance(vehicle_chosen, CarModel):  # Brand new car
                #ADOPTION SUBSIDY OPTIMIZATION
                if vehicle_chosen.transportType == 3:
                    adopt_sub = np.minimum(vehicle_chosen.price, self.rebate)    
                    self.policy_distortion +=  adopt_sub
                    self.net_policy_distortion -= adopt_sub    
            
                self.new_bought_vehicles.append(vehicle_chosen)#ADD NEW CAR TO NEW CAR LIST, used so can calculate the market concentration
                personalCar_id = self.id_generator.get_new_id()
                user.vehicle = PersonalCar(personalCar_id, vehicle_chosen.firm, user.user_id, vehicle_chosen.component_string, vehicle_chosen.parameters, vehicle_chosen.attributes_fitness, vehicle_chosen.price)
                if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
                    self.car_prices_sold_new.append(user.vehicle.price)
                    self.buy_new_car+=1
            else:
                raise(ValueError("invalid user transport behaviour"))
        else:#KEEP CAR
            if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
                self.keep_car +=1#KEEP CURRENT CAR

        # Update the age or timer of the chosen vehicle
        user.vehicle.update_timer_L_a_t()

        return vehicle_chosen, user.vehicle, choice_index, utilities_kappa
    
    def generate_utilities_current(self, vehicle_dict_vecs, beta_vec, gamma_vec, d_vec, nu_vec):# -> NDArray:
        """
        Compute utility values for users keeping their current vehicle.

        Args:
            vehicle_dict_vecs (dict): Feature matrix for current vehicles.
            beta_vec, gamma_vec, d_vec, nu_vec (np.ndarray): User-specific parameters.

        Returns:
            tuple: (utility matrix, utility vector)
        """

        U_a_i_t_vec = beta_vec*vehicle_dict_vecs["Quality_a_t"]**self.alpha + nu_vec*(vehicle_dict_vecs["B"]*vehicle_dict_vecs["Eff_omega_a_t"]*(1-vehicle_dict_vecs["delta"])**vehicle_dict_vecs["L_a_t"])**self.zeta - d_vec*(((1+self.r)*(1-vehicle_dict_vecs["delta"])*(vehicle_dict_vecs["fuel_cost_c"] + gamma_vec*vehicle_dict_vecs["e_t"]))/(vehicle_dict_vecs["Eff_omega_a_t"]*((1-vehicle_dict_vecs["delta"])**vehicle_dict_vecs["L_a_t"])*(self.r - vehicle_dict_vecs["delta"] - self.r*vehicle_dict_vecs["delta"])))
        
        # Initialize the matrix with -np.inf
        CV_utilities_matrix = np.full((len(U_a_i_t_vec), len(U_a_i_t_vec)), -np.inf)#its 
        
        # Set the diagonal values
        np.fill_diagonal(CV_utilities_matrix, U_a_i_t_vec)

        return  CV_utilities_matrix, U_a_i_t_vec
    
    def gen_current_vehicle_dict_vecs(self, list_vehicles):
        """
        Create a dictionary of vehicle attributes from a list of vehicles.

        Args:
            list_vehicles (list): List of vehicle objects.

        Returns:
            dict: Dictionary of vehicle attribute arrays.
        """
        # Extract properties using list comprehensions
        quality_a_t = np.array([vehicle.Quality_a_t for vehicle in list_vehicles])
        eff_omega_a_t = np.array([vehicle.Eff_omega_a_t for vehicle in list_vehicles])
        transport_type = np.array([vehicle.transportType for vehicle in list_vehicles])
        l_a_t = np.array([vehicle.L_a_t for vehicle in list_vehicles])
        fuel_cost_c = np.array([vehicle.fuel_cost_c for vehicle in list_vehicles])
        e_t = np.array([vehicle.e_t for vehicle in list_vehicles])
        delta = np.array([vehicle.delta for vehicle in list_vehicles])
        delta_P = np.array([vehicle.delta_P for vehicle in list_vehicles])
        B = np.array([vehicle.B for vehicle in list_vehicles])
        # Create the dictionary directly with NumPy arrays
        vehicle_dict_vecs = {
            "Quality_a_t": quality_a_t,
            "Eff_omega_a_t": eff_omega_a_t,
            "fuel_cost_c": fuel_cost_c,
            "e_t": e_t,
            "L_a_t": l_a_t,
            "transportType": transport_type,
            "delta": delta,
            "delta_P": delta_P,
            "B": B
        }

        return vehicle_dict_vecs

    def generate_utilities(self, beta_vec, gamma_vec, second_hand_merchant_offer_price, d_vec, nu_vec):
        """
        Compute utility values for all switchers over new and second-hand cars.

        Returns:
            tuple: (utility matrix, list of vehicle objects)
        """

        # Generate utilities
        #self.NC_vehicle_dict_vecs = self.gen_vehicle_dict_vecs_new_cars(self.new_cars)
        NC_utilities = self.vectorised_calculate_utility_new_cars(self.NC_vehicle_dict_vecs, beta_vec, gamma_vec, second_hand_merchant_offer_price, d_vec, nu_vec)

        # Calculate the total columns needed for utilities 
        total_columns = NC_utilities.shape[1]

        if self.second_hand_cars:
            SH_vehicle_dict_vecs = self.gen_vehicle_dict_vecs_second_hand(self.second_hand_cars)
            SH_utilities = self.vectorised_calculate_utility_second_hand_cars(SH_vehicle_dict_vecs, beta_vec, gamma_vec, second_hand_merchant_offer_price, d_vec, nu_vec)
            total_columns += SH_utilities.shape[1]

        # Preallocate arrays with the total required columns
        num_individuals_switchers = len(beta_vec)
        utilities_matrix = np.empty((num_individuals_switchers, total_columns))

        # Fill in preallocated arrays with submatrices
        col_idx = 0#start the counter at zero then increase as you set things up
        utilities_matrix[:, col_idx:col_idx + NC_utilities.shape[1]] = NC_utilities

        col_idx += NC_utilities.shape[1]

        if self.second_hand_cars:
            utilities_matrix[:, col_idx:col_idx + SH_utilities.shape[1]] = SH_utilities
            car_options = self.new_cars + self.second_hand_cars
        else:
            car_options = self.new_cars

        return utilities_matrix, car_options

    def filter_vehicle_dict_for_switchers(
        self,
        vehicle_dict_vecs: dict[str, np.ndarray],
        list_vehicles: list,
        switcher_indices: np.ndarray
    ) -> tuple[dict[str, np.ndarray], list]:
        """
        Filter an already-built 'vehicle_dict_vecs' so that it only includes
        rows corresponding to vehicles whose 'owner_id' is in 'switcher_indices'.

        Parameters
        ----------
        vehicle_dict_vecs : dict[str, np.ndarray]
            A dictionary of arrays (e.g. from gen_current_vehicle_dict_vecs),
            where each array has the same length = number_of_vehicles.
        list_vehicles : list
            The original list of vehicles in the same order used to build 'vehicle_dict_vecs'.
        switcher_indices : np.ndarray
            Array of user IDs who can switch (e.g., from np.where(switch_draws)[0]).

        Returns
        -------
        filtered_vehicle_dict : dict[str, np.ndarray]
            The same dictionary, but sliced down to only rows for switcher-owned vehicles.
        filtered_vehicles : list
            A filtered list of the actual vehicle objects corresponding to those owners.
        """

        # 1) Build an array of owner IDs for each vehicle in the same order used in vehicle_dict_vecs
        owner_ids = np.array([vehicle.owner_id for vehicle in list_vehicles], dtype=int)

        # 2) Create a boolean mask that is True if the owner is in switcher_indices
        #    (Note: np.isin will check for membership in switcher_indices)
        mask = np.isin(owner_ids, switcher_indices)

        # 3) Apply this mask to each array in vehicle_dict_vecs
        filtered_vehicle_dict = {}
        for key, arr in vehicle_dict_vecs.items():
            filtered_vehicle_dict[key] = arr[mask]

        # 4) Also build a filtered list of vehicle objects
        filtered_vehicles = [v for (v, keep) in zip(list_vehicles, mask) if keep]

        return filtered_vehicle_dict, filtered_vehicles

    def gen_vehicle_dict_vecs_new_cars(self, list_vehicles):
        """
        Generate attribute arrays for new cars.

        Args:
            list_vehicles (list): List of CarModel objects.

        Returns:
            dict: Dictionary of vehicle attributes.
        """
        # Extract properties using list comprehensions
        quality_a_t = np.array([vehicle.Quality_a_t for vehicle in list_vehicles])
        eff_omega_a_t = np.array([vehicle.Eff_omega_a_t for vehicle in list_vehicles])
        price = np.array([vehicle.price for vehicle in list_vehicles])
        production_emissions = np.array([vehicle.emissions for vehicle in list_vehicles])
        fuel_cost_c = np.array([vehicle.fuel_cost_c for vehicle in list_vehicles])
        e_t = np.array([vehicle.e_t for vehicle in list_vehicles])
        transport_type = np.array([vehicle.transportType for vehicle in list_vehicles])
        delta = np.array([vehicle.delta for vehicle in list_vehicles])
        rebate_vec = np.where(transport_type == 3, self.rebate_calibration + self.rebate, 0)
        B = np.array([vehicle.B for vehicle in list_vehicles])
        # Create the dictionary directly with NumPy arrays
        vehicle_dict_vecs = {
            "Quality_a_t": quality_a_t,
            "Eff_omega_a_t": eff_omega_a_t,
            "price": price,
            "production_emissions": production_emissions,
            "fuel_cost_c": fuel_cost_c,
            "e_t": e_t,
            "transportType": transport_type,
            "rebate": rebate_vec,
            "delta": delta,
            "B": B
        }

        return vehicle_dict_vecs

    def gen_vehicle_dict_vecs_second_hand(self, list_vehicles):
        """
        Generate attribute arrays for second-hand cars.

        Args:
            list_vehicles (list): List of second-hand vehicle objects.

        Returns:
            dict: Dictionary of vehicle attributes.
        """
        # Extract properties using list comprehensions
        quality_a_t = np.array([vehicle.Quality_a_t for vehicle in list_vehicles])
        eff_omega_a_t = np.array([vehicle.Eff_omega_a_t for vehicle in list_vehicles])
        price = np.array([vehicle.price for vehicle in list_vehicles])
        fuel_cost_c = np.array([vehicle.fuel_cost_c for vehicle in list_vehicles])
        e_t = np.array([vehicle.e_t for vehicle in list_vehicles])
        l_a_t = np.array([vehicle.L_a_t for vehicle in list_vehicles])
        transport_type = np.array([vehicle.transportType for vehicle in list_vehicles])
        delta = np.array([vehicle.delta for vehicle in list_vehicles])
        used_rebate_vec = np.where(transport_type == 3, self.used_rebate_calibration + self.used_rebate, 0)
        B = np.array([vehicle.B for vehicle in list_vehicles])
        # Create the dictionary directly with NumPy arrays
        vehicle_dict_vecs = {
            "Quality_a_t": quality_a_t,
            "Eff_omega_a_t": eff_omega_a_t,
            "price": price,
            "fuel_cost_c": fuel_cost_c,
            "e_t": e_t,
            "L_a_t": l_a_t,
            "transportType": transport_type,
            "used_rebate": used_rebate_vec,
            "delta": delta,
            "B":B
        }

        return vehicle_dict_vecs

    def vectorised_calculate_utility_second_hand_cars(self, vehicle_dict_vecs, beta_vec, gamma_vec, second_hand_merchant_offer_price, d_vec, nu_vec):
        """
        Compute user utilities for second-hand car options.

        Returns:
            np.ndarray: Utility matrix for second-hand options.
        """
        price_difference_raw = vehicle_dict_vecs["price"][:, np.newaxis] -  vehicle_dict_vecs["used_rebate"][:, np.newaxis]

        price_difference = np.maximum(0, price_difference_raw) - second_hand_merchant_offer_price

        price_difference_T = price_difference.T

        U_a_i_t_matrix  = -price_difference_T + beta_vec[:, np.newaxis]*vehicle_dict_vecs["Quality_a_t"]**self.alpha + nu_vec[:, np.newaxis]*(vehicle_dict_vecs["B"]*vehicle_dict_vecs["Eff_omega_a_t"]*(1-vehicle_dict_vecs["delta"])**vehicle_dict_vecs["L_a_t"])**self.zeta - d_vec[:, np.newaxis]*(((1+self.r)*(1-vehicle_dict_vecs["delta"])*(vehicle_dict_vecs["fuel_cost_c"] + gamma_vec[:, np.newaxis]*vehicle_dict_vecs["e_t"]))/(vehicle_dict_vecs["Eff_omega_a_t"]*((1-vehicle_dict_vecs["delta"])**vehicle_dict_vecs["L_a_t"])*(self.r - vehicle_dict_vecs["delta"] - self.r*vehicle_dict_vecs["delta"])))
        
        return U_a_i_t_matrix
    
    def vectorised_calculate_utility_new_cars(self, vehicle_dict_vecs, beta_vec, gamma_vec, second_hand_merchant_offer_price, d_vec, nu_vec):
        """
        Compute user utilities for new car options.

        Returns:
            np.ndarray: Utility matrix for new car options.
        """
        # Calculate price difference, applying rebate only for transportType == 3 (included in rebate calculation)
        price_difference_raw = (vehicle_dict_vecs["price"][:, np.newaxis] - vehicle_dict_vecs["rebate"][:, np.newaxis])  # Apply rebate

        #print("diff", np.maximum(0, price_difference_raw) )
        
        price_difference = np.maximum(0, price_difference_raw) - second_hand_merchant_offer_price

        price_difference_T = price_difference.T

        #U_a_i_t_matrix = ((1+self.r)*(beta_vec[:, np.newaxis]*vehicle_dict_vecs["Quality_a_t"]**self.alpha + nu_vec[:, np.newaxis]*(vehicle_dict_vecs["B"]*vehicle_dict_vecs["Eff_omega_a_t"])**self.zeta))/self.r - price_difference_T - gamma_vec[:, np.newaxis]*vehicle_dict_vecs["production_emissions"] - d_vec[:, np.newaxis]*(((1+self.r)*(1-vehicle_dict_vecs["delta"])*(vehicle_dict_vecs["fuel_cost_c"] + gamma_vec[:, np.newaxis]*vehicle_dict_vecs["e_t"]))/(vehicle_dict_vecs["Eff_omega_a_t"]*(self.r-vehicle_dict_vecs["delta"] - self.r*vehicle_dict_vecs["delta"])))
        U_a_i_t_matrix  = -price_difference_T - gamma_vec[:, np.newaxis]*vehicle_dict_vecs["production_emissions"] + beta_vec[:, np.newaxis]*vehicle_dict_vecs["Quality_a_t"]**self.alpha + nu_vec[:, np.newaxis]*(vehicle_dict_vecs["B"]*vehicle_dict_vecs["Eff_omega_a_t"])**self.zeta - d_vec[:, np.newaxis]*(((1+self.r)*(1-vehicle_dict_vecs["delta"])*(vehicle_dict_vecs["fuel_cost_c"] + gamma_vec[:, np.newaxis]*vehicle_dict_vecs["e_t"]))/(vehicle_dict_vecs["Eff_omega_a_t"]*(self.r - vehicle_dict_vecs["delta"] - self.r*vehicle_dict_vecs["delta"])))

        return U_a_i_t_matrix# Shape: (num_individuals, num_vehicles)
    
    def prep_counters(self):
        """
        Initialize all counters and tracking variables used during a simulation timestep.
        
        This includes emissions, utility, car attributes, purchase counts, and more.
        """
        self.users_driving_emissions_vec = np.zeros(self.num_individuals)
        self.users_distance_vec = np.zeros(self.num_individuals)
        self.users_utility_vec  = np.zeros(self.num_individuals)
        self.users_transport_type_vec  = np.full((self.num_individuals), np.nan)

        self.users_distance_vec_EV = np.full((self.num_individuals), np.nan)
        self.users_distance_vec_ICE = np.full((self.num_individuals), np.nan)
        #variable to track
    
        self.total_driving_emissions = 0
        self.total_driving_emissions_ICE = 0
        self.total_driving_emissions_EV = 0
        self.total_production_emissions = 0
        self.total_production_emissions_ICE = 0
        self.total_production_emissions_EV = 0
        self.total_utility = 0
        self.total_utility_bottom = 0
        self.total_utility_top = 0
        self.total_distance_travelled = 0
        self.total_distance_travelled_ICE = 0
        self.ICE_users = 0 
        self.EV_users = 0
        
        self.new_ICE_cars_bought = 0
        self.new_EV_cars_bought = 0
  
        self.second_hand_users = 0
        self.quality_vals = []
        self.efficiency_vals = []
        self.production_cost_vals = []
        self.quality_vals_ICE = []
        self.efficiency_vals_ICE = []
        self.production_cost_vals_ICE = []
        self.quality_vals_EV = []
        self.efficiency_vals_EV = []
        self.production_cost_vals_EV = []
        self.new_cars_bought = 0
        self.car_ages = []

        self.cars_cum_distances_driven = []
        self.cars_cum_driven_emissions = []
        self.cars_cum_emissions = []

        self.car_prices_sold_new = []
        self.car_prices_sold_second_hand = []

        self.keep_car = 0
        self.buy_new_car = 0
        self.buy_second_hand_car = 0

        self.second_hand_merchant_price_paid = []
        self.battery_EV = []
    
    def update_counters(self, person_index, vehicle_chosen, driven_distance, utility):
        """
        Update individual- and aggregate-level counters for emissions, utility, distance, and vehicle attributes.

        Args:
            person_index (int): Index of the user.
            vehicle_chosen (object): Vehicle object that the user ended up with.
            driven_distance (float): Distance driven by the user this timestep.
            utility (float): Utility derived from the choice.
        """
        
        #ADD TOTAL EMISSIONS     
        car_driving_emissions = (driven_distance/vehicle_chosen.Eff_omega_a_t)*vehicle_chosen.e_t 
        self.users_driving_emissions_vec[person_index] = car_driving_emissions

        self.users_distance_vec[person_index] = driven_distance
        self.users_utility_vec[person_index] =  utility
        self.users_transport_type_vec[person_index] = vehicle_chosen.transportType
        
        if vehicle_chosen.scenario == "new_car":  
            self.new_cars_bought +=1
            self.total_production_emissions += vehicle_chosen.emissions
            if vehicle_chosen.transportType == 2:
                self.new_ICE_cars_bought +=1
                self.total_production_emissions_ICE += vehicle_chosen.emissions
            else:
                self.new_EV_cars_bought +=1
                self.total_production_emissions_EV += vehicle_chosen.emissions
            
        
        self.total_driving_emissions += car_driving_emissions 

        if isinstance(vehicle_chosen, PersonalCar):
            vehicle_chosen.total_distance += driven_distance
            self.cars_cum_distances_driven.append(vehicle_chosen.total_distance)
            vehicle_chosen.total_driving_emmissions += car_driving_emissions
            self.cars_cum_driven_emissions.append(vehicle_chosen.total_driving_emmissions)
            vehicle_chosen.total_emissions += car_driving_emissions
            self.cars_cum_emissions.append(vehicle_chosen.total_emissions)

        self.car_ages.append(vehicle_chosen.L_a_t)
        self.quality_vals.append(vehicle_chosen.Quality_a_t)#done here for efficiency
        self.efficiency_vals.append(vehicle_chosen.Eff_omega_a_t)
        self.production_cost_vals.append(vehicle_chosen.ProdCost_t)

        if vehicle_chosen.transportType == 2:#ICE 
            self.users_distance_vec_ICE[person_index] = driven_distance
            self.quality_vals_ICE.append(vehicle_chosen.Quality_a_t)#done here for efficiency
            self.efficiency_vals_ICE.append(vehicle_chosen.Eff_omega_a_t)
            self.production_cost_vals_ICE.append(vehicle_chosen.ProdCost_t)
            self.total_driving_emissions_ICE += car_driving_emissions 
            self.total_distance_travelled_ICE += driven_distance
            self.ICE_users += 1
        else:#EV
            self.users_distance_vec_EV[person_index] = driven_distance
            self.quality_vals_EV.append(vehicle_chosen.Quality_a_t)#done here for efficiency
            self.efficiency_vals_EV.append(vehicle_chosen.Eff_omega_a_t)
            self.production_cost_vals_EV.append(vehicle_chosen.ProdCost_t)
            self.battery_EV.append(vehicle_chosen.B)
            self.total_driving_emissions_EV += car_driving_emissions 
            self.EV_users += 1
            
        self.total_utility +=  utility
        if self.beta_vec[person_index] < self.beta_median:
            self.total_utility_bottom += utility
        
        if self.beta_vec[person_index] > self.beta_rich:
            self.total_utility_top += utility

        self.total_distance_travelled += driven_distance
            
        if isinstance(vehicle_chosen, PersonalCar):
            self.second_hand_users +=1
      
    def set_up_time_series_social_network(self):
        """
        Initialize all time series data structures for tracking the evolution of the system over time.
        """
        self.emissions_flow_history = []
        self.history_utility_components = []
        self.history_max_index_segemnt = []
        
        self.history_driving_emissions = []
        self.history_driving_emissions_ICE = []
        self.history_driving_emissions_EV = []
        self.history_production_emissions = []
        self.history_production_emissions_ICE = []
        self.history_production_emissions_EV = []
        self.history_total_emissions = []
        self.history_total_utility = []
        self.history_total_utility_bottom = []
        self.history_total_utility_top = []
        self.history_total_distance_driven = []
        self.history_total_distance_driven_ICE = []
        self.history_ev_adoption_rate = []
        self.history_ev_adoption_rate_top = []
        self.history_ev_adoption_rate_bottom = []
        self.history_consider_ev_rate = []
        self.history_consider_ev = []
        self.history_ICE_users = []
        self.history_EV_users = []
        self.history_second_hand_users = []
        self.history_new_ICE_cars_bought = []
        self.history_new_EV_cars_bought = []
        # New history attributes for vehicle attributes
        self.history_quality = []
        self.history_efficiency = []
        self.history_production_cost = []

        self.history_quality_ICE = []
        self.history_efficiency_ICE  = []
        self.history_production_cost_ICE  = []

        self.history_quality_EV = []
        self.history_efficiency_EV = []
        self.history_production_cost_EV = []
        self.history_battery_EV = []

        self.history_attributes_EV_cars_on_sale_all_firms = []
        self.history_attributes_ICE_cars_on_sale_all_firms = []
        self.history_second_hand_bought = []
        self.history_new_car_bought = []
        self.history_car_age = []
        self.history_mean_car_age = []

        self.history_cars_cum_distances_driven = []
        self.history_cars_cum_driven_emissions = []
        self.history_cars_cum_emissions = []

        #INDIVIDUALS LEVEL DATA
        self.history_driving_emissions_individual = []
        self.history_distance_individual = []
        self.history_utility_individual = []
        self.history_transport_type_individual = []

        self.history_distance_individual_ICE = []
        self.history_distance_individual_EV = []
        self.history_count_buy = []

        #self.history_quality_index = []
        self.history_mean_price = []
        self.history_median_price = []

        self.history_mean_price_ICE_EV = []
        self.history_median_price_ICE_EV = []

        self.history_lower_percentile_price_ICE_EV = []
        self.history_upper_percentile_price_ICE_EV = []

        self.history_car_prices_sold_new = []
        self.history_car_prices_sold_second_hand = []

        self.history_quality_users_raw_adjusted = []

        self.history_second_hand_merchant_price_paid = []


        self.history_mean_efficiency_vals_EV = []
        self.history_mean_efficiency_vals_ICE = []
        self.history_drive_min_num = []

        self.history_mean_efficiency_vals = []

        self.history_second_hand_merchant_offer_price = []

    def save_timeseries_data_social_network(self):
        """
        Save current timestep's data to time series history for analysis and visualization.

        Tracks emissions, utility, prices, car attributes, EV adoption, and more.
        """
        self.history_second_hand_merchant_offer_price.append(self.second_hand_merchant_offer_price)

        self.history_count_buy.append([self.keep_car, self.buy_new_car, self.buy_second_hand_car])

        self.history_drive_min_num.append(self.drive_min_num/self.num_individuals)


        mean_price_new = np.mean([vehicle.price for vehicle in self.new_cars])
        median_price_new = np.median([vehicle.price for vehicle in self.new_cars])

        prices_ICE = [vehicle.price for vehicle in self.new_cars if vehicle.transportType == 2]
        #prices_EV = [np.maximum(0, vehicle.price - (self.rebate_calibration + self.rebate))  for vehicle in self.new_cars if vehicle.transportType == 3]
        prices_EV = [vehicle.price  for vehicle in self.new_cars if vehicle.transportType == 3]

        if prices_ICE:
            mean_price_new_ICE = np.mean(prices_ICE)
            median_price_new_ICE = np.median(prices_ICE)
            lower_price_new_ICE = np.percentile(prices_ICE,25)
            upper_price_new_ICE = np.percentile(prices_ICE,75)
        else:
            mean_price_new_ICE = np.nan
            median_price_new_ICE = np.nan
            lower_price_new_ICE = np.nan
            upper_price_new_ICE = np.nan
        
        if prices_EV:
            mean_price_new_EV = np.mean(prices_EV)
            median_price_new_EV = np.median(prices_EV)
            lower_price_new_EV = np.percentile(prices_EV,25)
            upper_price_new_EV = np.percentile(prices_EV,75)
        else:
            mean_price_new_EV = np.nan
            median_price_new_EV = np.nan
            lower_price_new_EV = np.nan
            upper_price_new_EV = np.nan

        if self.second_hand_cars:
            prices_second_hand_ICE = [vehicle.price for vehicle in self.second_hand_cars if vehicle.transportType == 2]
            #prices_second_hand_EV = [np.maximum(0, vehicle.price - (self.used_rebate_calibration + self.used_rebate)) for vehicle in self.second_hand_cars if vehicle.transportType == 3]
            prices_second_hand_EV = [vehicle.price for vehicle in self.second_hand_cars if vehicle.transportType == 3]

            if prices_second_hand_ICE:
                mean_price_second_hand_ICE = np.mean(prices_second_hand_ICE)
                median_price_second_hand_ICE = np.median(prices_second_hand_ICE)
                lower_price_second_hand_ICE = np.percentile(prices_second_hand_ICE,25)
                upper_price_second_hand_ICE = np.percentile(prices_second_hand_ICE,75)
            else:
                mean_price_second_hand_ICE = np.nan
                median_price_second_hand_ICE = np.nan
                lower_price_second_hand_ICE = np.nan
                upper_price_second_hand_ICE = np.nan
            
            if prices_second_hand_EV:
                mean_price_second_hand_EV = np.mean(prices_second_hand_EV)
                median_price_second_hand_EV = np.median(prices_second_hand_EV)
                lower_price_second_hand_EV = np.percentile(prices_second_hand_EV,25)
                upper_price_second_hand_EV = np.percentile(prices_second_hand_EV,75)
            else:
                mean_price_second_hand_EV = np.nan
                median_price_second_hand_EV = np.nan
                lower_price_second_hand_EV = np.nan
                upper_price_second_hand_EV = np.nan

        else:
            mean_price_second_hand_ICE = np.nan
            median_price_second_hand_ICE = np.nan
            mean_price_second_hand_EV = np.nan
            median_price_second_hand_EV = np.nan

            lower_price_second_hand_ICE = np.nan
            lower_price_second_hand_EV = np.nan
            upper_price_second_hand_ICE = np.nan
            upper_price_second_hand_EV = np.nan
            

        if self.second_hand_cars:
            mean_price_second_hand = np.mean([vehicle.price for vehicle in self.second_hand_cars])
            median_price_second_hand = np.median([vehicle.price for vehicle in self.second_hand_cars])
        else:
            mean_price_second_hand = np.nan#NO SECOND HAND CARS
            median_price_second_hand = np.nan#NO SECOND HAND CARS


        self.history_mean_price.append([mean_price_new, mean_price_second_hand])
        self.history_median_price.append([median_price_new, median_price_second_hand])

        self.history_mean_price_ICE_EV.append([(mean_price_new_ICE, mean_price_new_EV), (mean_price_second_hand_ICE,mean_price_second_hand_EV)])
        self.history_median_price_ICE_EV.append([(median_price_new_ICE, median_price_new_EV), (median_price_second_hand_ICE,median_price_second_hand_EV)])

        self.history_lower_percentile_price_ICE_EV.append([(lower_price_new_ICE, lower_price_new_EV), (lower_price_second_hand_ICE,lower_price_second_hand_EV)])
        self.history_upper_percentile_price_ICE_EV.append([(upper_price_new_ICE, upper_price_new_EV), (upper_price_second_hand_ICE,upper_price_second_hand_EV)])

        self.history_driving_emissions_individual.append(self.users_driving_emissions_vec)
        
        self.history_distance_individual.append(self.users_distance_vec)
        self.history_utility_individual.append(self.users_utility_vec)
        self.history_transport_type_individual.append(self.users_transport_type_vec)

        self.history_distance_individual_ICE.append(self.users_distance_vec_ICE)
        self.history_distance_individual_EV.append(self.users_distance_vec_EV)

        self.history_new_ICE_cars_bought.append(self.new_ICE_cars_bought)
        self.history_new_EV_cars_bought.append(self.new_EV_cars_bought)


        #self.history_max_index_segemnt.append(self.max_index_segemnt)

        #SUMS
        self.history_driving_emissions.append(self.total_driving_emissions)
        self.history_driving_emissions_ICE.append(self.total_driving_emissions_ICE)
        self.history_driving_emissions_EV.append(self.total_driving_emissions_EV)
        self.history_production_emissions.append(self.total_production_emissions)
        self.history_production_emissions_ICE.append(self.total_production_emissions_ICE)
        self.history_production_emissions_EV.append(self.total_production_emissions_EV)
        self.history_total_emissions.append(self.total_production_emissions + self.total_driving_emissions)
        self.history_total_utility.append(self.total_utility)
        self.history_total_utility_bottom.append(self.total_utility_bottom/self.num_poor)
        self.history_total_utility_top.append(self.total_utility_top/self.num_rich)
        self.history_total_distance_driven.append(self.total_distance_travelled)
        self.history_total_distance_driven_ICE.append(self.total_distance_travelled_ICE)
        self.history_ev_adoption_rate.append(np.mean(self.ev_adoption_vec))

        ev_adoption_rate_top = np.mean([i for i, j in zip(self.ev_adoption_vec, self.beta_vec) if j > self.beta_rich])
        ev_adoption_rate_bottom = np.mean([i for i, j in zip(self.ev_adoption_vec, self.beta_vec) if j < self.beta_median])
        self.history_ev_adoption_rate_top.append(ev_adoption_rate_top)
        self.history_ev_adoption_rate_bottom.append(ev_adoption_rate_bottom)

        self.history_consider_ev_rate.append(np.mean(self.consider_ev_vec))
        self.history_consider_ev.append(self.consider_ev_vec)
        self.history_ICE_users.append(self.ICE_users)
        self.history_EV_users.append(self.EV_users)
        self.history_second_hand_users.append(self.second_hand_users)
        self.history_second_hand_bought.append(self.second_hand_bought)
        self.history_new_car_bought.append(self.new_cars_bought)
        
        self.history_quality.append(self.quality_vals)
        self.history_efficiency.append(self.efficiency_vals)
        self.history_production_cost.append(self.production_cost_vals)

        self.history_mean_efficiency_vals.append(np.mean(self.efficiency_vals))

        if self.quality_vals_ICE:
            self.history_quality_ICE.append(self.quality_vals_ICE)
            self.history_efficiency_ICE.append(self.efficiency_vals_ICE)
            self.history_production_cost_ICE.append(self.production_cost_vals_ICE)

            self.history_mean_efficiency_vals_ICE.append(np.mean(self.efficiency_vals_ICE))
        else:
            self.history_quality_ICE.append([np.nan])
            self.history_efficiency_ICE.append([np.nan])
            self.history_production_cost_ICE.append([np.nan])
            self.history_mean_efficiency_vals_ICE.append([np.nan])

        if self.quality_vals_EV:
            self.history_quality_EV.append(self.quality_vals_EV)
            self.history_efficiency_EV.append(self.efficiency_vals_EV)
            self.history_production_cost_EV.append(self.production_cost_vals_EV)
            self.history_battery_EV.append(self.battery_EV)
            self.history_mean_efficiency_vals_EV.append(np.mean(self.efficiency_vals_EV))
        else:
            self.history_quality_EV.append([np.nan])
            self.history_efficiency_EV.append([np.nan])
            self.history_production_cost_EV.append([np.nan])
            self.history_battery_EV.append([np.nan])
            self.history_mean_efficiency_vals_EV.append([np.nan])

        data_ev = [[vehicle.Quality_a_t, vehicle.Eff_omega_a_t, vehicle.ProdCost_t]  for vehicle in self.all_vehicles_available if vehicle.transportType == 3]
        data_ice = [[vehicle.Quality_a_t ,vehicle.Eff_omega_a_t, vehicle.ProdCost_t]  for vehicle in self.all_vehicles_available if vehicle.transportType == 2]

        self.history_attributes_EV_cars_on_sale_all_firms.append(data_ev)
        self.history_attributes_ICE_cars_on_sale_all_firms.append(data_ice)

        self.history_car_age.append(self.car_ages)
        self.history_mean_car_age.append(np.mean(self.car_ages))

        self.history_cars_cum_distances_driven.append(self.cars_cum_distances_driven)
        self.history_cars_cum_driven_emissions.append(self.cars_cum_driven_emissions)
        self.history_cars_cum_emissions.append(self.cars_cum_emissions)

        self.history_car_prices_sold_new.append(self.car_prices_sold_new)
        self.history_car_prices_sold_second_hand.append(self.car_prices_sold_second_hand)

        self.history_quality_users_raw_adjusted.append([(car.Quality_a_t,car.Quality_a_t*(1-car.delta)**car.L_a_t ) for car in self.current_vehicles])

        self.history_second_hand_merchant_price_paid.append(self.second_hand_merchant_price_paid)
    
    def update_emisisons(self, vehicle_chosen, driven_distance):      
        """
        Update cumulative and flow emissions based on the selected vehicle and driven distance.

        Args:
            vehicle_chosen (object): The vehicle driven by the user.
            driven_distance (float): Distance the vehicle was driven this timestep.
        """
        emissions_flow = (driven_distance/vehicle_chosen.Eff_omega_a_t)*vehicle_chosen.e_t
        self.emissions_cumulative += emissions_flow
        self.emissions_cumulative_driving += emissions_flow
        self.emissions_flow += emissions_flow

        if vehicle_chosen.scenario == "new_car":  #if its a new car add emisisons
            self.emissions_cumulative += vehicle_chosen.emissions
            self.emissions_cumulative_production += vehicle_chosen.emissions
            self.emissions_flow += vehicle_chosen.emissions

    def update_EV_stock(self):
        """
        Update the proportion of users currently owning electric vehicles (EVs), and append it to the history.
        """
        self.EV_users_count = sum(1 if car.transportType == 3 else 0 for car in  self.current_vehicles)
        self.history_prop_EV.append(self.EV_users_count/self.num_individuals)

    def calc_price_mean_max_min(self):
        """
        Compute mean, min, and max prices among new cars.

        Returns:
            tuple: (mean_price, min_price, max_price)
        """
        prices = [car.price for car in self.new_cars]
        price_mean =  np.mean(prices)
        price_min =  np.min(prices)
        price_max = np.max(prices)

        return price_mean, price_min, price_max
    
    def calc_mean_car_age(self):
        """
        Calculate the average age of cars currently owned.

        Returns:
            float: Mean car age.
        """
        mean_car_age  = np.mean([car.L_a_t for car in self.current_vehicles])
        return mean_car_age

    def update_prices_and_emissions_intensity(self):
        """
        Update the fuel cost and emissions intensity of currently owned vehicles.
        """
        for car in self.current_vehicles:
            if car.transportType == 2:#ICE
                car.fuel_cost_c = self.gas_price
            elif car.transportType == 3:
                car.fuel_cost_c = self.electricity_price
                car.e_t = self.electricity_emissions_intensity  
        
    def next_step(self, carbon_price, second_hand_cars,new_cars, gas_price, electricity_price, electricity_emissions_intensity, rebate, used_rebate, electricity_price_subsidy_dollars, rebate_calibration, used_rebate_calibration):
        """
        Advance the simulation by one time step:
            - Update external parameters and policies.
            - Update vehicle attributes and user decisions.
            - Track emissions and adoption metrics.

        Args:
            carbon_price (float): Price of carbon emissions.
            second_hand_cars (list): Available second-hand cars.
            new_cars (list): Available new car models.
            gas_price (float): Current gasoline price.
            electricity_price (float): Current electricity price.
            electricity_emissions_intensity (float): Emissions per unit electricity.
            rebate (float): Rebate offered for new EV purchases.
            used_rebate (float): Rebate offered for used EV purchases.
            electricity_price_subsidy_dollars (float): Direct subsidy on electricity cost.
            rebate_calibration (float): Calibration offset for new EV rebate.
            used_rebate_calibration (float): Calibration offset for used EV rebate.

        Returns:
            tuple: (consider_ev_vec, new_bought_vehicles) indicating user intention and new purchases.
        """

        self.carbon_price = carbon_price
        self.gas_price =  gas_price
        self.electricity_price = electricity_price
        self.electricity_emissions_intensity = electricity_emissions_intensity
        self.rebate = rebate
        self.used_rebate = used_rebate
        self.rebate_calibration = rebate_calibration
        self.used_rebate_calibration = used_rebate_calibration
        self.electricity_price_subsidy_dollars = electricity_price_subsidy_dollars

        #update new tech and prices
        self.second_hand_cars, self.new_cars = second_hand_cars, new_cars
        self.all_vehicles_available = self.new_cars + self.second_hand_cars#ORDER IS VERY IMPORTANT

        self.update_prices_and_emissions_intensity()#UPDATE: the prices and emissions intensities of cars which are currently owned

        self.current_vehicles = self.update_VehicleUsers()
        
        self.consider_ev_vec, self.ev_adoption_vec = self.calculate_ev_adoption(ev_type=3)#BASED ON CONSUMPTION PREVIOUS TIME STEP

        self.update_EV_stock()
        
        self.t_social_network +=1
        
        return self.consider_ev_vec, self.new_bought_vehicles #self.chosen_vehicles instead of self.current_vehicles as firms can count pofits