# imports
import numpy as np
import networkx as nx
import numpy.typing as npt
import scipy.sparse as sp
import numpy as np
from package.model.personalCar import PersonalCar
from package.model.VehicleUser import VehicleUser
from package.model.carModel import CarModel
from package.model.secondHandMerchant import _attr_array

# Safety bound for _lifecycle_cost_term — see that method's docstring. Far
# above any value the model's realistic (non-extreme-policy) parameter
# ranges could ever legitimately produce.
MAX_LIFECYCLE_COST_TERM = 1e8

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

        # Mean age (months) of the owned fleet, recorded every step by
        # update_mean_car_age() independently of save_timeseries_data_state,
        # because the calibration matches against it. Deliberately NOT the same
        # quantity as history_mean_car_age, which is gated behind the save path
        # and averages only the vehicles CHOSEN this step (i.e. switchers), not
        # the whole fleet.
        self.history_mean_car_age_fleet = []

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

        # Always-on (unlike users_utility_vec/history_utility_individual, which are
        # gated behind save_timeseries_data_state): one (num_individuals,) array per
        # timestep, raw utility for every agent (switchers + non-switchers), whole
        # simulation. Reset to [] at the start of the future/policy period in
        # controller.setup_continued_run_future(), same as utility_cumulative is
        # reset to 0 there. Feeds the BAU-relative log-utility surrogate metric
        # (see sampling.compute_log_utility_metric) — entirely independent of the
        # existing prep_counters()/users_utility_vec instrumentation, so it changes
        # no existing behaviour.
        self.history_utility_individual_always = []

        self.init_network_settings(parameters_social_network)

        self.random_state = parameters_social_network["random_state"]
        self.seed_inputs = parameters_social_network["seed_inputs"]

        self.mu =  parameters_vehicle_user["mu"]
        self.r = parameters_vehicle_user["r"]
        self.kappa = parameters_vehicle_user["kappa"]

        # Backward compatible: absent => naive/permanent expectations (old behaviour).
        self.forward_looking_expectations = parameters_social_network.get("forward_looking_expectations", False)

        # Generate a list of indices and shuffle them
        self.user_indices = np.arange(self.num_individuals)

        # Efficient user list creation with list comprehension
        self.vehicleUsers_list = [VehicleUser(user_id=i) for i in range(self.num_individuals)]

        # Create network and calculate initial emissions
        self.adjacency_matrix, self.network = self.create_network()
        self.network_density = nx.density(self.network)

        self.include_self_social_state = parameters_social_network.get("include_self_social_state", False)
        if self.include_self_social_state:
            print("YOOOO")
            np.fill_diagonal(self.adjacency_matrix, 1)
            self.sparse_adjacency_matrix = sp.csr_matrix(self.adjacency_matrix)

        self.weighting_matrix = self._normlize_matrix(self.sparse_adjacency_matrix)#INTRODUCE HOMOPHILY INTO THE NETWORK BY ASSORTING BY BETA WITHING GROUPS

        #Assume nobody adopts EV at the start, THIS MAY BE AN ISSUE
        self.consider_ev_vec = np.zeros(self.num_individuals).astype(np.int8)

        self.current_vehicles = self.set_init_cars_selection(parameters_social_network)

        self._build_cv_cache()

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
        
        self.vehicle_type_vec = self._cv_cache["transportType"]

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
        # full_CV_utility_vec is now computed unconditionally (previously only
        # under save_timeseries_data_state) — it feeds the always-on
        # history_utility_individual_always buffer below. prep_counters() itself
        # stays gated exactly as before; it only resets state used by the
        # optional full history tracking, which this doesn't touch.
        # build_matrix=False: only the vector is used here (the caller discards
        # the matrix), and building it means allocating and -inf-filling a
        # num_individuals x num_individuals matrix every single timestep just to
        # write its diagonal — ~20% of total runtime at num_individuals=3000,
        # for a result that is immediately thrown away.
        __, full_CV_utility_vec = self.generate_utilities_current(CV_vehicle_dict_vecs, self.beta_vec, self.gamma_vec, self.d_vec, self.nu_vec, build_matrix=False)
        if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
            self.prep_counters()


        # Vectorised emissions + policy distortion for non-switchers
        ns_eff  = CV_vehicle_dict_vecs["Eff_omega_a_t"][non_switcher_indices]
        ns_e_t  = CV_vehicle_dict_vecs["e_t"][non_switcher_indices]
        ns_d    = self.d_vec[non_switcher_indices]
        ns_type = CV_vehicle_dict_vecs["transportType"][non_switcher_indices]

        ns_driving_emissions = (ns_d / ns_eff) * ns_e_t
        total_ns_emit = float(ns_driving_emissions.sum())
        self.emissions_cumulative        += total_ns_emit
        self.emissions_cumulative_driving += total_ns_emit
        self.emissions_flow              += total_ns_emit

        ev_mask = (ns_type == 3)
        elec_sub_total = float(np.where(ev_mask,  (self.electricity_price_subsidy_dollars * ns_d) / ns_eff, 0.0).sum())
        carbon_total   = float(np.where(~ev_mask, (self.carbon_price * ns_e_t * ns_d) / ns_eff, 0.0).sum())
        self.policy_distortion     += elec_sub_total + carbon_total
        self.net_policy_distortion += -elec_sub_total + carbon_total

        # Always-on per-timestep utility buffer (switchers + non-switchers) — see
        # history_utility_individual_always docstring in __init__. Independent of
        # save_timeseries_data_state and of the users_utility_vec/prep_counters
        # instrumentation above; doesn't affect anything else in this method.
        step_utility_buffer = np.zeros(self.num_individuals)
        step_utility_buffer[non_switcher_indices] = full_CV_utility_vec[non_switcher_indices]

        # The save-state test is loop-invariant, so it is evaluated once here
        # rather than once per agent per timestep (two attribute lookups and a
        # modulo each). Splitting on it also leaves the common calibration path
        # -- saving off -- as a bare ageing loop over Python ints.
        vehicle_users = self.vehicleUsers_list
        saving_this_step = bool(self.save_timeseries_data_state) and (self.t_social_network % self.compression_factor_state == 0)

        if saving_this_step:
            for person_index in non_switcher_indices.tolist():
                user = vehicle_users[person_index]
                user.vehicle.update_timer_L_a_t()

                self.keep_car += 1
                utility = full_CV_utility_vec[person_index]
                driven_distance = self.d_vec[person_index]
                self.update_counters(person_index, user.vehicle, driven_distance, utility)
        else:
            for person_index in non_switcher_indices.tolist():
                vehicle_users[person_index].vehicle.update_timer_L_a_t()

        self._cv_cache["L_a_t"][non_switcher_indices] += 1

                
        ##################################################################
        #SWITCHERS

        # 2) Shuffle only that subset of user indices
        shuffle_indices = self.random_state.permutation(switcher_indices)

        self.NC_vehicle_dict_vecs = self.gen_vehicle_dict_vecs_new_cars(self.new_cars)

        #THIS CAN BE DONE FOR THE SUBSET OF USERS
        CV_filtered_vechicles_dicts, CV_filtered_vehicles = self.filter_vehicle_dict_for_switchers(CV_vehicle_dict_vecs, self.current_vehicles, switcher_indices)
        utilities_current_matrix, __ = self.generate_utilities_current(CV_filtered_vechicles_dicts, self.sub_beta_vec, self.sub_gamma_vec, self.sub_d_vec, self.sub_nu_vec)

        self.second_hand_merchant_offer_price = self.calc_offer_prices_heursitic(self.NC_vehicle_dict_vecs, CV_filtered_vechicles_dicts, CV_filtered_vehicles)

        # pass those indices to generate_utilities. It allocates the full
        # (switchers x new+second_hand+current) matrix and fills the buy-side
        # blocks in place, so only the small current-car block is copied here.
        self.utilities_matrix_switchers, buying_vehicles_list, num_buying_columns = self.generate_utilities(self.sub_beta_vec, self.sub_gamma_vec, self.second_hand_merchant_offer_price, self.sub_d_vec, self.sub_nu_vec, extra_columns=utilities_current_matrix.shape[1])

        self.utilities_matrix_switchers[:, num_buying_columns:] = utilities_current_matrix

        # Combine the list of vehicles
        available_and_current_vehicles_list = buying_vehicles_list + CV_filtered_vehicles# ITS CURRENT VEHICLES AND NOT FILTERED VEHCILES AS THE SHUFFLING INDEX DOENST ACCOUNT FOR THE FILTERING

        utilities_kappa = self.masking_options(self.utilities_matrix_switchers, available_and_current_vehicles_list, self.consider_ev_vec[switcher_indices])

        #########################################################################################################################
        # Create a mapping from global to reduced indices. Built over plain
        # Python ints (.tolist()) rather than NumPy scalars: the keys compare
        # and hash the same either way, but boxing a NumPy scalar for every
        # dict operation and every list index below is not free.
        switcher_indices_list = switcher_indices.tolist()
        global_to_reduced = {global_idx: local_idx for local_idx, global_idx in enumerate(switcher_indices_list)}
        # Translate shuffle_indices to reduced indices
        shuffle_indices_reduced = [global_to_reduced[idx] for idx in shuffle_indices.tolist()]

        #########################################################################################################################

        # Pre-generate all uniform draws for the multinomial choices this step
        _u_draws = self.random_state.rand(num_switchers)

        for i, reduced_index in enumerate(shuffle_indices_reduced):
            # Map the reduced index back to the global index
            global_index = switcher_indices_list[reduced_index]

            user = vehicle_users[global_index]  # Use the global index to access the user
            vehicle_chosen, user_vehicle, vehicle_chosen_index, utilities_kappa = self.user_chooses(
                global_index, user, available_and_current_vehicles_list, utilities_kappa, reduced_index, index_current_cars_start, _u_draws[i]
            )
            user_vehicle_list[global_index] = user_vehicle  # Update using the global index
            self._update_cv_cache_row(global_index, user_vehicle)

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

            # Single scalar fetch. The chained [row][col] form built a throwaway
            # 1-D view object of the whole row first, once per switcher.
            utility = self.utilities_matrix_switchers[reduced_index, vehicle_chosen_index]
            self.utility_cumulative += utility
            step_utility_buffer[global_index] = utility

            if saving_this_step:

                self.update_counters(global_index, vehicle_chosen, driven_distance, utility)

        # Drop the cars sold out of the merchant's stock during the loop above.
        # remove_car() only marked them; see SecondHandMerchant.compact_stock.
        self.second_hand_merchant.compact_stock()

        if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
            self.emissions_flow_history.append(self.emissions_flow)

        self.history_utility_individual_always.append(step_utility_buffer)

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

        # Get the prices of the closest first-hand cars. The EV rebate only
        # exists on EVs (see the transportType == 3 gate on rebate_vec used for
        # the buy-side utility), so it may only be netted off an EV anchor
        # price. Matching is on (Quality, Efficiency, B), and B separates the
        # drivetrains (fuel tank vs battery), so a used ICE matches an ICE
        # anchor essentially always; deducting the EV rebate there collapsed
        # every ICE trade-in onto the scrap floor and cancelled the subsidy's
        # own effect on turnover.
        matched_is_ev = vehicle_dict_vecs_new_cars["transportType"][closest_idxs] == 3
        rebate_deduction = np.where(matched_is_ev, self.rebate_calibration + self.rebate, 0.0)
        closest_prices = np.maximum(first_hand_prices[closest_idxs] - rebate_deduction, 0)

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

        # A user who considers EVs may pick anything; one who does not may pick
        # only non-EVs. That is exactly `considers_i OR not_ev_j`, which the
        # original built as two full (people x cars) outer products that were
        # then OR-ed -- consider_ev_vec is 0/1, so `== 1` and `== 0` are
        # complementary and the second product only ever contributes on the
        # rows the first one left False. One broadcast gives the same matrix.
        # Kept as bool rather than cast to int: the only consumers are
        # np.where and np.any, both of which read it as a truth value, and the
        # int64 cast made the mask eight times larger than it needs to be.
        ev_mask_matrix = (consider_ev_vec == 1)[:, np.newaxis] | not_ev_vec

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

        if valid_rows.all():
            # Fast path -- the normal case, since any user who does not
            # consider EVs still has every ICE available, so a row is only
            # invalid in the degenerate all-EV-market case. Every row is kept,
            # so the boolean-index copy in, the copy out, and the zero-filled
            # destination are all pure overhead: the same three steps are done
            # in place on the one array instead, over four fewer full-size
            # (switchers x cars) buffers.
            row_max_utilities = np.max(masked_utilities, axis=1, keepdims=True)
            np.subtract(masked_utilities, row_max_utilities, out=masked_utilities)
            masked_utilities *= self.kappa
            utilities_kappa = np.exp(masked_utilities, out=masked_utilities)
        else:
            # Step 4: Compute row-wise max only for valid rows for numerical stability
            row_max_utilities = np.max(masked_utilities[valid_rows], axis=1, keepdims=True)

            exp_input = self.kappa * (masked_utilities[valid_rows] - row_max_utilities)

            # Step 6: Exponentiate, directly filling only valid entries
            utilities_kappa = np.zeros_like(utilities_matrix)
            utilities_kappa[valid_rows] = np.exp(exp_input)

        # Consolidate NaN guard here so user_chooses doesn't need per-row checks
        if np.isnan(utilities_kappa).any():
            np.nan_to_num(utilities_kappa, nan=0.0, copy=False)

        return utilities_kappa

    def user_chooses(self, person_index, user, available_and_current_vehicles_list, utilities_kappa, reduced_person_index, index_current_cars_start, u_draw):
        """
        Let a user choose a vehicle based on masked and exponentiated utility values.

        Args:
            person_index (int): Global user index.
            user (VehicleUser): User object.
            available_and_current_vehicles_list (list): All available vehicles.
            utilities_kappa (np.ndarray): Masked and exponentiated utility matrix.
            reduced_person_index (int): Row index in the utility matrix.
            index_current_cars_start (int): Starting index for current vehicles.
            u_draw (float): Pre-generated uniform [0,1) draw for this user's choice.

        Returns:
            tuple: (chosen vehicle, assigned vehicle, index of chosen vehicle, updated utilities matrix)
        """
        # Select individual-specific utilities (NaNs already handled in masking_options)
        individual_specific_util_kappa = utilities_kappa[reduced_person_index]

        # Cumulative sum as unnormalised CDF; sample by scaling u_draw to [0, total].
        # Computed before the switching clause rather than inside it, because
        # the total already answers the clause's question. Every entry is
        # exp() of a non-positive number or an exact zero written by the mask /
        # already-sold paths, so the row is non-negative, and a running sum of
        # non-negative values is zero only if every value is zero. Testing the
        # total is therefore identical to np.any(...) and saves a second full
        # pass over the row for every switcher, every timestep.
        cumsum = np.cumsum(individual_specific_util_kappa)
        total_util_kappa = cumsum[-1]

        #SWICHING_CLAUSE
        if total_util_kappa == 0:#NO car option all zero, THIS SHOULD ONLY REALLY BE TRIGGERED RIGHT AT THE START
            #keep current car
            choice_index = index_current_cars_start + reduced_person_index
            if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
                self.zero_util_count += 1
        else:
            choice_index = int(np.searchsorted(cumsum, u_draw * total_util_kappa, side='right'))
            # Clamp against floating-point overshoot
            if choice_index >= len(available_and_current_vehicles_list):
                choice_index = len(available_and_current_vehicles_list) - 1

        # Record the chosen vehicle
        vehicle_chosen = available_and_current_vehicles_list[choice_index]

        # A car is aged exactly once per month, by whoever held it at the START
        # of that month (see the update_timer_L_a_t() calls at the end of this
        # method and in secondHandMerchant.update_age_stock_prices_and_emissions_intensity).
        # A car bought from the merchant was already aged this step by the
        # merchant -- get_second_hand_cars() runs before update_social_network()
        # in controller.next_step() -- so it must NOT be aged again here.
        acquired_from_merchant = False

        # Handle consequences of the choice
        if user.user_id != vehicle_chosen.owner_id:  # New vehicle, not currently owned
            # Transfer the user's current vehicle to the second-hand merchant, if any
            if isinstance(user.vehicle, PersonalCar):#YOU SELL YOUR CAR?
                # The seller held this car for the whole month, so it is aged
                # here. Previously user.vehicle was rebound to the newly
                # acquired car before the update_timer_L_a_t() call at the end
                # of this method, so a car being sold silently skipped a month
                # -- which is why stocked cars read exactly one month too young
                # for their whole time on the second-hand market.
                user.vehicle.update_timer_L_a_t()
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
                acquired_from_merchant = True#already aged by the merchant this step
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

        # Update the age or timer of the chosen vehicle. Skipped for a car just
        # bought off the merchant, which the merchant already aged this step;
        # a kept car, and a newly produced one, are aged here.
        if not acquired_from_merchant:
            user.vehicle.update_timer_L_a_t()

        return vehicle_chosen, user.vehicle, choice_index, utilities_kappa
    
    def _lifecycle_cost_term(self, vehicle_dict_vecs, gamma_vec, d_vec, age_factor=1):
        """
        Present-value lifecycle fuel/emissions cost term (the d_vec-weighted
        term subtracted in the utility formulas below).

        forward_looking_expectations off: reproduces the paper's naive/
        permanent-policy closed form exactly (Appendix A.4) — bit-for-bit,
        since it's the same expression, just parameterised by age_factor.

        forward_looking_expectations on: uses the forward-looking discounted
        present-value indices (cost_index, emissions_index) computed once per
        timestep in controller.compute_discounted_indices() over the actual
        known future price/policy path, instead of assuming today's level
        persists forever. cost_index and emissions_index are kept separate
        (rather than combined like fuel_cost_c + gamma*e_t) because gamma is
        agent-heterogeneous and must be applied outside the discounted sum.

        The result is clipped at MAX_LIFECYCLE_COST_TERM: with a large ICE
        driving-ban penalty (see controller._unpack_ice_driving_ban_parameters)
        and a tail-end gamma_i (emissions willingness-to-pay) draw, this term
        can otherwise reach magnitudes far beyond anything a realistic
        fuel/carbon cost would ever produce, which previously led to a rare
        but real crash further down the choice pipeline. The clip is set far
        above any value the pre-existing (no-ban) model could ever legitimately
        produce, so it is a no-op for every scenario except this one.
        """
        delta = vehicle_dict_vecs["delta"]
        Eff = vehicle_dict_vecs["Eff_omega_a_t"]
        # Accumulated in place. The first product already allocates the full
        # (switchers x cars) result; every later step rewrites that same buffer
        # instead of allocating another one, which is what the original
        # expression did five times over. Each rewrite is the same IEEE
        # operation on the same operands (only the argument order of the
        # commutative products is flipped), so the result is bit-for-bit what
        # the original expression produced.
        if self.forward_looking_expectations:
            term = gamma_vec*vehicle_dict_vecs["emissions_index"]
            term += vehicle_dict_vecs["cost_index"]
            term /= (Eff*age_factor)
            term *= d_vec
        else:
            term = gamma_vec*vehicle_dict_vecs["e_t"]
            term += vehicle_dict_vecs["fuel_cost_c"]
            term *= (1+self.r)*(1-delta)
            term /= (Eff*age_factor*(self.r - delta - self.r*delta))
            term *= d_vec
        return np.minimum(term, MAX_LIFECYCLE_COST_TERM, out=term)

    def generate_utilities_current(self, vehicle_dict_vecs, beta_vec, gamma_vec, d_vec, nu_vec, build_matrix=True):# -> NDArray:
        """
        Compute utility values for users keeping their current vehicle.

        Args:
            vehicle_dict_vecs (dict): Feature matrix for current vehicles.
            beta_vec, gamma_vec, d_vec, nu_vec (np.ndarray): User-specific parameters.
            build_matrix (bool): If False, skip building the (n x n) diagonal
                matrix and return None in its place. The matrix exists only so
                that the current-car block of the switcher choice matrix is
                diagonal (each user may only pick their OWN current car); the
                all-users call site needs the vector alone, and allocating an
                n x n array there is pure waste.

        Returns:
            tuple: (utility matrix or None, utility vector)
        """

        age_factor = (1-vehicle_dict_vecs["delta"])**vehicle_dict_vecs["L_a_t"]
        U_a_i_t_vec = beta_vec*vehicle_dict_vecs["Quality_a_t"]**self.alpha + nu_vec*(vehicle_dict_vecs["B"]*vehicle_dict_vecs["Eff_omega_a_t"]*age_factor)**self.zeta - self._lifecycle_cost_term(vehicle_dict_vecs, gamma_vec, d_vec, age_factor)

        if not build_matrix:
            return None, U_a_i_t_vec

        # Initialize the matrix with -np.inf
        CV_utilities_matrix = np.full((len(U_a_i_t_vec), len(U_a_i_t_vec)), -np.inf)#its

        # Set the diagonal values
        np.fill_diagonal(CV_utilities_matrix, U_a_i_t_vec)

        return  CV_utilities_matrix, U_a_i_t_vec
    
    def _build_cv_cache(self):
        """Build (or rebuild) the vehicle attribute cache from current_vehicles."""
        vs = self.current_vehicles
        self._cv_cache = {
            "Quality_a_t":   np.array([v.Quality_a_t   for v in vs]),
            "Eff_omega_a_t": np.array([v.Eff_omega_a_t for v in vs]),
            "fuel_cost_c":   np.array([v.fuel_cost_c   for v in vs]),
            "e_t":           np.array([v.e_t           for v in vs]),
            "cost_index":      np.array([v.cost_index      for v in vs]),
            "emissions_index": np.array([v.emissions_index for v in vs]),
            "L_a_t":         np.array([v.L_a_t         for v in vs]),
            "transportType": np.array([v.transportType for v in vs]),
            "delta":         np.array([v.delta         for v in vs]),
            "delta_P":       np.array([v.delta_P       for v in vs]),
            "B":             np.array([v.B             for v in vs]),
        }

    def _update_cv_cache_row(self, idx, vehicle):
        """Update a single row in the vehicle attribute cache after a switcher chooses."""
        c = self._cv_cache
        c["Quality_a_t"][idx]   = vehicle.Quality_a_t
        c["Eff_omega_a_t"][idx] = vehicle.Eff_omega_a_t
        c["fuel_cost_c"][idx]   = vehicle.fuel_cost_c
        c["e_t"][idx]           = vehicle.e_t
        c["cost_index"][idx]      = vehicle.cost_index
        c["emissions_index"][idx] = vehicle.emissions_index
        c["L_a_t"][idx]         = vehicle.L_a_t
        c["transportType"][idx] = vehicle.transportType
        c["delta"][idx]         = vehicle.delta
        c["delta_P"][idx]       = vehicle.delta_P
        c["B"][idx]             = vehicle.B

    def gen_current_vehicle_dict_vecs(self, list_vehicles):
        """
        Return vehicle attribute arrays for all current vehicles.

        Uses the maintained cache (_cv_cache) instead of rebuilding from
        object attributes each step — kept in sync by update_prices_and_emissions_intensity,
        the non-switcher L_a_t increment, and _update_cv_cache_row.
        """
        return self._cv_cache

    def generate_utilities(self, beta_vec, gamma_vec, second_hand_merchant_offer_price, d_vec, nu_vec, extra_columns=0):
        """
        Compute utility values for all switchers over new and second-hand cars.

        Args:
            extra_columns (int): Number of trailing columns to leave
                uninitialised for the caller's current-car block. The buy-side
                blocks are written straight into this one array, so the
                second-hand block -- by far the largest -- is now materialised
                once instead of being built, copied here, and copied again into
                the caller's combined matrix.

        Returns:
            tuple: (utility matrix, list of vehicle objects, number of buy columns)
        """

        # Calculate the total columns needed for utilities
        num_new_cars = len(self.new_cars)
        num_second_hand = len(self.second_hand_cars) if self.second_hand_cars else 0
        total_columns = num_new_cars + num_second_hand

        # Preallocate arrays with the total required columns
        num_individuals_switchers = len(beta_vec)
        utilities_matrix = np.empty((num_individuals_switchers, total_columns + extra_columns))

        # Generate utilities directly into their column blocks
        #self.NC_vehicle_dict_vecs = self.gen_vehicle_dict_vecs_new_cars(self.new_cars)
        self.vectorised_calculate_utility_new_cars(self.NC_vehicle_dict_vecs, beta_vec, gamma_vec, second_hand_merchant_offer_price, d_vec, nu_vec, out=utilities_matrix[:, :num_new_cars])

        if self.second_hand_cars:
            SH_vehicle_dict_vecs = self.gen_vehicle_dict_vecs_second_hand(self.second_hand_cars)
            self.vectorised_calculate_utility_second_hand_cars(SH_vehicle_dict_vecs, beta_vec, gamma_vec, second_hand_merchant_offer_price, d_vec, nu_vec, out=utilities_matrix[:, num_new_cars:total_columns])
            car_options = self.new_cars + self.second_hand_cars
        else:
            car_options = self.new_cars

        return utilities_matrix, car_options, total_columns

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
        owner_ids = _attr_array(list_vehicles, "owner_id", np.int64)

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
        cost_index = np.array([vehicle.cost_index for vehicle in list_vehicles])
        emissions_index = np.array([vehicle.emissions_index for vehicle in list_vehicles])
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
            "cost_index": cost_index,
            "emissions_index": emissions_index,
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
        # Extract properties. _attr_array (np.fromiter over a C-level
        # attrgetter) rather than np.array over a list comprehension -- same
        # values and dtypes, about half the time, and this walks the entire
        # second-hand stock eleven times every timestep.
        quality_a_t = _attr_array(list_vehicles, "Quality_a_t")
        eff_omega_a_t = _attr_array(list_vehicles, "Eff_omega_a_t")
        price = _attr_array(list_vehicles, "price")
        fuel_cost_c = _attr_array(list_vehicles, "fuel_cost_c")
        e_t = _attr_array(list_vehicles, "e_t")
        cost_index = _attr_array(list_vehicles, "cost_index")
        emissions_index = _attr_array(list_vehicles, "emissions_index")
        l_a_t = _attr_array(list_vehicles, "L_a_t", np.int64)
        transport_type = _attr_array(list_vehicles, "transportType", np.int64)
        delta = _attr_array(list_vehicles, "delta")
        used_rebate_vec = np.where(transport_type == 3, self.used_rebate_calibration + self.used_rebate, 0)
        B = _attr_array(list_vehicles, "B")
        # Create the dictionary directly with NumPy arrays
        vehicle_dict_vecs = {
            "Quality_a_t": quality_a_t,
            "Eff_omega_a_t": eff_omega_a_t,
            "price": price,
            "fuel_cost_c": fuel_cost_c,
            "e_t": e_t,
            "cost_index": cost_index,
            "emissions_index": emissions_index,
            "L_a_t": l_a_t,
            "transportType": transport_type,
            "used_rebate": used_rebate_vec,
            "delta": delta,
            "B":B
        }

        return vehicle_dict_vecs

    def vectorised_calculate_utility_second_hand_cars(self, vehicle_dict_vecs, beta_vec, gamma_vec, second_hand_merchant_offer_price, d_vec, nu_vec, out=None):
        """
        Compute user utilities for second-hand car options.

        Args:
            out (np.ndarray): Optional (switchers x cars) destination to write
                the result into, so the caller's final choice matrix can be
                filled directly instead of building a separate block and
                copying it in.

        Returns:
            np.ndarray: Utility matrix for second-hand options.
        """
        # Built directly in (switchers x cars) orientation. The original formed
        # the (cars x switchers) price difference and transposed it, so every
        # subsequent term was accumulated across a transposed (column-major)
        # view -- the worst possible traversal order for an array this size.
        # -(net_price - offer) == offer - net_price exactly (IEEE negation is
        # exact and rounding is symmetric), so the values are unchanged.
        net_price = np.maximum(0, vehicle_dict_vecs["price"] - vehicle_dict_vecs["used_rebate"])

        if out is None:
            U_a_i_t_matrix = second_hand_merchant_offer_price[:, np.newaxis] - net_price
        else:
            U_a_i_t_matrix = out
            np.subtract(second_hand_merchant_offer_price[:, np.newaxis], net_price, out=U_a_i_t_matrix)

        age_factor = (1-vehicle_dict_vecs["delta"])**vehicle_dict_vecs["L_a_t"]

        # One scratch buffer, reused for both rank-1 terms, then accumulated in
        # the same left-to-right order as the original single expression.
        scratch = beta_vec[:, np.newaxis]*vehicle_dict_vecs["Quality_a_t"]**self.alpha
        U_a_i_t_matrix += scratch
        np.multiply(nu_vec[:, np.newaxis], (vehicle_dict_vecs["B"]*vehicle_dict_vecs["Eff_omega_a_t"]*age_factor)**self.zeta, out=scratch)
        U_a_i_t_matrix += scratch
        U_a_i_t_matrix -= self._lifecycle_cost_term(vehicle_dict_vecs, gamma_vec[:, np.newaxis], d_vec[:, np.newaxis], age_factor)

        return U_a_i_t_matrix

    def vectorised_calculate_utility_new_cars(self, vehicle_dict_vecs, beta_vec, gamma_vec, second_hand_merchant_offer_price, d_vec, nu_vec, out=None):
        """
        Compute user utilities for new car options.

        Args:
            out (np.ndarray): Optional (switchers x cars) destination, as above.

        Returns:
            np.ndarray: Utility matrix for new car options.
        """
        # Calculate price difference, applying rebate only for transportType == 3 (included in rebate calculation)
        # Same untransposed, in-place accumulation as the second-hand version.
        net_price = np.maximum(0, vehicle_dict_vecs["price"] - vehicle_dict_vecs["rebate"])

        if out is None:
            U_a_i_t_matrix = second_hand_merchant_offer_price[:, np.newaxis] - net_price
        else:
            U_a_i_t_matrix = out
            np.subtract(second_hand_merchant_offer_price[:, np.newaxis], net_price, out=U_a_i_t_matrix)

        scratch = gamma_vec[:, np.newaxis]*vehicle_dict_vecs["production_emissions"]
        U_a_i_t_matrix -= scratch
        np.multiply(beta_vec[:, np.newaxis], vehicle_dict_vecs["Quality_a_t"]**self.alpha, out=scratch)
        U_a_i_t_matrix += scratch
        np.multiply(nu_vec[:, np.newaxis], (vehicle_dict_vecs["B"]*vehicle_dict_vecs["Eff_omega_a_t"])**self.zeta, out=scratch)
        U_a_i_t_matrix += scratch
        U_a_i_t_matrix -= self._lifecycle_cost_term(vehicle_dict_vecs, gamma_vec[:, np.newaxis], d_vec[:, np.newaxis])

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
        self.EV_users_count = int(np.sum(self._cv_cache["transportType"] == 3))
        self.history_prop_EV.append(self.EV_users_count / self.num_individuals)

    def update_mean_car_age(self):
        """
        Record the mean age, in months, of the currently owned fleet.

        Reads the maintained _cv_cache (kept in sync per switcher by
        _update_cv_cache_row and per non-switcher by the L_a_t increment), so
        this costs one array mean per step rather than a pass over 3000 objects.
        Equivalent to calc_mean_car_age(), which the sensitivity analysis calls
        once at the end of a run.

        Called from next_step() right beside update_EV_stock(), so index t of
        history_mean_car_age_fleet lines up with index t of history_prop_EV.
        """
        self.history_mean_car_age_fleet.append(float(np.mean(self._cv_cache["L_a_t"])))

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
        # Loop-invariant lookups hoisted: this walks all num_individuals owned
        # vehicles every timestep, so each self.<attr> inside the body cost one
        # dict lookup per vehicle per timestep for a value that never changes
        # within the loop.
        gas_price = self.gas_price
        gas_cost_index = self.gas_cost_index
        gas_emissions_index = self.gas_emissions_index
        electricity_price = self.electricity_price
        electricity_emissions_intensity = self.electricity_emissions_intensity
        electricity_cost_index = self.electricity_cost_index
        electricity_emissions_index = self.electricity_emissions_index

        for car in self.current_vehicles:
            if car.transportType == 2:#ICE
                car.fuel_cost_c = gas_price
                car.cost_index = gas_cost_index
                car.emissions_index = gas_emissions_index
            elif car.transportType == 3:
                car.fuel_cost_c = electricity_price
                car.e_t = electricity_emissions_intensity
                car.cost_index = electricity_cost_index
                car.emissions_index = electricity_emissions_index

        ice_mask = self._cv_cache["transportType"] == 2
        ev_mask  = self._cv_cache["transportType"] == 3
        self._cv_cache["fuel_cost_c"][ice_mask] = self.gas_price
        self._cv_cache["fuel_cost_c"][ev_mask]  = self.electricity_price
        self._cv_cache["e_t"][ev_mask]          = self.electricity_emissions_intensity
        self._cv_cache["cost_index"][ice_mask]      = self.gas_cost_index
        self._cv_cache["cost_index"][ev_mask]       = self.electricity_cost_index
        self._cv_cache["emissions_index"][ice_mask] = self.gas_emissions_index
        self._cv_cache["emissions_index"][ev_mask]  = self.electricity_emissions_index

    def next_step(self, carbon_price, second_hand_cars,new_cars, gas_price, electricity_price, electricity_emissions_intensity, rebate, used_rebate, electricity_price_subsidy_dollars, rebate_calibration, used_rebate_calibration, gas_cost_index=0.0, gas_emissions_index=0.0, electricity_cost_index=0.0, electricity_emissions_index=0.0):
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
            gas_cost_index, gas_emissions_index, electricity_cost_index, electricity_emissions_index (float):
                Forward-looking present-value indices at this timestep (see controller.compute_discounted_indices).
                Only used when forward_looking_expectations is on.

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
        self.gas_cost_index = gas_cost_index
        self.gas_emissions_index = gas_emissions_index
        self.electricity_cost_index = electricity_cost_index
        self.electricity_emissions_index = electricity_emissions_index
        self.electricity_price_subsidy_dollars = electricity_price_subsidy_dollars

        #update new tech and prices
        self.second_hand_cars, self.new_cars = second_hand_cars, new_cars
        self.all_vehicles_available = self.new_cars + self.second_hand_cars#ORDER IS VERY IMPORTANT

        self.update_prices_and_emissions_intensity()#UPDATE: the prices and emissions intensities of cars which are currently owned

        self.current_vehicles = self.update_VehicleUsers()
        
        self.consider_ev_vec, self.ev_adoption_vec = self.calculate_ev_adoption(ev_type=3)#BASED ON CONSUMPTION PREVIOUS TIME STEP

        self.update_EV_stock()
        self.update_mean_car_age()

        self.t_social_network +=1
        
        return self.consider_ev_vec, self.new_bought_vehicles #self.chosen_vehicles instead of self.current_vehicles as firms can count pofits