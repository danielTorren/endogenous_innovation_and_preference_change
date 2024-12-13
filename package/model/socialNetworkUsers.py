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
        Constructs all the necessary attributes for the Social Network object.
        """
        self.t_social_network = 0
        self.emissions_cumulative = 0
        self.emissions_flow_history = []

        self.rebate = parameters_social_network["rebate"]
        self.used_rebate = parameters_social_network["used_rebate"]

        # Initialize parameters
        self.parameters_vehicle_user = parameters_vehicle_user
        self.init_initial_state(parameters_social_network)
        self.init_network_settings(parameters_social_network)
        self.init_preference_distribution(parameters_social_network)

        self.random_state_social_network = np.random.RandomState(parameters_social_network["social_network_seed"])

        self.alpha =  parameters_vehicle_user["alpha"]
        self.eta =  parameters_vehicle_user["eta"]
        self.mu =  parameters_vehicle_user["mu"]
        self.r = parameters_vehicle_user["r"]
        self.kappa = parameters_vehicle_user["kappa"]
        self.second_hand_car_max_consider = parameters_vehicle_user["second_hand_car_max_consider"]
        self.new_car_max_consider = parameters_vehicle_user["new_car_max_consider"]

        # Generate a list of indices and shuffle them
        self.user_indices = np.arange(self.num_individuals)

        # Efficient user list creation with list comprehension
        self.vehicleUsers_list = [VehicleUser(user_id=i) for i in range(self.num_individuals)]

        # Create network and calculate initial emissions
        self.adjacency_matrix, self.network = self.create_network()
        self.network_density = nx.density(self.network)

        self.weighting_matrix = self._calc_weighting_matrix_attribute(self.beta_vec)#INTRODUCE HOMOPHILY INTO THE NETWORK BY ASSORTING BY BETA WITHING GROUPS
        
        #Assume nobody adopts EV at the start, THIS MAY BE AN ISSUE
        self.consider_ev_vec = np.zeros(self.num_individuals).astype(np.int8)

        self.cars_init_state = parameters_social_network["cars_init_state"]
        if self.cars_init_state: 
            self.current_vehicles = self.set_init_cars_selection(parameters_social_network)
        else:
            self.set_init_vehicle_options(parameters_social_network)
            self.current_vehicles = self.update_VehicleUsers()

        self.consider_ev_vec, self.ev_adoption_vec = self.calculate_ev_adoption(ev_type=3)#BASED ON CONSUMPTION PREVIOUS TIME STEP

    ###############################################################################################################################################################
    ###############################################################################################################################################################
    #MODEL SETUP

    def init_initial_state(self, parameters_social_network):
        self.num_individuals = int(round(parameters_social_network["num_individuals"]))
        self.id_generator = parameters_social_network["IDGenerator_firms"]
        self.second_hand_merchant = parameters_social_network["second_hand_merchant"]
        self.save_timeseries_data_state = parameters_social_network["save_timeseries_data_state"]
        self.compression_factor_state = parameters_social_network["compression_factor_state"]
        self.carbon_price =  parameters_social_network["carbon_price"]

    def init_network_settings(self, parameters_social_network):
        self.selection_bias = parameters_social_network["selection_bias"]
        self.network_structure_seed = parameters_social_network["network_structure_seed"]
        self.K_social_network = parameters_social_network["SW_K"] 
        self.prob_rewire = parameters_social_network["SW_prob_rewire"]

    def init_preference_distribution(self, parameters_social_network):
        self.gamma_multiplier = parameters_social_network["gamma_multiplier"]
        self.beta_multiplier = parameters_social_network["beta_multiplier"]

        #GAMMA
        self.a_environment = parameters_social_network["a_environment"]
        self.b_environment = parameters_social_network["b_environment"]
        self.random_state_gamma = np.random.RandomState(parameters_social_network["init_vals_environmental_seed"])
        self.gamma_vec = self.random_state_gamma.beta(self.a_environment, self.b_environment, size=self.num_individuals)*self.gamma_multiplier

        #CHI
        self.a_innovativeness = parameters_social_network["a_innovativeness"]
        self.b_innovativeness = parameters_social_network["b_innovativeness"]

        self.random_state_chi = np.random.RandomState(parameters_social_network["init_vals_innovative_seed"])
        innovativeness_vec_init_unrounded = self.random_state_chi.beta(self.a_innovativeness, self.b_innovativeness, size=self.num_individuals)
        self.chi_vec = np.round(innovativeness_vec_init_unrounded, 1)

        self.ev_adoption_state_vec = np.zeros(self.num_individuals)

        #BETA
        self.a_price = parameters_social_network["a_price"]
        self.b_price = parameters_social_network["b_price"]
        self.random_state_beta = np.random.RandomState(parameters_social_network["init_vals_price_seed"])
        self.beta_vec = self.random_state_beta.beta(self.a_price, self.b_price, size=self.num_individuals)*self.beta_multiplier

        #d min
        self.d_i_min_vec = np.asarray(self.num_individuals*[parameters_social_network["d_i_min"]])


    def set_init_cars_selection(self, parameters_social_network):

        old_cars = parameters_social_network["old_cars"]
        for i, car in enumerate(old_cars):
            self.vehicleUsers_list[i].vehicle = car
            
            #SET USER ID OF CARS
        for i, individual in enumerate(self.vehicleUsers_list):
            individual.vehicle.owner_id = individual.user_id

        current_cars =  [user.vehicle for user in self.vehicleUsers_list]

        return current_cars#current cars

    def set_init_vehicle_options(self, parameters_social_network):
        self.second_hand_cars = []
        self.new_cars = parameters_social_network["init_car_options"]

        self.all_vehicles_available = self.second_hand_cars + self.new_cars
        
    def normalize_vec_sum(self, vec):
        return vec/sum(vec)

    def _split_into_groups(self) -> list:
        """
        Split agents into groups for SBM network creation.
        
        Returns:
            list: List of group sizes
        
        Raises:
            ValueError: If SBM_block_num is not positive
        """
        if self.SBM_block_num <= 0:
            raise ValueError("SBM_block_num must be greater than zero.")
        base_count = self.num_individuals//self.SBM_block_num
        remainder = self.num_individuals % self.SBM_block_num
        group_counts = [base_count + 1] * remainder + [base_count] * (self.SBM_block_num - remainder)
        
        return group_counts
    
    def _calc_weighting_matrix_attribute(self, attribute_array: np.ndarray) -> sp.csr_matrix:
        """
        Calculate weighting matrix based on attribute similarities.
        
        Args:
            attribute_array (np.ndarray): Array of attributes for calculating weights
            
        Returns:
            sp.csr_matrix: Sparse matrix of normalized weights
        """
        differences = attribute_array[self.row_indices_sparse] - attribute_array[self.col_indices_sparse]
        weights = np.exp(-self.selection_bias * np.abs(differences))
        non_diagonal_weighting_matrix = sp.csr_matrix(
            (weights, (self.row_indices_sparse, self.col_indices_sparse)),
            shape=self.adjacency_matrix.shape
        )
        norm_weighting_matrix = self._normlize_matrix(non_diagonal_weighting_matrix)

        return norm_weighting_matrix

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
        Create watts-strogatz small world graph using Networkx library

        parameters_social_network
        ----------
        None

        Returns
        -------
        weighting_matrix: npt.NDArray[bool]
            adjacency matrix, array giving social network structure where 1 represents a connection between agents and 0 no connection. It is symetric about the diagonal
        norm_weighting_matrix: npt.NDArray[float]
            an NxN array how how much each agent values the opinion of their neighbour. Note that is it not symetric and agent i doesn"t need to value the
            opinion of agent j as much as j does i"s opinion
        ws: nx.Graph
            a networkx watts strogatz small world graph
        """
        
        network = nx.watts_strogatz_graph(n=self.num_individuals, k=self.K_social_network, p=self.prob_rewire, seed=self.network_structure_seed)#FIX THE NETWORK STRUCTURE

        adjacency_matrix = nx.to_numpy_array(network)
        self.sparse_adjacency_matrix = sp.csr_matrix(adjacency_matrix)
        # Get the non-zero indices of the adjacency matrix
        self.row_indices_sparse, self.col_indices_sparse = self.sparse_adjacency_matrix.nonzero()

        self.network_density = nx.density(network)
        # Calculate the total number of neighbors for each user
        self.total_neighbors = np.array(self.sparse_adjacency_matrix.sum(axis=1)).flatten()
        return adjacency_matrix, network

    ###############################################################################################################################################################
    
    #DYNAMIC COMPONENTS

    def calculate_ev_adoption(self, ev_type=3):
        """
        Calculate the proportion of neighbors using EVs for each user, 
        and determine EV adoption consideration.
        """
        
        self.vehicle_type_vec = np.array([user.vehicle.transportType for user in self.vehicleUsers_list])  # Current vehicle types

        # Create a binary vec indicating EV users
        ev_adoption_vec = (self.vehicle_type_vec == ev_type).astype(int)

        # Calculate the number of EV-adopting neighbors using sparse matrix multiplication
        ev_neighbors = self.weighting_matrix.dot(ev_adoption_vec)

        consider_ev_vec = (ev_neighbors >= self.chi_vec).astype(np.int8)

        return consider_ev_vec, ev_adoption_vec

##########################################################################################################################################################
#MATRIX CALCULATION
#CHECK THE ACTUAL EQUATIONS

    def update_VehicleUsers(self):

        self.chosen_vehicles = []
        user_vehicle_list = [None]*self.num_individuals

        # Generate a single shuffle order
        shuffle_indices = self.random_state_social_network.permutation(self.num_individuals)

        self.second_hand_bought = 0#CAN REMOVE LATER ON IF I DONT ACTUALLY NEED TO COUNT

        if self.t_social_network > 0 or self.cars_init_state:
            # Vectorize current user vehicle attributes
            self.users_current_vehicle_price_vec = np.asarray([user.vehicle.price for user in self.vehicleUsers_list])
            self.users_current_vehicle_type_vec = np.asarray([user.vehicle.transportType for user in self.vehicleUsers_list])#USED TO CHECK IF YOU OWN A CAR

            # Generate current utilities and vehicles
            utilities_current_matrix, d_current_matrix = self.generate_utilities_current()

            # Generate buying utilities and vehicles
            utilities_buying_matrix, buying_vehicles_list, d_buying_matrix = self.generate_utilities()

            # Preallocate the final utilities and distance matrices
            total_columns = utilities_buying_matrix.shape[1] + utilities_current_matrix.shape[1]
            self.utilities_matrix = np.empty((utilities_buying_matrix.shape[0], total_columns))
            self.d_matrix = np.empty((d_buying_matrix.shape[0], total_columns))

            # Assign matrices directly
            self.utilities_matrix[:, :utilities_buying_matrix.shape[1]] = utilities_buying_matrix
            self.utilities_matrix[:, utilities_buying_matrix.shape[1]:] = utilities_current_matrix

            self.d_matrix[:, :d_buying_matrix.shape[1]] = d_buying_matrix
            self.d_matrix[:, d_buying_matrix.shape[1]:] = d_current_matrix

            # Combine the list of vehicles
            available_and_current_vehicles_list = buying_vehicles_list + self.current_vehicles

        else:
            # Initialize arrays for users without vehicles
            self.users_current_vehicle_type_vec = np.zeros(self.num_individuals, dtype=bool)#USED TO CHECK IF YOU OWN A CAR, DOESNT MATTER VALUE AS LONG AS BELOW 2
            self.users_current_vehicle_price_vec = np.zeros(self.num_individuals, dtype=np.int8)

            # Generate utilities for purchasing vehicles
            self.utilities_matrix, available_and_current_vehicles_list, self.d_matrix = self.generate_utilities()

        #Mask first hand cars
        self.mask_new_cars()

        # Mask the second-hand cars based on sampling for each individual
        if self.second_hand_cars:
            self.index_current_cars_start = len(self.new_cars) + len(self.second_hand_cars)
            self.mask_second_hand_cars()
        else:
            self.index_current_cars_start = len(self.new_cars)

        self.utilities_matrix[self.utilities_matrix < 0] = 0

        utilities_kappa = self.masking_options(self.utilities_matrix, available_and_current_vehicles_list)
        #self.chosen_already_mask = np.ones(len(available_and_current_vehicles_list), dtype=bool)

        self.emissions_flow = 0#MEASURIBNG THE FLOW

        if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
            self.prep_counters()
            for i, person_index in enumerate(shuffle_indices):
                user = self.vehicleUsers_list[person_index]
                vehicle_chosen, user_vehicle, vehicle_chosen_index, utilities_kappa = self.user_chooses(person_index, user, available_and_current_vehicles_list, utilities_kappa)
                user_vehicle_list[person_index] = user_vehicle
                self.update_counters(person_index, vehicle_chosen, vehicle_chosen_index)
                self.update_emisisons(person_index, vehicle_chosen_index, vehicle_chosen)
        else:
            for i, person_index in enumerate(shuffle_indices):
                user = self.vehicleUsers_list[person_index]
                vehicle_chosen, user_vehicle, vehicle_chosen_index, utilities_kappa = self.user_chooses(person_index, user, available_and_current_vehicles_list, utilities_kappa)
                user_vehicle_list[person_index] = user_vehicle
                self.update_emisisons(person_index, vehicle_chosen_index, vehicle_chosen)
        
        self.emissions_flow_history.append(self.emissions_flow)

        return user_vehicle_list#self.vehicle_chosen_list

    def mask_new_cars(self):
        num_new_cars = len(self.new_cars)
        max_consider = min(self.new_car_max_consider, num_new_cars)
        
        # Generate a 2D array of indices for each individual's sampled cars
        self.sampled_indices_new_cars = np.array([
            self.random_state_social_network.choice(num_new_cars, max_consider, replace=False)
            for _ in range(self.num_individuals)
        ])
        
        # Convert sampled indices to actual car objects for later reference
        self.sampled_new_cars = np.asarray([[self.new_cars[idx] for idx in indices] for indices in self.sampled_indices_new_cars])

    def mask_second_hand_cars(self):
        self.num_second_hand_cars = len(self.second_hand_cars)
        max_consider = min(self.second_hand_car_max_consider, self.num_second_hand_cars)

        if max_consider == 0:
            # Handle the edge case where there are no second-hand cars
            self.sampled_indices_second_hand = np.empty((self.num_individuals, 0), dtype=int)
            self.sampled_second_hand_cars = np.empty((self.num_individuals, 0), dtype=object)
            return

        # Generate all indices and shuffle for randomness
        all_indices = np.arange(self.num_second_hand_cars)
        sampled_indices = np.tile(all_indices, (self.num_individuals, 1))
        np.apply_along_axis(self.random_state_social_network.shuffle, 1, sampled_indices)#shuffle each row all the indicies

        # Select the first `max_consider` indices for each individual
        self.sampled_indices_second_hand = sampled_indices[:, :max_consider]

        # Convert cars list to numpy array for efficient indexing
        second_hand_cars_array = np.array(self.second_hand_cars)

        # Use advanced indexing to fetch sampled cars
        self.sampled_second_hand_cars = second_hand_cars_array[self.sampled_indices_second_hand]

    def _gen_mask_from_indices(self, sampled_indices, init_index, available_and_current_vehicles_list, total_number_type):
        # Start with a boolean matrix full of True (allowing all by default)
        mask = np.ones((self.num_individuals, len(available_and_current_vehicles_list)), dtype=bool)
        
        # Set False in the mask outside sampled indices for each individual
        for i, car_indices in enumerate(sampled_indices):
            mask[i, init_index:init_index + total_number_type] = False  # Block out all initially in the range
            mask[i, init_index + car_indices] = True  # Enable only selected indices
        return mask

    def gen_new_car_mask(self, available_and_current_vehicles_list):
        init_index = 0
        return self._gen_mask_from_indices(self.sampled_indices_new_cars, init_index, available_and_current_vehicles_list, len(self.new_cars))

    def gen_second_hand_mask(self, available_and_current_vehicles_list):
        init_index = len(self.new_cars)  # Adjust as per vehicle order
        return self._gen_mask_from_indices(self.sampled_indices_second_hand, init_index, available_and_current_vehicles_list, self.num_second_hand_cars)

    def gen_mask(self, available_and_current_vehicles_list):
        # Generate individual masks based on vehicle type and user conditions
        # Create a boolean vector where True indicates that a vehicle is NOT an EV (non-EV)
        not_ev_vec = np.array([vehicle.transportType != 3 for vehicle in available_and_current_vehicles_list], dtype=bool)
        consider_ev_all_vehicles = np.outer(self.consider_ev_vec == 1, np.ones(len(not_ev_vec), dtype=bool))
        
        dont_consider_evs_on_ev = np.outer(self.consider_ev_vec == 0, not_ev_vec)#This part masks EVs (False for EVs) only for individuals who do not consider EVs

        # Create an outer product to apply the EV consideration across all cars
        ev_mask_matrix = (consider_ev_all_vehicles | dont_consider_evs_on_ev).astype(int)
        sampled_new_car_mask = self.gen_new_car_mask(available_and_current_vehicles_list)

        if self.second_hand_cars:
            sampled_second_hand_car_mask = self.gen_second_hand_mask(available_and_current_vehicles_list)
            combined_mask = ev_mask_matrix & sampled_new_car_mask & sampled_second_hand_car_mask
        else:
            combined_mask = ev_mask_matrix & sampled_new_car_mask

        return combined_mask

    def calc_util_kappa(self,  utilities_matrix_masked):
        utilities_kappa = np.power(utilities_matrix_masked, self.kappa)
        return utilities_kappa
    
    def apply_mask(self, utilities_matrix, combined_mask):
        
        utilities_matrix_masked = utilities_matrix * combined_mask

        return utilities_matrix_masked

    def masking_options(self, utilities_matrix, available_and_current_vehicles_list):

        combined_mask = self.gen_mask(available_and_current_vehicles_list)
        utilities_matrix_masked = self.apply_mask(utilities_matrix, combined_mask)

        utilities_kappa = self.calc_util_kappa(utilities_matrix_masked)

        return utilities_kappa

    def user_chooses(self, person_index, user, available_and_current_vehicles_list, utilities_kappa):
        # Select individual-specific utilities
        individual_specific_util = utilities_kappa[person_index]  


        # Check if all utilities are zero after filtering
        if not np.any(individual_specific_util):#THIS SHOULD ONLY REALLY BE TRIGGERED RIGHT AT THE START
            """THIS SHOULD ACRUALLY BE USED"""
            if self.t_social_network == 0:
                #pick random vehicle which is available
                choice_index = self.random_state_social_network.choice(len(available_and_current_vehicles_list), p=probability_choose)
            else:#keep current car
                choice_index = self.index_current_cars_start + person_index #available_and_current_vehicles_list.index(user.vehicle)
        else:
            # Calculate the probability of choosing each vehicle
            """
            SHODY FIX THIS PROPERY
            """
            if np.isnan(np.sum(individual_specific_util)):
                individual_specific_util = np.nan_to_num(individual_specific_util)
                
            sum_prob = np.sum(individual_specific_util)

            probability_choose = individual_specific_util / sum_prob
            choice_index = self.random_state_social_network.choice(len(available_and_current_vehicles_list), p=probability_choose)

        # Record the chosen vehicle
        vehicle_chosen = available_and_current_vehicles_list[choice_index]

        self.chosen_vehicles.append(vehicle_chosen)#DONT NEED TO WORRY ABOUT ORDER

        # Handle consequences of the choice
        if user.user_id != vehicle_chosen.owner_id:  # New vehicle, not currently owned
            
            # Transfer the user's current vehicle to the second-hand merchant, if any
            if isinstance(user.vehicle, PersonalCar):
                user.vehicle.owner_id = self.second_hand_merchant.id
                self.second_hand_merchant.add_to_stock(user.vehicle)
                user.vehicle = None
            
            if vehicle_chosen.owner_id == self.second_hand_merchant.id:# Buy a second-hand car
                #SET THE UTILITY TO 0 of that second hand car
                utilities_kappa[:, choice_index] = 0

                self.second_hand_merchant.remove_car(vehicle_chosen)
                self.second_hand_bought += 1

                vehicle_chosen.owner_id = user.user_id
                vehicle_chosen.scenario = "current_car"
                user.vehicle = vehicle_chosen
            elif isinstance(vehicle_chosen, CarModel):  # Brand new car
                personalCar_id = self.id_generator.get_new_id()
                user.vehicle = PersonalCar(personalCar_id, vehicle_chosen.firm, user.user_id, vehicle_chosen.component_string, vehicle_chosen.parameters, vehicle_chosen.attributes_fitness, vehicle_chosen.price)
            else:
                raise(ValueError("invalid user transport behaviour"))

        # Update the age or timer of the chosen vehicle
        if isinstance(vehicle_chosen, PersonalCar):
            user.vehicle.update_timer()

        return vehicle_chosen, user.vehicle, choice_index, utilities_kappa

    def generate_utilities_current(self):

        """ Deal with the special case of utilities of current vehicles"""

        CV_vehicle_dict_vecs = self.gen_vehicle_dict_vecs(self.current_vehicles)

        CV_utilities, d_current = self.vectorised_calculate_utility_current(CV_vehicle_dict_vecs)

        CV_utilities_matrix = np.diag(CV_utilities)
        d_current_matrix = np.diag(d_current)

        return CV_utilities_matrix, d_current_matrix

    def generate_utilities(self):
        #CALC WHO OWNS CAR
        owns_car_mask = self.users_current_vehicle_type_vec > 1
        self.price_owns_car_vec = np.where(
            owns_car_mask,
            self.users_current_vehicle_price_vec / (1 + self.mu),
            0
        )

        # Generate utilities and distances for new cars
        NC_vehicle_dict_vecs = self.gen_vehicle_dict_vecs(self.new_cars)
        NC_utilities, d_NC = self.vectorised_calculate_utility_cars(NC_vehicle_dict_vecs)

        # Calculate the total columns needed for utilities and distance matrices
        total_columns = NC_utilities.shape[1]

        if self.second_hand_cars:
            SH_vehicle_dict_vecs = self.gen_vehicle_dict_vecs(self.second_hand_cars)
            SH_utilities, d_SH = self.vectorised_calculate_utility_second_hand_cars(SH_vehicle_dict_vecs)

            total_columns += SH_utilities.shape[1]

        # Preallocate arrays with the total required columns
        utilities_matrix = np.empty((self.num_individuals, total_columns))
        d_matrix = np.empty((self.num_individuals, total_columns))

        # Fill in preallocated arrays with submatrices
        col_idx = 0#start the counter at zero then increase as you set things up
        utilities_matrix[:, col_idx:col_idx + NC_utilities.shape[1]] = NC_utilities
        d_matrix[:, col_idx:col_idx + d_NC.shape[1]] = d_NC
        col_idx += NC_utilities.shape[1]

        if self.second_hand_cars:
            utilities_matrix[:, col_idx:col_idx + SH_utilities.shape[1]] = SH_utilities
            d_matrix[:, col_idx:col_idx + d_SH.shape[1]] = d_SH
            car_options = self.new_cars + self.second_hand_cars
        else:
            car_options = self.new_cars

        return utilities_matrix, car_options, d_matrix

    def gen_vehicle_dict_vecs(self, list_vehicles):
        # Initialize dictionary to hold lists of vehicle properties

        vehicle_dict_vecs = {
            "Quality_a_t": [], 
            "Eff_omega_a_t": [], 
            "price": [], 
            "delta_z": [], 
            "production_emissions": [],
            "fuel_cost_c_z": [], 
            "e_z_t": [],
            "L_a_t": [],
            "nu_z_i_t": [],
            "transportType": []
        }

        # Iterate over each vehicle to populate the arrays
        for vehicle in list_vehicles:
            vehicle_dict_vecs["Quality_a_t"].append(vehicle.Quality_a_t)
            vehicle_dict_vecs["Eff_omega_a_t"].append(vehicle.Eff_omega_a_t)
            vehicle_dict_vecs["price"].append(vehicle.price)
            vehicle_dict_vecs["delta_z"].append(vehicle.delta_z)
            vehicle_dict_vecs["production_emissions"].append(vehicle.emissions)
            vehicle_dict_vecs["fuel_cost_c_z"].append(vehicle.fuel_cost_c_z)
            vehicle_dict_vecs["e_z_t"].append(vehicle.e_z_t)
            vehicle_dict_vecs["L_a_t"].append(vehicle.L_a_t)
            vehicle_dict_vecs["nu_z_i_t"].append(vehicle.nu_z_i_t)
            vehicle_dict_vecs["transportType"].append(vehicle.transportType)

        # convert lists to numpy arrays for vectorised operations
        for key in vehicle_dict_vecs:
            vehicle_dict_vecs[key] = np.array(vehicle_dict_vecs[key])

        return vehicle_dict_vecs

    def vectorised_calculate_utility_current(self, vehicle_dict_vecs):
        """
        Optimized utility calculation assuming individuals compare either their current car, with price adjustments only applied for those who do not own a car.
        """

        # Compute basic utility components common across all scenarios
        d_i_t = np.maximum(
            self.d_i_min_vec,
            self.vectorised_optimal_distance_current(vehicle_dict_vecs)
        )  # Ensuring compatibility for element-wise comparison

        commuting_util_vec = self.vectorised_commuting_utility_current(vehicle_dict_vecs, d_i_t)
        base_utility_vec = commuting_util_vec / (self.r + (1 - vehicle_dict_vecs["delta_z"]) / (1 - self.alpha))

        # Create mask for users who do NOT own a car
        does_not_own_car_mask = self.users_current_vehicle_type_vec <= 1

        # Calculate the price difference and adjustment only for individuals without a car
        if np.any(does_not_own_car_mask):  # Check if there are any individuals without a car
            # Calculate the price difference for those without a car
            price_difference = vehicle_dict_vecs["price"]  # Only need the price of the transport option

            # Compute the price adjustment for individuals without a car
            price_adjustment = self.beta_vec * price_difference  # Shape: (num_individuals)

            # Initialize utility vec with base utility
            U_a_i_t_vec = base_utility_vec

            U_a_i_t_vec[does_not_own_car_mask] -= price_adjustment[does_not_own_car_mask]
        else:
            # EVERYONE OWNS A CAR SO THE UTILITY IS JUST THE BASE UTILITY
            U_a_i_t_vec = base_utility_vec

        return U_a_i_t_vec, d_i_t


    def vectorised_calculate_utility_second_hand_cars(self, vehicle_dict_vecs):
        # Compute shared base utility components
        d_i_t = np.maximum(self.d_i_min_vec[:, np.newaxis], self.vectorised_optimal_distance_cars(vehicle_dict_vecs))

        commuting_util_matrix = self.vectorised_commuting_utility_cars(vehicle_dict_vecs, d_i_t)
        base_utility_matrix = commuting_util_matrix / (self.r + (1 - vehicle_dict_vecs["delta_z"]) / (1 - self.alpha))


        price_difference = np.where(
            vehicle_dict_vecs["transportType"][:, np.newaxis] == 3,  # Check transportType
            (vehicle_dict_vecs["price"][:, np.newaxis] - self.used_rebate) - self.price_owns_car_vec,  # Apply rebate
            vehicle_dict_vecs["price"][:, np.newaxis] - self.price_owns_car_vec  # No rebate
        )

        # Calculate price and emissions adjustments once
        price_adjustment = np.multiply(self.beta_vec[:, np.newaxis], price_difference.T)
        
        # Use in-place modification to save memor
        U_a_i_t_matrix = base_utility_matrix - price_adjustment

        return U_a_i_t_matrix, d_i_t
    
    def vectorised_calculate_utility_cars(self, vehicle_dict_vecs):
        # Compute shared base utility components
        d_i_t = np.maximum(self.d_i_min_vec[:, np.newaxis], self.vectorised_optimal_distance_cars(vehicle_dict_vecs))
        
        commuting_util_matrix = self.vectorised_commuting_utility_cars(vehicle_dict_vecs, d_i_t)

        base_utility_matrix = commuting_util_matrix / (self.r + ((1 - vehicle_dict_vecs["delta_z"])/(1 - self.alpha)))
        

        #price_difference = vehicle_dict_vecs["price"][:, np.newaxis] - self.price_owns_car_vec

        # Calculate price difference, applying rebate only for transportType == 3
        price_difference = np.where(
            vehicle_dict_vecs["transportType"][:, np.newaxis] == 3,  # Check transportType
            (vehicle_dict_vecs["price"][:, np.newaxis] - self.rebate) - self.price_owns_car_vec,  # Apply rebate
            vehicle_dict_vecs["price"][:, np.newaxis] - self.price_owns_car_vec  # No rebate
        )
    
        # Calculate price and emissions adjustments once
        price_adjustment = np.multiply(self.beta_vec[:, np.newaxis], price_difference.T)
        
        # Use in-place modification to save memory
        emissions_penalty = np.multiply(self.gamma_vec[:, np.newaxis], vehicle_dict_vecs["production_emissions"])
        U_a_i_t_matrix = base_utility_matrix - price_adjustment - emissions_penalty
        return U_a_i_t_matrix, d_i_t
    
    def vectorised_optimal_distance_current(self, vehicle_dict_vecs):
        """
        Only does it for the 1 car, calculate the optimal distance for each individual considering only their corresponding vehicle.
        Assumes each individual has one vehicle, aligned by index.
        """
        # Compute numerator for each individual-vehicle pair
        numerator = (
            self.alpha * vehicle_dict_vecs["Quality_a_t"] *
            (1 - vehicle_dict_vecs["delta_z"]) ** vehicle_dict_vecs["L_a_t"]
        )  # Shape: (num_individuals,)

        denominator = np.where(
            vehicle_dict_vecs["transportType"] > 1,  # Shape: (num_vehicles,)
            ((self.beta_vec/vehicle_dict_vecs["Eff_omega_a_t"]) * (vehicle_dict_vecs["fuel_cost_c_z"] + self.carbon_price*vehicle_dict_vecs["e_z_t"])) +
            ((self.gamma_vec/vehicle_dict_vecs["Eff_omega_a_t"]) * vehicle_dict_vecs["e_z_t"]) +
            (self.eta * vehicle_dict_vecs["nu_z_i_t"]),
            (self.eta * vehicle_dict_vecs["nu_z_i_t"])
        )  # Resulting shape: (num_individuals, num_vehicles)

        # Calculate optimal distance vec for each individual-vehicle pair
        optimal_distance_vec = (numerator / denominator) ** (1 / (1 - self.alpha))

        return optimal_distance_vec  # Shape: (num_individuals,)

    def vectorised_optimal_distance_cars(self, vehicle_dict_vecs):
        """Distance of all cars for all agents"""
        # Compute numerator for all vehicles
        numerator = (
            self.alpha * vehicle_dict_vecs["Quality_a_t"] *
            ((1 - vehicle_dict_vecs["delta_z"]) ** vehicle_dict_vecs["L_a_t"])
        )  # Shape: (num_vehicles,)

        # Compute denominator for all individual-vehicle pairs using broadcasting
        # Reshape self.beta_vec and self.gamma_vec to (num_individuals, 1) for broadcasting across vehicles
        denominator = (
            ((self.beta_vec[:, np.newaxis]/vehicle_dict_vecs["Eff_omega_a_t"]) * (vehicle_dict_vecs["fuel_cost_c_z"] + self.carbon_price*vehicle_dict_vecs["e_z_t"])) +
            ((self.gamma_vec[:, np.newaxis]/ vehicle_dict_vecs["Eff_omega_a_t"]) * vehicle_dict_vecs["e_z_t"]) +
            (self.eta * vehicle_dict_vecs["nu_z_i_t"])
        )  # Shape: (num_individuals, num_vehicles)

        # Calculate optimal distance matrix for each individual-vehicle pair
        optimal_distance_matrix = (numerator / denominator) ** (1 / (1 - self.alpha))

        return optimal_distance_matrix  # Shape: (num_individuals, num_vehicles)

    def vectorised_commuting_utility_current(self, vehicle_dict_vecs, d_i_t):
        """
        Only one car. Calculate the commuting utility for each individual considering only their corresponding vehicle.
        Assumes each individual has one vehicle, aligned by index.
        """

        # Compute cost component based on transport type, without broadcasting
        cost_component = np.where(
            vehicle_dict_vecs["transportType"] > 1,  # Shape: (num_individuals,)
            ((self.beta_vec/ vehicle_dict_vecs["Eff_omega_a_t"]) * (vehicle_dict_vecs["fuel_cost_c_z"] + self.carbon_price*vehicle_dict_vecs["e_z_t"])) +
            ((self.gamma_vec/ vehicle_dict_vecs["Eff_omega_a_t"]) * vehicle_dict_vecs["e_z_t"]) +
            (self.eta * vehicle_dict_vecs["nu_z_i_t"]),
            self.eta * vehicle_dict_vecs["nu_z_i_t"]
        )  # Shape: (num_individuals,)

        # Calculate the commuting utility for each individual-vehicle pair
        commuting_utility_vec = np.maximum(
            0,
            vehicle_dict_vecs["Quality_a_t"] * (1 - vehicle_dict_vecs["delta_z"]) ** vehicle_dict_vecs["L_a_t"] *
            (d_i_t ** self.alpha) - d_i_t * cost_component
        )  # Shape: (num_individuals,)

        return commuting_utility_vec  # Shape: (num_individuals,)

    def vectorised_commuting_utility_cars(self, vehicle_dict_vecs, d_i_t):
        """utility of all cars for all agents"""
        # dit Shape: (num_individuals, num_vehicles)

        # Compute cost component based on transport type, with conditional operations
        cost_component = ((self.beta_vec[:, np.newaxis]/ vehicle_dict_vecs["Eff_omega_a_t"]) * (vehicle_dict_vecs["fuel_cost_c_z"] + self.carbon_price*vehicle_dict_vecs["e_z_t"]) +
            ((self.gamma_vec[:, np.newaxis]/ vehicle_dict_vecs["Eff_omega_a_t"]) * vehicle_dict_vecs["e_z_t"]) +
            (self.eta * vehicle_dict_vecs["nu_z_i_t"]))

        # Compute the commuting utility for each individual-vehicle pair

        commuting_utility_matrix = np.maximum(
            0,
            vehicle_dict_vecs["Quality_a_t"] * ((1 - vehicle_dict_vecs["delta_z"]) ** vehicle_dict_vecs["L_a_t"]) * (d_i_t ** self.alpha) - d_i_t * cost_component
        )  # Shape: (num_individuals, num_vehicles)

        return commuting_utility_matrix  # Shape: (num_individuals, num_vehicles)

####################################################################################################################################
    #TIMESERIES
    def prep_counters(self):
        
        self.users_driving_emissions_vec = np.zeros(self.num_individuals)
        self.users_distance_vec = np.zeros(self.num_individuals)
        self.users_utility_vec  = np.zeros(self.num_individuals)
        self.users_transport_type_vec  = np.full((self.num_individuals), np.nan)
    
        #variable to track
    
        self.total_driving_emissions = 0
        self.total_production_emissions = 0
        self.total_utility = 0
        self.total_distance_travelled = 0
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
    
    def update_counters(self, person_index, vehicle_chosen, vehicle_chosen_index):
        #ADD TOTAL EMISSIONS
        driven_distance = self.d_matrix[person_index][vehicle_chosen_index]           
        

        self.users_driving_emissions_vec[person_index] = (driven_distance/vehicle_chosen.Eff_omega_a_t)*vehicle_chosen.e_z_t

        self.users_distance_vec[person_index] = driven_distance
        self.users_utility_vec[person_index] = self.utilities_matrix[person_index][vehicle_chosen_index]
        self.users_transport_type_vec[person_index] = vehicle_chosen.transportType
        
        if vehicle_chosen.scenario == "new_car":  
            self.new_cars_bought +=1
            self.total_production_emissions += vehicle_chosen.emissions
            if vehicle_chosen.transportType == 2:
                self.new_ICE_cars_bought +=1
            else:
                self.new_EV_cars_bought +=1
            
        car_driving_emissions = (driven_distance/vehicle_chosen.Eff_omega_a_t)*vehicle_chosen.e_z_t 
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
        self.production_cost_vals.append(vehicle_chosen.ProdCost_z_t)

        if vehicle_chosen.transportType == 2:#ICE 
            self.quality_vals_ICE.append(vehicle_chosen.Quality_a_t)#done here for efficiency
            self.efficiency_vals_ICE.append(vehicle_chosen.Eff_omega_a_t)
            self.production_cost_vals_ICE.append(vehicle_chosen.ProdCost_z_t)
        else:
            self.quality_vals_EV.append(vehicle_chosen.Quality_a_t)#done here for efficiency
            self.efficiency_vals_EV.append(vehicle_chosen.Eff_omega_a_t)
            self.production_cost_vals_EV.append(vehicle_chosen.ProdCost_z_t)

        self.total_utility += self.utilities_matrix[person_index][vehicle_chosen_index]
        self.total_distance_travelled += driven_distance
            
        if isinstance(vehicle_chosen, PersonalCar):
            self.second_hand_users +=1

        if vehicle_chosen.transportType == 2:#ICE
            self.ICE_users += 1
        elif vehicle_chosen.transportType == 3:#EV
            self.EV_users += 1
        else:
            raise ValueError("Invalid transport type")
        
    def set_up_time_series_social_network(self):
        """
        self.history_driving_emissions = []
        self.history_production_emissions = []
        self.history_total_emissions = []
        self.history_total_utility = []
        self.history_total_distance_driven = []
        self.history_ev_adoption_rate = []
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

        self.history_attributes_EV_cars_on_sale_all_firms = []
        self.history_attributes_ICE_cars_on_sale_all_firms = []
        self.history_second_hand_bought = []
        self.history_new_car_bought = []
        self.history_car_age = []

        self.history_cars_cum_distances_driven = []
        self.history_cars_cum_driven_emissions = []
        self.history_cars_cum_emissions = []

        #INDIVIDUALS LEVEL DATA
        self.history_driving_emissions_individual = []
        self.history_distance_individual = []
        self.history_utility_individual = []
        self.history_transport_type_individual = []
        """
        self.history_prop_EV = []

    def save_timeseries_data_social_network(self):

        #INDIVIDUALS LEVEL DATA

        self.history_prop_EV.append(self.EV_users/(self.ICE_users + self.EV_users))

        """
        self.history_driving_emissions_individual.append(self.users_driving_emissions_vec)
        self.history_distance_individual.append(self.users_distance_vec)
        self.history_utility_individual.append(self.users_utility_vec)
        self.history_transport_type_individual.append(self.users_transport_type_vec)

        self.history_new_ICE_cars_bought.append(self.new_ICE_cars_bought)
        self.history_new_EV_cars_bought.append(self.new_EV_cars_bought)

        #SUMS
        self.history_driving_emissions.append(self.total_driving_emissions)
        self.history_production_emissions.append(self.total_production_emissions)
        self.history_total_emissions.append(self.total_production_emissions + self.total_driving_emissions)
        self.history_total_utility.append(self.total_utility)
        self.history_total_distance_driven.append(self.total_distance_travelled)
        self.history_ev_adoption_rate.append(np.mean(self.ev_adoption_vec))
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

        if self.quality_vals_ICE:
            self.history_quality_ICE.append(self.quality_vals_ICE)
            self.history_efficiency_ICE.append(self.efficiency_vals_ICE)
            self.history_production_cost_ICE.append(self.production_cost_vals_ICE)
        else:
            self.history_quality_ICE.append([np.nan])
            self.history_efficiency_ICE.append([np.nan])
            self.history_production_cost_ICE.append([np.nan])

        if self.quality_vals_EV:
            self.history_quality_EV.append(self.quality_vals_EV)
            self.history_efficiency_EV.append(self.efficiency_vals_EV)
            self.history_production_cost_EV.append(self.production_cost_vals_EV)
        else:
            self.history_quality_EV.append([np.nan])
            self.history_efficiency_EV.append([np.nan])
            self.history_production_cost_EV.append([np.nan])
    

        data_ev = [[vehicle.Quality_a_t, vehicle.Eff_omega_a_t, vehicle.ProdCost_z_t]  for vehicle in self.all_vehicles_available if vehicle.transportType == 3]
        data_ice = [[vehicle.Quality_a_t ,vehicle.Eff_omega_a_t, vehicle.ProdCost_z_t]  for vehicle in self.all_vehicles_available if vehicle.transportType == 2]

        self.history_attributes_EV_cars_on_sale_all_firms.append(data_ev)
        self.history_attributes_ICE_cars_on_sale_all_firms.append(data_ice)

        self.history_car_age.append(self.car_ages)

        self.history_cars_cum_distances_driven.append(self.cars_cum_distances_driven)
        self.history_cars_cum_driven_emissions.append(self.cars_cum_driven_emissions)
        self.history_cars_cum_emissions.append(self.cars_cum_emissions)
        """

####################################################################################################################################
    #TIMESERIES
    def update_emisisons(self, person_index, vehicle_chosen_index, vehicle_chosen):
        driven_distance = self.d_matrix[person_index][vehicle_chosen_index]           
        
        emissions_flow =   (driven_distance/vehicle_chosen.Eff_omega_a_t)*vehicle_chosen.e_z_t
        self.emissions_cumulative += emissions_flow
        self.emissions_flow += emissions_flow

        if vehicle_chosen.scenario == "new_car":  #if its a new car add emisisons
            self.emissions_cumulative += vehicle_chosen.emissions
            self.emissions_flow += vehicle_chosen.emissions

    def update_prices_and_emissions(self):
        #UPDATE EMMISSION AND PRICES, THIS WORKS FOR BOTH PRODUCTION AND INNOVATION
        for car in self.current_vehicles:
            if car.transportType == 2:#ICE
                car.fuel_cost_c_z = self.gas_price
            elif car.transportType == 3:#EV
                car.fuel_cost_c_z = self.electricity_price
                car.e_z_t = self.electricity_emissions_intensity
                car.nu_z_i_t = self.nu_z_i_t_EV

    def next_step(self, carbon_price, second_hand_cars,new_cars, gas_price, electricity_price, electricity_emissions_intensity, nu_z_i_t_EV, rebate, used_rebate):
        """
        Push the simulation forwards one time step. First advance time, then update individuals with data from previous timestep
        then produce new data and finally save it.

        parameters_social_network
        ----------
        None

        Returns
        -------
        None
        """

        self.t_social_network +=1

        self.carbon_price = carbon_price
        self.gas_price =  gas_price
        self.electricity_price = electricity_price
        self.electricity_emissions_intensity = electricity_emissions_intensity
        self.nu_z_i_t_EV = nu_z_i_t_EV
        self.rebate = rebate
        self.used_rebate = used_rebate

        #update new tech and prices
        self.second_hand_cars, self.new_cars = second_hand_cars, new_cars
        self.all_vehicles_available = self.new_cars + self.second_hand_cars#ORDER IS VERY IMPORTANT

        self.consider_ev_vec, self.ev_adoption_vec = self.calculate_ev_adoption(ev_type=3)#BASED ON CONSUMPTION PREVIOUS TIME STEP
 
        self.current_vehicles  = self.update_VehicleUsers()
        #print(self.total_driving_emissions)

        return self.consider_ev_vec,  self.chosen_vehicles #self.chosen_vehicles instead of self.current_vehicles as firms can count porfits
