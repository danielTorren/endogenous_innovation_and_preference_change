
# imports
import numpy as np
import networkx as nx
import numpy.typing as npt
import scipy.sparse as sp
import numpy as np
import random  # Import random module
from package.model.personalCar import PersonalCar
from package.model.VehicleUser import VehicleUser
from package.model.carModel import CarModel

from collections import Counter

class Social_Network:
    def __init__(self, parameters_social_network: dict, parameters_vehicle_user: dict):
        """
        Constructs all the necessary attributes for the Social Network object.
        """
        self.t_social_network = 0
        
        # Initialize parameters
        self.parameters_vehicle_user = parameters_vehicle_user
        self.init_initial_state(parameters_social_network)
        self.init_network_settings(parameters_social_network)
        self.init_preference_distribution(parameters_social_network)
        self.set_init_vehicle_options(parameters_social_network)

        self.alpha =  parameters_vehicle_user["alpha"]
        self.eta =  parameters_vehicle_user["eta"]
        self.mu =  parameters_vehicle_user["mu"]
        self.r = parameters_vehicle_user["r"]
        self.kappa = parameters_vehicle_user["kappa"]
        self.second_hand_car_max_consider = parameters_vehicle_user["second_hand_car_max_consider"]
        self.new_car_max_consider = parameters_vehicle_user["new_car_max_consider"]

        self.urban_public_transport_emissions = parameters_social_network["urban_public_transport_emissions"]
        self.rural_public_public_transport_emissions = parameters_social_network["urban_public_transport_emissions"]

        # Generate a list of indices and shuffle them
        self.user_indices = np.arange(self.num_individuals)

        self.num_PT_options = 2
        # Efficient user list creation with list comprehension
        self.vehicleUsers_list = [VehicleUser(user_id=i) for i in range(self.num_individuals)]

        # Create network and calculate initial emissions
        self.adjacency_matrix, self.network = self.create_network()
        self.network_density = nx.density(self.network)
        
        #Assume nobody adopts EV at the start, THIS MAY BE AN ISSUE
        self.consider_ev_vec = np.zeros(self.num_individuals).astype(np.int8)
        #individual choose their vehicle in the zeroth step

        self.history_user3 = []
        self.history_user2 = []
        self.history_user4 = []

        self.current_vehicles = self.update_VehicleUsers()
        #print("self.current_vehicles", self.current_vehicles)
        #quit()

        #self.chi_vec = np.array([user.chi for user in self.vehicleUsers_list])  # Innovation thresholds
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
        self.network_structure_seed = parameters_social_network["network_structure_seed"]
        #self.network_density_input = parameters_social_network["network_density"]
        #self.K_social_network = parameters_social_network["K"]#int(round((self.num_individuals - 1) * self.network_density_input))  # Calculate number of links
        #self.prob_rewire = parameters_social_network["prob_rewire"]
        self.SBM_block_num = int(parameters_social_network["SBM_block_num"])
        self.SBM_network_density_input_intra_block = parameters_social_network["SBM_network_density_input_intra_block"]
        self.SBM_network_density_input_inter_block = parameters_social_network["SBM_network_density_input_inter_block"]

    def init_preference_distribution(self, parameters_social_network):
        #GAMMA
        self.a_environment = parameters_social_network["a_environment"]
        self.b_environment = parameters_social_network["b_environment"]
        np.random.seed(parameters_social_network["init_vals_environmental_seed"])  # Initialize random seed
        self.gamma_vec = np.random.beta(self.a_environment, self.b_environment, size=self.num_individuals)

        #CHI
        self.a_innovativeness = parameters_social_network["a_innovativeness"]
        self.b_innovativeness = parameters_social_network["b_innovativeness"]
        np.random.seed(parameters_social_network["init_vals_innovative_seed"])  # Initialize random seed
        innovativeness_vec_init_unrounded = np.random.beta(self.a_innovativeness, self.b_innovativeness, size=self.num_individuals)
        self.chi_vec = np.round(innovativeness_vec_init_unrounded, 1)

        self.ev_adoption_state_vec = np.zeros(self.num_individuals)

        #BETA
        self.a_price = parameters_social_network["a_price"]
        self.b_price = parameters_social_network["b_price"]
        np.random.seed(parameters_social_network["init_vals_price_seed"])  # Initialize random seed
        self.beta_vec = np.random.beta(self.a_price, self.b_price, size=self.num_individuals)


        #origin
        #0 indicates urban and indicates rural
        self.prop_urban = parameters_social_network["prop_urban"]
        self.origin_vec = np.asarray([0]*(int(round(self.num_individuals*self.prop_urban))) + [1]*(int(round(self.num_individuals*(1-self.prop_urban)))))#THIS IS A PLACE HOLDER NEED TO DISCUSS THE DISTRIBUTION OF INDIVIDUALS

        self.origin_vec_invert = 1-self.origin_vec

        #d min
        np.random.seed(parameters_social_network["d_min_seed"]) 
        d_i_min = parameters_social_network["d_min_seed"]
        self.d_i_min_vec = np.random.uniform(size = self.num_individuals)*d_i_min

    def set_init_vehicle_options(self, parameters_social_network):
        self.second_hand_cars = []
        self.public_transport_options = parameters_social_network["public_transport"]
        self.new_cars = parameters_social_network["init_car_options"]

        self.all_vehicles_available = self.second_hand_cars + self.public_transport_options + self.new_cars
        
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

        self.origin_vec
        
        return group_counts
    
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

        self.SBM_block_sizes = [self.num_individuals - np.count_nonzero(self.origin_vec), np.count_nonzero(self.origin_vec)]

        block_probs = np.full((self.SBM_block_num, self.SBM_block_num), 
                                self.SBM_network_density_input_inter_block)
        np.fill_diagonal(block_probs, self.SBM_network_density_input_intra_block)
        network = nx.stochastic_block_model(sizes=self.SBM_block_sizes, p=block_probs, 
                                                   seed=self.network_structure_seed)
        self.block_id_list = np.asarray([i for i, size in enumerate(self.SBM_block_sizes) 
                                           for _ in range(size)])
        
        #network = nx.watts_strogatz_graph(n=self.num_individuals, k=self.K_social_network, p=self.prob_rewire, seed=self.network_structure_seed)#FIX THE NETWORK STRUCTURE

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
        ev_neighbors = self.sparse_adjacency_matrix.dot(ev_adoption_vec)


        # Calculate the proportion of neighbors with EVs
        proportion_ev_neighbors = np.divide(ev_neighbors, self.total_neighbors, where=self.total_neighbors != 0)

        consider_ev_vec = (proportion_ev_neighbors >= self.chi_vec).astype(np.int8)

        #consider_ev_vec = np.asarray([1]*self.num_individuals)
        #ev_adoption_vec =  np.asarray([1]*self.num_individuals)
        return consider_ev_vec, ev_adoption_vec

##########################################################################################################################################################
#MATRIX CALCULATION
#CHECK THE ACTUAL EQUATIONS

    def update_VehicleUsers(self):
        
        self.chosen_vehicles = []
        user_vehicle_list = [None]*self.num_individuals

        # Generate a single shuffle order
        shuffle_indices = np.random.permutation(self.num_individuals)##np.random.permutation(self.num_individuals)

        self.second_hand_bought = 0#CAN REMOVE LATER ON IF I DONT ACTUALLY NEED TO COUNT

        if self.t_social_network > 0:
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
            self.index_current_cars_start = len(self.public_transport_options) + len(self.new_cars) + len(self.second_hand_cars)
            self.mask_second_hand_cars()
        else:
            self.index_current_cars_start = len(self.public_transport_options) + len(self.new_cars)

        self.utilities_matrix[self.utilities_matrix < 0] = 0

        utilities_kappa = self.masking_options(self.utilities_matrix, available_and_current_vehicles_list)
        #self.chosen_already_mask = np.ones(len(available_and_current_vehicles_list), dtype=bool)

        if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
            self.prep_counters()
            for i, person_index in enumerate(shuffle_indices):
                user = self.vehicleUsers_list[person_index]
                vehicle_chosen, user_vehicle, vehicle_chosen_index, utilities_kappa = self.user_chooses(person_index, user, available_and_current_vehicles_list, utilities_kappa)
                user_vehicle_list[person_index] = user_vehicle
                self.update_counters(i, vehicle_chosen, vehicle_chosen_index)
        else:
            for i, person_index in enumerate(shuffle_indices):
                user = self.vehicleUsers_list[person_index]
                ____, user_vehicle, vehicle_chosen_index, utilities_kappa = self.user_chooses(person_index, user, available_and_current_vehicles_list, utilities_kappa)
                user_vehicle_list[person_index] = user_vehicle

        return user_vehicle_list#self.vehicle_chosen_list

    def mask_new_cars(self):
        num_new_cars = len(self.new_cars)
        max_consider = min(self.new_car_max_consider, num_new_cars)
        
        # Generate a 2D array of indices for each individual's sampled cars
        self.sampled_indices_new_cars = np.array([
            np.random.choice(num_new_cars, max_consider, replace=False)
            for _ in range(self.num_individuals)
        ])
        
        # Convert sampled indices to actual car objects for later reference
        self.sampled_new_cars = np.asarray([[self.new_cars[idx] for idx in indices] for indices in self.sampled_indices_new_cars])

    def mask_second_hand_cars(self):
        self.num_second_hand_cars = len(self.second_hand_cars)
        max_consider = min(self.second_hand_car_max_consider, self.num_second_hand_cars)
        
        # Generate a 2D array of indices for each individual's sampled cars
        self.sampled_indices_second_hand = np.array([
            np.random.choice(self.num_second_hand_cars, max_consider, replace=False)
            for _ in range(self.num_individuals)
        ])
        
        # Convert sampled indices to actual car objects for later reference
        self.sampled_second_hand_cars = np.asarray([[self.second_hand_cars[idx] for idx in indices] for indices in self.sampled_indices_second_hand])

    def _gen_mask_from_indices(self, sampled_indices, init_index, available_and_current_vehicles_list, total_number_type):
        # Start with a boolean matrix full of True (allowing all by default)
        mask = np.ones((self.num_individuals, len(available_and_current_vehicles_list)), dtype=bool)
        
        # Set False in the mask outside sampled indices for each individual
        for i, car_indices in enumerate(sampled_indices):
            mask[i, init_index:init_index + total_number_type] = False  # Block out all initially in the range
            mask[i, init_index + car_indices] = True  # Enable only selected indices
        return mask

    def gen_new_car_mask(self, available_and_current_vehicles_list):
        init_index = len(self.public_transport_options)  # Adjust as per vehicle order
        return self._gen_mask_from_indices(self.sampled_indices_new_cars, init_index, available_and_current_vehicles_list, len(self.new_cars))

    def gen_second_hand_mask(self, available_and_current_vehicles_list):
        init_index = len(self.public_transport_options) + len(self.new_cars)  # Adjust as per vehicle order
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
        
        # stop rural people from accessign urban public transport
        utilities_matrix[:, 0] *= self.origin_vec_invert
        # stop urban people from accessign rural public transport
        utilities_matrix[:, 1] *= self.origin_vec

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
            #keep current car
            choice_index = self.index_current_cars_start + person_index #available_and_current_vehicles_list.index(user.vehicle)
        else:
            # Calculate the probability of choosing each vehicle
            sum_prob = np.sum(individual_specific_util)
            probability_choose = individual_specific_util / sum_prob
            choice_index = np.random.choice(len(available_and_current_vehicles_list), p=probability_choose)

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
            else:  # Public transport
                user.vehicle = vehicle_chosen

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

        #HAS TO BE RECALCULATED EACH TIME STEP DUE TO THE AFFECT OF OWNING A CAR ON UTILTY
        PT_vehicle_dict_vecs = self.gen_vehicle_dict_vecs(self.public_transport_options)
        self.PT_utilities, self.d_PT = self.vectorised_calculate_utility_public_transport(PT_vehicle_dict_vecs)

        # Generate utilities and distances for new cars
        NC_vehicle_dict_vecs = self.gen_vehicle_dict_vecs(self.new_cars)
        NC_utilities, d_NC = self.vectorised_calculate_utility_cars(NC_vehicle_dict_vecs)

        # Calculate the total columns needed for utilities and distance matrices
        total_columns = self.num_PT_options + NC_utilities.shape[1]

        if self.second_hand_cars:
            SH_vehicle_dict_vecs = self.gen_vehicle_dict_vecs(self.second_hand_cars)
            SH_utilities, d_SH = self.vectorised_calculate_utility_second_hand_cars(SH_vehicle_dict_vecs)

            total_columns += SH_utilities.shape[1]

        # Preallocate arrays with the total required columns
        utilities_matrix = np.empty((self.num_individuals, total_columns))
        d_matrix = np.empty((self.num_individuals, total_columns))

        # Fill in preallocated arrays with submatrices
        col_idx = 0#start the counter at zero then increase as you set things up
        utilities_matrix[:, col_idx:col_idx + self.num_PT_options] = self.PT_utilities
        d_matrix[:, col_idx:col_idx + self.num_PT_options] = self.d_PT
        col_idx += self.num_PT_options

        utilities_matrix[:, col_idx:col_idx + NC_utilities.shape[1]] = NC_utilities
        d_matrix[:, col_idx:col_idx + d_NC.shape[1]] = d_NC
        col_idx += NC_utilities.shape[1]

        if self.second_hand_cars:
            utilities_matrix[:, col_idx:col_idx + SH_utilities.shape[1]] = SH_utilities
            d_matrix[:, col_idx:col_idx + d_SH.shape[1]] = d_SH
            car_options = self.public_transport_options + self.new_cars + self.second_hand_cars
        else:
            car_options = self.public_transport_options + self.new_cars

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
        Optimized utility calculation assuming individuals compare either their current car
        or public transport options, with price adjustments only applied for those who do not own a car.
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

    def vectorised_calculate_utility_public_transport(self, vehicle_dict_vecs):
        # Compute shared base utility components
        d_i_t = np.maximum(self.d_i_min_vec[:, np.newaxis], self.vectorised_optimal_distance_public_transport(vehicle_dict_vecs))

        base_utility_matrix = self.vectorised_commuting_utility_public_transport(vehicle_dict_vecs, d_i_t)

        price_difference = vehicle_dict_vecs["price"][:, np.newaxis] - self.price_owns_car_vec

        # Calculate price and emissions adjustments once
        price_adjustment = np.multiply(self.beta_vec[:, np.newaxis], price_difference.T)
        
        # Use in-place modification to save memor
        U_a_i_t_matrix = base_utility_matrix - price_adjustment

        return U_a_i_t_matrix, d_i_t

    def vectorised_calculate_utility_second_hand_cars(self, vehicle_dict_vecs):
        # Compute shared base utility components
        d_i_t = np.maximum(self.d_i_min_vec[:, np.newaxis], self.vectorised_optimal_distance_cars(vehicle_dict_vecs))

        commuting_util_matrix = self.vectorised_commuting_utility_cars(vehicle_dict_vecs, d_i_t)
        base_utility_matrix = commuting_util_matrix / (self.r + (1 - vehicle_dict_vecs["delta_z"]) / (1 - self.alpha))

        price_difference = vehicle_dict_vecs["price"][:, np.newaxis] - self.price_owns_car_vec

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
        
        price_difference = vehicle_dict_vecs["price"][:, np.newaxis] - self.price_owns_car_vec
    
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

    def vectorised_optimal_distance_public_transport(self, vehicle_dict_vecs):
        """Distance of all cars for all agents"""
        # Compute numerator for all vehicles
        numerator = (self.alpha * vehicle_dict_vecs["Quality_a_t"])  # Shape: (num_vehicles,)

        # Compute denominator for all individual-vehicle pairs using broadcasting

        denominator = self.eta * vehicle_dict_vecs["nu_z_i_t"]  # Resulting shape: (num_individuals, num_vehicles)

        # Calculate optimal distance matrix for each individual-vehicle pair
        optimal_distance_matrix = (numerator / denominator) ** (1 / (1 - self.alpha))

        return optimal_distance_matrix  # Shape: (num_individuals, num_vehicles)

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

    def vectorised_commuting_utility_public_transport(self, vehicle_dict_vecs, d_i_t):
        commuting_utility_matrix = np.maximum(
            0,
            vehicle_dict_vecs["Quality_a_t"] * (d_i_t ** self.alpha) - d_i_t * self.eta * vehicle_dict_vecs["nu_z_i_t"]
        )  # Shape: (num_individuals, num_vehicles)

        return commuting_utility_matrix  # Shape: (num_individuals, num_vehicles)
    
####################################################################################################################################
    #TIMESERIES
    def prep_counters(self):
        #variable to track
        self.total_driving_emissions = 0
        self.total_production_emissions = 0
        self.total_utility = 0
        self.total_distance_travelled = 0
        self.urban_public_transport_users = 0
        self.rural_public_transport_users = 0
        self.ICE_users = 0 
        self.EV_users = 0
        self.HEV_users = 0
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
        self.quality_vals_HEV = []
        self.efficiency_vals_HEV = []
        self.production_cost_vals_HEV = []
        self.new_cars_bought = 0
        self.car_ages = []

        self.cars_cum_distances_driven = []
        self.cars_cum_driven_emissions = []
        self.cars_cum_emissions = []
    
    def update_counters(self, person_index, vehicle_chosen, vehicle_chosen_index):
        #ADD TOTAL EMISSIONS
        driven_distance = self.d_matrix[person_index][vehicle_chosen_index]           
        if vehicle_chosen.scenario == "new_car":  
            self.new_cars_bought +=1
            self.total_production_emissions += vehicle_chosen.emissions
        
        if vehicle_chosen.transportType > 1:  
            
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
            #print(vehicle_chosen.L_a_t, type(vehicle_chosen))
            self.quality_vals.append(vehicle_chosen.Quality_a_t)#done here for efficiency
            self.efficiency_vals.append(vehicle_chosen.Eff_omega_a_t)
            self.production_cost_vals.append(vehicle_chosen.ProdCost_z_t)

            if vehicle_chosen.transportType == 2:#ICE 
                self.quality_vals_ICE.append(vehicle_chosen.Quality_a_t)#done here for efficiency
                self.efficiency_vals_ICE.append(vehicle_chosen.Eff_omega_a_t)
                self.production_cost_vals_ICE.append(vehicle_chosen.ProdCost_z_t)
            elif vehicle_chosen.transportType == 4:#HEV 
                self.quality_vals_HEV.append(vehicle_chosen.Quality_a_t)#done here for efficiency
                self.efficiency_vals_HEV.append(vehicle_chosen.Eff_omega_a_t)
                self.production_cost_vals_HEV.append(vehicle_chosen.ProdCost_z_t)
            else:
                self.quality_vals_EV.append(vehicle_chosen.Quality_a_t)#done here for efficiency
                self.efficiency_vals_EV.append(vehicle_chosen.Eff_omega_a_t)
                self.production_cost_vals_EV.append(vehicle_chosen.ProdCost_z_t)

        self.total_utility += self.utilities_matrix[person_index][vehicle_chosen_index]
        self.total_distance_travelled += driven_distance
            
        if isinstance(vehicle_chosen, PersonalCar):
            self.second_hand_users +=1

        if vehicle_chosen.transportType == 0:#URBAN
            self.urban_public_transport_users +=1
            #if self.origin_vec[i] == 1:
            #    print("WRONG")
        elif vehicle_chosen.transportType == 1:#RURAL
            self.rural_public_transport_users += 1
            #if self.origin_vec[i] == 0:
            #    print("WRONG")
        elif vehicle_chosen.transportType == 2:#ICE
            self.ICE_users += 1
        elif vehicle_chosen.transportType == 3:#EV
            self.EV_users += 1
        elif vehicle_chosen.transportType == 4:#HEV
            self.HEV_users += 1
        else:
            raise ValueError("Invalid transport type")
        
    def set_up_time_series_social_network(self):
        self.history_driving_emissions = []
        self.history_production_emissions = []
        self.history_total_emissions = []
        self.history_total_utility = []
        self.history_total_distance_driven = []
        self.history_ev_adoption_rate = []
        self.history_urban_public_transport_users = []
        self.history_rural_public_transport_users = []
        self.history_consider_ev_rate = []
        self.history_ICE_users = []
        self.history_EV_users = []
        self.history_HEV_users = []
        self.history_second_hand_users = []
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

        self.history_quality_HEV = []
        self.history_efficiency_HEV = []
        self.history_production_cost_HEV = []

        self.history_attributes_EV_cars_on_sale_all_firms = []
        self.history_attributes_HEV_cars_on_sale_all_firms = []
        self.history_attributes_ICE_cars_on_sale_all_firms = []
        self.history_second_hand_bought = []
        self.history_new_car_bought = []
        self.history_car_age = []

        self.history_cars_cum_distances_driven = []
        self.history_cars_cum_driven_emissions = []
        self.history_cars_cum_emissions = []

    def save_timeseries_data_social_network(self):

        self.history_driving_emissions.append(self.total_driving_emissions + self.urban_public_transport_emissions + self.rural_public_public_transport_emissions)
        self.history_production_emissions.append(self.total_production_emissions)
        self.history_total_emissions.append(self.total_production_emissions + self.total_driving_emissions + self.urban_public_transport_emissions + self.rural_public_public_transport_emissions)
        self.history_total_utility.append(self.total_utility)
        self.history_total_distance_driven.append(self.total_distance_travelled)
        self.history_ev_adoption_rate.append(np.mean(self.ev_adoption_vec))
        self.history_consider_ev_rate.append(np.mean(self.consider_ev_vec))
        #print("self", self.urban_public_transport_users, self.rural_public_transport_users)
        self.history_urban_public_transport_users.append(self.urban_public_transport_users)
        self.history_rural_public_transport_users.append(self.rural_public_transport_users)
        self.history_ICE_users.append(self.ICE_users)
        self.history_EV_users.append(self.EV_users)
        self.history_HEV_users.append(self.HEV_users)
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
        
        if self.quality_vals_HEV:
            self.history_quality_HEV.append(self.quality_vals_HEV)
            self.history_efficiency_HEV.append(self.efficiency_vals_HEV)
            self.history_production_cost_HEV.append(self.production_cost_vals_HEV)
        else:
            self.history_quality_HEV.append([np.nan])
            self.history_efficiency_HEV.append([np.nan])
            self.history_production_cost_HEV.append([np.nan])

        data_ev = [[vehicle.Quality_a_t, vehicle.Eff_omega_a_t, vehicle.ProdCost_z_t]  for vehicle in self.all_vehicles_available if vehicle.transportType == 3]
        data_ice = [[vehicle.Quality_a_t ,vehicle.Eff_omega_a_t, vehicle.ProdCost_z_t]  for vehicle in self.all_vehicles_available if vehicle.transportType == 2]
        data_hev = [[vehicle.Quality_a_t, vehicle.Eff_omega_a_t, vehicle.ProdCost_z_t]  for vehicle in self.all_vehicles_available if vehicle.transportType == 4]

        self.history_attributes_EV_cars_on_sale_all_firms.append(data_ev)
        self.history_attributes_HEV_cars_on_sale_all_firms.append(data_hev)
        self.history_attributes_ICE_cars_on_sale_all_firms.append(data_ice)

        self.history_car_age.append(self.car_ages)

        self.history_cars_cum_distances_driven.append(self.cars_cum_distances_driven)
        self.history_cars_cum_driven_emissions.append(self.cars_cum_driven_emissions)
        self.history_cars_cum_emissions.append(self.cars_cum_emissions)

########################################################################################################################

    def next_step(self, carbon_price, second_hand_cars,public_transport_options,new_cars):
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

        #update new tech and prices
        self.second_hand_cars,self.public_transport_options,self.new_cars = second_hand_cars, public_transport_options, new_cars
        self.all_vehicles_available = self.public_transport_options + self.new_cars + self.second_hand_cars#ORDER IS VERY IMPORTANT

        self.consider_ev_vec, self.ev_adoption_vec = self.calculate_ev_adoption(ev_type=3)#BASED ON CONSUMPTION PREVIOUS TIME STEP
 
        self.current_vehicles  = self.update_VehicleUsers()

        counter_same = Counter()  # Keeps track of unique vehicle counts

        # Iterate through all vehicles
        for vehicle in self.current_vehicles:
            if vehicle.transportType > 1 and isinstance(vehicle, PersonalCar):  # Check the condition
                # Use a unique identifier for the vehicle, such as a combination of attributes
                vehicle_id = vehicle.id  # Example attributes
                counter_same[vehicle_id] += 1

        return self.consider_ev_vec,  self.chosen_vehicles #self.chosen_vehicles instead of self.current_vehicles as firms can count porfits
