
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

        self.num_PT_options = 2
        # Efficient user list creation with list comprehension
        self.vehicleUsers_list = [VehicleUser(user_id=i) for i in range(self.num_individuals)]

        # Create network and calculate initial emissions
        self.adjacency_matrix, self.network = self.create_network()
        self.network_density = nx.density(self.network)
        
        #Assume nobody adopts EV at the start, THIS MAY BE AN ISSUE
        self.consider_ev_vec = np.zeros(self.num_individuals).astype(np.int8)
        #individual choose their vehicle in the zeroth step

        self.current_vehicles = self.update_VehicleUsers()

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

    def init_network_settings(self, parameters_social_network):
        self.network_structure_seed = parameters_social_network["network_structure_seed"]
        self.network_density_input = parameters_social_network["network_density"]
        self.K_social_network = int(round((self.num_individuals - 1) * self.network_density_input))  # Calculate number of links
        self.prob_rewire = parameters_social_network["prob_rewire"]

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
        self.origin_vec = np.asarray([0]*(int(round(self.num_individuals/2))) + [1]*(int(round(self.num_individuals/2))))#THIS IS A PLACE HOLDER NEED TO DISCUSS THE DISTRIBUTION OF INDIVIDUALS
        self.origin_vec_invert = 1-self.origin_vec
        #d min
        np.random.seed(parameters_social_network["d_min_seed"]) 
        self.d_i_min_vec = np.random.uniform(size = self.num_individuals)

    def set_init_vehicle_options(self, parameters_social_network):
        self.second_hand_cars = []
        self.public_transport_options = parameters_social_network["public_transport"]
        self.new_cars = parameters_social_network["init_car_options"]

        self.cars_on_sale_all_firms = self.second_hand_cars + self.public_transport_options + self.new_cars
        
    def normalize_vec_sum(self, vec):
        return vec/sum(vec)

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
        ev_neighbors = self.sparse_adjacency_matrix.dot(ev_adoption_vec)


        # Calculate the proportion of neighbors with EVs
        proportion_ev_neighbors = np.divide(ev_neighbors, self.total_neighbors, where=self.total_neighbors != 0)

        consider_ev_vec = (proportion_ev_neighbors >= self.chi_vec).astype(np.int8)

        #consider_ev_vec = np.asarray([1]*self.num_individuals)
        return consider_ev_vec, ev_adoption_vec

##########################################################################################################################################################
#CALC STUFF RELATED TO CAR

    def calc_emissions(self,vehicle_chosen_dict, driven_distance_vec):

        driving_emissions =  driven_distance_vec*(vehicle_chosen_dict["Eff_omega_a_t"]**-1)*vehicle_chosen_dict["e_z_t"]

        production_emissions = np.where(vehicle_chosen_dict["scenario"] == "current_car",vehicle_chosen_dict["emissions"],0)
        
        return driving_emissions, production_emissions
    
##########################################################################################################################################################
#MATRIX CALCULATION
#CHECK THE ACTUAL EQUATIONS

    def update_VehicleUsers_old(self):
        self.vehicle_chosen_list = []

        if self.t_social_network > 0:
            # Vectorize current user vehicle attributes
            self.users_current_vehicle_price_vec = np.asarray([user.vehicle.price for user in self.vehicleUsers_list])
            self.users_current_vehicle_type_vec = np.asarray([user.vehicle.transportType for user in self.vehicleUsers_list])

            # Generate current utilities and vehicles
            utilities_current_matrix, current_vehicles_list, d_current_matrix = self.generate_utilities_current()
            
            # Generate buying utilities and vehicles
            utilities_buying_matrix, buying_vehicles_list, d_buying_matrix = self.generate_utilities_old()

            # Preallocate the final utilities and distance matrices
            total_columns = utilities_buying_matrix.shape[1] + utilities_current_matrix.shape[1]
            self.utilities_matrix = np.empty((utilities_buying_matrix.shape[0], total_columns), dtype=utilities_buying_matrix.dtype)
            self.d_matrix = np.empty((d_buying_matrix.shape[0], total_columns), dtype=d_buying_matrix.dtype)

            # Assign matrices directly
            self.utilities_matrix[:, :utilities_buying_matrix.shape[1]] = utilities_buying_matrix
            self.utilities_matrix[:, utilities_buying_matrix.shape[1]:] = utilities_current_matrix

            self.d_matrix[:, :d_buying_matrix.shape[1]] = d_buying_matrix
            self.d_matrix[:, d_buying_matrix.shape[1]:] = d_current_matrix

            # Combine the list of vehicles
            all_vehicles_list = buying_vehicles_list + current_vehicles_list

        else:
            # Initialize arrays for users without vehicles
            self.users_current_vehicle_type_vec = np.zeros(self.num_individuals, dtype=np.int8)
            self.users_current_vehicle_price_vec = np.zeros(self.num_individuals, dtype=np.int8)

            # Generate utilities for purchasing vehicles
            self.utilities_matrix, all_vehicles_list, self.d_matrix = self.generate_utilities_old()

        #Mask first hand cars
        self.mask_new_cars()

        # Mask the second-hand cars based on sampling for each individual
        if self.second_hand_cars:
            self.mask_second_hand_cars()

        self.utilities_kappa = self.masking_options(self.utilities_matrix, all_vehicles_list)
        self.chosen_already_mask = np.ones(len(all_vehicles_list), dtype= np.int8)

        if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
            self.prep_counters()
            for i, user in enumerate(self.vehicleUsers_list):
                vehicle_chosen, vehicle_chosen_index = self.user_chooses_old(i, user, all_vehicles_list)
                self.update_counters(i, vehicle_chosen, vehicle_chosen_index)
        else:
            for i, user in enumerate(self.vehicleUsers_list):
                vehicle_chosen, vehicle_chosen_index = self.user_chooses_old(i, user, all_vehicles_list)

        return self.vehicle_chosen_list

    def mask_new_cars(self):
        num_new_cars = len(self.new_cars)
        max_consider = min(self.new_car_max_consider, num_new_cars)
        
        # Generate a 2D array of indices for each individual's sampled cars
        sampled_indices = np.array([
            np.random.choice(num_new_cars, max_consider, replace=False)
            for _ in range(self.num_individuals)
        ])
        
        # Convert sampled indices to actual car objects for later reference
        self.sampled_new_cars = [[self.new_cars[idx] for idx in indices] for indices in sampled_indices]

    def mask_second_hand_cars(self):
        num_second_hand_cars = len(self.second_hand_cars)
        max_consider = min(self.second_hand_car_max_consider, num_second_hand_cars)
        
        # Generate a 2D array of indices for each individual's sampled cars
        sampled_indices = np.array([
            np.random.choice(num_second_hand_cars, max_consider, replace=False)
            for _ in range(self.num_individuals)
        ])
        
        # Convert sampled indices to actual car objects for later reference
        self.sampled_second_hand_cars = [[self.second_hand_cars[idx] for idx in indices] for indices in sampled_indices]

    def gen_new_car_mask(self,all_vehicles_list):
        # Individualized mask for new car sampling
        sampled_new_car_mask = np.zeros((self.num_individuals, len(all_vehicles_list)), dtype=bool)
        for i, sampled_cars in enumerate(self.sampled_new_cars):
            for car in sampled_cars:
                car_index = all_vehicles_list.index(car)
                sampled_new_car_mask[i, car_index] = True
        return sampled_new_car_mask

    def gen_second_hand_mask(self, all_vehicles_list):
        sampled_car_mask =np.zeros((self.num_individuals, len(all_vehicles_list)), dtype=bool)
        for i, sampled_cars in enumerate(self.sampled_second_hand_cars):
            for car in sampled_cars:
                car_index = all_vehicles_list.index(car)
                sampled_car_mask[i, car_index] = True
        return sampled_car_mask

    def gen_mask(self, all_vehicles_list):

        # Generate individual masks based on vehicle type and user conditions
        self.invert_ev_mask = np.array([vehicle.transportType != 3 for vehicle in all_vehicles_list])
        self.invert_urban_mask = np.array([vehicle.transportType != 0 for vehicle in all_vehicles_list])
        self.invert_rural_mask = np.array([vehicle.transportType != 1 for vehicle in all_vehicles_list])

        ev_mask_matrix = np.outer(self.consider_ev_vec == 0, self.invert_ev_mask)
        origin_mask_matrix = np.outer(self.origin_vec_invert, self.invert_urban_mask) | np.outer(self.origin_vec, self.invert_rural_mask)
        
        sampled_new_car_mask = self.gen_new_car_mask(all_vehicles_list)

        if self.second_hand_cars:
            # Individualized mask for second-hand car sampling
            sampled_second_hand_car_mask = self.gen_second_hand_mask(all_vehicles_list)

            # Combine general eligibility and individualized sampling masks
            combined_mask = ev_mask_matrix & origin_mask_matrix & sampled_new_car_mask & sampled_second_hand_car_mask
        else:
            combined_mask = ev_mask_matrix & origin_mask_matrix & sampled_new_car_mask

        #combined_mask = ev_mask_matrix & origin_mask_matrix
        return combined_mask

    def calc_util_kappa(self, utilities_matrix, combined_mask):
        utilities_matrix_masked = utilities_matrix * combined_mask
        utilities_kappa = np.power(utilities_matrix_masked, self.kappa)
        return utilities_kappa
    
    def masking_options(self, utilities_matrix, all_vehicles_list):

        combined_mask = self.gen_mask(all_vehicles_list)

        utilities_kappa = self.calc_util_kappa(utilities_matrix, combined_mask)

        return utilities_kappa

    def user_chooses_old(self, i, user, all_vehicles_list):
        # Select individual-specific utilities
        individual_specific_util = self.utilities_kappa[i]        

        # Check if all utilities are zero after filtering
        if not np.any(individual_specific_util):#THIS SHOULD ONLY REALLY BE TRIGGERED RIGHT AT THE START
            # Default to public transport based on origin
            if self.origin_vec[i] == 0:  # Urban user
                choice_index = 0  # First index in all_vehicles_list for urban public transport
            else:  # Rural user
                choice_index = 1  # Second index in all_vehicles_list for rural public transport

        else:
            # Calculate the probability of choosing each vehicle
            probability_choose = individual_specific_util / individual_specific_util.sum()
            choice_index = np.random.choice(len(all_vehicles_list), p=probability_choose)

        # Record the chosen vehicle
        vehicle_chosen = all_vehicles_list[choice_index]
        self.vehicle_chosen_list.append(vehicle_chosen)

        if isinstance(vehicle_chosen, PersonalCar):
            # Remove the chosen vehicle's utility from all users for the next round
            self.utilities_kappa[:, choice_index] = 0
            self.chosen_already_mask[choice_index] = 0

        # Handle consequences of the choice
        if user.user_id != vehicle_chosen.owner_id:  # New vehicle, not currently owned
            # Transfer the user's current vehicle to the second-hand merchant, if any
            
            if isinstance(user.vehicle, PersonalCar):
                user.vehicle.owner_id = self.second_hand_merchant.id
                self.second_hand_merchant.add_to_stock(user.vehicle)
                user.vehicle = None


            # Buy a second-hand car
            if isinstance(vehicle_chosen, PersonalCar):  # Second-hand car or user's own
                # Remove from second-hand stock if still present and owned by merchant

                if vehicle_chosen in self.second_hand_merchant.cars_on_sale and vehicle_chosen.owner_id == self.second_hand_merchant.id:
                    self.second_hand_merchant.remove_car(vehicle_chosen)

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
        
        return vehicle_chosen, choice_index

    def generate_utilities_current(self):

        """ Deal with the special case of utilities of current vehicles"""

        CV_vehicle_dict_vecs = self.gen_vehicle_dict_vecs(self.current_vehicles)

        CV_utilities, d_current = self.vectorised_calculate_utility_current(CV_vehicle_dict_vecs)

        CV_utilities_matrix = np.diag(CV_utilities)
        d_current_matrix = np.diag(d_current)

        return CV_utilities_matrix, self.current_vehicles, d_current_matrix

    def generate_utilities_old(self):
        #CALC WHO OWNS CAR
        owns_car_mask = self.users_current_vehicle_type_vec > 1
        self.price_owns_car_vec = np.where(
            owns_car_mask,
            self.users_current_vehicle_price_vec / (1 + self.mu),
            0
        )

        #HAS TO BE RECALCULATED EACH TIME STEP DUE TO THE AFFECT OF OWNING A CAR ON UTILTY
        PT_vehicle_dict_vecs = self.gen_vehicle_dict_vecs(self.public_transport_options)
        self.PT_utilities, self.d_PT = self.vectorised_calculate_utility_public_second_hand_cars(PT_vehicle_dict_vecs)

        # Generate utilities and distances for new cars
        NC_vehicle_dict_vecs = self.gen_vehicle_dict_vecs(self.new_cars)
        NC_utilities, d_NC = self.vectorised_calculate_utility_cars(NC_vehicle_dict_vecs)

        # Calculate the total columns needed for utilities and distance matrices
        total_columns = self.num_PT_options + NC_utilities.shape[1]

        if self.second_hand_cars:
            SH_vehicle_dict_vecs = self.gen_vehicle_dict_vecs(self.second_hand_cars)
            SH_utilities, d_SH = self.vectorised_calculate_utility_public_second_hand_cars(SH_vehicle_dict_vecs)
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
            "emissions": [],
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
            vehicle_dict_vecs["emissions"].append(vehicle.emissions)
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
            # Initialize utility vec with base utility
            U_a_i_t_vec = base_utility_vec

        return U_a_i_t_vec, d_i_t
    
    def vectorised_calculate_utility_public_second_hand_cars(self, vehicle_dict_vecs):
        # Compute shared base utility components
        d_i_t = np.maximum(self.d_i_min_vec[:, np.newaxis], self.vectorised_optimal_distance(vehicle_dict_vecs))
        
        commuting_util_matrix = self.vectorised_commuting_utility(vehicle_dict_vecs, d_i_t)
        base_utility_matrix = commuting_util_matrix / (self.r + (1 - vehicle_dict_vecs["delta_z"]) / (1 - self.alpha))
        
        price_difference = vehicle_dict_vecs["price"][:, np.newaxis] - self.price_owns_car_vec

        # Calculate price and emissions adjustments once
        price_adjustment = np.multiply(self.beta_vec[:, np.newaxis], price_difference.T, dtype=np.float32)
        
        # Use in-place modification to save memor
        U_a_i_t_matrix = base_utility_matrix - price_adjustment
        return U_a_i_t_matrix, d_i_t
    
    def vectorised_calculate_utility_cars(self, vehicle_dict_vecs):
        # Compute shared base utility components
        d_i_t = np.maximum(self.d_i_min_vec[:, np.newaxis], self.vectorised_optimal_distance(vehicle_dict_vecs))
        
        commuting_util_matrix = self.vectorised_commuting_utility(vehicle_dict_vecs, d_i_t)
        base_utility_matrix = commuting_util_matrix / (self.r + (1 - vehicle_dict_vecs["delta_z"]) / (1 - self.alpha))
        
        price_difference = vehicle_dict_vecs["price"][:, np.newaxis] - self.price_owns_car_vec
    
        # Calculate price and emissions adjustments once
        price_adjustment = np.multiply(self.beta_vec[:, np.newaxis], price_difference.T, dtype=np.float32)
        
        # Use in-place modification to save memory
        emissions_penalty = np.multiply(self.gamma_vec[:, np.newaxis], vehicle_dict_vecs["emissions"], dtype=np.float32)
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

        # Compute denominator for each individual-vehicle pair without broadcasting
        denominator = (
            (self.beta_vec * vehicle_dict_vecs["Eff_omega_a_t"] ** -1 * vehicle_dict_vecs["fuel_cost_c_z"]) +
            (self.gamma_vec * vehicle_dict_vecs["Eff_omega_a_t"] ** -1 * vehicle_dict_vecs["e_z_t"]) +
            (self.eta * vehicle_dict_vecs["nu_z_i_t"])
        )  # Shape: (num_individuals,)

        # Calculate optimal distance vec for each individual-vehicle pair
        optimal_distance_vec = (numerator / denominator) ** (1 / (1 - self.alpha))

        return optimal_distance_vec  # Shape: (num_individuals,)

    def vectorised_optimal_distance(self, vehicle_dict_vecs):
        """Distance of all cars for all agents"""
        # Compute numerator for all vehicles
        numerator = (
            self.alpha * vehicle_dict_vecs["Quality_a_t"] *
            (1 - vehicle_dict_vecs["delta_z"]) ** vehicle_dict_vecs["L_a_t"]
        )  # Shape: (num_vehicles,)

        # Compute denominator for all individual-vehicle pairs using broadcasting
        # Reshape self.beta_vec and self.gamma_vec to (num_individuals, 1) for broadcasting across vehicles
        denominator = (
            (self.beta_vec[:, np.newaxis] * vehicle_dict_vecs["Eff_omega_a_t"] ** -1 * vehicle_dict_vecs["fuel_cost_c_z"]) +
            (self.gamma_vec[:, np.newaxis] * vehicle_dict_vecs["Eff_omega_a_t"] ** -1 * vehicle_dict_vecs["e_z_t"]) +
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
            (self.beta_vec * (1 / vehicle_dict_vecs["Eff_omega_a_t"]) * vehicle_dict_vecs["fuel_cost_c_z"]) +
            (self.gamma_vec * (1 / vehicle_dict_vecs["Eff_omega_a_t"]) * vehicle_dict_vecs["e_z_t"]) +
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

    def vectorised_commuting_utility(self, vehicle_dict_vecs, d_i_t):
        """utility of all cars for all agents"""
        # dit Shape: (num_individuals, num_vehicles)

        # Compute cost component based on transport type, with conditional operations
        cost_component = np.where(
            vehicle_dict_vecs["transportType"] > 1,  # Shape: (num_vehicles,)
            (self.beta_vec[:, np.newaxis] * (1 / vehicle_dict_vecs["Eff_omega_a_t"]) * vehicle_dict_vecs["fuel_cost_c_z"]) +
            (self.gamma_vec[:, np.newaxis] * (1 / vehicle_dict_vecs["Eff_omega_a_t"]) * vehicle_dict_vecs["e_z_t"]) +
            (self.eta * vehicle_dict_vecs["nu_z_i_t"]),
            self.eta * vehicle_dict_vecs["nu_z_i_t"]
        )  # Resulting shape: (num_individuals, num_vehicles)

        # Compute the commuting utility for each individual-vehicle pair

        commuting_utility_matrix = np.maximum(
            0,
            vehicle_dict_vecs["Quality_a_t"] * (1 - vehicle_dict_vecs["delta_z"]) ** vehicle_dict_vecs["L_a_t"] * (d_i_t ** self.alpha) - d_i_t * cost_component
        )  # Shape: (num_individuals, num_vehicles)

        return commuting_utility_matrix  # Shape: (num_individuals, num_vehicles)
    
    
####################################################################################################################   
#THE MASKED IMPLEMENTATION
    def update_VehicleUsers(self):
        self.vehicle_chosen_list = []

        if self.second_hand_cars:
            # Generate a 2D array of indices for each individual's sampled cars
            num_second_hand_cars = len(self.second_hand_cars)
            max_consider = min(self.second_hand_car_max_consider, num_second_hand_cars)
            sampled_indices = np.array([
                np.random.choice(num_second_hand_cars, max_consider, replace=False)
                for _ in range(self.num_individuals)
            ])

            # Create `sampled_car_mask` using `sampled_indices`
            self.sampled_car_mask = np.zeros((self.num_individuals, num_second_hand_cars), dtype=bool)
            np.put_along_axis(self.sampled_car_mask, sampled_indices, True, axis=1)

        if self.t_social_network > 0:
            # Vectorize current user vehicle attributes
            self.users_current_vehicle_price_vec = np.asarray([user.vehicle.price for user in self.vehicleUsers_list])
            self.users_current_vehicle_type_vec = np.asarray([user.vehicle.transportType for user in self.vehicleUsers_list])

            # Generate current utilities and vehicles
            utilities_current_matrix, current_vehicles_list, d_current_matrix = self.generate_utilities_current()
            
            # Generate buying utilities and vehicles
            utilities_buying_matrix, buying_vehicles_list, d_buying_matrix = self.generate_utilities()

            # Preallocate the final utilities and distance matrices
            total_columns = utilities_buying_matrix.shape[1] + utilities_current_matrix.shape[1]
            self.utilities_matrix = np.empty((utilities_buying_matrix.shape[0], total_columns), dtype=utilities_buying_matrix.dtype)
            self.d_matrix = np.empty((d_buying_matrix.shape[0], total_columns), dtype=d_buying_matrix.dtype)

            # Assign matrices directly
            self.utilities_matrix[:, :utilities_buying_matrix.shape[1]] = utilities_buying_matrix
            self.utilities_matrix[:, utilities_buying_matrix.shape[1]:] = utilities_current_matrix

            self.d_matrix[:, :d_buying_matrix.shape[1]] = d_buying_matrix
            self.d_matrix[:, d_buying_matrix.shape[1]:] = d_current_matrix

            # Combine the list of vehicles
            all_vehicles_list = buying_vehicles_list + current_vehicles_list

        else:
            # Initialize arrays for users without vehicles
            self.users_current_vehicle_type_vec = np.zeros(self.num_individuals, dtype=np.int8)
            self.users_current_vehicle_price_vec = np.zeros(self.num_individuals, dtype=np.int8)

            # Generate utilities for purchasing vehicles
            self.utilities_matrix, all_vehicles_list, self.d_matrix = self.generate_utilities()

        self.utilities_kappa = np.power(self.utilities_matrix, self.kappa)#I DONT NEED TO FILTER AS EVERYTHIGN IS ALREADY FILTERED IN THE UTILITY CALCULATIONS

        self.chosen_already_mask = np.ones(len(all_vehicles_list), dtype= np.int8)

        if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
            self.prep_counters()
            for i, user in enumerate(self.vehicleUsers_list):
                vehicle_chosen, vehicle_chosen_index = self.user_chooses(i, user, all_vehicles_list)
                self.update_counters(i, vehicle_chosen, vehicle_chosen_index)
        else:
            for i, user in enumerate(self.vehicleUsers_list):
                vehicle_chosen, vehicle_chosen_index = self.user_chooses(i, user, all_vehicles_list)

        return self.vehicle_chosen_list
    
    def user_chooses(self, i, user, all_vehicles_list):
        """PICK BASED ON UTILITIES AND THEN IF NO CHOICE JUST TAKE THE BUS"""
        # Select individual-specific utilities from the pre-filtered utilities matrix
        individual_specific_util = self.utilities_kappa[i]  

        # Check if all utilities are zero after filtering
        if not np.any(individual_specific_util):
            # Default to public transport based on origin
            if self.origin_vec[i] == 0:  # Urban user
                choice_index = 0  # First index in all_vehicles_list for urban public transport
            else:  # Rural user
                choice_index = 1  # Second index in all_vehicles_list for rural public transport
        else:
            # Calculate the probability of choosing each vehicle based on non-zero utilities
            probability_choose = individual_specific_util / individual_specific_util.sum()
            choice_index = np.random.choice(len(all_vehicles_list), p=probability_choose)

        # Record the chosen vehicle
        vehicle_chosen = all_vehicles_list[choice_index]
        self.vehicle_chosen_list.append(vehicle_chosen)

        if isinstance(vehicle_chosen, PersonalCar):
            # Remove the chosen vehicle's utility from all users for the next round
            self.utilities_kappa[:, choice_index] = 0
            self.chosen_already_mask[choice_index] = 0

        # Handle consequences of the choice
        if user.user_id != vehicle_chosen.owner_id:  # New vehicle, not currently owned
            # Transfer the user's current vehicle to the second-hand merchant, if any
            if isinstance(user.vehicle, PersonalCar):
                user.vehicle.owner_id = self.second_hand_merchant.id
                self.second_hand_merchant.add_to_stock(user.vehicle)
                user.vehicle = None

            # Buy a second-hand car
            if isinstance(vehicle_chosen, PersonalCar):
                # Remove from second-hand stock if still present and owned by merchant
                if vehicle_chosen in self.second_hand_merchant.cars_on_sale and vehicle_chosen.owner_id == self.second_hand_merchant.id:
                    self.second_hand_merchant.remove_car(vehicle_chosen)

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
        
        return vehicle_chosen, choice_index

    def generate_utilities(self):
        # Calculate who owns car first
        owns_car_mask = self.users_current_vehicle_type_vec > 1
        self.price_owns_car_vec = np.where(
            owns_car_mask,
            self.users_current_vehicle_price_vec / (1 + self.mu),
            0
        ).astype(np.int8)

        # Initialize arrays to store final utilities
        total_vehicles = (len(self.public_transport_options) + 
                        len(self.new_cars) + 
                        (len(self.second_hand_cars) if self.second_hand_cars else 0))
        
        utilities_matrix = np.zeros((self.num_individuals, total_vehicles))
        d_matrix = np.zeros((self.num_individuals, total_vehicles))
        current_idx = 0

        # Process public transport options
        utilities_matrix, d_matrix, current_idx = self.process_public(utilities_matrix, d_matrix, current_idx)

        # Process new cars
        utilities_matrix, d_matrix, current_idx = self.process_new_cars(utilities_matrix, d_matrix, current_idx)

        # Process second-hand cars if they exist
        if self.second_hand_cars:
            utilities_matrix, d_matrix, current_idx = self.process_second_hand(utilities_matrix, d_matrix, current_idx)

        return utilities_matrix, self.get_combined_vehicle_list(), d_matrix

    def process_public(self, utilities_matrix, d_matrix, current_idx):
        pt_mask = self.get_vehicle_type_mask(self.public_transport_options)

        if np.any(pt_mask):
            PT_vehicle_dict_vecs, filtered_pt_mask, _ = self.gen_vehicle_dict_vecs_masked(self.public_transport_options, pt_mask)
            if PT_vehicle_dict_vecs:
                pt_utilities, pt_distances = self.vectorised_calculate_utility_no_new_masked_alt(
                    PT_vehicle_dict_vecs, filtered_pt_mask)
                end_idx = current_idx + len(self.public_transport_options)
                utilities_matrix[:, current_idx:end_idx] = pt_utilities
                d_matrix[:, current_idx:end_idx] = pt_distances
                current_idx = end_idx
        return utilities_matrix, d_matrix, current_idx

    def process_new_cars(self, utilities_matrix, d_matrix, current_idx):
        new_car_mask = self.get_vehicle_type_mask(self.new_cars)
  
        if np.any(new_car_mask):
            NC_vehicle_dict_vecs, filtered_nc_mask, valid_nc_indices = self.gen_vehicle_dict_vecs_masked(self.new_cars, new_car_mask)
            if NC_vehicle_dict_vecs:
                nc_utilities, nc_distances = self.vectorised_calculate_utility_cars_masked_alt(
                    NC_vehicle_dict_vecs, filtered_nc_mask)
                # Place utilities/distances at correct positions using valid_nc_indices
                for idx, valid_idx in enumerate(valid_nc_indices):
                    utilities_matrix[:, current_idx + valid_idx] = nc_utilities[:, idx]
                    d_matrix[:, current_idx + valid_idx] = nc_distances[:, idx]
                current_idx += len(self.new_cars)
        #print("DOEN NEW")
        #quit()
        return utilities_matrix, d_matrix, current_idx

    def pre_processing_second_hand(self):
        # Apply combined mask in a single step

        sh_mask = self.get_vehicle_type_mask(self.second_hand_cars) & self.sampled_car_mask

        return sh_mask
        
    def process_second_hand(self, utilities_matrix, d_matrix, current_idx):
        
        sh_mask = self.pre_processing_second_hand()
        #print(sh_mask.shape)
        
        if np.any(sh_mask):
            SH_vehicle_dict_vecs, filtered_sh_mask, valid_sh_indices = self.gen_vehicle_dict_vecs_masked(
                self.second_hand_cars, sh_mask
            )
            if SH_vehicle_dict_vecs:
                sh_utilities, sh_distances = self.vectorised_calculate_utility_no_new_masked_alt(
                    SH_vehicle_dict_vecs, filtered_sh_mask
                )

                # Update `utilities_matrix` and `d_matrix` in blocks instead of in a loop
                utilities_matrix[:, current_idx:current_idx + len(valid_sh_indices)] = sh_utilities
                d_matrix[:, current_idx:current_idx + len(valid_sh_indices)] = sh_distances

        return utilities_matrix, d_matrix, current_idx

    def get_combined_vehicle_list(self):
        """Combine all vehicle lists in the correct order"""
        combined_list = (self.public_transport_options + 
                        self.new_cars + 
                        (self.second_hand_cars if self.second_hand_cars else []))
        return combined_list

    def get_vehicle_type_mask(self, vehicles):
        """Generate mask for vehicle type based on user preferences and vehicle characteristics"""
        mask = np.ones((self.num_individuals, len(vehicles)), dtype=np.int8)

        for j, vehicle in enumerate(vehicles):
            if vehicle.transportType == 3:  # EV
                mask[:, j] &= self.consider_ev_vec
            elif vehicle.transportType == 0:  # Urban
                mask[:, j] &= self.origin_vec
            elif vehicle.transportType == 1:  # Rural
                mask[:, j] &= ~self.origin_vec
                
        return mask

    def gen_vehicle_dict_vecs_masked(self, vehicles, mask):
        """Generate vehicle dictionaries only for masked vehicles, returning a filtered mask and indices."""
        if not np.any(mask):
            return None, None, []  # Return empty if nothing is selected

        # Initialize vehicle dictionary
        vehicle_dict_vecs = {
            "Quality_a_t": [], "Eff_omega_a_t": [], "price": [],
            "delta_z": [], "emissions": [], "fuel_cost_c_z": [],
            "e_z_t": [], "L_a_t": [], "nu_z_i_t": [], "transportType": []
        }
        
        # Track valid vehicles and create a filtered mask aligned with selected vehicles
        valid_vehicle_indices = []
        filtered_mask = np.zeros(mask.shape, dtype=bool)
        
        for i, vehicle in enumerate(vehicles):
            if np.any(mask[:, i]):
                valid_vehicle_indices.append(i)
                filtered_mask[:, len(valid_vehicle_indices) - 1] = mask[:, i]  # Update filtered mask

                # Add each vehicle's attributes to the dictionary
                vehicle_dict_vecs["Quality_a_t"].append(vehicle.Quality_a_t)
                vehicle_dict_vecs["Eff_omega_a_t"].append(vehicle.Eff_omega_a_t)
                vehicle_dict_vecs["price"].append(vehicle.price)
                vehicle_dict_vecs["delta_z"].append(vehicle.delta_z)
                vehicle_dict_vecs["emissions"].append(vehicle.emissions)
                vehicle_dict_vecs["fuel_cost_c_z"].append(vehicle.fuel_cost_c_z)
                vehicle_dict_vecs["e_z_t"].append(vehicle.e_z_t)
                vehicle_dict_vecs["L_a_t"].append(vehicle.L_a_t)
                vehicle_dict_vecs["nu_z_i_t"].append(vehicle.nu_z_i_t)
                vehicle_dict_vecs["transportType"].append(vehicle.transportType)

        # Convert lists to arrays for efficient computation
        vehicle_dict_vecs = {k: np.array(v) for k, v in vehicle_dict_vecs.items()}

        return vehicle_dict_vecs, filtered_mask[:, :len(valid_vehicle_indices)], valid_vehicle_indices

    def vectorised_calculate_utility_no_new_masked(self, vehicle_dict_vecs, filtered_mask):
        """Calculate utilities for public transport options with masking using boolean indexing."""
        if vehicle_dict_vecs is None or filtered_mask is None:
            return np.array([]), np.array([])

        # Initialize result arrays with zeros
        d_i_t = np.zeros_like(filtered_mask, dtype=np.float32)
        U_a_i_t_matrix = np.zeros_like(filtered_mask, dtype=np.float32)
 
        # Calculate optimal distance only for masked entries
        d_i_t[filtered_mask] = np.maximum(
            self.d_i_min_vec,
            self.vectorised_optimal_distance_masked(vehicle_dict_vecs, filtered_mask)[filtered_mask]
        )

        # Calculate commuting utility only for masked entries
        commuting_util = np.zeros_like(filtered_mask, dtype=np.float32)
        commuting_util[filtered_mask] = self.vectorised_commuting_utility_masked(vehicle_dict_vecs, d_i_t, filtered_mask)[filtered_mask]

        # Calculate base utility only for masked entries
        base_utility = np.zeros_like(filtered_mask, dtype=np.float32)
        base_utility[filtered_mask] = commuting_util[filtered_mask] / (
            self.r + (1 - vehicle_dict_vecs["delta_z"]) / (1 - self.alpha)
        )

        # Calculate price difference and adjustment only for masked entries
        price_diff = np.zeros_like(filtered_mask, dtype=np.float32)
        price_diff[filtered_mask] = (vehicle_dict_vecs["price"][np.newaxis, :] - self.price_owns_car_vec[:, np.newaxis])[filtered_mask]

        price_adjustment = np.zeros_like(filtered_mask, dtype=np.float32)
        price_adjustment[filtered_mask] = (self.beta_vec[:, np.newaxis] * price_diff)[filtered_mask]

        # Final utility calculation only for masked entries
        U_a_i_t_matrix[filtered_mask] = base_utility[filtered_mask] - price_adjustment[filtered_mask]

        return U_a_i_t_matrix, d_i_t


    def vectorised_calculate_utility_cars_masked(self, vehicle_dict_vecs, filtered_mask):
        """Calculate utilities only for masked vehicles using boolean indexing."""
        if vehicle_dict_vecs is None or filtered_mask is None:
            return np.array([]), np.array([])

        # Initialize result arrays with zeros
        d_i_t = np.zeros_like(filtered_mask, dtype=np.float32)
        U_a_i_t_matrix = np.zeros_like(filtered_mask, dtype=np.float32)

        # Calculate optimal distance only for masked entries
        d_i_t[filtered_mask] = np.maximum(
            self.d_i_min_vec[:, np.newaxis][filtered_mask],
            self.vectorised_optimal_distance_masked(vehicle_dict_vecs, filtered_mask)[filtered_mask]
        )

        # Calculate commuting utility only for masked entries
        commuting_util = np.zeros_like(filtered_mask, dtype=np.float32)
        commuting_util[filtered_mask] = self.vectorised_commuting_utility_masked(vehicle_dict_vecs, d_i_t, filtered_mask)[filtered_mask]

        # Calculate base utility only for masked entries
        base_utility = np.zeros_like(filtered_mask, dtype=np.float32)
        base_utility[filtered_mask] = commuting_util[filtered_mask] / (
            self.r + (1 - vehicle_dict_vecs["delta_z"]) / (1 - self.alpha)
        )

        # Calculate price adjustment only for masked entries
        price_diff = np.zeros_like(filtered_mask, dtype=np.float32)
        price_diff[filtered_mask] = (vehicle_dict_vecs["price"][:, np.newaxis] - self.price_owns_car_vec).T[filtered_mask]

        price_adjustment = np.zeros_like(filtered_mask, dtype=np.float32)
        price_adjustment[filtered_mask] = (self.beta_vec[:, np.newaxis] * price_diff)[filtered_mask]

        # Calculate emissions penalty only for masked entries
        emissions_penalty = np.zeros_like(filtered_mask, dtype=np.float32)
        emissions_penalty[filtered_mask] = (self.gamma_vec[:, np.newaxis] * vehicle_dict_vecs["emissions"])[filtered_mask]

        # Final utility calculation only for masked entries
        U_a_i_t_matrix[filtered_mask] = base_utility[filtered_mask] - price_adjustment[filtered_mask] - emissions_penalty[filtered_mask]

        return U_a_i_t_matrix, d_i_t


    def vectorised_optimal_distance_masked(self, vehicle_dict_vecs, filtered_mask):
        """Calculate optimal distance only for masked combinations with filtered mask using boolean indexing."""

        # Initialize the result array with zeros
        optimal_distance = np.zeros_like(filtered_mask, dtype=np.float32)

        # Apply calculations only where filtered_mask is True
        numerator = (self.alpha * vehicle_dict_vecs["Quality_a_t"] * 
                    (1 - vehicle_dict_vecs["delta_z"]) ** vehicle_dict_vecs["L_a_t"])

        denominator = (
            (self.beta_vec[:, np.newaxis] * vehicle_dict_vecs["Eff_omega_a_t"] ** -1 * vehicle_dict_vecs["fuel_cost_c_z"]) +
            (self.gamma_vec[:, np.newaxis] * vehicle_dict_vecs["Eff_omega_a_t"] ** -1 * vehicle_dict_vecs["e_z_t"]) +
            (self.eta * vehicle_dict_vecs["nu_z_i_t"])
        )

        # Calculate only for True values in filtered_mask
        optimal_distance[filtered_mask] = (numerator[filtered_mask] / denominator[filtered_mask]) ** (1 / (1 - self.alpha))
        
        return optimal_distance


    def vectorised_commuting_utility_masked(self, vehicle_dict_vecs, d_i_t, filtered_mask):
        """Calculate commuting utility only for masked combinations using boolean indexing."""

        # Initialize the result array with zeros
        commuting_utility = np.zeros_like(filtered_mask, dtype=np.float32)

        # Calculate cost component only for masked entries
        cost_component = np.zeros_like(filtered_mask, dtype=np.float32)
        mask_transport = filtered_mask & (vehicle_dict_vecs["transportType"] > 1)

        cost_component[mask_transport] = (
            (self.beta_vec[:, np.newaxis] * (1 / vehicle_dict_vecs["Eff_omega_a_t"]) * vehicle_dict_vecs["fuel_cost_c_z"]) +
            (self.gamma_vec[:, np.newaxis] * (1 / vehicle_dict_vecs["Eff_omega_a_t"]) * vehicle_dict_vecs["e_z_t"]) +
            (self.eta * vehicle_dict_vecs["nu_z_i_t"])
        )[mask_transport]

        # For non-masked entries, use a default cost component
        cost_component[filtered_mask & ~mask_transport] = (self.eta * vehicle_dict_vecs["nu_z_i_t"])[filtered_mask & ~mask_transport]

        # Calculate the commuting utility only for masked entries
        commuting_utility[filtered_mask] = np.maximum(
            0,
            (vehicle_dict_vecs["Quality_a_t"] * (1 - vehicle_dict_vecs["delta_z"]) ** vehicle_dict_vecs["L_a_t"] *
            (d_i_t ** self.alpha) - d_i_t * cost_component
            )[filtered_mask]
        )

        return commuting_utility

####################################################################################################################################
    
    def vectorised_calculate_utility_no_new_masked_alt(self, vehicle_dict_vecs, filtered_mask):
        """Calculate utilities for public transport options with masking."""
        if vehicle_dict_vecs is None or filtered_mask is None:
            return np.array([]), np.array([])

        # Calculate optimal distance only for masked combinations
        d_i_t = np.where(
            filtered_mask,
            np.maximum(
                self.d_i_min_vec[:, np.newaxis],
                self.vectorised_optimal_distance_masked_alt(vehicle_dict_vecs, filtered_mask)
            ),
            0
        )

        # Calculate utilities only where masked
        commuting_util = np.where(
            filtered_mask,
            self.vectorised_commuting_utility_masked_alt(vehicle_dict_vecs, d_i_t, filtered_mask),
            0
        )

        base_utility = np.where(
            filtered_mask,
            commuting_util / (self.r + (1 - vehicle_dict_vecs["delta_z"]) / (1 - self.alpha)),
            0
        )

        # Apply price adjustment only to masked entries
        price_diff = np.where(
            filtered_mask,
            vehicle_dict_vecs["price"][np.newaxis, :] - self.price_owns_car_vec[:, np.newaxis],
            0
        )

        price_adjustment = np.where(
            filtered_mask,
            np.multiply(self.beta_vec[:, np.newaxis], price_diff),
            0
        )

        U_a_i_t_matrix = base_utility - price_adjustment
        
        return U_a_i_t_matrix, d_i_t
    
    def vectorised_calculate_utility_cars_masked_alt(self, vehicle_dict_vecs, filtered_mask):
        """Calculate utilities only for masked vehicles."""
        if vehicle_dict_vecs is None or filtered_mask is None:
            return np.array([]), np.array([])

        # Calculate optimal distance only for masked combinations
        d_i_t = np.where(
            filtered_mask,
            np.maximum(
                self.d_i_min_vec[:, np.newaxis],
                self.vectorised_optimal_distance_masked_alt(vehicle_dict_vecs, filtered_mask)
            ),
            0
        )
        
        # Calculate utilities only where masked
        commuting_util = np.where(
            filtered_mask,
            self.vectorised_commuting_utility_masked_alt(vehicle_dict_vecs, d_i_t, filtered_mask),
            0
        )
        
        base_utility = np.where(
            filtered_mask,
            commuting_util / (self.r + (1 - vehicle_dict_vecs["delta_z"]) / (1 - self.alpha)),
            0
        )

        # Apply price and emissions adjustments only to masked entries
        price_diff = np.where(
            filtered_mask,
            (vehicle_dict_vecs["price"][:, np.newaxis] - self.price_owns_car_vec).T,
            0
        )
        
        price_adjustment = np.where(
            filtered_mask,
            np.multiply(self.beta_vec[:, np.newaxis], price_diff),
            0
        )
        
        emissions_penalty = np.where(
            filtered_mask,
            np.multiply(self.gamma_vec[:, np.newaxis], vehicle_dict_vecs["emissions"]),
            0
        )
        
        U_a_i_t_matrix = base_utility - price_adjustment - emissions_penalty
        
        return U_a_i_t_matrix, d_i_t

    def vectorised_optimal_distance_masked_alt(self, vehicle_dict_vecs, filtered_mask):
        """Calculate optimal distance only for masked combinations with filtered mask."""

        # Perform the calculation only where the mask is True
        optimal_distance = np.where(
            filtered_mask,
            ((self.alpha * vehicle_dict_vecs["Quality_a_t"] * 
            (1 - vehicle_dict_vecs["delta_z"]) ** vehicle_dict_vecs["L_a_t"])[np.newaxis, :] /
            ((self.beta_vec[:, np.newaxis] * vehicle_dict_vecs["Eff_omega_a_t"] ** -1 * vehicle_dict_vecs["fuel_cost_c_z"]) +
            (self.gamma_vec[:, np.newaxis] * vehicle_dict_vecs["Eff_omega_a_t"] ** -1 * vehicle_dict_vecs["e_z_t"]) +
            (self.eta * vehicle_dict_vecs["nu_z_i_t"]))
            ) ** (1 / (1 - self.alpha)),
            0
        )
        
        return optimal_distance

    def vectorised_commuting_utility_masked_alt(self, vehicle_dict_vecs, d_i_t, filtered_mask):
        """Calculate commuting utility only for masked combinations"""
        # Calculate cost component based on transport type with masking
        cost_component = np.where(
            filtered_mask & (vehicle_dict_vecs["transportType"] > 1),
            (self.beta_vec[:, np.newaxis] * (1 / vehicle_dict_vecs["Eff_omega_a_t"]) * 
            vehicle_dict_vecs["fuel_cost_c_z"]) +
            (self.gamma_vec[:, np.newaxis] * (1 / vehicle_dict_vecs["Eff_omega_a_t"]) * 
            vehicle_dict_vecs["e_z_t"]) +
            (self.eta * vehicle_dict_vecs["nu_z_i_t"]),
            self.eta * vehicle_dict_vecs["nu_z_i_t"]
        )

        # Calculate the commuting utility with masking
        commuting_utility = np.where(
            filtered_mask,
            np.maximum(
                0,
                vehicle_dict_vecs["Quality_a_t"] * 
                (1 - vehicle_dict_vecs["delta_z"]) ** vehicle_dict_vecs["L_a_t"] *
                (d_i_t ** self.alpha) - d_i_t * cost_component
            ),
            0
        )

        return commuting_utility

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
        self.second_hand_users = 0
        self.quality_vals = []
        self.efficiency_vals = []
        self.production_cost_vals = []
    
    def update_counters(self, i, vehicle_chosen, vehicle_chosen_index):
        #ADD TOTAL EMISSIONS
        driven_distance = self.d_matrix[i][vehicle_chosen_index]           
        if vehicle_chosen.scenario == "new_car":  
            self.total_production_emissions += vehicle_chosen.emissions
        self.total_driving_emissions += driven_distance*(vehicle_chosen.Eff_omega_a_t**-1)*vehicle_chosen.e_z_t 
        self.total_utility += self.utilities_matrix[i][vehicle_chosen_index]
        self.total_distance_travelled += driven_distance

        self.quality_vals.append(vehicle_chosen.Quality_a_t)#done here for efficiency
        self.efficiency_vals.append(vehicle_chosen.Eff_omega_a_t)
        self.production_cost_vals.append(vehicle_chosen.ProdCost_z_t)
            
        if isinstance(vehicle_chosen, PersonalCar):
            self.second_hand_users +=1
        
        if vehicle_chosen.transportType == 0:
            self.urban_public_transport_users+=1
        elif vehicle_chosen.transportType == 1:
            self.rural_public_transport_users += 1
        elif vehicle_chosen.transportType == 2:
            self.ICE_users += 1
        else:
            self.EV_users += 1

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

        self.history_quality.append(self.quality_vals)
        self.history_efficiency.append(self.efficiency_vals)
        self.history_production_cost.append(self.production_cost_vals)

        data_ev = [[vehicle.Quality_a_t, vehicle.Eff_omega_a_t, vehicle.ProdCost_z_t]  for vehicle in self.cars_on_sale_all_firms if vehicle.transportType == 3]
        data_ice = [[vehicle.Quality_a_t ,vehicle.Eff_omega_a_t, vehicle.ProdCost_z_t]  for vehicle in self.cars_on_sale_all_firms if vehicle.transportType == 2]

        self.history_attributes_EV_cars_on_sale_all_firms.append(data_ev)
        self.history_attributes_ICE_cars_on_sale_all_firms.append(data_ice)

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
        self.cars_on_sale_all_firms = self.second_hand_cars + self.public_transport_options + self.new_cars

        self.consider_ev_vec, self.ev_adoption_vec = self.calculate_ev_adoption(ev_type=3)#BASED ON CONSUMPTION PREVIOUS TIME STEP
        #self.current_vehicles = self.update_VehicleUsers()
        self.current_vehicles = self.update_VehicleUsers_old()
        

        return self.consider_ev_vec, self.current_vehicles
