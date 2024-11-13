
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

        self.num_PT_options = 2
        # Efficient user list creation with list comprehension
        self.vehicleUsers_list = [VehicleUser(user_id=i) for i in range(self.num_individuals)]


        # Create network and calculate initial emissions
        self.adjacency_matrix, self.network = self.create_network()
        self.network_density = nx.density(self.network)
        
        #Assume nobody adopts EV at the start, THIS MAY BE AN ISSUE
        self.consider_ev_vec = np.zeros(self.num_individuals)
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

        consider_ev_vec = (proportion_ev_neighbors >= self.chi_vec).astype(int)

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

    def update_VehicleUsers(self):
        self.vehicle_chosen_list = []

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

            # Generate public transport utility and distance, if applicable
            self.generate_public_transport_utility_distance()

            # Generate utilities for purchasing vehicles
            self.utilities_matrix, all_vehicles_list, self.d_matrix = self.generate_utilities()


        self.utilities_kappa, combined_mask = self.masking_options(self.utilities_matrix, all_vehicles_list)
        self.chosen_already_mask = np.ones(len(all_vehicles_list), dtype= np.int8)

        if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
            self.prep_counters()
            for i, user in enumerate(self.vehicleUsers_list):
                vehicle_chosen, vehicle_chosen_index = self.user_chooses(i, user, combined_mask, all_vehicles_list)
                self.update_counters(i, vehicle_chosen, vehicle_chosen_index)
        else:
            for i, user in enumerate(self.vehicleUsers_list):
                vehicle_chosen, vehicle_chosen_index = self.user_chooses(i, user, combined_mask, all_vehicles_list)

        return self.vehicle_chosen_list

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


    def masking_options(self, utilities_matrix, all_vehicles_list):
        # Generate individual masks based on vehicle type and user conditions
        self.invert_ev_mask = np.array([vehicle.transportType != 3 for vehicle in all_vehicles_list])
        self.invert_urban_mask = np.array([vehicle.transportType != 0 for vehicle in all_vehicles_list])
        self.invert_rural_mask = np.array([vehicle.transportType != 1 for vehicle in all_vehicles_list])

        ev_mask_matrix = np.outer(self.consider_ev_vec == 0, self.invert_ev_mask)
        origin_mask_matrix = np.outer(self.origin_vec_invert, self.invert_urban_mask) | np.outer(self.origin_vec, self.invert_rural_mask)
        combined_mask = ev_mask_matrix & origin_mask_matrix
        utilities_kappa = np.power(utilities_matrix, self.kappa) * combined_mask
        return utilities_kappa, combined_mask

    def user_chooses(self, i, user, combined_mask, all_vehicles_list):
        # Select individual-specific utilities
        individual_specific_util = self.utilities_kappa[i]        

        # Check if all utilities are zero after filtering
        if not np.any(individual_specific_util):#THIS SHOULD ONLY REALLY BE TRIGGERED RIGHT AT THE START
            bad_options = (combined_mask[i])*self.chosen_already_mask
            #print(bad_options)
            choice_index = np.random.choice(bad_options)

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
                #print("before adding",len(self.second_hand_merchant.cars_on_sale))
                user.vehicle.owner_id = self.second_hand_merchant.id
                self.second_hand_merchant.add_to_stock(user.vehicle)
                user.vehicle = None
                #print("after adding", len(self.second_hand_merchant.cars_on_sale))

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

    def generate_public_transport_utility_distance(self):
        PT_vehicle_dict_vecs = self.gen_vehicle_dict_vecs(self.public_transport_options)
        self.PT_utilities, self.d_PT = self.vectorised_calculate_utility(PT_vehicle_dict_vecs, scenario= "public_transport")

    def generate_utilities_alt(self):

        NC_vehicle_dict_vecs = self.gen_vehicle_dict_vecs(self.new_cars)
        NC_utilities, d_NC = self.vectorised_calculate_utility(NC_vehicle_dict_vecs, scenario= "new_car")

        if self.second_hand_cars:
            SH_vehicle_dict_vecs = self.gen_vehicle_dict_vecs(self.second_hand_cars)
            SH_utilities, d_SH = self.vectorised_calculate_utility( SH_vehicle_dict_vecs, scenario= "second_hand")
        
            utilities_matrix = np.hstack((self.PT_utilities, NC_utilities,SH_utilities))
            d_matrix = np.hstack((self.d_PT, d_NC,d_SH))
            car_options =  self.public_transport_options + self.new_cars + self.second_hand_cars
        else:
            utilities_matrix = np.hstack((self.PT_utilities, NC_utilities))
            d_matrix = np.hstack((self.d_PT, d_NC))
            car_options = self.public_transport_options + self.new_cars

        return utilities_matrix, car_options, d_matrix
    
    def generate_utilities(self):
        # Generate utilities and distances for new cars
        NC_vehicle_dict_vecs = self.gen_vehicle_dict_vecs(self.new_cars)
        NC_utilities, d_NC = self.vectorised_calculate_utility(NC_vehicle_dict_vecs, scenario="new_car")

        # Calculate the total columns needed for utilities and distance matrices
        total_columns = self.num_PT_options + NC_utilities.shape[1]
        if self.second_hand_cars:
            SH_vehicle_dict_vecs = self.gen_vehicle_dict_vecs(self.second_hand_cars)
            SH_utilities, d_SH = self.vectorised_calculate_utility(SH_vehicle_dict_vecs, scenario="second_hand")
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
        #print(self.vectorised_optimal_distance_current(vehicle_dict_vecs))
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


    def vectorised_calculate_utility(self, vehicle_dict_vecs, scenario):
        # Precompute EV consideration mask based on user preference and vehicle type
        ev_mask = self.consider_ev_vec[:, np.newaxis]  # Shape: (num_individuals, 1)

        # Only calculate for users who consider EVs or vehicles of non-EV type
        if vehicle_dict_vecs["transportType"][0] == 3:  # Assuming EVs have transportType == 3
            consider_indices = np.where(ev_mask == 1)
        else:
            consider_indices = np.where(ev_mask == 0)

        # Calculate only for relevant individuals
        d_i_t = np.maximum(self.d_i_min_vec[consider_indices[0], np.newaxis], self.vectorised_optimal_distance(vehicle_dict_vecs)[consider_indices])
        commuting_util_matrix = self.vectorised_commuting_utility(vehicle_dict_vecs, d_i_t)[consider_indices]

        base_utility_matrix = commuting_util_matrix / (self.r + (1 - vehicle_dict_vecs["delta_z"]) / (1 - self.alpha))

        # Avoid redundant calculations for those not owning a car
        owns_car_mask = self.users_current_vehicle_type_vec[consider_indices[0]] > 1
        price_difference = vehicle_dict_vecs["price"][consider_indices[1], np.newaxis] - np.where(
            owns_car_mask,
            self.users_current_vehicle_price_vec[consider_indices[0]] / (1 + self.mu),
            0
        )

        # Calculate price and emissions adjustments
        price_adjustment = np.multiply(self.beta_vec[consider_indices[0], np.newaxis], price_difference.T, dtype=np.float32)

        # Initialize U_a_i_t_matrix with zeros for all users
        U_a_i_t_matrix = np.zeros((self.num_individuals, len(vehicle_dict_vecs["price"])), dtype=np.float32)
        if scenario in ["public_transport", "second_hand"]:
            U_a_i_t_matrix[consider_indices] = base_utility_matrix - price_adjustment
        else:
            emissions_penalty = np.multiply(self.gamma_vec[consider_indices[0], np.newaxis], vehicle_dict_vecs["emissions"][consider_indices[1]], dtype=np.float32)
            U_a_i_t_matrix[consider_indices] = base_utility_matrix - price_adjustment - emissions_penalty

        return U_a_i_t_matrix, d_i_t


    def vectorised_calculate_utility(self, vehicle_dict_vecs, scenario):
        # Compute shared base utility components
        d_i_t = np.maximum(self.d_i_min_vec[:, np.newaxis], self.vectorised_optimal_distance(vehicle_dict_vecs))
        
        commuting_util_matrix = self.vectorised_commuting_utility(vehicle_dict_vecs, d_i_t)
        base_utility_matrix = commuting_util_matrix / (self.r + (1 - vehicle_dict_vecs["delta_z"]) / (1 - self.alpha))
        
        # Avoid redundant calculations
        owns_car_mask = self.users_current_vehicle_type_vec > 1
        price_difference = vehicle_dict_vecs["price"][:, np.newaxis] - np.where(
            owns_car_mask,
            self.users_current_vehicle_price_vec / (1 + self.mu),
            0
        )

        # Calculate price and emissions adjustments once
        price_adjustment = np.multiply(self.beta_vec[:, np.newaxis], price_difference.T, dtype=np.float32)
        
        # Use in-place modification to save memory
        if scenario in ["public_transport", "second_hand"]:
            U_a_i_t_matrix = base_utility_matrix - price_adjustment
        else:
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
            vehicle_dict_vecs["Quality_a_t"] * (1 - vehicle_dict_vecs["delta_z"]) ** vehicle_dict_vecs["L_a_t"] *
            (d_i_t ** self.alpha) - d_i_t * cost_component
        )  # Shape: (num_individuals, num_vehicles)

        return commuting_utility_matrix  # Shape: (num_individuals, num_vehicles)
    
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
        self.current_vehicles = self.update_VehicleUsers()
        

        return self.consider_ev_vec, self.current_vehicles
