"""Create social network with individuals

Created: 10/10/2022
"""

# imports
import numpy as np
import networkx as nx
import numpy.typing as npt
from collections import defaultdict
#from package.model.public_transport import Rural_Public_Transport, Urban_Public_Transport
import scipy.sparse as sp

import numpy as np
from scipy.spatial import distance_matrix

class SocialNetwork:
    def __init__(self, users, adjacency_matrix):
        self.adjacency_matrix = adjacency_matrix  # Small-world network structure (N x N matrix)
        self.N = len(users)  # Number of users

class Social_Network:
    def __init__(self, parameters_social_network: dict):
        """
        Constructs all the necessary attributes for the Social Network object.
        """
        self.t_social_network = 0
        
        # Initialize parameters
        self.init_time_parameters(parameters_social_network)
        self.init_initial_state(parameters_social_network)
        self.init_fixed_preferences(parameters_social_network)
        self.init_cumulative_emissions(parameters_social_network)
        self.init_network_settings(parameters_social_network)
        self.init_learning_and_bias(parameters_social_network)
        self.init_preference_distribution(parameters_social_network)
        self.init_vehicle_and_transport(parameters_social_network)
        
        self.createVehicleUsers()
        
        # Create network and calculate initial emissions
        self.adjacency_matrix, self.network = self.create_network()
        self.network_density = nx.density(self.network)
        
        self.extract_VehicleUser_data()

        #record emissions
        self.total_carbon_emissions_flow = self.calc_total_emissions()
        self.total_carbon_emissions_cumulative = self.total_carbon_emissions_flow
        
        self.save_timeseries_data_state = parameters_social_network["save_timeseries_data_state"]
        if self.save_timeseries_data_state:
            self.set_up_time_series_social_network()

    def init_time_parameters(self, params):
        self.duration_no_OD_no_stock_no_policy = params["duration_no_OD_no_stock_no_policy"] 
        self.duration_OD_no_stock_no_policy = params["duration_OD_no_stock_no_policy"] 
        self.duration_OD_stock_no_policy = params["duration_OD_stock_no_policy"] 
        self.duration_OD_stock_policy = params["duration_OD_stock_policy"]
        self.policy_start_time = params["policy_start_time"]

    def init_initial_state(self, params):
        self.preference_drift_state = params["preference_drift_state"]
        self.fixed_preferences_state = params["fixed_preferences_state"]
        self.heterogenous_init_preferences = params["heterogenous_init_preferences"]
        self.emissions_flow_social_influence_state = params["emissions_flow_social_influence_state"]
        self.num_individuals = int(round(params["num_individuals"]))

    def init_fixed_preferences(self):
        if self.fixed_preferences_state: 
            self.fixed_preferences_state_instant = 1
        elif self.t_social_network < self.duration_no_OD_no_stock_no_policy:
            self.fixed_preferences_state_instant = 1
        else:
            self.fixed_preferences_state_instant = 0

    def init_cumulative_emissions(self, params):
        self.cumulative_emissions_preference_state = params["cumulative_emissions_preference_state"]
        if self.cumulative_emissions_preference_state:
            self.cumulative_emissions_preference_start_time = self.duration_no_OD_no_stock_no_policy + self.duration_OD_no_stock_no_policy
            if self.t_social_network < self.cumulative_emissions_preference_start_time:
                self.cumulative_emissions_preference_state_instant = 0
            else:
                self.cumulative_emissions_preference_state_instant = 1
            
            self.upsilon_E_center = params["upsilon_E"]
            self.heterogenous_reaction_cumulative_emissions_state = params["heterogenous_reaction_cumulative_emissions_state"]
            if self.heterogenous_reaction_cumulative_emissions_state:
                self.upsilon_E_std = params["upsilon_E_std"]
                self.upsilon_E_uncapped = np.random.normal(self.upsilon_E_center, self.upsilon_E_std, size=(self.num_individuals))
                self.upsilon_E = np.clip(self.upsilon_E_uncapped, 0, 1)
            else:
                self.upsilon_E = self.upsilon_E_center

    def init_network_settings(self, params):
        np.random.seed(params["init_vals_seed"])  # Initialize random seed
        self.network_structure_seed = params["network_structure_seed"]
        self.init_vals_seed = params["init_vals_seed"]
        self.preference_drift_seed = int(round(params["preference_drift_seed"]))
        self.network_density_input = params["network_density"]
        self.K_social_network = int(round((self.num_individuals - 1) * self.network_density_input))  # Calculate number of links
        self.prob_rewire = params["prob_rewire"]

    def init_learning_and_bias(self, params):
        self.upsilon = params["upsilon"]
        self.consumption_imitation_state = params["consumption_imitation_state"]
        self.confirmation_bias = params["confirmation_bias"]
        if self.preference_drift_state:
            self.preference_drift_std = params["preference_drift_std"]
            self.clipping_epsilon = params["clipping_epsilon"]
        else:
            self.preference_drift_std = 0
            self.clipping_epsilon = 0

    def init_preference_distribution(self, params):
        self.a_preferences = params["a_preferences"]
        self.b_preferences = params["b_preferences"]
        self.environmental_preference_arr = np.random.beta(self.a_preferences, self.b_preferences, size=self.num_individuals)
        self.a_innovativeness = params["a_innovativeness"]
        self.b_innovativeness = params["b_innovativeness"]
        self.innovativeness_arr_init = np.random.beta(self.a_innovativeness, self.b_innovativeness, size=self.num_individuals)
        self.ev_adoption_state_arr = np.zeros(self.num_individuals)
        self.a_price = params["a_price"]
        self.b_price = params["b_price"]
        self.price_preference_arr = np.random.beta(self.a_price, self.b_price, size=self.num_individuals)

    def init_vehicle_and_transport(self, params):
        self.markup = params["markup"]
        self.delta = params["delta"]
        self.kappa = params["kappa"]
        self.init_car_vec = params["init_car_vec"]
        self.utility_boost_const = params["utility_boost_const"]
        self.price_constant = params["price_constant"]
        self.init_public_transport_state = params["init_public_transport_state"]

        #Import the already made "cars" for urban and rural public transport
        self.rural_public_transport = params["rural_public_transport"]
        self.urban_public_transport = params["urban_public_transport"]

        if self.init_public_transport_state:
            self.car_owned_vec = np.asarray([self.public_option] * self.num_individuals)
            self.new_car_bool_vec = self.decide_purchase(self.init_car_vec)
        else:
            self.car_owned_vec = np.asarray([None] * self.num_individuals)
            replacement_candidate_vec, _ = self.choose_replacement_candidate(self.init_car_vec)
            self.new_car_bool_vec = np.ones(self.num_individuals)
            self.car_owned_vec = replacement_candidate_vec
            self.car_age_vec = np.zeros(self.num_individuals)
    
    def createVehicleUsers(self):
        self.vehicleUsers_list = users  # List of VehicleUser objects
             
    def normalize_vector_sum(self, vec):
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

        adjacency_matrix = nx.to_numpy_array(self.network)
        self.sparse_adjacency_matrix = sp.csr_matrix(adjacency_matrix)
        # Get the non-zero indices of the adjacency matrix
        self.row_indices_sparse, self.col_indices_sparse = self.sparse_adjacency_matrix.nonzero()

        self.network_density = nx.density(self.network)
        return adjacency_matrix, network

    def calc_total_emissions(self) -> int:
        total_network_emissions = sum([(1-car.environmental_score) for car in self.car_owned_vec])
        return total_network_emissions

    def extract_VehicleUser_data(self):
        # Extract user attributes into vectors for efficient processing
        self.chi_vector = np.array([user.chi for user in self.vehicleUsers_list])  # Innovation thresholds
        self.gamma_vector = np.array([user.gamma for user in self.vehicleUsers_list])  # Environmental concern
        self.beta_vector = np.array([user.beta for user in self.vehicleUsers_list])  # Cost sensitivity
        self.vehicle_type_vector = np.array([user.current_vehicle_type for user in self.vehicleUsers_list])  # Current vehicle types
        self.origin_vector = np.array([user.origin for user in self.vehicleUsers_list])  # Urban/rural origin
    
    def calculate_ev_adoption(self, ev_type=3):
        # Create a binary matrix where 1 indicates neighbors using EVs
        ev_adoption_matrix = np.where(self.vehicle_type_vector == ev_type, 1, 0)
        
        # Calculate the proportion of neighbors with EVs for each user (matrix multiplication)
        ev_neighbors = np.dot(self.adjacency_matrix, ev_adoption_matrix)
        total_neighbors = np.sum(self.adjacency_matrix, axis=1)
        proportion_ev_neighbors = np.divide(ev_neighbors, total_neighbors, where=total_neighbors != 0)

        # Determine whether each user considers buying an EV (1 if proportion of EVs in neighborhood ≥ chi)
        consider_ev_matrix = (proportion_ev_neighbors >= self.chi_vector).astype(int)
        
        return consider_ev_matrix

    def update_vehicle_choices(self, ev_type=3):
        # Determine who considers an EV based on their neighbors and chi thresholds
        consider_ev_matrix = self.calculate_ev_adoption(ev_type=ev_type)

        # Update the vehicle type vector for those considering EVs and switch their vehicle type
        self.vehicle_type_vector = np.where(consider_ev_matrix == 1, ev_type, self.vehicle_type_vector)
        
        # Optionally, update the users' current vehicle types
        for i, user in enumerate(self.vehicleUsers_list):
            user.current_vehicle_type = self.vehicle_type_vector[i]

    def calc_bought_firm_car_count(self):

        # Initialize a defaultdict for counting car purchases by firm
        firm_car_count = defaultdict(nested_defaultdict)

        for bought_new, car in zip(self.new_car_bool_vec, self.car_owned_vec):
            if bought_new:
                firm_car_count[car.firm_id][car.id] += 1

        return firm_car_count

    #########################################################################
    #CHAT GPT ATTEMPT AT CONSUMPTION
    def utility_buy_matrix(self, car_attributes_matrix):
        """UPDATE THIS TO THE NEW ONE - CAR ATTRIBUTES SHOULD BE ALREADY LIMITED BASED ON THE INNOVATIVENESS OF INDIVIDUAL"""
        low_carbon_preference_matrix = self.low_carbon_preference_arr[:, np.newaxis]
        gamma_vals_matrix = self.gamma_vals[:, np.newaxis]
        price = (1 + self.markup) * car_attributes_matrix[:,0] + self.carbon_price*(1-car_attributes_matrix[:,1])
        #print("price comp",(1 + self.markup) * car_attributes_matrix[0] , self.carbon_price*(1-car_attributes_matrix[1]))
        utilities = low_carbon_preference_matrix*car_attributes_matrix[:,1] + gamma_vals_matrix*car_attributes_matrix[:,2] - self.price_constant * price

        return utilities + self.utility_boost_const

    def utility_buy_vec(self, car_attributes_matrix):
        low_carbon_preference_matrix = self.low_carbon_preference_arr[:, np.newaxis]
        gamma_vals_matrix = self.gamma_vals[:, np.newaxis]
        price = (1 + self.markup) * car_attributes_matrix[0] + self.carbon_price*(1-car_attributes_matrix[1])
        utilities = low_carbon_preference_matrix*car_attributes_matrix[1] + gamma_vals_matrix * car_attributes_matrix[2] - self.price_constant * price
  
        return utilities + self.utility_boost_const
    
    def utility_keep(self, cars_owned_attributes_matrix):
        utilities = (self.low_carbon_preference_arr*cars_owned_attributes_matrix[:,1] + self.gamma_vals*cars_owned_attributes_matrix[:,2] + self.utility_boost_const)*(1 - self.delta) ** self.car_age_vec
        return utilities 

    def choose_replacement_candidate(self, cars):
        car_attributes_matrix = np.asarray([x.attributes_fitness for x in cars])  # ARRAY OF ALL THE CARS
        
        utilities_matrix = self.utility_buy_matrix(car_attributes_matrix)  # FOR EACH INDIVIDUAL WHAT IS THE UTILITY OF THE DIFFERENT CARS
        utilities_matrix[utilities_matrix < 0] = 0  # IF NEGATIVE UTILITY PUT IT AT 0
        
        # Calculate the denominator vector
        denominator_vec = np.sum(utilities_matrix ** self.kappa, axis=1)
        
        # Create a boolean mask for non-zero denominators
        non_zero_denominator_mask = denominator_vec != 0
        
        # Initialize probabilities matrix with zeros
        probabilities = np.zeros_like(utilities_matrix)
        
        # Calculate probabilities for non-zero denominators
        probabilities[non_zero_denominator_mask] = (utilities_matrix[non_zero_denominator_mask] ** self.kappa) / denominator_vec[non_zero_denominator_mask, np.newaxis]
        
        # Handle cases where the denominator is zero by assigning uniform probabilities
        zero_denominator_indices = np.where(denominator_vec == 0)[0]
        probabilities[zero_denominator_indices] = 1.0 / len(cars)
        
        # Create cumulative probabilities for each individual
        cumulative_probabilities = np.cumsum(probabilities, axis=1)#THIS MIGHT NEED TO BE IN THE OTHER AXIS
        
        # Generate random values for each individual
        random_values = np.random.rand(self.num_individuals, 1)
        
        # Select indices based on cumulative probabilities
        #IDEA OF USING CUMULATIVE PROBABILITIES IS THAT I DONT NEED TO USE A FOR LOOP
        replacement_index_vec = (cumulative_probabilities > random_values).argmax(axis=1)# CHECK IF THIS WORKS HOW I THINK IT DOES

        replacement_candidate_vec = cars[replacement_index_vec]
        utility_replacement_vec = utilities_matrix[np.arange(self.num_individuals), replacement_index_vec]#ADVANCED INDEXING; THE ARANGE LOOPS THROUGH EACH ROW AND THEN PICKS OUT THE COLUMN, SELECTING USING TWO 1D VECTORS DOES STUFF ELEMENT WISE

        return replacement_candidate_vec, utility_replacement_vec

    def decide_purchase(self, cars):

        replacement_candidate_vec, utility_replacement_vec = self.choose_replacement_candidate(cars)

        # Create utility_old_vec based on whether omega is None or not
        if self.init_public_transport_state:
            has_car_mask = np.asarray([car is not self.public_option for car in self.car_owned_vec])
            cars_owned_attributes_matrix = np.asarray([car.attributes_fitness for car in self.car_owned_vec])#calc extra utility but throw it away
            
            utility_old_vec = np.zeros(self.num_individuals)

            #print(has_car_mask)
            if self.t_social_network > 0:
                utility_old_vec[has_car_mask] = self.utility_keep(cars_owned_attributes_matrix)[has_car_mask] 
            
            #self.owned_car_utility_vec = utility_old_vec

            #NEED TO CALCULATE THE UTILITY OF PUBLIC TRANSPORT HERE
            
            utility_public = np.squeeze(self.utility_buy_vec(self.public_transport_attributes))
            # Stack the utility vectors into a single 2D array
            utilities = np.vstack([utility_public, utility_replacement_vec, utility_old_vec])


            # Find the index of the maximum utility along the first axis (i.e., across the rows)
            chosen_option_indices = np.argmax(utilities, axis=0)
            row_indices = np.arange(self.num_individuals)
            utilities_trans = utilities.T
            self.owned_car_utility_vec = utilities_trans[row_indices, chosen_option_indices]
            self.public_transport_prop = np.sum(chosen_option_indices == 0)/self.num_individuals#RECORD NUM PEOPLE PUBLIC TRANPORT

            # Determine the new boolean vector
            new_car_bool_vec = (chosen_option_indices == 1)  # 1 corresponds to new car

            # Update the car_owned_vec based on the chosen options
            self.car_owned_vec = np.where(
                chosen_option_indices == 0,  # 0 corresponds to public transport
                self.public_option,
                np.where(
                    chosen_option_indices == 1,  # 1 corresponds to new car
                    replacement_candidate_vec,
                    self.car_owned_vec
                )
            )
            self.car_owned_vec_no_public = np.where(
                chosen_option_indices == 0,  # 0 corresponds to public transport
                None,
                np.where(
                    chosen_option_indices == 1,  # 1 corresponds to new car
                    replacement_candidate_vec,
                    self.car_owned_vec
                )
            )

            # Update the car_age_vec based on the chosen options
            self.car_age_vec = np.where(
                chosen_option_indices == 0,  # Reset age for public transport
                0,
                np.where(
                    chosen_option_indices == 1,  # Reset age for new car
                    0,
                    self.car_age_vec + 1  # Increment age for keeping old car
                )
            )

            #NEED TO SORT OUT THE BOOL 
        else:
            has_car_mask = np.asarray([car is not None for car in self.car_owned_vec])
            cars_owned_attributes_matrix = np.asarray([car.attributes_fitness for car in self.car_owned_vec[has_car_mask]])

            utility_old_vec = np.zeros(self.num_individuals)

            utility_old_vec[has_car_mask] = self.utility_keep(cars_owned_attributes_matrix[has_car_mask])
            
            self.owned_car_utility_vec = utility_old_vec
            new_car_bool_vec = utility_replacement_vec > utility_old_vec
            self.car_owned_vec = np.where(new_car_bool_vec, replacement_candidate_vec, self.car_owned_vec)
            self.car_age_vec = np.where(new_car_bool_vec, 0, self.car_age_vec + 1)

        return new_car_bool_vec

    def update_burn_in_state(self):
        #
        if (self.t_social_network > self.duration_no_OD_no_stock_no_policy) and (not self.fixed_preferences_state):
            self.fixed_preferences_state_instant = 0 
        
        if (self.t_social_network > (self.duration_no_OD_no_stock_no_policy + self.duration_OD_no_stock_no_policy)) and self.cumulative_emissions_preference_state:
            self.cumulative_emissions_preference_state_instant = 1

    def set_up_time_series_social_network(self):
        self.history_utility_vec = [np.asarray([0]*self.num_individuals)]
        self.history_flow_carbon_emissions = [self.total_carbon_emissions_flow]
        self.history_cumulative_carbon_emissions = [self.total_carbon_emissions_cumulative]#FIX
        self.history_time_social_network = [self.t_social_network]
        self.history_firm_count = [self.firm_count]
        self.history_car_owned_vec = [self.car_owned_vec]
        
        #self.histor_raw_utility_buy_0 = [np.asarray([0]*30)]
        if self.init_public_transport_state:
            self.history_public_transport_prop = [self.public_transport_prop]
            self.history_car_owned_vec_no_public = [self.car_owned_vec_no_public]
    
    def save_timeseries_data_social_network(self):
        """
        Save time series data

        parameters_social_network
        ----------
        None

        Returns
        -------
        None
        """
        self.history_cumulative_carbon_emissions.append(self.total_carbon_emissions_cumulative)#THIS IS FUCKED#
        self.history_utility_vec.append(self.owned_car_utility_vec)
        self.history_flow_carbon_emissions.append(self.total_carbon_emissions_flow)
        self.history_time_social_network.append(self.t_social_network)
        self.history_firm_count.append(self.firm_count)
        self.history_car_owned_vec.append(self.car_owned_vec)
        #self.histor_raw_utility_buy_0.append(self.raw_utility_buy_0)
        if self.init_public_transport_state:
            self.history_public_transport_prop.append(self.public_transport_prop)
            self.history_car_owned_vec_no_public.append(self.car_owned_vec_no_public)

    def next_step(self, carbon_price, cars_on_sale_all_firms):
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
         
        self.update_burn_in_state()

        #update new tech and prices
        self.cars_on_sale_all_firms = cars_on_sale_all_firms

        #update adoption_proportion
        self.ev_adoption_state_arr = self.update_ev_adoption_proportion()
        self.cars_on_sale_all_firms_limited = self.select_cars_available()

        #decide to buy new cars
        self.new_car_bool = self.decide_purchase(self.cars_on_sale_all_firms)

        #calc consumption quantities
        self.firm_count = self.calc_bought_firm_car_count()

        #calc emissions
        self.total_carbon_emissions_flow = self.calc_total_emissions()
        self.total_carbon_emissions_cumulative = self.total_carbon_emissions_cumulative + self.total_carbon_emissions_flow
            
        if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
            self.save_timeseries_data_social_network()

        return self.ev_adoption_state_arr, self.environmental_preference_arr, self.price_preference_arr
