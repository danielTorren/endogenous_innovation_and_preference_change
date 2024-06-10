"""Create social network with individuals

Created: 10/10/2022
"""

# imports
import numpy as np
import networkx as nx
import numpy.typing as npt
from collections import defaultdict

# modules

# Define the default factory function outside the method
def nested_defaultdict():
    return defaultdict(int)

class Social_Network:

    def __init__(self, parameters_social_network: list):#FEED THE STUFF STRAIGHT THAT ISNT USED BY SOCIAL NETWORK
        """
        Constructs all the necessary attributes for the Network object.

        parameters_social_network
        ----------
        parameters_social_network : dict
            Dictionary of parameters_social_network used to generate attributes, dict used for readability instead of super long list of input parameters_social_network

        """
        self.t_social_network = 0
        

        #TIME 
        self.duration_no_OD_no_stock_no_policy = parameters_social_network["duration_no_OD_no_stock_no_policy"] 
        self.duration_OD_no_stock_no_policy = parameters_social_network["duration_OD_no_stock_no_policy"] 
        self.duration_OD_stock_no_policy = parameters_social_network["duration_OD_stock_no_policy"] 
        self.duration_OD_stock_policy = parameters_social_network["duration_OD_stock_policy"]
        self.policy_start_time = parameters_social_network["policy_start_time"]  

        #INITAL STATE OF THE SYSTEMS, WHAT ARE THE RUN CONDITIONS
        self.preference_drift_state = parameters_social_network["preference_drift_state"]
        self.fixed_preferences_state = parameters_social_network["fixed_preferences_state"]#DEAL WITH BURN IN
        self.heterogenous_init_preferences = parameters_social_network["heterogenous_init_preferences"]
        
        self.num_individuals = int(round(parameters_social_network["num_individuals"]))
        
        #FIXED PREFERENCES
        #NOTE THE DIFFERENCE BETWEEN THE STATE AND INSTANT STATE
        if self.fixed_preferences_state: 
            self.fixed_preferences_state_instant = 1
        elif self.t_social_network < self.duration_no_OD_no_stock_no_policy:
            self.fixed_preferences_state_instant = 1
        else:
            self.fixed_preferences_state_instant = 0
            
        #CUMULATIVE EMISSIONS
        #NOTE THE DIFFERENCE BETWEEN STATE AND INSTANT
        self.cumulative_emissions_preference_state = parameters_social_network["cumulative_emissions_preference_state"]
        if self.cumulative_emissions_preference_state: 
            self.cumulative_emissions_preference_start_time = self.duration_no_OD_no_stock_no_policy + self.duration_OD_no_stock_no_policy
            if self.t_social_network < self.cumulative_emissions_preference_start_time:
                self.cumulative_emissions_preference_state_instant = 0
            else:
                self.cumulative_emissions_preference_state_instant = 1
            
            self.upsilon_E_center = parameters_social_network["upsilon_E"]
            self.heterogenous_reaction_cumulative_emissions_state  = parameters_social_network["heterogenous_reaction_cumulative_emissions_state"]
            if self.heterogenous_reaction_cumulative_emissions_state: 
                self.upsilon_E_std = parameters_social_network["upsilon_E_std"]
                self.upsilon_E_uncapped = np.random.normal(self.upsilon_E_center, self.upsilon_E_std, size=(self.num_individuals))
                self.upsilon_E = np.clip(self.upsilon_E_uncapped, 0, 1)
            else:
                self.upsilon_E = self.upsilon_E_center


        self.save_timeseries_data_state = parameters_social_network["save_timeseries_data_state"]
        self.compression_factor_state = parameters_social_network["compression_factor_state"]

        #seeds
        self.network_structure_seed = parameters_social_network["network_structure_seed"] 
        self.init_vals_seed = parameters_social_network["init_vals_seed"] 
        self.preference_drift_seed = int(round(parameters_social_network["preference_drift_seed"]))

        #carbon price
        self.carbon_price = parameters_social_network["carbon_price"]

        #Emissiosn
        self.emissions_max = parameters_social_network["emissions_max"]#NEEDS TO BE SET ACCOUNTING FOR THE NUMBER OF TIME STEPS, PEOPLE AND CARBON INTENSITY, MAX = steps*people

        np.random.seed(self.init_vals_seed)#For inital construction set a seed, this is the same for all runs, then later change it to preference_drift_seed

        # network
        self.network_density_input = parameters_social_network["network_density"]
        self.K_social_network = int(round((self.num_individuals - 1)*self.network_density_input)) #reverse engineer the links per person using the density  d = 2m/n(n-1) where n is nodes and m number of edges
        self.prob_rewire = parameters_social_network["prob_rewire"]

        # social learning and bias
        self.upsilon = parameters_social_network["upsilon"]
        
        self.confirmation_bias = parameters_social_network["confirmation_bias"]
        if self.preference_drift_state:
            self.preference_drift_std = parameters_social_network["preference_drift_std"]
            self.clipping_epsilon = parameters_social_network["clipping_epsilon"]
        else:
            self.preference_drift_std = 0
            self.clipping_epsilon = 0     
        
        # create network
        (
            self.adjacency_matrix,
            self.weighting_matrix,
            self.network,
        ) = self.create_weighting_matrix()

        self.network_density = nx.density(self.network)

        if self.heterogenous_init_preferences:
            self.a_preferences = parameters_social_network["a_preferences"]
            self.b_preferences = parameters_social_network["b_preferences"]
            (
                self.low_carbon_preference_arr_init
            ) = self.generate_init_data_preferences()
        else:
            self.low_carbon_preference_arr_init = np.asarray([self.clipping_epsilon]*self.num_individuals)
        

        #NOW SET SEED FOR THE IMPERFECT LEARNING
        np.random.seed(self.preference_drift_seed)

        self.low_carbon_preference_arr = self.low_carbon_preference_arr_init
        self.low_carbon_preference_matrix = self.low_carbon_preference_arr[:, np.newaxis]

        if not self.fixed_preferences_state_instant:
            self.weighting_matrix = self.update_weightings()

        #FIX
        self.total_carbon_emissions_cumulative = 0#this are for post tax

        #CARS
        self.car_age_vec = np.zeros(self.num_individuals)
        self.car_owned_vec = np.asarray([None]*self.num_individuals)
        self.gamma = parameters_social_network["gamma"]  # weight for car quality
        self.markup = parameters_social_network["markup"]  # industry mark-up on production costs
        self.delta =  parameters_social_network["delta"]  # depreciation rate
        self.kappa = parameters_social_network["kappa"]  # parameter indicating consumers' ability to make rational choices
        self.init_car_vec = parameters_social_network["init_car_vec"]

        #calc consumption quantities
        replacement_candidate_vec, _ = self.choose_replacement_candidate(self.init_car_vec)
        self.new_car_bool_vec = np.ones(self.num_individuals)
        self.car_owned_vec = replacement_candidate_vec
        self.car_age_vec = np.zeros(self.num_individuals)   
        
        self.firm_count = self.calc_bought_firm_car_count()

        self.total_carbon_emissions_flow = self.calc_total_emissions()
        self.total_carbon_emissions_cumulative = self.total_carbon_emissions_cumulative + self.total_carbon_emissions_flow

        if self.save_timeseries_data_state:
            self.set_up_time_series_social_network()
    
    def normalize_vector_sum(self, vec):
        return vec/sum(vec)
    
    def normlize_matrix(self, matrix: npt.NDArray) -> npt.NDArray:
        """
        Row normalize an array

        parameters_social_network
        ----------
        matrix: npt.NDArrayf
            array to be row normalized

        Returns
        -------
        norm_matrix: npt.NDArray
            row normalized array
        """
        row_sums = matrix.sum(axis=1)
        norm_matrix = matrix / row_sums[:, np.newaxis]

        return norm_matrix

    def create_weighting_matrix(self) -> tuple[npt.NDArray, npt.NDArray, nx.Graph]:
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

        G = nx.watts_strogatz_graph(n=self.num_individuals, k=self.K_social_network, p=self.prob_rewire, seed=self.network_structure_seed)#FIX THE NETWORK STRUCTURE

        weighting_matrix = nx.to_numpy_array(G)

        norm_weighting_matrix = self.normlize_matrix(weighting_matrix)

        return (
            weighting_matrix,
            norm_weighting_matrix,
            G,
        )
        
    def generate_init_data_preferences(self) -> tuple[npt.NDArray, npt.NDArray]:
        low_carbon_preference_arr = np.random.beta( self.a_preferences, self.b_preferences, size=self.num_individuals)
        return low_carbon_preference_arr


##################################################################################################################
    #IMPORTED FROM PAPER 2 VECTORISED SO IT WILL RUN FAST!
    #UPDATE PREFERENCES

    def update_preferences(self):
        social_influence = np.matmul(self.weighting_matrix, self.low_carbon_preference_arr)

        #idea is that noise is constant proportion of signal over timer
        if self.cumulative_emissions_preference_state_instant:
            cumulative_emissions_influence = self.total_carbon_emissions_cumulative/self.emissions_max
            low_carbon_preferences = (1 - self.upsilon)*self.low_carbon_preference_arr + self.upsilon*((1 - self.upsilon_E) * social_influence + self.upsilon_E*cumulative_emissions_influence) + np.random.normal(0, self.preference_drift_std, size=(self.num_individuals))  # Gaussian noise
        else:
            low_carbon_preferences = (1 - self.upsilon)*self.low_carbon_preference_arr + self.upsilon*social_influence + np.random.normal(0, self.preference_drift_std, size=(self.num_individuals))  # Gaussian noise
        
        low_carbon_preferences  = np.clip(low_carbon_preferences, 0 + self.clipping_epsilon, 1- self.clipping_epsilon)#this stops the guassian error from causing A to be too large or small thereby producing nans
        
        return low_carbon_preferences

    #UPDATE WEIGHTING
    def update_weightings(self):

        difference_matrix = np.subtract.outer(self.low_carbon_preference_arr, self.low_carbon_preference_arr) 
        alpha_numerator = np.exp(-np.multiply(self.confirmation_bias, np.abs(difference_matrix)))

        non_diagonal_weighting_matrix = (
            self.adjacency_matrix*alpha_numerator
        )  # We want onlythose values that have network connections

        norm_weighting_matrix = self.normlize_matrix(
            non_diagonal_weighting_matrix
        )  # normalize the matrix row wise
    
        return norm_weighting_matrix
    
##################################################################################################################

    def calc_total_emissions(self) -> int:
        total_network_emissions = sum([(1-car.emissions) for car in self.car_owned_vec])
        return total_network_emissions

    def calc_bought_firm_car_count(self):

        # Initialize a defaultdict for counting car purchases by firm
        firm_car_count = defaultdict(nested_defaultdict)

        # Iterate over the agents and update the purchase counts
        #print(self.new_car_bool_vec, self.car_owned_vec)
        #print(self.new_car_bool_vec.shape, self.car_owned_vec.shape)
        #quit()
        for bought_new, car in zip(self.new_car_bool_vec, self.car_owned_vec):
            if bought_new:
                firm_car_count[car.firm_id][car.id] += 1

        return firm_car_count


    #########################################################################
    #CHAT GPT ATTEMPT AT CONSUMPTION
    def utility_buy_matrix(self, car_attributes_matrix):
        low_carbon_preference_matrix = self.low_carbon_preference_arr[:, np.newaxis]
        utilities = low_carbon_preference_matrix*car_attributes_matrix[:,1] + (1 -  low_carbon_preference_matrix) * (self.gamma * car_attributes_matrix[:,2] - (1 - self.gamma) * ((1 + self.markup) * car_attributes_matrix[:,0] + self.carbon_price*car_attributes_matrix[:,1]))
        return utilities
    
    def utility_keep(self, cars_owned_attributes_matrix):
        utilities = (self.low_carbon_preference_arr*cars_owned_attributes_matrix[:,1] + (1 - self.low_carbon_preference_arr) * cars_owned_attributes_matrix[:,2])*(1 - self.delta) ** self.car_age_vec
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
        has_car_mask = np.asarray([car is not None for car in self.car_owned_vec])
        cars_owned_attributes_matrix = np.asarray([car.attributes_fitness for car in self.car_owned_vec[has_car_mask]])
        #print("cars_owned_attributes_matrix",cars_owned_attributes_matrix)
        utility_old_vec = np.zeros(self.num_individuals)
        #print(self.t_social_network, has_car_mask)
        #print(cars_owned_attributes_matrix)
        #a = cars_owned_attributes_matrix[has_car_mask]
        #print(a)
        #util_car_kept = self.utility_keep(cars_owned_attributes_matrix[has_car_mask])
        #print(util_car_kept.shape)
        #print("DONE")
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
        self.history_preference_list = [self.low_carbon_preference_arr]
        self.history_flow_carbon_emissions = [self.total_carbon_emissions_flow]
        self.history_cumulative_carbon_emissions = [self.total_carbon_emissions_cumulative]#FIX
        self.history_time_social_network = [self.t_social_network]
        self.history_firm_count = [self.firm_count]
        self.history_car_owned_vec = [self.car_owned_vec]

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
        self.history_cumulative_carbon_emissions.append(self.total_carbon_emissions_cumulative)#THIS IS FUCKED
        self.history_flow_carbon_emissions.append(self.total_carbon_emissions_flow)
        self.history_preference_list.append(self.low_carbon_preference_arr)
        self.history_time_social_network.append(self.t_social_network)
        self.history_firm_count.append(self.firm_count)
        self.history_car_owned_vec.append(self.car_owned_vec)

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

        #update preferences 
        if not self.fixed_preferences_state_instant:
            self.weighting_matrix = self.update_weightings()#UNSURE WHAT THE ORDER SHOULD BE HERE
            self.low_carbon_preference_arr = self.update_preferences()

        #decide to buy new cars
        self.new_car_bool = self.decide_purchase(self.cars_on_sale_all_firms)

        #calc consumption quantities
        self.firm_count = self.calc_bought_firm_car_count()

        #calc emissions
        self.total_carbon_emissions_flow = self.calc_total_emissions()
        self.total_carbon_emissions_cumulative = self.total_carbon_emissions_cumulative + self.total_carbon_emissions_flow
            
        if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
            self.save_timeseries_data_social_network()

        return self.low_carbon_preference_arr
