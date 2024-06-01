"""Create social network with individuals

Created: 10/10/2022
"""

# imports
import numpy as np
import networkx as nx
import numpy.typing as npt
from package.model.individual import Individual
from collections import defaultdict

# modules

# Define the default factory function outside the method
def nested_defaultdict():
    return defaultdict(int)

class Social_Network:

    def __init__(self, parameters_social_network: list, parameters_individual):#FEED THE STUFF STRAIGHT THAT ISNT USED BY SOCIAL NETWORK
        """
        Constructs all the necessary attributes for the Network object.

        parameters_social_network
        ----------
        parameters_social_network : dict
            Dictionary of parameters_social_network used to generate attributes, dict used for readability instead of super long list of input parameters_social_network

        """
        self.t_social_network = 0
        
        self.parameters_individual = parameters_individual
        #TIME 
        self.burn_in_no_OD = parameters_social_network["burn_in_no_OD"] 
        self.burn_in_duration_no_policy = parameters_social_network["burn_in_duration_no_policy"] 
        self.policy_duration = parameters_social_network["policy_duration"]

        #INITAL STATE OF THE SYSTEMS, WHAT ARE THE RUN CONDITIONS
        self.imperfect_learning_state = parameters_social_network["imperfect_learning_state"]
        self.fixed_preferences_state = parameters_social_network["fixed_preferences_state"]#DEAL WITH BURN IN

        if self.fixed_preferences_state: 
            self.fixed_preferences_state_instant = 1
        elif self.t_social_network < self.burn_in_no_OD:
            self.fixed_preferences_state_instant = 1
        else:
            self.fixed_preferences_state_instant = 0

        self.save_timeseries_data_state = parameters_social_network["save_timeseries_data_state"]
        self.compression_factor_state = parameters_social_network["compression_factor_state"]

        #seeds
        self.network_structure_seed = parameters_social_network["network_structure_seed"] 
        self.init_vals_seed = parameters_social_network["init_vals_seed"] 
        self.imperfect_learning_seed = int(round(parameters_social_network["imperfect_learning_seed"]))

        #carbon price
        self.carbon_price = parameters_social_network["carbon_price"]

        #Emissiosn
        self.emissions_max = parameters_social_network["emissions_max"]#NEEDS TO BE SET ACCOUNTING FOR THE NUMBER OF TIME STEPS, PEOPLE AND CARBON INTENSITY, MAX = steps*people

        np.random.seed(self.init_vals_seed)#For inital construction set a seed, this is the same for all runs, then later change it to imperfect_learning_seed

        # network
        self.network_density_input = parameters_social_network["network_density"]
        self.num_individuals = int(round(parameters_social_network["num_individuals"]))
        self.K_social_network = int(round((self.num_individuals - 1)*self.network_density_input)) #reverse engineer the links per person using the density  d = 2m/n(n-1) where n is nodes and m number of edges
        self.prob_rewire = parameters_social_network["prob_rewire"]

        # social learning and bias
        self.upsilon = parameters_social_network["upsilon"]
        self.upsilon_E = parameters_social_network["upsilon_E"]
        self.confirmation_bias = parameters_social_network["confirmation_bias"]
        if self.imperfect_learning_state:
            self.preference_drift_std = parameters_social_network["preference_drift_std"]
            self.clipping_epsilon = parameters_social_network["clipping_epsilon"]
        else:
            self.preference_drift_std = 0
            self.clipping_epsilon = 0     

        # network homophily
        self.homophily = parameters_social_network["homophily"]  # 0-1
        self.shuffle_reps = int(
            round(self.num_individuals*((1 - self.homophily)**1.5))#1.5 to just make the mixing stronger
        )
        
        # create network
        (
            self.adjacency_matrix,
            self.weighting_matrix,
            self.network,
        ) = self.create_weighting_matrix()

        self.network_density = nx.density(self.network)

        self.a_preferences = parameters_social_network["a_preferences"]
        self.b_preferences = parameters_social_network["b_preferences"]
        (
            self.low_carbon_preference_arr_init
        ) = self.generate_init_data_preferences()
            
        self.agent_list = self.create_agent_list()
        self.shuffle_agent_list()#partial shuffle of the list based on prefernece

        #NOW SET SEED FOR THE IMPERFECT LEARNING
        np.random.seed(self.imperfect_learning_seed)

        self.low_carbon_preference_arr = self.low_carbon_preference_arr_init
        self.weighting_matrix = self.update_weightings()

        #FIX
        self.total_carbon_emissions_cumulative = 0#this are for post tax

        #calc consumption quantities
        self.firm_count = self.calc_consumption_vec()

        #print(self.agent_list[0].omega)
        #quit()
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
    
    def circular_agent_list(self) -> list:
        """
        Makes an ordered list circular so that the start and end values are matched in value and value distribution is symmetric

        parameters_social_network
        ----------
        list: list
            an ordered list e.g [1,2,3,4,5]
        Returns
        -------
        circular: list
            a circular list symmetric about its middle entry e.g [1,3,5,4,2]
        """

        first_half = self.agent_list[::2]  # take every second element in the list, even indicies
        second_half = (self.agent_list[1::2])[::-1]  # take every second element , odd indicies
        self.agent_list = first_half + second_half

    def partial_shuffle_agent_list(self) -> list:
        """
        Partially shuffle a list using Fisher Yates shuffle
        """

        for _ in range(self.shuffle_reps):
            a, b = np.random.randint(
                low=0, high=self.num_individuals, size=2
            )  # generate pair of indicies to swap
            self.agent_list[b], self.agent_list[a] = self.agent_list[a], self.agent_list[b]
    
    def generate_init_data_preferences(self) -> tuple[npt.NDArray, npt.NDArray]:
        low_carbon_preference_arr = np.random.beta( self.a_preferences, self.b_preferences, size=self.num_individuals)
        return low_carbon_preference_arr

    def create_agent_list(self) -> list[Individual]:
        """
        Create list of Individual objects that each have behaviours

        parameters_social_network
        ----------
        None

        Returns 
        -------
        agent_list: list[Individual]
            List of Individual objects 
        """

        agent_list = [
            Individual(
                self.parameters_individual,
                self.low_carbon_preference_arr_init[n],
                n
            )
            for n in range(self.num_individuals)
        ]

        return agent_list
        
    def shuffle_agent_list(self): 
        #make list cirucalr then partial shuffle it
        self.agent_list.sort(key=lambda x: x.low_carbon_preference)#sorted by preference
        self.circular_agent_list()#agent list is now circular in terms of preference
        self.partial_shuffle_agent_list()#partial shuffle of the list


##################################################################################################################
    #IMPORTED FROM PAPER 2 VECTORISED SO IT WILL RUN FAST!
    #UPDATE PREFERENCES

    def update_preferences(self):
        social_influence = np.matmul(self.weighting_matrix, self.low_carbon_preference_arr)

        cumulative_emissions_influence = self.total_carbon_emissions_cumulative/self.emissions_max

        low_carbon_preferences = (1 - self.upsilon)*self.low_carbon_preference_arr + self.upsilon*((1 - self.upsilon_E) * social_influence + self.upsilon_E*cumulative_emissions_influence) + np.random.normal(0, self.preference_drift_std, size=(self.num_individuals))  # Gaussian noise
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
        total_network_emissions = sum([x.omega.emissions for x in self.agent_list])
        return total_network_emissions

    def calc_consumption_vec(self):
        # Extract the new car boolean vector and car vector from the agent list
        new_car_bool_vec = [x.new_car_bool for x in self.agent_list]
        car_owned_list = [x.omega for x in self.agent_list]

        # Initialize a defaultdict for counting car purchases by firm
        firm_car_count = defaultdict(nested_defaultdict)

        # Iterate over the agents and update the purchase counts
        for bought_new, car in zip(new_car_bool_vec, car_owned_list):
            if bought_new:
                firm_car_count[car.firm_id][car.id] += 1

        return firm_car_count


    def update_individuals(self):
        """
        Update Individual objects with new information regarding social interactions, prices and dividend
        """

        # Assuming you have self.agent_list as the list of objects
        for i, agent in enumerate(self.agent_list):
            agent.next_step(self.low_carbon_preference_arr[i], self.cars_on_sale_all_firms, self.carbon_price)
    
    def update_burn_in_OD(self):
        if (self.t_social_network > self.burn_in_no_OD) and (not self.fixed_preferences_state):
            self.fixed_preferences_state_instant = 0 

    def set_up_time_series_social_network(self):
        if not self.fixed_preferences_state:
            self.history_preference_list = [self.low_carbon_preference_arr]
        self.history_flow_carbon_emissions = [self.total_carbon_emissions_flow]
        self.history_cumulative_carbon_emissions = [self.total_carbon_emissions_cumulative]#FIX
        self.history_time_social_network = [self.t_social_network]
        self.history_firm_count = [self.firm_count]

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
        if not self.fixed_preferences_state:
            self.history_preference_list.append(self.low_carbon_preference_arr)
        self.history_time_social_network.append(self.t_social_network)
        self.history_firm_count.append(self.firm_count)

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
         
        self.update_burn_in_OD()

        #update new tech and prices
        self.cars_on_sale_all_firms = cars_on_sale_all_firms

        #update preferences 
        if not self.fixed_preferences_state:
            self.weighting_matrix = self.update_weightings()#UNSURE WHAT THE ORDER SHOULD BE HERE
            self.low_carbon_preference_arr = self.update_preferences()

        # execute step
        self.update_individuals()

        #calc consumption quantities
        self.firm_count = self.calc_consumption_vec()

        #calc emissions
        self.total_carbon_emissions_flow = self.calc_total_emissions()
        self.total_carbon_emissions_cumulative = self.total_carbon_emissions_cumulative + self.total_carbon_emissions_flow
            
        if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
            self.save_timeseries_data_social_network()

        return self.low_carbon_preference_arr
