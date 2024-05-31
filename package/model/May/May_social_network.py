"""Create social network with individuals
A module that use input data to generate a social network containing individuals who each have multiple 
behaviours. The weighting of individuals within the social network is determined by the preference distance 
between neighbours. The simulation evolves over time saving data at set intervals to reduce data output.


Created: 10/10/2022
"""


# imports
import numpy as np
import networkx as nx
import numpy.typing as npt
from package.model.individual import Individual
from operator import attrgetter
from collections import defaultdict

# modules
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

        np.random.seed(self.init_vals_seed)#For inital construction set a seed, this is the same for all runs, then later change it to imperfect_learning_seed
        

        self.segment_number = int(parameters_social_network["segment_number"])
        self.segement_preference_bounds = np.linspace(0, 1, self.segment_number+1) 

        # network
        self.network_density_input = parameters_social_network["network_density"]
        self.num_individuals = int(round(parameters_social_network["num_individuals"]))
        self.K_social_network = int(round((self.num_individuals - 1)*self.network_density_input)) #reverse engineer the links per person using the density  d = 2m/n(n-1) where n is nodes and m number of edges
        #print("self.K",self.K)
        self.prob_rewire = parameters_social_network["prob_rewire"]

        #firms stuff
        self.num_firms = parameters_social_network["J"]
        self.emissions_intensities_vec = parameters_social_network["emissions_intensities_vec"]
        #price
        self.prices_vec =  parameters_social_network["prices_vec"]#THIS NEEDS TO BE SET AFTER THE STUFF IS RUN.
        
        self.carbon_price_state = parameters_social_network["carbon_price_state"]
        self.carbon_price_policy = parameters_social_network["carbon_price"]

        #print("self.carbon_price_policy",self.carbon_price_policy)
        #quit()

        if self.carbon_price_state == "AR1":
                self.ar_1_coefficient = parameters_social_network["ar_1_coefficient"] 
                self.noise_mean = parameters_social_network["noise_mean"]  
                self.noise_sigma = parameters_social_network["noise_sigma"] 
                self.carbon_price_AR1 = self.generate_ar1(self.carbon_price_policy,self.ar_1_coefficient, self.noise_mean, self.noise_sigma, self.policy_duration+2)
        elif self.carbon_price_state == "normal":
            self.noise_sigma = parameters_social_network["noise_sigma"] 
            self.carbon_price_normal = np.random.normal(self.carbon_price_policy,self.noise_sigma, size = self.policy_duration + 2)

        if self.t_social_network == self.burn_in_no_OD + self.burn_in_duration_no_policy:
            if self.carbon_price_state == "flat":
                self.carbon_price = self.carbon_price_policy
            elif self.carbon_price_state == "AR1":
                self.carbon_price = self.carbon_price_AR1[0]
            elif self.carbon_price_state == "normal":
                self.carbon_price = self.carbon_price_normal[0]
        else:
            self.carbon_price = 0

        # social learning and bias
        self.confirmation_bias = parameters_social_network["confirmation_bias"]
        if self.imperfect_learning_state:
            self.std_learning_error = parameters_social_network["std_learning_error"]
            self.clipping_epsilon = parameters_social_network["clipping_epsilon"]
        else:
            self.std_learning_error = 0
            self.clipping_epsilon = 0     

        self.clipping_epsilon_init_preference = parameters_social_network["clipping_epsilon_init_preference"]

        # network homophily
        self.homophily = parameters_social_network["homophily"]  # 0-1
        self.shuffle_reps = int(
            round(self.num_individuals*(1 - self.homophily))
        )
        
        # create network
        (
            self.adjacency_matrix,
            self.weighting_matrix,
            self.network,
        ) = self.create_weighting_matrix()

        self.network_density = nx.density(self.network)

        self.a_preferences = parameters_social_network["a_preferences"]#A #IN THIS BRANCH CONSISTEN BEHAVIOURS USE THIS FOR THE IDENTITY DISTRIBUTION
        self.b_preferences = parameters_social_network["b_preferences"]#A #IN THIS BRANCH CONSISTEN BEHAVIOURS USE THIS FOR THE IDENTITY DISTRIBUTION
        #self.shift_preferences = parameters_social_network["shift_preferences"]
        self.std_low_carbon_preference = parameters_social_network["std_low_carbon_preference"]
        (
            self.low_carbon_preference_arr_init
        ) = self.generate_init_data_preferences()
        self.preference_mul = parameters_social_network["preference_mul"]
        self.low_carbon_preference_arr_init = self.low_carbon_preference_arr_init*self.preference_mul
        #print("MEAN vals", sum(self.low_carbon_preference_arr_init))
        #print("network", self.adjacency_matrix[0])
        #quit()
        
        self.heterogenous_expenditure_state = parameters_social_network["heterogenous_expenditure_state"]
        self.total_expenditure = parameters_social_network["total_expenditure"]
        if self.heterogenous_expenditure_state:
            self.expenditure_inequality_const = parameters_social_network["expenditure_inequality_const"]
            u = np.linspace(0.01,1,self.num_individuals)
            no_norm_individual_expenditure_array = u**(-1/self.expenditure_inequality_const)       
            self.individual_expenditure_array =  self.total_expenditure*self.normalize_vector_sum(no_norm_individual_expenditure_array)
        else:   
            self.individual_expenditure_array =  np.asarray([self.total_expenditure/(self.num_individuals)]*self.num_individuals)#sums to 1
        
        self.substitutability = parameters_social_network["substitutability"]
        self.heterogenous_substitutability_state = parameters_social_network["heterogenous_substitutability_state"]
        if self.heterogenous_substitutability_state:
            self.std_substitutability = parameters_social_network["std_substitutability"]
            self.substitutability_vec = np.clip(np.random.normal(loc = self.substitutability , scale = self.std_substitutability, size = self.num_individuals), 1.05, None)#limit between 0.01
        else:
            self.substitutability_vec = np.asarray([self.substitutability]*self.num_individuals)
            
        self.heterogenous_emissions_intensity_penalty_state = parameters_social_network["heterogenous_emissions_intensity_penalty_state"]
        self.emissions_intensity_penalty = parameters_social_network["emissions_intensity_penalty"]
        if self.heterogenous_emissions_intensity_penalty_state:
            self.std_emissions_intensity_penalty = parameters_social_network["std_emissions_intensity_penalty"]
            self.emissions_intensity_penalty_vec = np.clip(np.random.normal(loc = self.emissions_intensity_penalty , scale = self.std_emissions_intensity_penalty, size = self.num_individuals), 0.01, None)#limit between 0.01
        else:
            self.emissions_intensity_penalty_vec = np.asarray([self.emissions_intensity_penalty]*self.num_individuals)


        self.agent_list = self.create_agent_list()
        #self.low_carbon_preference_arr = [x.low_carbon_preference for x in self.agent_list]#list(map(attrgetter("low_carbon_preference"), self.agent_list))

        self.shuffle_agent_list()#partial shuffle of the list based on prefernece

        #NOW SET SEED FOR THE IMPERFECT LEARNING
        np.random.seed(self.imperfect_learning_seed)

        if self.fixed_preferences_state_instant:
            self.social_component_matrix = np.asarray([n.low_carbon_preference for n in self.agent_list])#DUMBY FEED IT ITSELF? DO I EVEN NEED TO DEFINE IT
        else:
            self.weighting_matrix = self.update_weightings()
            self.social_component_matrix = self.calc_social_component_matrix()

        #FIX
        self.total_carbon_emissions_cumulative = 0#this are for post tax

        #calc consumption quantities
        self.consumption_matrix, self.consumed_quantities_vec, self.consumed_quantities_vec_firms = self.calc_consumption_vec()

        self.redistribution_state = parameters_social_network["redistribution_state"]
        #redistribute, for next turn
        if self.redistribution_state:
            self.carbon_dividend_array = self.calc_carbon_dividend_array()
        else:
            self.carbon_dividend_array = np.asarray([0]*self.num_individuals)

        self.total_carbon_emissions_flow = self.calc_total_emissions()
        self.total_carbon_emissions_cumulative = self.total_carbon_emissions_cumulative + self.total_carbon_emissions_flow

        if self.save_timeseries_data_state:
            self.set_up_time_series_social_network()
   
    def generate_ar1(self, mean, acf, mu, sigma, N):
        data = [mean]
        for i in range(1,N):
            noise = np.random.normal(mu,sigma)
            data.append(mean + acf * (data[-1] - mean) + noise)
        return np.array(data)
    
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

        preferences_uncapped = np.random.beta( self.a_preferences, self.b_preferences, size=self.num_individuals)# + self.shift_preferences
       
        low_carbon_preference_arr = np.clip(preferences_uncapped, 0 + self.clipping_epsilon_init_preference, 1- self.clipping_epsilon_init_preference)
        #low_carbon_preference_arr = preferences_uncapped

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
        self.parameters_individual["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_individual["compression_factor_state"] = self.compression_factor_state
        self.parameters_individual["carbon_price"] = self.carbon_price
        self.parameters_individual["prices_vec"] = self.prices_vec
        self.parameters_individual["emissions_intensities_vec"] = self.emissions_intensities_vec
        self.parameters_individual["clipping_epsilon"] = self.clipping_epsilon
        self.parameters_individual["fixed_preferences_state"] = self.fixed_preferences_state
        self.parameters_individual["num_firms"] = self.num_firms

        agent_list = [
            Individual(
                self.parameters_individual,
                self.low_carbon_preference_arr_init[n],
                #self.sector_preference_matrix_init,
                self.individual_expenditure_array[n],
                self.emissions_intensity_penalty_vec[n],
                self.substitutability_vec[n],
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
        low_carbon_preferences = (1 - self.upsilon)*self.low_carbon_preference_arr + self.upsilon*((1 - self.upsilon_E) * social_influence + self.upsilon_E*cumulative_emissions_influence) + np.random.normal(0, self.preference_drift_std, size=(self.num_individuals, self.num_individuals))  # Gaussian noise
        low_carbon_preferences  = np.clip(low_carbon_preferences, 0 + self.clipping_epsilon_init_preference, 1- self.clipping_epsilon_init_preference)#this stops the guassian error from causing A to be too large or small thereby producing nans
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
        total_network_emissions = sum([x.omega.emissions_intensity for x in self.agent_list])
        return total_network_emissions

    def calc_consumption_vec(self):
        # Extract the new car boolean vector and car vector from the agent list
        new_car_bool_vec = [x.new_car_bool for x in self.agent_list]
        car_vec = [x.omega for x in self.agent_list]

        # Initialize a defaultdict for counting car purchases by firm
        firm_car_count = defaultdict(lambda: defaultdict(int))

        # Iterate over the agents and update the purchase counts
        for bought_new, car in zip(new_car_bool_vec, car_vec):
            if bought_new:
                firm_car_count[car.firm_id][car.id] += 1

        return firm_car_count

    
    def calc_carbon_dividend_array(self):

        tax_income_R =  self.carbon_price*self.total_carbon_emissions_flow     
        carbon_dividend_array =  np.asarray([tax_income_R/self.num_individuals]*self.num_individuals)

        return carbon_dividend_array

    def update_individuals(self):
        """
        Update Individual objects with new information regarding social interactions, prices and dividend
        """

        # Assuming you have self.agent_list as the list of objects
        ____ = list(map(
            lambda agent, preference: agent.next_step(preference, self.car_vec, self.carbon_price),
            self.agent_list,
            self.low_carbon_preference_arr
        ))
    
    def update_burn_in_OD(self):
        if (self.t_social_network > self.burn_in_no_OD) and (not self.fixed_preferences_state):
            self.fixed_preferences_state_instant = 0 

    def update_carbon_price(self):
        if self.t_social_network == self.burn_in_no_OD+self.burn_in_duration_no_policy:
            if self.carbon_price_state == "flat":
                self.carbon_price = self.carbon_price_policy
            elif self.carbon_price_state == "AR1":
                self.carbon_price = self.carbon_price_AR1[0]
            elif self.carbon_price_state == "normal":
                self.carbon_price = self.carbon_price_normal[0]
        elif (self.t_social_network > self.burn_in_no_OD+self.burn_in_duration_no_policy): 
            if self.carbon_price_state == "AR1":
                self.carbon_price = self.carbon_price_AR1[self.t_social_network - (self.burn_in_no_OD+self.burn_in_duration_no_policy)]
            elif self.carbon_price_state == "normal":
                self.carbon_price = self.carbon_price_normal[self.t_social_network - (self.burn_in_no_OD+self.burn_in_duration_no_policy)]

    def update_prices(self):
        
        self.prices_vec_instant = self.prices_vec + self.emissions_intensities_vec*self.carbon_price
        #CHANGE THIS TO BE DONE BY THE SOCIAL NETWORK
    
    def set_up_time_series_social_network(self):
        if not self.fixed_preferences_state:
            self.history_preference_list = [self.preference_list]
        #self.history_weighting_matrix = [self.weighting_matrix]
        #self.weighting_matrix_convergence = 0  # there is no convergence in the first step, to deal with time issues when plotting
        #self.history_weighting_matrix_convergence = [self.weighting_matrix_convergence]
        self.history_flow_carbon_emissions = [self.total_carbon_emissions_flow]
        self.history_cumulative_carbon_emissions = [self.total_carbon_emissions_cumulative]#FIX
        self.history_consumed_quantities_vec = [self.consumed_quantities_vec]#consumtion associated with each firm quantity
        self.history_consumed_quantities_vec_firms = [self.consumed_quantities_vec_firms]#consumtion associated with each firm quantity
        self.history_time_social_network = [self.t_social_network]

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
        #self.history_weighting_matrix.append(self.weighting_matrix)
        #self.history_weighting_matrix_convergence.append(self.weighting_matrix_convergence)
        self.history_cumulative_carbon_emissions.append(self.total_carbon_emissions_cumulative)#THIS IS FUCKED
        self.history_flow_carbon_emissions.append(self.total_carbon_emissions_flow)
        if not self.fixed_preferences_state:
            self.history_preference_list.append(self.preference_list)
        self.history_consumed_quantities_vec.append(self.consumed_quantities_vec)
        self.history_consumed_quantities_vec_firms.append(self.consumed_quantities_vec_firms)
        self.history_time_social_network.append(self.t_social_network)

    def next_step(self, car_vec, prices_vec):
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
         
        self.update_burn_in_OD()

        #update new tech and prices
        self.car_vec = car_vec
        self.emissions_intensities_vec = [x.emissions_intensity for x in self.car_vec]
        self.prices_vec = prices_vec

        #PRICE
        self.update_carbon_price()
        self.update_prices()

        #update preferences 
        if not self.fixed_preferences_state:
            self.weighting_matrix = self.update_weightings()#UNSURE WHAT THE ORDER SHOULD BE HERE
            self.low_carbon_preference_arr = self.update_preferences()

        self.carbon_dividend_array = self.calc_carbon_dividend_array()

        # execute step
        self.update_individuals()

        #calc consumption quantities
        self.firm_count = self.calc_consumption_vec()

        #calc emissions
        self.total_carbon_emissions_flow = self.calc_total_emissions()
        self.total_carbon_emissions_cumulative = self.total_carbon_emissions_cumulative + self.total_carbon_emissions_flow

        #calc emissions dividend
        if self.redistribution_state:
            self.carbon_dividend_array = self.calc_carbon_dividend_array()

        self.segment_consumer_count, __ = np.histogram(self.low_carbon_preference_arr, bins = self.segement_preference_bounds)
            
        if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
            self.save_timeseries_data_social_network()

        return self.firm_count, self.carbon_price, self.segment_consumer_count
