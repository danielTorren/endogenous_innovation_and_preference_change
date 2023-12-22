"""Create social network with individuals
A module that use input data to generate a social network containing individuals who each have multiple 
behaviours. The weighting of individuals within the social network is determined by the identity distance 
between neighbours. The simulation evolves over time saving data at set intervals to reduce data output.


Created: 10/10/2022
"""


# imports
import numpy as np
import networkx as nx
import numpy.typing as npt
from package.model.individual import Individual
from operator import attrgetter


# modules
class Social_Network:

    def __init__(self, parameters_social_network: list):#FEED THE STUFF STRAIGHT THAT ISNT USED BY SOCIAL NETWORK
        """
        Constructs all the necessary attributes for the Network object.

        parameters_social_network
        ----------
        parameters_social_network : dict
            Dictionary of parameters_social_network used to generate attributes, dict used for readability instead of super long list of input parameters_social_network

        """

        #INITAL STATE OF THE SYSTEMS, WHAT ARE THE RUN CONDITIONS
        self.imperfect_learning_state = parameters_social_network["imperfect_learning_state"]
        self.nu_change_state = parameters_social_network["nu_change_state"]
        self.save_timeseries_data_state = parameters_social_network["save_timeseries_data_state"]
        self.compression_factor_state = parameters_social_network["compression_factor_state"]

        #seeds
        self.init_vals_seed = parameters_social_network["init_vals_seed"] 
        self.imperfect_learning_seed = int(round(parameters_social_network["imperfect_learning_seed"]))

        np.random.seed(self.init_vals_seed)#For inital construction set a seed, this is the same for all runs, then later change it to imperfect_learning_seed
        
        # network
        self.network_density_input = parameters_social_network["network_density"]
        self.num_individuals = int(round(parameters_social_network["N"]))
        self.K = int(round((self.num_individuals - 1)*self.network_density_input)) #reverse engineer the links per person using the density  d = 2m/n(n-1) where n is nodes and m number of edges
        #print("self.K",self.K)
        self.prob_rewire = parameters_social_network["prob_rewire"]

        #price
        self.prices =  parameters_social_network["prices"]
        self.carbon_price = parameters_social_network["carbon_price"]

        # social learning and bias
        self.confirmation_bias = parameters_social_network["confirmation_bias"]
        if self.imperfect_learning_state:
            self.std_learning_error = parameters_social_network["std_learning_error"]
            self.clipping_epsilon = parameters_social_network["clipping_epsilon"]
        else:
            self.std_learning_error = 0
            self.clipping_epsilon = 0     

        self.clipping_epsilon_init_preference = parameters_social_network["clipping_epsilon_init_preference"]

        self.individual_phi = parameters_social_network["phi"]

        # network homophily
        self.homophily = parameters_social_network["homophily"]  # 0-1
        self.shuffle_reps = int(
            round(self.num_individuals*(1 - self.homophily))
        )


        self.quantity_state = parameters_social_network["quantity_state"]
        self.chi_ms = parameters_social_network["chi_ms"]
        
        # create network
        (
            self.adjacency_matrix,
            self.weighting_matrix,
            self.network,
        ) = self.create_weighting_matrix()

        self.network_density = nx.density(self.network)

        self.a_preferences = parameters_social_network["a_preferences"]#A #IN THIS BRANCH CONSISTEN BEHAVIOURS USE THIS FOR THE IDENTITY DISTRIBUTION
        self.b_preferences = parameters_social_network["b_preferences"]#A #IN THIS BRANCH CONSISTEN BEHAVIOURS USE THIS FOR THE IDENTITY DISTRIBUTION
        self.std_low_carbon_preference = parameters_social_network["std_low_carbon_preference"]
        (
            self.low_carbon_preference_matrix_init
        ) = self.generate_init_data_preferences()
        
        self.individual_expenditure_array =  np.asarray([parameters_social_network["expenditure"]]*self.num_individuals)#sums to 1
        
        self.substitutability = parameters_social_network["substitutability"]
        
        self.agent_list = self.create_agent_list()

        self.shuffle_agent_list()#partial shuffle of the list based on identity

        #NOW SET SEED FOR THE IMPERFECT LEARNING
        np.random.seed(self.imperfect_learning_seed)

        if self.nu_change_state == "fixed_preferences":
            self.social_component_matrix = np.asarray([n.low_carbon_preference for n in self.agent_list])#DUMBY FEED IT ITSELF? DO I EVEN NEED TO DEFINE IT
        else:
            if self.nu_change_state in ("uniform_network_weighting","static_culturally_determined_weights","dynamic_culturally_determined_weights"):
                self.weighting_matrix = self.update_weightings()
            self.social_component_matrix = self.calc_social_component_matrix()

        self.t_social_networkotal_carbon_emissions_stock = 0#this are for post tax

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
            an NxN array how how much each agent values the opinion of their neighbour. Note that is it not symetric and agent i doesn't need to value the
            opinion of agent j as much as j does i's opinion
        ws: nx.Graph
            a networkx watts strogatz small world graph
        """

        G = nx.watts_strogatz_graph(n=self.num_individuals, k=self.K, p=self.prob_rewire, seed=self.network_structure_seed)#FIX THE NETWORK STRUCTURE

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

        preferences_uncapped = np.random.beta( self.a_preferences, self.b_preferences, size=self.num_individuals)

        low_carbon_preference_matrix = np.clip(preferences_uncapped, 0 + self.clipping_epsilon_init_preference, 1- self.clipping_epsilon_init_preference)

        return low_carbon_preference_matrix

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

        individual_params = {
            "save_timeseries_data_state": self.save_timeseries_data_state,
            "individual_phi": self.individual_phi,
            "compression_factor_state": self.compression_factor_state,
            "carbon_price": self.carbon_price,
            "prices": self.prices,
            "clipping_epsilon" :self.clipping_epsilon,
            "nu_change_state": self.nu_change_state,
            "substitutability": self.substitutability,
            "quantity_state": self.quantity_state,
            "chi_ms": self.chi_ms,
        }

        agent_list = [
            Individual(
                individual_params,
                self.low_carbon_preference_matrix_init[n],
                #self.sector_preference_matrix_init,
                self.individual_expenditure_array[n],
                n
            )
            for n in range(self.num_individuals)
        ]

        return agent_list
        
    def shuffle_agent_list(self): 
        #make list cirucalr then partial shuffle it
        self.agent_list.sort(key=lambda x: x.low_carbon_preference)#sorted by identity
        self.circular_agent_list()#agent list is now circular in terms of identity
        self.partial_shuffle_agent_list()#partial shuffle of the list

    
    def calc_ego_influence_degroot(self) -> npt.NDArray:

        attribute_matrix =np.asarray(list(map(attrgetter('outward_social_influence'), self.agent_list))) 

        neighbour_influence = np.matmul(self.weighting_matrix, attribute_matrix)    
        
        return neighbour_influence

    def calc_social_component_matrix(self) -> npt.NDArray:
        """
        Combine neighbour influence and social learning error to updated individual behavioural attitudes

        parameters_social_network
        ----------
        None

        Returns
        -------
        social_influence: npt.NDArray
            NxM array giving the influence of social learning from neighbours for that time step
        """

        ego_influence = self.calc_ego_influence_degroot()           
         
        social_influence = ego_influence + np.random.normal(loc=0, scale=self.std_learning_error, size=(self.num_individuals))

        return social_influence

    def calc_weighting_matrix_attribute(self,attribute_array):

        difference_matrix = np.subtract.outer(attribute_array, attribute_array) #euclidean_distances(attribute_array,attribute_array)# i think this actually not doing anything? just squared at the moment

        alpha_numerator = np.exp(-np.multiply(self.confirmation_bias, np.abs(difference_matrix)))

        non_diagonal_weighting_matrix = (
            self.adjacency_matrix*alpha_numerator
        )  # We want onlythose values that have network connections

        norm_weighting_matrix = self.normlize_matrix(
            non_diagonal_weighting_matrix
        )  # normalize the matrix row wise
    
        return norm_weighting_matrix

    def update_weightings(self) -> tuple[npt.NDArray, float]:
        """
        Update the link strength array according to the new agent identities

        parameters_social_network
        ----------
        None

        Returns
        -------
        norm_weighting_matrix: npt.NDArray
            Row normalized weighting array giving the strength of inter-Individual connections due to similarity in identity
        total_identity_differences
        total_difference: float
            total element wise difference between the previous weighting arrays
        """

        ##THE WEIGHTING FOR THE IDENTITY IS DONE IN INDIVIDUALS

        self.preference_list = list(map(attrgetter('low_carbon_preference'), self.agent_list))

        norm_weighting_matrix = self.calc_weighting_matrix_attribute(self.preference_list)

        return norm_weighting_matrix

    def calc_total_emissions(self) -> int:
        """
        Calculate total carbon emissions of N*M behaviours

        parameters_social_network
        ----------
        None

        Returns
        -------
        total_network_emissions: float
            total network emissions from each Individual object
        """

        total_network_emissions = sum(map(attrgetter('flow_carbon_emissions'), self.agent_list))
        return total_network_emissions
    
    def calc_consumption_vec(self):
        consumption_matrix = np.asarray([map(attrgetter('quantities'), self.agent_list)])
        print("consumption_matrix shape", consumption_matrix.shape)
        consumption_vec = np.sum(consumption_matrix, axis=1)
        print("cosnumption_vec shape",consumption_vec.shape)

        return consumption_vec

    def update_individuals(self):
        """
        Update Individual objects with new information regarding social interactions, prices and dividend
        """

        # Assuming you have self.agent_list as the list of objects
        ____ = list(map(
            lambda agent, scm: agent.next_step(scm, self.carbon_price, self.emissions_intensities),
            self.agent_list,
            self.social_component_matrix
        ))
    
    def set_up_time_series_social_network(self):
        self.history_preference_list = [self.preference_list]
        self.history_weighting_matrix = [self.weighting_matrix]
        self.weighting_matrix_convergence = 0  # there is no convergence in the first step, to deal with time issues when plotting
        self.history_weighting_matrix_convergence = [self.weighting_matrix_convergence]
        self.history_flow_carbon_emissions = [self.t_social_networkotal_carbon_emissions_flow]
        self.history_stock_carbon_emissions = [self.t_social_networkotal_carbon_emissions_stock]

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
        self.history_weighting_matrix.append(self.weighting_matrix)
        self.history_weighting_matrix_convergence.append(
            self.weighting_matrix_convergence
        )
        self.history_stock_carbon_emissions.append(self.t_social_networkotal_carbon_emissions_stock)
        self.history_flow_carbon_emissions.append(self.t_social_networkotal_carbon_emissions_flow)
        self.history_preference_list.append(self.preference_list)

    def next_step(self, emissions_intensities_vec, prices_vec):
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

        #update new tech and prices
        self.emissions_intensities = emissions_intensities_vec
        self.prices = prices_vec

        # execute step
        self.update_individuals()

        #calc consumption quantities
        self.consumption_vec = self.calc_consumption_vec()

        # update network parameters_social_network for next step
        if self.nu_change_state != "fixed_preferences":
            if self.nu_change_state == "dynamic_culturally_determined_weights":
                self.weighting_matrix = self.update_weightings()
            else:
                pass #this is for "uniform_network_weighting", "static_socially_determined_weights","static_culturally_determined_weights"

            self.social_component_matrix = self.calc_social_component_matrix()


        self.t_social_networkotal_carbon_emissions_flow = self.calc_total_emissions()
        self.t_social_networkotal_carbon_emissions_stock = self.t_social_networkotal_carbon_emissions_stock + self.t_social_networkotal_carbon_emissions_flow
            
        if self.save_timeseries_data_state:
            self.save_timeseries_data_social_network()

        return self.consumption_vec
