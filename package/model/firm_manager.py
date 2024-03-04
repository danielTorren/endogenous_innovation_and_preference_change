"""Define firm manager that creates and manages different firms


Created: 22/12/2023
"""

# imports

from tkinter import N
from package.model.firm import Firm
import numpy as np
import random
from package.model.technology import Technology

# modules
class Firm_Manager:

    def __init__(self, parameters_firm_manager: dict, parameters_firm):
        
        self.t_firm_manager = 0

        self.landscape_seed = parameters_firm_manager["landscape_seed"]
        self.init_tech_seed = parameters_firm_manager["init_tech_seed"] 

        self.J = parameters_firm_manager["J"]
        self.N = parameters_firm_manager["N"]
        self.K = parameters_firm_manager["K"]
        self.alpha = parameters_firm_manager["alpha"]
        self.rho = parameters_firm_manager["rho"]
        self.save_timeseries_data_state = parameters_firm_manager["save_timeseries_data_state"]
        self.compression_factor_state = parameters_firm_manager["compression_factor_state"]
        self.init_tech_heterogenous_state = parameters_firm_manager["init_tech_heterogenous_state"]
        #print("self.init_tech_heterogenous_state",self.init_tech_heterogenous_state)
        self.carbon_price = parameters_firm_manager["carbon_price"]
        self.nk_multiplier = parameters_firm_manager["nk_multiplier"]
        self.c_min = parameters_firm_manager["c_min"]
        self.c_max = parameters_firm_manager["c_max"]
        self.ei_min = parameters_firm_manager["ei_min"]
        self.ei_max = parameters_firm_manager["ei_max"]
        self.num_individuals = parameters_firm_manager["num_individuals"]
        self.survey_cost = self.num_individuals/(2*self.J)
        self.research_cost = self.num_individuals/(2*self.J)

        self.parameters_firm = parameters_firm

        #GEN INIT TECH
        np.random.seed(self.init_tech_seed)#set seed for numpy
        random.seed(self.init_tech_seed)#set seed for random

        self.init_tech_component_string = f"{random.getrandbits(self.N):=0{self.N}b}"
        if self.init_tech_heterogenous_state:
            decimal_value = int(self.init_tech_component_string, 2) 
            init_tech_component_string_list_N = self.invert_bits_one_at_a_time(decimal_value, len(self.init_tech_component_string))
            init_tech_component_string_list = np.random.choice(init_tech_component_string_list_N, self.J)
        
        #################################################################################################################################################
        #BELOW STUFF IS VARIED IN MONTECARLO SIMULATIONS
        #################################################################################################################################################

        np.random.seed(self.landscape_seed)#set seed for numpy
        random.seed(self.landscape_seed)#set seed for random

        self.value_matrix_cost, self.value_matrix_emissions_intensity = self.create_NK_model()

        if self.init_tech_heterogenous_state:
            init_tech_emissions_list, inti_tech_cost_list = zip(*[self.calc_tech_emission_cost(x) for x in init_tech_component_string_list])
            self.init_tech_list = [Technology(init_tech_component_string_list[x], init_tech_emissions_list[x], inti_tech_cost_list[x], choosen_tech_bool = 1) for x in range(self.J)]
        else:
            self.init_tech_emissions, self.inti_tech_cost = self.calc_tech_emission_cost(self.init_tech_component_string)
            self.technology_init = Technology(self.init_tech_component_string, self.init_tech_emissions, self.inti_tech_cost, choosen_tech_bool = 1)
            self.init_tech_list = [self.technology_init]*self.J

        self.parameters_firm["value_matrix_cost"] = self.value_matrix_cost
        self.parameters_firm["value_matrix_emissions_intensity"] = self.value_matrix_emissions_intensity
        self.parameters_firm["N"] = self.N
        self.parameters_firm["K"] = self.K
        self.parameters_firm["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_firm["compression_factor_state"] = self.compression_factor_state
        self.parameters_firm["init_market_share"] = 1/self.J
        self.parameters_firm["J"] = self.J
        self.parameters_firm["carbon_price"] = self.carbon_price
        self.parameters_firm["c_min"] = self.c_min 
        self.parameters_firm["c_max"] = self.c_max 
        self.parameters_firm["ei_min"] = self.ei_min 
        self.parameters_firm["ei_max"] = self.ei_max 
        self.parameters_firm["survey_cost"] = self.survey_cost
        self.parameters_firm["research_cost"] = self.research_cost

        self.firms_list = self.create_firms()

        #set up init stuff
        self.emissions_intensities_vec, self.prices_vec, self.cost_vec, self.budget_vec = self.get_firm_properties()
        self.market_share_vec = [firm.current_market_share for firm in self.firms_list]
        self.weighted_emissions_intensities_vec = self.emissions_intensities_vec*self.market_share_vec
        self.weighted_emissions_intensity = sum(self.weighted_emissions_intensities_vec) 
        
        if self.save_timeseries_data_state:
            self.set_up_time_series_firm_manager()

    def calc_tech_emission_cost(self, random_technology_string):
        """JUST FOR CALCULATING INITIAL CONDITIONS"""

        fitness_vector_cost = np.zeros((self.N))
        fitness_vector_emissions_intensity = np.zeros((self.N))

        for n in range(self.N):#Look through substrings
            # Create the binary substring
            substring = random_technology_string[n:n+self.K]
            #print("substring",substring)
            
            # If the substring is shorter than K, wrap around (toroidal)
            if len(substring) < self.K:
                substring += random_technology_string[:self.K-len(substring)]
                #print("wrap around",substring)
            # Convert the binary substring to decimal
            decimal = int(substring, 2)
            # Retrieve the value from the value matrix
            #ISSUE, i think value matrix shoudl be at least (14,10) but its acctually 33,0? 0 makes sense, but not the 33
            #THE decimal conversion thing is wrong its giving values that are much larger than the size of the table
            fitness_vector_cost[n] = self.value_matrix_cost[decimal, n]
            fitness_vector_emissions_intensity[n] = self.value_matrix_emissions_intensity[decimal, n]


        cost = self.c_min +((self.c_max-self.c_min)/self.N)*np.sum(fitness_vector_cost, axis = 0)
        emissions = self.ei_min +((self.ei_max-self.ei_min)/self.N)*np.sum(fitness_vector_emissions_intensity, axis = 0)

        return emissions, cost

    def invert_bits_one_at_a_time(self,decimal_value, length):
        # Convert decimal value to binary with leading zeros to achieve length N
        # binary_value = format(decimal_value, f'0{length}b')

        # Initialize an empty list to store inverted binary values
        inverted_binary_values = []

        # Iterate through each bit position
        for bit_position in range(length):
            """
            NEED TO UNDERSTAND BETTER HOW THIS WORKS!!
            """
            inverted_value = decimal_value^(1 << bit_position)

            # Convert the inverted decimal value to binary
            inverted_binary_value = format(inverted_value, f'0{length}b')

            # Append the inverted binary value to the list
            inverted_binary_values.append(inverted_binary_value)

        return inverted_binary_values

    def create_firms(self):

        firms_list = [Firm(
                self.parameters_firm,
                self.init_tech_list[j],
                j
            ) 
            for j in range(self.J) 
        ]

        return firms_list

    def create_NK_model(self):
        """
            We make a landscape for cost of technologies
        """
        # Step 1: Create the value matrix
        #value_matrix_cost = np.random.uniform(0, 1*self.nk_multiplier, (2**(self.K+1), self.N)) * self.alpha#THIS IS THE COST

        value_matrix_cost = np.random.uniform(0, 1*self.nk_multiplier, (2**(self.K+1), self.N)) * self.alpha#THIS IS THE COST
        #print(value_matrix_cost.shape)
        #print(self.N, self.K, 2**(self.K+1))
        #print(2**self.N-1)
        #quit()

        #normalized_value_matrix_cost = value_matrix_cost + 1
        #print("normalized_value_matrix_cost", np.min(normalized_value_matrix_cost), np.max(normalized_value_matrix_cost))
        #value_matrix_emissions_intensity = self.convert_technology_cost_to_emissions_intensities(normalized_value_matrix_cost)
        value_matrix_emissions_intensity = self.convert_technology_cost_to_emissions_intensities(value_matrix_cost)

        #STEP 2 Normalize both
        #normalized_cost_matrix = self.c_min +((self.c_max-self.c_min)/self.N)*np.sum(value_matrix_cost, axis = 1)
        #normalized_emissions_intensity_matrix = self.ei_min +((self.ei_max-self.ei_min)/self.N)*np.sum(value_matrix_emissions_intensity, axis = 1)
        #print(normalized_cost_matrix.shape)
        #quit()
        #return normalized_cost_matrix, normalized_emissions_intensity_matrix
        return value_matrix_cost, value_matrix_emissions_intensity
        
    def convert_technology_cost_to_emissions_intensities(self, cost):

        if self.rho >= 0:
            emissions_intensity = (self.rho*cost + ((np.random.uniform(0,1*self.nk_multiplier, size = cost.shape))**(self.alpha))*(1-self.rho**2)**(0.5))/(self.rho + (1-self.rho**2)**(0.5))
        else:
            emissions_intensity = (self.rho*cost + ((np.random.uniform(0,1*self.nk_multiplier, size = cost.shape))**(self.alpha))*(1-self.rho**2)**(0.5) - self.rho)/(-self.rho + (1-self.rho**2)**(0.5))
        
        #normalized_EI = emissions_intensity + 1
        #return normalized_EI
        return emissions_intensity
    
    def calc_market_share(self, consumed_quantities_vec):
        """EXPENDITURE MARKET SHARE"""
        market_share_vec = (consumed_quantities_vec*self.prices_vec)/np.matmul(consumed_quantities_vec,self.prices_vec) #price is the previous time step, so is the consumed quantity!!

        return market_share_vec
    
    def get_firm_properties(self):
        #emiussions_intes
        emissions_intensities_vec = []
        prices_vec = []
        cost_vec = []
        budget_vec = []

        for j, firm in enumerate(self.firms_list):
            emissions_intensities_vec.append(firm.firm_emissions_intensity)
            prices_vec.append(firm.firm_price)
            cost_vec.append(firm.firm_cost)
            budget_vec.append(firm.firm_budget)

        return np.asarray(emissions_intensities_vec), np.asarray(prices_vec), np.asarray(cost_vec), np.asarray(budget_vec)

    def set_up_time_series_firm_manager(self):

        self.history_emissions_intensities_vec = [self.emissions_intensities_vec]
        self.history_weighted_emissions_intensities_vec = [self.weighted_emissions_intensities_vec]
        self.history_prices_vec = [self.prices_vec]
        self.history_market_share_vec = [self.market_share_vec]#this may be off by 1 time step??
        self.history_cost_vec = [self.cost_vec]
        self.history_budget_vec = [self.budget_vec]

        self.history_time_firm_manager = [self.t_firm_manager]

    def save_timeseries_data_firm_manager(self):
        """
        Save time series data

        parameters_social_network
        ----------
        None

        Returns
        -------
        None
        """

        self.history_emissions_intensities_vec.append(self.emissions_intensities_vec)
        self.history_weighted_emissions_intensities_vec.append(self.weighted_emissions_intensities_vec)
        self.history_prices_vec.append(self.prices_vec)
        self.history_market_share_vec.append(self.market_share_vec)#this may be off by 1 time step??
        self.history_cost_vec.append(self.cost_vec)
        self.history_budget_vec.append(self.budget_vec)
        self.history_time_firm_manager.append(self.t_firm_manager)

    def update_firms(self):
        for j,firm in enumerate(self.firms_list):
            #                    market_share_vec,     consumed_quantities_vec,       emissions_intensities_vec,     cost_vec, carbon_price, consumer_preferences_vec
            firm.next_step(self.market_share_vec, self.consumed_quantities_vec, self.emissions_intensities_vec, self.cost_vec, self.carbon_price, self.consumer_preferences_vec)

    def next_step(self, consumed_quantities_vec, carbon_price, consumer_preferences_vec):

        self.t_firm_manager  +=1
        self.carbon_price = carbon_price
        self.market_share_vec = self.calc_market_share(consumed_quantities_vec)
        self.consumed_quantities_vec = consumed_quantities_vec
        self.consumer_preferences_vec = consumer_preferences_vec

        self.update_firms()

        #calc stuff for next step to pass on to consumersm, get the new prices and emissiosn internsities for consumers
        self.emissions_intensities_vec, self.prices_vec, self.cost_vec, self.budget_vec = self.get_firm_properties()
        self.weighted_emissions_intensities_vec = self.emissions_intensities_vec*self.market_share_vec
        self.weighted_emissions_intensity = sum(self.weighted_emissions_intensities_vec) 

        if self.save_timeseries_data_state and (self.t_firm_manager % self.compression_factor_state == 0):
            self.save_timeseries_data_firm_manager()

        return self.emissions_intensities_vec, self.prices_vec

        