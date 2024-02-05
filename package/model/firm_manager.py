"""Define firm manager that creates and manages different firms


Created: 22/12/2023
"""

# imports

from package.model.firm import Firm
import numpy as np
import random
from package.model.technology import Technology

# modules
class Firm_Manager:

    def __init__(self, parameters_firm_manager: dict, parameters_firm):
        
        self.t_firm_manager = 0

        self.landscape_seed = parameters_firm_manager["landscape_seed"]
        np.random.seed(self.landscape_seed)#set seed for numpy
        random.seed(self.landscape_seed)#set seed for random

        self.J = parameters_firm_manager["J"]
        self.N = parameters_firm_manager["N"]
        self.K = parameters_firm_manager["K"]
        self.alpha = parameters_firm_manager["alpha"]
        self.rho = parameters_firm_manager["rho"]
        self.save_timeseries_data_state = parameters_firm_manager["save_timeseries_data_state"]
        self.compression_factor_state = parameters_firm_manager["compression_factor_state"]
        self.init_tech_heterogenous_state = parameters_firm_manager["init_tech_heterogenous_state"]
        #print("self.init_tech_heterogenous_state",self.init_tech_heterogenous_state)
        self.init_carbon_premium_heterogenous_state = parameters_firm_manager["init_carbon_premium_heterogenous_state"]

        self.parameters_firm = parameters_firm

        self.value_matrix_cost, self.value_matrix_emissions_intensity = self.create_NK_model()
        self.parameters_firm["value_matrix_cost"] = self.value_matrix_cost
        self.parameters_firm["value_matrix_emissions_intensity"] = self.value_matrix_emissions_intensity
        self.parameters_firm["N"] = self.N
        self.parameters_firm["K"] = self.K
        self.parameters_firm["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_firm["compression_factor_state"] = self.compression_factor_state
        self.parameters_firm["init_market_share"] = 1/self.J
        self.parameters_firm["J"] = self.J

        
        if self.init_tech_heterogenous_state:
            init_tech_component_string_list = [f'{random.getrandbits(self.N):=0{self.N}b}' for _ in range(self.J)]#GENERATE A RANDOM STRING OF LENGTH N
            init_tech_emissions_list, inti_tech_cost_list = zip(*[self.calc_tech_emission_cost(x) for x in init_tech_component_string_list])
            self.init_tech_list = [Technology(init_tech_component_string_list[x], init_tech_emissions_list[x], inti_tech_cost_list[x], choosen_tech_bool = 1) for x in range(self.J)]
        else:
            self.init_tech_component_string = f'{random.getrandbits(self.N):=0{self.N}b}'#GENERATE A RANDOM STRING OF LENGTH N
            self.init_tech_emissions, self.inti_tech_cost = self.calc_tech_emission_cost(self.init_tech_component_string)
            self.technology_init = Technology(self.init_tech_component_string, self.init_tech_emissions, self.inti_tech_cost, choosen_tech_bool = 1)
            #self.parameters_firm["technology_init"] = self.technology_init
            self.init_tech_list = [self.technology_init]*self.J

        if self.init_carbon_premium_heterogenous_state:
            self.expected_carbon_premium_list = np.random.normal(parameters_firm_manager["expected_carbon_premium"], parameters_firm_manager["expected_carbon_premium_init_sigma"], self.J)
        else:
            self.expected_carbon_premium_list = [parameters_firm_manager["expected_carbon_premium"]]*self.J

        self.firms_list = self.create_firms()

        #set up init stuff
        self.emissions_intensities_vec, self.prices_vec, self.cost_vec, self.budget_vec, self.expected_carbon_premium_vec = self.get_firm_properties()
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
            #print("decimal", decimal)
            # Retrieve the value from the value matrix
            #print("self.value_matrix_cost",self.value_matrix_cost.shape)
            #print("decimal, n",decimal, n)
            #quit()
            #ISSUE, i think value matrix shoudl be at least (14,10) but its acctually 33,0? 0 makes sense, but not the 33
            #THE decimal conversion thing is wrong its giving values that are much larger than the size of the table
            fitness_vector_cost[n] = self.value_matrix_cost[decimal, n]
            fitness_vector_emissions_intensity[n] = self.value_matrix_emissions_intensity[decimal, n]

        emissions = np.mean(fitness_vector_emissions_intensity)
        cost = np.mean(fitness_vector_cost)
        return emissions, cost


    def create_firms(self):

        firms_list = [Firm(
                self.parameters_firm,
                self.init_tech_list[j],
                self.expected_carbon_premium_list[j],
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
        value_matrix_cost = np.random.uniform(0, 1, (2**(self.K+1), self.N)) * self.alpha#THIS IS THE COST
        value_matrix_emissions_intensity = self.convert_technology_cost_to_emissions_intensities(value_matrix_cost)
    
        return value_matrix_cost, value_matrix_emissions_intensity
        
    def convert_technology_cost_to_emissions_intensities(self, cost):

        if self.rho >= 0:
            emissions_intensity = (self.rho*cost + ((np.random.uniform(0,1, size = cost.shape))**(self.alpha))*(1-self.rho**2)**(0.5))/(self.rho + (1-self.rho**2)**(0.5))
        else:
            emissions_intensity = (self.rho*cost + ((np.random.uniform(0,1, size = cost.shape))**(self.alpha))*(1-self.rho**2)**(0.5) - self.rho)/(-self.rho + (1-self.rho**2)**(0.5))
        return emissions_intensity
    
    def calc_market_share(self, consumed_quantities_vec):
        market_share_vec = (consumed_quantities_vec*self.prices_vec)/np.matmul(consumed_quantities_vec,self.prices_vec) #price is the previous time step, so is the consumed quantity!!

        return market_share_vec
    
    def get_firm_properties(self):
        #emiussions_intes
        emissions_intensities_vec = []
        prices_vec = []
        cost_vec = []
        budget_vec = []
        expected_carbon_premium_vec = []

        for j, firm in enumerate(self.firms_list):
            emissions_intensities_vec.append(firm.firm_emissions_intensity)
            prices_vec.append(firm.firm_price)
            cost_vec.append(firm.firm_cost)
            budget_vec.append(firm.firm_budget)
            expected_carbon_premium_vec.append(firm.expected_carbon_premium)

        return np.asarray(emissions_intensities_vec), np.asarray(prices_vec), np.asarray(cost_vec), np.asarray(budget_vec), np.asarray(expected_carbon_premium_vec)

    def set_up_time_series_firm_manager(self):

        self.history_emissions_intensities_vec = [self.emissions_intensities_vec]
        self.history_weighted_emissions_intensities_vec = [self.weighted_emissions_intensities_vec]
        self.history_prices_vec = [self.prices_vec]
        self.history_market_share_vec = [self.market_share_vec]#this may be off by 1 time step??
        self.history_cost_vec = [self.cost_vec]
        self.history_budget_vec = [self.budget_vec]
        self.history_expected_carbon_premium_vec = [self.expected_carbon_premium_vec]
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
        self.history_expected_carbon_premium_vec.append(self.expected_carbon_premium_vec)
        self.history_time_firm_manager.append(self.t_firm_manager)

    def update_firms(self):
        for j,firm in enumerate(self.firms_list):
            firm.next_step(self.market_share_vec, self.consumed_quantities_vec, self.emissions_intensities_vec, self.cost_vec)

    def next_step(self, consumed_quantities_vec):

        self.t_firm_manager  +=1

        self.market_share_vec = self.calc_market_share(consumed_quantities_vec)
        self.consumed_quantities_vec = consumed_quantities_vec

        self.update_firms()

        #calc stuff for next step to pass on to consumersm, get the new prices and emissiosn internsities for consumers
        self.emissions_intensities_vec, self.prices_vec, self.cost_vec, self.budget_vec, self.expected_carbon_premium_vec = self.get_firm_properties()
        self.weighted_emissions_intensities_vec = self.emissions_intensities_vec*self.market_share_vec
        self.weighted_emissions_intensity = sum(self.weighted_emissions_intensities_vec) 

        if self.save_timeseries_data_state and (self.t_firm_manager % self.compression_factor_state == 0):
            self.save_timeseries_data_firm_manager()

        return self.emissions_intensities_vec, self.prices_vec

        