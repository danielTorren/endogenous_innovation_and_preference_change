"""Define firm manager that creates and manages different firms


Created: 10/10/2022
"""

# imports

from re import M
from package.model.firm import Firm
import numpy as np
from package.model.technology import Technology

# modules
class Firm_Manager:

    def __init__(self, parameters_firm_manager: dict):
        #do stufff

        self.N = parameters_firm_manager["N"]
        self.K = parameters_firm_manager["K"]
        self.alpha = parameters_firm_manager["alpha"]
        self.rho = parameters_firm_manager["rho"]
        self.parameters_firm = parameters_firm_manager["parameters_firm"]    

        self.value_matrix_cost, self.value_matrix_emissions_intensity = self.create_NK_model(self.N,self.K, self.alpha, self.rho)
        self.parameters_firm["value_matrix_cost"] =self.value_matrix_cost
        self.parameters_firm["value_matrix_emissions_intensity"] =self.value_matrix_emissions_intensity

        self.firms_list = self.create_firms()

    def create_firms(self):

        firms_list = [Firm(
                self.parameters_firm,
                j
            ) 
            for j in range(self.J) 
        ]

        return firms_list

    def create_NK_model(self,N,K, alpha=1, rho=1):
        """
            We make a landscape for cost of technologies
        """
        # Step 1: Create the value matrix
        value_matrix_cost = np.random.uniform(0, 1, (2*(K+1), N)) * alpha#THIS IS THE COST
        value_matrix_emissions_intensity = self.convert_technology_cost_to_emissions_intensities(value_matrix_cost)
    
        return value_matrix_cost, value_matrix_emissions_intensity
        
    def convert_technology_cost_to_emissions_intensities(self, cost):
        if self.rho >= 0:
            emissions_intensity = (self.rho*cost + ((np.random.uniform(0,1))**(self.alpha))*(1-self.rho**2)**(0.5))/(self.rho + (1-self.rho**2)**(0.5))
        else:
            emissions_intensity = (self.rho*cost + ((np.random.uniform(0,1))**(self.alpha))*(1-self.rho**2)**(0.5) - self.rho)/(-self.rho + (1-self.rho**2)**(0.5))
            
        return emissions_intensity
    
    def calc_market_share(self, consumed_quantities_vec):
        market_share_vec = (consumed_quantities_vec*self.price_vec)/np.matmul(consumed_quantities_vec*self.price_vec) #price is the previous time step, so is the consumed quantity!!
        return market_share_vec
    
    def get_firm_price_and_intensities(self):
        #emiussions_intes
        emissions_intensities_vec = []
        price_vec = []

        for j,firm in enumerate(self.firms_list):
            emissions_intensities_vec.append(firm.firm_emissions_intensities)
            price_vec.append(firm.firm_price)
        
        return np.asarray(emissions_intensities_vec), np.asarray(price_vec)


    def update_firms(self):
        for j,firm in enumerate(self.firms_list):
            firm.next_step(self, self.market_share_vec, self.consumed_quantities_vec, self.emissions_intensities_vec, self.price_vec)

    def next_step(self, consumed_quantities_vec):
        #calcl stuff from preious step
        
        self.market_share_vec = self.calc_market_share(consumed_quantities_vec)
        self.consumed_quantities_vec = consumed_quantities_vec

        self.update_firms()

        #calc stuff for next step to pass on to consumersm, get the new prices and emissiosn internsities for consumers
        self.emissions_intensities_vec, self.price_vec = self.get_firm_price_and_intensities()

        