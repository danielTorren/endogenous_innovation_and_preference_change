"""Define firm manager that creates and manages different firms


Created: 22/12/2023
"""

# imports

from package.model.firm import Firm
import numpy as np

# modules
class Firm_Manager:

    def __init__(self, parameters_firm_manager: dict, parameters_firm):
        #do stufff

        self.N = parameters_firm_manager["N"]
        self.K = parameters_firm_manager["K"]
        self.alpha = parameters_firm_manager["alpha"]
        self.rho = parameters_firm_manager["rho"]
        self.save_timeseries_data_state = parameters_firm_manager["save_timeseries_data_state"]
        self.compression_factor_state = parameters_firm_manager["compression_factor_state"]
        self.burn_in_duration_firm_manager = parameters_firm_manager["burn_in_duration_firm_manager"]
        self.parameters_firm = parameters_firm

        self.value_matrix_cost, self.value_matrix_emissions_intensity = self.create_NK_model(self.N,self.K, self.alpha, self.rho)
        self.parameters_firm["value_matrix_cost"] = self.value_matrix_cost
        self.parameters_firm["value_matrix_emissions_intensity"] = self.value_matrix_emissions_intensity
        self.parameters_firm["N"] = self.N
        self.parameters_firm["K"] = self.K
        self.parameters_firm["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_firm["compression_factor_state"] = self.compression_factor_state
        self.parameters_firm["burn_in_duration_firm_manager"] = self.burn_in_duration_firm_manager

        # time
        self.t_firm_manager = 0

        self.firms_list = self.create_firms()

    def create_firms(self):

        firms_list = [Firm(
                self.parameters_firm,
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
        value_matrix_cost = np.random.uniform(0, 1, (2*(self.K+1), self.N)) * self.alpha#THIS IS THE COST
        value_matrix_emissions_intensity = self.convert_technology_cost_to_emissions_intensities(value_matrix_cost)
    
        return value_matrix_cost, value_matrix_emissions_intensity
        
    def convert_technology_cost_to_emissions_intensities(self, cost):
        if self.rho >= 0:
            emissions_intensity = (self.rho*cost + ((np.random.uniform(0,1))**(self.alpha))*(1-self.rho**2)**(0.5))/(self.rho + (1-self.rho**2)**(0.5))
        else:
            emissions_intensity = (self.rho*cost + ((np.random.uniform(0,1))**(self.alpha))*(1-self.rho**2)**(0.5) - self.rho)/(-self.rho + (1-self.rho**2)**(0.5))
            
        return emissions_intensity
    
    def calc_market_share(self, consumed_quantities_vec):
        market_share_vec = (consumed_quantities_vec*self.prices_vec)/np.matmul(consumed_quantities_vec*self.prices_vec) #price is the previous time step, so is the consumed quantity!!
        return market_share_vec
    
    def get_firm_prices_and_intensities(self):
        #emiussions_intes
        emissions_intensities_vec = []
        prices_vec = []

        for j,firm in enumerate(self.firms_list):
            emissions_intensities_vec.append(firm.firm_emissions_intensities)
            prices_vec.append(firm.firm_price)
        
        return np.asarray(emissions_intensities_vec), np.asarray(prices_vec)

    def set_up_time_series_firm_manager(self):
        self.history_time_firm_manager = [self.t_firm_manager]
        self.history_emissions_intensities_vec = [self.emissions_intensities_vec]
        self.history_prices_vec = [self.prices_vec]
        self.history_market_share_vec = [self.market_share_vec]#this may be off by 1 time step??

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
        self.history_time_firm_manager.append(self.t_firm_manager)
        self.history_emissions_intensities_vec.append(self.emissions_intensities_vec)
        self.history_prices_vec.append(self.prices_vec)
        self.history_market_share_vec.append(self.market_share_vec)#this may be off by 1 time step??

    def update_firms(self):
        for j,firm in enumerate(self.firms_list):
            firm.next_step(self, self.t_firm_manager, self.market_share_vec, self.consumed_quantities_vec, self.emissions_intensities_vec, self.price_vec)

    def next_step(self, consumed_quantities_vec):

        # advance a time step
        self.t_firm_manager += 1

        self.market_share_vec = self.calc_market_share(consumed_quantities_vec)
        self.consumed_quantities_vec = consumed_quantities_vec

        self.update_firms()

        #calc stuff for next step to pass on to consumersm, get the new prices and emissiosn internsities for consumers
        self.emissions_intensities_vec, self.prices_vec = self.get_firm_prices_and_intensities()

        if self.save_timeseries_data_state:
            if self.t_firm_manager == self.burn_in_duration_firm_manager + 1:
                self.set_up_time_series_firm_manager()
            elif (self.t_firm_manager % self.compression_factor_state == 0) and (self.t_firm_manager > self.burn_in_duration_firm_manager):
                self.save_timeseries_data_firm_manager()

        return self.emissions_intensities_vec, self.prices_vec

        