"""Define controller than manages exchange of information between social network and firms


Created: 22/12/2023
"""

# imports

from package.model.social_network import Social_Network
from package.model.firm_manager import Firm_Manager 

class Controller:
    def __init__(self, parameters_controller):
        
        self.t_controller = 0
        self.save_timeseries_data_state = parameters_controller["save_timeseries_data_state"]
        self.compression_factor_state = parameters_controller["compression_factor_state"]
        self.carbon_price = parameters_controller["carbon_price"]

        #create firm manager
        self.parameters_firm_manager = parameters_controller["parameters_firm_manager"]
        self.parameters_firm_manager["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_firm_manager["compression_factor_state"] = self.compression_factor_state
        self.parameters_firm_manager["carbon_price"] = self.carbon_price
        self.parameters_firm = parameters_controller["parameters_firm"]
        self.parameters_firm["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_firm["compression_factor_state"] = self.compression_factor_state

        self.firm_manager = Firm_Manager(self.parameters_firm_manager, self.parameters_firm)

        #create social network
        self.parameters_social_network = parameters_controller["parameters_social_network"]
        self.parameters_social_network["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_social_network["compression_factor_state"] = self.compression_factor_state
        self.parameters_social_network["J"] = self.parameters_firm_manager["J"]
        self.parameters_social_network["carbon_price"] = self.carbon_price
        #print("self.parameters_social_network",self.parameters_social_network)
        #quit()
        #GET FIRM PRICES
        self.parameters_social_network["prices_vec"] = self.firm_manager.prices_vec
        self.parameters_social_network["emissions_intensities_vec"] = self.firm_manager.emissions_intensities_vec
        #print("self.parameters_social_network",self.parameters_social_network)
        #quit()
        self.social_network = Social_Network(self.parameters_social_network)


        #update values for the next step
        self.emissions_intensities_vec = self.firm_manager.emissions_intensities_vec
        self.prices_vec = self.firm_manager.prices_vec
        self.consumed_quantities_vec_firms = self.social_network.consumed_quantities_vec_firms
        self.consumer_preferences_vec = self.social_network.preference_list

    def next_step(self):
        self.t_controller+=1

        #NOTE HERE THAT FIRMS REACT FIRST TO THE 

        # Update firms based on the social network and market conditions
        emissions_intensities_vec, prices_vec = self.firm_manager.next_step(self.consumed_quantities_vec_firms, self.carbon_price, self.consumer_preferences_vec)

        # Update social network based on firm preferences
        consumed_quantities_vec_firms, carbon_price, consumer_preferences_vec = self.social_network.next_step(self.emissions_intensities_vec, self.prices_vec)
        
        #update values for the next step
        self.consumer_preferences_vec = consumer_preferences_vec
        self.emissions_intensities_vec = emissions_intensities_vec
        self.prices_vec = prices_vec
        self.consumed_quantities_vec_firms = consumed_quantities_vec_firms
        self.carbon_price = carbon_price


