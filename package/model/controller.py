"""Define controller than manages exchange of information between social network and firms
Created: 22/12/2023
"""

# imports

from package.model.social_network import Social_Network
from package.model.firm_manager import Firm_Manager 
from package.model.centralized_ID_generator import IDGenerator

class Controller:
    def __init__(self, parameters_controller):
        
        self.parameters_controller = parameters_controller#save copy in the object for ease of access
        self.parameters_social_network = parameters_controller["parameters_social_network"]
        self.parameters_firm_manager = parameters_controller["parameters_firm_manager"]
        self.parameters_individual =  parameters_controller["parameters_individual"]
        self.parameters_carbon_policy = parameters_controller["parameters_carbon_policy"]

        self.t_controller = 0
        self.save_timeseries_data_state = parameters_controller["save_timeseries_data_state"]
        self.compression_factor_state = parameters_controller["compression_factor_state"]

        #TIME STUFF
        self.burn_in_no_OD = parameters_controller["burn_in_no_OD"] 
        self.burn_in_duration_no_policy = parameters_controller["burn_in_duration_no_policy"] 
        self.policy_duration = parameters_controller["policy_duration"]
        
        #CARBON PRICE
        self.carbon_price_state = self.parameters_carbon_policy["carbon_price_state"]
        self.carbon_price_policy = self.parameters_carbon_policy["carbon_price"]
        self.carbon_price = self.update_carbon_price()

        # CREATE ID GENERATOR FOR FIRMS
        self.IDGenerator_firms = IDGenerator()

        #TRANSFERING COMMON INFORMATION
        #FIRM MANAGER
        self.parameters_firm_manager["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_firm_manager["compression_factor_state"] = self.compression_factor_state
        self.parameters_firm_manager["num_individuals"] = self.parameters_social_network["num_individuals"]
        self.parameters_firm_manager["gamma"] = self.parameters_individual["gamma"] 
        self.parameters_firm_manager["carbon_price"] = self.carbon_price
        self.parameters_firm_manager["IDGenerator_firms"] = self.IDGenerator_firms

        #FIRM
        self.parameters_firm = parameters_controller["parameters_firm"]
        self.parameters_firm["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_firm["compression_factor_state"] = self.compression_factor_state
        self.parameters_firm["num_individuals"] = self.parameters_social_network["num_individuals"]
        self.parameters_firm["kappa"] = self.parameters_individual["kappa"]
        self.parameters_firm["N"] = self.parameters_firm_manager["N"]
        self.parameters_firm["K"] = self.parameters_firm_manager["K"]
        self.parameters_firm["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_firm["compression_factor_state"] = self.compression_factor_state
        self.parameters_firm["J"] = self.parameters_firm_manager["J"]
        self.parameters_firm["carbon_price"] = self.carbon_price
        self.parameters_firm["id_generator"] = self.IDGenerator_firms#pass the genertor on so that it can be used by the technology generation at the firm level
        self.parameters_firm["gamma"] = self.parameters_firm_manager["gamma"]
        self.parameters_firm["markup"] = self.parameters_firm_manager["markup"]

        #create social network
        self.parameters_social_network["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_social_network["compression_factor_state"] = self.compression_factor_state
        self.parameters_social_network["burn_in_no_OD"] = self.burn_in_no_OD
        self.parameters_social_network["burn_in_duration_no_policy"] = self.burn_in_duration_no_policy
        self.parameters_social_network["policy_duration"] = self.policy_duration      
        self.parameters_social_network["carbon_price"] = self.carbon_price
        self.parameters_social_network["carbon_price_state"] = self.parameters_carbon_policy["carbon_price_state"]

        #INDIVIDUALS
        self.parameters_individual["markup"] = self.parameters_firm_manager["markup"]
        self.parameters_individual["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_individual["compression_factor_state"] = self.compression_factor_state
        self.parameters_individual["carbon_price"] = self.carbon_price

        #CREATE FIRMS    
        self.firm_manager = Firm_Manager(self.parameters_firm_manager, self.parameters_firm)

        self.parameters_individual["init_car_vec"] = self.firm_manager.cars_on_sale_all_firms

        #GET FIRM PRICES
        self.social_network = Social_Network(self.parameters_social_network, self.parameters_individual)#MUST GO SECOND AS CONSUMERS NEED TO MAKE FIRST CAR CHOICE
        
        #update values for the next step
        self.controller_low_carbon_preference_arr = self.social_network.low_carbon_preference_arr

    def update_carbon_price(self):
        if self.t_controller == self.burn_in_no_OD+self.burn_in_duration_no_policy:
            if self.carbon_price_state == "flat":
                carbon_price = self.carbon_price_policy
        else:
            carbon_price = 0 

        return carbon_price

    def next_step(self):
        self.t_controller+=1
        print("self.t_controller", self.t_controller)
        self.carbon_price = self.update_carbon_price()
        # Update firms based on the social network and market conditions
        cars_on_sale_all_firms = self.firm_manager.next_step(self.carbon_price, self.controller_low_carbon_preference_arr)

        # Update social network based on firm preferences
        controller_low_carbon_preference_arr = self.social_network.next_step(self.carbon_price, cars_on_sale_all_firms)

        self.cars_on_sale_all_firms  = cars_on_sale_all_firms#
        self.controller_low_carbon_preference_arr = controller_low_carbon_preference_arr



