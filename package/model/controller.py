"""Define controller than manages exchange of information between social network and firms
Created: 22/12/2023
"""

# imports

from package.model.social_network import Social_Network
#from package.model.gpt_firm_manager import Firm_Manager 
from package.model.firm_manager import Firm_Manager 
from package.model.centralized_ID_generator import IDGenerator

class Controller:
    def __init__(self, parameters_controller):
        
        self.parameters_controller = parameters_controller#save copy in the object for ease of access
        self.parameters_social_network = parameters_controller["parameters_social_network"]
        self.parameters_firm_manager = parameters_controller["parameters_firm_manager"]
        self.parameters_carbon_policy = parameters_controller["parameters_carbon_policy"]

        self.t_controller = 0
        self.save_timeseries_data_state = parameters_controller["save_timeseries_data_state"]
        self.compression_factor_state = parameters_controller["compression_factor_state"]
        self.utility_boost_const = parameters_controller["utility_boost_const"]

        #TIME STUFF
        self.duration_no_OD_no_stock_no_policy = parameters_controller["duration_no_OD_no_stock_no_policy"] 
        self.duration_OD_no_stock_no_policy = parameters_controller["duration_OD_no_stock_no_policy"] 
        self.duration_OD_stock_no_policy = parameters_controller["duration_OD_stock_no_policy"] 
        self.duration_OD_stock_policy = parameters_controller["duration_OD_stock_policy"]

        self.policy_start_time = self.duration_no_OD_no_stock_no_policy + self.duration_OD_no_stock_no_policy + self.duration_OD_stock_no_policy
        
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
        self.parameters_firm_manager["gamma"] = self.parameters_social_network["gamma"] 
        self.parameters_firm_manager["carbon_price"] = self.carbon_price
        self.parameters_firm_manager["IDGenerator_firms"] = self.IDGenerator_firms
        self.parameters_firm_manager["kappa"] = self.parameters_social_network["kappa"]
        self.parameters_firm_manager["utility_boost_const"] = self.utility_boost_const

        #create social network
        self.parameters_social_network["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_social_network["compression_factor_state"] = self.compression_factor_state
        self.parameters_social_network["duration_no_OD_no_stock_no_policy"] = self.duration_no_OD_no_stock_no_policy
        self.parameters_social_network["duration_OD_no_stock_no_policy"] = self.duration_OD_no_stock_no_policy
        self.parameters_social_network["duration_OD_stock_no_policy"] = self.duration_OD_stock_no_policy
        self.parameters_social_network["duration_OD_stock_policy"] = self.duration_OD_stock_policy
        self.parameters_social_network["policy_start_time"] = self.policy_start_time      
        self.parameters_social_network["carbon_price"] = self.carbon_price
        self.parameters_social_network["carbon_price_state"] = self.parameters_carbon_policy["carbon_price_state"]
        self.parameters_social_network["markup"] = self.parameters_firm_manager["markup"]
        self.parameters_social_network["utility_boost_const"] = self.utility_boost_const
        #CREATE FIRMS    
        #self.firm_manager = Firm_Manager(self.parameters_firm_manager, self.parameters_firm)
        self.firm_manager = Firm_Manager(self.parameters_firm_manager)

        self.parameters_social_network["init_car_vec"] = self.firm_manager.cars_on_sale_all_firms

        #GET FIRM PRICES
        self.social_network = Social_Network(self.parameters_social_network)#MUST GO SECOND AS CONSUMERS NEED TO MAKE FIRST CAR CHOICE
        
        #update values for the next step
        self.controller_low_carbon_preference_arr = self.social_network.low_carbon_preference_arr

    def update_carbon_price(self):
        if self.t_controller == self.policy_start_time:
            if self.carbon_price_state == "flat":
                carbon_price = self.carbon_price_policy
        else:
            carbon_price = 0 

        return carbon_price

    def next_step(self):
        self.t_controller+=1
        #print("self.t_controller", self.t_controller)
        self.carbon_price = self.update_carbon_price()
        # Update firms based on the social network and market conditions
        cars_on_sale_all_firms = self.firm_manager.next_step(self.carbon_price, self.controller_low_carbon_preference_arr)

        # Update social network based on firm preferences
        controller_low_carbon_preference_arr = self.social_network.next_step(self.carbon_price, cars_on_sale_all_firms)

        self.cars_on_sale_all_firms  = cars_on_sale_all_firms#
        self.controller_low_carbon_preference_arr = controller_low_carbon_preference_arr



