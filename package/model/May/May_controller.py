"""Define controller than manages exchange of information between social network and firms


Created: 22/12/2023
"""

# imports

from package.model.social_network import Social_Network
from package.model.firm_manager import Firm_Manager 

class Controller:
    def __init__(self, parameters_controller):
        
        self.parameters_controller = parameters_controller#save copy in the object for ease of access
        self.parameters_social_network = parameters_controller["parameters_social_network"]
        self.parameters_firm_manager = parameters_controller["parameters_firm_manager"]
        self.parameters_carbon_policy = parameters_controller["parameters_carbon_policy"]

        self.t_controller = 0
        self.save_timeseries_data_state = parameters_controller["save_timeseries_data_state"]
        self.compression_factor_state = parameters_controller["compression_factor_state"]


        #TIME STUFF
        self.burn_in_no_OD = parameters_controller["burn_in_no_OD"] 
        self.burn_in_duration_no_policy = parameters_controller["burn_in_duration_no_policy"] 
        self.policy_duration = parameters_controller["policy_duration"]

        #create firm manager
        
        self.parameters_firm_manager["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_firm_manager["compression_factor_state"] = self.compression_factor_state
        self.parameters_firm_manager["num_individuals"] = self.parameters_social_network["num_individuals"]
        self.parameters_firm = parameters_controller["parameters_firm"]
        self.parameters_firm["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_firm["compression_factor_state"] = self.compression_factor_state
        self.parameters_firm["emissions_intensity_penalty"] = self.parameters_social_network["emissions_intensity_penalty"]
        #self.parameters_firm["firm_budget"] = self.parameters_firm_manager["N"]*self.parameters_firm["research_cost"]
        #print("Budget", self.parameters_firm["firm_budget"])

        #create social network
        self.parameters_social_network["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_social_network["compression_factor_state"] = self.compression_factor_state
        self.parameters_social_network["J"] = self.parameters_firm_manager["J"]

        #CARBON PRICING
        self.parameters_social_network["carbon_price"] = self.parameters_carbon_policy["carbon_price"]
        self.parameters_social_network["carbon_price_state"] = self.parameters_carbon_policy["carbon_price_state"]

        if (self.burn_in_no_OD + self.burn_in_duration_no_policy) == 0:
            self.carbon_price = self.parameters_carbon_policy["carbon_price"]
            self.parameters_firm_manager["carbon_price"] = self.carbon_price
        else:
            self.carbon_price = 0

        self.parameters_firm_manager["carbon_price"] = self.carbon_price
        #IN THE CASE OF AR1 SET UP STUFF IN SOCIAL NETWORK
        if  self.parameters_carbon_policy["carbon_price_state"] == "AR1":
            self.parameters_social_network["ar_1_coefficient"] = self.parameters_carbon_policy["ar_1_coefficient"] 
            self.parameters_social_network["noise_mean"] = self.parameters_carbon_policy["noise_mean"]  
            self.parameters_social_network["noise_sigma"] = self.parameters_carbon_policy["noise_sigma"] 
        elif self.parameters_carbon_policy["carbon_price_state"] == "normal":
            self.parameters_social_network["noise_sigma"] = self.parameters_carbon_policy["noise_sigma"] 

        #CREATE FIRMS    
        self.firm_manager = Firm_Manager(self.parameters_firm_manager, self.parameters_firm)

        self.parameters_social_network["burn_in_no_OD"] = self.burn_in_no_OD
        self.parameters_social_network["burn_in_duration_no_policy"] = self.burn_in_duration_no_policy
        self.parameters_social_network["policy_duration"] = self.policy_duration      

        #GET FIRM PRICES
        self.parameters_social_network["prices_vec"] = self.firm_manager.prices_vec
        self.parameters_social_network["emissions_intensities_vec"] = self.firm_manager.emissions_intensities_vec
        #print("self.parameters_social_network",self.parameters_social_network)
        #quit()
        self.parameters_individual = parameters_controller["parameters_individual"]
        self.social_network = Social_Network(self.parameters_social_network, self.parameters_individual)


        #update values for the next step
        self.emissions_intensities_vec = self.firm_manager.emissions_intensities_vec
        self.prices_vec = self.firm_manager.prices_vec
        self.consumed_quantities_vec_firms = self.social_network.consumed_quantities_vec_firms
        self.segment_consumer_count_vec = self.social_network.segment_consumer_count

    def next_step(self):
        self.t_controller+=1

        #NOTE HERE THAT FIRMS REACT FIRST TO THE 

        # Update firms based on the social network and market conditions
        car_vec, prices_vec = self.firm_manager.next_step(self.consumed_quantities_vec_firms, self.carbon_price, self.consumer_preferences_vec)

        # Update social network based on firm preferences
        car_count_vec_firms, carbon_price, segment_consumer_count = self.social_network.next_step(car_vec, prices_vec)
        
        #update values for the next step
        self.segment_consumer_count_vec = segment_consumer_count
        self.car_vec = car_vec
        self.car_count_vec_firms =  car_count_vec_firms
        self.carbon_price = carbon_price


