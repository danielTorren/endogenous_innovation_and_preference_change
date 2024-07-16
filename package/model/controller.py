"""Define controller than manages exchange of information between social network and firms
Created: 22/12/2023
"""

# imports

from package.model.social_network import Social_Network
#from package.model.gpt_firm_manager import Firm_Manager 
from package.model.firm_manager import Firm_Manager 
from package.model.centralized_ID_generator import IDGenerator
import numpy as np

class Controller:
    def __init__(self, parameters_controller):
        
        self.parameters_controller = parameters_controller#save copy in the object for ease of access
        self.parameters_social_network = parameters_controller["parameters_social_network"]
        self.parameters_firm_manager = parameters_controller["parameters_firm_manager"]
        self.parameters_carbon_policy = parameters_controller["parameters_carbon_policy"]
        self.parameters_future_carbon_policy = parameters_controller["parameters_future_carbon_policy"]
        self.parameters_future_information_provision_policy = parameters_controller["parameters_future_information_provision_policy"]

        self.t_controller = 0
        self.save_timeseries_data_state = parameters_controller["save_timeseries_data_state"]
        self.compression_factor_state = parameters_controller["compression_factor_state"]
        self.time_steps_max = parameters_controller["time_steps_max"]

        #TIME STUFF
        self.duration_no_OD_no_stock_no_policy = parameters_controller["duration_no_OD_no_stock_no_policy"] 
        self.duration_OD_no_stock_no_policy = parameters_controller["duration_OD_no_stock_no_policy"] 
        self.duration_OD_stock_no_policy = parameters_controller["duration_OD_stock_no_policy"] 
        self.duration_OD_stock_policy = parameters_controller["duration_OD_stock_policy"]
        self.duration_future = parameters_controller["duration_future"]

        self.policy_start_time = self.duration_no_OD_no_stock_no_policy + self.duration_OD_no_stock_no_policy + self.duration_OD_stock_no_policy
        self.future_policy_start_time = self.policy_start_time + self.duration_OD_stock_policy

        #CARBON PRICE - PAST
        self.carbon_price_state = self.parameters_carbon_policy["carbon_price_state"]
        self.carbon_price_policy = self.parameters_carbon_policy["carbon_price"]
        if self.carbon_price_state == "linear" and self.duration_OD_stock_policy > 0:
            self.carbon_price_init = self.parameters_carbon_policy["carbon_price_init"]
            self.carbon_price_policy_gradient = (self.carbon_price_policy - self.carbon_price_init) /self.duration_OD_stock_policy
        
        #CARBON PRICE - FUTURE
        self.future_carbon_price_state = self.parameters_future_carbon_policy["carbon_price_state"]
        self.future_carbon_price_policy = self.parameters_future_carbon_policy["carbon_price"]
        if self.future_carbon_price_state == "linear" and self.duration_future > 0:
            self.future_carbon_price_init = self.parameters_future_carbon_policy["carbon_price_init"]
            self.future_carbon_price_policy_gradient = (self.future_carbon_price_policy - self.future_carbon_price_init)/self.duration_future
        
        self.carbon_price_time_series = self.calculate_carbon_price_time_series()
        self.carbon_price = self.carbon_price_time_series[0]

        # CREATE ID GENERATOR FOR FIRMS
        self.IDGenerator_firms = IDGenerator()

        #TRANSFERING COMMON INFORMATION
        #FIRM MANAGER
        self.parameters_firm_manager["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_firm_manager["compression_factor_state"] = self.compression_factor_state
        self.parameters_firm_manager["num_individuals"] = self.parameters_social_network["num_individuals"]
        #self.parameters_firm_manager["gamma"] = self.parameters_social_network["gamma"] 
        self.parameters_firm_manager["carbon_price"] = self.carbon_price
        self.parameters_firm_manager["IDGenerator_firms"] = self.IDGenerator_firms
        self.parameters_firm_manager["kappa"] = self.parameters_social_network["kappa"]
        self.parameters_firm_manager["price_constant"] = self.parameters_social_network["price_constant"]

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


        self.utility_boost_const = 2#self.parameters_firm_manager["markup"] + 1 + self.parameters_carbon_policy["carbon_price"] + 0.1#JUST TO BE SAFE
        self.parameters_firm_manager["utility_boost_const"] = self.utility_boost_const
        self.parameters_social_network["utility_boost_const"] = self.utility_boost_const

        ##################################
        #HET GAMMA
        self.heterogenous_gamma_state = self.parameters_social_network["heterogenous_gamma_state"]
        if self.heterogenous_gamma_state:
            self.min_value_gamma = self.parameters_social_network["gamma"] 
            self.lambda_poisson_gamma = self.parameters_social_network["lambda_poisson_gamma"]
            self.gamma_vals = self.generate_gamma_values(self.min_value_gamma, self.lambda_poisson_gamma, self.parameters_social_network["num_individuals"])
        else:
            self.gamma_vals = np.asarray([self.parameters_social_network["gamma"]]*self.parameters_social_network["num_individuals"])  # weight for car quality

        self.parameters_social_network["gamma_vals"] = self.gamma_vals
        self.parameters_firm_manager["gamma"] = np.median(self.gamma_vals)#TAKE THE MEDIAN OF VALS

        #CREATE FIRMS    
        #self.firm_manager = Firm_Manager(self.parameters_firm_manager, self.parameters_firm)
        self.firm_manager = Firm_Manager(self.parameters_firm_manager)

        self.parameters_social_network["init_car_vec"] = self.firm_manager.cars_on_sale_all_firms

        #GET FIRM PRICES
        self.social_network = Social_Network(self.parameters_social_network)#MUST GO SECOND AS CONSUMERS NEED TO MAKE FIRST CAR CHOICE
        
        #update values for the next step
        self.controller_low_carbon_preference_arr = self.social_network.low_carbon_preference_arr

        self.history_carbon_price = [self.carbon_price]

    def generate_gamma_values(self, min_value, lambda_poisson, num_individuals):
        """
        Generates heterogeneous gamma values for n agents, constrained between min_value and 1,
        using a Poisson-like distribution.

        Parameters:
        n (int): Number of agents.
        min_value (float): Minimum value for gamma (0 < min_value < 1).
        lambda_poisson (float): Lambda parameter for the Poisson distribution.

        Returns:
        np.ndarray: Array of gamma values for n agents.
        """
        if min_value <= 0 or min_value >= 1:
            raise ValueError("min_value must be between 0 and 1.")
        
        # Generate Poisson-like distributed values
        poisson_values = np.random.poisson(lambda_poisson, num_individuals)
        
        # Normalize the Poisson values to fit between min_value and 1
        max_poisson_value = np.max(poisson_values)
        normalized_values = poisson_values / max_poisson_value
        
        # Scale normalized values to fit within the range [min_value, 1]
        gamma_values = min_value + (1 - min_value) * normalized_values

        #print("GAMAM VALS",gamma_values)

        return gamma_values

    def assign_carbon_price(self, time, carbon_price):
        if time < self.policy_start_time:
            carbon_price = 0
        elif time == self.policy_start_time:
            if self.carbon_price_state == "flat":
                carbon_price = self.carbon_price_policy
            elif self.carbon_price_state == "linear":
                carbon_price = self.carbon_price_init
        elif (time > self.policy_start_time) and (time < self.future_policy_start_time):
            if self.carbon_price_state == "linear":
                carbon_price += self.carbon_price_policy_gradient
            else:
                carbon_price = self.carbon_price_policy
        elif time == self.future_policy_start_time:
            if self.future_carbon_price_state == "flat":
                carbon_price = self.future_carbon_price_policy
            elif self.future_carbon_price_state == "linear":
                carbon_price = self.future_carbon_price_init
        elif time > self.future_policy_start_time:
            if self.future_carbon_price_state == "linear":
                carbon_price += self.future_carbon_price_policy_gradient
            else:
                carbon_price = self.future_carbon_price_policy
        return carbon_price
    
    def calculate_carbon_price_time_series(self):
        time_series = np.arange(self.time_steps_max)
        carbon_price  = 0
        carbon_price_series = [carbon_price]
        for t in time_series:
            carbon_price = self.assign_carbon_price(t, carbon_price)
            carbon_price_series.append(carbon_price)
        return carbon_price_series


    def next_step(self):
        self.t_controller+=1
        #print("self.t_controller", self.t_controller)
        #self.update_carbon_price()
        self.carbon_price = self.carbon_price_time_series[self.t_controller]

        self.history_carbon_price.append(self.carbon_price)
        # Update firms based on the social network and market conditions
        cars_on_sale_all_firms = self.firm_manager.next_step(self.carbon_price, self.controller_low_carbon_preference_arr)

        # Update social network based on firm preferences
        controller_low_carbon_preference_arr = self.social_network.next_step(self.carbon_price, cars_on_sale_all_firms)

        self.cars_on_sale_all_firms  = cars_on_sale_all_firms#
        self.controller_low_carbon_preference_arr = controller_low_carbon_preference_arr



