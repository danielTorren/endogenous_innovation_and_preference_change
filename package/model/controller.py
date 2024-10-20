"""Define controller than manages exchange of information between social network and firms
Created: 22/12/2023
"""

# imports
from package.model.nk_model import NKModel
from package.model.social_network import Social_Network
#from package.model.gpt_firm_manager import Firm_Manager 
from package.model.firm_manager import Firm_Manager 
from package.model.public_transport import Rural_Public_Transport
from package.model.public_transport import Urban_Public_Transport
from package.model.centralized_ID_generator import IDGenerator
import numpy as np

class Controller:
    def __init__(self, parameters_controller):
        
        self.unpack_controller_parameters(parameters_controller)
        self.manage_time()
        self.manage_carbon_price()
        self.setup_id_gen()

        #SET UP LANDSCAPES
        self.setup_ICE_landscape_paramters()
        self.setup_EV_landscape_paramters()
        self.setup_urban_public_transport_landscape_paramters()
        self.setup_rural_public_transport_landscape_paramters()
        self.setup_ICE_landscape()
        self.setup_EV_landscape()
        self.setup_urban_public_transport()
        self.setup_rural_public_transport()

        #create firms and social networks
        self.setup_firm_manager_parameters()
        self.setup_social_network_parameters()

        self.setup_gamma()
        self.gen_firms()
        self.gen_social_network()

        self.set_up_time_series_controller()

    def unpack_controller_parameters(self,parameters_controller):
        
        #CONTROLLER PARAMETERS:
        self.parameters_controller = parameters_controller#save copy in the object for ease of access
        self.parameters_social_network = parameters_controller["parameters_social_network"]
        self.parameters_firm_manager = parameters_controller["parameters_firm_manager"]
        self.parameters_ICE_landscape = parameters_controller["parameters_ICE_landscape"]
        self.parameters_EV_landscape = parameters_controller["parameters_EVandscape"]
        self.parameters_urban_public_transport = parameters_controller["parameters_urban_public_transport"]
        self.parameters_rural_public_transport = parameters_controller["parameters_rural_public_transport"]

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

    def manage_time(self):
        #Manage time
        self.policy_start_time = self.duration_no_OD_no_stock_no_policy + self.duration_OD_no_stock_no_policy + self.duration_OD_stock_no_policy
        self.future_policy_start_time = self.policy_start_time + self.duration_OD_stock_policy

    def manage_carbon_price(self):

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

    def calculate_carbon_price_time_series(self):
        time_series = np.arange(self.time_steps_max)
        carbon_price  = 0
        carbon_price_series = [carbon_price]
        for t in time_series:
            carbon_price = self.assign_carbon_price(t, carbon_price)
            carbon_price_series.append(carbon_price)
        return carbon_price_series
    
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

    def setup_id_gen(self):
        self.IDGenerator_firms = IDGenerator()# CREATE ID GENERATOR FOR FIRMS

    def setup_ICE_landscape_paramters(self):
        #Create NK model
        self.landscape_seed_ICE= self.parameters_ICE_landscape["landscape_seed"]
        self.N_ICE = int(round(self.parameters_ICE_landscape["N"]))
        self.K_ICE = int(round(self.parameters_ICE_landscape["K"]))
        self.A_ICE = self.parameters_ICE_landscape["A"]
        self.rho_ICE = self.parameters_ICE_landscape["rho"]

    def setup_EV_landscape_paramters(self):
        #Create NK model
        self.landscape_seed_EV= self.parameters_EV_landscape["landscape_seed"]
        self.N_EV = int(round(self.parameters_EV_landscape["N"]))
        self.K_EV = int(round(self.parameters_EV_landscape["K"]))
        self.A_EV = self.parameters_EV_landscape["A"]
        self.rho_EV = self.parameters_EV_landscape["rho"]
    
    def setup_urban_public_transport_landscape_paramters(self):
        #Create NK model
        self.urban_public_transport_attributes = self.parameters_urban_public_transport["attributes"]

    def setup_rural_public_transport_landscape_paramters(self):
        #Create NK model
        self.rural_public_transport_attributes = self.parameters_rural_public_transport["attributes"]

    def setup_firm_manager_parameters(self):
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
        #ADD THE LANDSCAPES
        self.parameters_firm_manager["ICE_landscape"] = self.landscape_seed_ICE
        self.parameters_firm_manager["EV_landscape"] = self.landscape_seed_EV
    
    def setup_social_network_parameters(self):
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
        #ADD PUBLIC TRANSPORT OPTIONS
        self.parameters_social_network["urban_public_transport"] = self.urban_public_tranport
        self.parameters_social_network["rural_public_transport"] = self.rural_public_tranport

    def setup_ICE_landscape(self):    
        self.ICE_landscape = NKModel(self.N_ICE, self.K_ICE, self.A_ICE, self.rho_ICE, self.landscape_seed_ICE)

    def setup_EV_landscape(self):
        self.EV_landscape = NKModel(self.N_EV, self.K_EV, self.A_EV, self.rho_EV, self.landscape_seed_EV)

    def setup_urban_public_transport(self):
        self.urban_public_tranport = Urban_Public_Transport(self.urban_public_transport_attributes)

    def setup_rural_public_transport(self):
        self.rural_public_tranport = Rural_Public_Transport(self.rural_public_transport_attributes)

    def setup_gamma(self):
        """
        This is to set up a heterogenous sensitivity to the price of cars, this is used to represent wealth inequality within the model
        """
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
    
    def gen_firms(self):
        #CREATE FIRMS    
        self.firm_manager = Firm_Manager(self.parameters_firm_manager)
        self.parameters_social_network["init_car_vec"] = self.firm_manager.cars_on_sale_all_firms
    
    def gen_social_network(self):
        self.social_network = Social_Network(self.parameters_social_network)#MUST GO SECOND AS CONSUMERS NEED TO MAKE FIRST CAR CHOICE
        #update values for the next step
        self.controller_ev_adoption_state_arr = self.social_network.ev_adoption_state_arr
        self.controller_environmental_preference_arr =  self.social_network.environmental_preference_arr
        self.controller_price_preference_arr = self.social_network.price_preference_arr

    def update_carbon_price(self):
        self.carbon_price = self.carbon_price_time_series[self.t_controller]

    def update_firms(self):
        # Update firms based on the social network and market conditions
        cars_on_sale_all_firms = self.firm_manager.next_step(self.carbon_price, self.ev_adoption_state_arr, self.environmental_preference_arr, self.price_preference_arr)
        return cars_on_sale_all_firms
    
    def update_social_network(self):
        # Update social network based on firm preferences
        self.controller_ev_adoption_state_arr, self.controller_environmental_preference_arr, self.controller_price_preference_arr = self.social_network.next_step(self.carbon_price, cars_on_sale_all_firms)
    
    def set_up_time_series_controller(self):
        self.history_carbon_price = [self.carbon_price]

    def save_timeseries_data_controller(self):
        self.history_carbon_price.append(self.carbon_price)

    def next_step(self):
        self.t_controller+=1

        self.update_carbon_price()
        self.save_timeseries_data_controller()
        cars_on_sale_all_firms = self.update_firms()
        self.update_social_network()

        self.cars_on_sale_all_firms  = cars_on_sale_all_firms




