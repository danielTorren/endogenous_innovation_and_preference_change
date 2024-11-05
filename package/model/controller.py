"""Define controller than manages exchange of information between social network and firms
Created: 22/12/2023
"""

# imports
from package.model import firmManager
from package.model.nkModel import NKModel
from package.model.socialNetwork import Social_Network
from package.model.firmManager import Firm_Manager 
from package.model.publicTransport import Rural_Public_Transport
from package.model.publicTransport import Urban_Public_Transport
from package.model.centralizedIdGenerator import IDGenerator
from package.model.secondHandMerchant import SecondHandMerchant
import numpy as np

class Controller:
    def __init__(self, parameters_controller):

        self.unpack_controller_parameters(parameters_controller)
        self.manage_carbon_price()
        self.manage_time()
        self.setup_id_gen()

        #SET UP LANDSCAPES
        self.setup_ICE_landscape_parameters()
        self.setup_EV_landscape_parameters()
        self.setup_ICE_landscape()
        self.setup_EV_landscape()
        self.setup_urban_public_transport()
        self.setup_rural_public_transport()
        self.setup_second_hand_market()

        #create firms and social networks
        self.setup_firm_manager_parameters()
        self.setup_social_network_parameters()

        self.setup_gamma()
        self.gen_firms()
        #NEED TO CREATE INIT OPTIONS
        self.parameters_social_network["init_car_vec"] = self.firm_manager.cars_on_sale_all_firms
        self.gen_social_network()#users have chosen a vehicle
        #pass information across one time
        self.firm_manager.input_social_network_data(self.social_network.beta_vector, self.social_network.origin_vector, self.social_network.environmental_awareness_vec, self.social_network.ev_adoption_vec)
        #Need to calculate sum U give the consumption choices by individuals
        self.firm_manager.generate_market_data_init()

    def unpack_controller_parameters(self,parameters_controller):
        
        #CONTROLLER PARAMETERS:
        self.parameters_controller = parameters_controller#save copy in the object for ease of access
        self.parameters_social_network = parameters_controller["parameters_social_network"]
        self.parameters_firm_manager = parameters_controller["parameters_firm_manager"]
        self.parameters_ICE_landscape = parameters_controller["parameters_ICE_landscape"]
        self.parameters_EV_landscape = parameters_controller["parameters_EV_landscape"]
        self.parameters_urban_public_transport = parameters_controller["parameters_urban_public_transport"]
        self.parameters_rural_public_transport = parameters_controller["parameters_rural_public_transport"]
        self.parameters_public_transport =  parameters_controller["parameters_public_transport"]

        self.parameters_carbon_policy = parameters_controller["parameters_carbon_policy"]
        self.parameters_future_carbon_policy = parameters_controller["parameters_future_carbon_policy"]
        self.parameters_future_information_provision_policy = parameters_controller["parameters_future_information_provision_policy"]

        self.t_controller = 0
        self.save_timeseries_data_state = parameters_controller["save_timeseries_data_state"]
        self.compression_factor_state = parameters_controller["compression_factor_state"]
        
        #TIME STUFF
        self.duration_no_carbon_price = parameters_controller["duration_no_carbon_price"] 
        self.duration_small_carbon_price = parameters_controller["duration_small_carbon_price"] 
        self.duration_large_carbon_price = parameters_controller["duration_large_carbon_price"] 

        self.duration_no_EV = parameters_controller["duration_no_EV"]
        #self.duration_EV = parameters_controller["duration_EV"]

        self.time_steps_max = parameters_controller["time_steps_max"]

    #############################################################################################################################
    #DEAL WITH CARBON PRICE

    def manage_time(self):
        #Manage time
        self.policy_start_time = self.duration_no_carbon_price
        self.future_policy_start_time = self.policy_start_time + self.duration_small_carbon_price

    def manage_carbon_price(self):

        #CARBON PRICE - PAST (this is small fuel tax)
        self.carbon_price_state = self.parameters_carbon_policy["carbon_price_state"]
        self.carbon_price_policy = self.parameters_carbon_policy["carbon_price"]

        if self.carbon_price_state == "linear" and self.duration_small_carbon_price > 0:
            self.carbon_price_init = self.parameters_carbon_policy["carbon_price_init"]
            self.carbon_price_policy_gradient = (self.carbon_price_policy - self.carbon_price_init) /self.duration_small_carbon_price
        
        #CARBON PRICE - FUTURE
        self.future_carbon_price_state = self.parameters_future_carbon_policy["carbon_price_state"]
        self.future_carbon_price_policy = self.parameters_future_carbon_policy["carbon_price"]
        if self.future_carbon_price_state == "linear" and self.duration_large_carbon_price > 0:
            self.future_carbon_price_init = self.parameters_future_carbon_policy["carbon_price_init"]
            self.future_carbon_price_policy_gradient = (self.future_carbon_price_policy - self.future_carbon_price_init)/self.duration_large_carbon_price
        
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
    
    #############################################################################################################################

    def setup_id_gen(self):
        self.IDGenerator_firms = IDGenerator()# CREATE ID GENERATOR FOR FIRMS

    def setup_ICE_landscape_parameters(self):
        #Create NK model
        self.landscape_seed_ICE= self.parameters_ICE_landscape["landscape_seed"]
        self.N_ICE = int(round(self.parameters_ICE_landscape["N"]))
        self.K_ICE = int(round(self.parameters_ICE_landscape["K"]))
        self.A_ICE = self.parameters_ICE_landscape["A"]
        self.rho_ICE = self.parameters_ICE_landscape["rho"]

    def setup_EV_landscape_parameters(self):
        #Create NK model
        self.landscape_seed_EV= self.parameters_EV_landscape["landscape_seed"]
        self.N_EV = int(round(self.parameters_EV_landscape["N"]))
        self.K_EV = int(round(self.parameters_EV_landscape["K"]))
        self.A_EV = self.parameters_EV_landscape["A"]
        self.rho_EV = self.parameters_EV_landscape["rho"]

    def setup_public_transport_parameters(self):

        self.urban_public_transport_attributes = self.parameters_urban_public_transport["attributes"]
        self.price_Urban_Public_Transport = self.parameters_urban_public_transport["price"]

        self.rural_public_transport_attributes = self.parameters_rural_public_transport["attributes"]
        self.price_Rural_Public_Transport = self.parameters_rural_public_transport["price"]

    def setup_firm_manager_parameters(self):
        #TRANSFERING COMMON INFORMATION
        #FIRM MANAGER
        self.parameters_firm_manager["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_firm_manager["compression_factor_state"] = self.compression_factor_state
        self.parameters_firm_manager["num_individuals"] = self.parameters_social_network["num_individuals"]
        self.parameters_firm_manager["carbon_price"] = self.carbon_price
        self.parameters_firm_manager["IDGenerator_firms"] = self.IDGenerator_firms
        self.parameters_firm_manager["kappa"] = self.parameters_social_network["kappa"]
        self.parameters_firm_manager["price_constant"] = self.parameters_social_network["price_constant"]

    
    def setup_social_network_parameters(self):
        #create social network
        self.parameters_social_network["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_social_network["compression_factor_state"] = self.compression_factor_state
        self.parameters_social_network["policy_start_time"] = self.policy_start_time      
        self.parameters_social_network["carbon_price"] = self.carbon_price
        self.parameters_social_network["carbon_price_state"] = self.parameters_carbon_policy["carbon_price_state"]

    def setup_ICE_landscape(self):    
        self.ICE_landscape = NKModel(self.N_ICE, self.K_ICE, self.A_ICE, self.rho_ICE, self.landscape_seed_ICE)

    def setup_EV_landscape(self):
        self.EV_landscape = NKModel(self.N_EV, self.K_EV, self.A_EV, self.rho_EV, self.landscape_seed_EV)

    def setup_urban_public_transport(self):
        self.urban_public_tranport = Urban_Public_Transport(self.urban_public_transport_attributes, self.parameters_public_transport, self.price_Urban_Public_Transport)

    def setup_rural_public_transport(self):
        self.rural_public_tranport = Rural_Public_Transport(self.rural_public_transport_attributes, self.parameters_public_transport, self.price_Rural_Public_Transport)

    def setup_second_hand_market(self):
        self.second_hand_merchant = SecondHandMerchant(unique_id = -3)

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

        return gamma_values
    
    def gen_firms(self):
        #CREATE FIRMS    
        self.firm_manager = Firm_Manager(self.parameters_firm_manager)
    
    def gen_social_network(self):
        self.social_network = Social_Network(self.parameters_social_network)#MUST GO SECOND AS CONSUMERS NEED TO MAKE FIRST CAR CHOICE

    def update_carbon_price(self):
        self.carbon_price = self.carbon_price_time_series[self.t_controller]

    def update_firms(self):
        # Update firms based on the social network and market conditions   #carbon_price, ev_adoption_state_vec, environmental_awareness_arr
        cars_on_sale_all_firms = self.firm_manager.next_step(self.carbon_price, self.ev_adoption_state_vec)
        return cars_on_sale_all_firms
    
    def update_social_network(self, vehicles_available):
        # Update social network based on firm preferences
        ev_adoption_state_vec, vehicles_chosen_list = self.social_network.next_step(self.carbon_price, vehicles_available)

        return ev_adoption_state_vec, vehicles_chosen_list
    
    def mix_in_vehicles(self):
        
        vehicles_available = [self.urban_public_tranport,self.rural_public_tranport] + self.second_hand_merchant.cars_on_sale + self.cars_on_sale_all_firms

        return vehicles_available

    def next_step(self):
        self.t_controller+=1

        self.update_carbon_price()
        self.cars_on_sale_all_firms = self.update_firms()
        vehicles_available = self.mix_in_vehicles()
        self.ev_adoption_state_vec, self.vehicles_chosen_list = self.update_social_network(vehicles_available)




