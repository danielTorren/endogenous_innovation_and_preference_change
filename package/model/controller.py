"""Define controller than manages exchange of information between social network and firms
Created: 22/12/2023
"""

# imports
from package.model.nkModel import NKModel
from package.model.firmManager import Firm_Manager 
from package.model.publicTransport import Public_Transport
from package.model.centralizedIdGenerator import IDGenerator
from package.model.secondHandMerchant import SecondHandMerchant
import numpy as np
from package.model.socialNetworkUsers import Social_Network
import pandas as pd

class Controller:
    def __init__(self, parameters_controller):

        self.unpack_controller_parameters(parameters_controller)
        self.manage_time()
        self.manage_carbon_price()
        self.update_prices_and_emmisions()
        self.setup_id_gen()

        #SET UP LANDSCAPES
        self.setup_ICE_landscape(self.parameters_ICE)
        self.setup_EV_landscape(self.parameters_EV)
        self.setup_second_hand_market()

        self.setup_urban_public_transport(self.parameters_urban_public_transport)
        self.setup_rural_public_transport(self.parameters_rural_public_transport)
        
        #create firms and social networks
        self.setup_firm_manager_parameters()
        self.setup_firm_parameters()
        self.setup_social_network_parameters()
        self.setup_vehicle_users_parameters()

        self.gen_firms()

        #NEED TO CREATE INIT OPTIONS
        self.cars_on_sale_all_firms = self.firm_manager.cars_on_sale_all_firms
        self.public_option_list = [self.urban_public_tranport,self.rural_public_tranport]
        self.second_hand_cars = self.get_second_hand_cars()
        
        self.parameters_social_network["public_transport"] = self.public_option_list
        self.parameters_social_network["init_car_options"] =  self.cars_on_sale_all_firms 

        #self.parameters_social_network["init_vehicle_options"] = self.mix_in_vehicles()
        self.gen_social_network()#users have chosen a vehicle
        self.consider_ev_vec = self.social_network.consider_ev_vec
        #NEED THE LIST OF VEHICLES CHOSEN to record data
        self.vehicles_chosen_list = self.social_network.current_vehicles

        #pass information across one time
        self.firm_manager.input_social_network_data(self.social_network.beta_vec, self.social_network.origin_vec, self.social_network.gamma_vec, self.social_network.consider_ev_vec)
        #Need to calculate sum U give the consumption choices by individuals
        self.firm_manager.generate_market_data()

        np.random.seed(parameters_controller["choice_seed"])#SET ONCE ALL SET UP HAS BEEN DONE

        if self.save_timeseries_data_state:
            self.social_network.set_up_time_series_social_network()
            self.firm_manager.set_up_time_series_firm_manager()
            self.rural_public_tranport.set_up_time_series_firm()
            self.urban_public_tranport.set_up_time_series_firm()
            self.time_series = []

    def unpack_controller_parameters(self,parameters_controller):
        
        #CONTROLLER PARAMETERS:
        self.parameters_controller = parameters_controller#save copy in the object for ease of access
        self.parameters_calibration_data = parameters_controller["calibration_data"]
        
        self.gas_price_california_vec = self.parameters_calibration_data["Real Dollars per Kilowatt-Hour"].to_numpy()
        self.electricity_price_vec = self.parameters_calibration_data["Real Dollars per Kilowatt-Hour (City Average)"].to_numpy()
        self.electricity_emissions_intensity_vec = self.parameters_calibration_data["KgCO2 per Kilowatt-Hour"].to_numpy()
        self.calibration_time_steps = len(self.electricity_emissions_intensity_vec)
        
        self.parameters_social_network = parameters_controller["parameters_social_network"]
        self.parameters_vehicle_user = parameters_controller["parameters_vehicle_user"]
        self.parameters_firm_manager = parameters_controller["parameters_firm_manager"]
        self.parameters_firm = parameters_controller["parameters_firm"]
        self.parameters_ICE = parameters_controller["parameters_ICE"]
        self.parameters_EV = parameters_controller["parameters_EV"]

        self.parameters_urban_public_transport = parameters_controller["parameters_urban_public_transport"]
        self.parameters_rural_public_transport = parameters_controller["parameters_rural_public_transport"]

    
        self.parameters_carbon_policy = parameters_controller["parameters_carbon_policy"]
        self.parameters_future_carbon_policy = parameters_controller["parameters_future_carbon_policy"]

        self.t_controller = 0
        self.save_timeseries_data_state = parameters_controller["save_timeseries_data_state"]
        self.compression_factor_state = parameters_controller["compression_factor_state"]
        
        self.age_limit_second_hand = parameters_controller["age_limit_second_hand"]

        #TIME STUFF
        self.duration_no_carbon_price = parameters_controller["duration_no_carbon_price"] 
        self.duration_small_carbon_price = parameters_controller["duration_small_carbon_price"] 
        self.duration_large_carbon_price = parameters_controller["duration_large_carbon_price"] 

        #############################################################################################################################
        #DEAL WITH EV RESEARCH
        self.ev_research_start_time = self.parameters_controller["ev_research_start_time"]
        
        self.time_steps_max = parameters_controller["time_steps_max"]
        
    #############################################################################################################################
    #DEAL WITH CARBON PRICE

    def manage_time(self):
        #Manage time
        self.policy_start_time = self.duration_no_carbon_price
        self.future_policy_start_time = self.policy_start_time + self.duration_small_carbon_price

    def manage_carbon_price(self):
        # Initial setup from parameters
        self.carbon_price_state = self.parameters_carbon_policy["carbon_price_state"]  # "linear", "quadratic", "exponential", "logarithmic"
        self.carbon_price_init = self.parameters_carbon_policy.get("carbon_price_init", 0)
        self.carbon_price_policy = self.parameters_carbon_policy["carbon_price"]
        
        self.future_carbon_price_state = self.parameters_future_carbon_policy["carbon_price_state"]
        self.future_carbon_price_init = self.parameters_future_carbon_policy.get("carbon_price_init", 0)
        self.future_carbon_price_policy = self.parameters_future_carbon_policy["carbon_price"]
        
        
        self.carbon_price_time_series = self.calculate_carbon_price_time_series()

        self.carbon_price = self.carbon_price_time_series[0]

    def calculate_carbon_price_time_series(self):
        time_series = np.arange(self.time_steps_max + 1)
        carbon_price_series = []
        
        for t in time_series:
            carbon_price = self.calculate_price_at_time(t)
            carbon_price_series.append(carbon_price)
        
        return carbon_price_series

    def calculate_price_at_time(self, t):
        if t < self.policy_start_time:
            return 0
            
        # First policy period
        if t >= self.policy_start_time and t < self.future_policy_start_time:
            duration = t - self.policy_start_time
            total_duration = self.future_policy_start_time - self.policy_start_time
            return self.calculate_growth(
                duration, 
                total_duration,
                self.carbon_price_init,
                self.carbon_price_policy,
                self.carbon_price_state
            )
        
        # Future policy period
        if t >= self.future_policy_start_time:
            duration = t - self.future_policy_start_time
            total_duration = self.time_steps_max - self.future_policy_start_time
            return self.calculate_growth(
                duration,
                total_duration,
                self.future_carbon_price_init,
                self.future_carbon_price_policy,
                self.future_carbon_price_state
            )

    def calculate_growth(self, t, total_duration, start_price, end_price, growth_type):
        if growth_type == "flat":
            return end_price
            
        elif growth_type == "linear":
            slope = (end_price - start_price) / total_duration
            return start_price + slope * t
            
        elif growth_type == "quadratic":
            a = (end_price - start_price) / (total_duration ** 2)
            return start_price + a * (t ** 2)
            
        elif growth_type == "exponential":
            r = np.log(end_price / start_price) / total_duration if start_price > 0 else 0
            return start_price * np.exp(r * t)
            
        elif growth_type == "logarithmic":
            if t == 0:
                return start_price
            k = (end_price - start_price) / np.log(total_duration + 1)
            return start_price + k * np.log(t + 1)
            
        else:
            raise ValueError(f"Unknown growth type: {growth_type}")
    

    #############################################################################################################################


    def setup_id_gen(self):
        self.IDGenerator_firms = IDGenerator()# CREATE ID GENERATOR FOR FIRMS

    def setup_firm_manager_parameters(self):
        #TRANSFERING COMMON INFORMATION
        #FIRM MANAGER
        #self.parameters_firm_manager["save_timeseries_data_state"] = self.save_timeseries_data_state
        #self.parameters_firm_manager["compression_factor_state"] = self.compression_factor_state
        self.parameters_firm_manager["num_individuals"] = self.parameters_social_network["num_individuals"]
        self.parameters_firm_manager["carbon_price"] = self.carbon_price
        self.parameters_firm_manager["IDGenerator_firms"] = self.IDGenerator_firms
        self.parameters_firm_manager["kappa"] = self.parameters_vehicle_user["kappa"]
        self.parameters_firm_manager["N"] = self.parameters_ICE["N"]

    def setup_firm_parameters(self):
        self.parameters_firm["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_firm["compression_factor_state"] = self.compression_factor_state
        self.parameters_firm["IDGenerator_firms"] = self.IDGenerator_firms
        self.parameters_firm["kappa"] = self.parameters_vehicle_user["kappa"]
        self.parameters_firm["alpha"] = self.parameters_vehicle_user["alpha"]
        self.parameters_firm["ICE_landscape"] = self.ICE_landscape
        self.parameters_firm["EV_landscape"] = self.EV_landscape
        self.parameters_firm["eta"] = self.parameters_vehicle_user["eta"]
        self.parameters_firm["r"] = self.parameters_vehicle_user["r"]
        self.parameters_firm["delta_z"] = self.parameters_ICE["delta_z"]#ASSUME THAT BOTH ICE AND EV HAVE SAME DEPRECIATIONS RATE
        self.parameters_firm["carbon_price"] = self.carbon_price
        self.parameters_firm["gas_price"] = self.gas_price
        self.parameters_firm["electricity_price"] = self.electricity_price
        self.parameters_firm["electricity_emissions_intensity"] = self.electricity_emissions_intensity

    def setup_social_network_parameters(self):
        #create social network
        self.parameters_social_network["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_social_network["compression_factor_state"] = self.compression_factor_state
        self.parameters_social_network["policy_start_time"] = self.policy_start_time      
        self.parameters_social_network["carbon_price"] = self.carbon_price
        self.parameters_social_network["carbon_price_state"] = self.parameters_carbon_policy["carbon_price_state"]
        self.parameters_social_network["IDGenerator_firms"] = self.IDGenerator_firms
        self.parameters_social_network["second_hand_merchant"] = self.second_hand_merchant
        self.parameters_social_network["urban_public_transport_emissions"] = self.parameters_urban_public_transport["production_emissions"]
        self.parameters_social_network["rural_public_transport_emissions"] = self.parameters_rural_public_transport["production_emissions"]
        self.parameters_social_network["gas_price"] = self.gas_price
        self.parameters_social_network["electricity_price"] = self.electricity_price
        self.parameters_social_network["electricity_emissions_intensity"] = self.electricity_emissions_intensity

    def setup_vehicle_users_parameters(self):
        self.parameters_vehicle_user["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_vehicle_user["compression_factor_state"] = self.compression_factor_state

    def setup_ICE_landscape(self, parameters_ICE):    
        self.ICE_landscape = NKModel(parameters_ICE)

    def setup_EV_landscape(self, parameters_EV):
        self.EV_landscape = NKModel(parameters_EV)

    def setup_urban_public_transport(self, parameters_urban_public_transport):
        parameters_urban_public_transport["eta"] = self.parameters_vehicle_user["eta"]
        self.urban_public_tranport = Public_Transport(parameters=parameters_urban_public_transport)

    def setup_rural_public_transport(self, parameters_rural_public_transport):
        parameters_rural_public_transport["eta"] = self.parameters_vehicle_user["eta"]
        self.rural_public_tranport = Public_Transport(parameters=parameters_rural_public_transport)

    def setup_second_hand_market(self):
        self.second_hand_merchant = SecondHandMerchant(unique_id = -3, age_limit_second_hand = self.age_limit_second_hand)
    
    def gen_firms(self):
        #CREATE FIRMS    
        self.parameters_ICE["eta"] = self.parameters_vehicle_user["eta"]
        self.parameters_ICE["fuel_cost_c_z"]  = self.gas_price

        self.parameters_EV["eta"] = self.parameters_vehicle_user["eta"]
        self.parameters_EV["fuel_cost_c_z"] = self.electricity_price 
        self.parameters_EV["e_z_t"] = self.electricity_emissions_intensity

        self.firm_manager = Firm_Manager(self.parameters_firm_manager, self.parameters_firm, self.parameters_ICE, self.parameters_EV, self.ICE_landscape, self.EV_landscape)
    
    def gen_social_network(self):
        #self.social_network = Social_Network(self.parameters_social_network, self.parameters_vehicle_user)#MUST GO SECOND AS CONSUMERS NEED TO MAKE FIRST CAR CHOICE
        self.social_network = Social_Network(self.parameters_social_network, self.parameters_vehicle_user)#MUST GO SECOND AS CONSUMERS NEED TO MAKE FIRST CAR CHOICE

    def update_carbon_price(self):
        self.carbon_price = self.carbon_price_time_series[self.t_controller]

    def update_prices_and_emmisions(self):
        self.gas_price = self.gas_price_california_vec[self.t_controller]
        self.electricity_price = self.electricity_price_vec[self.t_controller]
        self.electricity_emissions_intensity = self.electricity_emissions_intensity_vec[self.t_controller]

    def update_firms(self):
        cars_on_sale_all_firms = self.firm_manager.next_step(self.carbon_price, self.consider_ev_vec, self.vehicles_chosen_list, self.gas_price, self.electricity_price, self.electricity_emissions_intensity)
        return cars_on_sale_all_firms
    
    def update_social_network(self):
        # Update social network based on firm preferences
        consider_ev_vec, vehicles_chosen_list = self.social_network.next_step(self.carbon_price,  self.second_hand_cars, self.public_option_list, self.cars_on_sale_all_firms, self.gas_price, self.electricity_price, self.electricity_emissions_intensity)

        return consider_ev_vec, vehicles_chosen_list
    
    def update_public_transport(self):
        #DO it hear to avoid having to record the time in the subobjects
        if self.save_timeseries_data_state and (self.t_controller % self.compression_factor_state == 0):
            self.social_network.save_timeseries_data_social_network()
            self.firm_manager.save_timeseries_data_firm_manager()
            self.rural_public_tranport.save_timeseries_data_firm()
            self.urban_public_tranport.save_timeseries_data_firm()
            self.second_hand_merchant.save_timeseries_second_hand_merchant()
            self.time_series.append(self.t_controller)

    def get_second_hand_cars(self):

        self.second_hand_merchant.next_step(self.gas_price, self.electricity_price, self.electricity_emissions_intensity)

        return self.second_hand_merchant.cars_on_sale

    def update_ev_reserach_state(self):
        if self.t_controller == self.ev_research_start_time:
            for firm in self.firm_manager.firms_list:
                firm.ev_reserach_bool = True
    ################################################################################################

    def next_step(self):
        self.t_controller+=1
        print("TIME STEP", self.t_controller)

        self.update_ev_reserach_state()
        self.update_carbon_price()
        if self.t_controller < self.calibration_time_steps:
            self.update_prices_and_emmisions()
        self.second_hand_cars = self.get_second_hand_cars()
        self.cars_on_sale_all_firms = self.update_firms()
        #vehicles_available = self.mix_in_vehicles()
        self.consider_ev_vec, self.vehicles_chosen_list = self.update_social_network()

        self.update_public_transport()



