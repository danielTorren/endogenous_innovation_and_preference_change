"""Define controller than manages exchange of information between social network and firms
Created: 22/12/2023
"""

# imports
from ast import Raise
from package.model.nkModel import NKModel
from package.model.firmManager import Firm_Manager 
from package.model.centralizedIdGenerator import IDGenerator
from package.model.secondHandMerchant import SecondHandMerchant
import numpy as np
from package.model.socialNetworkUsers import Social_Network
import pandas as pd

class Controller:
    def __init__(self, parameters_controller):

        self.unpack_controller_parameters(parameters_controller)

        self.gen_time_series_calibration_scenarios_policies()
        self.update_time_series_data()

        self.setup_id_gen()

        #SET UP LANDSCAPES
        self.setup_ICE_landscape(self.parameters_ICE)
        self.setup_EV_landscape(self.parameters_EV)
        self.setup_second_hand_market()
        
        #create firms and social networks
        self.setup_firm_manager_parameters()
        self.setup_firm_parameters()
        self.setup_social_network_parameters()
        self.setup_vehicle_users_parameters()

        self.gen_firms()

        #NEED TO CREATE INIT OPTIONS
        self.cars_on_sale_all_firms = self.firm_manager.cars_on_sale_all_firms
        self.second_hand_cars = self.get_second_hand_cars()
        
        self.parameters_social_network["init_car_options"] =  self.cars_on_sale_all_firms 
        self.parameters_social_network["old_cars"] = self.firm_manager.old_cars

        #self.parameters_social_network["init_vehicle_options"] = self.mix_in_vehicles()
        self.gen_social_network()#users have chosen a vehicle
        self.consider_ev_vec = self.social_network.consider_ev_vec
        #NEED THE LIST OF VEHICLES CHOSEN to record data
        self.vehicles_chosen_list = self.social_network.current_vehicles

        #pass information across one time
        self.firm_manager.input_social_network_data(self.social_network.beta_vec, self.social_network.gamma_vec, self.social_network.consider_ev_vec)
        #Need to calculate sum U give the consumption choices by individuals
        self.firm_manager.generate_market_data()

        if self.save_timeseries_data_state:
            self.social_network.set_up_time_series_social_network()
            self.firm_manager.set_up_time_series_firm_manager()
            self.time_series = []
            self.set_up_time_series_controller()
        
    def unpack_controller_parameters(self,parameters_controller):
        
        #CONTROLLER PARAMETERS:
        self.parameters_controller = parameters_controller#save copy in the object for ease of access



        self.parameters_social_network = parameters_controller["parameters_social_network"]
        self.parameters_vehicle_user = parameters_controller["parameters_vehicle_user"]
        self.parameters_firm_manager = parameters_controller["parameters_firm_manager"]
        self.parameters_firm = parameters_controller["parameters_firm"]
        self.parameters_ICE = parameters_controller["parameters_ICE"]
        self.parameters_EV = parameters_controller["parameters_EV"]

        self.EV_nu_diff_state = parameters_controller["EV_nu_diff_state"]

        self.t_controller = 0
        self.save_timeseries_data_state = parameters_controller["save_timeseries_data_state"]
        self.compression_factor_state = parameters_controller["compression_factor_state"]
        
        self.age_limit_second_hand = parameters_controller["age_limit_second_hand"]

        #TIME STUFF
        self.duration_no_carbon_price = parameters_controller["duration_no_carbon_price"] 
        self.duration_future = parameters_controller["duration_future"] 

        if self.duration_future > 0: 
            self.full_run_state = True
        else:
            self.full_run_state = False

        #############################################################################################################################
        #DEAL WITH EV RESEARCH
        self.ev_research_start_time = self.parameters_controller["ev_research_start_time"]
        
        self.time_steps_max = parameters_controller["time_steps_max"]

    #############################################################################################################################
    #DEAL WITH CALIBRATION
    def manage_calibration(self):

        
        self.parameters_calibration_data = self.parameters_controller["calibration_data"]
        self.gas_price_california_vec = self.parameters_calibration_data["gas_price_california_vec"]
        self.electricity_price_vec = self.parameters_calibration_data["electricity_price_vec"]
        self.electricity_emissions_intensity_vec = self.parameters_calibration_data["electricity_emissions_intensity_vec"]
        self.tank_ratio_vec = self.parameters_calibration_data["tank_ratio_vec"]
        self.parameters_ICE["e_t"] = self.parameters_calibration_data["gasoline_Kgco2_per_Kilowatt_Hour"]
        
        self.calibration_time_steps = len(self.electricity_emissions_intensity_vec)
        if self.EV_nu_diff_state:
            self.nu_i_t_EV_vec = self.tank_ratio_vec*self.parameters_ICE["nu_i_t"]
        else:
            self.nu_i_t_EV_vec = [self.parameters_ICE["nu_i_t"]]*self.calibration_time_steps 
        
        self.parameters_rebate = self.parameters_controller["parameters_rebate"]
        self.rebate_time_series = np.zeros(self.duration_no_carbon_price)
        self.used_rebate_time_series = np.zeros(self.duration_no_carbon_price)
        
        if self.parameters_controller["EV_rebate_state"]:#IF TRUE IMPLEMENTED
            self.rebate_time_series[self.parameters_rebate["start_time"]:] = self.parameters_rebate["rebate"]
            self.used_rebate_time_series[self.parameters_rebate["start_time"]:] = self.parameters_rebate["used_rebate"]

        self.parameters_social_network["income"] = self.parameters_calibration_data["income"]

    #############################################################################################################################
    #DEAL WITH SCENARIOS

    def manage_scenario(self):

        self.Gas_price_state = self.parameters_controller["parameters_scenarios"]["States"]["Gas_price"]
        self.Electricity_price_state =  self.parameters_controller["parameters_scenarios"]["States"]["Electricity_price"]
        self.Grid_emissions_intensity_state =  self.parameters_controller["parameters_scenarios"]["States"]["Grid_emissions_intensity"]
        self.EV_Substitutability_state =  self.parameters_controller["parameters_scenarios"]["States"]["EV_Substitutability"]
        
        self.Gas_price_2022 = self.parameters_calibration_data["Gas_price_2022"]
        if self.Gas_price_state == "Low":
            self.Gas_price_future = self.Gas_price_2022*self.parameters_controller["parameters_scenarios"]["Values"]["Gas_price"]["Low"]
        elif self.Gas_price_state == "Current":
            self.Gas_price_future = self.Gas_price_2022*self.parameters_controller["parameters_scenarios"]["Values"]["Gas_price"]["Current"]
        elif self.Gas_price_state == "High":
            self.Gas_price_future = self.Gas_price_2022*self.parameters_controller["parameters_scenarios"]["Values"]["Gas_price"]["High"]
        else:
            raise ValueError("Invalid gas price state")
        self.gas_price_series_future = np.linspace(self.Gas_price_2022, self.Gas_price_future, self.duration_future)

        self.Electricity_price_2022 = self.parameters_calibration_data["Electricity_price_2022"]
        if self.Electricity_price_state == "Low":
            self.Electricity_price_future = self.Electricity_price_2022*self.parameters_controller["parameters_scenarios"]["Values"]["Electricity_price"]["Low"]
        elif self.Electricity_price_state == "Current":
            self.Electricity_price_future = self.Electricity_price_2022*self.parameters_controller["parameters_scenarios"]["Values"]["Electricity_price"]["Current"]
        elif self.Electricity_price_state == "High":
            self.Electricity_price_future = self.Electricity_price_2022*self.parameters_controller["parameters_scenarios"]["Values"]["Electricity_price"]["High"]
        else:
            raise ValueError("Invalid electricity price state")
        self.electricity_price_series_future = np.linspace(self.Electricity_price_2022, self.Electricity_price_future, self.duration_future)
        
        self.Grid_emissions_intensity_2022 = self.parameters_calibration_data["Electricity_emissions_intensity_2022"]
        if self.Grid_emissions_intensity_state == "Weaker":
            self.Grid_emissions_intensity_future = self.Grid_emissions_intensity_2022*self.parameters_controller["parameters_scenarios"]["Values"]["Grid_emissions_intensity"]["Weaker"]
        elif self.Grid_emissions_intensity_state == "Decarbonised":
            self.Grid_emissions_intensity_future = self.Grid_emissions_intensity_2022*self.parameters_controller["parameters_scenarios"]["Values"]["Grid_emissions_intensity"]["Decarbonised"]
        else:
            raise ValueError("Invalid Grid emissions intensity state")
        self.grid_emissions_intensity_series_future = np.linspace(self.Grid_emissions_intensity_2022, self.Grid_emissions_intensity_future, self.duration_future)
        
        self.EV_Substitutability_2022 = self.nu_i_t_EV_vec[-1]#TAKE THE LAST VALUE
        if self.EV_Substitutability_state == "Improved":
            self.EV_Substitutability_future = self.EV_Substitutability_2022*self.parameters_controller["parameters_scenarios"]["Values"]["EV_Substitutability"]["Improved"]
        elif self.EV_Substitutability_state == "Parity":
            self.EV_Substitutability_future = self.EV_Substitutability_2022*self.parameters_controller["parameters_scenarios"]["Values"]["EV_Substitutability"]["Parity"]
        else:
            raise ValueError("Invalid EV Substitutability state")
        self.EV_Substitutability_future = np.linspace(self.EV_Substitutability_2022, self.EV_Substitutability_future, self.duration_future)

    #############################################################################################################################
    #DEAL WITH POLICIES
    def manage_policies(self):
        
        self.Carbon_price_state = self.parameters_controller["parameters_policies"]["States"]["Carbon_price"]
        self.Adoption_subsidy_state =  self.parameters_controller["parameters_policies"]["States"]["Adoption_subsidy"]

        # Carbon price calculation
        if self.Carbon_price_state == "Zero":
            self.future_carbon_price_state = self.parameters_controller["parameters_policies"]["Values"]["Carbon_price"]["Zero"]["carbon_price_state"]
            self.future_carbon_price_init = self.parameters_controller["parameters_policies"]["Values"]["Carbon_price"]["Zero"]["carbon_price_init"]
            self.future_carbon_price_policy = self.parameters_controller["parameters_policies"]["Values"]["Carbon_price"]["Zero"]["carbon_price"]
        elif self.Carbon_price_state == "Low":
            self.future_carbon_price_state = self.parameters_controller["parameters_policies"]["Values"]["Carbon_price"]["Low"]["carbon_price_state"]
            self.future_carbon_price_init = self.parameters_controller["parameters_policies"]["Values"]["Carbon_price"]["Low"]["carbon_price_init"]
            self.future_carbon_price_policy = self.parameters_controller["parameters_policies"]["Values"]["Carbon_price"]["Low"]["carbon_price"]
        elif self.Carbon_price_state == "High":
            self.future_carbon_price_state = self.parameters_controller["parameters_policies"]["Values"]["Carbon_price"]["High"]["carbon_price_state"]
            self.future_carbon_price_init = self.parameters_controller["parameters_policies"]["Values"]["Carbon_price"]["High"]["carbon_price_init"]
            self.future_carbon_price_policy = self.parameters_controller["parameters_policies"]["Values"]["Carbon_price"]["High"]["carbon_price"]
        else:
            raise ValueError("Invalid Carbon price state")
        #DEAL WITH CARBON PRICE
        self.carbon_price_time_series = self.calculate_carbon_price_time_series()

        # Adoption subsidy calculation
        if self.Adoption_subsidy_state == "Zero":
            self.Adoption_subsidy = self.parameters_controller["parameters_policies"]["Values"]["Adoption_subsidys"]["Zero"]["rebate"]
            self.Used_adoption_subsidy = self.parameters_controller["parameters_policies"]["Values"]["Adoption_subsidys"]["Zero"]["used_rebate"]
        elif self.Adoption_subsidy_state == "Low":
            self.Adoption_subsidy = self.parameters_controller["parameters_policies"]["Values"]["Adoption_subsidys"]["Low"]["rebate"]
            self.Used_adoption_subsidy = self.parameters_controller["parameters_policies"]["Values"]["Adoption_subsidys"]["Low"]["used_rebate"]
        elif self.Adoption_subsidy_state == "High":
            self.Adoption_subsidy = self.parameters_controller["parameters_policies"]["Values"]["Adoption_subsidys"]["High"]["rebate"]
            self.Used_adoption_subsidy = self.parameters_controller["parameters_policies"]["Values"]["Adoption_subsidys"]["High"]["used_rebate"]
        else:
            raise ValueError("Invalid Adoption subsidy state")
        self.rebate_time_series_future = np.asarray([self.Adoption_subsidy]*self.duration_future)
        self.used_rebate_time_series_future = np.asarray([self.Used_adoption_subsidy]*self.duration_future)
        
    #############################################################################################################################
    #DEAL WITH CARBON PRICE

    def calculate_carbon_price_time_series(self):
        time_series = np.arange(self.time_steps_max + 1)
        carbon_price_series = []
        
        for t in time_series:
            carbon_price = self.calculate_price_at_time(t)
            carbon_price_series.append(carbon_price)
        
        return carbon_price_series

    def calculate_price_at_time(self, t):
        if self.future_carbon_price_policy > 0 and self.duration_future > 0:
            if t < self.duration_no_carbon_price:
                return 0
            
            if t >= self.duration_no_carbon_price:
                relative_time = t - self.duration_no_carbon_price
                return self.calculate_growth(
                    relative_time, 
                    self.duration_future,
                    self.future_carbon_price_init,
                    self.future_carbon_price_policy,
                    self.future_carbon_price_state
                )
        else:
            return 0

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

            
        else:
            raise ValueError(f"Unknown growth type: {growth_type}")

    #############################################################################################################################

    def gen_time_series_calibration_scenarios_policies(self):
        """Put together the calibration, scenarios and policies data"""

        self.manage_calibration()
        
        if self.full_run_state:
            self.manage_scenario()
            self.manage_policies() 

            #NOW STAPLE THE STUFF TOGETHER TO GET ONE THING
            #CALIRBATION TIME_STEPS
            self.gas_price_california_vec = np.concatenate((self.gas_price_california_vec, self.gas_price_series_future), axis=None) 
            self.electricity_price_vec =  np.concatenate((self.electricity_price_vec, self.electricity_price_series_future ), axis=None) 
            self.electricity_emissions_intensity_vec = np.concatenate((self.electricity_emissions_intensity_vec,self.grid_emissions_intensity_series_future ), axis=None) 
            self.nu_i_t_EV_vec = np.concatenate((self.nu_i_t_EV_vec,self.EV_Substitutability_future), axis=None) 
            self.rebate_time_series = np.concatenate((self.rebate_time_series,self.rebate_time_series_future ), axis=None) 
            self.used_rebate_time_series = np.concatenate((self.used_rebate_time_series,self.used_rebate_time_series_future ), axis=None) 
        else:
            self.carbon_price_time_series = np.asarray(([0])*len( self.electricity_price_vec))
        #FINISH JOING THE STUFF HERE FOR THE SCENARIOS AND POLICY TIME SERIES

    def update_time_series_data(self):
        #EV research state
        if self.t_controller == self.ev_research_start_time:
            for firm in self.firm_manager.firms_list:
                firm.ev_reserach_bool = True
                
                    
        #carbon price
        self.carbon_price = self.carbon_price_time_series[self.t_controller]

        #update_prices_and_emmisions
        self.gas_price = self.gas_price_california_vec[self.t_controller]
        self.electricity_price = self.electricity_price_vec[self.t_controller]
        self.electricity_emissions_intensity = self.electricity_emissions_intensity_vec[self.t_controller]
        self.nu_i_t_EV = self.nu_i_t_EV_vec[self.t_controller]    
        self.rebate = self.rebate_time_series[self.t_controller]
        self.used_rebate = self.used_rebate_time_series[self.t_controller]

    #############################################################################################################################
    def setup_id_gen(self):
        self.IDGenerator_firms = IDGenerator()# CREATE ID GENERATOR FOR FIRMS

    def setup_firm_manager_parameters(self):
        #TRANSFERING COMMON INFORMATION
        #FIRM MANAGER
        self.parameters_firm_manager["num_individuals"] = self.parameters_social_network["num_individuals"]
        self.parameters_firm_manager["carbon_price"] = self.carbon_price
        self.parameters_firm_manager["IDGenerator_firms"] = self.IDGenerator_firms
        self.parameters_firm_manager["kappa"] = self.parameters_vehicle_user["kappa"]
        self.parameters_firm_manager["N"] = self.parameters_ICE["N"]
        self.parameters_firm_manager["cars_init_state"] = self.parameters_controller["cars_init_state"]

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
        self.parameters_firm["delta"] = self.parameters_ICE["delta"]#ASSUME THAT BOTH ICE AND EV HAVE SAME DEPRECIATIONS RATE
        self.parameters_firm["carbon_price"] = self.carbon_price
        self.parameters_firm["gas_price"] = self.gas_price
        self.parameters_firm["electricity_price"] = self.electricity_price
        self.parameters_firm["electricity_emissions_intensity"] = self.electricity_emissions_intensity
        self.parameters_firm["rebate"] = self.rebate 

        if self.t_controller == self.ev_research_start_time:
            self.parameters_firm["ev_reserach_bool"] = True
        else:
            self.parameters_firm["ev_reserach_bool"] = False

    def setup_social_network_parameters(self):
        #create social network
        self.parameters_social_network["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_social_network["compression_factor_state"] = self.compression_factor_state
        self.parameters_social_network["policy_start_time"] = self.duration_no_carbon_price
        self.parameters_social_network["carbon_price"] = self.carbon_price
        self.parameters_social_network["IDGenerator_firms"] = self.IDGenerator_firms
        self.parameters_social_network["second_hand_merchant"] = self.second_hand_merchant
        self.parameters_social_network["gas_price"] = self.gas_price
        self.parameters_social_network["electricity_price"] = self.electricity_price
        self.parameters_social_network["electricity_emissions_intensity"] = self.electricity_emissions_intensity
        self.parameters_social_network["rebate"] = self.rebate 

        self.parameters_social_network["used_rebate"] = self.used_rebate 
        self.parameters_social_network["used_rebate"] = self.used_rebate 
        self.parameters_social_network["cars_init_state"] = self.parameters_controller["cars_init_state"]

    def setup_vehicle_users_parameters(self):
        self.parameters_vehicle_user["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_vehicle_user["compression_factor_state"] = self.compression_factor_state

    def setup_ICE_landscape(self, parameters_ICE):    
        self.ICE_landscape = NKModel(parameters_ICE)

    def setup_EV_landscape(self, parameters_EV):
        self.EV_landscape = NKModel(parameters_EV)

    def setup_second_hand_market(self):
        self.second_hand_merchant = SecondHandMerchant(unique_id = -3, age_limit_second_hand = self.age_limit_second_hand)
    
    def gen_firms(self):
        #CREATE FIRMS    
        self.parameters_ICE["eta"] = self.parameters_vehicle_user["eta"]
        self.parameters_ICE["fuel_cost_c"]  = self.gas_price
        self.parameters_EV["eta"] = self.parameters_vehicle_user["eta"]
        self.parameters_EV["fuel_cost_c"] = self.electricity_price 
        self.parameters_EV["e_t"] = self.electricity_emissions_intensity
        self.parameters_EV["nu_i_t"] = self.nu_i_t_EV

        self.firm_manager = Firm_Manager(self.parameters_firm_manager, self.parameters_firm, self.parameters_ICE, self.parameters_EV, self.ICE_landscape, self.EV_landscape)
    
    def gen_social_network(self):
        #self.social_network = Social_Network(self.parameters_social_network, self.parameters_vehicle_user)#MUST GO SECOND AS CONSUMERS NEED TO MAKE FIRST CAR CHOICE
        self.social_network = Social_Network(self.parameters_social_network, self.parameters_vehicle_user)#MUST GO SECOND AS CONSUMERS NEED TO MAKE FIRST CAR CHOICE

    def update_firms(self):
        cars_on_sale_all_firms = self.firm_manager.next_step(self.carbon_price, self.consider_ev_vec, self.vehicles_chosen_list, self.gas_price, self.electricity_price, self.electricity_emissions_intensity, self.nu_i_t_EV, self.rebate)
        return cars_on_sale_all_firms
    
    def update_social_network(self):
        # Update social network based on firm preferences
        consider_ev_vec, vehicles_chosen_list = self.social_network.next_step(self.carbon_price,  self.second_hand_cars, self.cars_on_sale_all_firms, self.gas_price, self.electricity_price, self.electricity_emissions_intensity, self.nu_i_t_EV, self.rebate, self.used_rebate)

        return consider_ev_vec, vehicles_chosen_list
    
    def set_up_time_series_controller(self):
        self.history_gas_price = []
        self.history_electricity_price = []
        self.history_electricity_emissions_intensity = []
        self.history_nu_i_t_EV = []
        self.history_rebate = []
        self.history_used_rebate = []

    def save_timeseries_controller(self):
        self.history_gas_price.append(self.gas_price)
        self.history_electricity_price.append(self.electricity_price)
        self.history_electricity_emissions_intensity.append(self.electricity_emissions_intensity)
        self.history_nu_i_t_EV.append(self.nu_i_t_EV)
        self.history_rebate.append(self.rebate)
        self.history_used_rebate.append(self.used_rebate)

    def manage_saves(self):
        #DO it hear to avoid having to record the time in the subobjects
        if self.save_timeseries_data_state and (self.t_controller % self.compression_factor_state == 0):
            self.social_network.save_timeseries_data_social_network()
            self.firm_manager.save_timeseries_data_firm_manager()
            self.second_hand_merchant.save_timeseries_second_hand_merchant()
            self.time_series.append(self.t_controller)

            self.save_timeseries_controller()

    def get_second_hand_cars(self):

        self.second_hand_merchant.next_step(self.gas_price, self.electricity_price, self.electricity_emissions_intensity, self.nu_i_t_EV)

        return self.second_hand_merchant.cars_on_sale

    ################################################################################################

    def next_step(self):
        self.t_controller+=1#I DONT KNOW IF THIS SHOULD BE AT THE START OR THE END OF THE TIME STEP? But the code works if its at the end lol
        #print("TIME STEP", self.t_controller)

        self.update_time_series_data()
        self.second_hand_cars = self.get_second_hand_cars()
        self.cars_on_sale_all_firms = self.update_firms()
        self.consider_ev_vec, self.vehicles_chosen_list = self.update_social_network()

        self.manage_saves()


        



        



