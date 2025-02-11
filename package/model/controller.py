"""Define controller than manages exchange of information between social network and firms
Created: 22/12/2023
"""
from package.model.nkModel import NKModel
from package.model.firmManager import Firm_Manager 
from package.model.centralizedIdGenerator import IDGenerator
from package.model.secondHandMerchant import SecondHandMerchant
from package.model.socialNetworkUsers import Social_Network
import numpy as np

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

        self.gen_users_parameters()

        self.gen_firms()

        #NEED TO CREATE INIT OPTIONS
        self.cars_on_sale_all_firms = self.firm_manager.cars_on_sale_all_firms

        
        self.second_hand_cars = []#EMPTY INITIATILLY
        
        self.parameters_social_network["init_car_options"] =  self.cars_on_sale_all_firms 
        self.parameters_social_network["old_cars"] = self.firm_manager.old_cars

        #self.parameters_social_network["init_vehicle_options"] = self.mix_in_vehicles()
        self.gen_social_network()#users have chosen a vehicle

        self.firm_manager.add_social_network(self.social_network)

        self.consider_ev_vec = self.social_network.consider_ev_vec
        #NEED THE LIST OF VEHICLES CHOSEN to record data
        self.new_bought_vehicles = self.social_network.current_vehicles

        #UPDATE SECOND HADN MERCHANT WITH THE DATA
        self.second_hand_merchant.calc_median(self.social_network.beta_vec, self.social_network.gamma_vec)        

        #pass information across one time
        self.firm_manager.input_social_network_data(self.social_network.beta_vec, self.social_network.gamma_vec, self.social_network.consider_ev_vec, self.beta_bins)

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
        self.parameters_second_hand = parameters_controller["parameters_second_hand"]
        
        self.parameters_rebate_calibration = self.parameters_controller["parameters_rebate_calibration"]

        self.parameters_EV["min_Quality"] = self.parameters_EV["min_Quality"]
        self.parameters_EV["max_Quality"] = self.parameters_EV["max_Quality"]
        self.parameters_EV["min_Cost"] = self.parameters_ICE["min_Cost"]
        self.parameters_EV["max_Cost"] = self.parameters_ICE["max_Cost"]
        self.parameters_EV["delta"] = self.parameters_ICE["delta"]
                           
        self.t_controller = 0
        self.save_timeseries_data_state = parameters_controller["save_timeseries_data_state"]
        self.compression_factor_state = parameters_controller["compression_factor_state"]
        
        #TIME STUFF#
        self.duration_burn_in = parameters_controller["duration_burn_in"] 
        self.duration_no_carbon_price = parameters_controller["duration_no_carbon_price"] 
        self.duration_future = parameters_controller["duration_future"] 

        if self.duration_future > 0: 
            self.full_run_state = True
        else:
            self.full_run_state = False

        #############################################################################################################################
        #DEAL WITH EV RESEARCH
        self.ev_research_start_time = self.duration_burn_in + self.parameters_controller["ev_research_start_time"]
        self.ev_production_start_time = self.duration_burn_in + self.parameters_controller["ev_production_start_time"]

        if self.ev_research_start_time > self.ev_production_start_time:
            raise ValueError("EV Production before research")

        self.time_steps_max = parameters_controller["time_steps_max"]

    def gen_users_parameters(self):

        self.num_individuals = self.parameters_social_network["num_individuals"]
         
        #CHI
        self.a_innovativeness = self.parameters_social_network["a_innovativeness"]
        self.b_innovativeness = self.parameters_social_network["b_innovativeness"]
        self.random_state_chi = np.random.RandomState(self.parameters_social_network["init_vals_innovative_seed"])
        innovativeness_vec_init_unrounded = self.random_state_chi.beta(self.a_innovativeness, self.b_innovativeness, size=self.num_individuals)
        self.chi_vec = np.round(innovativeness_vec_init_unrounded, 1)
        self.ev_adoption_state_vec = np.zeros(self.num_individuals)

        #BETA
        self.random_state_beta = np.random.RandomState(self.parameters_social_network["init_vals_price_seed"])
        self.beta_vec = self.generate_beta_values_quintiles(self.num_individuals,  self.parameters_social_network["income"])

        unique_beta_vals, counts = np.unique(self.beta_vec, return_counts=True)

        # Add a small delta to the start and end to ensure proper binning
        delta = (unique_beta_vals[1] - unique_beta_vals[0]) / 2
        self.beta_bins = np.concatenate([
            [unique_beta_vals[0] - delta],  # Lower edge
            unique_beta_vals[:-1] + delta,  # Midpoints
            [unique_beta_vals[-1] + delta]  # Upper edge
        ])


        #GAMMA
        self.random_state_gamma = np.random.RandomState(self.parameters_social_network["init_vals_environmental_seed"])
        self.WTP_mean = self.parameters_social_network["WTP_mean"]
        self.WTP_sd = self.parameters_social_network["WTP_sd"]
        self.car_lifetime_months = self.parameters_social_network["car_lifetime_months"]
        WTP_vec_unclipped = self.random_state_gamma.normal(loc = self.WTP_mean, scale = self.WTP_sd, size = self.num_individuals)
        self.WTP_vec = np.clip(WTP_vec_unclipped, a_min = self.parameters_social_network["gamma_epsilon"], a_max = np.inf)
        self.gamma_vec = self.beta_vec*self.WTP_vec/self.car_lifetime_months

        #social network data
        self.parameters_social_network["beta_vec"] = self.beta_vec 
        self.parameters_social_network["gamma_vec"] = self.gamma_vec 
        self.parameters_social_network["chi_vec"] = self.chi_vec 
        self.parameters_social_network["beta_median"] = np.median(self.beta_vec)
        self.parameters_social_network["gamma_median"] = np.median(self.beta_vec)

        #firm data
        self.parameters_firm_manager["beta_threshold"] = np.percentile(self.beta_vec, self.parameters_firm_manager["beta_threshold_percentile"])
        self.parameters_firm_manager["beta_val_empty_upper"] = np.percentile(self.beta_vec, self.parameters_firm_manager["beta_threshold_percentile"]/2)
        self.parameters_firm_manager["beta_val_empty_lower"] = np.percentile(self.beta_vec, self.parameters_firm_manager["beta_threshold_percentile"]+ (1-self.parameters_firm_manager["beta_threshold_percentile"])/2)
        
        self.parameters_firm_manager["gamma_threshold"] = np.percentile(self.beta_vec, self.parameters_firm_manager["gamma_threshold_percentile"])
        self.parameters_firm_manager["gamma_val_empty_upper"] = np.percentile(self.gamma_vec , self.parameters_firm_manager["gamma_threshold_percentile"]/2)
        self.parameters_firm_manager["gamma_val_empty_upper"] = np.percentile(self.gamma_vec,  self.parameters_firm_manager["gamma_threshold_percentile"]+ (1-self.parameters_firm_manager["gamma_threshold_percentile"])/2)
        
    def generate_beta_values_quintiles(self,n, quintile_incomes):
        """
        Generate a list of beta values for n agents based on quintile incomes.
        Beta for each quintile is calculated as:
            beta = 1 * (lowest_quintile_income / quintile_income)
        
        Args:
            n (int): Total number of agents.
            quintile_incomes (list): List of incomes for each quintile (from lowest to highest).
            
        Returns:
            list: A list of beta values of length n.
        """
        # Calculate beta values for each quintile
        lowest_income = quintile_incomes[0]
        beta_vals = [lowest_income / income for income in quintile_incomes]
        
        # Assign proportions for each quintile (evenly split 20% each)
        proportions = [0.2] * len(quintile_incomes)
        
        # Compute the number of agents for each quintile
        agent_counts = [int(round(p * n)) for p in proportions]
        
        # Adjust for rounding discrepancies to ensure sum(agent_counts) == n
        while sum(agent_counts) < n:
            agent_counts[agent_counts.index(min(agent_counts))] += 1
        while sum(agent_counts) > n:
            agent_counts[agent_counts.index(max(agent_counts))] -= 1
        
        # Generate the beta values list
        beta_list = []
        for count, beta in zip(agent_counts, beta_vals):
            beta_list.extend([beta] * count)
        
        # Shuffle to randomize the order of agents
        self.random_state_beta.shuffle(beta_list)
        
        return np.asarray(beta_list)
    
    #######################################################################################################################################
    def manage_burn_in(self):
        self.burn_in_gas_price_vec = np.asarray([self.calibration_gas_price_california_vec[0]]*self.duration_burn_in)
        self.burn_in_electricity_price_vec = np.asarray([self.calibration_electricity_price_vec[0]]*self.duration_burn_in)
        self.burn_in_electricity_emissions_intensity_vec = np.asarray([self.calibration_electricity_emissions_intensity_vec[0]]*self.duration_burn_in)
        self.burn_in_rebate_time_series = np.zeros(self.duration_burn_in)
        self.burn_in_used_rebate_time_series = np.zeros(self.duration_burn_in)

    #############################################################################################################################
    #DEAL WITH CALIBRATION
    
    def manage_calibration(self):

        self.parameters_ICE["e_t"] = self.parameters_calibration_data["gasoline_Kgco2_per_Kilowatt_Hour"]

        self.calibration_rebate_time_series = np.zeros(self.duration_no_carbon_price + self.duration_future )
        self.calibration_used_rebate_time_series = np.zeros(self.duration_no_carbon_price + self.duration_future)
        
        if self.parameters_controller["EV_rebate_state"]:#IF TRUE IMPLEMENTED
            self.calibration_rebate_time_series[self.parameters_rebate_calibration["start_time"]:] = self.parameters_rebate_calibration["rebate"]
            self.calibration_used_rebate_time_series[self.parameters_rebate_calibration["start_time"]:] = self.parameters_rebate_calibration["used_rebate"]

        self.parameters_social_network["income"] = self.parameters_calibration_data["income"]

    #############################################################################################################################
    #DEAL WITH SCENARIOS

    def manage_scenario(self):

        self.Gas_price_state = self.parameters_controller["parameters_scenarios"]["States"]["Gas_price"]
        self.Electricity_price_state =  self.parameters_controller["parameters_scenarios"]["States"]["Electricity_price"]
        self.Grid_emissions_intensity_state =  self.parameters_controller["parameters_scenarios"]["States"]["Grid_emissions_intensity"]
        
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
        
    #############################################################################################################################
    #DEAL WITH POLICIES

    def manage_policies(self):
        
        self.Carbon_price_state = self.parameters_controller["parameters_policies"]["States"]["Carbon_price"]
        self.Discriminatory_corporate_tax_state =  self.parameters_controller["parameters_policies"]["States"]["Discriminatory_corporate_tax"]
        self.Electricity_subsidy_state =  self.parameters_controller["parameters_policies"]["States"]["Electricity_subsidy"]
        self.Adoption_subsidy_state =  self.parameters_controller["parameters_policies"]["States"]["Adoption_subsidy"]
        self.Adoption_subsidy_used_state =  self.parameters_controller["parameters_policies"]["States"]["Adoption_subsidy_used"]
        self.Production_subsidy_state =  self.parameters_controller["parameters_policies"]["States"]["Production_subsidy"]
        self.Research_subsidy_state =  self.parameters_controller["parameters_policies"]["States"]["Research_subsidy"]

        # Carbon price calculation
        if self.Carbon_price_state == "Zero":
            self.future_carbon_price_state = self.parameters_controller["parameters_policies"]["Values"]["Carbon_price"]["Zero"]["Carbon_price_state"]
            self.future_carbon_price_init = self.parameters_controller["parameters_policies"]["Values"]["Carbon_price"]["Zero"]["Carbon_price_init"]
            self.future_carbon_price_policy = self.parameters_controller["parameters_policies"]["Values"]["Carbon_price"]["Zero"]["Carbon_price"]
        elif self.Carbon_price_state == "Low":
            self.future_carbon_price_state = self.parameters_controller["parameters_policies"]["Values"]["Carbon_price"]["Low"]["Carbon_price_state"]
            self.future_carbon_price_init = self.parameters_controller["parameters_policies"]["Values"]["Carbon_price"]["Low"]["Carbon_price_init"]
            self.future_carbon_price_policy = self.parameters_controller["parameters_policies"]["Values"]["Carbon_price"]["Low"]["Carbon_price"]
        elif self.Carbon_price_state == "High":
            self.future_carbon_price_state = self.parameters_controller["parameters_policies"]["Values"]["Carbon_price"]["High"]["Carbon_price_state"]
            self.future_carbon_price_init = self.parameters_controller["parameters_policies"]["Values"]["Carbon_price"]["High"]["Carbon_price_init"]
            self.future_carbon_price_policy = self.parameters_controller["parameters_policies"]["Values"]["Carbon_price"]["High"]["Carbon_price"]
        else:
            raise ValueError("Invalid Carbon price state")
        #DEAL WITH CARBON PRICE
        self.carbon_price_time_series = self.calculate_carbon_price_time_series()

        # Discriminatory_corporate_tax calculation
        if self.Discriminatory_corporate_tax_state == "Zero":
            self.Discriminatory_corporate_tax = self.parameters_controller["parameters_policies"]["Values"]["Discriminatory_corporate_tax"]["Zero"]["corporate_tax"]
        elif self.Discriminatory_corporate_tax_state == "Low":
            self.Discriminatory_corporate_tax = self.parameters_controller["parameters_policies"]["Values"]["Discriminatory_corporate_tax"]["Low"]["corporate_tax"]
        elif self.Discriminatory_corporate_tax_state == "High":
            self.Discriminatory_corporate_tax = self.parameters_controller["parameters_policies"]["Values"]["Discriminatory_corporate_tax"]["High"]["corporate_tax"]
        else:
            raise ValueError("Invalid Discriminatory_corporate_tax state")
        self.discriminatory_corporate_tax_time_series_future = np.asarray([self.Discriminatory_corporate_tax]*self.duration_future)

        # Electricity_subsidy calculation
        if self.Electricity_subsidy_state == "Zero":
            self.Electricity_subsidy = self.parameters_controller["parameters_policies"]["Values"]["Electricity_subsidy"]["Zero"]["electricity_price_subsidy"]
        elif self.Electricity_subsidy_state == "Low":
            self.Electricity_subsidy = self.parameters_controller["parameters_policies"]["Values"]["Electricity_subsidy"]["Low"]["electricity_price_subsidy"]
        elif self.Electricity_subsidy_state == "High":
            self.Electricity_subsidy = self.parameters_controller["parameters_policies"]["Values"]["Electricity_subsidy"]["High"]["electricity_price_subsidy"]
        else:
            raise ValueError("Invalid electricity_price_subsidy state")
        self.electricity_price_subsidy_time_series_future = np.asarray([self.Electricity_subsidy]*self.duration_future)

        # Adoption subsidy calculation
        if self.Adoption_subsidy_state == "Zero":
            self.Adoption_subsidy = self.parameters_controller["parameters_policies"]["Values"]["Adoption_subsidy"]["Zero"]["rebate"]
        elif self.Adoption_subsidy_state == "Low":
            self.Adoption_subsidy = self.parameters_controller["parameters_policies"]["Values"]["Adoption_subsidy"]["Low"]["rebate"]
        elif self.Adoption_subsidy_state == "High":
            self.Adoption_subsidy = self.parameters_controller["parameters_policies"]["Values"]["Adoption_subsidy"]["High"]["rebate"]
        else:
            raise ValueError("Invalid Adoption subsidy state")
        self.rebate_time_series_future = np.asarray([self.Adoption_subsidy]*self.duration_future)

        if self.Adoption_subsidy_used_state == "Zero":
            self.Used_adoption_subsidy = self.parameters_controller["parameters_policies"]["Values"]["Adoption_subsidy"]["Zero"]["rebate"]
        elif self.Adoption_subsidy_used_state == "Low":
            self.Used_adoption_subsidy = self.parameters_controller["parameters_policies"]["Values"]["Adoption_subsidy"]["Low"]["rebate"]
        elif self.Adoption_subsidy_used_state == "High":
            self.Used_adoption_subsidy = self.parameters_controller["parameters_policies"]["Values"]["Adoption_subsidy"]["High"]["rebate"]
        else:
            raise ValueError("Invalid Adoption subsidy state")
        self.used_rebate_time_series_future = np.asarray([self.Used_adoption_subsidy]*self.duration_future)

        # Production_subsidy calculation
        if self.Production_subsidy_state == "Zero":
            self.Production_subsidy = self.parameters_controller["parameters_policies"]["Values"]["Production_subsidy"]["Zero"]["rebate"]
        elif self.Production_subsidy_state == "Low":
            self.Production_subsidy = self.parameters_controller["parameters_policies"]["Values"]["Production_subsidy"]["Low"]["rebate"]
        elif self.Production_subsidy_state == "High":
            self.Production_subsidy = self.parameters_controller["parameters_policies"]["Values"]["Production_subsidy"]["High"]["rebate"]
        else:
            raise ValueError("Invalid Production_subsidy state")
        self.production_subsidy_time_series_future = np.asarray([self.Production_subsidy]*self.duration_future)

        # Research_subsidy calculation
        if self.Research_subsidy_state == "Zero":
            self.Research_subsidy = self.parameters_controller["parameters_policies"]["Values"]["Research_subsidy"]["Zero"]["rebate"]
        elif self.Research_subsidy_state == "Low":
            self.Research_subsidy = self.parameters_controller["parameters_policies"]["Values"]["Research_subsidy"]["Low"]["rebate"]
        elif self.Research_subsidy_state == "High":
            self.Research_subsidy = self.parameters_controller["parameters_policies"]["Values"]["Research_subsidy"]["High"]["rebate"]
        else:
            raise ValueError("Invalid Research_subsidy state")
        self.research_subsidy_time_series_future = np.asarray([self.Research_subsidy]*self.duration_future)

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
            if t < (self.duration_burn_in + self.duration_no_carbon_price):
                return 0
            
            if t >= (self.duration_burn_in + self.duration_no_carbon_price):
                relative_time = t - (self.duration_burn_in  + self.duration_no_carbon_price)
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
        
        self.parameters_calibration_data = self.parameters_controller["calibration_data"]
        self.calibration_gas_price_california_vec = self.parameters_calibration_data["gas_price_california_vec"]
        self.calibration_electricity_price_vec = self.parameters_calibration_data["electricity_price_vec"]
        self.calibration_electricity_emissions_intensity_vec = self.parameters_calibration_data["electricity_emissions_intensity_vec"]


        self.manage_burn_in()

        self.manage_calibration()

        #JOIN BURN IN AND CALIBRATION
        self.pre_future_gas_price_california_vec = np.concatenate((self.burn_in_gas_price_vec,self.calibration_gas_price_california_vec), axis=None) 
        self.pre_future_electricity_price_vec =  np.concatenate((self.burn_in_gas_price_vec,self.calibration_electricity_price_vec), axis=None) 
        self.pre_future_electricity_emissions_intensity_vec = np.concatenate((self.burn_in_gas_price_vec,self.calibration_electricity_emissions_intensity_vec), axis=None) 
        self.pre_future_rebate_time_series = np.concatenate((self.burn_in_rebate_time_series, self.calibration_rebate_time_series), axis=None) 
        self.pre_future_used_rebate_time_series = np.concatenate((self.burn_in_used_rebate_time_series, self.calibration_used_rebate_time_series), axis=None) 
                
        if self.full_run_state:
            self.manage_scenario()
            self.manage_policies() 

            #NOW STAPLE THE STUFF TOGETHER TO GET ONE THING
            #CALIRBATION TIME_STEPS
            self.gas_price_california_vec = np.concatenate((self.pre_future_gas_price_california_vec, self.gas_price_series_future), axis=None) 
            self.electricity_price_vec =  np.concatenate((self.pre_future_electricity_price_vec, self.electricity_price_series_future ), axis=None) 
            self.electricity_emissions_intensity_vec = np.concatenate((self.pre_future_electricity_emissions_intensity_vec,self.grid_emissions_intensity_series_future ), axis=None) 
            
            #self.rebate_time_series = np.concatenate((self.pre_future_rebate_time_series,self.rebate_time_series_future ), axis=None) 
            #self.used_rebate_time_series = np.concatenate((self.pre_future_used_rebate_time_series,self.used_rebate_time_series_future ), axis=None) 

            #FOR REBATE THE POLICY IS ADDITIONAL TO THE ONE THAT EXISTS SO ADD IT ON TOP NOT CONCATINATED
            self.rebate_time_series =  self.pre_future_rebate_time_series  
            self.rebate_time_series[self.duration_burn_in + self.duration_no_carbon_price:] += self.rebate_time_series_future

            self.used_rebate_time_series = self.pre_future_used_rebate_time_series
            self.used_rebate_time_series[self.duration_burn_in + self.duration_no_carbon_price:] += self.used_rebate_time_series_future

            self.discriminatory_corporate_tax_time_series =  np.concatenate(( np.zeros(self.duration_burn_in + self.duration_no_carbon_price), self.discriminatory_corporate_tax_time_series_future), axis=None) 
            self.electricity_price_subsidy_time_series = np.concatenate(( np.zeros(self.duration_burn_in + self.duration_no_carbon_price), self.electricity_price_subsidy_time_series_future), axis=None) 
            self.production_subsidy_time_series = np.concatenate(( np.zeros(self.duration_burn_in + self.duration_no_carbon_price), self.production_subsidy_time_series_future), axis=None) 
            self.research_subsidy_time_series = np.concatenate(( np.zeros(self.duration_burn_in + self.duration_no_carbon_price), self.research_subsidy_time_series_future), axis=None) 

        else:
            self.gas_price_california_vec = self.pre_future_gas_price_california_vec 
            self.electricity_price_vec = self.pre_future_electricity_price_vec 
            self.electricity_emissions_intensity_vec = self.pre_future_electricity_emissions_intensity_vec
            self.rebate_time_series = self.pre_future_rebate_time_series 
            self.used_rebate_time_series = self.pre_future_used_rebate_time_series
                    
            self.carbon_price_time_series = np.zeros(self.duration_burn_in + self.duration_no_carbon_price)
            self.discriminatory_corporate_tax_time_series = np.zeros(self.duration_burn_in + self.duration_no_carbon_price)
            self.electricity_price_subsidy_time_series = np.zeros(self.duration_burn_in + self.duration_no_carbon_price)
            self.production_subsidy_time_series = np.zeros(self.duration_burn_in + self.duration_no_carbon_price)
            self.research_subsidy_time_series = np.zeros(self.duration_burn_in + self.duration_no_carbon_price)
        
        #FINISH JOING THE STUFF HERE FOR THE SCENARIOS AND POLICY TIME SERIES


    #############################################################################################################################
    def setup_id_gen(self):
        self.IDGenerator_firms = IDGenerator()# CREATE ID GENERATOR FOR FIRMS

    def setup_firm_manager_parameters(self):
        #TRANSFERING COMMON INFORMATION
        #FIRM MANAGER
        self.parameters_firm_manager["num_individuals"] = self.parameters_social_network["num_individuals"]
        self.parameters_firm_manager["carbon_price"] = self.carbon_price
        self.parameters_firm_manager["IDGenerator_firms"] = self.IDGenerator_firms
        self.parameters_firm_manager["kappa"] = int(round(self.parameters_vehicle_user["kappa"]))
        self.parameters_firm_manager["N"] = self.parameters_ICE["N"]
        self.rebate_users_per_pop = self.parameters_social_network["num_individuals"]/self.parameters_rebate_calibration["pop"]
        self.parameters_firm_manager["rebate_count_cap_adjusted"] = self.parameters_rebate_calibration["rebate_count_cap"]*self.rebate_users_per_pop

    def setup_firm_parameters(self):
        self.parameters_firm["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_firm["compression_factor_state"] = self.compression_factor_state
        self.parameters_firm["IDGenerator_firms"] = self.IDGenerator_firms
        self.parameters_firm["kappa"] = int(round(self.parameters_vehicle_user["kappa"]))
        self.parameters_firm["alpha"] = self.parameters_vehicle_user["alpha"]
        self.parameters_firm["ICE_landscape"] = self.ICE_landscape
        self.parameters_firm["EV_landscape"] = self.EV_landscape
        self.parameters_firm["r"] = self.parameters_vehicle_user["r"]
        self.parameters_firm["delta"] = self.parameters_ICE["delta"]#ASSUME THAT BOTH ICE AND EV HAVE SAME DEPRECIATIONS RATE
        self.parameters_firm["carbon_price"] = self.carbon_price
        self.parameters_firm["gas_price"] = self.gas_price
        self.parameters_firm["electricity_price"] = self.electricity_price
        self.parameters_firm["electricity_emissions_intensity"] = self.electricity_emissions_intensity
        self.parameters_firm["rebate"] = self.rebate 
        self.parameters_firm["rebate_low"] = self.parameters_rebate_calibration["rebate_low"]
        self.parameters_firm["d_max"] = self.parameters_social_network["d_max"]
        self.parameters_firm["nu"] = self.parameters_vehicle_user["nu"]

        if self.t_controller == self.ev_research_start_time:
            self.parameters_firm["ev_research_bool"] = True
        else:
            self.parameters_firm["ev_research_bool"] = False

        if self.t_controller == self.ev_production_start_time:
            self.parameters_firm["ev_production_bool"] = True
        else:
            self.parameters_firm["ev_production_bool"] = False

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
        self.parameters_social_network["rebate_low"] = self.parameters_rebate_calibration["rebate_low"]
        self.parameters_social_network["used_rebate_low"] = self.parameters_rebate_calibration["used_rebate_low"]
        self.parameters_social_network["delta"] = self.parameters_ICE["delta"]
        self.parameters_social_network["nu"] = self.parameters_vehicle_user["nu"]

    def setup_vehicle_users_parameters(self):
        self.parameters_vehicle_user["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_vehicle_user["compression_factor_state"] = self.compression_factor_state

    def setup_ICE_landscape(self, parameters_ICE):    

        parameters_ICE["alpha"] = self.parameters_vehicle_user["alpha"]
        parameters_ICE["r"] = self.parameters_vehicle_user["r"]
        parameters_ICE["median_beta"] = 0.5
        parameters_ICE["median_gamma"] = 0.5
        parameters_ICE["fuel_cost"] = self.parameters_calibration_data["gas_price_california_vec"][0]
        parameters_ICE["e_t"] = self.parameters_calibration_data["gasoline_Kgco2_per_Kilowatt_Hour"]
        parameters_ICE["d_max"]= self.parameters_social_network["d_max"]
        self.parameters_ICE["nu"] = self.parameters_vehicle_user["nu"]
        self.ICE_landscape = NKModel(parameters_ICE)

    def setup_EV_landscape(self, parameters_EV):
        parameters_EV["alpha"] = self.parameters_vehicle_user["alpha"]
        parameters_EV["r"] = self.parameters_vehicle_user["r"]
        parameters_EV["delta"] = self.parameters_ICE["delta"]
        parameters_EV["median_beta"] = 0.5#representative 
        parameters_EV["median_gamma"] = 0.5#representative 
        parameters_EV["fuel_cost"] = self.parameters_calibration_data["electricity_price_vec"][0]
        parameters_EV["e_t"] = self.parameters_calibration_data["electricity_emissions_intensity_vec"][0]
        parameters_EV["d_max"]= self.parameters_social_network["d_max"]
        self.parameters_EV["nu"] = self.parameters_vehicle_user["nu"]
        self.EV_landscape = NKModel(parameters_EV)

    def setup_second_hand_market(self):
        self.parameters_second_hand["alpha"] = self.parameters_vehicle_user["alpha"]
        self.parameters_second_hand["r"] = self.parameters_vehicle_user["r"]
        self.parameters_second_hand["d_max"] = self.parameters_social_network["d_max"]
        self.parameters_second_hand["delta"] = self.parameters_ICE["delta"]
        self.parameters_second_hand["kappa"] = self.parameters_vehicle_user["kappa"]
        self.parameters_second_hand["nu"] = self.parameters_vehicle_user["nu"]

        self.second_hand_merchant = SecondHandMerchant(unique_id = -3, parameters_second_hand= self.parameters_second_hand)
    
    def gen_firms(self):
        #CREATE FIRMS    
        self.parameters_ICE["fuel_cost_c"]  = self.gas_price
        self.parameters_EV["fuel_cost_c"] = self.electricity_price 
        self.parameters_EV["e_t"] = self.electricity_emissions_intensity

        self.firm_manager = Firm_Manager(self.parameters_firm_manager, self.parameters_firm, self.parameters_ICE, self.parameters_EV, self.ICE_landscape, self.EV_landscape)
    
    def gen_social_network(self):

        #self.social_network = Social_Network(self.parameters_social_network, self.parameters_vehicle_user)#MUST GO SECOND AS CONSUMERS NEED TO MAKE FIRST CAR CHOICE
        self.social_network = Social_Network(self.parameters_social_network, self.parameters_vehicle_user)#MUST GO SECOND AS CONSUMERS NEED TO MAKE FIRST CAR CHOICE

##########################################################################################################################################
    
    def set_up_time_series_controller(self):
        self.history_gas_price = []
        self.history_electricity_price = []
        self.history_electricity_emissions_intensity = []
        self.history_rebate = []
        self.history_used_rebate = []

    def save_timeseries_controller(self):
        self.history_gas_price.append(self.gas_price)
        self.history_electricity_price.append(self.electricity_price)
        self.history_electricity_emissions_intensity.append(self.electricity_emissions_intensity)
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

##########################################################################################################################################

    def update_time_series_data(self):
        #EV research state
        if self.t_controller == self.ev_research_start_time:
            for firm in self.firm_manager.firms_list:
                firm.ev_research_bool = True
                firm.list_technology_memory = firm.list_technology_memory_ICE + firm.list_technology_memory_EV

        if self.t_controller == self.ev_production_start_time:
            for firm in self.firm_manager.firms_list:
                firm.ev_production_bool = True
                
        #carbon price
        self.carbon_price = self.carbon_price_time_series[self.t_controller]

        #update_prices_and_emmisions
        self.gas_price = self.gas_price_california_vec[self.t_controller]
        self.electricity_price_subsidy = self.electricity_price_subsidy_time_series[self.t_controller]
        self.electricity_price = self.electricity_price_vec[self.t_controller] -  self.electricity_price_subsidy#ADJUST THE PRICE HERE HERE!

        self.electricity_emissions_intensity = self.electricity_emissions_intensity_vec[self.t_controller]
        self.rebate = self.rebate_time_series[self.t_controller]
        self.used_rebate = self.used_rebate_time_series[self.t_controller]
        self.discriminatory_corporate_tax = self.discriminatory_corporate_tax_time_series[self.t_controller]
        self.production_subsidy = self.production_subsidy_time_series[self.t_controller]
        self.research_subsidy = self.research_subsidy_time_series[self.t_controller]

        #HANDLE REBATE EXCLUSION, REMOVE ALL THE FIRMS FROM THE LIST, RESET THE COMULATIVE COST OF POLICY
        if self.t_controller == self.duration_burn_in + self.duration_no_carbon_price:
            for firm_id in self.social_network.firm_rebate_exclusion_set:
                self.social_network.remove_firm_rebate_exclusion_set(firm_id)

            self.social_network.policy_distortion = 0
            self.firm_manager.policy_distortion = 0
            for firm in self.firm_manager.firms_list:
                firm.policy_distortion = 0

    def update_firms(self):
        cars_on_sale_all_firms, U_sum_total = self.firm_manager.next_step(self.carbon_price, self.consider_ev_vec, self.new_bought_vehicles, self.gas_price, self.electricity_price, self.electricity_emissions_intensity, self.rebate, self.discriminatory_corporate_tax, self.production_subsidy, self.research_subsidy)
        return cars_on_sale_all_firms, U_sum_total
    
    def update_social_network(self):
        # Update social network based on firm preferences
        consider_ev_vec, new_bought_vehicles = self.social_network.next_step(self.carbon_price,  self.second_hand_cars, self.cars_on_sale_all_firms, self.gas_price, self.electricity_price, self.electricity_emissions_intensity, self.rebate, self.used_rebate, self.electricity_price_subsidy)

        return consider_ev_vec, new_bought_vehicles

    def get_second_hand_cars(self,U_sum_total):
        self.second_hand_merchant.next_step(self.gas_price, self.electricity_price, self.electricity_emissions_intensity, self.cars_on_sale_all_firms, self.carbon_price, U_sum_total)

        return self.second_hand_merchant.cars_on_sale

    ################################################################################################
    #POLICY OUTPUTS
    def calc_total_policy_distortion(self):
        """RUN ONCE AT THE END OF SIMULATION"""
        policy_distortion_firms = sum(firm.policy_distortion for firm in self.firm_manager.firms_list)
        policy_distortion = self.social_network.policy_distortion + self.firm_manager.policy_distortion + policy_distortion_firms
        return policy_distortion

    def calc_EV_prop(self):
        EV_stock_prop = sum(1 if car.transportType == 3 else 0 for car in self.social_network.current_vehicles)/self.social_network.num_individuals#NEED FOR OPTIMISATION, measures the uptake EVS
        return EV_stock_prop
    ################################################################################################

    def next_step(self):
        self.t_controller+=1#I DONT KNOW IF THIS SHOULD BE AT THE START OR THE END OF THE TIME STEP? But the code works if its at the end lol

        #print("TIME STEP", self.t_controller)

        self.update_time_series_data()
        self.cars_on_sale_all_firms, U_sum_total = self.update_firms()
        self.second_hand_cars = self.get_second_hand_cars(U_sum_total)

        self.consider_ev_vec, self.new_bought_vehicles = self.update_social_network()

        self.manage_saves()

    #################################################################################################

    def setup_continued_run_future(self, updated_parameters):
        """Allows future runs to be made using a single loadable controller for a given seed. Used for policy analysis"""

        self.parameters_controller = updated_parameters

        self.duration_future = self.parameters_controller["duration_future"]
        self.time_steps_max = self.parameters_controller["time_steps_max"]

        self.manage_scenario()
        self.manage_policies() 

        self.gas_price_california_vec = np.concatenate((self.pre_future_gas_price_california_vec, self.gas_price_series_future), axis=None) 
        self.electricity_price_vec =  np.concatenate((self.pre_future_electricity_price_vec, self.electricity_price_series_future ), axis=None) 
        self.electricity_emissions_intensity_vec = np.concatenate((self.pre_future_electricity_emissions_intensity_vec,self.grid_emissions_intensity_series_future ), axis=None) 
        self.rebate_time_series = np.concatenate((self.pre_future_rebate_time_series,self.rebate_time_series_future ), axis=None) 
        self.used_rebate_time_series = np.concatenate((self.pre_future_used_rebate_time_series,self.used_rebate_time_series_future ), axis=None) 

        self.discriminatory_corporate_tax_time_series =  np.concatenate(( np.zeros(self.duration_burn_in + self.duration_no_carbon_price), self.discriminatory_corporate_tax_time_series_future), axis=None) 
        self.electricity_price_subsidy_time_series = np.concatenate(( np.zeros(self.duration_burn_in + self.duration_no_carbon_price), self.electricity_price_subsidy_time_series_future), axis=None) 
        self.production_subsidy_time_series = np.concatenate(( np.zeros(self.duration_burn_in + self.duration_no_carbon_price), self.production_subsidy_time_series_future), axis=None) 
        self.research_subsidy_time_series = np.concatenate(( np.zeros(self.duration_burn_in + self.duration_no_carbon_price), self.research_subsidy_time_series_future), axis=None) 

        



        



