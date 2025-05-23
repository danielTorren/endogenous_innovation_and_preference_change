### controller.py
"""
Define controller that manages exchange of information between social network and firms.
"""

# Imports
from package.model.nkModel_ICE import NKModel_ICE 
from package.model.nkModel_EV import NKModel_EV 
from package.model.firmManager import Firm_Manager
from package.model.centralizedIdGenerator import IDGenerator
from package.model.secondHandMerchant import SecondHandMerchant
from package.model.socialNetworkUsers import Social_Network

import numpy as np
import itertools
from scipy.stats import lognorm
from copy import deepcopy

class Controller:
    """
    Controller class orchestrates interactions between agents in a vehicle market simulation
    involving social networks, firms, policy, and vehicle technologies (ICE and EV).
    """
    def __init__(self, parameters_controller):
        """
        Initialize the Controller object with all components and parameters.
        Sets up initial conditions, landscapes, firms, and social network users.

        Args:
            parameters_controller (dict): Dictionary of all model configuration parameters.
        """
        self.absolute_2035 = 144#number of months from the end of calibration period to 2035
        self.unpack_controller_parameters(parameters_controller)
        
        self.parameters_EV["delta"] = self.parameters_ICE["delta"] 
        self.parameters_EV["min_Quality"] = self.parameters_ICE["min_Quality"] 
        self.parameters_EV["max_Quality"] = self.parameters_ICE["max_Quality"]

        self.handle_seed()

        self.gen_time_series_calibration_scenarios_policies()
        self.gen_users_parameters()

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

        ####################################################################################################
        #BURN IN THE RESEARCH WITHOUT THE SOCIAL NETWORK WITH NO CARS
        self.consider_ev_vec = np.zeros(self.num_individuals).astype(np.int8)
        self.firm_manager.input_social_network_data(self.beta_vec, self.gamma_vec, self.consider_ev_vec, self.beta_bins, self.gamma_bins)
       
        #Need to calculate sum U give the consumption choices by individuals
        self.firm_manager.generate_market_data()
        for i in range(self.duration_burn_in_firms):
            self.firm_manager.next_step_burn_in()
        old_cars = self.firm_manager.gen_initial_cars()

        ####################################################################################################

        #NEED TO CREATE INIT OPTIONS
        self.cars_on_sale_all_firms = self.firm_manager.cars_on_sale_all_firms
        
        self.second_hand_cars = []#EMPTY INITIATILLY
        
        self.parameters_social_network["init_car_options"] =  self.cars_on_sale_all_firms 
        self.parameters_social_network["old_cars"] = old_cars
        
        #self.parameters_social_network["init_vehicle_options"] = self.mix_in_vehicles()
        self.gen_social_network()#users have chosen a vehicle

        #NEED THE LIST OF VEHICLES CHOSEN to record data
        self.new_bought_vehicles = self.social_network.current_vehicles

        #UPDATE SECOND HADN MERCHANT WITH THE DATA
        self.second_hand_merchant.calc_median(self.social_network.beta_vec, self.social_network.gamma_vec)        

        #pass information across one time
        #self.firm_manager.input_social_network_data(self.social_network.beta_vec, self.social_network.gamma_vec, self.social_network.consider_ev_vec, self.beta_bins)


        if self.save_timeseries_data_state:
            self.social_network.set_up_time_series_social_network()
            self.firm_manager.set_up_time_series_firm_manager()
            self.time_series = []
            self.set_up_time_series_controller()
        
    def unpack_controller_parameters(self,parameters_controller):
        """
        Unpack and store all input parameters from the controller configuration.

        Args:
            parameters_controller (dict): Configuration parameters for model components.
        """
        #CONTROLLER PARAMETERS:
        self.parameters_controller = parameters_controller#save copy in the object for ease of access

        self.parameters_social_network = parameters_controller["parameters_social_network"]
        self.parameters_vehicle_user = parameters_controller["parameters_vehicle_user"]
        self.parameters_firm_manager = parameters_controller["parameters_firm_manager"]
        self.parameters_firm = parameters_controller["parameters_firm"]
        self.parameters_ICE = parameters_controller["parameters_ICE"]
        self.parameters_EV = parameters_controller["parameters_EV"]
        self.parameters_second_hand = parameters_controller["parameters_second_hand"]
        self.parameters_calibration_data = self.parameters_controller["calibration_data"]
        
        self.parameters_rebate_calibration = self.parameters_controller["parameters_rebate_calibration"]        

        self.t_controller = 0
        self.save_timeseries_data_state = parameters_controller["save_timeseries_data_state"]
        self.compression_factor_state = parameters_controller["compression_factor_state"]
        
        #TIME STUFF
        self.duration_burn_in = parameters_controller["duration_burn_in"] 
        self.duration_calibration = parameters_controller["duration_calibration"] 
        self.duration_future = parameters_controller["duration_future"] 
        self.duration_burn_in_firms = parameters_controller["duration_burn_in_research"] 

        self.t_2030 = int(self.duration_burn_in + self.duration_calibration + 6*12)#1/1/20230, only used to measure a medium run utility in 2030

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

    def handle_seed(self):
        """
        Initialize random seeds for various components using controller parameters.
        Ensures reproducibility of random processes in model.
        """
        #Seed for inputs
        self.random_state_inputs = np.random.RandomState(self.parameters_controller["seed_inputs"])
        self.parameters_social_network["seed_inputs"] = self.parameters_controller["seed_inputs"]
        self.parameters_ICE["random_state_inputs"] = self.random_state_inputs
        self.parameters_EV["random_state_inputs"] = self.random_state_inputs
        self.parameters_firm_manager["random_state_input"] = self.random_state_inputs

        #Variable stuff
        self.random_state = np.random.RandomState(self.parameters_controller["seed"])
        self.parameters_social_network["random_state"] = self.random_state
        self.parameters_firm["random_state"] = self.random_state
        self.parameters_second_hand["random_state"] = self.random_state

    def gen_users_parameters(self):
        """
        Generate user-specific parameters for social network individuals:
        - Driving distances
        - Innovativeness (chi)
        - Emissions willingness to pay (gamma)
        - Vehicle range preference (nu)
        - Quality sensitivity (beta)
        """
        self.num_individuals = self.parameters_social_network["num_individuals"]
        self.ev_adoption_state_vec = np.zeros(self.num_individuals)

        #########################################
        #GENERATING DISTANCES
        self.gen_distance()

        ########################################################################
        # CHI
        self.gen_chi()

        ####################################################################################################################################
        #GAMMA
        self.gen_gamma()

        ####################################################################################################################################
        #NU  
        self.gen_nu()
        
        ####################################################################################################################################
        #BETA
        self.gen_beta()

        self.parameters_social_network["beta_vec"] = self.beta_vec 
        self.parameters_social_network["gamma_vec"] = self.gamma_vec 
        self.parameters_social_network["chi_vec"] = self.chi_vec 
        self.parameters_social_network["nu_vec"] = self.nu_vec 
        self.parameters_social_network["d_vec"] = self.d_vec

        self.beta_median = np.median(self.beta_vec)
        self.gamma_median = np.median(self.gamma_vec)
        self.nu_median = np.median(self.nu_vec)

        self.parameters_social_network["beta_median"] = self.beta_median
        self.parameters_social_network["gamma_median"] = self.gamma_median 
        self.parameters_social_network["nu_median"] = self.nu_median

        #create the beta and gamma vectors:
        beta_values = []
        gamma_values = []

        segment_codes = list(itertools.product(range(self.num_beta_segments), range(self.num_gamma_segments), range(2)))
        self.num_segments = len(segment_codes) 

        for code in segment_codes:
            beta_idx, gamma_idx, _ = code  # Unpack the segment code
            beta_values.append(self.beta_s[beta_idx])
            gamma_values.append(self.gamma_s[gamma_idx])

        self.beta_segment_vals = np.array(beta_values)
        self.gamma_segment_vals = np.array(gamma_values)


    def gen_distance(self):
        """
        Generate driving distance vector (d_vec) using a fitted Poisson distribution.
        Derived from empirical California vehicle travel data.
        """
                
        #data from https://www.energy.ca.gov/data-reports/surveys/california-vehicle-survey/vehicle-miles-traveled-fuel-type
        bin_centers = np.array([
            335.2791667,
            1005.8375,
            1676.331279,
            2346.892946,
            3017.452946,
            5028.99
        ])
        fitted_lambda = 1.5428809450394916
        #Fitted_lambda_10_plus_year_old_cars =  0.2345307913567248
        #Monthly_depreciation_rate_distance =  0.010411 #THIS ASSUMES CARS OF 15 years for 10+years with max distance driven as 50% more than the second larger upper bin limit.
        # Step 4: Generate N random distances using the fitted Poisson parameter
        N = self.num_individuals # Number of individuals
        poisson_samples = self.random_state_inputs.poisson(lam=fitted_lambda, size=N)
        # Step 5: Map the Poisson samples back to the distance range of the original data
        min_bin, max_bin = bin_centers[0], bin_centers[-1]
        scale_factor = (max_bin - min_bin) / (len(bin_centers) - 1)
        self.d_vec = poisson_samples * scale_factor + min_bin
        self.parameters_social_network["d_vec"] = self.d_vec

    def gen_chi(self):
        """
        Generate chi_vec representing innovativeness of users using a beta distribution.
        Injects proportion of zeros to simulate innovative users. 
        """
        self.a_chi = self.parameters_social_network["a_chi"]
        self.b_chi = self.parameters_social_network["b_chi"]
        self.chi_max = self.parameters_social_network["chi_max"]
        self.proportion_zero_target = self.parameters_social_network["proportion_zero_target"]  # Define your target proportion here

        # Step 1: Generate continuous Beta distribution
        innovativeness_vec_continuous = self.random_state_inputs.beta(self.a_chi, self.b_chi, size=self.num_individuals)

        # Step 2: Introduce zeros based on target proportion
        num_zeros = int(self.proportion_zero_target * self.num_individuals)
        zero_indices = self.random_state_inputs.choice(self.num_individuals, size=num_zeros, replace=False)
        innovativeness_vec_continuous[zero_indices] = 0

        # Step 3: Scale to chi_max without rounding
        self.chi_vec = innovativeness_vec_continuous * self.chi_max

        # Check the actual proportion of zeros
        self.proportion_zero_chi = np.mean(self.chi_vec == 0)
        self.parameters_social_network["chi_vec"] = self.chi_vec 

    def gen_nu(self):
        """
        Generate constant nu vector representing vehicle range preference.
        """
        self.nu_vec = np.asarray([self.parameters_social_network["nu"]] * self.num_individuals)
        self.nu_median = np.median(self.nu_vec)
        self.parameters_social_network["nu_median"] = self.nu_median
        self.parameters_social_network["nu_vec"] = self.nu_vec 

    def gen_gamma(self):
        """
        Compute gamma values (willingness to pay per emissions reduced) for individuals.
        Derived from normal distribution of WTP_E scaled by distance.
        Used to create segmentation bins.
        """
        r = self.parameters_vehicle_user["r"]
        delta = self.parameters_ICE["delta"]
        if (r <= delta/(1-delta)) or (r <= self.parameters_EV["delta"]/(1-self.parameters_EV["delta"])):
            print("r and delta: r, delta/1-delta",r, delta,delta/(1-delta))
            raise Exception("r <= delta/(1-delta)), raise r or lower delta")
        
        self.WTP_E_mean = self.parameters_social_network["WTP_E_mean"]
        self.WTP_E_sd = self.parameters_social_network["WTP_E_sd"]     
        WTP_E_vec_unclipped = self.random_state_inputs.normal(loc = self.WTP_E_mean, scale = self.WTP_E_sd, size = self.num_individuals)
        self.WTP_E_vec = np.clip(WTP_E_vec_unclipped, a_min = self.parameters_social_network["gamma_epsilon"], a_max = np.inf)     
        self.gamma_vec = (self.WTP_E_vec/self.d_vec)*((r - delta - r*delta)/((1+r)*(1-delta)))

        self.num_gamma_segments = self.parameters_firm_manager["num_gamma_segments"]
        # Calculate the bin edges using quantiles
        quants_gamma = np.linspace(0,1,self.num_gamma_segments + 1)
        self.gamma_bins =  np.quantile(self.gamma_vec, quants_gamma)

        # Step 1: Define the percentiles for the midpoints
        percentiles_gamma = np.linspace(1 / (2 * self.num_gamma_segments), 1 - 1 / (2 * self.num_gamma_segments), self.num_gamma_segments)
        self.gamma_s = np.quantile(self.gamma_vec, percentiles_gamma)

    def gen_beta(self):
        """
        Generate beta_vec reflecting consumer quality sensitivity based on income.
        Inverse relationship to income distribution drawn from a log-normal.
        """
        #BETA
        median_beta = self.calc_beta_median()
        #GIVEN THAT YOU DO MEDIAN INCOME/ INCOME, DONT NEED TO SCALE INCOME
        incomes = lognorm.rvs(s=self.parameters_social_network["income_sigma"], scale=np.exp(self.parameters_social_network["income_mu"]), size=self.num_individuals, random_state=self.random_state_inputs)
        median_income = np.median(incomes)
        self.beta_vec = median_beta*(median_income/incomes)
        self.random_state_inputs.shuffle(self.beta_vec)# Shuffle to randomize the order of agents

        self.num_beta_segments = self.parameters_firm_manager["num_beta_segments"]
        # Calculate the bin edges using quantiles
        quants = np.linspace(0,1,self.num_beta_segments + 1)
        self.beta_bins =  np.quantile(self.beta_vec, quants)

        # Step 1: Define the percentiles for the midpoints
        percentiles = np.linspace(1 / (2 * self.num_beta_segments), 1 - 1 / (2 * self.num_beta_segments), self.num_beta_segments)
        self.beta_s = np.quantile(self.beta_vec, percentiles)

    def generate_beta_values_quintiles(self,n, quintile_incomes, median_beta):
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

        median_income = quintile_incomes[2]

        beta_vals = [median_beta*median_income/income for income in quintile_incomes]
        
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
        self.random_state_inputs.shuffle(beta_list)
        
        return np.asarray(beta_list)

    def calc_beta_median(self):
        """
        Calculate the median beta value.
        """
        # Extract parameters
        E = self.parameters_ICE["production_emissions"]
        delta = self.parameters_ICE["delta"]
        r = self.parameters_vehicle_user["r"]
        kappa = self.parameters_vehicle_user["kappa"]
        alpha = self.parameters_vehicle_user["alpha"]
        zeta = self.parameters_vehicle_user["zeta"]
        
        # Calculate average gasoline cost and emissions
        c = np.mean(self.calibration_gas_price_california_vec)
        e = self.parameters_calibration_data["gasoline_Kgco2_per_Kilowatt_Hour"]
        
        # Calculate mean efficiency
        omega_mean = (self.parameters_ICE["min_Efficiency"] + self.parameters_ICE["max_Efficiency"]) / 2
        # Calculate maximum cost
        C_mean = (self.parameters_ICE["max_Cost"] + self.parameters_ICE["min_Cost"])/2
        # Set other parameters
        omega = omega_mean
        gamma = np.median(self.gamma_vec)
        nu = np.median(self.nu_vec)
        P = self.parameters_ICE["mean_Price"]
        W = self.parameters_vehicle_user["W_calibration"]
        D = np.median(self.d_vec)
        
        # Calculate Q_mt (assuming Q_mt is given or calculated elsewhere)
        Q_mt = (self.parameters_ICE["min_Quality"] + self.parameters_ICE["max_Quality"])/2
        B = self.parameters_ICE["fuel_tank"]

        # Calculate the components of the equation
        term1 = np.log(W * (kappa * (P - C_mean) - 1))* (1 / kappa)
        term2 = P + gamma * E
        term3 = -(nu*(B*omega)**zeta)
        term4 = D * ((1 + r) * (1 - delta) * (c + gamma * e)) / (omega * (r - delta - r * delta))
        beta_s = (1/(Q_mt**alpha))*(term1 + term2 + term3 + term4)

        return beta_s

    #####################################################################################################################################
    def manage_burn_in(self):
        """
        Create static time series inputs for burn-in period of model.
        Gas prices, electricity, emissions, and rebate data are held constant.
        """
        self.burn_in_gas_price_vec = np.asarray([self.calibration_gas_price_california_vec[0]]*self.duration_burn_in)
        self.burn_in_electricity_price_vec = np.asarray([self.calibration_electricity_price_vec[0]]*self.duration_burn_in)
        self.burn_in_electricity_emissions_intensity_vec = np.asarray([self.calibration_electricity_emissions_intensity_vec[0]]*self.duration_burn_in)

        self.burn_in_rebate_time_series = np.zeros(self.duration_burn_in)
        self.burn_in_used_rebate_time_series = np.zeros(self.duration_burn_in)

    def manage_calibration(self):
        """
        Generate calibration period rebate and emissions parameters from config.
        Handles EV rebate policy setup.
        """
        self.gas_emissions_intensity = self.parameters_calibration_data["gasoline_Kgco2_per_Kilowatt_Hour"]
        
        self.calibration_rebate_time_series = np.zeros(self.duration_calibration + self.duration_future )
        self.calibration_used_rebate_time_series = np.zeros(self.duration_calibration + self.duration_future)
        
        if self.parameters_controller["EV_rebate_state"]:#CONTROLLS WHETHER OR NOT THE REBATE POLICY APPLIED DURING FUTURE
            if self.duration_future > self.absolute_2035:
                self.calibration_rebate_time_series[self.parameters_rebate_calibration["start_time"]:self.duration_calibration + self.absolute_2035] = self.parameters_rebate_calibration["rebate"]
                self.calibration_used_rebate_time_series[self.parameters_rebate_calibration["start_time"]:self.duration_calibration + self.absolute_2035] = self.parameters_rebate_calibration["used_rebate"]
            else:
                self.calibration_rebate_time_series[self.parameters_rebate_calibration["start_time"]:] = self.parameters_rebate_calibration["rebate"]
                self.calibration_used_rebate_time_series[self.parameters_rebate_calibration["start_time"]:] = self.parameters_rebate_calibration["used_rebate"]
        else:
            self.calibration_rebate_time_series[self.parameters_rebate_calibration["start_time"]:self.duration_calibration] = self.parameters_rebate_calibration["rebate"]
            self.calibration_used_rebate_time_series[self.parameters_rebate_calibration["start_time"]:self.duration_calibration] = self.parameters_rebate_calibration["used_rebate"]

    def manage_scenario(self):
        """
        Generate future scenario projections for gas prices, electricity, and emissions.
        Based on linear extrapolations from 2023 values and future scenario multipliers.
        """
        self.Gas_price_2023 = self.parameters_calibration_data["Gas_price_2023"]
        self.Gas_price_future = self.Gas_price_2023*self.parameters_controller["parameters_scenarios"]["Gas_price"]
        if self.duration_future > self.absolute_2035:
            gas_price_series_future_im = np.linspace(self.Gas_price_2023, self.Gas_price_future, self.absolute_2035)
            self.gas_price_series_future = np.concatenate((gas_price_series_future_im, np.asarray([gas_price_series_future_im[-1]]*(self.duration_future - self.absolute_2035))), axis=None)
        else:
            self.gas_price_series_future = np.linspace(self.Gas_price_2023, self.Gas_price_future, self.duration_future)

        self.Electricity_price_2023 = self.parameters_calibration_data["Electricity_price_2023"]
        self.Electricity_price_future = self.Electricity_price_2023*self.parameters_controller["parameters_scenarios"]["Electricity_price"]
        if self.duration_future > self.absolute_2035:
            electricity_price_series_future_im = np.linspace(self.Electricity_price_2023, self.Electricity_price_future, self.absolute_2035)
            self.electricity_price_series_future = np.concatenate((electricity_price_series_future_im, np.asarray([electricity_price_series_future_im[-1]]*(self.duration_future - self.absolute_2035))), axis=None)
        else:
            self.electricity_price_series_future = np.linspace(self.Electricity_price_2023, self.Electricity_price_future, self.duration_future)
        
        self.Grid_emissions_intensity_2023 = self.parameters_calibration_data["Electricity_emissions_intensity_2023"]
        self.Grid_emissions_intensity_future = self.Grid_emissions_intensity_2023*self.parameters_controller["parameters_scenarios"]["Grid_emissions_intensity"]
        if self.duration_future > self.absolute_2035:
            grid_emissions_intensity_series_future_im = np.linspace(self.Grid_emissions_intensity_2023, self.Grid_emissions_intensity_future, self.absolute_2035)
            self.grid_emissions_intensity_series_future = np.concatenate((grid_emissions_intensity_series_future_im, np.asarray([grid_emissions_intensity_series_future_im[-1]]*(self.duration_future - self.absolute_2035))), axis=None)
        else:
            self.grid_emissions_intensity_series_future = np.linspace(self.Grid_emissions_intensity_2023, self.Grid_emissions_intensity_future, self.duration_future)

    def manage_policies(self):
        """
        Generate policy time series data such as carbon prices, subsidies, rebates.
        Applies configured policies from the controller settings.
        """
        self.Carbon_price_state = self.parameters_controller["parameters_policies"]["States"]["Carbon_price"]
        
        self.Electricity_subsidy_state =  self.parameters_controller["parameters_policies"]["States"]["Electricity_subsidy"]
        self.Adoption_subsidy_state =  self.parameters_controller["parameters_policies"]["States"]["Adoption_subsidy"]
        self.Adoption_subsidy_used_state =  self.parameters_controller["parameters_policies"]["States"]["Adoption_subsidy_used"]
        self.Production_subsidy_state =  self.parameters_controller["parameters_policies"]["States"]["Production_subsidy"]


        # Carbon price calculation
        if self.Carbon_price_state:
            self.future_carbon_price_state = self.parameters_controller["parameters_policies"]["Values"]["Carbon_price"]["Carbon_price_state"]
            self.future_carbon_price_init = self.parameters_controller["parameters_policies"]["Values"]["Carbon_price"]["Carbon_price_init"]
            self.future_carbon_price_policy = self.parameters_controller["parameters_policies"]["Values"]["Carbon_price"]["Carbon_price"]
        else:
            self.future_carbon_price_state = 0
            self.future_carbon_price_init = 0
            self.future_carbon_price_policy = 0


        #DEAL WITH CARBON PRICE
        self.carbon_price_time_series = self.calculate_carbon_price_time_series()


        # Electricity_subsidy calculation
        if self.Electricity_subsidy_state:
            self.Electricity_subsidy = self.parameters_controller["parameters_policies"]["Values"]["Electricity_subsidy"]
        else:
            self.Electricity_subsidy = 0
        if self.duration_future > self.absolute_2035:
            electricity_price_subsidy_time_series_future = np.asarray([self.Electricity_subsidy]*self.absolute_2035)
            self.electricity_price_subsidy_time_series_future = np.concatenate((electricity_price_subsidy_time_series_future, np.asarray([0]*(self.duration_future - self.absolute_2035))), axis=None)
        else:
            self.electricity_price_subsidy_time_series_future = np.asarray([self.Electricity_subsidy]*self.duration_future)

        # Adoption subsidy calculation
        if self.Adoption_subsidy_state:
            self.Adoption_subsidy = self.parameters_controller["parameters_policies"]["Values"]["Adoption_subsidy"]
        else:
            self.Adoption_subsidy = 0
        
        if self.duration_future > self.absolute_2035:
            rebate_time_series_future =  np.asarray([self.Adoption_subsidy]*self.absolute_2035)
            self.rebate_time_series_future = np.concatenate((rebate_time_series_future, np.asarray([0]*(self.duration_future - self.absolute_2035))), axis=None)
        else:
            self.rebate_time_series_future = np.asarray([self.Adoption_subsidy]*self.duration_future)

        #USED 
        if self.Adoption_subsidy_used_state:
            self.Used_adoption_subsidy = self.parameters_controller["parameters_policies"]["Values"]["Adoption_subsidy_used"]
        else:
            self.Used_adoption_subsidy = 0
        if self.duration_future > self.absolute_2035:
            used_rebate_time_series_future =  np.asarray([self.Used_adoption_subsidy]*self.absolute_2035)
            self.used_rebate_time_series_future = np.concatenate((used_rebate_time_series_future, np.asarray([0]*(self.duration_future - self.absolute_2035))), axis=None)
        else:
            self.used_rebate_time_series_future = np.asarray([self.Used_adoption_subsidy]*self.duration_future)
        
        # Production_subsidy calculation
        if self.Production_subsidy_state:
            self.Production_subsidy = self.parameters_controller["parameters_policies"]["Values"]["Production_subsidy"]
        else:
            self.Production_subsidy = 0
        if self.duration_future > self.absolute_2035:
            production_subsidy_time_series_future =  np.asarray([self.Production_subsidy]*self.absolute_2035)
            self.production_subsidy_time_series_future = np.concatenate((production_subsidy_time_series_future, np.asarray([0]*(self.duration_future - self.absolute_2035))), axis=None)
        else:
            self.production_subsidy_time_series_future = np.asarray([self.Production_subsidy]*self.duration_future)

    def calculate_carbon_price_time_series(self):
        """
        Generate a time series for carbon price based on user-defined growth type.

        Returns:
            list: Carbon price values for each time step.
        """
        time_series = np.arange(self.time_steps_max + 1)
        carbon_price_series = []

        for t in time_series:
            carbon_price = self.calculate_price_at_time(t)
            carbon_price_series.append(carbon_price)
        return carbon_price_series

    def calculate_price_at_time(self, t):
        """
        Calculate carbon price at specific time t using policy schedule.

        Args:
            t (int): Time step.

        Returns:
            float: Carbon price at time t.
        """
        if self.future_carbon_price_policy > 0 and self.duration_future > 0:
            if t > (self.duration_burn_in + self.duration_calibration + self.absolute_2035):
                return 0#self.future_carbon_price_policy
            elif t >= (self.duration_burn_in + self.duration_calibration):
                relative_time = t - (self.duration_burn_in  + self.duration_calibration)
                return self.calculate_growth(
                    relative_time, 
                    self.absolute_2035,
                    self.future_carbon_price_init,
                    self.future_carbon_price_policy,
                    self.future_carbon_price_state
                )
            else:
                return 0
        else:
            return 0

    def calculate_growth(self, t, total_duration, start_price, end_price, growth_type):
        """
        Growth function for carbon pricing policy over time.

        Args:
            t (int): Time step.
            total_duration (int): Duration for full policy growth.
            start_price (float): Initial price.
            end_price (float): Final price.
            growth_type (str): Type of growth (flat, linear, quadratic, exponential).

        Returns:
            float: Price at time t.
        """
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

    def gen_time_series_calibration_scenarios_policies(self):
        """
        Combine time series data for calibration, scenarios, and policy regimes.
        Outputs vectors for pricing, emissions, rebates across all simulation phases.
        """
        
        self.calibration_gas_price_california_vec = self.parameters_calibration_data["gas_price_california_vec"]
        self.calibration_electricity_price_vec = self.parameters_calibration_data["electricity_price_vec"]
        self.calibration_electricity_emissions_intensity_vec = self.parameters_calibration_data["electricity_emissions_intensity_vec"]

        self.manage_burn_in()
        self.manage_calibration()

        #JOIN BURN IN AND CALIBRATION
        self.pre_future_gas_price_california_vec = np.concatenate((self.burn_in_gas_price_vec,self.calibration_gas_price_california_vec), axis=None) 
        self.pre_future_electricity_price_vec =  np.concatenate((self.burn_in_electricity_price_vec,self.calibration_electricity_price_vec), axis=None) 
        self.pre_future_electricity_emissions_intensity_vec = np.concatenate((self.burn_in_electricity_emissions_intensity_vec,self.calibration_electricity_emissions_intensity_vec), axis=None) 
        
        #THIS IS THE REBATE ASSOCIATED WITH THE BACKED IN POLICY
        self.rebate_calibration_time_series = np.concatenate((self.burn_in_rebate_time_series, self.calibration_rebate_time_series), axis=None) #THIS IS BOTH BURN IN CALIBRATION AND FUTURE
        self.used_rebate_calibration_time_series = np.concatenate((self.burn_in_used_rebate_time_series, self.calibration_used_rebate_time_series), axis=None) 

        if self.full_run_state:
            self.manage_scenario()
            self.manage_policies() 

            #NOW STAPLE THE STUFF TOGETHER TO GET ONE THING
            #CALIRBATION TIME_STEPS
            self.gas_price_california_vec = np.concatenate((self.pre_future_gas_price_california_vec, self.gas_price_series_future), axis=None) 
            self.electricity_price_vec =  np.concatenate((self.pre_future_electricity_price_vec, self.electricity_price_series_future ), axis=None) 
            self.electricity_emissions_intensity_vec = np.concatenate((self.pre_future_electricity_emissions_intensity_vec,self.grid_emissions_intensity_series_future ), axis=None) 
            
            self.rebate_time_series = np.concatenate(( np.zeros(self.duration_burn_in + self.duration_calibration), self.rebate_time_series_future), axis=None) 
            self.used_rebate_time_series = np.concatenate(( np.zeros(self.duration_burn_in + self.duration_calibration), self.used_rebate_time_series_future), axis=None) 

            self.electricity_price_subsidy_time_series = np.concatenate(( np.zeros(self.duration_burn_in + self.duration_calibration), self.electricity_price_subsidy_time_series_future), axis=None) 
            self.production_subsidy_time_series = np.concatenate(( np.zeros(self.duration_burn_in + self.duration_calibration), self.production_subsidy_time_series_future), axis=None) 
        else:
            self.gas_price_california_vec = self.pre_future_gas_price_california_vec 
            self.electricity_price_vec = self.pre_future_electricity_price_vec 
            self.electricity_emissions_intensity_vec = self.pre_future_electricity_emissions_intensity_vec
            
            self.rebate_time_series =  np.zeros(self.duration_burn_in + self.duration_calibration)
            self.used_rebate_time_series = np.zeros(self.duration_burn_in + self.duration_calibration)

            self.carbon_price_time_series = np.zeros(self.duration_burn_in + self.duration_calibration)

            self.electricity_price_subsidy_time_series = np.zeros(self.duration_burn_in + self.duration_calibration)
            self.production_subsidy_time_series = np.zeros(self.duration_burn_in + self.duration_calibration)

    def setup_id_gen(self):
        """
        Initialize ID generator for firms to ensure unique IDs across agents(individuals, firms, cars and technologies).
        """
        self.IDGenerator_firms = IDGenerator()# CREATE ID GENERATOR FOR FIRMS

    def setup_firm_manager_parameters(self):
        """
        Configure firm manager parameters with shared and calculated model values.
        """
        #TRANSFERING COMMON INFORMATION
        #FIRM MANAGER
        self.parameters_firm_manager["num_individuals"] = self.parameters_social_network["num_individuals"]
        self.parameters_firm_manager["carbon_price"] = self.carbon_price
        self.parameters_firm_manager["IDGenerator_firms"] = self.IDGenerator_firms
        self.parameters_firm_manager["kappa"] = self.parameters_vehicle_user["kappa"]
        self.parameters_firm_manager["N"] = self.parameters_ICE["N"]
        self.parameters_firm_manager["nu"] = self.nu_median
        self.parameters_firm_manager["min_W"] = self.parameters_vehicle_user["min_W"]
        self.parameters_firm_manager["zeta"] = self.parameters_vehicle_user["zeta"]

    def setup_firm_parameters(self):
        """
        Prepare firm configuration including landscapes, pricing, and emission settings.
        """
        self.parameters_firm["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_firm["compression_factor_state"] = self.compression_factor_state
        self.parameters_firm["IDGenerator_firms"] = self.IDGenerator_firms
        self.parameters_firm["kappa"] = self.parameters_vehicle_user["kappa"]

        self.parameters_firm["ICE_landscape"] = self.ICE_landscape
        self.parameters_firm["EV_landscape"] = self.EV_landscape
        self.parameters_firm["r"] = self.parameters_vehicle_user["r"]
        #self.parameters_firm["delta"] = self.parameters_ICE["delta"]#ASSUME THAT BOTH ICE AND EV HAVE SAME DEPRECIATIONS RATE
        self.parameters_firm["carbon_price"] = self.carbon_price
        self.parameters_firm["gas_price"] = self.gas_price
        self.parameters_firm["electricity_price"] = self.electricity_price
        self.parameters_firm["electricity_emissions_intensity"] = self.electricity_emissions_intensity
        self.parameters_firm["rebate"] = self.rebate 
        self.parameters_firm["rebate_calibration"] = self.rebate_calibration
        self.parameters_firm["d_mean"] = np.mean(self.d_vec)
        self.parameters_firm["U_segments_init"] = self.parameters_vehicle_user["U_segments_init"]
        self.parameters_firm["nu"] = self.nu_median
        self.parameters_firm["alpha"] = self.parameters_vehicle_user["alpha"]
        self.parameters_firm["zeta"] = self.parameters_vehicle_user["zeta"]
        self.parameters_firm["beta_segment_vals"] = self.beta_segment_vals 
        self.parameters_firm["gamma_segment_vals"] = self.gamma_segment_vals 

        if self.t_controller == self.ev_research_start_time:
            self.parameters_firm["ev_research_bool"] = True
        else:
            self.parameters_firm["ev_research_bool"] = False

        if self.t_controller == self.ev_production_start_time:
            self.parameters_firm["ev_production_bool"] = True
        else:
            self.parameters_firm["ev_production_bool"] = False

    def setup_social_network_parameters(self):
        """
        Set up social network agent parameters, including policies, emissions, and utilities.
        """
        self.parameters_social_network["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_social_network["compression_factor_state"] = self.compression_factor_state
        self.parameters_social_network["policy_start_time"] = self.duration_calibration
        self.parameters_social_network["carbon_price"] = self.carbon_price
        self.parameters_social_network["IDGenerator_firms"] = self.IDGenerator_firms
        self.parameters_social_network["second_hand_merchant"] = self.second_hand_merchant
        self.parameters_social_network["gas_price"] = self.gas_price
        self.parameters_social_network["electricity_price"] = self.electricity_price
        self.parameters_social_network["electricity_emissions_intensity"] = self.electricity_emissions_intensity
        
        self.parameters_social_network["rebate"] = self.rebate 
        self.parameters_social_network["used_rebate"] = self.used_rebate
        self.parameters_social_network["rebate_calibration"] = self.rebate_calibration 
        self.parameters_social_network["used_rebate_calibration"] = self.used_rebate_calibration 

        self.parameters_social_network["beta_segment_vals"] = self.beta_segment_vals 
        self.parameters_social_network["gamma_segment_vals"] = self.gamma_segment_vals 
        self.parameters_social_network["scrap_price"] = self.parameters_second_hand["scrap_price"]
        self.parameters_social_network["alpha"] = self.parameters_vehicle_user["alpha"]
        self.parameters_social_network["zeta"] = self.parameters_vehicle_user["zeta"]


    def setup_vehicle_users_parameters(self):
        """
        Configure basic user-specific parameters for social network agents.
        """
        self.parameters_vehicle_user["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_vehicle_user["compression_factor_state"] = self.compression_factor_state

    def setup_ICE_landscape(self, parameters_ICE):   
        """
        Initialize ICE technology landscape model from NK landscape model.

        Args:
            parameters_ICE (dict): ICE-related model parameters.
        """ 
        parameters_ICE["init_price_multiplier"] = self.parameters_firm["init_price_multiplier"]
        parameters_ICE["r"] = self.parameters_vehicle_user["r"]
        parameters_ICE["median_beta"] = self.beta_median
        parameters_ICE["median_gamma"] = self.gamma_median
        parameters_ICE["median_nu"] = self.nu_median
        parameters_ICE["fuel_cost_c"] = self.parameters_calibration_data["gas_price_california_vec"][0]
        parameters_ICE["e_t"] = self.gas_emissions_intensity
        parameters_ICE["d_mean"] = np.mean(self.d_vec)
        parameters_ICE["alpha"] = self.parameters_vehicle_user["alpha"]
        parameters_ICE["zeta"] = self.parameters_vehicle_user["zeta"]
        self.ICE_landscape = NKModel_ICE(parameters_ICE)

    def setup_EV_landscape(self, parameters_EV):
        """
        Initialize EV technology landscape model from NK landscape model.

        Args:
            parameters_EV (dict): EV-related model parameters.
        """
        parameters_EV["init_price_multiplier"] = self.parameters_firm["init_price_multiplier"]
        parameters_EV["r"] = self.parameters_vehicle_user["r"]
        parameters_EV["median_beta"] = self.beta_median 
        parameters_EV["median_gamma"] = self.gamma_median
        parameters_EV["median_nu"] = self.nu_median
        parameters_EV["fuel_cost_c"] = self.parameters_calibration_data["electricity_price_vec"][self.parameters_controller["ev_production_start_time"]]
        parameters_EV["e_t"] = self.parameters_calibration_data["electricity_emissions_intensity_vec"][self.parameters_controller["ev_production_start_time"]]
        parameters_EV["d_mean"] = np.mean(self.d_vec)
        parameters_EV["alpha"] = self.parameters_vehicle_user["alpha"]
        parameters_EV["zeta"] = self.parameters_vehicle_user["zeta"]
        self.parameters_EV["min_Quality"] = self.parameters_ICE["min_Quality"]
        self.parameters_EV["max_Quality"] = self.parameters_ICE["max_Quality"]
        self.EV_landscape = NKModel_EV(parameters_EV)


    def setup_second_hand_market(self):
        """
        Configure and initialize the second-hand car market simulation object.
        """
        self.parameters_second_hand["r"] = self.parameters_vehicle_user["r"]
        self.parameters_second_hand["kappa"] = self.parameters_vehicle_user["kappa"]

        self.parameters_second_hand["beta_segment_vals"] = self.beta_segment_vals 
        self.parameters_second_hand["gamma_segment_vals"] = self.gamma_segment_vals 
        self.parameters_second_hand["max_num_cars"] = int(np.round(self.parameters_social_network["num_individuals"] * self.parameters_second_hand["max_num_cars_prop"]))  

        self.second_hand_merchant = SecondHandMerchant(unique_id = -3, parameters_second_hand= self.parameters_second_hand)
    
    def gen_firms(self):
        """
        Instantiate firm manager and all firm agents for the simulation.
        """
        self.firm_manager = Firm_Manager(self.parameters_firm_manager, self.parameters_firm, self.parameters_ICE, self.parameters_EV, self.ICE_landscape, self.EV_landscape)
    
    def gen_social_network(self):
        """
        Instantiate social network object and create initial vehicle choices for users.
        """
        self.social_network = Social_Network(self.parameters_social_network, self.parameters_vehicle_user)


    def set_up_time_series_controller(self):
        """
        Initialize data structures for recording controller-level time series history.
        """
        self.history_gas_price = []
        self.history_electricity_price = []
        self.history_electricity_emissions_intensity = []
        self.history_rebate = []
        self.history_used_rebate = []
        self.history_policy_net_cost = []

    def save_timeseries_controller(self):
        """
        Save current time step's values to time series for later analysis.
        """
        self.history_gas_price.append(self.gas_price)
        self.history_electricity_price.append(self.electricity_price)
        self.history_electricity_emissions_intensity.append(self.electricity_emissions_intensity)
        self.history_rebate.append(self.rebate)
        self.history_used_rebate.append(self.used_rebate)
        self.history_policy_net_cost.append(self.calc_net_policy_distortion())

    def manage_saves(self):
        """
        Trigger save operations for all model components if conditions met.
        Applies compression factor to reduce frequency.
        """
        if self.save_timeseries_data_state and (self.t_controller % self.compression_factor_state == 0):
            self.social_network.save_timeseries_data_social_network()
            self.firm_manager.save_timeseries_data_firm_manager()
            self.second_hand_merchant.save_timeseries_second_hand_merchant()
            self.time_series.append(self.t_controller)
            self.save_timeseries_controller()

##########################################################################################################################################

    def update_time_series_data(self):
        """
        Refresh controller variables each time step based on scenario and policy vectors.
        Includes prices, emissions, subsidies, and rebates.
        """
                
        #EV research state
        if self.t_controller == self.ev_research_start_time:#turn on ev research
            for firm in self.firm_manager.firms_list:
                firm.ev_research_bool = True
                firm.list_technology_memory = firm.list_technology_memory_ICE + firm.list_technology_memory_EV

        if self.t_controller == self.ev_production_start_time:
            self.firm_manager.ev_production_bool = 1
            for firm in self.firm_manager.firms_list:
                firm.ev_production_bool = True

        #carbon price
        self.carbon_price = self.carbon_price_time_series[self.t_controller]
        #update_prices_and_emmisions
        self.gas_price = self.gas_price_california_vec[self.t_controller] + self.carbon_price*self.gas_emissions_intensity

        self.electricity_emissions_intensity = self.electricity_emissions_intensity_vec[self.t_controller]
        
        #pre tax
        self.electricity_price_subsidy_prop = self.electricity_price_subsidy_time_series[self.t_controller]
        self.electricity_price_subsidy_dollars = self.electricity_price_vec[self.t_controller]*self.electricity_price_subsidy_prop

        #add on tax
        self.electricity_price = self.electricity_price_vec[self.t_controller]*(1-self.electricity_price_subsidy_prop) + self.carbon_price*self.electricity_emissions_intensity

        self.rebate_calibration = self.rebate_calibration_time_series[self.t_controller]
        self.rebate = self.rebate_time_series[self.t_controller]
        self.used_rebate_calibration = self.used_rebate_calibration_time_series[self.t_controller]
        self.used_rebate = self.used_rebate_time_series[self.t_controller]

        self.production_subsidy = self.production_subsidy_time_series[self.t_controller]

    def update_firms(self):
        """
        Advance firm behavior for the current time step.

        Returns:
            list: Cars currently on sale across all firms.
        """
        cars_on_sale_all_firms = self.firm_manager.next_step(self.carbon_price, self.consider_ev_vec, self.new_bought_vehicles, self.gas_price, self.electricity_price, self.electricity_emissions_intensity, self.rebate, self.production_subsidy, self.rebate_calibration)
        return cars_on_sale_all_firms
    
    def update_social_network(self):
        """
        Advance social network behavior for the current time step.

        Returns:
            tuple: (consider_ev_vec, new_bought_vehicles)
        """
        consider_ev_vec, new_bought_vehicles = self.social_network.next_step(self.carbon_price,  self.second_hand_cars, self.cars_on_sale_all_firms, self.gas_price, self.electricity_price, self.electricity_emissions_intensity, self.rebate, self.used_rebate, self.electricity_price_subsidy_dollars, self.rebate_calibration, self.used_rebate_calibration)

        if self.t_controller == self.t_2030:
            self.utility_cum_2030 = deepcopy(self.social_network.utility_cumulative)     

        return consider_ev_vec, new_bought_vehicles

    def get_second_hand_cars(self):
        """
        Retrieve list of cars available from the second-hand market.

        Returns:
            list: Cars for sale in second-hand market.
        """
        self.second_hand_merchant.next_step(self.gas_price, self.electricity_price, self.electricity_emissions_intensity, self.cars_on_sale_all_firms, self.rebate_calibration, self.rebate)
        cars_on_sale_second_hand = self.second_hand_merchant.cars_on_sale

        return cars_on_sale_second_hand

    def calc_price_range_ice(self):
        """
        Calculate range of prices for ICE cars currently on sale. Used to record data.

        Returns:
            float: Difference between max and min ICE vehicle price.
        """
                
        prices = [car.price for car in self.cars_on_sale_all_firms if car.transportType == 2]
        min_price = np.min(prices)
        max_price = np.max(prices)
        return max_price - min_price

    def calc_total_policy_distortion(self):
        """
        Compute the total policy distortion cost at the end of simulation.

        Returns:
            float: Total distortion from all actors.
        """
        policy_distortion_firms = sum(firm.policy_distortion for firm in self.firm_manager.firms_list)
        policy_distortion = self.social_network.policy_distortion + self.firm_manager.policy_distortion + policy_distortion_firms
        return policy_distortion
    
    def calc_net_policy_distortion(self):
        """
        Compute net cost of policy distortion.

        Returns:
            float: Net policy cost from all actors.
        """
        policy_distortion_firms = sum(firm.policy_distortion for firm in self.firm_manager.firms_list)
        policy_distortion = self.social_network.net_policy_distortion - self.firm_manager.policy_distortion - policy_distortion_firms
        return -policy_distortion#NEGATIVE AS ITS COSTS

    def calc_EV_prop(self):
        """
        Compute proportion of EVs in the current vehicle fleet.

        Returns:
            float: EV share among all users.
        """
        EV_stock_prop = sum(1 if car.transportType == 3 else 0 for car in self.social_network.current_vehicles)/self.social_network.num_individuals#NEED FOR OPTIMISATION, measures the uptake EVS

        return EV_stock_prop
    
    ################################################################################################

    def next_step(self):
        """
        Advance the full model by one time step. Updates firms, users, second-hand market.
        """

        self.update_time_series_data()
        self.cars_on_sale_all_firms = self.update_firms()
        self.second_hand_cars  = self.get_second_hand_cars()
        self.consider_ev_vec, self.new_bought_vehicles = self.update_social_network()

        self.manage_saves()

        self.t_controller+=1

    def setup_continued_run_future(self, updated_parameters):
        """
        Reset and continue simulation using an updated future parameter config.

        Args:
            updated_parameters (dict): New configuration for future period.
        """
        self.parameters_controller = updated_parameters

        self.duration_future = self.parameters_controller["duration_future"]
        self.time_steps_max = self.parameters_controller["time_steps_max"]
        self.save_timeseries_data_state = self.parameters_controller["save_timeseries_data_state"]


        if self.save_timeseries_data_state:#SAVE DATA
            self.set_up_time_series_controller()
            self.time_series = []

            self.second_hand_merchant.save_timeseries_data_state = 1
            self.social_network.save_timeseries_data_state = 1
            self.firm_manager.save_timeseries_data_state = 1

            self.second_hand_merchant.set_up_time_series_second_hand_car()
            self.social_network.set_up_time_series_social_network()
            self.firm_manager.set_up_time_series_firm_manager()

            for firm in self.firm_manager.firms_list:
                firm.save_timeseries_data_state = 1
                firm.set_up_time_series_firm()

        #RESET COUNTERS FOR POLICY
        self.social_network.emissions_cumulative = 0
        self.social_network.emissions_cumulative_driving = 0
        self.social_network.emissions_cumulative_production = 0
        self.social_network.utility_cumulative = 0
        self.firm_manager.profit_cumulative = 0

        self.manage_calibration()
        self.manage_scenario()
        self.manage_policies() 

        self.gas_price_california_vec = np.concatenate((self.pre_future_gas_price_california_vec, self.gas_price_series_future), axis=None) 
        self.electricity_price_vec =  np.concatenate((self.pre_future_electricity_price_vec, self.electricity_price_series_future ), axis=None) 
        self.electricity_emissions_intensity_vec = np.concatenate((self.pre_future_electricity_emissions_intensity_vec, self.grid_emissions_intensity_series_future ), axis=None) 

        self.rebate_time_series = np.concatenate(( np.zeros(self.duration_burn_in + self.duration_calibration), self.rebate_time_series_future), axis=None) 
        self.used_rebate_time_series = np.concatenate(( np.zeros(self.duration_burn_in + self.duration_calibration), self.used_rebate_time_series_future), axis=None) 
        
        self.rebate_calibration_time_series = np.concatenate((self.burn_in_rebate_time_series, self.calibration_rebate_time_series), axis=None) #THIS IS BOTH BURN IN CALIBRATION AND FUTURE
        self.used_rebate_calibration_time_series = np.concatenate((self.burn_in_used_rebate_time_series, self.calibration_used_rebate_time_series), axis=None) 

        self.electricity_price_subsidy_time_series = np.concatenate(( np.zeros(self.duration_burn_in + self.duration_calibration), self.electricity_price_subsidy_time_series_future), axis=None) 
        self.production_subsidy_time_series = np.concatenate(( np.zeros(self.duration_burn_in + self.duration_calibration), self.production_subsidy_time_series_future), axis=None) 

