import copy
import numpy as np
from package.model.carModel import CarModel
from package.model.personalCar import PersonalCar
from package.model.firm import Firm
from collections import defaultdict
import itertools

class Firm_Manager:
    def __init__(self, parameters_firm_manager: dict, parameters_firm: dict, parameters_car_ICE: dict, parameters_car_EV: dict, ICE_landscape: dict, EV_landscape: dict):
        self.t_firm_manager = 0
        self.parameters_firm = parameters_firm

        self.policy_distortion = 0

        self.zero_profit_options_prod_sum = 0


        self.init_tech_seed = parameters_firm_manager["init_tech_seed"]
        self.J = int(round(parameters_firm_manager["J"]))
        self.N = int(round(parameters_firm_manager["N"]))

        #self.save_timeseries_data_state = parameters_firm_manager["save_timeseries_data_state"]
        #self.compression_factor_state = parameters_firm_manager["compression_factor_state"]
        self.carbon_price = parameters_firm_manager["carbon_price"]
        self.id_generator = parameters_firm_manager["IDGenerator_firms"]
        self.kappa = parameters_firm_manager["kappa"]
        self.num_individuals = parameters_firm_manager["num_individuals"]

        self.rebate_count_cap = parameters_firm_manager["rebate_count_cap_adjusted"]

        self.time_steps_tracking_market_data = parameters_firm_manager["time_steps_tracking_market_data"]

        self.num_beta_segments = parameters_firm_manager["num_beta_segments"]

        self.all_segment_codes = list(itertools.product(range(self.num_beta_segments), range(2), range(2)))

        #landscapes
        self.landscape_ICE = ICE_landscape
        self.landscape_EV = EV_landscape

        #car paramets
        self.parameters_car_ICE = parameters_car_ICE
        self.parameters_car_EV = parameters_car_EV 

        #PRICE, NOT SURE IF THIS IS NECESSARY
        self.beta_threshold = parameters_firm_manager["beta_threshold"]
        self.beta_val_empty_upper =  parameters_firm_manager["beta_val_empty_upper"]
        self.beta_val_empty_lower =  parameters_firm_manager["beta_val_empty_lower"]
        
        self.gamma_threshold = parameters_firm_manager["gamma_threshold"]
        self.gamma_val_empty_upper = parameters_firm_manager["gamma_val_empty_upper"]
        self.gamma_val_empty_lower = parameters_firm_manager["gamma_val_empty_upper"]

        self.random_state = np.random.RandomState(self.init_tech_seed)  # Local random state

        self.innovation_seed_list = self.random_state.randint(0,1000, self.J)

        self.init_firms()
        
        #calculate the inital attributes of all the cars on sale
        self.cars_on_sale_all_firms = self.generate_cars_on_sale_all_firms()
        self.age_max = parameters_firm_manager["init_car_age_max"]
        self.old_cars = self.gen_old_cars()        

    ###########################################################################################################
    #INITIALISATION

    def gen_old_cars(self):
        """
        Using random assortment of cars intially pick some random cars and set a random age distribution
        """

        model_choices = self.random_state.choice(self.cars_on_sale_all_firms, self.num_individuals)
        age_range = np.arange(0,self.age_max)
        age_list = self.random_state.choice(age_range, self.num_individuals)
        car_list = []
        for i, car in enumerate(model_choices):
            personalCar_id = self.id_generator.get_new_id()
            car_real = PersonalCar(personalCar_id, car.firm, None, car.component_string, car.parameters, car.attributes_fitness, car.price, init_car=1)
            car_real.L_a_t = age_list[i]
            car_list.append(car_real)
        return car_list

    def init_firms(self):
        """
        Generate a initial ICE and EV TECH TO START WITH 
        """

        #Generate the initial fitness values of the starting tecnology(ies)

        self.init_tech_component_string_ICE = self.landscape_ICE.min_fitness_string
        self.init_tech_component_string_EV = self.landscape_EV.min_fitness_string


        decimal_value_ICE = int(self.init_tech_component_string_ICE, 2)
        decimal_value_EV = int(self.init_tech_component_string_EV, 2)


        init_tech_component_string_list_N_ICE = self.invert_bits_one_at_a_time(decimal_value_ICE, len(self.init_tech_component_string_ICE))
        init_tech_component_string_list_N_EV = self.invert_bits_one_at_a_time(decimal_value_EV, len(self.init_tech_component_string_EV))

        init_tech_component_string_list_ICE = self.random_state.choice(init_tech_component_string_list_N_ICE, self.J)
        init_tech_component_string_list_EV = self.random_state.choice(init_tech_component_string_list_N_EV, self.J)

        self.init_tech_list_ICE = [CarModel(init_tech_component_string_list_ICE[j], self.landscape_ICE, parameters = self.parameters_car_ICE, choosen_tech_bool=1) for j in range(self.J)]
        self.init_tech_list_EV = [CarModel(init_tech_component_string_list_EV[j], self.landscape_EV, parameters = self.parameters_car_EV, choosen_tech_bool=1) for j in range(self.J)]

        #global repo
        self.universal_model_repo_ICE = {} 
        self.universal_model_repo_EV = {}

        self.parameters_firm["universal_model_repo_EV"] = self.universal_model_repo_EV
        self.parameters_firm["universal_model_repo_ICE"] = self.universal_model_repo_ICE
        self.parameters_firm["segment_codes"] = self.all_segment_codes
        #Create the firms, these store the data but dont do anything otherwise
        self.firms_list = [Firm(j, self.init_tech_list_ICE[j], self.init_tech_list_EV[j],  self.parameters_firm, self.parameters_car_ICE, self.parameters_car_EV, self.innovation_seed_list[j]) for j in range(self.J)]

    def invert_bits_one_at_a_time(self, decimal_value, length):
        """THIS IS ONLY USED ONCE TO GENERATE HETEROGENOUS INITIAL TECHS"""
        inverted_binary_values = []
        for bit_position in range(length):
            inverted_value = decimal_value ^ (1 << bit_position)
            inverted_binary_value = format(inverted_value, f'0{length}b')
            inverted_binary_values.append(inverted_binary_value)
        return inverted_binary_values

    
    def generate_cars_on_sale_all_firms(self):
        """ONLY USED ONCE AT INITITALISATION"""
        cars_on_sale_all_firms = []
        for firm in self.firms_list:
            cars_on_sale_all_firms.extend(firm.cars_on_sale)
        return cars_on_sale_all_firms

    ###########################################################################################################

        ###########################################################################################################
    # BETA-SEGMETING LOGIC

    def input_social_network_data(self, beta_vec, environmental_awareness_vec, consider_ev_vec, beta_bins):
        """
        Set up the individual-level vectors. 
        We'll convert:
          - beta_vec into 5 segments (0..4)
          - gamma_vec into binary (0..1)
          - consider_ev_vec is presumably 0..1
        """
        self.beta_vec = beta_vec
        self.gamma_vec = environmental_awareness_vec
        self.consider_ev_vec = consider_ev_vec

        # Convert beta into 5 segments
        
        # Suppose your beta values are roughly in [0, 1].  If not, pick different bin edges.
        self.beta_bins = beta_bins#np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.01])  
        # np.digitize returns an integer from 1..len(self.beta_bins)-1
        # so we subtract 1 to get 0..4
        self.beta_segment_idx = np.digitize(self.beta_vec, self.beta_bins) - 1  
        # Now each entry in self.beta_segment_idx is in {0,1,2,3,4}.

        # Convert gamma to 0 or 1 based on threshold
        self.gamma_binary = (self.gamma_vec > self.gamma_threshold).astype(int)

    def generate_market_data(self):
        """

        For each segment code, we store:
         - I_s_t     (count of individuals)
         - beta_s_t  (average beta)
         - gamma_s_t (average gamma)
         - W     (will be computed after we calculate utilities)
        """
        # 1) Build a dictionary for ALL possible combos
        
        self.market_data = {}
        for code in self.all_segment_codes:
            self.market_data[code] = {
                "I_s_t": 0,
                "beta_s_t": 0.0,
                "gamma_s_t": 0.0,
                "W": 0.0,
                "history_I_s_t": [],
                "history_W": []
            }

        # 2) Count how many individuals fall into each segment code
        #    also sum up their beta, gamma for averages
        segment_counts = defaultdict(int)
        beta_sums = defaultdict(float)
        gamma_sums = defaultdict(float)

        # Go through each individual
        for i in range(self.num_individuals):
            b_idx = self.beta_segment_idx[i]         # in [0..4]
            g_idx = self.gamma_binary[i]             # in [0..1]
            e_idx = self.consider_ev_vec[i]          # in [0..1]
            seg_code = (b_idx, g_idx, e_idx)

            segment_counts[seg_code] += 1
            beta_sums[seg_code] += self.beta_vec[i]
            gamma_sums[seg_code] += self.gamma_vec[i]

        # 3) Compute averages for each segment
        for code in self.all_segment_codes:
            count = segment_counts[code]
            if count > 0:
                avg_beta = beta_sums[code] / count
                avg_gamma = gamma_sums[code] / count
            else:
                #FIX THIS SO THAT THE CORRECT BETA VALUE IS CHOSEN!
                b_idx, g_idx, _ = code
                if b_idx >= 2:
                    avg_beta = self.beta_val_empty_upper
                else:
                    avg_beta = self.beta_val_empty_lower

                if g_idx == 1:
                    avg_gamma = self.gamma_val_empty_upper
                else:
                    avg_gamma = self.gamma_val_empty_lower

            self.market_data[code]["I_s_t"] = count
            self.market_data[code]["beta_s_t"] = avg_beta
            self.market_data[code]["gamma_s_t"] = avg_gamma

        # 4) Now we need to compute the initial W for each segment
        for firm in self.firms_list:
            firm.calc_utility_cars_segments(self.market_data, self.cars_on_sale_all_firms)

        # 5) Sum up the utilities across all cars for each segment
        segment_W = defaultdict(float)
        for firm in self.firms_list:
            for car in firm.cars_on_sale:
                for code, U in car.car_utility_segments_U.items():
                    segment_W[code] += U**2

        # 6) Store the U_sum in market_data
        for code in self.all_segment_codes:
            self.market_data[code]["W"] = segment_W[code]


    ############################################################################################################################################################
    #GENERATE MARKET DATA DYNAMIC

    def generate_cars_on_sale_all_firms_and_sum_U(self, market_data, gas_price, electricity_price, electricity_emissions_intensity, rebate, discriminatory_corporate_tax, production_subsidy, research_subsidy):
        cars_on_sale_all_firms = []
        segment_W = defaultdict(float)
        self.zero_profit_options_prod_sum = 0
        self.zero_profit_options_research_sum = 0

        #print("In firm manager data: ", market_data, self.carbon_price, gas_price, electricity_price, electricity_emissions_intensity, rebate)
        for firm in self.firms_list:
            self.zero_profit_options_prod_sum += firm.zero_profit_options_prod#CAN DELETE OCNE FIXED ISSUE O uitlity in firms prod
            self.zero_profit_options_research_sum += firm.zero_profit_options_research
            cars_on_sale = firm.next_step(market_data, self.carbon_price, gas_price, electricity_price, electricity_emissions_intensity, rebate, discriminatory_corporate_tax, production_subsidy, research_subsidy)

            cars_on_sale_all_firms.extend(cars_on_sale)
            for car in cars_on_sale:
                for segment, U in car.car_utility_segments_U.items():
                    segment_W[segment] += U**2

        return cars_on_sale_all_firms, segment_W

    def update_market_data_moving_average(self, W_segment):
        """
        If you still want a moving average approach:
        """
        segment_counts = defaultdict(int)
        for i in range(self.num_individuals):
            b_idx = self.beta_segment_idx[i]
            g_idx = self.gamma_binary[i]
            e_idx = self.consider_ev_vec[i]
            code = (b_idx, g_idx, e_idx)
            segment_counts[code] += 1

        self.total_W = 0.0

        for code in self.market_data.keys():
            # Append current values to history
            self.market_data[code]["history_I_s_t"].append(segment_counts[code])
            self.market_data[code]["history_W"].append(W_segment[code])

            # Trim history to the last N time steps
            if len(self.market_data[code]["history_I_s_t"]) > self.time_steps_tracking_market_data:
                self.market_data[code]["history_I_s_t"].pop(0)
            if len(self.market_data[code]["history_W"]) > self.time_steps_tracking_market_data:
                self.market_data[code]["history_W"].pop(0)

            # Calculate moving averages
            moving_avg_I_s_t = np.mean(self.market_data[code]["history_I_s_t"])
            moving_avg_W = np.mean(self.market_data[code]["history_W"])

            # Store the moving averages
            self.market_data[code]["I_s_t"] = moving_avg_I_s_t
            self.market_data[code]["W"] = moving_avg_W

            self.total_W += moving_avg_W

    ######################################################################################################################

    def calc_total_profits(self, past_new_bought_vehicles):

        total_profit_all_firms = 0         
        for car in self.cars_on_sale_all_firms:#LOOP OVER ALL CARS ON SALE TO DO IT IN ONE GO I GUESS
            num_vehicle_sold = past_new_bought_vehicles.count(car)
            
            if num_vehicle_sold > 0:#ONLY COUNT WHEN A CAR HAS ACTUALLY BEEN SOLD
                profit = (car.price - car.ProdCost_t)
                total_profit = num_vehicle_sold*profit
                car.firm.firm_profit += total_profit#I HAVE NO IDEA IF THIS WILL WORK
                total_profit_all_firms += total_profit
                
                #OPTIMIZATION OF DISCRIMINATORY CORPORATE TAX
                if car.transportType == 2 and total_profit > 0:#ICE CARS THAT MAKE ACTUAL PROFITS
                    self.policy_distortion += total_profit*self.discriminatory_corporate_tax

                #OPTIMIZATION OF PRODUCTION SUBSIDY
                if car.transportType == 3:
                    self.policy_distortion += num_vehicle_sold*self.production_subsidy

        return total_profit_all_firms

    def calculate_market_share(self, firm, past_new_bought_vehicles, total_sales):
        """
        Calculate market share for a specific firm.
        Parameters:
            firm : Firm - The firm object for which to calculate market share
            past_new_bought_vehicles : list - List of car objects representing vehicles sold in the past period
        Returns:
            MS_firm : float - Market share for the specified firm, 0 if no cars sold
        """
        # Calculate total sales for the specified firm by summing prices of cars sold by this firm
        firm_sales = sum(car.price for car in past_new_bought_vehicles if car.firm == firm)
        
        # If total_sales is zero, return 0 to avoid division by zero; otherwise, calculate firm market share
        MS_firm = firm_sales / total_sales if total_sales > 0 else 0
        return MS_firm

    def calculate_market_concentration(self, past_new_bought_vehicles):
        """
        Calculate market concentration (HHI) based on market shares of all firms.
        Parameters:
            past_new_bought_vehicles : list - List of car objects representing vehicles sold in the past period
        Returns:
            HHI : float - Market concentration (HHI)
        """
        # Calculate total market sales by summing prices of all cars sold in the market
        total_sales = sum(car.price for car in past_new_bought_vehicles)

        # Calculate the HHI by summing the squares of market shares for each firm in firm_list
        HHI = sum(self.calculate_market_share(firm, past_new_bought_vehicles, total_sales)**2 for firm in self.firms_list)
        return HHI

    #######################################################################################################################
    #DEAL WITH REBATE
    def add_social_network(self,social_network):
        self.social_network = social_network

    def handle_limited_rebate(self):#THIS BLOCKS FIRMS IF THEY RECIEVE TOO MUCH REBATE
        for vehicle in self.past_new_bought_vehicles:
            if vehicle.transport_type == 3:
                vehicle.firm.EVs_sold += 1
        
        for firm in self.firms_list:
            if firm.EVs_sold > self.rebate_count_cap:
                self.social_network.add_firm_rebate_exclusion_set(firm.firm_id)
                print("firm got to the limit:", self.t_firm_manager,firm.firm_id)

    #####################################################################################################################

    def set_up_time_series_firm_manager(self):
        #self.history_cars_on_sale_all_firms = [self.cars_on_sale_all_firms]
        self.history_total_profit = []
        self.history_market_concentration = []
        self.history_segment_count = []
        self.history_cars_on_sale_EV_prop = []
        self.history_cars_on_sale_ICE_prop = []
        self.history_cars_on_sale_price = []

        self.history_market_data = []
        self.history_zero_profit_options_prod_sum = []
        self.history_zero_profit_options_research_sum = []

    def save_timeseries_data_firm_manager(self):
        #self.history_cars_on_sale_all_firms.append(self.cars_on_sale_all_firms)
        #self.total_profit = self.calc_total_profits(self.past_new_bought_vehicles)
        self.HHI = self. calculate_market_concentration(self.past_new_bought_vehicles)
        self.calc_vehicles_chosen_list(self.past_new_bought_vehicles)
        self.history_cars_on_sale_price.append([car.price for car in self.cars_on_sale_all_firms])

        self.history_total_profit.append(self.total_profit)
        self.history_market_concentration.append(self.HHI)
        self.history_segment_count.append([segment_data["I_s_t"] for segment_data in self.market_data.values()])

        count_transport_type_2 = sum(1 for car in self.cars_on_sale_all_firms if car.transportType == 2)
        count_transport_type_3 = sum(1 for car in self.cars_on_sale_all_firms if car.transportType == 3)

        self.history_cars_on_sale_ICE_prop.append(count_transport_type_2)
        self.history_cars_on_sale_EV_prop.append(count_transport_type_3)

        self.history_market_data.append(copy.deepcopy(self.market_data))

        self.history_zero_profit_options_prod_sum.append(self.zero_profit_options_prod_sum/self.J)
        self.history_zero_profit_options_research_sum.append(self.zero_profit_options_research_sum/self.J)

    def calc_vehicles_chosen_list(self, past_new_bought_vehicles):
        for firm in self.firms_list:
            firm.firm_cars_users = sum(1 for car in past_new_bought_vehicles if car.firm == firm)

    #####################################################################################################################

    def next_step(self, carbon_price, consider_ev_vec, new_bought_vehicles,  gas_price, electricity_price, electricity_emissions_intensity, rebate,  discriminatory_corporate_tax, production_subsidy, research_subsidy):
        
        self.t_firm_manager += 1
        
        self.past_new_bought_vehicles = new_bought_vehicles
        self.total_profit = self.calc_total_profits(self.past_new_bought_vehicles)#NEED TO CALC TOTAL PROFITS NOW before the cars on sale change?
         
        self.carbon_price = carbon_price

        self.discriminatory_corporate_tax = discriminatory_corporate_tax
        self.production_subsidy = production_subsidy
        self.cars_on_sale_all_firms, W_segment = self.generate_cars_on_sale_all_firms_and_sum_U(self.market_data, gas_price, electricity_price, electricity_emissions_intensity, rebate, discriminatory_corporate_tax, production_subsidy, research_subsidy)#WE ASSUME THAT FIRMS DONT CONSIDER SECOND HAND MARKET
    
        self.consider_ev_vec = consider_ev_vec#UPDATE THIS TO NEW CONSIDERATION
        #self.update_market_data(sums_U_segment)
        self.update_market_data_moving_average(W_segment)

        return self.cars_on_sale_all_firms, self.total_W