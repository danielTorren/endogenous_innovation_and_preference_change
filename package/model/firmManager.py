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

        self.carbon_price = parameters_firm_manager["carbon_price"]
        self.id_generator = parameters_firm_manager["IDGenerator_firms"]
        self.kappa = parameters_firm_manager["kappa"]

        self.num_individuals = parameters_firm_manager["num_individuals"]
        self.time_steps_tracking_market_data = parameters_firm_manager["time_steps_tracking_market_data"]
        self.min_W = parameters_firm_manager["min_W"]
        self.num_beta_segments = parameters_firm_manager["num_beta_segments"]
        self.num_gamma_segments = parameters_firm_manager["num_gamma_segments"]

        self.all_segment_codes = list(itertools.product(range(self.num_beta_segments), range(self.num_gamma_segments), range(2)))
        self.num_segments = len(self.all_segment_codes)

        #landscapes
        self.landscape_ICE = ICE_landscape
        self.landscape_EV = EV_landscape

        #car paramets
        self.parameters_car_ICE = parameters_car_ICE
        self.parameters_car_EV = parameters_car_EV 
    

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
        # Define mean and standard deviation for normal distribution
        mu = 120#self.age_max / 2  # Mean age at half of max age
        sigma = 60#self.age_max / 4  # Standard deviation (adjustable)

        # Generate normally distributed ages, ensuring values are within range
        age_list = np.clip(self.random_state.normal(mu, sigma, self.num_individuals), 0, self.age_max).astype(int)

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


    ############################################################################################################################################################
    #GENERATE MARKET DATA DYNAMIC

    def input_social_network_data(self, beta_vec, gamma_vec, consider_ev_vec, beta_bins, gamma_bins):
        """
        Set up the individual-level vectors. 
        We'll convert:
          - beta_vec into 5 segments (0..4)
          - gamma_vec into binary (0..1)
          - consider_ev_vec is presumably 0..1
        """
        self.beta_vec = beta_vec
        self.gamma_vec = gamma_vec
        self.consider_ev_vec = consider_ev_vec


        self.beta_bins = beta_bins
        self.gamma_bins = gamma_bins
        self.beta_segment_idx = np.digitize(self.beta_vec, self.beta_bins) - 1  
        self.gamma_segment_idx = np.digitize(self.gamma_vec, self.gamma_bins) - 1  

        # Now each entry in self.beta_segment_idx is in {0,1,2,3,4}.
        # Convert gamma to 0 or 1 based on threshold
        #self.gamma_binary = (self.gamma_vec > self.gamma_threshold).astype(int)

    def calc_exp(self, U):
        exp_input = self.kappa*U
        #exp_input = np.clip(exp_input, -700, 700)#CLIP SO DONT GET OVERFLOWS
        comp = np.exp(exp_input)
        return comp

    def generate_market_data(self):
        """
        CALLED ONCE AT THE END OF THE SET UP BY CONTROLLER AND ONLY AFTER THE SOCIAL NETWORK INFORMATION HAS BEEN RECEIVIED
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
                "W": self.min_W,#0.0,
                "history_I_s_t": [],
                "history_W": [],
                "maxU": 0
            }

        # 2) Count how many individuals fall into each segment code
        segment_counts = defaultdict(int)

        for i in range(self.num_individuals):
            b_idx = self.beta_segment_idx[i]         # in [0..4]
            g_idx = self.gamma_segment_idx[i]            # in [0..1]
            e_idx = self.consider_ev_vec[i]          # in [0..1]
            seg_code = (b_idx, g_idx, e_idx)

            segment_counts[seg_code] += 1

        # 3) Compute midpoints for each segment
        for i, code in enumerate(self.all_segment_codes):
            b_idx, g_idx, _ = code

            # Assign values to the market data
            self.market_data[code]["I_s_t"] = segment_counts[code]
        
        #4) calc the utility of each car (already did base utility in the car but need the full value including price and emissiosn production)
        for firm in self.firms_list:
            firm.calc_init_U_segments()


        # 5) Sum up the utilities across all cars for each segment
        segment_W = defaultdict(float)

        for segment in self.all_segment_codes:
            segment_W[segment] = self.min_W#SET THE MINIMUM
        
        #5.5) calc the max U 
        for firm in self.firms_list:
            for car in firm.cars_on_sale:
                for code, U in car.car_utility_segments_U.items():
                    if U > self.market_data[code]["maxU"]:
                        self.market_data[code]["maxU"] = U


        for firm in self.firms_list:
            for car in firm.cars_on_sale:
                for code, U in car.car_utility_segments_U.items():
                        segment_W[code] += self.calc_exp(U)

        # 6) Store the U_sum in market_data
        for code in self.all_segment_codes:
            self.market_data[code]["W"] = segment_W[code]
            #print("INIT W", self.market_data[code]["W"])
    
        self.I_s_t_vec = np.asarray([self.market_data[code]["I_s_t"] for code in self.all_segment_codes])
        self.W_vec = np.asarray([self.market_data[code]["W"] for code in self.all_segment_codes])
        self.maxU_vec = np.asarray([self.market_data[code]["maxU"] for code in self.all_segment_codes])

    def update_W_immediate(self):
        #calc the total "probability of selection" of the market based on max utility in the segment
        segment_W = defaultdict(float)

        for segment in self.all_segment_codes:
            segment_W[segment] = self.min_W#RESET THEM INCASE
        
        #UPDATE U MAX
        for car in self.cars_on_sale_all_firms:
            for segment, U in car.car_utility_segments_U.items():
                    if U > self.market_data[segment]["maxU"]:
                            self.market_data[segment]["maxU"] = U

        for car in self.cars_on_sale_all_firms:
            for segment, U in car.car_utility_segments_U.items():
                segment_W[segment] += self.calc_exp(U)

        maxU_vec = np.asarray([self.market_data[code]["maxU"] for code in self.all_segment_codes])

        return segment_W, maxU_vec
        
    def update_market_data_moving_average(self, W_segment):

        segment_counts = defaultdict(int)
        for i in range(self.num_individuals):
            b_idx = self.beta_segment_idx[i]
            g_idx = self.gamma_segment_idx[i]
            e_idx = self.consider_ev_vec[i]
            code = (b_idx, g_idx, e_idx)
            segment_counts[code] += 1

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
        
        I_s_t_vec = np.asarray([self.market_data[code]["I_s_t"] for code in self.all_segment_codes])
        W_vec = np.asarray([self.market_data[code]["W"] for code in self.all_segment_codes])

        return I_s_t_vec, W_vec
        
    
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

    def calc_profit_margin(self, past_new_bought_vehicles):
        
        profit_margin_ICE = []
        profit_margin_EV = []

        for car in past_new_bought_vehicles:
            profit_margin = (car.price - car.ProdCost_t)/car.ProdCost_t
            if car.transportType == 3:
                profit_margin_EV.append(profit_margin)
            else:
                profit_margin_ICE.append(profit_margin)
        #if not profit_margin_EV:
        #    profit_margin_EV.append(np.nan)#if no evs then just add nan
        return profit_margin_ICE, profit_margin_EV
    
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


        self.history_profit_margins_EV = []
        self.history_profit_margins_ICE = []
        self.history_W = []

        self.history_quality_ICE = []
        self.history_efficiency_ICE  = []
        self.history_production_cost_ICE  = []

        self.history_quality_EV = []
        self.history_efficiency_EV = []
        self.history_production_cost_EV = []
        self.history_battery_EV = []


    def save_timeseries_data_firm_manager(self):
        #self.history_cars_on_sale_all_firms.append(self.cars_on_sale_all_firms)
        #self.total_profit = self.calc_total_profits(self.past_new_bought_vehicles)
        self.HHI = self. calculate_market_concentration(self.past_new_bought_vehicles)
        profit_margin_ICE, profit_margin_EV = self.calc_profit_margin(self.past_new_bought_vehicles)

        self.history_profit_margins_EV.append(profit_margin_EV)
        self.history_profit_margins_ICE.append(profit_margin_ICE)

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

        self.history_W.append(list(self.W_segment.values()))

        self.quality_vals_ICE = []
        self.efficiency_vals_ICE = []
        self.production_cost_vals_ICE = []
        self.quality_vals_EV = []
        self.efficiency_vals_EV = []
        self.production_cost_vals_EV = []
        self.battery_EV = []

        for car in self.cars_on_sale_all_firms:
            if car.transportType == 2:#ICE 
                self.quality_vals_ICE.append(car.Quality_a_t)#done here for efficiency
                self.efficiency_vals_ICE.append(car.Eff_omega_a_t)
                self.production_cost_vals_ICE.append(car.ProdCost_t)
            else:#EV
                self.quality_vals_EV.append(car.Quality_a_t)#done here for efficiency
                self.efficiency_vals_EV.append(car.Eff_omega_a_t)
                self.production_cost_vals_EV.append(car.ProdCost_t)
                self.battery_EV.append(car.B)

        if self.quality_vals_EV:
            self.history_quality_EV.append(self.quality_vals_EV)
            self.history_efficiency_EV.append(self.efficiency_vals_EV)
            self.history_production_cost_EV.append(self.production_cost_vals_EV)
            self.history_battery_EV.append(self.battery_EV)
        else:
            self.history_quality_EV.append([np.nan])
            self.history_efficiency_EV.append([np.nan])
            self.history_production_cost_EV.append([np.nan])
            self.history_battery_EV.append([np.nan])

        if self.quality_vals_ICE:
            self.history_quality_ICE.append(self.quality_vals_ICE)
            self.history_efficiency_ICE.append(self.efficiency_vals_ICE)
            self.history_production_cost_ICE.append(self.production_cost_vals_ICE)
        else:
            self.history_quality_ICE.append([np.nan])
            self.history_efficiency_ICE.append([np.nan])
            self.history_production_cost_ICE.append([np.nan])

    def calc_vehicles_chosen_list(self, past_new_bought_vehicles):
        for firm in self.firms_list:
            firm.firm_cars_users = sum(1 for car in past_new_bought_vehicles if car.firm == firm)


#####################################################################################################################################################
    def update_firms(self, gas_price, electricity_price, electricity_emissions_intensity, rebate, discriminatory_corporate_tax, production_subsidy, research_subsidy, rebate_calibration):
        cars_on_sale_all_firms = []
        
        self.zero_profit_options_prod_sum = 0
        self.zero_profit_options_research_sum = 0


        # 4) Now we need to compute the initial W for each segment
        for firm in self.firms_list:
            self.zero_profit_options_prod_sum += firm.zero_profit_options_prod#CAN DELETE OCNE FIXED ISSUE O uitlity in firms prod
            self.zero_profit_options_research_sum += firm.zero_profit_options_research
            cars_on_sale = firm.next_step(self.I_s_t_vec, self.W_vec, self.maxU_vec, self.carbon_price, gas_price, electricity_price, electricity_emissions_intensity, rebate, discriminatory_corporate_tax, production_subsidy, research_subsidy, rebate_calibration)

            cars_on_sale_all_firms.extend(cars_on_sale)

        return cars_on_sale_all_firms

#####################################################################################################################

    def next_step(self, carbon_price, consider_ev_vec, new_bought_vehicles,  gas_price, electricity_price, electricity_emissions_intensity, rebate,  discriminatory_corporate_tax, production_subsidy, research_subsidy, rebate_calibration):
        
        self.t_firm_manager += 1
        self.past_new_bought_vehicles = new_bought_vehicles
        self.total_profit = self.calc_total_profits(self.past_new_bought_vehicles)#NEED TO CALC TOTAL PROFITS NOW before the cars on sale change?
        
        self.consider_ev_vec = consider_ev_vec#UPDATE THIS TO NEW CONSIDERATION
        self.carbon_price = carbon_price
        
        self.discriminatory_corporate_tax = discriminatory_corporate_tax
        self.production_subsidy = production_subsidy
        
        self.cars_on_sale_all_firms  = self.update_firms(gas_price, electricity_price, electricity_emissions_intensity, rebate, discriminatory_corporate_tax, production_subsidy, research_subsidy, rebate_calibration)#WE ASSUME THAT FIRMS DONT CONSIDER SECOND HAND MARKET
        self.W_segment, self.maxU_vec = self.update_W_immediate()#calculate the competiveness of the market current

        #print("W im:",np.min(list(self.W_segment.values())),np.max(list(self.W_segment.values())))
        self.I_s_t_vec, self.W_vec = self.update_market_data_moving_average(self.W_segment)#update the rollign vlaues

        return self.cars_on_sale_all_firms