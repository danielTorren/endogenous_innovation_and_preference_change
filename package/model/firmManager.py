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
        self.profit_cumulative = 0
        self.random_state_input = parameters_firm_manager["random_state_input"]

        self.zero_profit_options_prod_sum = 0

        self.HHI_past_new_bought_vehicles_history = []
        self.margin_past_new_bought_vehicles_history = []

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

        self.init_car_age_mean = parameters_firm_manager["init_car_age_mean"]
        self.init_car_age_std = parameters_firm_manager["init_car_age_std"]

        self.ev_production_bool = 0

        self.production_subsidy = 0 #NEEDED FOR TIEM STEP 0

        self.all_segment_codes = list(itertools.product(range(self.num_beta_segments), range(self.num_gamma_segments), range(2)))
        self.num_segments = len(self.all_segment_codes)

        #landscapes
        self.landscape_ICE = ICE_landscape
        self.landscape_EV = EV_landscape

        #car paramets
        self.parameters_car_ICE = parameters_car_ICE
        self.parameters_car_EV = parameters_car_EV 

        self.init_firms()
        
        #calculate the inital attributes of all the cars on sale
        self.cars_on_sale_all_firms = self.generate_cars_on_sale_all_firms()
             

    ###########################################################################################################
    #INITIALISATION

    def gen_initial_cars(self):
        model_choices = self.random_state_input.choice(self.cars_on_sale_all_firms, self.num_individuals)

        car_list = []
        for i, car in enumerate(model_choices):
            personalCar_id = self.id_generator.get_new_id()
            car_real = PersonalCar(personalCar_id, car.firm, None, car.component_string, car.parameters, car.attributes_fitness, car.price, init_car=1)
            car_real.L_a_t = 0
            car_list.append(car_real)

        self.old_cars = car_list   
        return self.old_cars
    

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

        init_tech_component_string_list_ICE = self.random_state_input.choice(init_tech_component_string_list_N_ICE, self.J)
        init_tech_component_string_list_EV = self.random_state_input.choice(init_tech_component_string_list_N_EV, self.J)

        self.init_tech_list_ICE = [CarModel(init_tech_component_string_list_ICE[j], self.landscape_ICE, parameters = self.parameters_car_ICE, choosen_tech_bool=1) for j in range(self.J)]
        self.init_tech_list_EV = [CarModel(init_tech_component_string_list_EV[j], self.landscape_EV, parameters = self.parameters_car_EV, choosen_tech_bool=1) for j in range(self.J)]

        #global repo
        self.universal_model_repo_ICE = {} 
        self.universal_model_repo_EV = {}

        self.parameters_firm["universal_model_repo_EV"] = self.universal_model_repo_EV
        self.parameters_firm["universal_model_repo_ICE"] = self.universal_model_repo_ICE
        self.parameters_firm["segment_codes"] = self.all_segment_codes

        #Create the firms, these store the data but dont do anything otherwise
        self.firms_list = [Firm(j, self.init_tech_list_ICE[j], self.init_tech_list_EV[j],  self.parameters_firm, self.parameters_car_ICE, self.parameters_car_EV) for j in range(self.J)]

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
            b_idx, g_idx, e_idx = code

            if (e_idx == 0) or (e_idx == 1 and self.ev_production_bool):
                self.market_data[code]["I_s_t"] = segment_counts[code]#IS NOT AN EV SEGMENT or CAN PRODUCE EVS AND THE SEGMENT ALLOWS IT
            else:
                self.market_data[code]["I_s_t"] = 0#CANT PRODUCE AN EV
        
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
            #SEGMENT COUNTS
            e_idx = code[2]
            if (e_idx == 0) or (e_idx == 1 and self.ev_production_bool):
                count = segment_counts[code]#IS NOT AN EV SEGMENT or CAN PRODUCE EVS AND THE SEGMENT ALLOWS IT
            else:
                count = 0#CANT PRODUCE AN EV
            self.market_data[code]["history_I_s_t"].append(count)

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

    def calc_total_profits(self, past_new_bought_vehicles, prod_subsidy):

        total_profit_all_firms = 0         
        for car in self.cars_on_sale_all_firms:#LOOP OVER ALL CARS ON SALE TO DO IT IN ONE GO I GUESS
            num_vehicle_sold = past_new_bought_vehicles.count(car)
            
            if num_vehicle_sold > 0:#ONLY COUNT WHEN A CAR HAS ACTUALLY BEEN SOLD
                
                if car.transportType == 3:
                    profit = car.price - np.maximum(0,car.ProdCost_t - prod_subsidy)
                else:
                    profit = car.price - car.ProdCost_t

                total_profit = num_vehicle_sold*profit
                car.firm.firm_profit += total_profit#I HAVE NO IDEA IF THIS WILL WORK
                total_profit_all_firms += total_profit

                #OPTIMIZATION OF PRODUCTION SUBSIDY
                if car.transportType == 3:
                    self.policy_distortion += num_vehicle_sold * np.minimum(car.ProdCost_t, self.production_subsidy)

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

    def calculate_market_concentration(self, new_bought_vehicles):
        """
        Calculate market concentration (HHI) using all cars bought over the last 12 time steps.
        This function maintains its own history of purchases.
        """

        # Append the new purchases to history
        self.HHI_past_new_bought_vehicles_history.append(new_bought_vehicles)

        # Trim to last 12 time steps
        if len(self.HHI_past_new_bought_vehicles_history) > 12:
            self.HHI_past_new_bought_vehicles_history.pop(0)

        # Flatten the list to get all purchases from the last 12 time steps
        all_purchases = list(itertools.chain(*self.HHI_past_new_bought_vehicles_history))

        # Calculate total market sales over the last 12 time steps
        total_sales = sum(car.price for car in all_purchases)

        # If no sales, return HHI as zero
        if total_sales == 0:
            return 0

        # Calculate the HHI by summing the squares of market shares for each firm
        HHI = sum(self.calculate_market_share(firm, all_purchases, total_sales) ** 2 for firm in self.firms_list)

        return HHI

    def calc_profit_margin(self, new_bought_vehicles):
        """
        Calculate market concentration (HHI) using all cars bought over the last 12 time steps.
        This function maintains its own history of purchases.
        """

        # Append the new purchases to history
        self.margin_past_new_bought_vehicles_history.append(new_bought_vehicles)

        # Trim to last 12 time steps
        if len(self.margin_past_new_bought_vehicles_history) > 12:
            self.margin_past_new_bought_vehicles_history.pop(0)

        # Flatten the list to get all purchases from the last 12 time steps
        all_purchases = list(itertools.chain(*self.HHI_past_new_bought_vehicles_history))

        profit_margin_ICE = []
        profit_margin_EV = []

        # Calculate the HHI by summing the squares of market shares for each firm
        for car in all_purchases:
            if car.transportType == 3:
                prod_cost = np.maximum(0, car.ProdCost_t - self.production_subsidy)
                
                if prod_cost == 0:
                    profit_margin = np.inf
                else:
                    profit_margin = (car.price - prod_cost)/prod_cost
                
                profit_margin_EV.append(profit_margin)
            else:
                
                prod_cost = car.ProdCost_t
                if prod_cost == 0:
                    profit_margin = np.inf
                else:
                    profit_margin = (car.price - prod_cost)/prod_cost
                profit_margin_ICE.append(profit_margin)

        return profit_margin_ICE, profit_margin_EV
    

    def calc_last_step_profit_margin(self):
        profit_margin_ICE, profit_margin_EV = self.calc_profit_margin(self.past_new_bought_vehicles)
        all_profit_margins = profit_margin_ICE +  profit_margin_EV
        return np.mean(all_profit_margins)

    def calc_last_step_HHI(self):
        HHI = self.calculate_market_concentration(self.past_new_bought_vehicles)
        return HHI
    
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


        self.history_mean_profit_margins_EV = []
        self.history_mean_profit_margins_ICE = []

        self.history_median_profit_margins_EV = []
        self.history_median_profit_margins_ICE = []

        self.history_W = []

        self.history_quality_ICE = []
        self.history_efficiency_ICE  = []
        self.history_production_cost_ICE  = []

        self.history_quality_EV = []
        self.history_efficiency_EV = []
        self.history_production_cost_EV = []
        self.history_battery_EV = []
        self.history_prop_EV = []

        self.history_prop_EV_research = []
        self.history_prop_ICE_research = []

        self.history_past_new_bought_vehicles_prop_ev = []


    def save_timeseries_data_firm_manager(self):
        #self.history_cars_on_sale_all_firms.append(self.cars_on_sale_all_firms)
        #self.total_profit = self.calc_total_profits(self.past_new_bought_vehicles)
        if self.past_new_bought_vehicles:
            self.history_past_new_bought_vehicles_prop_ev.append(sum([1 for car in self.past_new_bought_vehicles if car.transportType == 3])/len(self.past_new_bought_vehicles))
        else:
            self.history_past_new_bought_vehicles_prop_ev.append(np.nan)

        # Extract research type history for each firm (last 12 years, or fewer if not available)
        research_history = [firm.history_research_type[-12:] for firm in self.firms_list]

        # Convert to numpy array, ensuring a consistent dtype for handling NaNs
        research_history = np.array(research_history, dtype=float)  # Convert to float to handle NaNs

        # Count valid (non-NaN) research occurrences per firm
        valid_counts = np.sum(~np.isnan(research_history), axis=1)  # Count non-NaN entries per firm

        # Count occurrences of EV (1) and ICE (0) research
        ev_counts = np.nansum(research_history == 1, axis=1)  # Count EV research (ignoring NaNs)
        ice_counts = np.nansum(research_history == 0, axis=1)  # Count ICE research (ignoring NaNs)

        # **Handle cases where valid_counts == 0**
        ev_proportion = np.zeros_like(valid_counts, dtype=float)  # Default to 0
        ice_proportion = np.zeros_like(valid_counts, dtype=float)  # Default to 0

        # Compute proportions only where valid_counts > 0
        nonzero_mask = valid_counts > 0
        ev_proportion[nonzero_mask] = ev_counts[nonzero_mask] / valid_counts[nonzero_mask]
        ice_proportion[nonzero_mask] = ice_counts[nonzero_mask] / valid_counts[nonzero_mask]

        # Compute the **overall average proportion** across all firms (ignoring empty cases)
        avg_ev_proportion = np.mean(ev_proportion)  # No need for nanmean since we ensured no NaNs
        avg_ice_proportion = np.mean(ice_proportion)
        self.history_prop_EV_research.append(avg_ev_proportion)
        self.history_prop_ICE_research.append(avg_ice_proportion)

        self.EV_users_count = sum(1 if car.transportType == 3 else 0 for car in  self.cars_on_sale_all_firms)
        self.history_prop_EV.append(self.EV_users_count/len(self.cars_on_sale_all_firms))
    
        self.HHI = self.calculate_market_concentration(self.past_new_bought_vehicles)
        profit_margin_ICE, profit_margin_EV = self.calc_profit_margin(self.past_new_bought_vehicles)

        if profit_margin_EV:
            self.history_mean_profit_margins_EV.append(np.nanmean(profit_margin_EV))
            self.history_median_profit_margins_EV.append(np.nanmedian(profit_margin_EV))
        else:
            self.history_mean_profit_margins_EV.append(np.nan)
            self.history_median_profit_margins_EV.append(np.nan)

        if profit_margin_ICE:
            self.history_mean_profit_margins_ICE.append(np.nanmean(profit_margin_ICE))
            self.history_median_profit_margins_ICE.append(np.nanmedian(profit_margin_ICE))
        else:
            self.history_mean_profit_margins_ICE.append(np.nan)
            self.history_median_profit_margins_ICE.append(np.nan)

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
    def update_firms(self, gas_price, electricity_price, electricity_emissions_intensity, rebate, production_subsidy,  rebate_calibration):
        cars_on_sale_all_firms = []
        
        self.zero_profit_options_prod_sum = 0
        self.zero_profit_options_research_sum = 0


        # 4) Now we need to compute the initial W for each segment
        for firm in self.firms_list:
            self.zero_profit_options_prod_sum += firm.zero_profit_options_prod#CAN DELETE OCNE FIXED ISSUE O uitlity in firms prod
            self.zero_profit_options_research_sum += firm.zero_profit_options_research
            cars_on_sale = firm.next_step(self.I_s_t_vec, self.W_vec, self.maxU_vec, self.carbon_price, gas_price, electricity_price, electricity_emissions_intensity, rebate, production_subsidy,  rebate_calibration)

            cars_on_sale_all_firms.extend(cars_on_sale)

        return cars_on_sale_all_firms
    
    def update_firms_burn_in(self):
        cars_on_sale_all_firms = []
        for firm in self.firms_list:
            cars_on_sale = firm.next_step_burn_in(self.I_s_t_vec, self.W_vec, self.maxU_vec)
            cars_on_sale_all_firms.extend(cars_on_sale)
        return cars_on_sale_all_firms

#####################################################################################################################

    def next_step(self, carbon_price, consider_ev_vec, new_bought_vehicles,  gas_price, electricity_price, electricity_emissions_intensity, rebate,  production_subsidy,  rebate_calibration):
        
        self.t_firm_manager += 1
        self.past_new_bought_vehicles = new_bought_vehicles
        self.total_profit = self.calc_total_profits(self.past_new_bought_vehicles, self.production_subsidy)#NEED TO CALC TOTAL PROFITS NOW before the cars on sale change?
        self.profit_cumulative += self.total_profit

        self.consider_ev_vec = consider_ev_vec#UPDATE THIS TO NEW CONSIDERATION
        self.carbon_price = carbon_price
        self.production_subsidy = production_subsidy
        
        self.cars_on_sale_all_firms  = self.update_firms(gas_price, electricity_price, electricity_emissions_intensity, rebate, production_subsidy,  rebate_calibration)#WE ASSUME THAT FIRMS DONT CONSIDER SECOND HAND MARKET
        self.W_segment, self.maxU_vec = self.update_W_immediate()#calculate the competiveness of the market current

        self.I_s_t_vec, self.W_vec = self.update_market_data_moving_average(self.W_segment)#update the rollign vlaues

        return self.cars_on_sale_all_firms
    

    def next_step_burn_in(self):
        
        self.cars_on_sale_all_firms  = self.update_firms_burn_in()#WE ASSUME THAT FIRMS DONT CONSIDER SECOND HAND MARKET
        self.W_segment, self.maxU_vec = self.update_W_immediate()#calculate the competiveness of the market current
        self.I_s_t_vec, self.W_vec = self.update_market_data_moving_average(self.W_segment)#update the rollign vlaues

    