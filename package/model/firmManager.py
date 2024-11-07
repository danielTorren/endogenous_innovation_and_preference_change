import numpy as np
import random
from package.model.carModel import CarModel
from package.model.firm import Firm
from collections import defaultdict

class Firm_Manager:
    def __init__(self, parameters_firm_manager: dict, parameters_firm: dict, parameters_car_ICE: dict, parameters_car_EV: dict, ICE_landscape: dict, EV_landscape: dict):
        self.t_firm_manager = 0

        self.parameters_firm = parameters_firm
        #print(self.parameters_firm)
        #quit()

        self.init_tech_seed = parameters_firm_manager["init_tech_seed"]
        self.J = int(round(parameters_firm_manager["J"]))
        self.N = int(round(parameters_firm_manager["N"]))

        #self.save_timeseries_data_state = parameters_firm_manager["save_timeseries_data_state"]
        #self.compression_factor_state = parameters_firm_manager["compression_factor_state"]
        self.carbon_price = parameters_firm_manager["carbon_price"]
        self.id_generator = parameters_firm_manager["IDGenerator_firms"]
        self.kappa = parameters_firm_manager["kappa"]


        #landscapes
        self.landscape_ICE = ICE_landscape
        self.landscape_EV = EV_landscape

        #car paramets
        self.parameters_car_ICE = parameters_car_ICE
        self.parameters_car_EV = parameters_car_EV 

        #PRICE, NOT SURE IF THIS IS NECESSARY
        self.beta_threshold = 0.5#parameters_firm_manager["beta_threshold"]
        self.beta_val_empty = (1+self.beta_threshold)/2
        self.gamma_threshold = 0.5#parameters_firm_manager["gamma_threshold"]
        self.gamma_val_empty = (1+self.gamma_threshold)/2

        #PRice sentitivity BETA
        self.segment_number_price_sensitivity = 2
        self.expected_segment_share_price_sensitivity = [1 / self.segment_number_price_sensitivity] * self.segment_number_price_sensitivity
        self.segment_price_sensitivity_bounds = np.linspace(0, 1, self.segment_number_price_sensitivity + 1)
        self.width_price_segment = self.segment_price_sensitivity_bounds[1] - self.segment_price_sensitivity_bounds[0]
        self.segment_price_sensitivity = np.arange(self.width_price_segment / 2, 1, self.width_price_segment)
        self.segment_price_reshaped = self.segment_price_sensitivity[:, np.newaxis]

        # ENVIRONMENTAL PREFERENCE, GAMMA
        self.segment_number_environmental_preference = 2
        self.expected_segment_share_environmental_preference = [1 / self.segment_number_environmental_preference] * self.segment_number_environmental_preference
        self.segment_environmental_preference_bounds = np.linspace(0, 1, self.segment_number_environmental_preference + 1)
        self.width_environmental_segment = self.segment_environmental_preference_bounds[1] - self.segment_environmental_preference_bounds[0]
        self.segment_environmental_preference = np.arange(self.width_environmental_segment / 2, 1, self.width_environmental_segment)
        self.segment_environmental_reshaped = self.segment_environmental_preference[:, np.newaxis]

        np.random.seed(self.init_tech_seed)
        random.seed(self.init_tech_seed)

        self.init_firms()
        
        #calculate the inital attributes of all the cars on sale
        self.cars_on_sale_all_firms = self.generate_cars_on_sale_all_firms()

    ###########################################################################################################
    #INITIALISATION

    def init_firms(self):
        """
        Generate a initial ICE and EV TECH TO START WITH 
        """
        #Pick the initial technology
        self.init_tech_component_string = f"{random.getrandbits(self.N):0{self.N}b}"#CAN USE THE SAME STRING FOR BOTH THE EV AND ICE

        #Generate the initial fitness values of the starting tecnology(ies)

        decimal_value = int(self.init_tech_component_string, 2)
        init_tech_component_string_list_N = self.invert_bits_one_at_a_time(decimal_value, len(self.init_tech_component_string))
        init_tech_component_string_list = np.random.choice(init_tech_component_string_list_N, self.J)
        
        self.init_tech_list_ICE = [CarModel(self.id_generator.get_new_id(), init_tech_component_string_list[j], self.landscape_ICE, parameters = self.parameters_car_ICE, choosen_tech_bool=1) for j in range(self.J)]
        self.init_tech_list_EV = [CarModel(self.id_generator.get_new_id(), init_tech_component_string_list[j], self.landscape_EV, parameters = self.parameters_car_EV, choosen_tech_bool=1) for j in range(self.J)]

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

    def input_social_network_data(self, beta_vector, origin_vector, environmental_awareness_vec,consider_ev_vec):
        self.beta_vec = beta_vector
        self.origin_vec = origin_vector
        self.gamma_vec = environmental_awareness_vec
        self.consider_ev_vec = consider_ev_vec

        # Convert beta and gamma arrays to binary based on threshold values
        self.beta_binary = (self.beta_vec > self.beta_threshold).astype(int)
        self.gamma_binary = (self.gamma_vec > self.gamma_threshold).astype(int)

    def generate_market_data(self):
        """Used once at the start of model run, need to generate the counts then the market data without U then calc U and add it in!"""

        segment_codes = (self.beta_binary << 3) | (self.gamma_binary << 2) | (self.consider_ev_vec << 1) | self.origin_vec
        segment_counts = np.bincount(segment_codes, minlength=16)
        # Store the counts for each of the segments as binary strings ('0000' to '1111')
        self.market_data = {format(i, '04b'): {"I_s_t":segment_counts[i]} for i in range(16)}

        #DEAL WITH BETA AND GAMMA
        # Iterate over all possible segments (0 to 15)
        for segment_code in self.market_data.keys():
            # Identify the indices of individuals belonging to the current segment
            indices = np.where(segment_codes == segment_code)[0]

            # If there are individuals in the segment, calculate average beta and gamma
            if len(indices) > 0:
                avg_beta = np.mean(self.beta_vec[indices])
                avg_gamma = np.mean(self.gamma_vec[indices])
            else:
                avg_beta = self.beta_val_empty
                avg_gamma = self.gamma_val_empty

            # Add data for the segment
            self.market_data[segment_code]["beta_s_t"] =  avg_beta
            self.market_data[segment_code]["gamma_s_t"] =  avg_gamma

        #NEED TO HAVE MARKET DATA WITHOUT THE U SUM IN ORDER TO CALCULATE U SUM
        #CALC THE U SUM DATA
        for firm in self.firms_list:#CREATE THE UTILITY SEGMENT DATA
            firm.calc_utility_cars_segments(self.market_data, self.cars_on_sale_all_firms)

        segment_U_sums = defaultdict(float)#GET SUM U NOW
        for firm in self.firms_list:
            for car in firm.cars_on_sale:
                for segment, U in car.car_utility_segments_U.items():
                    segment_U_sums[segment] += (U ** self.kappa)

        #ADD IN THE U SUM DATA
        for segment_code in self.market_data.keys():
            self.market_data[segment_code]["sum_U_kappa"] =  segment_U_sums[segment_code]
        
    ############################################################################################################################################################
    #GENERATE MARKET DATA DYNAMIC

    def generate_cars_on_sale_all_firms_and_sum_U(self, market_data):
        cars_on_sale_all_firms = []
        segment_U_sums = defaultdict(float)
        for firm in self.firms_list:
            cars_on_sale = firm.next_step(market_data)

            cars_on_sale_all_firms.extend(cars_on_sale)
            for car in cars_on_sale:
                for segment, U in car.car_utility_segments_U.items():
                    segment_U_sums[segment] += U ** self.kappa

        return cars_on_sale_all_firms, segment_U_sums

    def update_market_data(self, sums_U_segment):
        """Update market data with segment counts and sums U for each segment"""

        # Calculate segment codes based on the provided binary vectors
        segment_codes = (self.beta_binary << 3) | (self.gamma_binary << 2) | (self.consider_ev_vec << 1) | self.origin_vec
        
        # Calculate segment counts
        segment_counts = np.bincount(segment_codes, minlength=16)
        
        # Directly update the market data without the staging dictionary
        for i in range(16):
            segment_code = format(i, '04b')
            self.market_data[segment_code]["I_s_t"] = segment_counts[i]
            self.market_data[segment_code]["sum_U_kappa"] = sums_U_segment[segment_code]

    ######################################################################################################################

    def calc_total_profits(self, past_chosen_vehicles):

        total_profit_all_firms = 0         
        for car in self.cars_on_sale_all_firms:
            num_vehicle_sold = past_chosen_vehicles.count(car)
            #print(num_vehicle_sold)
            profit = (car.price - car.c_z_t)
            total_profit = num_vehicle_sold*profit
            #print(total_profit)
            #car.total_profit = total_profit
            car.firm.firm_profit += total_profit#I HAVE NO IDEA IF THIS WILL WORK
            total_profit_all_firms += total_profit
        #quit()
        return total_profit_all_firms

    def calculate_market_share(self, firm, past_chosen_vehicles, total_sales):
        """
        Calculate market share for a specific firm.
        Parameters:
            firm : Firm - The firm object for which to calculate market share
            past_chosen_vehicles : list - List of car objects representing vehicles sold in the past period
        Returns:
            MS_firm : float - Market share for the specified firm, 0 if no cars sold
        """
        # Calculate total sales for the specified firm by summing prices of cars sold by this firm
        firm_sales = sum(car.price for car in past_chosen_vehicles if car.firm == firm)
        
        # If total_sales is zero, return 0 to avoid division by zero; otherwise, calculate firm market share
        MS_firm = firm_sales / total_sales if total_sales > 0 else 0
        return MS_firm


    def calculate_market_concentration(self, past_chosen_vehicles):
        """
        Calculate market concentration (HHI) based on market shares of all firms.
        Parameters:
            past_chosen_vehicles : list - List of car objects representing vehicles sold in the past period
        Returns:
            HHI : float - Market concentration (HHI)
        """
        # Calculate total market sales by summing prices of all cars sold in the market
        total_sales = sum(car.price for car in past_chosen_vehicles)

        # Calculate the HHI by summing the squares of market shares for each firm in firm_list
        HHI = sum(self.calculate_market_share(firm, past_chosen_vehicles, total_sales)**2 for firm in self.firms_list)
        return HHI

    def calc_vehicles_chosen_list(self, past_chosen_vehicles):
        for firm in self.firms_list:
            firm.firm_cars_users = sum(1 for car in past_chosen_vehicles if car.firm == firm)

    #####################################################################################################################

    def set_up_time_series_firm_manager(self):
        #self.history_cars_on_sale_all_firms = [self.cars_on_sale_all_firms]
        self.history_total_profit = []
        self.history_market_concentration = []

    def save_timeseries_data_firm_manager(self):
        #self.history_cars_on_sale_all_firms.append(self.cars_on_sale_all_firms)
        total_profit = self.calc_total_profits(self.past_chosen_vehicles)
        HHI = self. calculate_market_concentration(self.past_chosen_vehicles)
        self.calc_vehicles_chosen_list(self.past_chosen_vehicles)

        self.history_total_profit.append(total_profit)
        self.history_market_concentration.append(HHI)

    def next_step(self, carbon_price, consider_ev_vec, chosen_vehicles):
        
        self.t_firm_manager += 1

        self.past_chosen_vehicles =  chosen_vehicles
         
        self.carbon_price = carbon_price

        self.cars_on_sale_all_firms, sums_U_segment = self.generate_cars_on_sale_all_firms_and_sum_U(self.market_data)#WE ASSUME THAT FIRMS DONT CONSIDER SECOND HAND MARKET OR PUBLIC TRANSPORT
    
        self.consider_ev_vec = consider_ev_vec#UPDATE THIS TO NEW CONSIDERATION
        self.update_market_data(sums_U_segment)

        return self.cars_on_sale_all_firms