import copy
import numpy as np
from package.model.carModel import CarModel
from package.model.personalCar import PersonalCar
from package.model.firm import Firm
from collections import defaultdict

class Firm_Manager:
    def __init__(self, parameters_firm_manager: dict, parameters_firm: dict, parameters_car_ICE: dict, parameters_car_EV: dict, ICE_landscape: dict, EV_landscape: dict):
        self.t_firm_manager = 0

        self.total_U_sum = 0
        self.parameters_firm = parameters_firm

        self.init_tech_seed = parameters_firm_manager["init_tech_seed"]
        self.J = int(round(parameters_firm_manager["J"]))
        self.N = int(round(parameters_firm_manager["N"]))

        #self.save_timeseries_data_state = parameters_firm_manager["save_timeseries_data_state"]
        #self.compression_factor_state = parameters_firm_manager["compression_factor_state"]
        self.carbon_price = parameters_firm_manager["carbon_price"]
        self.id_generator = parameters_firm_manager["IDGenerator_firms"]
        self.kappa = parameters_firm_manager["kappa"]
        self.num_individuals = parameters_firm_manager["num_individuals"]

        #landscapes
        self.landscape_ICE = ICE_landscape
        self.landscape_EV = EV_landscape

        #car paramets
        self.parameters_car_ICE = parameters_car_ICE
        self.parameters_car_EV = parameters_car_EV 

        #PRICE, NOT SURE IF THIS IS NECESSARY
        self.beta_threshold = 0.5#parameters_firm_manager["beta_threshold"]
        self.beta_val_empty_upper = (1+self.beta_threshold)/2
        self.beta_val_empty_lower = (1-self.beta_threshold)/2
        
        self.gamma_threshold = 0.5#parameters_firm_manager["gamma_threshold"]
        self.gamma_val_empty_upper = (1+self.gamma_threshold)/2
        self.gamma_val_empty_lower = (1-self.gamma_threshold)/2

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

    def input_social_network_data(self, beta_vec, environmental_awareness_vec,consider_ev_vec):
        self.beta_vec = beta_vec
        self.gamma_vec = environmental_awareness_vec
        self.consider_ev_vec = consider_ev_vec

        # Convert beta and gamma arrays to binary based on threshold values
        self.beta_binary = (self.beta_vec > self.beta_threshold).astype(int)
        self.gamma_binary = (self.gamma_vec > self.gamma_threshold).astype(int)

    def generate_market_data(self):
        """Used once at the start of model run, need to generate the counts then the market data without U then calc U and add it in!"""

        segment_codes = (self.beta_binary << 2) | (self.gamma_binary << 1) | (self.consider_ev_vec << 0)

        segment_counts = np.bincount(segment_codes, minlength=8)
        # Store the counts for each of the segments as binary strings ('0000' to '1111')
        self.market_data = {format(i, '03b'): {"I_s_t":segment_counts[i]} for i in range(8)}

        #DEAL WITH BETA AND GAMMA
        # Iterate over all possible segments (0 to 7)
        for segment_code in self.market_data.keys():
            # Identify the indices of individuals belonging to the current segment
            indices = np.where(segment_codes == segment_code)[0]

            # If there are individuals in the segment, calculate average beta and gamma
            if len(indices) > 0:
                avg_beta = np.mean(self.beta_vec[indices])
                avg_gamma = np.mean(self.gamma_vec[indices])
            else:
                if segment_code[0] == str(1):
                    avg_beta = self.beta_val_empty_upper
                else:
                    avg_beta = self.beta_val_empty_lower
                    
                if segment_code[1]  == str(1):
                    avg_gamma = self.gamma_val_empty_upper
                else:
                    avg_gamma = self.gamma_val_empty_lower

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
                    segment_U_sums[segment] += U 

        #ADD IN THE U SUM DATA
        for segment_code in self.market_data.keys():
            self.market_data[segment_code]["U_sum"] =  segment_U_sums[segment_code]
        
    ############################################################################################################################################################
    #GENERATE MARKET DATA DYNAMIC

    def generate_cars_on_sale_all_firms_and_sum_U(self, market_data, gas_price, electricity_price, electricity_emissions_intensity, rebate):
        cars_on_sale_all_firms = []
        segment_U_sums = defaultdict(float)

        #print("In firm manager data: ", market_data, self.carbon_price, gas_price, electricity_price, electricity_emissions_intensity, rebate)
        for firm in self.firms_list:
            cars_on_sale = firm.next_step(market_data, self.carbon_price, gas_price, electricity_price, electricity_emissions_intensity, rebate)

            cars_on_sale_all_firms.extend(cars_on_sale)
            for car in cars_on_sale:
                for segment, U in car.car_utility_segments_U.items():
                    segment_U_sums[segment] += U
        #print("segment_U_sums", segment_U_sums)
        return cars_on_sale_all_firms, segment_U_sums

    def update_market_data(self, sums_U_segment):
        """Update market data with segment counts and sums U for each segment"""

        # Calculate segment codes based on the provided binary vecs
        segment_codes = (self.beta_binary << 2) | (self.gamma_binary << 1) | (self.consider_ev_vec << 0)
        # Calculate segment counts
        segment_counts = np.bincount(segment_codes, minlength=8)

        for i in range(8):
            segment_code = format(i, '03b')
            
            """
            # Determine if the segment considers ICE cars
            if segment_code[2] == '0':  
                # Flip the bit at position 2 (count from left, 0-indexed)
                i_flipped = i ^ (1 << 2)  # Flip the 2nd bit from the right
                segment_code_flipped = format(i_flipped, '03b')

                # Update market data
                self.market_data[segment_code]["I_s_t"] = segment_counts[i] + segment_counts[i_flipped]
                self.market_data[segment_code]["U_sum"] = sums_U_segment[segment_code] + sums_U_segment[segment_code_flipped]
            else:
                self.market_data[segment_code]["I_s_t"] = segment_counts[i]
                self.market_data[segment_code]["U_sum"] = sums_U_segment[segment_code]
            """
            self.market_data[segment_code]["I_s_t"] = segment_counts[i]
            self.market_data[segment_code]["U_sum"] = sums_U_segment[segment_code]
            self.total_U_sum += sums_U_segment[segment_code]
            
            #print("segemtn count",segment_code,  self.market_data[segment_code]["I_s_t"])
        #quit()

    ######################################################################################################################

    def calc_total_profits(self, past_chosen_vehicles):

        total_profit_all_firms = 0         
        for car in self.cars_on_sale_all_firms:
            num_vehicle_sold = past_chosen_vehicles.count(car)

            profit = (car.price - car.ProdCost_t)
            total_profit = num_vehicle_sold*profit
            car.firm.firm_profit += total_profit#I HAVE NO IDEA IF THIS WILL WORK
            total_profit_all_firms += total_profit

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
        self.history_segment_count = []
        self.history_cars_on_sale_EV_prop = []
        self.history_cars_on_sale_ICE_prop = []
        self.history_cars_on_sale_price = []

        self.history_market_data = []

    def save_timeseries_data_firm_manager(self):
        #self.history_cars_on_sale_all_firms.append(self.cars_on_sale_all_firms)
        self.total_profit = self.calc_total_profits(self.past_chosen_vehicles)
        self.HHI = self. calculate_market_concentration(self.past_chosen_vehicles)
        self.calc_vehicles_chosen_list(self.past_chosen_vehicles)
        self.history_cars_on_sale_price.append([car.price for car in self.cars_on_sale_all_firms])

        self.history_total_profit.append(self.total_profit)
        self.history_market_concentration.append(self.HHI)
        self.history_segment_count.append([segment_data["I_s_t"] for segment_data in self.market_data.values()])

        count_transport_type_2 = sum(1 for car in self.cars_on_sale_all_firms if car.transportType == 2)
        count_transport_type_3 = sum(1 for car in self.cars_on_sale_all_firms if car.transportType == 3)

        self.history_cars_on_sale_ICE_prop.append(count_transport_type_2)
        self.history_cars_on_sale_EV_prop.append(count_transport_type_3)

        self.history_market_data.append(copy.deepcopy(self.market_data))


    def next_step(self, carbon_price, consider_ev_vec, chosen_vehicles,  gas_price, electricity_price, electricity_emissions_intensity, rebate):
        
        self.t_firm_manager += 1

        self.total_U_sum = 0
        
        self.past_chosen_vehicles =  chosen_vehicles
         
        self.carbon_price = carbon_price

        self.cars_on_sale_all_firms, sums_U_segment = self.generate_cars_on_sale_all_firms_and_sum_U(self.market_data, gas_price, electricity_price, electricity_emissions_intensity, rebate)#WE ASSUME THAT FIRMS DONT CONSIDER SECOND HAND MARKET
    
        self.consider_ev_vec = consider_ev_vec#UPDATE THIS TO NEW CONSIDERATION
        self.update_market_data(sums_U_segment)

        return self.cars_on_sale_all_firms, self.total_U_sum