import numpy as np
import random
from package.model import carModel
from package.model.carModel import CarModel
from package.model.firm import Firm
from collections import defaultdict

class Firm_Manager:
    def __init__(self, parameters_firm_manager: dict, parameters_firm: dict):
        self.t_firm_manager = 0

        self.landscape_seed = parameters_firm_manager["landscape_seed"]
        self.init_tech_seed = parameters_firm_manager["init_tech_seed"]
        self.J = int(round(parameters_firm_manager["J"]))
        self.N = int(round(parameters_firm_manager["N"]))
        self.K = int(round(parameters_firm_manager["K"]))
        self.A = parameters_firm_manager["A"]
        self.rho = parameters_firm_manager["rho"]
        self.save_timeseries_data_state = parameters_firm_manager["save_timeseries_data_state"]
        self.compression_factor_state = parameters_firm_manager["compression_factor_state"]
        self.init_tech_heterogenous_state = parameters_firm_manager["init_tech_heterogenous_state"]
        self.carbon_price = parameters_firm_manager["carbon_price"]
        self.num_individuals = parameters_firm_manager["num_individuals"]
        self.id_generator = parameters_firm_manager["IDGenerator_firms"]
        self.gamma = parameters_firm_manager["gamma"]
        self.kappa = parameters_firm_manager["kappa"]
        self.markup = parameters_firm_manager["markup"]
        self.rank_number = parameters_firm_manager['rank_number']
        self.memory_cap = parameters_firm_manager["memory_cap"]
        self.static_tech_state = parameters_firm_manager["static_tech_state"]
        self.utility_boost_const = parameters_firm_manager["utility_boost_const"]
        self.price_constant = parameters_firm_manager["price_constant"]

        self.beta_threshold = parameters_firm_manager["beta_threshold"]
        self.beta_val_empty = (1+self.beta_threshold)/2
        self.gamma_threshold = parameters_firm_manager["gamma_threshold"]
        self.gamma_val_empty = (1+self.gamma_threshold)/2

        #landscapes
        self.landscape_ICE = parameters_firm_manager["landscape_ICE"]
        self.landscape_EV = parameters_firm_manager["lanscape_EV"]

        #PRICE, NOT SURE IF THIS IS NECESSARY
        self.segment_number_price_sensitivity = 2
        self.expected_segment_share_price_sensitivity = [1 / self.segment_number_price_sensitivity] * self.segment_number_price_sensitivity
        self.segment_price_sensitivity_bounds = np.linspace(0, 1, self.segment_number_price_sensitivity + 1)
        self.width_price_segment = self.segment_price_sensitivity_bounds[1] - self.segment_price_sensitivity_bounds[0]
        self.segment_price_sensitivity = np.arange(self.width_price_segment / 2, 1, self.width_price_segment)
        self.segment_price_reshaped = self.segment_price_sensitivity[:, np.newaxis]

        # ENVIRONMENTAL PREFERENCE
        self.segment_number_environmental_preference = 2
        self.expected_segment_share_environmental_preference = [1 / self.segment_number_environmental_preference] * self.segment_number_environmental_preference
        self.segment_environmental_preference_bounds = np.linspace(0, 1, self.segment_number_environmental_preference + 1)
        self.width_environmental_segment = self.segment_environmental_preference_bounds[1] - self.segment_environmental_preference_bounds[0]
        self.segment_environmental_preference = np.arange(self.width_environmental_segment / 2, 1, self.width_environmental_segment)
        self.segment_environmental_reshaped = self.segment_environmental_preference[:, np.newaxis]

        np.random.seed(self.init_tech_seed)
        random.seed(self.init_tech_seed)

        self.parameters_firm = parameters_firm
        self.parameters_car_ICE, self.parameters_car_EV = self.generate_parameters_firm(self.parameters_firm)
        self.firms_list = self.init_firms()
        #calculate the inital attributes of all the cars on sale
        self.cars_on_sale_all_firms = self.generate_cars_on_sale_all_firms()
        
        if self.save_timeseries_data_state:
            self.set_up_time_series_firm_manager()

    ###########################################################################################################

    def generate_parameters_firm(self, parameters_firm):
        """Need to add in all the parameters required for the firms"""
        parameters_car_ICE = {}
        parameters_car_EV = {}

        parameters_car_ICE["transportType"] = 2
        parameters_car_ICE["nk_landscape"] = self.landscape_ICE
        parameters_car_ICE["delta_z"] = parameters_firm["delta_ICE"]
        parameters_car_ICE["e_z_t"] = parameters_firm["e_ICE"]  
        parameters_car_ICE["nu_z_i_t"] = parameters_firm["nu_ICE"]  
        parameters_car_ICE["emissions"] = parameters_firm["emissions_ICE"] 
        parameters_car_ICE["eta"] = parameters_firm["eta"] 

        parameters_car_EV["transportType"] = 3
        parameters_car_EV["nk_landscape"] = self.landscape_EV
        parameters_car_EV["delta_z"] = parameters_firm["delta_EV"]
        parameters_car_EV["e_z_t"] = parameters_firm["e_EV"]  
        parameters_car_EV["nu_z_i_t"] = parameters_firm["nu_EV"]  
        parameters_car_EV["emissions"] = parameters_firm["emissions_EV"]
        parameters_car_EV["eta"] = parameters_firm["eta"] 

        return parameters_car_ICE, parameters_car_EV
    
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
        
        self.init_tech_list_ICE = [carModel(self.id_generator.get_new_id(), j, init_tech_component_string_list[j], parameters =self.parameters_car_ICE, choosen_tech_bool=1) for j in range(self.J)]
        self.init_tech_list_EV = [carModel(self.id_generator.get_new_id(), j, init_tech_component_string_list[j], parameters =self.parameters_car_EV, choosen_tech_bool=1) for j in range(self.J)]

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
    
    def set_up_time_series_firm_manager(self):
        self.history_cars_on_sale_all_firms = [self.cars_on_sale_all_firms]

    def save_timeseries_data_firm_manager(self):
        self.history_cars_on_sale_all_firms.append(self.cars_on_sale_all_firms)
    
    def generate_cars_on_sale_all_firms(self):
        """ONLY USED ONCE AT INITITALISATION"""
        cars_on_sale_all_firms = []
        for firm in self.firms_list:
            cars_on_sale_all_firms.extend(firm.cars_on_sale)

    def generate_cars_on_sale_all_firms_and_sum_U(self, market_data):
        cars_on_sale_all_firms = []
        segment_U_sums = defaultdict(float)
        for firm in self.firms_list:
            cars_on_sale = firm.next_step(market_data)
            cars_on_sale_all_firms.extend(cars_on_sale)
            for car in cars_on_sale:
                for segment, U in car.car_utility_segments_U.items():
                    segment_U_sums[segment] += U ** self.kappa

        return np.asarray(cars_on_sale_all_firms), segment_U_sums

    def input_social_network_data(self, beta_vector, origin_vector, environmental_awareness_vec,adoption_state_vec):
        self.beta_vec = beta_vector
        self.origin_vec = origin_vector
        self.gamma_vec = environmental_awareness_vec
        self.adoption_state_vec = adoption_state_vec

    def generate_market_data_init(self):
        """Used once at the start of model run, need to generate the counts then the market data without U then calc U and add it in!"""

        self.update_segment_count_intersections(self, self.beta_vec, self.gamma_vec, self.adoption_state_vec, self.origin_vec)

        # Dictionary to hold market data for each segment
        market_data = {}

        # Iterate over all possible segments (0 to 15)
        for segment_code in range(16):
            # Binary representation of the segment code (4-bit string)
            segment_code_str = format(segment_code, '04b')

            # Identify the indices of individuals belonging to the current segment
            indices = np.where(segment_codes == segment_code)[0]

            # If there are individuals in the segment, calculate average beta and gamma
            if len(indices) > 0:
                avg_beta = np.mean(beta_vec[indices])
                avg_gamma = np.mean(gamma_vec[indices])
            else:
                avg_beta = self.beta_val_empty
                avg_gamma = self.gamma_val_empty

            # Add data for the segment
            market_data[segment_code_str] = {
                "I_s_t": segment_counts[segment_code_str],
                "beta_s_t": avg_beta,
                "gamma_s_t": avg_gamma,
            }

        for firm in self.firms_list:#CREATE THE UTILITY SEGMENT DATA
            firm.calc_utility_cars_segments(self, market_data, self.cars_on_sale_all_firms)

        segment_U_sums = defaultdict(float)#GET SUM U NOW
        for firm in self.firms_list:
            for car in firm.cars_on_sale:
                for segment, U in car.car_utility_segments_U.items():
                    segment_U_sums[segment] += U ** self.kappa

        market_data_full = self.generate_market_data(self, segment_codes, segment_counts, beta_vec, gamma_vec, segment_U_sums)
        
        return market_data_full
    
    ############################################################################################################################################################
    #GENERATE MARKET DATA

    def commuting_utility(self, vehicle, d_i_t, z):
        """
        Calculate the commuting utility based on different conditions.

        Parameters:
        vehicle (Vehicle): The vehicle being considered for commuting.
        d_i_t (float): Distance traveled during a time-step.
        z (float): Condition parameter to differentiate public and private.

        Returns:
        float: The calculated commuting utility u_{a,i,t}.
        """
        Q_a_t = vehicle.Q_a_t
        delta_z = vehicle.delta_z
        L_a_t = vehicle.L_a_t
        omega_a_t = vehicle.omega_a_t
        c_z_t = vehicle.c_z_t
        e_z_t = vehicle.e_z_t
        nu_z_i_t = vehicle.nu_z_i_t

        # Calculate commuting utility based on conditions for z
        if z > 1:
            # If z' > 1, include all cost components
            cost_component = self.beta * (1 / omega_a_t) * c_z_t + self.gamma * (1 / omega_a_t) * e_z_t + self.eta * nu_z_i_t
            utility = Q_a_t * (1 - delta_z) ** L_a_t * (d_i_t ** self.alpha) - d_i_t * cost_component
        else:
            # If z' <= 1, include only the eta * nu component
            utility = Q_a_t * (1 - delta_z) ** L_a_t * (d_i_t ** self.alpha) - d_i_t * (self.eta * nu_z_i_t)

        # Ensure utility is non-negative
        utility = max(0, utility)

        return utility

    def optimal_distance(self, vehicle):
        """
        Calculate the optimal distance based on the vehicle properties.

        Parameters:
        vehicle (Vehicle): The vehicle for which the optimal distance is calculated.

        Returns:
        float: The calculated optimal distance, d^*_{a,i,t}.
        """
        numerator = self.alpha * vehicle.Q_a_t * (1 - vehicle.delta_z) ** vehicle.L_a_t
        denominator = (self.beta * vehicle.omega_a_t ** -1 * vehicle.c_z_t +
                       self.gamma * vehicle.omega_a_t ** -1 * vehicle.e_z_t +
                       vehicle.eta * vehicle.nu_z_i_t)

        # Compute optimal distance
        if denominator == 0:
            raise ValueError("The denominator is zero, adjust the parameters to avoid division by zero.")

        optimal_d = (numerator / denominator) ** (1 / (1 - self.alpha))
        return optimal_d

    def calculate_utility(self, vehicle):
        """
        Calculate the lifetime utility using the closed-form solution based on different conditions.

        Parameters:
        vehicle (Vehicle): The vehicle for which the utility is being calculated.
        scenario (str): The scenario to determine how the utility is adjusted.

        Returns:
        float: The calculated lifetime utility U_{a,i,t}.
        """

        scenario = vehicle.scenario #is it second hand, first hand or public transport.

        # Calculate distance and commuting utility
        d_i_t = self.optimal_distance(vehicle)
        commuting_util = self.commuting_utility(vehicle, d_i_t, z=2)  # Example z value (should be scenario-specific)

        # Closed-form solution for lifetime utility
        denominator = self.r + (1 - self.delta) / (1 - self.alpha)
        if denominator == 0:
            raise ValueError("The denominator is zero, adjust the parameters to avoid division by zero.")
        
        # Calculate the base lifetime utility using the closed form
        base_utility = commuting_util / denominator

        """
        Cases:
        1. buy brand new car and you have no old car (could be literally no car or that you use public transport)
        2. buy brand new car and you have an old car which you sell to the second hand man
        3. buy second hand car and you have no old car (could be literally no car or that you use public transport)
        4. buy second hand car and you have old car which you sell to the second hand man
        5. you choose public transport and you have old car which you sell to the second hand man
        6. you choose public tranpsort and you have no old car (could be literally no car or that you use public transport)
        7. you own car and you keep the same car
        """
        
        # Adjust the lifetime utility based on the scenario
        if self.vehicle.transportType > 0:#YOU OWN A CAR
            if scenario == "current_car":#CASE 7
                U_a_i_t = base_utility
            elif scenario == ("public_optional" or "private_unassigned"):# CASE 4 and 5, PUBLIC TRANSPORT or second hand car, whilst owning a second hand car!
                U_a_i_t = base_utility - self.beta * (vehicle.price -  self.vehicle.price/(1+self.mu))
            elif scenario == "private_emissions":#CASE 2, you buy a new car and you own one
                U_a_i_t = base_utility - self.beta * (vehicle.price - self.vehicle.price/(1+self.mu)) - self.gamma * vehicle.emissions
            else:
                raise ValueError("Invalid scenario specified. Owns second hand car")
        else:#you dont own a car!
            if scenario == ("public_optional" or "private_unassigned"):#Cases 3 and 6, choosing PUBLIC TRANSPORT or second hand car without owning a second hand car due to public tranport or no car
                U_a_i_t = base_utility - self.beta * vehicle.price
            elif scenario == "private_emissions":#CASE 1, buyign new without owning a second hand car
                U_a_i_t = base_utility - self.beta * vehicle.price - self.gamma * vehicle.emissions
            else:
                raise ValueError("Invalid scenario specified. No car is owned")

        return U_a_i_t

    def update_segment_count_intersections(self, beta_vec, gamma_vec, adoption_state_vec, origin_vec):
        """
        Updates the count of people in each segment for the intersections of `beta`, `gamma`, `adoption_state`, and `origin`.
        The segments are defined based on threshold values for `beta` and `gamma`.

        Parameters:
        - beta_vec: array of `beta` values (between 0 and 1)
        - gamma_vec: array of `gamma` values (between 0 and 1)
        - adoption_state_vec: array of boolean values (0 or 1) for adoption state
        - origin_vec: array of boolean values (0 or 1) for origin
        - beta_threshold: threshold for categorizing `beta` as low or high (default 0.5)
        - gamma_threshold: threshold for categorizing `gamma` as low or high (default 0.5)
        """
        
        # Convert beta and gamma arrays to binary based on threshold values
        beta_binary = (beta_vec > self.beta_threshold).astype(int)
        gamma_binary = (gamma_vec > self.gamma_threshold).astype(int)

        # Create a combined code to represent each segment uniquely
        # The code is a binary representation of the four categories (beta, gamma, adoption state, origin)
        segment_codes = (beta_binary << 3) | (gamma_binary << 2) | (adoption_state_vec << 1) | origin_vec

        # Calculate the count for each of the 16 segments (0 to 15)
        segment_counts = np.bincount(segment_codes, minlength=16)

        # Store the counts for each of the segments as binary strings ('0000' to '1111')
        segment_counts_dict = {format(i, '04b'): segment_counts[i] for i in range(16)}

        return segment_codes, segment_counts_dict, beta_binary, gamma_binary


    def generate_market_data(self, segment_codes, segment_counts, beta_vec, gamma_vec, sums_U_segment):
        """
        Generate market data containing the number of people in each segment and the beta and gamma for those segments.
        The keys of the returned dictionary are segment codes (as binary strings), and values are dictionaries containing:
        - "I_s_t": Population of the segment.
        - "beta_s_t": Average beta value of the segment.
        - "gamma_s_t": Average gamma value of the segment.
        """

        """I NEED TO GENEATE HERE sum_U_kappa and add one in persegment! i think i dont have to do the calcualtion again if i can feed back somethign form firms and chosen cars"""

        # Dictionary to hold market data for each segment
        market_data = {}

        # Iterate over all possible segments (0 to 15)
        for segment_code in range(16):
            # Binary representation of the segment code (4-bit string)
            segment_code_str = format(segment_code, '04b')

            # Identify the indices of individuals belonging to the current segment
            indices = np.where(segment_codes == segment_code)[0]

            # If there are individuals in the segment, calculate average beta and gamma
            if len(indices) > 0:
                avg_beta = np.mean(beta_vec[indices])
                avg_gamma = np.mean(gamma_vec[indices])
            else:
                avg_beta = self.beta_val_empty
                avg_gamma = self.gamma_val_empty

            # Add data for the segment
            market_data[segment_code_str] = {
                "I_s_t": segment_counts[segment_code_str],
                "beta_s_t": avg_beta,
                "gamma_s_t": avg_gamma,
                "sum_U_kappa": sums_U_segment
            }

        return market_data

    def next_step(self, carbon_price, ev_adoption_state_vec):
        
        self.t_firm_manager += 1

        self.carbon_price = carbon_price

        segment_codes, segment_counts_dict, beta_binary, gamma_binary = self.update_segment_count_intersections(self, self.beta_vec, self.gamma_vec, ev_adoption_state_vec, self.origin_vec)

        self.cars_on_sale_all_firms, sums_U_segment = self.generate_cars_on_sale_all_firms_and_sum_U(market_data)#WE ASSUME THAT FIRMS DONT CONSIDER SECOND HAND MARKET OR PUBLIC TRANSPORT
        
        market_data = self.generate_market_data(segment_codes, segment_counts_dict, beta_binary, gamma_binary, sums_U_segment)

        if self.save_timeseries_data_state and (self.t_firm_manager % self.compression_factor_state == 0):
            self.save_timeseries_data_firm_manager()

        return self.cars_on_sale_all_firms