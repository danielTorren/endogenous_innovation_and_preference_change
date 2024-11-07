from networkx import general_random_intersection_graph
import numpy as np
from package.model.carModel import CarModel

class Firm:
    def __init__(self, firm_id, init_tech_ICE, init_tech_EV, parameters_firm, parameters_car_ICE, parameters_car_EV):
        
        self.t_firm = 0
        self.save_timeseries_data_state = parameters_firm["save_timeseries_data_state"]
        self.compression_factor_state = parameters_firm["compression_factor_state"]

        self.firm_id = firm_id
        #ICE
        self.init_tech_ICE = init_tech_ICE
        self.init_tech_ICE.firm = self
        self.list_technology_memory_ICE = [init_tech_ICE]

        self.last_researched_car_ICE = self.init_tech_ICE
        #EV
        self.init_tech_EV = init_tech_EV
        self.init_tech_EV.firm = self
        self.list_technology_memory_EV = [init_tech_EV]
        self.last_researched_car_EV = self.init_tech_EV

        self.firm_profit = 0
        self.firm_cars_users = 0

        self.list_technology_memory = self.list_technology_memory_ICE + self.list_technology_memory_EV 

        self.parameters_firm = parameters_firm
        self.eta = parameters_firm["eta"]
        self.alpha = parameters_firm["alpha"]
        self.kappa = self.parameters_firm["kappa"]
        self.memory_cap = self.parameters_firm["memory_cap"]
        self.prob_innovate = self.parameters_firm["prob_innovate"]
        self.r = self.parameters_firm["r"]
        self.id_generator = parameters_firm["IDGenerator_firms"]
        self.lambda_pow = parameters_firm["lambda_pow"]

        self.ICE_landscape = self.parameters_firm["ICE_landscape"]
        self.EV_landscape = self.parameters_firm["EV_landscape"]

        self.parameters_car_ICE = parameters_car_ICE
        self.parameters_car_EV = parameters_car_EV

        self.expected_profits_segments = {}

        #ALL TYPES
        self.cars_on_sale = [init_tech_ICE, init_tech_EV]
        self.set_car_init_price_and_U()

        if self.save_timeseries_data_state:
            self.set_up_time_series_firm()

    def set_car_init_price_and_U(self):
        for car in self.cars_on_sale:
            car.price = self.parameters_firm["init_price"]
            for segment_code in range(16):
                # Binary representation of the segment code (4-bit string)
                segment_code_str = format(segment_code, '04b')
                # Add data for the segment
                car.optimal_price_segments[segment_code_str] = self.parameters_firm["init_price"]
                car.car_utility_segments[segment_code_str] = 0
    
    def optimal_distance(self, vehicle, beta, gamma):
        """
        Calculate the optimal distance based on the vehicle properties.

        Parameters:
        vehicle (Vehicle): The vehicle for which the optimal distance is calculated.

        Returns:
        float: The calculated optimal distance, d^*_{a,i,t}.
        """
        numerator = self.alpha * vehicle.Q_a_t * (1 - vehicle.delta_z) ** vehicle.L_a_t
        denominator = (beta * vehicle.omega_a_t ** -1 * vehicle.c_z_t +
                       gamma * vehicle.omega_a_t ** -1 * vehicle.e_z_t +
                       vehicle.eta * vehicle.nu_z_i_t)

        # Compute optimal distance
        if denominator == 0:
            raise ValueError("The denominator is zero, adjust the parameters to avoid division by zero.")

        optimal_d = (numerator / denominator) ** (1 / (1 - self.alpha))
        
        return optimal_d

    def calc_commuting_utility(self, vehicle, d_i_t, beta_s, gamma_s):
        """
        Calculate the commuting utility based on different conditions.

        Parameters:
        vehicle (Vehicle): The vehicle being considered for commuting.
        d_i_t (float): Distance traveled during a time-step.
        beta_s (float): Price sensitivity parameter for the segment.
        gamma_s (float): Environmental sensitivity parameter for the segment.

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
        cost_component = beta_s * (1 / omega_a_t) * c_z_t + gamma_s * (1 / omega_a_t) * e_z_t + self.eta * nu_z_i_t
        utility = Q_a_t * (1 - delta_z) ** L_a_t * (d_i_t ** self.alpha) - d_i_t * cost_component

        # Ensure utility is non-negative
        utility = max(0, utility)

        return utility


    def calc_optimal_price_cars(self, market_data, car_list): 
        """Calculate the optimal price for each car in the car list based on market data."""

        for car in car_list:
            # Iterate over each market segment to calculate utilities and distances
            for segment_code, segment_data in market_data.items():
                beta_s = segment_data["beta_s_t"]
                gamma_s = segment_data["gamma_s_t"]

                E_m = car.emissions  # Emissions for the current car
                C_m = car.c_z_t       # Cost for the current car

                # Calculate optimal distance for the given segment
                d_i_t = self.optimal_distance(car, beta_s, gamma_s)
                car.car_distance_segments[segment_code] = d_i_t  # Save the calculated distance

                # Calculate the utility for the given segment
                utility_segment = self.calc_commuting_utility(car, d_i_t, beta_s, gamma_s)
                car.car_utility_segments[segment_code] = utility_segment  # Save the utility

                # Set the calculated optimal price for the car
                car.optimal_price_segments[segment_code] = (utility_segment - gamma_s * E_m + beta_s * C_m) / (2 * beta_s)

        return car_list

    def calculate_utility(self, vehicle, commuting_util, beta, gamma, price):
        """
        Calculate the lifetime utility using the closed-form solution based on different conditions.

        Parameters:
        vehicle (Vehicle): The vehicle for which the utility is being calculated.
        scenario (str): The scenario to determine how the utility is adjusted.

        Returns:
        float: The calculated lifetime utility U_{a,i,t}.
        """
        # Closed-form solution for lifetime utility
        denominator = self.r + (1 - vehicle.delta_z) / (1 - self.alpha)
        if denominator == 0:
            raise ValueError("The denominator is zero, adjust the parameters to avoid division by zero.")
        
        # Calculate the base lifetime utility using the closed form
        base_utility = commuting_util / denominator

        # Adjust the lifetime utility based on the scenario
        U_a_i_t = base_utility - beta *price - gamma * vehicle.emissions

        return U_a_i_t
    
    def calc_utility_cars_segments(self, market_data, vehicle_list):

        for segment_code, segment_data in market_data.items():
           
            beta_s =  segment_data["beta_s_t"]
            gamma_s = segment_data["gamma_s_t"]

            for car in vehicle_list:
                price_s = car.optimal_price_segments[segment_code]#price for that specific segment
                if  (car.transportType == 2) or (segment_code[-1] == 1 and car.transportType == 3):#THE EV ADOPTION BIT GOES LAST
                    utility_segment_U  = self.calculate_utility(car, car.car_utility_segments[segment_code], beta_s, gamma_s, price_s)
                    car.car_utility_segments_U[segment_code] = utility_segment_U 
                else:
                    car.car_utility_segments_U[segment_code] = 0 
    

    def calc_predicted_profit_segments(self, market_data, car_list):

        """
        Calculate the expected profit for market m', firm j, at time t given price P.

        """
        expected_profits_segments = {}
        #IS THER A BETTER WAY TO DO THIS?
        for segment_code, segment_data in market_data.items():
            expected_profits_segments[segment_code] = {}#create the dict that the data goes in

        for vehicle in car_list:
            
            # Loop through each market segment in market_data
            for segment_code, segment_data in market_data.items():
                profit = (vehicle.optimal_price_segments[segment_code] - vehicle.c_z_t)
                # Extract the data for the segment
                I_s_t = segment_data["I_s_t"]  # Size of individuals in the segment at time t
                sum_U_kappa = segment_data["sum_U_kappa"]
                # Accumulate the expected profit for this segment
                expected_profit =  profit * I_s_t * (vehicle.car_utility_segments_U[segment_code] ** self.kappa)/sum_U_kappa
                vehicle.expected_profit_segments[segment_code] = expected_profit

                expected_profits_segments[segment_code][expected_profit] = vehicle

        return expected_profits_segments
    
    def choose_cars_segments(self, market_data):
        """
        market_data: includes data on the population of each of the segments and the differnet values of the preferences/sensitivity of those segments
        """

        self.calc_optimal_price_cars(market_data,  self.list_technology_memory)#calc the optimal price of all cars, also does the utility and distance!

        self.calc_utility_cars_segments(market_data, self.list_technology_memory)#utility of all the cars for all the segments
        
        expected_profits = self.calc_predicted_profit_segments(market_data, self.list_technology_memory)#calculte the predicted profit of each segment 
        
        cars_selected = self.select_car_lambda_production(expected_profits)#pick the cars for each car

        return cars_selected


    def select_car_lambda_research(self, expected_profits_segments):
        """
        Probabilistically select a vehicle for research, where the probability of selecting a vehicle
        is proportional to its expected profit, rather than always selecting the one with the highest profit.

        Parameters:
        - expected_profits_segments (dict): A dictionary containing segment data with vehicles and their expected profits.

        Returns:
        - CarModel: The vehicle selected for research.
        """

        # List to store all vehicles and their corresponding profits
        vehicles = []
        profits = []

        # Iterate over all segments and collect profits and vehicles
        for segment_data in expected_profits_segments.values():
            for profit, vehicle in segment_data.items():
                vehicles.append(vehicle)
                profits.append(profit)

        # Convert profits list to numpy array
        profits = np.array(profits)

        # To avoid numerical instability with large or very small profit values, use a scaling factor for softmax
        # Here we subtract the max profit for numerical stability
        max_profit = np.max(profits)
        scaled_profits = profits - max_profit

        # Compute the softmax probabilities
        lambda_profits = scaled_profits**self.lambda_pow
        probabilities = lambda_profits / np.sum(lambda_profits)

        # Select a vehicle based on the computed probabilities
        selected_index = np.random.choice(len(vehicles), p=probabilities)
        selected_vehicle = vehicles[selected_index]

        return selected_vehicle


    def select_car_lambda_production(self, expected_profits_segments):
        """
        Probabilistically select a vehicle for production, where the probability of selecting a vehicle
        is proportional to its expected profit in each segemnt, rather than always selecting the one with the highest profit.

        Parameters:
        - expected_profits_segments (dict): A dictionary containing segment data with vehicles and their expected profits.

        Returns:
        - list CarModel: list of vehicles selected for production.
        """

        # List to store all vehicles and their corresponding profits
        
        vehicles_selected = []

        # Iterate over all segments and collect profits and vehicles
        for segement_code, segment_data in expected_profits_segments.items():
            vehicles = []        
            profits = []
            for profit, vehicle in segment_data.items():
                vehicles.append(vehicle)
                profits.append(profit)

            if len(profits) == 1:
                #IF there is only one car just pick that one
                selected_index = 0
            else:
                # Convert profits list to numpy array
                profits = np.array(profits)               
                # Compute the softmax probabilities
                profits[profits < 0] = 0#REPLACE NEGATIVE VALUES OF PROFIT WITH 0, SO PROBABILITY IS 0
                
                lambda_profits = profits**self.lambda_pow
                sum_profits = np.sum(lambda_profits)
                #("profits", profits, sum_profits)
                if sum_profits ==  0:
                    #num_cars = len(profits)
                    #probabilities = [1/num_cars]*len(num_cars)
                    #pick random car if non are profitable
                    selected_index = np.random.choice(len(vehicles))
                else:
                    probabilities = lambda_profits / np.sum(lambda_profits)
                    # Select a vehicle based on the computed probabilities
                    selected_index = np.random.choice(len(vehicles), p=probabilities)

            selected_vehicle = vehicles[selected_index]
            #SET PRICE OF CAR TO THAT OF THE SEGMENT ITS CHOSEN FOR
            selected_vehicle.price = selected_vehicle.optimal_price_segments[segement_code]
            vehicles_selected.append(selected_vehicle)

        return vehicles_selected


    def add_new_vehicle(self, vehicle_model_research):

        #add the vehicle
        if vehicle_model_research not in self.list_technology_memory:
            self.list_technology_memory.append(vehicle_model_research)
            if vehicle_model_research.transportType == 3:
                self.list_technology_memory_EV.append(vehicle_model_research)
            else:
                self.list_technology_memory_ICE.append(vehicle_model_research)

    def update_memory_timer(self):

        #change the timer for the techs that are not the ones being used
        for technology in self.list_technology_memory:
            technology.update_timer()

    def update_memory_len(self):
        #is the memory list is too long then remove data
        if len(self.list_technology_memory) > self.memory_cap:
            tech_to_remove = max((tech for tech in self.list_technology_memory if not tech.choosen_tech_bool), key=lambda x: x.timer, default=None)#PICK TECH WITH MAX TIMER WHICH IS NOT ACTIVE
            self.list_technology_memory.remove(tech_to_remove)#last thing is remove the item
            if tech_to_remove.transportType == 3:
                self.list_technology_memory_EV.remove(tech_to_remove)
            else:
                self.list_technology_memory_ICE.remove(tech_to_remove)

    def innovate(self, market_data):
        # create a list of cars in neighbouring memory space                               #last_researched_car, list_technology_memory, landscape, parameters_car
        unique_neighbouring_technologies_EV = self.generate_neighbouring_technologies(self.last_researched_car_EV,  self.list_technology_memory_EV, self.EV_landscape, self.parameters_car_EV )
        unique_neighbouring_technologies_ICE = self.generate_neighbouring_technologies(self.last_researched_car_ICE,  self.list_technology_memory_ICE, self.ICE_landscape, self.parameters_car_ICE)

        unique_neighbouring_technologies = unique_neighbouring_technologies_EV + unique_neighbouring_technologies_ICE + [self.last_researched_car_EV, self.last_researched_car_ICE]

        # calculate the optimal price of cars in the memory 
        self.calc_optimal_price_cars(market_data, unique_neighbouring_technologies)

        # calculate the utility of car segements
        self.calc_utility_cars_segments(market_data, unique_neighbouring_technologies)
        # calc the predicted profits of cars
        expected_profits_segments = self.calc_predicted_profit_segments(market_data, unique_neighbouring_technologies)
        # select the car to innovate
        vehicle_model_research = self.select_car_lambda_research(expected_profits_segments)

        if vehicle_model_research.transportType == 3:#EV
            self.last_researched_car_EV = vehicle_model_research
        else:
            self.last_researched_car_ICE = vehicle_model_research

        #add vehicle to memory, MAKE THE MEMORY INTO A CAR HERE?
        self.add_new_vehicle(vehicle_model_research)

        # adjust memory bank
        self.update_memory_timer()
        self.update_memory_len()

    ########################################################################################################################################
    #Memory
    def gen_neighbour_carsModel(self, tech_strings, nk_landscape, parameters_car):
        # Generate CarModel instances for the unique neighboring technologies
        neighbouring_technologies = []
        for tech_string in  tech_strings:
            # Calculate fitness for the new technology
            unique_tech_id = self.id_generator.get_new_id()
        
            new_car_model = CarModel(
                unique_id=unique_tech_id,
                component_string=tech_string,
                nk_landscape = nk_landscape,
                parameters = parameters_car,
                firm = self
            )
            neighbouring_technologies.append(new_car_model)

        return neighbouring_technologies

    def gen_neighbour_strings(self, memory_string_list, last_researched_car):

        unique_neighbouring_technologies_strings = set()
        unique_neighbouring_technologies_strings.update(last_researched_car.inverted_tech_strings)
        #get strings from memory
        list_technology_memory_strings = [vehicle.component_string for vehicle in memory_string_list]
        # Remove the existing technologies from neighboring options to avoid duplicates
        unique_neighbouring_technologies_strings -= set(list_technology_memory_strings)

        return unique_neighbouring_technologies_strings

    def generate_neighbouring_technologies(self, last_researched_car, list_technology_memory, landscape, parameters_car):
        """Generate neighboring technologies for cars. Roaming point"""
        # Set to track unique neighboring technology strings

        string_list = self.gen_neighbour_strings(list_technology_memory, last_researched_car)
        neighbouring_technologies= self.gen_neighbour_carsModel(string_list, landscape, parameters_car)

        return neighbouring_technologies
    
    ########################################################################################################################################

    def set_up_time_series_firm(self):
        self.history_profit = []
        self.history_firm_cars_users = []

    def save_timeseries_data_firm(self):
        self.history_profit.append(self.firm_profit)
        self.history_firm_cars_users.append(self.firm_cars_users)

    def next_step(self, market_data):
        self.t_firm += 1

        #decide cars to sell
        self.cars_on_sale = self.choose_cars_segments(market_data)

        #decide whether to innovate
        if np.random.rand() < self.prob_innovate:
            self.innovate(market_data)

        if self.save_timeseries_data_state and (self.t_firm % self.compression_factor_state == 0):
            self.save_timeseries_data_firm()
        
        return self.cars_on_sale

