import numpy as np
from package.model.carModel import CarModel

class Firm:
    def __init__(self, firm_id, init_tech_ICE, init_tech_EV, parameters_firm, parameters_car_ICE, parameters_car_EV):
        self.firm_id = firm_id
        #ICE
        self.init_tech_ICE = init_tech_ICE
        self.list_technology_memory_ICE = [init_tech_ICE]
        self.last_researched_car_ICE = self.init_tech_ICE
        #EV
        self.init_tech_EV = init_tech_EV
        self.list_technology_memory_EV = [init_tech_EV]
        self.last_researched_car_EV = self.init_tech_EV

        self.list_technology_memory = self.list_technology_memory_ICE + self.list_technology_memory_EV 

        self.parameters_firm = parameters_firm
        self.eta = parameters_firm["eta"]
        self.alpha = parameters_firm["alpha"]
        self.kappa = self.parameters_firm["kappa"]
        self.memory_cap = self.parameters_firm["memory_cap"]
        self.prob_innovate = self.parameters_firm["prob_innovate"]
        self.delta = self.parameters_firm["delta"]
        self.r = self.parameters_firm["r"]
        self.id_generator = parameters_firm["IDGenerator_firms"]
        self.lambda_pow = parameters_firm["lambda_pow"]

        self.parameters_car_ICE = parameters_car_ICE
        self.parameters_car_EV = parameters_car_EV

        self.expected_profits_segments = {}

        #ALL TYPES
        self.cars_on_sale = [init_tech_ICE, init_tech_EV]
        self.expected_profit_research_alternatives = None
        self.last_tech_expected_profit = None

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

    def calc_optimal_price_cars(self, market_data, car_list): 
        """Calculate the optimal price for each car in the car list based on market data."""

        for car in car_list:
            # Iterate over each market segment to calculate utilities and distances
            for segment_code, segment_data in market_data.items():
                beta_s = segment_data["beta_s_t"]
                gamma_s = segment_data["gamma_s_t"]
                E_m = car.emissions  # Emissions for the current car
                C_m = car.cost       # Cost for the current car

                # Calculate optimal distance for the given segment
                d_i_t = self.optimal_distance(car, beta_s, gamma_s)
                car.car_distance_segments[segment_code] = d_i_t  # Save the calculated distance

                # Calculate the utility for the given segment
                utility_segment = self.commuting_utility(car, d_i_t, beta_s, gamma_s)
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
        denominator = self.r + (1 - self.delta) / (1 - self.alpha)
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
                    utility_segment_U  = self.calculate_utility(self, car, car.car_utility_segments[segment_code], beta_s, gamma_s, price_s)
                    car.car_utility_segments_U[segment_code] = utility_segment_U 
                else:
                    car.car_utility_segments_U[segment_code] = 0 
    

    def calc_predicted_profit_segments(self, market_data, car_list):

        """
        Calculate the expected profit for market m', firm j, at time t given price P.

        """
        expected_profits_segments = {}
        for vehicle in car_list:

            profit = (vehicle.price - vehicle.c_z_t)
            # Loop through each market segment in market_data
            for segment_code, segment_data in market_data.items():
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
        
        cars_selected = self.select_car_lambda_research(expected_profits)#pick the cars for each car

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
        for segment_data in expected_profits_segments.values():
            vehicles = []        
            profits = []
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

            vehicles_selected.append(selected_vehicle)

        return vehicles_selected


    def add_new_vehicle(self, vehicle_model_research):

        #add the vehicle
        if vehicle_model_research not in self.list_technology_memory:
            self.list_technology_memory.append(vehicle_model_research)
            if vehicle_model_research.typeTransport == 3:
                self.list_technology_memory_EV.append(vehicle_model_research)
            else:
                self.list_technology_memory_ICE.append(vehicle_model_research)

    def update_memory_innovate(self):

        #change the timer for the techs that are not the ones being used
        for technology in self.list_technology_memory:
            technology.update_timer()

        #is the memory list is too long then remove data
        if len(self.list_technology_memory) > self.memory_cap:
            tech_to_remove = max((tech for tech in self.list_technology_memory if not tech.choosen_tech_bool), key=lambda x: x.timer, default=None)#PICK TECH WITH MAX TIMER WHICH IS NOT ACTIVE
            self.list_technology_memory.remove(tech_to_remove)#last thing is remove the item
            if tech_to_remove.typeTransport == 3:
                self.list_technology_memory_EV.remove(tech_to_remove)
            else:
                self.list_technology_memory_ICE.remove(tech_to_remove)

    def innovate(self, market_data):
        # create a list of cars in neighbouring memory space
        unique_neighbouring_technologies_EV = self.generate_neighbouring_technologies(self, self.last_researched_car_EV, self.parameters_car_EV, self.list_technology_memory_ICE )
        unique_neighbouring_technologies_ICE = self.generate_neighbouring_technologies(self, self.last_researched_car_ICE, self.parameters_car_ICE, self.list_technology_memory_ICE)

        unique_neighbouring_technologies = unique_neighbouring_technologies_EV + unique_neighbouring_technologies_ICE + [self.last_researched_car_EV, self.last_researched_car_ICE]

        # calculate the optimal price of cars in the memory 
        self.calc_optimal_price_cars(market_data, unique_neighbouring_technologies)
        # calculate the utility of car segements
        self.calc_utility_cars_segments(self, market_data, unique_neighbouring_technologies)
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
        self.update_memory_innovate(vehicle_model_research)

    ########################################################################################################################################
    #Memory

    def generate_neighbouring_technologies(self, last_researched_car, parameters_car, list_technology_memory):
        """Generate neighboring technologies for cars. Roaming point"""
        # Set to track unique neighboring technology strings
        unique_neighbouring_technologies_strings = set()

        unique_neighbouring_technologies_strings.update(last_researched_car.inverted_tech_strings)

        #get strings from memory
        list_technology_memory_strings = [vehicle.component_string for vehicle in list_technology_memory]

        # Remove the existing technologies from neighboring options to avoid duplicates
        unique_neighbouring_technologies_strings -= set(list_technology_memory_strings)

        # Generate CarModel instances for the unique neighboring technologies
        neighbouring_technologies = []
        for tech_string in unique_neighbouring_technologies_strings:
            # Calculate fitness for the new technology
            unique_tech_id = self.id_generator.get_new_id()
            
            # Create a new CarModel instance
            new_car_model = CarModel(
                unique_id=unique_tech_id,
                firm_id=self.firm_id,
                component_string=tech_string,
                parameters = parameters_car
            )
            neighbouring_technologies.append(new_car_model)

        return neighbouring_technologies
    
    ########################################################################################################################################

    def next_step(self, market_data):

        #decide cars to sell
        self.cars_on_sale = self.choose_cars_segments(market_data)

        #decide whether to innovate
        if np.random.rand() < self.prob_innovate:
            self.innovate(market_data)
        
        return self.cars_on_sale

