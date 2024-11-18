import numpy as np
from package.model.carModel import CarModel

class Firm:
    def __init__(self, firm_id, init_tech_ICE, init_tech_EV, parameters_firm, parameters_car_ICE, parameters_car_EV):
        
        self.t_firm = 0
        self.save_timeseries_data_state = parameters_firm["save_timeseries_data_state"]
        self.compression_factor_state = parameters_firm["compression_factor_state"]
        self.id_generator = parameters_firm["IDGenerator_firms"]    

        self.firm_id = firm_id
        #ICE
        self.init_tech_ICE = init_tech_ICE
        self.init_tech_ICE.firm = self
        self.init_tech_ICE.unique_id = self.id_generator.get_new_id()
        self.list_technology_memory_ICE = [init_tech_ICE]
        self.last_researched_car_ICE = self.init_tech_ICE

        #EV
        self.init_tech_EV = init_tech_EV
        self.init_tech_EV.firm = self
        self.init_tech_EV.unique_id = self.id_generator.get_new_id()
        self.list_technology_memory_EV = [init_tech_EV]
        self.last_researched_car_EV = self.init_tech_EV

        self.firm_profit = 0
        self.firm_cars_users = 0
        self.research_bool = 0

        self.list_technology_memory = self.list_technology_memory_ICE + self.list_technology_memory_EV 

        self.parameters_firm = parameters_firm
        self.eta = parameters_firm["eta"]
        self.alpha = parameters_firm["alpha"]
        self.kappa = self.parameters_firm["kappa"]
        self.memory_cap = self.parameters_firm["memory_cap"]
        self.prob_innovate = self.parameters_firm["prob_innovate"]
        self.r = self.parameters_firm["r"]
        self.delta_z = self.parameters_firm["delta_z"]
        
        self.carbon_price =  self.parameters_firm["carbon_price"]

        self.lambda_pow = parameters_firm["lambda_pow"]

        self.universal_model_repo_ICE = parameters_firm["universal_model_repo_ICE"]#THIS NEEDS TO BE SHARED AMONGST ALL FIRMS
        self.universal_model_repo_EV = parameters_firm["universal_model_repo_EV"]#THIS NEEDS TO BE SHARED AMONGST ALL FIRMS

        self.ICE_landscape = self.parameters_firm["ICE_landscape"]
        self.EV_landscape = self.parameters_firm["EV_landscape"]

        self.parameters_car_ICE = parameters_car_ICE
        self.parameters_car_EV = parameters_car_EV

        self.expected_profits_segments = {}

        #ALL TYPES
        self.cars_on_sale = [init_tech_ICE, init_tech_EV]
        self.set_car_init_price_and_U()

        np.random.seed(parameters_firm["innovation_seed"])

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
                car.car_base_utility_segments[segment_code_str] = self.parameters_firm["init_base_U"]
    
    def optimal_distance(self, vehicle, beta, gamma):
        """
        Calculate the optimal distance based on the vehicle properties.

        Parameters:
        vehicle (Vehicle): The vehicle for which the optimal distance is calculated.

        Returns:
        float: The calculated optimal distance, d^*_{a,i,t}.
        """

        numerator = self.alpha * vehicle.Quality_a_t * (1 - vehicle.delta_z) ** vehicle.L_a_t
        denominator = (beta * vehicle.Eff_omega_a_t ** -1 * (vehicle.fuel_cost_c_z + self.carbon_price*vehicle.e_z_t) +
                       gamma * vehicle.Eff_omega_a_t ** -1 * vehicle.e_z_t +
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
        Quality_a_t = vehicle.Quality_a_t
        delta_z = vehicle.delta_z
        L_a_t = vehicle.L_a_t
        Eff_omega_a_t = vehicle.Eff_omega_a_t
        e_z_t = vehicle.e_z_t
        fuel_cost_c_z = vehicle.fuel_cost_c_z
        nu_z_i_t = vehicle.nu_z_i_t

        # Calculate commuting utility based on conditions for z
        cost_component = beta_s * (1 / Eff_omega_a_t) * (fuel_cost_c_z + self.carbon_price*e_z_t) + gamma_s * (1 / Eff_omega_a_t) * e_z_t + self.eta * nu_z_i_t
        utility = Quality_a_t * (1 - delta_z) ** L_a_t * (d_i_t ** self.alpha) - d_i_t * cost_component

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
                U_sum = segment_data["U_sum"]
                E_m = car.emissions  # Emissions for the current car
                C_m = car.ProdCost_z_t  + self.carbon_price*E_m       # Cost for the current car

                # Calculate optimal distance for the given segment
                d_i_t = self.optimal_distance(car, beta_s, gamma_s)
                car.car_distance_segments[segment_code] = d_i_t  # Save the calculated distance

                # Calculate the commuting  utility for the given segment
                utility_segment = self.calc_commuting_utility(car, d_i_t, beta_s, gamma_s)
                
                # Save the base utility
                B = utility_segment/(self.r + (1-self.delta_z)/(1-self.alpha))
                car.car_base_utility_segments[segment_code] = B
                # Set the calculated optimal price for the car
                car.optimal_price_segments[segment_code] = max(C_m,(U_sum  + B - gamma_s * E_m - np.sqrt(U_sum*(U_sum + B - gamma_s*E_m - beta_s*C_m )) )/beta_s   )

        return car_list
    
    def calc_utility_cars_segments(self, market_data, vehicle_list):

        for segment_code, segment_data in market_data.items():
           
            beta_s =  segment_data["beta_s_t"]
            gamma_s = segment_data["gamma_s_t"]

            for car in vehicle_list:
                price_s = car.optimal_price_segments[segment_code]#price for that specific segment
                if  car.transportType == 2 or all((segment_code[-1] == str(1), car.transportType == 3)):#THE EV ADOPTION BIT GOES LAST
                    utility_segment_U  = car.car_base_utility_segments[segment_code] - beta_s *price_s - gamma_s * car.emissions
                    car.car_utility_segments_U[segment_code] = utility_segment_U 
                else:
                    car.car_utility_segments_U[segment_code] = 0 

    
    def calc_predicted_profit_segments(self, market_data, car_list):
        """
        Calculate the expected profit for each segment, with segments that consider EVs able to buy both EVs and ICE cars, 
        and segments that do not consider EVs only able to buy ICE cars.
        """
        expected_profits_segments = {}
        
        # Initialize expected profits dictionary for each segment
        for segment_code, segment_data in market_data.items():
            expected_profits_segments[segment_code] = {}  # Initialize dict to store profits by vehicle
        
        # Loop through each vehicle in car_list to calculate profit for each segment
        for vehicle in car_list:
            is_ev = vehicle.transportType == 3  # Assuming transportType 3 is EV; adjust as needed

            # Calculate profit for each segment
            for segment_code, segment_data in market_data.items():

                consider_ev = segment_code[2] == str(1)  # Determine if the segment considers EVs

                # For segments that consider EVs, calculate profit for both EV and ICE vehicles
                if consider_ev or not is_ev:  # Include EV only if the segment considers EV, always include ICE                    
                    # Calculate profit for this vehicle and segment
                    profit_per_sale = vehicle.optimal_price_segments[segment_code] - (vehicle.ProdCost_z_t  + self.carbon_price*vehicle.emissions) 
                    I_s_t = segment_data["I_s_t"]  # Size of individuals in the segment at time t
                    U_sum = segment_data["U_sum"]
                    
                    # Expected profit calculation
                    expected_profit = profit_per_sale * I_s_t * (max(0,vehicle.car_utility_segments_U[segment_code])**self.kappa)/(U_sum + vehicle.car_utility_segments_U[segment_code])**self.kappa
                    
                    # Store profit in the vehicle's expected profit attribute and update the main dictionary
                    vehicle.expected_profit_segments[segment_code] = expected_profit
                    expected_profits_segments[segment_code][expected_profit] = vehicle
                else:
                    # Store profit in the vehicle's expected profit attribute and update the main dictionary
                    expected_profit = 0
                    vehicle.expected_profit_segments[segment_code] = expected_profit
                    expected_profits_segments[segment_code][expected_profit] = vehicle

        return expected_profits_segments


    def choose_cars_segments(self, market_data):
        """
        market_data: includes data on the population of each of the segments and the differnet values of the preferences/sensitivity of those segments
        """

        self.list_technology_memory = self.list_technology_memory_EV + self.list_technology_memory_ICE

        self.calc_optimal_price_cars(market_data,  self.list_technology_memory)#calc the optimal price of all cars, also does the utility and distance!

        self.calc_utility_cars_segments(market_data, self.list_technology_memory)#utility of all the cars for all the segments
        
        expected_profits = self.calc_predicted_profit_segments(market_data, self.list_technology_memory)#calculte the predicted profit of each segment 
        
        cars_selected = self.select_car_lambda_production(expected_profits)#pick the cars for each car

        #DO THIS FASTER
        for car in self.list_technology_memory_EV:
            if car in cars_selected:
                car.choosen_tech_bool = 1 
            else:
                car.choosen_tech_bool = 0

        for car in self.list_technology_memory_ICE:
            if car in cars_selected:
                car.choosen_tech_bool = 1 
            else:
                car.choosen_tech_bool = 0

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
        profits[profits < 0] = 0#REPLACE NEGATIVE VALUES OF PROFIT WITH 0, SO PROBABILITY IS 0

        # Compute the softmax probabilities
        lambda_profits = profits**self.lambda_pow
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

        # Dictionary to store the best profit and segment for each vehicle
        vehicle_best_segment = {}

        for segment_code, segment_data in expected_profits_segments.items():
            vehicles = []
            profits = []
            
            # Extract vehicles and profits for the current segment
            for profit, vehicle in segment_data.items():
                vehicles.append(vehicle)
                profits.append(profit)

            if len(profits) == 1:
                # If there is only one car, select it
                selected_index = 0
            else:
                # Convert profits list to a numpy array
                profits = np.array(profits)
                
                # Replace negative profits with zero (no selection probability)
                profits[profits < 0] = 0
                
                # Select the index of the vehicle with the highest profit
                selected_index = np.argmax(profits)

            # Select the vehicle based on the index
            selected_vehicle = vehicles[selected_index]
            selected_profit = profits[selected_index]

            # Check if this vehicle is already in the best segment tracker
            if selected_vehicle in vehicle_best_segment:
                # If already present, compare profits
                current_best_profit = vehicle_best_segment[selected_vehicle]['profit']
                if selected_profit > current_best_profit:
                    # Update to the segment with the higher profit
                    vehicle_best_segment[selected_vehicle] = {
                        'profit': selected_profit,
                        'segment_code': segment_code
                    }
            else:
                # Add the vehicle to the tracker with its current best segment
                vehicle_best_segment[selected_vehicle] = {
                    'profit': selected_profit,
                    'segment_code': segment_code
                }

        # Final selection of vehicles with the highest utility prices
        vehicles_selected = []
        for vehicle, info in vehicle_best_segment.items():
            segment_code = info['segment_code']
            vehicle.price = vehicle.optimal_price_segments[segment_code]
            vehicles_selected.append(vehicle)

        return vehicles_selected

    def add_new_vehicle(self, vehicle_model_research):

        #add the vehicle
        if vehicle_model_research.transportType == 2:
            if vehicle_model_research not in self.list_technology_memory_ICE:
                self.list_technology_memory_ICE.append(vehicle_model_research)
        else:
            if vehicle_model_research not in self.list_technology_memory_ICE:
                self.list_technology_memory_EV.append(vehicle_model_research)
                
    def update_memory_timer(self):
        #change the timer for the techs that are not the ones being used
        for technology in self.list_technology_memory:
            technology.update_timer()

    def update_memory_len(self):
        #is the memory list is too long then remove data
        #print(len(self.list_technology_memory))
        
        if len(self.list_technology_memory) > self.memory_cap:

            tech_to_remove = max((tech for tech in self.list_technology_memory if not tech.choosen_tech_bool), key=lambda x: x.timer, default=None)#PICK TECH WITH MAX TIMER WHICH IS NOT ACTIVE
            
            if tech_to_remove.transportType == 3:
                self.list_technology_memory_EV.remove(tech_to_remove)
            else:
                self.list_technology_memory_ICE.remove(tech_to_remove)

    def innovate(self, market_data):
        # create a list of cars in neighbouring memory space                                #self, last_researched_car,  list_technology_memory,        list_technology_memory_neighbouring, landscape, parameters_car, transportType
        self.unique_neighbouring_technologies_EV = self.generate_neighbouring_technologies(self.last_researched_car_EV,  self.list_technology_memory_EV, self.EV_landscape, self.parameters_car_EV, transportType = 3 )
        self.unique_neighbouring_technologies_ICE = self.generate_neighbouring_technologies(self.last_researched_car_ICE,  self.list_technology_memory_ICE, self.ICE_landscape, self.parameters_car_ICE, transportType = 2)

        self.unique_neighbouring_technologies = self.unique_neighbouring_technologies_EV + self.unique_neighbouring_technologies_ICE + [self.last_researched_car_EV, self.last_researched_car_ICE]

        # calculate the optimal price of cars in the memory 
        self.calc_optimal_price_cars(market_data, self.unique_neighbouring_technologies)

        # calculate the utility of car segements
        self.calc_utility_cars_segments(market_data, self.unique_neighbouring_technologies)
        # calc the predicted profits of cars
        expected_profits_segments = self.calc_predicted_profit_segments(market_data, self.unique_neighbouring_technologies)
        # select the car to innovate
        self.vehicle_model_research = self.select_car_lambda_research(expected_profits_segments)

        #print("vehicle_model_research",self.firm_id, vehicle_model_research.component_string)

        if self.vehicle_model_research.transportType == 3:#EV
            self.last_researched_car_EV = self.vehicle_model_research
        else:
            self.last_researched_car_ICE = self.vehicle_model_research

        #add vehicle to memory, MAKE THE MEMORY INTO A CAR HERE?
        self.add_new_vehicle(self.vehicle_model_research)

        # adjust memory bank
        self.update_memory_timer()
        self.update_memory_len()

    ########################################################################################################################################
    #Memory
    def gen_neighbour_carsModel(self, tech_strings, nk_landscape, parameters_car, transportType):
        # Generate CarModel instances for the unique neighboring technologies
        neighbouring_technologies = []

        if transportType == 2:
            universal_model_repo = self.universal_model_repo_ICE
        else:
            universal_model_repo = self.universal_model_repo_EV

        for tech_string in  tech_strings:
            if tech_string in universal_model_repo.keys():
                tech_to_add = universal_model_repo[tech_string]
            else:
                tech_to_add = CarModel(
                    component_string=tech_string,
                    nk_landscape = nk_landscape,
                    parameters = parameters_car
                )
                universal_model_repo[tech_string] = tech_to_add

            unique_tech_id = self.id_generator.get_new_id()
            tech_to_add.unique_id = unique_tech_id
            tech_to_add.firm = self
            neighbouring_technologies.append(tech_to_add)

        return neighbouring_technologies
    
    def gen_neighbour_strings(self, memory_string_list, last_researched_car):

        unique_neighbouring_technologies_strings = set()
        unique_neighbouring_technologies_strings.update(last_researched_car.inverted_tech_strings)
        #get strings from memory
        list_technology_memory_strings = [vehicle.component_string for vehicle in memory_string_list]
        # Remove the existing technologies from neighboring options to avoid duplicates
        unique_neighbouring_technologies_strings -= set(list_technology_memory_strings)

        return unique_neighbouring_technologies_strings

    def generate_neighbouring_technologies(self, last_researched_car, list_technology_memory, landscape, parameters_car, transportType):
        """Generate neighboring technologies for cars. Roaming point"""
        # Set to track unique neighboring technology strings

        #string_list = self.gen_neighbour_strings(list_technology_memory, last_researched_car)
        string_list = self.gen_neighbour_strings_tumor(list_technology_memory)
        
        #string_list = self.gen_neighbour_strings_alt(list_technology_memory)
        self.neighbouring_technologies = self.gen_neighbour_carsModel(string_list, landscape, parameters_car, transportType)

        return self.neighbouring_technologies
    
    ########################################################################################################################################
    #TUMOR ALTERNATIVE
    
    def gen_neighbour_strings_tumor(self, list_technology_memory):

        unique_neighbouring_technologies_strings = set(
            tech_string  # Each individual tech string we want to add to the set
            for vehicleModel in list_technology_memory  # For each vehicleModel in the list
            for tech_string in vehicleModel.inverted_tech_strings  # For each tech_string in the vehicleModel's inverted_tech_strings
        )
        #get strings from memory
        list_technology_memory_strings = [vehicle.component_string for vehicle in list_technology_memory]
        # Remove the existing technologies from neighboring options to avoid duplicates
        unique_neighbouring_technologies_strings -= set(list_technology_memory_strings)

        return unique_neighbouring_technologies_strings
    
    ########################################################################################################################################

    def set_up_time_series_firm(self):
        self.history_profit = []
        self.history_firm_cars_users = []
        self.history_attributes_researched = []
        self.history_research_type = []

    def save_timeseries_data_firm(self):
        self.history_profit.append(self.firm_profit)
        self.history_firm_cars_users.append(self.firm_cars_users)
        if self.research_bool == 1:
            self.history_attributes_researched.append(self.vehicle_model_research.attributes_fitness)
            if self.vehicle_model_research.transportType == 3:
                EVbool_res = 1
            else:
                EVbool_res = 0
            self.history_research_type.append(EVbool_res)
        else:
            self.history_attributes_researched.append([np.nan, np.nan,np.nan ])
            self.history_research_type.append(np.nan)
        
    def next_step(self, market_data, carbon_price):
        self.t_firm += 1
        self.carbon_price = carbon_price
        #decide cars to sell
        self.cars_on_sale = self.choose_cars_segments(market_data)

        #decide whether to innovate
        if np.random.rand() < self.prob_innovate:
            #print("HIGH", self.firm_id, self.t_firm)
            self.innovate(market_data)
            self.research_bool = 1

        if self.save_timeseries_data_state and (self.t_firm % self.compression_factor_state == 0):
            self.save_timeseries_data_firm()
            self.research_bool = 0

        return self.cars_on_sale

