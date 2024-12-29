import numpy as np
from package.model.carModel import CarModel

class Firm:
    def __init__(self, firm_id, init_tech_ICE, init_tech_EV, parameters_firm, parameters_car_ICE, parameters_car_EV, innovation_seed):
        

        self.rebate = parameters_firm["rebate"]#7000#JUST TO TRY TO GET TRANSITION

        self.t_firm = 0
        self.production_change_bool = 0
        self.policy_distortion = 0
        
        self.save_timeseries_data_state = parameters_firm["save_timeseries_data_state"]
        self.compression_factor_state = parameters_firm["compression_factor_state"]
        self.id_generator = parameters_firm["IDGenerator_firms"]  

        self.num_cars_production =   parameters_firm["num_cars_production"]  
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
        
        self.ev_research_bool = parameters_firm["ev_research_bool"]
        self.ev_production_bool = parameters_firm["ev_production_bool"]

        self.firm_profit = 0
        self.firm_cars_users = 0
        self.research_bool = 0
        self.EVs_sold = 0

        self.list_technology_memory = self.list_technology_memory_ICE + self.list_technology_memory_EV

        self.parameters_firm = parameters_firm
        self.alpha = parameters_firm["alpha"]
        self.kappa = self.parameters_firm["kappa"]
        self.memory_cap = self.parameters_firm["memory_cap"]
        self.prob_innovate = self.parameters_firm["prob_innovate"]
        self.prob_change_production = self.parameters_firm["prob_change_production"]
        self.r = self.parameters_firm["r"]
        self.delta = self.parameters_firm["delta"]

        self.init_U_sum = self.parameters_firm["init_U_sum"]
        self.init_price_multiplier = self.parameters_firm["init_price_multiplier"]
        
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
        if self.ev_production_bool:
            self.cars_on_sale = [init_tech_ICE, init_tech_EV]
        else:
            self.cars_on_sale = [init_tech_ICE]

        self.set_car_init_price_and_U()

        if self.save_timeseries_data_state:
            self.set_up_time_series_firm()

        self.random_state = np.random.RandomState(innovation_seed)  # Local random state
    
    def set_car_init_price_and_U(self):
        for car in self.cars_on_sale:
            car.price = car.ProdCost_t*self.init_price_multiplier
            for segment_code in range(8):
                # Binary representation of the segment code (4-bit string)
                segment_code_str = format(segment_code, '03b')
                # Add data for the segment
                car.optimal_price_segments[segment_code_str] = car.price
                car.car_base_utility_segments[segment_code_str] = self.init_U_sum
    
    def optimal_distance(self, vehicle, beta, gamma):
        """
        Calculate the optimal distance based on the vehicle properties.

        Parameters:
        vehicle (Vehicle): The vehicle for which the optimal distance is calculated.

        Returns:
        float: The calculated optimal distance, d^*_{a,i,t}.
        """

        numerator = self.alpha * vehicle.Quality_a_t * (1 - vehicle.delta) ** vehicle.L_a_t
        denominator = ((beta/vehicle.Eff_omega_a_t) * (vehicle.fuel_cost_c + self.carbon_price*vehicle.e_t) +
                       (gamma/vehicle.Eff_omega_a_t) * vehicle.e_t)

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
        delta = vehicle.delta
        L_a_t = vehicle.L_a_t
        Eff_omega_a_t = vehicle.Eff_omega_a_t
        e_t = vehicle.e_t
        fuel_cost_c = vehicle.fuel_cost_c

        # Calculate commuting utility based on conditions for z
        
        cost_component = (beta_s / Eff_omega_a_t) * (fuel_cost_c + self.carbon_price*e_t) + (gamma_s/ Eff_omega_a_t) * e_t
        utility = Quality_a_t * (1 - delta) ** L_a_t * (d_i_t ** self.alpha) - d_i_t * cost_component

        # Ensure utility is non-negative
        utility_final = max(0, utility)

        return utility_final


    def calc_optimal_price_cars(self, market_data, car_list): 
        """Calculate the optimal price for each car in the car list based on market data. USED WHEN FIRST STUDYING A CAR"""

        for car in car_list:
            E_m = car.emissions  # Emissions for the current car                                                  
            C_m = car.ProdCost_t  #+ self.carbon_price*E_m  # Cost for the current car

            #UPDATE EMMISSION AND PRICES, THIS WORKS FOR BOTH PRODUCTION AND INNOVATION
            if car.transportType == 3:#EV
                C_m  = car.ProdCost_t - self.production_subsidy

            # Iterate over each market segment to calculate utilities and distances
            for segment_code, segment_data in market_data.items():
                beta_s = segment_data["beta_s_t"]
                gamma_s = segment_data["gamma_s_t"]
                U_sum = segment_data["U_sum"]
 
                # Calculate optimal distance for the given segment
                d_i_t = self.optimal_distance(car, beta_s, gamma_s)
                car.car_distance_segments[segment_code] = d_i_t  # Save the calculated distance

                # Calculate the commuting  utility for the given segment
                utility_segment = self.calc_commuting_utility(car, d_i_t, beta_s, gamma_s)

                # Save the base utility
                #B = utility_segment/(self.r + (np.log(1+self.delta))/(1-self.alpha))
                B = utility_segment*(1+self.r) / ((1+self.r) - (1 - self.delta)**(1/(1 - self.alpha)))

                car.car_base_utility_segments[segment_code] = B

                inside_component = U_sum*(U_sum + B - gamma_s*E_m - beta_s*C_m )
                if inside_component < 0 or self.t_firm == 1:
                    car.optimal_price_segments[segment_code] = C_m#STOP NEGATIVE SQUARE ROOTS
                else:
                    car.optimal_price_segments[segment_code] = max(C_m,(U_sum  + B - gamma_s * E_m - np.sqrt(inside_component) )/beta_s )

        return car_list
    
    def calc_utility_cars_segments(self, market_data, vehicle_list):

        for segment_code, segment_data in market_data.items():
           
            beta_s =  segment_data["beta_s_t"]
            gamma_s = segment_data["gamma_s_t"]

            for car in vehicle_list:
                price_s = car.optimal_price_segments[segment_code]#price for that specific segment

                if (car.transportType == 2) or all((segment_code[2] == str(1), car.transportType == 3)):#THE EV ADOPTION BIT GOES SECOND LAST
                    #ADD IN A SUBSIDY
                    if car.transportType == 3:
                        utility_segment_U  = car.car_base_utility_segments[segment_code] - beta_s *(price_s - self.rebate) - gamma_s * car.emissions
                    else:
                        utility_segment_U  = car.car_base_utility_segments[segment_code] - beta_s *price_s - gamma_s * car.emissions
                    #utility_segment_U  = car.car_base_utility_segments[segment_code] - beta_s *price_s - gamma_s * car.emissions
                    car.car_utility_segments_U[segment_code] = utility_segment_U 
                    #print(self.t_firm, self.firm_id,segment_code,car.car_base_utility_segments[segment_code], utility_segment_U ,price_s)
                else:
                    car.car_utility_segments_U[segment_code] = 0 

    
    def calc_predicted_profit_segments_production(self, market_data, car_list):
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
                    if is_ev:#PRODUCTION SUBSIDY
                        profit_per_sale = vehicle.optimal_price_segments[segment_code] - (vehicle.ProdCost_t - self.production_subsidy) # + self.carbon_price*vehicle.emissions 
                    else:
                        profit_per_sale = vehicle.optimal_price_segments[segment_code] - (vehicle.ProdCost_t)

                    I_s_t = segment_data["I_s_t"]  # Size of individuals in the segment at time t
                    U_sum = segment_data["U_sum"]
                    
                    # Expected profit calculation
                    utility_car = max(0,vehicle.car_utility_segments_U[segment_code])

                    utility_proportion = ((utility_car)**self.kappa)/((U_sum + utility_car)**self.kappa)

                    raw_profit = profit_per_sale * I_s_t * utility_proportion

                    if is_ev:
                        expected_profit = raw_profit
                    else:
                        expected_profit = raw_profit*(1-self.discriminatory_corporate_tax)

                    # Store profit in the vehicle's expected profit attribute and update the main dictionary
                    vehicle.expected_profit_segments[segment_code] = expected_profit
                    
                    expected_profits_segments[segment_code][expected_profit] = vehicle
                else:
                    # Store profit in the vehicle's expected profit attribute and update the main dictionary
                    expected_profit = 0
                    vehicle.expected_profit_segments[segment_code] = expected_profit
                    expected_profits_segments[segment_code][expected_profit] = vehicle

        return expected_profits_segments

    def calc_predicted_profit_segments_research(self, market_data, car_list):
        """
        THIS INCLUDES THE FLAT SUBSIDY FOR EV RESEARCH!
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
                    if is_ev:#PRODUCTION SUBSIDY
                        profit_per_sale = vehicle.optimal_price_segments[segment_code] - (vehicle.ProdCost_t - self.production_subsidy) # + self.carbon_price*vehicle.emissions 
                    else:
                        profit_per_sale = vehicle.optimal_price_segments[segment_code] - (vehicle.ProdCost_t)
                    
                    I_s_t = segment_data["I_s_t"]  # Size of individuals in the segment at time t
                    U_sum = segment_data["U_sum"]
                    
                    # Expected profit calculation
                    utility_car = max(0,vehicle.car_utility_segments_U[segment_code])

                    #print(vehicle.car_utility_segments_U[segment_code])
                    utility_proportion = ((utility_car)**self.kappa)/((U_sum + utility_car)**self.kappa)

                    raw_profit = profit_per_sale * I_s_t * utility_proportion

                    if is_ev:
                        expected_profit = raw_profit + self.research_subsidy
                    else:
                        expected_profit = raw_profit*(1-self.discriminatory_corporate_tax)
                    #print("expected_profit ",expected_profit )
                    # Store profit in the vehicle's expected profit attribute and update the main dictionary
                    vehicle.expected_profit_segments[segment_code] = expected_profit 
                    
                    expected_profits_segments[segment_code][expected_profit] = vehicle
                else:
                    # Store profit in the vehicle's expected profit attribute and update the main dictionary
                    expected_profit = self.research_subsidy
                    vehicle.expected_profit_segments[segment_code] = expected_profit 
                    expected_profits_segments[segment_code][expected_profit] = vehicle

        return expected_profits_segments

    def select_car_lambda_research(self, expected_profits_segments):
        """
        Probabilistically select a vehicle for research, where the probability of selecting a vehicle
        is proportional to its expected profit, rather than always selecting the one with the highest profit.

        Parameters:
        - expected_profits_segments (dict): A dictionary containing segment data with vehicles and their expected profits.

        Returns:
        - CarModel: The vehicle selected for research.
        """

        # Dictionary to store the highest profit for each vehicle
        vehicle_profits = {}

        # Iterate through the segments to determine the highest profit for each vehicle
        for segment_data in expected_profits_segments.values():
            for profit, vehicle in segment_data.items():
                # Update the profit if the vehicle is not in the dictionary or if the new profit is higher
                if vehicle not in vehicle_profits or profit > vehicle_profits[vehicle]:
                    vehicle_profits[vehicle] = profit

        # Separate the vehicles and profits into lists
        vehicles = list(vehicle_profits.keys())
        profits = list(vehicle_profits.values())

        # Convert profits list to numpy array
        profits = np.array(profits)

        profits[profits < 0] = 0#REPLACE NEGATIVE VALUES OF PROFIT WITH 0, SO PROBABILITY IS 0
        profits[profits == np.nan] = 0#REPLACE NEGATIVE VALUES OF PROFIT WITH 0, SO PROBABILITY IS 0

        # Compute the softmax probabilities
        lambda_profits = profits**self.lambda_pow        
        sum_prob = np.sum(lambda_profits)
        len_vehicles = len(vehicles)#literally just cos i do it 3 times
        if sum_prob == 0:
            selected_index = self.random_state.choice(len_vehicles)#pick one
        else:
            probabilities = lambda_profits/np.sum(sum_prob)
            selected_index = self.random_state.choice(len_vehicles, p=probabilities)
        # Select a vehicle based on the computed probabilities
        selected_vehicle = vehicles[selected_index]

        return selected_vehicle

    def select_car_lambda_production_old(self, expected_profits_segments):
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
        vehicles_selected_profits = []
        vehicle_segments = []  # To keep track of vehicle segments

        for vehicle, info in vehicle_best_segment.items():
            segment_code = info['segment_code']
            vehicle.price = vehicle.optimal_price_segments[segment_code]
            vehicles_selected_profits.append((vehicle, info["profit"]))
            vehicle_segments.append((vehicle, segment_code))  # Store segment info

        # SELECT THE most profitable cars
        if len(vehicles_selected_profits) < self.num_cars_production:
            vehicles_selected = [x[0] for x in vehicles_selected_profits]
        else:
            vehicles_selected = [x[0] for x in sorted(vehicles_selected_profits, key=lambda x: x[1], reverse=True)]

        # Count segments of selected vehicles
        self.selected_vehicle_segment_counts = {}

        for vehicle, segment_code in vehicle_segments:
            if vehicle in vehicles_selected:
                if segment_code not in self.selected_vehicle_segment_counts:
                    self.selected_vehicle_segment_counts[segment_code] = 0
                self.selected_vehicle_segment_counts[segment_code] += 1

        return vehicles_selected

    def select_car_lambda_production_alt(self, market_data, car_list):
        """
        Select vehicles for production based on a matrix of expected profits for each segment and technology combination.
        Iteratively selects the most profitable combination, removes the segment, and updates the technology column.

        Parameters:
        - market_data (dict): Market data for each segment.
        - car_list (list): List of CarModel objects.

        Returns:
        - list (tuple): List of tuples containing selected vehicles and their associated chosen segments.
        """
        # Step 0: Build the profit matrix (segments x technologies)
        segments = list(market_data.keys())
        technologies = car_list
        profit_matrix = np.zeros((len(segments), len(technologies)))

        vehicles_selected_profits = []  # Used to track the vehicle and the expected profit
        vehicle_segments = []  # To keep track of vehicle segment which it was targeting

        # Populate the profit matrix with expected profits
        for i, segment_code in enumerate(segments):
            for j, car in enumerate(technologies):
                #print("MATRIX IN",self.t_firm,self.firm_id,car.expected_profit_segments[segment_code])
                profit_matrix[i, j] = car.expected_profit_segments[segment_code]
        
        if not np.any(profit_matrix): #PROFIT IS ALL 0, pick up to 
            if len(technologies) > self.num_cars_production:
                #print("ALL ZEROS AND MORE THAN NUM CARS PRODUCTION")
                vehicles_selected = technologies[:self.num_cars_production]
            else:
                #print("ALL ZEROS AND LESS THAN NUM CARS PRODUCTION")
                vehicles_selected = technologies

            if self.save_timeseries_data_state and (self.t_firm % self.compression_factor_state == 0):
                self.selected_vehicle_segment_counts = {}
                for segment_code in range(8):
                    segment_code_str = format(segment_code, '03b')
                    self.selected_vehicle_segment_counts[segment_code_str] = np.nan
        else:
            selected_vehicles = []  # To store selected vehicles

            # Iterate until all segments are covered or the max number of cars are chosen
            for _ in range(len(segments)):
                # Step 1: Choose the highest value square in the matrix
                max_index = np.unravel_index(np.argmax(profit_matrix, axis=None), profit_matrix.shape)
                max_row, max_col = max_index
                max_profit = profit_matrix[max_row, max_col]

                # Select the car and mark the segment as covered
                selected_vehicle = technologies[max_col]
                vehicles_selected_profits.append((selected_vehicle, max_profit, segments[max_row]))
                vehicle_segments.append((selected_vehicle, segments[max_row]))

                # Step 2: Remove the segment (set the row to negative infinity)
                profit_matrix[max_row, :] = -np.inf

                # Step 3: Update the column for the selected technology using the optimal price for the chosen segment
                optimal_price_chosen_segment = selected_vehicle.optimal_price_segments[segments[max_row]]
                selected_vehicle.price = optimal_price_chosen_segment  # SET PRICE

                for i, segment_code in enumerate(segments):
                    if profit_matrix[i, max_col] != -np.inf:  # Skip covered rows
                        selected_vehicle.optimal_price_segments[segment_code] = optimal_price_chosen_segment

                        beta_s = market_data[segment_code]["beta_s_t"]
                        gamma_s = market_data[segment_code]["gamma_s_t"]

                        if (selected_vehicle.transportType == 2) or all(
                            (segment_code[2] == str(1), selected_vehicle.transportType == 3)
                        ):
                            if selected_vehicle.transportType == 3:
                                utility_segment_U = (
                                    selected_vehicle.car_base_utility_segments[segment_code]
                                    - beta_s * (selected_vehicle.optimal_price_segments[segment_code] - self.rebate)
                                    - gamma_s * selected_vehicle.emissions
                                )
                            else:
                                utility_segment_U = (
                                    selected_vehicle.car_base_utility_segments[segment_code]
                                    - beta_s * selected_vehicle.optimal_price_segments[segment_code]
                                    - gamma_s * selected_vehicle.emissions
                                )
                            selected_vehicle.car_utility_segments_U[segment_code] = utility_segment_U
                        else:
                            selected_vehicle.car_utility_segments_U[segment_code] = 0

                        if selected_vehicle.transportType == 3:#PRODUCTION SUBSIDY
                            profit_per_sale = selected_vehicle.optimal_price_segments[segment_code] - (selected_vehicle.ProdCost_t - self.production_subsidy)
                        else:
                            profit_per_sale = selected_vehicle.optimal_price_segments[segment_code] - (selected_vehicle.ProdCost_t)

                        #profit_per_sale = selected_vehicle.optimal_price_segments[segment_code] - selected_vehicle.ProdCost_t

                        I_s_t = market_data[segment_code]["I_s_t"]
                        U_sum = market_data[segment_code]["U_sum"]
                        utility_proportion = (
                            (selected_vehicle.car_utility_segments_U[segment_code] ** self.kappa)
                            / ((U_sum + selected_vehicle.car_utility_segments_U[segment_code]) ** self.kappa)
                        )
                        raw_profit = profit_per_sale * I_s_t * utility_proportion

                        if selected_vehicle.transportType == 3:  # EV
                            updated_profit = raw_profit
                        else:  # ICE
                            updated_profit = raw_profit * (1 - self.discriminatory_corporate_tax)

                        profit_matrix[i, max_col] = updated_profit

            # Final selection of vehicles with the highest expected profits
            vehicle_to_max_profit = {}

            # Track the highest profit and corresponding segment for each vehicle
            for vehicle, profit, segment in vehicles_selected_profits:
                if vehicle not in vehicle_to_max_profit or profit > vehicle_to_max_profit[vehicle]["profit"]:
                    vehicle_to_max_profit[vehicle] = {"profit": profit, "segment": segment}

            # Sort vehicles by their maximum profit in descending order
            sorted_vehicles = sorted(
                vehicle_to_max_profit.items(), key=lambda x: x[1]["profit"], reverse=True
            )

            if len(selected_vehicles) > self.num_cars_production:
                # Select up to the allowed number of cars
                vehicles_selected_with_segment = [
                    (x[0], x[1]["segment"]) for x in sorted_vehicles[:self.num_cars_production]
                ]
                vehicles_selected = [
                    x[0] for x in sorted_vehicles[:self.num_cars_production]
                ]
            else:
                vehicles_selected_with_segment = [
                    (x[0], x[1]["segment"]) for x in sorted_vehicles
                ]
                vehicles_selected = [
                    x[0] for x in sorted_vehicles
                ]

            if self.save_timeseries_data_state and (self.t_firm % self.compression_factor_state == 0):
                # Count segments of selected vehicles
                self.selected_vehicle_segment_counts = {}

                for vehicle, segment_code in vehicles_selected_with_segment:
                    if segment_code not in self.selected_vehicle_segment_counts:
                        self.selected_vehicle_segment_counts[segment_code] = 0
                    self.selected_vehicle_segment_counts[segment_code] += 1

        return vehicles_selected

    def add_new_vehicle(self, vehicle_model_research):

        #add the vehicle
        if vehicle_model_research.transportType == 2:
            if vehicle_model_research not in self.list_technology_memory_ICE:
                self.list_technology_memory_ICE.append(vehicle_model_research)
        else:
            if vehicle_model_research not in self.list_technology_memory_EV:
                self.list_technology_memory_EV.append(vehicle_model_research)
                
    def update_memory_timer(self):
        #change the timer for the techs that are not the ones being used
        if self.ev_research_bool:
            for technology in self.list_technology_memory_EV:
                if not technology.choosen_tech_bool:
                    technology.update_timer()

        for technology in self.list_technology_memory_ICE:
            if not technology.choosen_tech_bool:
                technology.update_timer()

    def update_memory_len(self):
        #is the memory list is too long then remove data
        if len(self.list_technology_memory) > self.memory_cap:
            tech_to_remove = max((tech for tech in self.list_technology_memory if not tech.choosen_tech_bool), key=lambda x: x.timer, default=None)#PICK TECH WITH MAX TIMER WHICH IS NOT ACTIVE
            if tech_to_remove.transportType == 3:
                if tech_to_remove in self.list_technology_memory_EV:
                    self.list_technology_memory_EV.remove(tech_to_remove)
                else:
                    print(tech_to_remove)
                    raise ValueError("Tech being removed without being in the EV list")
            else:
                if tech_to_remove in self.list_technology_memory_ICE:
                    self.list_technology_memory_ICE.remove(tech_to_remove)
                else:
                    print(tech_to_remove)
                    raise ValueError("Tech being removed without being in the ICE list")
                
    def innovate(self, market_data):
        # create a list of cars in neighbouring memory space                                #self, last_researched_car,  list_technology_memory,        list_technology_memory_neighbouring, landscape, parameters_car, transportType

        self.unique_neighbouring_technologies_ICE = self.generate_neighbouring_technologies(self.last_researched_car_ICE,  self.list_technology_memory_ICE, self.ICE_landscape, self.parameters_car_ICE, transportType = 2)

        if self.ev_research_bool:
            self.unique_neighbouring_technologies_EV = self.generate_neighbouring_technologies(self.last_researched_car_EV,  self.list_technology_memory_EV, self.EV_landscape, self.parameters_car_EV, transportType = 3 )
            self.unique_neighbouring_technologies = self.unique_neighbouring_technologies_EV + self.unique_neighbouring_technologies_ICE + [self.last_researched_car_EV, self.last_researched_car_ICE]
        else:
            self.unique_neighbouring_technologies = self.unique_neighbouring_technologies_ICE +  [self.last_researched_car_ICE]

        # update the prices of models to consider        
        self.update_prices_and_emissions_intensity(self.unique_neighbouring_technologies)
        # calculate the optimal price of cars in the memory 
        car_list = self.calc_optimal_price_cars(market_data, self.unique_neighbouring_technologies)
        # calculate the utility of car segements
        self.calc_utility_cars_segments(market_data, self.unique_neighbouring_technologies)
        # calc the predicted profits of cars
        expected_profits_segments = self.calc_predicted_profit_segments_research(market_data, self.unique_neighbouring_technologies)
        # select the car to innovate
        self.vehicle_model_research = self.select_car_lambda_research(expected_profits_segments)


        if self.vehicle_model_research.transportType == 3:#EV
            self.last_researched_car_EV = self.vehicle_model_research

            #OPTIMIZATION OF RESEARCH SUBSIDY
            self.policy_distortion += self.research_subsidy
        else:
            self.last_researched_car_ICE = self.vehicle_model_research

        #add vehicle to memory, MAKE THE MEMORY INTO A CAR HERE?
        self.add_new_vehicle(self.vehicle_model_research)

        # adjust memory bank
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
    
    def gen_neighbour_strings(self, memory_list, last_researched_car):
        # Initialize a list to store unique neighboring technology strings
        unique_neighbouring_technologies_strings = []

        # Add inverted technology strings from the last researched car, what technologies are adjacent to the car you just researched
        for tech_string in last_researched_car.inverted_tech_strings:
            if tech_string not in unique_neighbouring_technologies_strings:#only add then if they arent already in the list, this really shoulldnt be activated
                unique_neighbouring_technologies_strings.append(tech_string)

        # Get strings from the memory ie the ones you all ready know
        list_technology_memory_strings = [vehicle.component_string for vehicle in memory_list]

        # Remove overlapping strings
        result = []
        for tech in unique_neighbouring_technologies_strings:
            if tech not in list_technology_memory_strings:
                result.append(tech)

        return result

    def generate_neighbouring_technologies(self, last_researched_car, list_technology_memory, landscape, parameters_car, transportType):
        """Generate neighboring technologies for cars. Roaming point"""
        # Set to track unique neighboring technology strings

        string_list = self.gen_neighbour_strings(list_technology_memory, last_researched_car)

        self.neighbouring_technologies = self.gen_neighbour_carsModel(string_list, landscape, parameters_car, transportType)

        return self.neighbouring_technologies
    
    ########################################################################################################################################

    def choose_cars_segments(self, market_data):
        """
        market_data: includes data on the population of each of the segments and the differnet values of the preferences/sensitivity of those segments
        """

        if self.ev_production_bool:
            self.list_technology_memory = self.list_technology_memory_EV + self.list_technology_memory_ICE 
        else:
            self.list_technology_memory = self.list_technology_memory_ICE 

        self.update_prices_and_emissions_intensity(self.list_technology_memory)#UPDATE TECHNOLOGY WITH NEW PRICES FOR PRODUCTION SELECTION

        car_list = self.calc_optimal_price_cars(market_data,  self.list_technology_memory)#calc the optimal price of all cars, also does the utility and distance!
        
        self.calc_utility_cars_segments(market_data, self.list_technology_memory)#utility of all the cars for all the segments
        
        expected_profits = self.calc_predicted_profit_segments_production(market_data, self.list_technology_memory)#calculte the predicted profit of each segment 

        #cars_selected = self.select_car_lambda_production_old(expected_profits)#pick the cars for each car
        cars_selected = self.select_car_lambda_production_alt(market_data, car_list)#pick the cars for each car

        #DO THIS FASTER
        if self.ev_production_bool:
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
 
####################################################################################################

    def set_up_time_series_firm(self):
        self.history_profit = []
        self.history_firm_cars_users = []
        self.history_attributes_researched = []
        self.history_research_type = []
        self.history_num_cars_on_sale = []
        self.history_segment_production_counts = []

    def save_timeseries_data_firm(self):
        self.history_profit.append(self.firm_profit)
        self.history_firm_cars_users.append(self.firm_cars_users)
        self.history_num_cars_on_sale.append(len(self.cars_on_sale))
        if self.production_change_bool == 1:
            self.history_segment_production_counts.append(self.selected_vehicle_segment_counts)
        else:
            self.selected_vehicle_segment_counts = {}
            for segment_code in range(8):
                segment_code_str = format(segment_code, '03b')
                self.selected_vehicle_segment_counts[segment_code_str] = np.nan
            self.history_segment_production_counts.append(self.selected_vehicle_segment_counts)

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
    
    def update_prices_and_emissions_intensity(self, list_cars):
        for car in list_cars:
            if car.transportType == 2:#ICE
                car.fuel_cost_c = self.gas_price
            elif car.transportType == 3:
                car.fuel_cost_c = self.electricity_price
                car.e_t = self.electricity_emissions_intensity

    def next_step(self, market_data, carbon_price, gas_price, electricity_price, electricity_emissions_intensity, rebate, discriminatory_corporate_tax, production_subsidy, research_subsidy):
        self.t_firm += 1

        self.carbon_price = carbon_price
        self.gas_price =  gas_price
        self.electricity_price = electricity_price
        self.electricity_emissions_intensity = electricity_emissions_intensity
        self.rebate = rebate
        self.discriminatory_corporate_tax =  discriminatory_corporate_tax
        self.production_subsidy = production_subsidy
        self.research_subsidy = research_subsidy

        self.update_prices_and_emissions_intensity(self.cars_on_sale)#update the prices of cars on sales with changes, this is required for calculations made by users

        #update cars to sell
        if self.random_state.rand() < self.prob_change_production:
            self.cars_on_sale = self.choose_cars_segments(market_data)
            self.production_change_bool = 1

        self.update_memory_timer()
        if self.random_state.rand() < self.prob_innovate:

            self.innovate(market_data)
            self.research_bool = 1#JUST USED FOR THE SAVE TIME SERIES DAT

        if self.save_timeseries_data_state and (self.t_firm % self.compression_factor_state == 0):
            self.save_timeseries_data_firm()
            self.research_bool = 0

        return self.cars_on_sale

