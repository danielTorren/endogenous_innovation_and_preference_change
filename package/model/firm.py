import numpy as np
from scipy.special import lambertw
from package.model.carModel import CarModel

class Firm:
    def __init__(self, firm_id, init_tech_ICE, init_tech_EV, parameters_firm, parameters_car_ICE, parameters_car_EV, innovation_seed):
        

        self.rebate = parameters_firm["rebate"]#7000#JUST TO TRY TO GET TRANSITION
        self.rebate_calibration = parameters_firm["rebate_calibration"]
        

        self.t_firm = 0
        self.production_change_bool = 0
        self.policy_distortion = 0
        self.zero_profit_options_prod = 0
        self.zero_profit_options_research = 0

        #DELETE THIS LATER
        self.prod_counter = 0
        self.research_counter = 0

        self.segment_codes = parameters_firm["segment_codes"]
        
        self.save_timeseries_data_state = parameters_firm["save_timeseries_data_state"]
        self.compression_factor_state = parameters_firm["compression_factor_state"]
        self.id_generator = parameters_firm["IDGenerator_firms"]  

        self.d_mean = parameters_firm["d_mean"]  
        self.alpha = parameters_firm["alpha"]

        self.firm_id = firm_id
        #ICE
        self.init_tech_ICE = init_tech_ICE
        self.init_tech_ICE.firm = self
        self.init_tech_ICE.unique_id = self.id_generator.get_new_id()
        self.list_technology_memory_ICE = [self.init_tech_ICE ]
        self.last_researched_car_ICE = self.init_tech_ICE 

        #EV
        self.init_tech_EV = init_tech_EV
        self.init_tech_EV.firm = self
        self.init_tech_EV.unique_id = self.id_generator.get_new_id()
        self.list_technology_memory_EV = [self.init_tech_EV]
        self.last_researched_car_EV = self.init_tech_EV
        
        self.ev_research_bool = parameters_firm["ev_research_bool"]
        self.ev_production_bool = parameters_firm["ev_production_bool"]

        self.firm_profit = 0
        self.firm_cars_users = 0
        self.research_bool = 0
        self.EVs_sold = 0

        self.parameters_firm = parameters_firm

        self.kappa = self.parameters_firm["kappa"]
        self.memory_cap = self.parameters_firm["memory_cap"]
        self.prob_innovate = self.parameters_firm["prob_innovate"]
        self.prob_change_production = self.parameters_firm["prob_change_production"]
        self.r = self.parameters_firm["r"]
        #self.delta = self.parameters_firm["delta"]

        self.min_profit = self.parameters_firm["min profit"]

        self.U_segments_init = self.parameters_firm["U_segments_init"]
        
        self.init_price_multiplier = self.parameters_firm["init_price_multiplier"]
        
        self.carbon_price =  self.parameters_firm["carbon_price"]

        self.lambda_exp = parameters_firm["lambda"]

        self.universal_model_repo_ICE = parameters_firm["universal_model_repo_ICE"]#THIS NEEDS TO BE SHARED AMONGST ALL FIRMS
        self.universal_model_repo_EV = parameters_firm["universal_model_repo_EV"]#THIS NEEDS TO BE SHARED AMONGST ALL FIRMS

        self.ICE_landscape = self.parameters_firm["ICE_landscape"]
        self.EV_landscape = self.parameters_firm["EV_landscape"]

        self.parameters_car_ICE = parameters_car_ICE
        self.parameters_car_EV = parameters_car_EV

        self.expected_profits_segments = {}      
        
        if self.ev_production_bool:
            self.cars_on_sale = [self.init_tech_ICE, self.init_tech_EV]
        else:
            self.cars_on_sale = [self.init_tech_ICE] 

        self.set_car_init_price_and_base_U()
        

        if self.save_timeseries_data_state:
            self.set_up_time_series_firm()

        self.random_state = np.random.RandomState(innovation_seed)  # Local random state
    
    def set_car_init_price_and_base_U(self):
        for car in self.cars_on_sale:
            car.price = car.ProdCost_t*self.init_price_multiplier
            for segment_code in self.segment_codes:
                # Add data for the segment
                car.optimal_price_segments[segment_code] = car.price
                #car.B_segments[segment_code] = self.B_segments_init

        #need to do EV IN MEMORY FOR THE FIRST STEP as well
        for car in self.list_technology_memory_EV:
            car.price = car.ProdCost_t*self.init_price_multiplier
            for segment_code in self.segment_codes:
                # Add data for the segment
                car.optimal_price_segments[segment_code] = car.price
                #car.B_segments[segment_code] = self.B_segments_init

    def calc_init_U_segments(self, market_data):
        for segment_code, segment_data in market_data.items():          
            # Unpack the tuple
            b_idx, g_idx, e_idx = segment_code  # if your codes are (b, g, e)
            for car in self.cars_on_sale:
                if (car.transportType == 2) or (e_idx == 1 and car.transportType == 3):
                    car.car_utility_segments_U[segment_code] = self.U_segments_init#SET AS FIXED CONSTANT TO START FIRST TURN
                else:
                    car.car_utility_segments_U[segment_code] = -np.inf

    def calc_utility_prop(self,U,W, nu_maxU):
        
        #print("bits",self.kappa*U, self.kappa*self.nu_maxU, W)
        utility_proportion = np.exp(self.kappa*U - self.kappa*nu_maxU)/(np.exp(-self.kappa*nu_maxU)*W + np.exp(self.kappa*U - self.kappa*nu_maxU))
        #utility_proportion = np.exp(self.kappa*U)/(W + np.exp(self.kappa*U))

        return utility_proportion

    def calc_utility(self, Q, beta, gamma, c, omega, e, E_new, P_adjust, delta):
        #U = self.d_mean*(Q**self.alpha)*((1+self.r)/(self.r-self.delta)) - beta*(self.d_mean*c/(self.r*omega) + P_adjust) - gamma*(self.d_mean*e/(self.r*omega) + E_new)
        U = self.d_mean*(Q**self.alpha)*((1+self.r)/(self.r - (1 - delta)**self.alpha + 1)) - beta*(self.d_mean*c*(1+self.r)/(self.r*omega) + P_adjust) - gamma*(self.d_mean*e*(1+self.r)/(self.r*omega) + E_new)
        return U
    
    def calc_optimal_price_cars(self, market_data, car_list): 
        """Calculate the optimal price for each car in the car list based on market data. USED WHEN FIRST STUDYING A CAR"""

        for car in car_list:
            E_m = car.emissions  # Emissions for the current car                                                  
            C_m = car.ProdCost_t  
            C_m_cost = car.ProdCost_t  
            C_m_price = car.ProdCost_t 
            delta = car.delta

            #UPDATE EMMISSION AND PRICES, THIS WORKS FOR BOTH PRODUCTION AND INNOVATION
            if car.transportType == 3:#EV
                #C_m  = car.ProdCost_t - self.production_subsidy
                C_m_cost  = np.maximum(0,car.ProdCost_t - self.production_subsidy)
                C_m_price = np.maximum(0,car.ProdCost_t - (self.production_subsidy + self.rebate + self.rebate_calibration))
                #C_m_price = np.maximum(0,car.ProdCost_t - (self.production_subsidy))

            # Iterate over each market segment to calculate utilities and distances
            for segment_code, segment_data in market_data.items():
                beta_s = segment_data["beta_s_t"]
                gamma_s = segment_data["gamma_s_t"]
                W_s_t = segment_data["W"]
                nu_maxU = segment_data["nu_maxU"]
                
                term = self.kappa*(self.d_mean*(car.Quality_a_t**self.alpha)*((1+self.r)/(self.r - (1 - delta)**self.alpha + 1)) - beta_s*self.d_mean*car.fuel_cost_c*(1+self.r)/(self.r*car.Eff_omega_a_t) - gamma_s*(self.d_mean*car.e_t*(1+self.r)/(self.r*car.Eff_omega_a_t) + E_m) - beta_s*C_m_price) - 1.0 

                log_term = term - np.log(W_s_t)
                Arg = np.exp(log_term)

                LW   = lambertw(Arg, 0).real  # principal branch
                
                P = C_m_cost + (1.0 + LW)/(self.kappa*beta_s)

                if P < C_m:
                    print(P,C_m, Arg)
                    raise ValueError("P LESS THAN C")
                #CHECK THAT THIS IS POSITIVE AND MORE THAN C
                car.optimal_price_segments[segment_code] = P

        return car_list
    
    def calc_utility_cars_segments(self, market_data, vehicle_list):
        for segment_code, segment_data in market_data.items():
            beta_s =  segment_data["beta_s_t"]
            gamma_s = segment_data["gamma_s_t"]
            # Unpack the tuple
            b_idx, g_idx, e_idx = segment_code  # if your codes are (b, g, e)
            for car in vehicle_list:
                if (car.transportType == 2) or (e_idx == 1 and car.transportType == 3):
                    price_s = car.optimal_price_segments[segment_code]#price for that specific segment
                    #ADD IN A SUBSIDY
                    if car.transportType == 3:
                        price_adjust = np.maximum(0,price_s - (self.rebate + self.rebate_calibration))
                    else:
                        price_adjust = price_s                                                  
                    car.car_utility_segments_U[segment_code] =  self.calc_utility(car.Quality_a_t, beta_s, gamma_s, car.fuel_cost_c, car.Eff_omega_a_t, car.e_t, car.emissions, price_adjust, car.delta)
                else:
                    car.car_utility_segments_U[segment_code] = -np.inf

        return vehicle_list

    ########################################################################################################################################
    #INNOVATION
    def innovate(self, market_data):
        # create a list of cars in neighbouring memory space                       
        unique_neighbouring_technologies_ICE = self.generate_neighbouring_technologies(self.last_researched_car_ICE,  self.list_technology_memory_ICE, self.ICE_landscape, self.parameters_car_ICE, transportType = 2)

        if self.ev_research_bool:
            unique_neighbouring_technologies_EV = self.generate_neighbouring_technologies(self.last_researched_car_EV,  self.list_technology_memory_EV, self.EV_landscape, self.parameters_car_EV, transportType = 3 )
            unique_neighbouring_technologies = unique_neighbouring_technologies_EV + unique_neighbouring_technologies_ICE + [self.last_researched_car_EV, self.last_researched_car_ICE]
        else:
            unique_neighbouring_technologies = unique_neighbouring_technologies_ICE +  [self.last_researched_car_ICE]

        
        # update the prices of models to consider        
        unique_neighbouring_technologies = self.update_prices_and_emissions_intensity(unique_neighbouring_technologies)
        # calculate the optimal price of cars in the memory 
        unique_neighbouring_technologies = self.calc_optimal_price_cars(market_data, unique_neighbouring_technologies)
        # calculate the utility of car segements
        unique_neighbouring_technologies = self.calc_utility_cars_segments(market_data, unique_neighbouring_technologies)
        # calc the predicted profits of cars
        #unique_neighbouring_technologies, expected_profits_segments = self.calc_predicted_profit_segments_research(market_data, unique_neighbouring_technologies)
        
        unique_neighbouring_technologies = self.calc_predicted_profit_segments_research(market_data, unique_neighbouring_technologies)
        
        self.vehicle_model_research = self.select_car_lambda_research(unique_neighbouring_technologies)

        if self.vehicle_model_research.transportType == 3:#EV
            self.last_researched_car_EV = self.vehicle_model_research

            #OPTIMIZATION OF RESEARCH SUBSIDY
            self.policy_distortion += self.research_subsidy
        else:
            self.last_researched_car_ICE = self.vehicle_model_research

        #add vehicle to memory, MAKE THE MEMORY INTO A CAR HERE?
        self.add_new_vehicle_memory(self.vehicle_model_research)

        # adjust memory bank
        self.update_memory_len()

    def calc_predicted_profit_segments_research(self, market_data, car_list):
        """
        THIS INCLUDES THE FLAT SUBSIDY FOR EV RESEARCH!
        Calculate the expected profit for each segment, with segments that consider EVs able to buy both EVs and ICE cars, 
        and segments that do not consider EVs only able to buy ICE cars.
        """

        # Loop through each vehicle in car_list to calculate profit for each segment
        for vehicle in car_list:
            is_ev = vehicle.transportType == 3  # Assuming transportType 3 is EV; adjust as needed

            # Calculate profit for each segment
            for segment_code, segment_data in market_data.items():

                b_idx, g_idx, e_idx = segment_code
                consider_ev = (e_idx == 1) # Determine if the segment considers EVs

                # For segments that consider EVs, calculate profit for both EV and ICE vehicles
                if (is_ev == False) or (consider_ev == True):# if its an ice car, or people dont consider evs
                    # Include EV only if the segment considers EV, always include ICE                    

                    # Calculate profit for this vehicle and segment
                    if is_ev:#PRODUCTION SUBSIDY
                        profit_per_sale = vehicle.optimal_price_segments[segment_code] - np.maximum(0,vehicle.ProdCost_t - self.production_subsidy) # + self.carbon_price*vehicle.emissions 
                    else:
                        profit_per_sale = vehicle.optimal_price_segments[segment_code] - vehicle.ProdCost_t
                    
                    I_s_t = segment_data["I_s_t"]  # Size of individuals in the segment at time t
                    W = segment_data["W"]
                    
                    # Expected profit calculation
                    utility_value = vehicle.car_utility_segments_U[segment_code]

                    #print("utility_value",utility_value)

                    if utility_value == -np.inf:
                        utility_proportion = 0
                        raw_profit = 0
                    else:
                        #utility_proportion = np.exp(self.kappa*utility_value - self.nu_maxU)/(W + np.exp(self.kappa*utility_value - self.nu_maxU))
                        nu_maxU = segment_data["nu_maxU"]
                        utility_proportion = self.calc_utility_prop(utility_value,W, nu_maxU)
                        #print("utility_proportion", utility_proportion)
                        raw_profit = profit_per_sale * I_s_t * utility_proportion

                    if is_ev:
                        expected_profit = raw_profit + self.research_subsidy
                    else:
                        expected_profit = raw_profit*(1-self.discriminatory_corporate_tax)

                    # Store profit in the vehicle's expected profit attribute and update the main dictionary
                    vehicle.expected_profit_segments[segment_code] = expected_profit 
                else:#THIS SEGEMNT CANT BUY EVS
                    # Store profit in the vehicle's expected profit attribute and update the main dictionary
                    expected_profit = self.research_subsidy
                    vehicle.expected_profit_segments[segment_code] = expected_profit 
                #print("expected_profit ",expected_profit )
        return  car_list
    
    def select_car_lambda_research(self, car_list):
        """
        Probabilistically select a vehicle for research, where the probability of selecting a vehicle
        is proportional to its expected profit, rather than always selecting the one with the highest profit.

        Parameters:
        - expected_profits_segments (dict): A dictionary containing segment data with vehicles and their expected profits.

        Returns:
        - CarModel: The vehicle selected for research.
        """

        # Dictionary to store the highest profit for each vehicle
        
        #PICK OUT EACH CARS BEST SEGMENT
        profits = []

        #num_ev = sum(1 for car in car_list if car.transportType == 3)
        #print("num ev ad prop: ",num_ev, num_ev/len(car_list))

        for vehicle in car_list:
            # Calculate profit for each segment
            max_profit = 0
            #if vehicle.transportType == 3:
            #    print("profts car", list(vehicle.expected_profit_segments.values()))
            for segment_code, segment_profit in vehicle.expected_profit_segments.items():
                if segment_profit > max_profit:
                    max_profit = segment_profit
            profits.append(max_profit)
            
        # Convert profits list to numpy array
        profits = np.array(profits)

        #ev_profit = sum(profit for car, profit in zip(car_list, profits) if car.transportType == 3)
        #print("ev_profit", ev_profit)

        len_vehicles = len(car_list)  # Length of the car list

        if np.sum(profits) == 0:#ALL TECHH HAS 0 UTILITY DUE TO BEIGN VERY BAD, exp caps out
            #print("0 profit", profits)
            self.zero_profit_options_research = 1
            selected_index = self.random_state.choice(len_vehicles)
        else:
            #print("proftis raw", profits)
            profits[profits == 0] = -np.inf#if pofit is zero you cant choose it

            # Compute the softmax probabilities
            lambda_profits = np.zeros_like(profits)
            valid_profits_mask = profits != -np.inf
            #print("valid_profits_mask", valid_profits_mask)

            exp_input = self.lambda_exp*(profits[valid_profits_mask]  - np.max(profits[valid_profits_mask]))
            #print("exp input",exp_input)

            exp_input = np.clip(exp_input, -700, 700)#CLIP TO AVOID OVERFLOWS

            lambda_profits[valid_profits_mask] = np.exp(exp_input)
            #lambda_profits = profits**self.lambda_exp

            #print("lambda_profits", lambda_profits)
            sum_profit = np.sum(lambda_profits)

            self.zero_profit_options_research = 0
            probabilities = lambda_profits/sum_profit

            #ev_probability = sum(prob for car, prob in zip(car_list, probabilities) if car.transportType == 3)
            #print("ev_probability", ev_probability)


            selected_index = self.random_state.choice(len_vehicles, p=probabilities)

        # Select a vehicle based on the computed probabilities
        selected_vehicle = car_list[selected_index]
        #quit()

        return selected_vehicle

    #Memory
    def gen_neighbour_carsModel(self, tech_strings, nk_landscape, parameters_car, transportType):
        # Generate CarModel instances for the unique neighboring technologies
        neighbouring_technologies = []

        if transportType == 2:
            universal_model_repo = self.universal_model_repo_ICE
        else:
            universal_model_repo = self.universal_model_repo_EV

        for tech_string in tech_strings:
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
    
    def add_new_vehicle_memory(self, vehicle_model_research):

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

        list_technology_memory_all = list(self.list_technology_memory_EV +self.list_technology_memory_ICE)

        if len(list_technology_memory_all) > self.memory_cap:
            tech_to_remove = max((tech for tech in list_technology_memory_all if not tech.choosen_tech_bool), key=lambda x: x.timer, default=None)#PICK TECH WITH MAX TIMER WHICH IS NOT ACTIVE
            
            # If there's no unchosen tech to remove, do nothing (or handle differently)
            if tech_to_remove is None:
                print("Warning: Memory is full, but no unchosen technology to remove.")
                return  # Or decide on a different policy to remove a chosen tech, etc.
        
            if tech_to_remove.transportType == 3:
                if tech_to_remove in self.list_technology_memory_EV:
                    self.list_technology_memory_EV.remove(tech_to_remove)
                else:
                    raise ValueError("Tech being removed without being in the EV list")
            else:
                if tech_to_remove in self.list_technology_memory_ICE:
                    self.list_technology_memory_ICE.remove(tech_to_remove)
                else:
                    raise ValueError("Tech being removed without being in the ICE list")
                
    ########################################################################################################################################
    #PRODUCTION
    def calc_predicted_profit_segments_production(self, market_data, car_list):
        """
        Calculate the expected profit for each segment, with segments that consider EVs able to buy both EVs and ICE cars, 
        and segments that do not consider EVs only able to buy ICE cars.
        """

        # Loop through each vehicle in car_list to calculate profit for each segment
        for vehicle in car_list:
            is_ev = vehicle.transportType == 3  # Assuming transportType 3 is EV; adjust as needed

            # Calculate profit for each segment
            for segment_code, segment_data in market_data.items():
                b_idx, g_idx, e_idx = segment_code
                consider_ev = (e_idx == 1) # Determine if the segment considers EVs

                # For segments that consider EVs, calculate profit for both EV and ICE vehicles
                if (is_ev == False) or (consider_ev == True):  # Include EV only if the segment considers EV, always include ICE                    

                    # Calculate profit for this vehicle and segment
                    if is_ev:#PRODUCTION SUBSIDY
                        profit_per_sale = vehicle.optimal_price_segments[segment_code] - np.maximum(0,vehicle.ProdCost_t - self.production_subsidy) # + self.carbon_price*vehicle.emissions 
                    else:
                        profit_per_sale = vehicle.optimal_price_segments[segment_code] - np.maximum(0,vehicle.ProdCost_t)

                    I_s_t = segment_data["I_s_t"]  # Size of individuals in the segment at time t
                    W = segment_data["W"]
                    
                    # Expected profit calculation
                    utility_car = vehicle.car_utility_segments_U[segment_code]#max(0,vehicle.car_utility_segments_U[segment_code])

                    #utility_proportion = np.exp(self.kappa*utility_car - self.nu_maxU)/(W + np.exp(self.kappa*utility_car - self.nu_maxU))
                    nu_maxU = segment_data["nu_maxU"]
                    utility_proportion = self.calc_utility_prop(utility_car,W, nu_maxU)

                    raw_profit = profit_per_sale * I_s_t * utility_proportion

                    if is_ev:
                        expected_profit = raw_profit
                    else:
                        expected_profit = raw_profit*(1-self.discriminatory_corporate_tax)

                    # Store profit in the vehicle's expected profit attribute and update the main dictionary
                    vehicle.expected_profit_segments[segment_code] = expected_profit
                else:
                    # Store profit in the vehicle's expected profit attribute and update the main dictionary
                    expected_profit = 0
                    vehicle.expected_profit_segments[segment_code] = expected_profit

        return car_list

    def set_utility_and_profit_grid_production(self, selected_vehicle, market_data, segment_code):

        beta_s = market_data[segment_code]["beta_s_t"]
        gamma_s = market_data[segment_code]["gamma_s_t"]
        b_idx, g_idx, e_idx = segment_code

        if (selected_vehicle.transportType == 2) or (e_idx == 1 and selected_vehicle.transportType == 3):
            price_s = selected_vehicle.optimal_price_segments[segment_code]
            if selected_vehicle.transportType == 3:
                price_adjust = np.maximum(0,price_s - (self.rebate + self.rebate_calibration))
            else:
                price_adjust = price_s
            selected_vehicle.car_utility_segments_U[segment_code] = self.calc_utility(selected_vehicle.Quality_a_t, beta_s, gamma_s, selected_vehicle.fuel_cost_c, selected_vehicle.Eff_omega_a_t, selected_vehicle.e_t, selected_vehicle.emissions, price_adjust, selected_vehicle.delta)
        else:
            selected_vehicle.car_utility_segments_U[segment_code] = -np.inf

        if selected_vehicle.transportType == 3:#PRODUCTION SUBSIDY
            profit_per_sale = selected_vehicle.optimal_price_segments[segment_code] - np.maximum(0,selected_vehicle.ProdCost_t - self.production_subsidy)
        else:
            profit_per_sale = selected_vehicle.optimal_price_segments[segment_code] - np.maximum(0,selected_vehicle.ProdCost_t)

        I_s_t = market_data[segment_code]["I_s_t"]
        W = market_data[segment_code]["W"]

        utility_value = selected_vehicle.car_utility_segments_U[segment_code]

        if utility_value == -np.inf:
            utility_proportion = 0
            raw_profit = 0
        else:
            #utility_proportion = np.exp(self.kappa*utility_value - self.nu_maxU)/(W + np.exp(self.kappa*utility_value - self.nu_maxU))
            nu_maxU = market_data[segment_code]["nu_maxU"]
            utility_proportion = self.calc_utility_prop(utility_value,W, nu_maxU)
            raw_profit = profit_per_sale * I_s_t * utility_proportion

        if selected_vehicle.transportType == 3:  # EV
            updated_profit = raw_profit
        else:  # ICE
            updated_profit = raw_profit * (1 - self.discriminatory_corporate_tax)
        
        return updated_profit


    def select_car_lambda_production(self, market_data, car_list):
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
                profit_matrix[i, j] = car.expected_profit_segments[segment_code]
        
        if not np.any(profit_matrix): #PROFIT IS ALL 0, pick up to 
            self.zero_profit_options_prod = 1
            
            vehicles_selected = technologies

            #NEED TO SET PRICES HERE, WHAT TO DO WHEN ALL THE CARS ARE TERRIBLE, SET THE LOWEST PRICE?
            for i, vehicle in enumerate(vehicles_selected):
                price_options = vehicle.optimal_price_segments.values()
                lowest_price = min(price_options)
                vehicle.price = lowest_price

            if self.save_timeseries_data_state and (self.t_firm % self.compression_factor_state == 0):
                self.selected_vehicle_segment_counts = {}
                for segment_code in self.segment_codes:
                    self.selected_vehicle_segment_counts[segment_code] = np.nan
        else:
            self.zero_profit_options_prod = 0

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
                        selected_vehicle.optimal_price_segments[segment_code] = optimal_price_chosen_segment#UPDATE THE PRICE FOR THE SEGEMENT, THIS LOCKS THE PRICE FOR ALL OTHERS SEGMENTS!
                        updated_profit = self.set_utility_and_profit_grid_production(selected_vehicle, market_data, segment_code)#RECALCULATE TEH UTILITY AND PROFIT BASED ON THIS PRICE
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

    def choose_cars_segments(self, market_data):
        """
        market_data: includes data on the population of each of the segments and the differnet values of the preferences/sensitivity of those segments
        """

        if self.ev_production_bool:
            list_technology_memory_all = self.list_technology_memory_EV + self.list_technology_memory_ICE 
        else:
            list_technology_memory_all = self.list_technology_memory_ICE 

        # Create a shallow copy of the list to keep the list structure independent, THIS STOPS THE MEMORY LIST AND THE CURRENT CARS LIST FROM LINKING!!!
        list_technology_memory_all = list(list_technology_memory_all)

        list_technology_memory_all = self.update_prices_and_emissions_intensity(list_technology_memory_all)#UPDATE TECHNOLOGY WITH NEW PRICES FOR PRODUCTION SELECTION

        list_technology_memory_all = self.calc_optimal_price_cars(market_data,  list_technology_memory_all)#calc the optimal price of all cars, also does the utility and distance!
        
        list_technology_memory_all = self.calc_utility_cars_segments(market_data, list_technology_memory_all)#utility of all the cars for all the segments
        
        list_technology_memory_all = self.calc_predicted_profit_segments_production(market_data, list_technology_memory_all)#calculte the predicted profit of each segment 

        cars_selected = self.select_car_lambda_production(market_data, list_technology_memory_all)#pick the cars for each car

        for car in list_technology_memory_all:#put the cars that are in the list as selected
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
            for segment_code in self.segment_codes:
                self.selected_vehicle_segment_counts[segment_code] = np.nan
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
    
    def update_prices_and_emissions_intensity(self, car_list):
        for car in car_list:
            if car.transportType == 2:#ICE
                car.fuel_cost_c = self.gas_price
            else:#EV
                car.fuel_cost_c = self.electricity_price
                car.e_t = self.electricity_emissions_intensity
        return car_list

    def next_step(self, market_data, carbon_price, gas_price, electricity_price, electricity_emissions_intensity, rebate, discriminatory_corporate_tax, production_subsidy, research_subsidy, rebate_calibration):
        self.t_firm += 1

        self.carbon_price = carbon_price
        self.gas_price =  gas_price
        self.electricity_price = electricity_price
        self.electricity_emissions_intensity = electricity_emissions_intensity
        self.rebate = rebate
        self.rebate_calibration = rebate_calibration
        self.discriminatory_corporate_tax =  discriminatory_corporate_tax
        self.production_subsidy = production_subsidy
        self.research_subsidy = research_subsidy

        self.cars_on_sale = self.update_prices_and_emissions_intensity(self.cars_on_sale)#update the prices of cars on sales with changes, this is required for calculations made by users

        #update cars to sell   
        if self.random_state.rand() < self.prob_change_production:
            self.cars_on_sale = self.choose_cars_segments(market_data)
            self.production_change_bool = 1
            self.prod_counter += 1

        self.update_memory_timer()

        if self.random_state.rand() < self.prob_innovate:
            self.innovate(market_data)
            self.research_bool = 1#JUST USED FOR THE SAVE TIME SERIES DAT
            self.research_counter += 1

        if self.save_timeseries_data_state and (self.t_firm % self.compression_factor_state == 0):
            self.save_timeseries_data_firm()
            self.research_bool = 0

        return self.cars_on_sale

