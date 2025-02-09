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
        self.num_segments = len(self.segment_codes)

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

    def calc_init_U_segments(self):
        for segment_code in self.segment_codes:          
            # Unpack the tuple
            b_idx, g_idx, e_idx = segment_code  # if your codes are (b, g, e)
            for car in self.cars_on_sale:
                if (car.transportType == 2) or (e_idx == 1 and car.transportType == 3):
                    car.car_utility_segments_U[segment_code] = self.U_segments_init#SET AS FIXED CONSTANT TO START FIRST TURN
                else:
                    car.car_utility_segments_U[segment_code] = -np.inf

    def input_beta_gamma_segments(self,beta_s_values, gamma_s_values):
        self.beta_s_values = beta_s_values
        self.gamma_s_values = gamma_s_values

    def calc_utility_prop(self,U,W, nu_maxU):
        utility_proportion = np.exp(self.kappa*U - self.kappa*nu_maxU)/(np.exp(-self.kappa*nu_maxU)*W + np.exp(self.kappa*U - self.kappa*nu_maxU))
        return utility_proportion

    def create_car_data(self, car_list):
        """
        Converts a list of car objects into a dictionary of NumPy arrays for vectorized calculations.

        Args:
            car_list: A list of car objects.  Each car object is assumed to have 
                    attributes like emissions, ProdCost_t, transportType, etc.

        Returns:
            A dictionary where keys are car attribute names (e.g., "emissions", "ProdCost_t")
            and values are NumPy arrays containing the corresponding attribute values for all cars.
            Returns None if car_list is empty or if car objects don't have the expected attributes.
        """

        if not car_list:
            return None

        # Check if car objects have the necessary attributes (you might want to add more checks)
        required_attributes = ["emissions", "ProdCost_t", "transportType", "delta", "Quality_a_t", "Eff_omega_a_t", "e_t", "fuel_cost_c"]
        for attr in required_attributes:
            if not hasattr(car_list[0], attr):
                print(f"Error: Car object does not have attribute '{attr}'")
                return None  # Or raise an exception

        car_data = {}

        for attr in required_attributes:
            # Use a list comprehension to extract the attribute values and then convert to a NumPy array
            car_data[attr] = np.array([getattr(car, attr) for car in car_list])

        return car_data

    def calc_optimal_price_cars(self, car_list, car_data):
        """Fully vectorized calculation of optimal prices."""

        # Convert car data to NumPy arrays (CRITICAL CHANGE)
        E_m = car_data["emissions"]  # Array of emissions for all cars
        C_m = car_data["ProdCost_t"]
        transport_types = car_data["transportType"]
        delta = car_data["delta"]
        Quality_a_t = car_data["Quality_a_t"]
        Eff_omega_a_t = car_data["Eff_omega_a_t"]
        e_t = car_data["e_t"]
        fuel_cost_c = car_data["fuel_cost_c"]

        # Apply EV-specific calculations using boolean indexing
        ev_mask = transport_types == 3  # Boolean mask for EV cars
        C_m_cost = C_m.copy()  # Important: Create a copy to avoid modifying original
        C_m_price = C_m.copy()
        C_m_cost[ev_mask] = np.maximum(0, C_m[ev_mask] - self.production_subsidy)
        C_m_price[ev_mask] = np.maximum(0, C_m[ev_mask] - (self.production_subsidy + self.rebate + self.rebate_calibration))

        # Fully vectorized calculation of 'term'
        term = self.kappa * (self.d_mean * (Quality_a_t[:, np.newaxis]**self.alpha) * ((1 + self.r) / (self.r - (1 - delta[:, np.newaxis])**self.alpha + 1)) \
            - self.beta_s_values[np.newaxis, :] * self.d_mean * fuel_cost_c[:, np.newaxis] * (1 + self.r) / (self.r * Eff_omega_a_t[:, np.newaxis]) \
            - self.gamma_s_values[np.newaxis, :] * (self.d_mean * e_t[:, np.newaxis] * (1 + self.r) / (self.r * Eff_omega_a_t[:, np.newaxis]) + E_m[:, np.newaxis]) \
            - self.beta_s_values[np.newaxis, :] * C_m_price[:, np.newaxis]) - 1.0

        exp_input = term - np.log(self.W_vec[np.newaxis, :])
        np.clip(exp_input, -700, 700, out=exp_input)#CLIP SO DONT GET OVERFLOWS
        Arg = np.exp(exp_input)
        LW = lambertw(Arg, 0).real

        P = C_m_cost[:, np.newaxis] + (1.0 + LW) / (self.kappa * self.beta_s_values[np.newaxis, :])
        
        # Store results in the original car objects (CRITICAL CHANGE)
        for i, car in enumerate(car_list):
                for j, segment_code in enumerate(self.segment_codes):  # Use enumerate directly on the dictionary
                    car.optimal_price_segments[segment_code] = P[i, j]
        return car_list  # Return a dictionary of optimal prices by segment.

    def calc_utility(self, Q, beta, gamma, c, omega, e, E_new, P_adjust, delta):
        #U = self.d_mean*(Q**self.alpha)*((1+self.r)/(self.r-self.delta)) - beta*(self.d_mean*c/(self.r*omega) + P_adjust) - gamma*(self.d_mean*e/(self.r*omega) + E_new)
        U = self.d_mean*(Q**self.alpha)*((1+self.r)/(self.r - (1 - delta)**self.alpha + 1)) - beta*(self.d_mean*c*(1+self.r)/(self.r*omega) + P_adjust) - gamma*(self.d_mean*e*(1+self.r)/(self.r*omega) + E_new)
        return U

    def calc_utility_cars_segments(self, car_list, car_data):
        num_cars = len(car_list)

        U = np.full((num_cars, self.num_segments), -np.inf)

        # Broadcasting car data to match (num_cars, num_segments)
        Q_values = np.tile(car_data["Quality_a_t"][:, np.newaxis], (1, self.num_segments))
        c_values = np.tile(car_data["fuel_cost_c"][:, np.newaxis], (1, self.num_segments))
        omega_values = np.tile(car_data["Eff_omega_a_t"][:, np.newaxis], (1, self.num_segments))
        e_values = np.tile(car_data["e_t"][:, np.newaxis], (1, self.num_segments))
        E_new_values = np.tile(car_data["emissions"][:, np.newaxis], (1, self.num_segments))
        delta_values = np.tile(car_data["delta"][:, np.newaxis], (1, self.num_segments))
        transport_types = np.tile(car_data["transportType"][:, np.newaxis], (1, self.num_segments))

        # Broadcast segment parameters to match (num_cars, num_segments)
        beta_s_broadcast = np.tile(self.beta_s_values, (num_cars, 1))
        gamma_s_broadcast = np.tile(self.gamma_s_values, (num_cars, 1))

        # Initialize P_adjust and valid_mask
        P_adjust_values = np.zeros((num_cars, self.num_segments))
        valid_mask = np.zeros((num_cars, self.num_segments), dtype=bool)

        # Populate valid_mask and P_adjust_values
        for i, car in enumerate(car_list):
            for j, segment_code in enumerate(self.segment_codes):
                if (car.transportType == 2) or (segment_code[2] == 1 and car.transportType == 3):
                    valid_mask[i, j] = True
                    P_adjust_values[i, j] = car.optimal_price_segments.get(segment_code, 0)

        # Apply rebates for EVs
        ev_mask = (transport_types == 3) & valid_mask
        P_adjust_values[ev_mask] = np.maximum(0, P_adjust_values[ev_mask] - (self.rebate + self.rebate_calibration))

        # Calculate utilities only for valid entries
        U[valid_mask] = self.calc_utility(
            Q_values[valid_mask], 
            beta_s_broadcast[valid_mask],
            gamma_s_broadcast[valid_mask],
            c_values[valid_mask],
            omega_values[valid_mask],
            e_values[valid_mask],
            E_new_values[valid_mask],
            P_adjust_values[valid_mask],
            delta_values[valid_mask]
        )

        # Assign calculated utilities back to car objects
        for i, car in enumerate(car_list):
            for j, segment_code in enumerate(self.segment_codes):
                car.car_utility_segments_U[segment_code] = U[i, j]

        return car_list

    ########################################################################################################################################
    #INNOVATION

    def innovate(self):
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
    
        # Preprocess car data once and pass it to subsequent functions
        car_data = self.create_car_data(unique_neighbouring_technologies)

        unique_neighbouring_technologies = self.calc_optimal_price_cars(unique_neighbouring_technologies, car_data)
        # calculate the utility of car segements
        unique_neighbouring_technologies = self.calc_utility_cars_segments(unique_neighbouring_technologies, car_data)
        # calc the predicted profits of cars
        
        unique_neighbouring_technologies = self.calc_predicted_profit_segments_research(unique_neighbouring_technologies, car_data)
        
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


    def calc_predicted_profit_segments_research(self, car_list, car_data):
        """
        THIS INCLUDES THE FLAT SUBSIDY FOR EV RESEARCH!
        Calculate the expected profit for each segment, with segments that consider EVs able to buy both EVs and ICE cars,
        and segments that do not consider EVs only able to buy ICE cars.
        """

        num_cars = len(car_list)

        # Extract necessary car data
        transport_types = car_data["transportType"]#np.array([car.transportType for car in car_list])
        prod_costs = car_data["ProdCost_t"]#np.array([car.ProdCost_t for car in car_list])
        optimal_prices = np.array([
            [car.optimal_price_segments.get(segment_code, 0) for segment_code in self.segment_codes] 
            for car in car_list
        ])
        utilities = np.array([
            [car.car_utility_segments_U.get(segment_code, -np.inf) for segment_code in self.segment_codes] 
            for car in car_list
        ])

        # Extract market data
        consider_ev_mask = np.array([segment_code[2] == 1 for segment_code in self.segment_codes])  # Shape (num_segments,)

        # Create masks
        is_ev_mask = transport_types == 3  # Shape (num_cars,)
        consider_ev_broadcast = np.tile(consider_ev_mask, (num_cars, 1))
        include_vehicle_mask = (~is_ev_mask[:, np.newaxis]) | consider_ev_broadcast

        # Profit per sale
        prod_subsidy_adjustment = np.maximum(0, prod_costs[:, np.newaxis] - self.production_subsidy)
        profit_per_sale = np.where(
            is_ev_mask[:, np.newaxis],
            optimal_prices - prod_subsidy_adjustment,
            optimal_prices - prod_costs[:, np.newaxis]
        )

        # Utility proportion
        utility_proportion = np.where(
            utilities == -np.inf,
            0,
            self.calc_utility_prop(utilities, self.W_vec[np.newaxis, :], self.nu_maxU_vec[np.newaxis, :])
        )

        # Raw profit
        raw_profit = profit_per_sale * self.I_s_t_vec[np.newaxis, :] * utility_proportion

        # Expected profit
        expected_profit = np.where(
            is_ev_mask[:, np.newaxis],
            raw_profit + self.research_subsidy,
            raw_profit * (1 - self.discriminatory_corporate_tax)
        )

        # Apply research subsidy for segments that can't buy EVs
        expected_profit = np.where(
            include_vehicle_mask,
            expected_profit,
            self.research_subsidy
        )

        # Assign expected profits back to cars
        for i, car in enumerate(car_list):
            for j, segment_code in enumerate(self.segment_codes):
                car.expected_profit_segments[segment_code] = expected_profit[i, j]

        return car_list

    def select_car_lambda_research(self, car_list):
        """
        Probabilistically select a vehicle for research, where the probability of selecting a vehicle
        is proportional to its expected profit, rather than always selecting the one with the highest profit.

        Parameters:
        - expected_profits_segments (dict): A dictionary containing segment data with vehicles and their expected profits.

        Returns:
        - CarModel: The vehicle selected for research.
        """
        
        #PICK OUT EACH CARS BEST SEGMENT
        profits = []

        for vehicle in car_list:
            # Calculate profit for each segment
            max_profit = 0
            for segment_code, segment_profit in vehicle.expected_profit_segments.items():
                if segment_profit > max_profit:
                    max_profit = segment_profit
            profits.append(max_profit)
            
        # Convert profits list to numpy array
        profits = np.array(profits)

        len_vehicles = len(car_list)  # Length of the car list

        if np.sum(profits) == 0:#ALL TECHH HAS 0 UTILITY DUE TO BEIGN VERY BAD, exp caps out
            self.zero_profit_options_research = 1
            selected_index = self.random_state.choice(len_vehicles)
        else:
            profits[profits == 0] = -np.inf#if pofit is zero you cant choose it

            # Compute the softmax probabilities
            lambda_profits = np.zeros_like(profits)
            valid_profits_mask = profits != -np.inf
            #print("valid_profits_mask", valid_profits_mask)

            exp_input = self.lambda_exp*(profits[valid_profits_mask]  - np.max(profits[valid_profits_mask]))
            #print("exp input",exp_input)

            exp_input = np.clip(exp_input, -700, 700)#CLIP TO AVOID OVERFLOWS

            lambda_profits[valid_profits_mask] = np.exp(exp_input)
            sum_profit = np.sum(lambda_profits)

            self.zero_profit_options_research = 0
            probabilities = lambda_profits/sum_profit
            selected_index = self.random_state.choice(len_vehicles, p=probabilities)

        selected_vehicle = car_list[selected_index]

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

        if vehicle_model_research.transportType == 2:#ICE
            if vehicle_model_research not in self.list_technology_memory_ICE:
                self.list_technology_memory_ICE.append(vehicle_model_research)
        else:#EV
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

        list_technology_memory_all = list(self.list_technology_memory_EV + self.list_technology_memory_ICE)

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
    def calc_predicted_profit_segments_production(self, car_list, car_data):
        """
        Calculate the expected profit for each segment, with segments that consider EVs able to buy both EVs and ICE cars,
        and segments that do not consider EVs only able to buy ICE cars.
        """
        num_cars = len(car_list)

        # Extract necessary car data
        transport_types = car_data["transportType"]#np.array([car.transportType for car in car_list])
        prod_costs = car_data["ProdCost_t"]#np.array([car.ProdCost_t for car in car_list])
        optimal_prices = np.array([
            [car.optimal_price_segments.get(segment_code, 0) for segment_code in self.segment_codes] 
            for car in car_list
        ])
        utilities = np.array([
            [car.car_utility_segments_U.get(segment_code, -np.inf) for segment_code in self.segment_codes] 
            for car in car_list
        ])

        # Extract market data
        consider_ev_mask = np.array([segment_code[2] == 1 for segment_code in self.segment_codes])  # Shape (num_segments,)

        # Create masks
        is_ev_mask = transport_types == 3  # Shape (num_cars,)
        consider_ev_broadcast = np.tile(consider_ev_mask, (num_cars, 1))
        include_vehicle_mask = (~is_ev_mask[:, np.newaxis]) | consider_ev_broadcast

        # Profit per sale
        prod_subsidy_adjustment = np.maximum(0, prod_costs[:, np.newaxis] - self.production_subsidy)
        profit_per_sale = np.where(
            is_ev_mask[:, np.newaxis],
            optimal_prices - prod_subsidy_adjustment,
            optimal_prices - np.maximum(0, prod_costs[:, np.newaxis])
        )

        # Utility proportion
        utility_proportion = np.where(
            utilities == -np.inf,
            0,
            self.calc_utility_prop(utilities, self.W_vec[np.newaxis, :], self.nu_maxU_vec[np.newaxis, :])
        )

        # Raw profit
        raw_profit = profit_per_sale * self.I_s_t_vec[np.newaxis, :] * utility_proportion

        # Expected profit
        expected_profit = np.where(
            is_ev_mask[:, np.newaxis],
            raw_profit,
            raw_profit * (1 - self.discriminatory_corporate_tax)
        )

        # Apply zero profit for segments that can't buy EVs
        expected_profit = np.where(
            include_vehicle_mask,
            expected_profit,
            0
        )

        # Assign expected profits back to cars
        for i, car in enumerate(car_list):
            for j, segment_code in enumerate(self.segment_codes):
                car.expected_profit_segments[segment_code] = expected_profit[i, j]

        return car_list


    def set_utility_and_profit_grid_production(self, selected_vehicle, segment_codes_reduc, valid_indices):

        beta_s_values = self.beta_s_values[valid_indices]
        gamma_s_values = self.gamma_s_values[valid_indices]
        I_s_t_values = self.I_s_t_vec[valid_indices]
        W_values = self.W_vec[valid_indices]
        nu_maxU_values = self.nu_maxU_vec[valid_indices]

        e_indices = np.array([code[2] for code in segment_codes_reduc])

        valid_segments = (selected_vehicle.transportType == 2) | ((e_indices == 1) & (selected_vehicle.transportType == 3))

        price_s_values = np.array([selected_vehicle.optimal_price_segments[code] for code in segment_codes_reduc])
        price_adjust_values = np.where(
            selected_vehicle.transportType == 3,
            np.maximum(0, price_s_values - (self.rebate + self.rebate_calibration)),
            price_s_values
        )

        utilities = np.full(len(segment_codes_reduc), -np.inf)
        utilities[valid_segments] = self.calc_utility(
            selected_vehicle.Quality_a_t,
            beta_s_values[valid_segments],
            gamma_s_values[valid_segments],
            selected_vehicle.fuel_cost_c,
            selected_vehicle.Eff_omega_a_t,
            selected_vehicle.e_t,
            selected_vehicle.emissions,
            price_adjust_values[valid_segments],
            selected_vehicle.delta
        )

        for idx, code in enumerate(segment_codes_reduc):
            selected_vehicle.car_utility_segments_U[code] = utilities[idx]

        profit_per_sale = np.where(
            selected_vehicle.transportType == 3,
            price_s_values - np.maximum(0, selected_vehicle.ProdCost_t - self.production_subsidy),
            price_s_values - np.maximum(0, selected_vehicle.ProdCost_t)
        )

        utility_proportions = np.where(
            utilities == -np.inf,
            0,
            self.calc_utility_prop(utilities, W_values, nu_maxU_values)
        )

        raw_profits = profit_per_sale * I_s_t_values * utility_proportions

        updated_profits = np.where(
            selected_vehicle.transportType == 3,
            raw_profits,
            raw_profits * (1 - self.discriminatory_corporate_tax)
        )

        return updated_profits
    
    def select_car_lambda_production(self, car_list):
        """
        Select vehicles for production based on a matrix of expected profits for each segment and technology combination.
        Iteratively selects the most profitable combination, removes the segment, and updates the technology column.

        Parameters:
        - car_list (list): List of CarModel objects.

        Returns:
        - list (tuple): List of tuples containing selected vehicles and their associated chosen segments.
        """
        # Step 0: Build the profit matrix (segments x technologies)
       
        technologies = car_list
        profit_matrix = np.zeros((self.num_segments, len(technologies)))

        vehicles_selected_profits = []  # Used to track the vehicle and the expected profit
        vehicle_segments = []  # To keep track of vehicle segment which it was targeting

        # Populate the profit matrix with expected profits
        for i, segment_code in enumerate(self.segment_codes):
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
            for _ in range(self.num_segments):
                # Step 1: Choose the highest value square in the matrix
                max_index = np.unravel_index(np.argmax(profit_matrix, axis=None), profit_matrix.shape)
                max_row, max_col = max_index
                max_profit = profit_matrix[max_row, max_col]

                # Select the car and mark the segment as covered
                selected_vehicle = technologies[max_col]
                vehicles_selected_profits.append((selected_vehicle, max_profit, self.segment_codes[max_row]))
                vehicle_segments.append((selected_vehicle, self.segment_codes[max_row]))

                # Step 2: Remove the segment (set the row to negative infinity)
                profit_matrix[max_row, :] = -np.inf

                # Step 3: Update the column for the selected technology using the optimal price for the chosen segment
                optimal_price_chosen_segment = selected_vehicle.optimal_price_segments[self.segment_codes[max_row]]
                selected_vehicle.price = optimal_price_chosen_segment  # SET PRICE

                valid_indices = np.where(profit_matrix[:, max_col] != -np.inf)[0]
                valid_segment_codes = [self.segment_codes[i] for i in valid_indices]

                for segment_code in valid_segment_codes:
                    selected_vehicle.optimal_price_segments[segment_code] = optimal_price_chosen_segment

                updated_profits = self.set_utility_and_profit_grid_production(selected_vehicle, valid_segment_codes, valid_indices)

                profit_matrix[valid_indices, max_col] = updated_profits

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


    def choose_cars_segments(self):

        if self.ev_production_bool:
            list_technology_memory_all = self.list_technology_memory_EV + self.list_technology_memory_ICE 
        else:
            list_technology_memory_all = self.list_technology_memory_ICE 

        # Create a shallow copy of the list to keep the list structure independent, THIS STOPS THE MEMORY LIST AND THE CURRENT CARS LIST FROM LINKING!!!
        list_technology_memory_all = list(list_technology_memory_all)

        list_technology_memory_all = self.update_prices_and_emissions_intensity(list_technology_memory_all)#UPDATE TECHNOLOGY WITH NEW PRICES FOR PRODUCTION SELECTION

        # Preprocess car data once and pass it to subsequent functions
        car_data = self.create_car_data(list_technology_memory_all)

        list_technology_memory_all = self.calc_optimal_price_cars(list_technology_memory_all, car_data)#calc the optimal price of all cars, also does the utility and distance!
        
        list_technology_memory_all = self.calc_utility_cars_segments(list_technology_memory_all, car_data)#utility of all the cars for all the segments
        
        list_technology_memory_all = self.calc_predicted_profit_segments_production(list_technology_memory_all, car_data)#calculte the predicted profit of each segment 

        cars_selected = self.select_car_lambda_production( list_technology_memory_all)#pick the cars for each car

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

    def next_step(self, I_s_t_vec, W_vec, nu_UMax_vec, carbon_price, gas_price, electricity_price, electricity_emissions_intensity, rebate, discriminatory_corporate_tax, production_subsidy, research_subsidy, rebate_calibration):
        self.t_firm += 1

        self.I_s_t_vec = I_s_t_vec
        self.W_vec =  W_vec
        self.nu_maxU_vec = nu_UMax_vec
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
            self.cars_on_sale = self.choose_cars_segments()
            self.production_change_bool = 1
            self.prod_counter += 1

        self.update_memory_timer()

        if self.random_state.rand() < self.prob_innovate:
            self.innovate()
            self.research_bool = 1#JUST USED FOR THE SAVE TIME SERIES DAT
            self.research_counter += 1

        if self.save_timeseries_data_state and (self.t_firm % self.compression_factor_state == 0):
            self.save_timeseries_data_firm()
            self.research_bool = 0

        return self.cars_on_sale

