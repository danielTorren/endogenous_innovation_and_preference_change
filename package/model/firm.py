import numpy as np
from scipy.special import lambertw
from package.model.carModel import CarModel

class Firm:
    """
    Firm class represents a car manufacturer in the simulation.
    Each firm manages its own innovation, production, pricing, and sales strategy
    for both ICE and EV technologies.
    """

    def __init__(self, firm_id, init_tech_ICE, init_tech_EV, parameters_firm, parameters_car_ICE, parameters_car_EV):
        """
        Initialize a Firm object with its technologies, parameters, and firm-specific attributes.

        Args:
            firm_id (int): Unique identifier for the firm.
            init_tech_ICE (CarModel): Initial ICE car model.
            init_tech_EV (CarModel): Initial EV car model.
            parameters_firm (dict): Configuration parameters for the firm.
            parameters_car_ICE (dict): Parameters for ICE car models.
            parameters_car_EV (dict): Parameters for EV car models.
        """

        self.rebate = parameters_firm["rebate"]#7000#JUST TO TRY TO GET TRANSITION
        self.rebate_calibration = parameters_firm["rebate_calibration"]
        
        self.random_state = parameters_firm["random_state"]

        self.t_firm = 0
        self.production_change_bool = 0
        self.policy_distortion = 0
        self.zero_profit_options_prod = 0
        self.zero_profit_options_research = 0

        self.segment_codes = parameters_firm["segment_codes"]
        self.num_segments = len(self.segment_codes)

        self.save_timeseries_data_state = parameters_firm["save_timeseries_data_state"]
        self.compression_factor_state = parameters_firm["compression_factor_state"]
        self.id_generator = parameters_firm["IDGenerator_firms"]  

        self.beta_s_values = parameters_firm["beta_segment_vals"] 
        self.gamma_s_values = parameters_firm["gamma_segment_vals"] 

        self.d_mean = parameters_firm["d_mean"]
        self.nu = parameters_firm["nu"]  
        self.zeta = parameters_firm["zeta"]
        self.alpha = parameters_firm["alpha"]

        self.max_cars_prod = parameters_firm["max_cars_prod"] 

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

        self.min_profit = self.parameters_firm["min_profit"]

        self.U_segments_init = self.parameters_firm["U_segments_init"]
        
        self.init_price_multiplier = self.parameters_firm["init_price_multiplier"]

        self.lambda_exp = parameters_firm["lambda"]

        self.universal_model_repo_ICE = parameters_firm["universal_model_repo_ICE"]#THIS NEEDS TO BE SHARED AMONGST ALL FIRMS
        self.universal_model_repo_EV = parameters_firm["universal_model_repo_EV"]#THIS NEEDS TO BE SHARED AMONGST ALL FIRMS

        self.ICE_landscape = self.parameters_firm["ICE_landscape"]
        self.EV_landscape = self.parameters_firm["EV_landscape"]

        self.parameters_car_ICE = parameters_car_ICE
        self.parameters_car_EV = parameters_car_EV

        self.carbon_price =  self.parameters_firm["carbon_price"]
        self.gas_price =  self.parameters_firm["gas_price"]
        self.electricity_price =  self.parameters_firm["electricity_price"]
        self.electricity_emissions_intensity =  self.parameters_firm["electricity_emissions_intensity"]
        self.rebate_calibration =  self.parameters_firm["rebate_calibration"]
        self.rebate =  0
        self.production_subsidy =  0

        self.expected_profits_segments = {}      
        
        if self.ev_production_bool:
            self.cars_on_sale = [self.init_tech_ICE, self.init_tech_EV]
        else:
            self.cars_on_sale = [self.init_tech_ICE] 

        self.set_car_init_price()

        if self.save_timeseries_data_state:
            self.set_up_time_series_firm()

        
    def set_car_init_price(self):
        """
        Set initial price for all initial cars on sale.
        """
        for car in self.cars_on_sale:
            car.price = car.ProdCost_t*self.init_price_multiplier
            for segment_code in self.segment_codes:
                # Add data for the segment
                car.optimal_price_segments[segment_code] = car.price

        #need to do EV IN MEMORY FOR THE FIRST STEP as well
        for car in self.list_technology_memory_EV:
            car.price = car.ProdCost_t*self.init_price_multiplier
            for segment_code in self.segment_codes:
                # Add data for the segment
                car.optimal_price_segments[segment_code] = car.price
                #car.B_segments[segment_code] = self.B_segments_init

    def calc_init_U_segments(self):
        """
        Initialize utility values across segments for each car on sale at time step zero.
        """
        for segment_code in self.segment_codes:          
            # Unpack the tuple
            b_idx, g_idx, e_idx = segment_code  # if your codes are (b, g, e)
            for car in self.cars_on_sale:
                if (car.transportType == 2) or (e_idx == 1 and car.transportType == 3):
                    car.car_utility_segments_U[segment_code] = self.U_segments_init#SET AS FIXED CONSTANT TO START FIRST TURN
                else:
                    car.car_utility_segments_U[segment_code] = -np.inf

    def calc_utility_prop(self,U,W, maxU):
        """
        Compute utility proportion using exponential scaling based on segment utility.

        Args:
            U (float): Utility value.
            W (float): Segment-wide scaling factor.
            maxU (float): Maximum utility in the segment.

        Returns:
            float: Normalized utility proportion for car choice modeling.
        """
        exp_input = self.kappa*U - self.kappa*maxU
        norm_exp_input = -self.kappa*maxU
        
        utility_proportion = np.exp(exp_input)/(np.exp(norm_exp_input)*W + np.exp(exp_input))

        return utility_proportion

    def create_car_data(self, car_list):
        """
        Converts a list of car objects into a dictionary of NumPy arrays for vectorized calculations.

        Args:
            car_list (list): List of CarModel objects.

        Returns:
            dict: Dictionary mapping attribute names to numpy arrays.
        """

        if not car_list:
            return None

        # Check if car objects have the necessary attributes (you might want to add more checks)
        required_attributes = ["emissions", "ProdCost_t", "transportType", "delta", "Quality_a_t", "Eff_omega_a_t", "e_t", "fuel_cost_c", "B"]
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
        """
        Vectorized calculation of profit-maximizing prices for each car and segment.

        Args:
            car_list (list): List of cars to price.
            car_data (dict): Dictionary of car attributes.

        Returns:
            list: Updated car list with segment-specific optimal prices.
        """

        # Convert car data to NumPy arrays (CRITICAL CHANGE)
        E_m = car_data["emissions"]  # Array of emissions for all cars
        C_m = car_data["ProdCost_t"]
        transport_types = car_data["transportType"]
        delta = car_data["delta"]
        Quality_a_t = car_data["Quality_a_t"]
        Eff_omega_a_t = car_data["Eff_omega_a_t"]
        e_t = car_data["e_t"]
        fuel_cost_c = car_data["fuel_cost_c"]
        B = car_data["B"]

        # Apply EV-specific calculations using boolean indexing
        ev_mask = transport_types == 3  # Boolean mask for EV cars
        C_m_cost = C_m.copy()  # Important: Create a copy to avoid modifying original
        C_m_price = C_m.copy()
        C_m_cost[ev_mask] = np.maximum(0, C_m[ev_mask] - self.production_subsidy)
        C_m_price[ev_mask] = np.maximum(0, C_m[ev_mask] - (self.production_subsidy + self.rebate + self.rebate_calibration))

        term1 = - C_m_price[:, np.newaxis] - self.gamma_s_values[np.newaxis, :]*E_m[:, np.newaxis] # Matrix with shape: num cars x num segments
        term2 = self.beta_s_values[np.newaxis, :]*(Quality_a_t[:, np.newaxis]**self.alpha)# Matrix with shape: num cars x num segments
        term3 =self.nu*(B[:, np.newaxis]*Eff_omega_a_t[:, np.newaxis])**self.zeta# Matrix with shape: num cars x num segments
        term4 = - self.d_mean * (((1 + self.r) * (1 - delta[:, np.newaxis]) * (fuel_cost_c[:, np.newaxis] + self.gamma_s_values[np.newaxis, :] * e_t[:, np.newaxis])) / (Eff_omega_a_t[:, np.newaxis] * (self.r - delta[:, np.newaxis] - self.r * delta[:, np.newaxis])))
        
        U = term1 + term2 + term3 + term4# Matrix with shape: num cars x num segments

        exp_input = (self.kappa*U - 1) - np.log(self.W_vec[np.newaxis, :])

        Arg = np.exp(exp_input)
        LW = lambertw(Arg, 0).real

        P = C_m_cost[:, np.newaxis] + (1.0 + LW)/self.kappa
        
        # Store results in the original car objects (CRITICAL CHANGE)
        for i, car in enumerate(car_list):
                for j, segment_code in enumerate(self.segment_codes):  # Use enumerate directly on the dictionary
                    car.optimal_price_segments[segment_code] = P[i, j]#CALC PRICE REGARDLESS OF WHETHER OR NOT THE CAR CAN BE USED BY THAT SEGMENT

        return car_list  # Return a dictionary of optimal prices by segment.

    def calc_utility(self, Q, beta, gamma, c, omega, e, E_new, P_adjust, delta, B):
        """
        Compute utility value for a single car across segments.

        Returns:
            float: Utility value for given inputs.
        """
        U = - P_adjust - gamma*E_new + beta*Q**self.alpha + self.nu*(B*omega)**self.zeta - self.d_mean*(((1+self.r)*(1-delta)*(c + gamma*e))/(omega*(self.r - delta - self.r*delta)))
        
        return U

    def calc_utility_cars_segments(self, car_list, car_data):
        """
        Compute utility for all car-segment combinations.

        Args:
            car_list (list): Cars to evaluate.
            car_data (dict): Dictionary of car attributes.

        Returns:
            list: Updated car list with utility values by segment.
        """
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
        B_values = np.tile(car_data["B"][:, np.newaxis], (1, self.num_segments))

        # Broadcast segment parameters to match (num_cars, num_segments)
        beta_s_broadcast = np.tile(self.beta_s_values, (num_cars, 1))
        gamma_s_broadcast = np.tile(self.gamma_s_values, (num_cars, 1))

        # Initialize P_adjust and valid_mask
        P_adjust_values = np.zeros((num_cars, self.num_segments))
        valid_mask = np.zeros((num_cars, self.num_segments), dtype=bool)

        # Populate valid_mask and P_adjust_values
        for i, car in enumerate(car_list):
            for j, segment_code in enumerate(self.segment_codes):
                if (car.transportType == 2) or (segment_code[2] == 1):
                    valid_mask[i, j] = True
                    P_adjust_values[i, j] = car.optimal_price_segments.get(segment_code, 0)

        # Apply rebates for EVs
        ev_mask = (transport_types == 3)#JUST APPLY REBATE TO EV CARS
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
            delta_values[valid_mask],
            B_values[valid_mask]
        )

        # Assign calculated utilities back to car objects
        for i, car in enumerate(car_list):
            for j, segment_code in enumerate(self.segment_codes):
                car.car_utility_segments_U[segment_code] = U[i, j]

        return car_list

    ########################################################################################################################################
    #INNOVATION

    def innovate(self):
        """
        Perform innovation step for the firm.
        Evaluate neighboring technologies and select a new car to add to memory.
        """
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
        else:
            self.last_researched_car_ICE = self.vehicle_model_research

        #add vehicle to memory, MAKE THE MEMORY INTO A CAR HERE?
        self.add_new_vehicle_memory(self.vehicle_model_research)

        # adjust memory bank
        self.update_memory_len()

    def calc_predicted_profit_segments_research(self, car_list, car_data):
        """
        Calculate expected profits for cars being considered for research.
        Incorporates production subsidies and considers segment compatibility.

        Args:
            car_list (list): Cars to evaluate.
            car_data (dict): Dictionary of car attributes.

        Returns:
            list: Updated cars with expected profit by segment.
        """
        num_cars = len(car_list)

        # Extract necessary car data
        transport_types = car_data["transportType"]
        prod_costs = car_data["ProdCost_t"]
        optimal_prices = np.array([
            [car.optimal_price_segments.get(segment_code, 0) for segment_code in self.segment_codes] 
            for car in car_list
        ])
        utilities = np.array([
            [car.car_utility_segments_U.get(segment_code, -np.inf) for segment_code in self.segment_codes] 
            for car in car_list
        ])#THES ARE ADJUST FOR ICE CARS AND THE ABOLUTY TO ChOOSE EVS, the -np.inf is incase the utility doesnt exist

        # Extract market data
        consider_ev_mask = np.array([segment_code[2] == 1 for segment_code in self.segment_codes])  # Shape (num_segments,)
        is_ev_mask = transport_types == 3  # Shape (num_cars,)#CAR IS AN EV not ICEV
        consider_ev_broadcast = np.tile(consider_ev_mask, (num_cars, 1))
        include_vehicle_mask = (~is_ev_mask[:, np.newaxis]) | consider_ev_broadcast#IS ICE OR CONSIDERS EV 

        # Profit per sale
        prod_subsidy_adjustment = np.maximum(0, prod_costs[:, np.newaxis] - self.production_subsidy)
        profit_per_sale = np.where(
            is_ev_mask[:, np.newaxis],
            optimal_prices - prod_subsidy_adjustment,#if car is EV then the poduction subsidy increases the profit
            optimal_prices - prod_costs[:, np.newaxis]#if ICE CAR then profit it prices minus production cost
        )

        # Utility proportion
        utility_proportion = np.where(
            utilities == -np.inf,
            0,
            self.calc_utility_prop(utilities, self.W_vec[np.newaxis, :], self.maxU_vec[np.newaxis, :])
        )

        # Raw profit
        raw_profit = profit_per_sale * self.I_s_t_vec[np.newaxis, :] * utility_proportion


        # Apply research subsidy for segments that can't buy EVs, you get the resarch subsidy if you choose an EV regardless of whether or not the segemnt can buy it
        expected_profit = np.where(
            include_vehicle_mask,
            raw_profit,
            0 
        )
        
        # Assign expected profits back to cars
        for i, car in enumerate(car_list):
            for j, segment_code in enumerate(self.segment_codes):
                car.expected_profit_segments[segment_code] = expected_profit[i, j]

        return car_list

    def select_car_lambda_research(self, car_list):
        """
        Select a car for R&D using softmax over expected profits.

        Args:
            car_list (list): Cars to select from.

        Returns:
            CarModel: Selected car for innovation.
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

        if np.sum(profits) == 0:
            self.zero_profit_options_research = 1
            selected_index = self.random_state.choice(len_vehicles)
        else:
            profits[profits == 0] = -np.inf#if pofit is zero you cant choose it

            # Compute the softmax probabilities
            lambda_profits = np.zeros_like(profits)
            valid_profits_mask = profits != -np.inf

            exp_input = self.lambda_exp*(profits[valid_profits_mask]  - np.max(profits[valid_profits_mask]))
            lambda_profits[valid_profits_mask] = np.exp(exp_input)
            sum_profit = np.sum(lambda_profits)

            self.zero_profit_options_research = 0
            probabilities = lambda_profits/sum_profit
            selected_index = self.random_state.choice(len_vehicles, p=probabilities)

        selected_vehicle = car_list[selected_index]

        return selected_vehicle

    def gen_neighbour_carsModel(self, tech_strings, nk_landscape, parameters_car, transportType):
        """
        Generate CarModel instances for neighboring technology strings.

        Returns:
            list: List of CarModel instances.
        """
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
        """
        Generate strings representing neighboring technology designs.

        Returns:
            list: List of unique neighboring technology strings.
        """
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
        """
        Combine generation of neighbor strings and CarModel creation.

        Returns:
            list: List of neighboring CarModels.
        """
        # Set to track unique neighboring technology strings

        string_list = self.gen_neighbour_strings(list_technology_memory, last_researched_car)

        self.neighbouring_technologies = self.gen_neighbour_carsModel(string_list, landscape, parameters_car, transportType)

        return self.neighbouring_technologies
    
    def add_new_vehicle_memory(self, vehicle_model_research):
        """
        Add a new vehicle model to the firm's memory.
        """
        if vehicle_model_research.transportType == 2:#ICE
            if vehicle_model_research not in self.list_technology_memory_ICE:
                self.list_technology_memory_ICE.append(vehicle_model_research)
        else:#EV
            if vehicle_model_research not in self.list_technology_memory_EV:
                self.list_technology_memory_EV.append(vehicle_model_research)

    def update_memory_timer(self):
        """
        Update memory timer for technologies not currently in use.
        """
        #change the timer for the techs that are not the ones being used
        if self.ev_research_bool:
            for technology in self.list_technology_memory_EV:
                if not technology.choosen_tech_bool:
                    technology.update_timer()

        for technology in self.list_technology_memory_ICE:
            if not technology.choosen_tech_bool:
                technology.update_timer()

    def update_memory_len(self):
        """
        Trim the memory bank if it exceeds the allowed capacity.
        Removes the oldest unused car.
        """

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
        Calculate expected production profits for cars across segments.

        Returns:
            list: Updated car list with profit expectations by segment.
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
            self.calc_utility_prop(utilities, self.W_vec[np.newaxis, :], self.maxU_vec[np.newaxis, :])
        )

        # Raw profit
        raw_profit = profit_per_sale * self.I_s_t_vec[np.newaxis, :] * utility_proportion

        # Expected profit, if ice car then apply the discriminatory tax
        expected_profit = np.where(
            is_ev_mask[:, np.newaxis],
            raw_profit,
            raw_profit
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
        """
        Compute utility and profit for a selected vehicle across valid segments.

        Returns:
            np.ndarray: Array of expected profits per segment.
        """
        beta_s_values = self.beta_s_values[valid_indices]
        gamma_s_values = self.gamma_s_values[valid_indices]
        I_s_t_values = self.I_s_t_vec[valid_indices]
        W_values = self.W_vec[valid_indices]
        maxU_values = self.maxU_vec[valid_indices]

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
            selected_vehicle.delta,
            selected_vehicle.B
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
            self.calc_utility_prop(utilities, W_values, maxU_values)
        )

        raw_profits = profit_per_sale * I_s_t_values * utility_proportions

        updated_profits = np.where(
            selected_vehicle.transportType == 3,
            raw_profits,
            raw_profits
        )

        return updated_profits
    
    def select_car_lambda_production(self, car_list):
        """
        Select cars for production using a greedy approach over expected profits.

        Returns:
            list: Selected cars for production.
        """
        # Step 0: Build the profit matrix (segments x technologies)

        technologies = car_list
        profit_matrix_raw = np.zeros((self.num_segments, len(technologies)))

        vehicles_selected_profits = []  # Used to track the vehicle and the expected profit
        vehicle_segments = []  # To keep track of vehicle segment which it was targeting

        # Populate the profit matrix with expected profits
        for i, segment_code in enumerate(self.segment_codes):
            for j, car in enumerate(technologies):

                profit_matrix_raw[i, j] = car.expected_profit_segments[segment_code]

        # Step 2: Identify rows with at least one nonzero value
        valid_rows_mask = np.any(profit_matrix_raw > 0, axis=1)  # Boolean mask for rows with nonzero values
        
        # Step 3: Filter the profit matrix and corresponding segment codes
        profit_matrix = profit_matrix_raw[valid_rows_mask, :]
        valid_segment_codes_conditions = [self.segment_codes[i] for i in range(self.num_segments) if valid_rows_mask[i]]

        if not np.any(profit_matrix): #PROFIT IS ALL 0, pick up to 
            self.zero_profit_options_prod = 1
            vehicles_selected = technologies

            #NEED TO SET PRICES HERE, WHAT TO DO WHEN ALL THE CARS ARE TERRIBLE, SET THE LOWEST PRICE?
            for i, vehicle in enumerate(vehicles_selected):
                price_options = vehicle.optimal_price_segments.values()
                lowest_price = min(price_options)
                vehicle.price = lowest_price
        else:
            self.zero_profit_options_prod = 0

            # Iterate until all segments are covered or the max number of cars are chosen
            for _ in range(len(valid_segment_codes_conditions)):
                # Step 1: Choose the highest value square in the matrix
                max_index = np.unravel_index(np.argmax(profit_matrix, axis=None), profit_matrix.shape)
                max_row, max_col = max_index
                max_profit = profit_matrix[max_row, max_col]

                # Select the car and mark the segment as covered
                selected_vehicle = technologies[max_col]
                vehicles_selected_profits.append((selected_vehicle, max_profit, valid_segment_codes_conditions[max_row]))
                vehicle_segments.append((selected_vehicle, valid_segment_codes_conditions[max_row]))

                # Step 2: Remove the segment (set the row to negative infinity)
                profit_matrix[max_row, :] = -np.inf

                # Step 3: Update the column for the selected technology using the optimal price for the chosen segment
                optimal_price_chosen_segment = selected_vehicle.optimal_price_segments[valid_segment_codes_conditions[max_row]]
                selected_vehicle.price = optimal_price_chosen_segment  # SET PRICE

                valid_indices = np.where(profit_matrix[:, max_col] != -np.inf)[0]
                valid_segment_codes = [valid_segment_codes_conditions[i] for i in valid_indices]

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

        vehicle_to_max_profit = {}
        for vehicle, profit, segment in vehicles_selected_profits:
            if vehicle not in vehicle_to_max_profit or profit > vehicle_to_max_profit[vehicle]["profit"]:
                vehicle_to_max_profit[vehicle] = {"profit": profit, "segment": segment}
        
        sorted_vehicles = sorted(vehicle_to_max_profit.items(), key=lambda x: x[1]["profit"], reverse=True)

        vehicles_selected = [x[0] for x in sorted_vehicles[:self.max_cars_prod]]

        return vehicles_selected


    def choose_cars_segments(self):
        """
        Main method to choose which cars will be sold based on profitability and utility.

        Returns:
            list: Selected cars to be offered on the market.
        """
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
    
    def set_up_time_series_firm(self):
        """
        Initialize storage for time-series data tracking firm performance.
        """
        self.history_profit = []
        self.history_firm_cars_users = []
        self.history_attributes_researched = []
        self.history_research_type = []
        self.history_num_cars_on_sale = []
        self.history_segment_production_counts = []

    def save_timeseries_data_firm(self):
        """
        Record current firm performance metrics into time-series storage.
        """
        self.history_profit.append(self.firm_profit)
        self.history_firm_cars_users.append(self.firm_cars_users)
        self.history_num_cars_on_sale.append(len(self.cars_on_sale))

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
        """
        Update each car's fuel cost and emissions intensity based on current market inputs.

        Returns:
            list: Updated car list.
        """
        for car in car_list:
            if car.transportType == 2:#ICE
                car.fuel_cost_c = self.gas_price
            else:#EV
                car.fuel_cost_c = self.electricity_price
                car.e_t = self.electricity_emissions_intensity
        return car_list

    def next_step(self, I_s_t_vec, W_vec, nu_UMax_vec, carbon_price, gas_price, electricity_price, electricity_emissions_intensity, rebate, production_subsidy, rebate_calibration):
        """
        Advance the firm to the next time step. Updates cars, memory, and innovations.

        Returns:
            list: Cars on sale after production and innovation decisions.
        """
        
        self.t_firm += 1

        self.I_s_t_vec = I_s_t_vec
        self.W_vec =  W_vec
        self.maxU_vec = nu_UMax_vec
        self.carbon_price = carbon_price
        self.gas_price =  gas_price
        self.electricity_price = electricity_price
        self.electricity_emissions_intensity = electricity_emissions_intensity
        self.rebate = rebate
        self.rebate_calibration = rebate_calibration
        self.production_subsidy = production_subsidy

        self.cars_on_sale = self.update_prices_and_emissions_intensity(self.cars_on_sale)#update the prices of cars on sales with changes, this is required for calculations made by users

        #update cars to sell   
        if (self.random_state.rand() < self.prob_change_production):
            self.cars_on_sale = self.choose_cars_segments()
            self.production_change_bool = 1                 

        self.update_memory_timer()

        if self.random_state.rand() < self.prob_innovate:
            self.innovate()
            self.research_bool = 1#JUST USED FOR THE SAVE TIME SERIES DATA

        if self.save_timeseries_data_state and (self.t_firm % self.compression_factor_state == 0):
            self.save_timeseries_data_firm()
            self.research_bool = 0

        return self.cars_on_sale
    
    def next_step_burn_in(self, I_s_t_vec, W_vec, nu_UMax_vec):
        """
        Execute firm actions during the burn-in phase.

        Returns:
            list: Cars on sale for this pre-initialization step.
        """
        self.I_s_t_vec = I_s_t_vec
        self.W_vec =  W_vec
        self.maxU_vec = nu_UMax_vec

        #update cars to sell   
        if (self.random_state.rand() < self.prob_change_production):
            self.cars_on_sale = self.choose_cars_segments()
          
        self.update_memory_timer()

        if self.random_state.rand() < self.prob_innovate:
            self.innovate()

        return self.cars_on_sale

