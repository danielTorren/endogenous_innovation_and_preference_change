import numpy as np

class SecondHandMerchant:
    def __init__(self, unique_id, parameters_second_hand):
        """
        Initialize the SecondHandMerchant with its unique ID and configuration.

        Args:
            unique_id (int): Unique identifier for the merchant.
            parameters_second_hand (dict): Dictionary containing configuration parameters such as age limits,
                                        discount rates, scrap prices, and random state.
        """
        self.id = unique_id
        self.cars_on_sale = []

        self.t_second_hand_cars = 0

        self.age_limit_second_hand = parameters_second_hand["age_limit_second_hand"]
        self.set_up_time_series_second_hand_car()

        self.r = parameters_second_hand["r"]
        self.max_num_cars = parameters_second_hand["max_num_cars"]
        self.burn_in_second_hand_market = parameters_second_hand["burn_in_second_hand_market"]

        self.random_state = parameters_second_hand["random_state"]

        #self.delta = parameters_second_hand["delta"]

        self.scrap_price = parameters_second_hand["scrap_price"]

        self.beta_segment_vec = parameters_second_hand["beta_segment_vals"] 
        self.gamma_segment_vec = parameters_second_hand["gamma_segment_vals"] 

        self.spent = 0
        self.income = 0
        self.assets = 0
        self.profit = 0
        self.scrap_loss = 0
        self.age_second_hand_car_removed = []
        
    def calc_median(self, beta_vec, gamma_vec):
        """
        Calculate and store the median values of beta and gamma across users.

        Args:
            beta_vec (np.ndarray): Array of beta values for users.
            gamma_vec (np.ndarray): Array of gamma values for users.
        """
        self.median_beta =  np.median(beta_vec)
        self.median_gamma = np.median(gamma_vec)

    def gen_vehicle_dict_vecs_second_hand(self, list_vehicles):
        """
        Generate attribute arrays from a list of second-hand vehicle objects.

        Args:
            list_vehicles (list): List of vehicle objects.

        Returns:
            dict: Dictionary mapping attribute names to NumPy arrays.
        """
            
        # Initialize dictionary to hold lists of vehicle properties
        vehicle_dict_vecs = {
            "Quality_a_t": [], 
            "Eff_omega_a_t": [], 
            "price": [], 
            "L_a_t": [],
            "delta_P": [],
            "B": []
        }

        # Iterate over each vehicle to populate the arrays
        for vehicle in list_vehicles:
            vehicle_dict_vecs["Quality_a_t"].append(vehicle.Quality_a_t)
            vehicle_dict_vecs["Eff_omega_a_t"].append(vehicle.Eff_omega_a_t)
            #vehicle_dict_vecs["price"].append(vehicle.price)
            vehicle_dict_vecs["L_a_t"].append(vehicle.L_a_t)
            vehicle_dict_vecs["delta_P"].append(vehicle.delta_P)
            vehicle_dict_vecs["B"].append(vehicle.B)

        # convert lists to numpy arrays for vectorised operations
        for key in vehicle_dict_vecs:
            vehicle_dict_vecs[key] = np.array(vehicle_dict_vecs[key])

        return vehicle_dict_vecs
    
    def gen_vehicle_dict_vecs_new_cars(self, list_vehicles):
        """
        Generate attribute arrays from a list of new vehicle objects.

        Args:
            list_vehicles (list): List of vehicle objects.

        Returns:
            dict: Dictionary mapping attribute names to NumPy arrays.
        """
            
        # Initialize dictionary to hold lists of vehicle properties
        vehicle_dict_vecs = {
            "Quality_a_t": [], 
            "Eff_omega_a_t": [], 
            "price": [],
            "B": []
        }

        # Iterate over each vehicle to populate the arrays
        for vehicle in list_vehicles:
            vehicle_dict_vecs["Quality_a_t"].append(vehicle.Quality_a_t)
            vehicle_dict_vecs["Eff_omega_a_t"].append(vehicle.Eff_omega_a_t)
            vehicle_dict_vecs["price"].append(vehicle.price)       
            vehicle_dict_vecs["B"].append(vehicle.B)       

        # convert lists to numpy arrays for vectorised operations
        for key in vehicle_dict_vecs:
            vehicle_dict_vecs[key] = np.array(vehicle_dict_vecs[key])

        return vehicle_dict_vecs
    
    def calc_car_price_heuristic(self, vehicle_dict_vecs_new_cars, vehicle_dict_vecs_second_hand_cars):
        """
        Estimate second-hand car prices using a heuristic based on similarity to new cars.

        Args:
            vehicle_dict_vecs_new_cars (dict): Attributes of new cars.
            vehicle_dict_vecs_second_hand_cars (dict): Attributes of second-hand cars.

        Returns:
            np.ndarray: Estimated prices for second-hand cars.
        """
        # Extract Quality, Efficiency, and Prices of first-hand cars
        first_hand_quality = vehicle_dict_vecs_new_cars["Quality_a_t"]
        first_hand_efficiency =  vehicle_dict_vecs_new_cars["Eff_omega_a_t"]
        first_hand_prices = vehicle_dict_vecs_new_cars["price"]
        first_hand_B = vehicle_dict_vecs_new_cars["B"]

        # Extract Quality, Efficiency, and Age of second-hand cars
        second_hand_quality = vehicle_dict_vecs_second_hand_cars["Quality_a_t"]
        second_hand_efficiency = vehicle_dict_vecs_second_hand_cars["Eff_omega_a_t"]
        second_hand_ages = vehicle_dict_vecs_second_hand_cars["L_a_t"]
        second_hand_delta_P = vehicle_dict_vecs_second_hand_cars["delta_P"]
        second_hand_B = vehicle_dict_vecs_second_hand_cars["B"]

        first_hand_quality_max = np.max(first_hand_quality)
        first_hand_efficiency_max = np.max(first_hand_efficiency)
        first_hand_B_max = np.max(first_hand_B)

        normalized_first_hand_quality = first_hand_quality / first_hand_quality_max 
        normalized_first_hand_efficiency = first_hand_efficiency / first_hand_efficiency_max 
        normalized_first_hand_B = first_hand_B/first_hand_B_max

        normalized_second_hand_quality = second_hand_quality  / first_hand_quality_max 
        normalized_second_hand_efficiency = second_hand_efficiency / first_hand_efficiency_max
        normalized_second_hand_B = second_hand_B / first_hand_B_max

        # Compute proximity (Euclidean distance) for all second-hand cars to all first-hand cars
        diff_quality = normalized_second_hand_quality[:, np.newaxis] - normalized_first_hand_quality
        diff_efficiency = normalized_second_hand_efficiency[:, np.newaxis] - normalized_first_hand_efficiency
        diff_B = normalized_second_hand_B[:, np.newaxis] - normalized_first_hand_B

        distances = np.sqrt(diff_quality ** 2 + diff_efficiency ** 2 + diff_B ** 2)

        # Find the closest first-hand car for each second-hand car
        closest_idxs = np.argmin(distances, axis=1)

        # Get the prices of the closest first-hand cars
        #closest_prices = first_hand_prices[closest_idxs]
        closest_prices = np.maximum(first_hand_prices[closest_idxs] - (self.rebate_calibration + self.rebate),0)

        # Adjust prices based on car age and depreciation
        adjusted_prices = closest_prices * (1 - second_hand_delta_P) ** second_hand_ages

        return adjusted_prices

    def update_stock_contents(self):
        """
        Update the stock of second-hand cars:
            - Remove overaged or underpriced cars.
            - Update prices using a heuristic method.
            - Enforce max inventory constraint.
        """
            
        #check len of list    
        for vehicle in self.cars_on_sale:       
            if vehicle.second_hand_counter > self.age_limit_second_hand:
                self.age_second_hand_car_removed.append(vehicle.L_a_t)
                self.assets -= vehicle.cost_second_hand_merchant
                self.scrap_loss += vehicle.cost_second_hand_merchant
                self.cars_on_sale.remove(vehicle)

        data_dicts_second_hand = self.gen_vehicle_dict_vecs_second_hand(self.cars_on_sale)
        # Calculate the price vector
        data_dicts_new_cars = self.gen_vehicle_dict_vecs_new_cars(self.vehicles_on_sale)

        price_vec = self.calc_car_price_heuristic(data_dicts_new_cars, data_dicts_second_hand)

        # Update the prices of the remaining cars
        for i, vehicle in enumerate(self.cars_on_sale):
            vehicle.price = price_vec[i]

        # Vectorized approach to identify cars below the scrap price
        below_scrap_mask = price_vec < self.scrap_price

        # Remove cars below the scrap price
        self.cars_on_sale = [
            vehicle for i, vehicle in enumerate(self.cars_on_sale) if not below_scrap_mask[i]
        ]

        #REMOVE EXCESS CARS
        if len(self.cars_on_sale) > self.max_num_cars:
            # Calculate how many cars to remove
            num_cars_to_remove = len(self.cars_on_sale) - self.max_num_cars
            # Randomly select cars to remove
            cars_to_remove = self.random_state.choice(
                self.cars_on_sale, num_cars_to_remove, replace=False
            )
            # Add ages of removed cars
            self.age_second_hand_car_removed.extend(vehicle.L_a_t for vehicle in cars_to_remove)
            # Use list comprehension to filter out the cars to remove
            self.cars_on_sale = [car for car in self.cars_on_sale if car not in cars_to_remove] 
        """
        #REMOVE EXCESS CARS
        if len(self.cars_on_sale) > self.max_num_cars:
            # Calculate how many cars to remove
            num_cars_to_remove = len(self.cars_on_sale) - self.max_num_cars

            # Sort cars by age in descending order (oldest first)
            self.cars_on_sale.sort(key=lambda car: car.L_a_t, reverse=True)

            # Select the oldest cars to remove
            cars_to_remove = self.cars_on_sale[:num_cars_to_remove]

            # Add ages of removed cars
            self.age_second_hand_car_removed.extend(car.L_a_t for car in cars_to_remove)

            # Keep only the remaining cars (i.e., exclude the removed ones)
            self.cars_on_sale = self.cars_on_sale[num_cars_to_remove:]
        """
        
    def add_to_stock(self,vehicle):
        """
        Add a new second-hand vehicle to the merchant's stock.

        Args:
            vehicle (object): Vehicle object to add.
        """
            
        #add new car to stock
        vehicle.price = vehicle.price_second_hand_merchant
        vehicle.scenario = "second_hand"
        vehicle.second_hand_counter = 0
        self.cars_on_sale.append(vehicle)
    
    def remove_car(self, vehicle):
        """
        Remove a vehicle from the merchant's stock.

        Args:
            vehicle (object): Vehicle object to remove.
        """
            
        self.cars_on_sale.remove(vehicle)

    def set_up_time_series_second_hand_car(self):
        """
        Initialize time series trackers for second-hand inventory metrics.
        """
            
        self.history_num_second_hand = []
        self.history_profit = []
        self.history_age_second_hand_car_removed = []

    def save_timeseries_second_hand_merchant(self):
        """
        Save current values of second-hand inventory, profit, and removal history to time series.
        """
            
        self.history_num_second_hand.append(len(self.cars_on_sale))
        self.history_profit.append(self.profit )
        self.history_age_second_hand_car_removed.append(self.age_second_hand_car_removed)

    def update_age_stock_prices_and_emissions_intensity(self, list_cars):
        """
        Increment the age and update fuel costs and emissions intensities for a list of cars.

        Args:
            list_cars (list): List of vehicle objects currently in stock.
        """
        for car in list_cars:
            if self.t_second_hand_cars > self.burn_in_second_hand_market:
                car.L_a_t += 1
                car.second_hand_counter += 1#UPDATE THE STEPS ITS BEEN HERE
            if car.transportType == 2:#ICE
                car.fuel_cost_c = self.gas_price
            else:#EV
                car.fuel_cost_c = self.electricity_price
                car.e_t = self.electricity_emissions_intensity

    def next_step(self,gas_price, electricity_price, electricity_emissions_intensity, vehicles_on_sale, rebate_calibration,rebate):
        """
        Advance the second-hand merchant's state by one timestep:
            - Update fuel prices and emission intensities.
            - Age vehicles and adjust stock post-burn-in.
            - Update profit and return current stock.

        Args:
            gas_price (float): Current gasoline price.
            electricity_price (float): Current electricity price.
            electricity_emissions_intensity (float): Grid carbon intensity.
            vehicles_on_sale (list): List of new cars for price reference.
            rebate_calibration (float): Policy calibration value for EV rebates.
            rebate (float): Rebate for used electric vehicles.

        Returns:
            list: Updated list of second-hand vehicles in stock.
        """
        self.t_second_hand_cars += 1

        self.gas_price =  gas_price
        self.electricity_price = electricity_price
        self.electricity_emissions_intensity = electricity_emissions_intensity
        self.vehicles_on_sale = vehicles_on_sale
        self.rebate_calibration = rebate_calibration
        self.rebate = rebate
        self.update_age_stock_prices_and_emissions_intensity(self.cars_on_sale)

        self.age_second_hand_car_removed = []

        if self.cars_on_sale and self.t_second_hand_cars > self.burn_in_second_hand_market:
            self.update_stock_contents()
        

        self.profit = self.income - self.spent

        return self.cars_on_sale