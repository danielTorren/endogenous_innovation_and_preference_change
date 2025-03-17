import numpy as np

class SecondHandMerchant:
    def __init__(self, unique_id, parameters_second_hand):
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
        self.median_beta =  np.median(beta_vec)
        self.median_gamma = np.median(gamma_vec)

    def create_second_hand_car_list(self, old_cars):
        self.cars_on_sale_pre_burn_in = []
        for car in old_cars:
            car.scenario = "second_hand"
            car.second_hand_counter = 0
            car.owner_id = self.id
            car.cost_second_hand_merchant = (car.ProdCost_t*1.2)*((1 - car.delta_P) ** car.L_a_t)/2#Need a value as a place holder, use 25% increase on production cost then assume a pay out of 50% correct price
            self.cars_on_sale_pre_burn_in.append(car)


    ###############################################################################################################

    def gen_vehicle_dict_vecs_second_hand(self, list_vehicles):
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
    

####################################################################################################################################

    def calc_car_price_heuristic(self, vehicle_dict_vecs_new_cars, vehicle_dict_vecs_second_hand_cars):

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
        normalized_first_hand_B = first_hand_B / first_hand_B_max

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
        closest_prices = first_hand_prices[closest_idxs]

        # Adjust prices based on car age and depreciation
        adjusted_prices = closest_prices * (1 - second_hand_delta_P) ** second_hand_ages

        return adjusted_prices

#############################################################################################################################

    def update_stock_contents(self):
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
        
        
    def add_to_stock(self,vehicle):
        #add new car to stock
        vehicle.price = vehicle.price_second_hand_merchant
        vehicle.scenario = "second_hand"
        vehicle.second_hand_counter = 0
        self.cars_on_sale.append(vehicle)
    
    def remove_car(self, vehicle):
        self.cars_on_sale.remove(vehicle)

############################################################################################

    def set_up_time_series_second_hand_car(self):
        self.history_num_second_hand = []
        self.history_profit = []
        self.history_age_second_hand_car_removed = []

    def save_timeseries_second_hand_merchant(self):
        self.history_num_second_hand.append(len(self.cars_on_sale))
        self.history_profit.append(self.profit )
        self.history_age_second_hand_car_removed.append(self.age_second_hand_car_removed)

#########################################################################################

    def update_age_stock_prices_and_emissions_intensity(self, list_cars):

        """
        Update ages of cars and the prices and emissiosn intensities
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

    def next_step(self,gas_price, electricity_price, electricity_emissions_intensity, vehicles_on_sale):
        
        self.t_second_hand_cars += 1

        self.gas_price =  gas_price
        self.electricity_price = electricity_price
        self.electricity_emissions_intensity = electricity_emissions_intensity
        self.vehicles_on_sale = vehicles_on_sale
        self.update_age_stock_prices_and_emissions_intensity(self.cars_on_sale)

        self.age_second_hand_car_removed = []
        
        if self.t_second_hand_cars == self.burn_in_second_hand_market:
            self.cars_on_sale = self.cars_on_sale_pre_burn_in

        if self.cars_on_sale and self.t_second_hand_cars > self.burn_in_second_hand_market:
            self.update_stock_contents()
            #ages = [car.L_a_t for car in self.cars_on_sale]
            #ages_EV = [car.L_a_t for car in self.cars_on_sale if car.transportType == 3]
            #ages_ICE = [car.L_a_t for car in self.cars_on_sale if car.transportType == 2]
            #print(self.t_second_hand_cars,":",np.mean(ages_ICE),  np.mean(ages_EV), len(self.cars_on_sale))

        self.profit = self.income - self.spent

        return self.cars_on_sale