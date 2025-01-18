import numpy as np
from scipy.special import lambertw
from sklearn.linear_model import LinearRegression

class SecondHandMerchant:


    def __init__(self, unique_id, parameters_second_hand):
        self.id = unique_id
        self.cars_on_sale = []

        self.age_limit_second_hand = parameters_second_hand["age_limit_second_hand"]
        self.set_up_time_series_social_network()
        self.car_bin = []

        self.alpha = parameters_second_hand["alpha"]
        self.r = parameters_second_hand["r"]
        self.max_num_cars = parameters_second_hand["max_num_cars"]
        self.burn_in_second_hand_market = parameters_second_hand["burn_in_second_hand_market"]
        self.random_state_second_hand = np.random.RandomState(parameters_second_hand["remove_seed"])
        self.d_max = parameters_second_hand["d_max"]
        self.delta = parameters_second_hand["delta"]
        self.kappa = parameters_second_hand["kappa"]
        self.nu = parameters_second_hand["nu"]
        self.scrap_price = parameters_second_hand["scrap_price"]

        self.beta_segment_vec = parameters_second_hand["beta_segment_vals"] 
        self.gamma_segment_vec = parameters_second_hand["gamma_segment_vals"] 


        self.spent = 0
        self.income = 0
        self.assets = 0
        self.profit = 0
        self.scrap_loss = 0

    def calc_median(self, beta_vec, gamma_vec):
        self.median_beta =  np.median(beta_vec)
        self.median_gamma = np.median(gamma_vec)

    ###############################################################################################################

    def gen_vehicle_dict_vecs(self, list_vehicles):
        # Initialize dictionary to hold lists of vehicle properties
        vehicle_dict_vecs = {
            "Quality_a_t": [], 
            "Eff_omega_a_t": [], 
            "price": [], 
            "fuel_cost_c": [], 
            "e_t": [],
            "L_a_t": [],
            "transportType": [],
            "cost_second_hand_merchant": []
        }

        # Iterate over each vehicle to populate the arrays
        for vehicle in list_vehicles:
            vehicle_dict_vecs["Quality_a_t"].append(vehicle.Quality_a_t)
            vehicle_dict_vecs["Eff_omega_a_t"].append(vehicle.Eff_omega_a_t)
            vehicle_dict_vecs["price"].append(vehicle.price)
            vehicle_dict_vecs["fuel_cost_c"].append(vehicle.fuel_cost_c)
            vehicle_dict_vecs["e_t"].append(vehicle.e_t)
            vehicle_dict_vecs["L_a_t"].append(vehicle.L_a_t)
            vehicle_dict_vecs["transportType"].append(vehicle.transportType)
            vehicle_dict_vecs["cost_second_hand_merchant"].append(vehicle.cost_second_hand_merchant)

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
            "L_a_t": [],
        }

        # Iterate over each vehicle to populate the arrays
        for vehicle in list_vehicles:
            vehicle_dict_vecs["Quality_a_t"].append(vehicle.Quality_a_t)
            vehicle_dict_vecs["Eff_omega_a_t"].append(vehicle.Eff_omega_a_t)
            vehicle_dict_vecs["price"].append(vehicle.price)
            vehicle_dict_vecs["L_a_t"].append(vehicle.L_a_t)

        # convert lists to numpy arrays for vectorised operations
        for key in vehicle_dict_vecs:
            vehicle_dict_vecs[key] = np.array(vehicle_dict_vecs[key])

        return vehicle_dict_vecs
    

####################################################################################################################################
    def calc_driving_utility_direct_old(self, Quality_a_t_vec,L_a_t_vec, X):

        utility_vec = Quality_a_t_vec * ((1 - self.delta) ** L_a_t_vec)/(self.alpha*X +1)

        return utility_vec
    
    def calc_car_price_vec_old(self, vehicle_dict_vecs, beta, gamma, W_s_t):

        #present UTILITY
        # Compute cost component based on transport type, with conditional operations
        X = (beta *vehicle_dict_vecs["fuel_cost_c"] + gamma * vehicle_dict_vecs["e_t"])/vehicle_dict_vecs["Eff_omega_a_t"]

        driving_utility_vec = self.calc_driving_utility_direct(vehicle_dict_vecs["Quality_a_t"],vehicle_dict_vecs["L_a_t"], X)

        B_vec = driving_utility_vec*((1+self.r)/(self.r + self.delta))

        Arg_vec = (np.exp(self.kappa*self.nu*(B_vec - beta*vehicle_dict_vecs["cost_second_hand_merchant"] )- 1.0)) / W_s_t
        LW_vec   = lambertw(Arg_vec, 0).real  # principal branch
        P_vec = vehicle_dict_vecs["cost_second_hand_merchant"] + (1.0 + LW_vec) / (self.kappa *self.nu* beta)

        #if P < vehicle.cost_second_hand_merchant:
        #    print(P,vehicle.cost_second_hand_merchant, Arg)
        #    raise ValueError("P LESS THAN vehicle.cost_second_hand_merchant")
        
        #inside_component = W_s_t ** 2 *(beta * vehicle_dict_vecs["cost_second_hand_merchant"]) ** 2 + W_s_t + U_vec
        
        # Adjust the component to avoid negative square roots
        #inside_component_adjusted = np.maximum(inside_component, 0)  # Replace negatives with 0

        #price_vec = np.where(
        #    inside_component < 0,
        #    vehicle_dict_vecs["cost_second_hand_merchant"],
        #    np.maximum(vehicle_dict_vecs["cost_second_hand_merchant"],(beta * vehicle_dict_vecs["cost_second_hand_merchant"] * W_s_t + np.sqrt( inside_component_adjusted))/(beta*W_s_t))
        #)

        return P_vec
    
    def calc_driving_utility_direct_single_old(self, Quality_a_t, L_a_t, X):

        driving_utility = Quality_a_t*(1-self.delta)**(L_a_t)/(self.alpha*X +1)

        return driving_utility
    
    def calc_car_price_single(self, vehicle, beta, gamma, W_s_t):

        #present UTILITY
        X = (beta *vehicle.fuel_cost_c + gamma * vehicle.e_t)/vehicle.Eff_omega_a_t

        driving_utility = self.calc_driving_utility_direct_single( vehicle.Quality_a_t, vehicle.L_a_t, X)
        B = driving_utility*((1+self.r)/(self.r + self.delta))#treat it as a current car, NO EMISSIOSN COST OR PURCHASSE COST? 
            
        Arg = (np.exp(self.kappa*self.nu*(B - beta*vehicle.cost_second_hand_merchant )- 1.0)) / W_s_t
        LW   = lambertw(Arg, 0)  # principal branch
        P = vehicle.cost_second_hand_merchant + (1.0 + LW) / (self.kappa *self.nu* beta)

        if P < vehicle.cost_second_hand_merchant:
            print(P,vehicle.cost_second_hand_merchant, Arg)
            raise ValueError("P LESS THAN vehicle.cost_second_hand_merchant")
        
        #inside_component = W_s_t ** 2 * ((gamma * vehicle.emissions) ** 2 + (beta * vehicle.cost_second_hand_merchant) ** 2 +  2 * vehicle.cost_second_hand_merchant * vehicle.emissions) + W_s_t + U
        
        #if inside_component < 0:
        #    price = vehicle.cost_second_hand_merchant
        #else:
        #    price = max(vehicle.cost_second_hand_merchant,(beta * vehicle.cost_second_hand_merchant * W_s_t + np.sqrt(inside_component))/(beta*W_s_t) )

        return P

    def calc_driving_utility_direct(self,Quality_a_t_vec,L_a_t_vec, X_matrix):

        # Compute commuting utility for individual-vehicle pairs
        B_matrix = Quality_a_t_vec*((1 - self.delta) ** L_a_t_vec)/(self.alpha*X_matrix + 1)

        return B_matrix

    def calc_car_price_vec(self, vehicle_dict_vecs):
        """ Calc the price at which utility of user would be 0 and set offer based on that"""
        #present UTILITY
        # Compute cost component based on transport type, with conditional operations
        # Adjust costs based on transport type

        X_matrix = ((self.beta_segment_vec[:, np.newaxis]*vehicle_dict_vecs["fuel_cost_c"]) + (self.gamma_segment_vec[:, np.newaxis]* vehicle_dict_vecs["e_t"]))/vehicle_dict_vecs["Eff_omega_a_t"]
        driving_utility_matrix = self.calc_driving_utility_direct(vehicle_dict_vecs["Quality_a_t"] ,vehicle_dict_vecs["L_a_t"] , X_matrix)
        B_s = (driving_utility_matrix*(1+self.r)/(self.r + self.delta))

        #quit()
        market_component = np.log(self.U_vec_on_sale)/(self.nu* self.beta_segment_vec)
        Price_s = B_s/self.beta_segment_vec[:, np.newaxis] - market_component[:, np.newaxis]
        Price_sale_vec = np.max(Price_s, axis = 0)

        return Price_sale_vec

    def calc_car_price_heuristic(self, vehicle_dict_vecs_new_cars, vehicle_dict_vecs_second_hand_cars):

        # Extract Quality, Efficiency, and Prices of first-hand cars
        first_hand_quality = vehicle_dict_vecs_new_cars["Quality_a_t"]
        first_hand_efficiency =  vehicle_dict_vecs_new_cars["Eff_omega_a_t"]
        first_hand_prices = vehicle_dict_vecs_new_cars["price"]

        # Extract Quality, Efficiency, and Age of second-hand cars
        second_hand_quality = vehicle_dict_vecs_second_hand_cars["Quality_a_t"]
        second_hand_efficiency = vehicle_dict_vecs_second_hand_cars["Eff_omega_a_t"]
        second_hand_ages = vehicle_dict_vecs_second_hand_cars["L_a_t"]

        # Normalize Quality and Efficiency for both first-hand and second-hand cars
        all_quality = np.concatenate([first_hand_quality, second_hand_quality])
        all_efficiency = np.concatenate([first_hand_efficiency, second_hand_efficiency])

        quality_min, quality_max = np.min(all_quality), np.max(all_quality)
        efficiency_min, efficiency_max = np.min(all_efficiency), np.max(all_efficiency)

        normalized_first_hand_quality = (first_hand_quality - quality_min) / (quality_max - quality_min)
        normalized_first_hand_efficiency = (first_hand_efficiency - efficiency_min) / (efficiency_max - efficiency_min)

        normalized_second_hand_quality = (second_hand_quality - quality_min) / (quality_max - quality_min)
        normalized_second_hand_efficiency = (second_hand_efficiency - efficiency_min) / (efficiency_max - efficiency_min)

        # Compute proximity (Euclidean distance) for all second-hand cars to all first-hand cars
        diff_quality = normalized_second_hand_quality[:, np.newaxis] - normalized_first_hand_quality
        diff_efficiency = normalized_second_hand_efficiency[:, np.newaxis] - normalized_first_hand_efficiency

        distances = np.sqrt(diff_quality ** 2 + diff_efficiency ** 2)

        # Find the closest first-hand car for each second-hand car
        closest_idxs = np.argmin(distances, axis=1)

        # Get the prices of the closest first-hand cars
        closest_prices = first_hand_prices[closest_idxs]

        # Adjust prices based on car age and depreciation
        adjusted_prices = closest_prices * (1 - self.delta) ** second_hand_ages

        return adjusted_prices

    def generate_ols(self,vehicle_dict_vecs_new_cars):
        """
        Use OLS regression to predict second-hand car prices based on quality and efficiency of new cars.
        
        Args:
            vehicle_dict_vecs_new_cars: Dictionary of new car attributes (Quality, Efficiency, Price).
            vehicle_dict_vecs_current_cars: Dictionary of second-hand car attributes (Quality, Efficiency, Age).
        """
        # Extract features and target variable from new cars
        first_hand_quality = vehicle_dict_vecs_new_cars["Quality_a_t"]
        first_hand_efficiency = vehicle_dict_vecs_new_cars["Eff_omega_a_t"]
        first_hand_prices = vehicle_dict_vecs_new_cars["price"]

        # Combine features into a matrix
        first_hand_features = np.column_stack((first_hand_quality, first_hand_efficiency))

        # Fit OLS model
        self.ols_model = LinearRegression()
        self.ols_model.fit(first_hand_features, first_hand_prices)

    def calc_car_price_ols(self, vehicle_dict_vecs_second_hand_cars):
        """
        Use OLS regression to predict second-hand car prices based on quality and efficiency of new cars.
        
        Args:
            vehicle_dict_vecs_new_cars: Dictionary of new car attributes (Quality, Efficiency, Price).
            vehicle_dict_vecs_current_cars: Dictionary of second-hand car attributes (Quality, Efficiency, Age).
        """

        # Extract features from second-hand cars
        # Extract Quality, Efficiency, and Age of second-hand cars
        second_hand_quality = vehicle_dict_vecs_second_hand_cars["Quality_a_t"]
        second_hand_efficiency = vehicle_dict_vecs_second_hand_cars["Eff_omega_a_t"]
        second_hand_ages = vehicle_dict_vecs_second_hand_cars["L_a_t"]

        # Combine second-hand features into a matrix
        second_hand_features = np.column_stack((second_hand_quality, second_hand_efficiency))

        # Predict second-hand prices using the OLS model
        predicted_prices = self.ols_model.predict(second_hand_features)

        # Adjust prices for depreciation based on age
        adjusted_prices = predicted_prices * (1 - self.delta) ** second_hand_ages

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

        data_dicts_second_hand = self.gen_vehicle_dict_vecs(self.cars_on_sale)
        # Calculate the price vector
        data_dicts_new_cars = self.gen_vehicle_dict_vecs_new_cars(self.vehicles_on_sale)

        price_vec = self.calc_car_price_heuristic(data_dicts_new_cars, data_dicts_second_hand)

        #price_vec = self.calc_car_price_ols(data_dicts_second_hand)

        #price_vec = self.calc_car_price_heuristic(data_dicts_new_cars, data_dicts_second_hand)
        #price_vec = self.calc_car_price_vec_old( data_dicts_second_hand, self.median_beta, self.median_gamma, self.U_sum)

        # Vectorized approach to identify cars below the scrap price
        below_scrap_mask = price_vec < self.scrap_price

        # Remove cars below the scrap price
        self.cars_on_sale = [
            vehicle for i, vehicle in enumerate(self.cars_on_sale) if not below_scrap_mask[i]
        ]

        # Update the prices of the remaining cars
        for i, vehicle in enumerate(self.cars_on_sale):
            vehicle.price = price_vec[i]

        #REMOVE EXCESS CARS
        if len(self.cars_on_sale) > self.max_num_cars:
            # Calculate how many cars to remove
            num_cars_to_remove = len(self.cars_on_sale) - self.max_num_cars
            # Randomly select cars to remove
            cars_to_remove = self.random_state_second_hand.choice(
                self.cars_on_sale, num_cars_to_remove, replace=False
            )
            # Add ages of removed cars
            self.age_second_hand_car_removed.extend(vehicle.L_a_t for vehicle in cars_to_remove)
            # Use list comprehension to filter out the cars to remove
            self.cars_on_sale = [car for car in self.cars_on_sale if car not in cars_to_remove] 
        

    def add_to_stock(self,vehicle):
        #add new car to stock
        #vehicle.price = self.calc_car_price_single(vehicle, self.median_beta, self.median_gamma, self.U_sum)
        vehicle.price = vehicle.price_second_hand_merchant
        vehicle.scenario = "second_hand"
        vehicle.second_hand_counter = 0
        self.cars_on_sale.append(vehicle)
    
    def remove_car(self, vehicle):
        self.car_bin.append(vehicle)
        self.cars_on_sale.remove(vehicle)

############################################################################################

    def set_up_time_series_social_network(self):
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
            car.L_a_t += 1
            car.second_hand_counter += 1#UPDATE THE STEPS ITS BEEN HERE
        
            if car.transportType == 2:#ICE
                car.fuel_cost_c = self.gas_price
            elif car.transportType == 3:
                car.fuel_cost_c = self.electricity_price
                car.e_t = self.electricity_emissions_intensity

    def next_step(self,gas_price, electricity_price, electricity_emissions_intensity, vehicles_on_sale, carbon_price, U_sum, U_vec_on_sale):
        
        self.gas_price =  gas_price
        self.electricity_price = electricity_price
        self.electricity_emissions_intensity = electricity_emissions_intensity
        self.vehicles_on_sale = vehicles_on_sale
        self.carbon_price = carbon_price
        self.U_vec_on_sale = U_vec_on_sale
        self.update_age_stock_prices_and_emissions_intensity(self.cars_on_sale)

        self.age_second_hand_car_removed = []
        
        self.U_sum = U_sum

        if self.cars_on_sale:
            self.update_stock_contents()

        self.profit = self.income - self.spent

        return self.cars_on_sale