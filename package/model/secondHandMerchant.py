import numpy as np
from pyro import param

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
        self.price_adjust_monthly = parameters_second_hand["price_adjust_monthly"]
        self.fixed_alternative_mark_up = parameters_second_hand["fixed_alternative_mark_up"]
        self.random_state_second_hand = np.random.RandomState(parameters_second_hand["remove_seed"])

        self.spent = 0
        self.income = 0
        self.profit = 0

    def calc_median(self, beta_vec, gamma_vec):
        self.median_beta =  np.median(beta_vec)
        self.median_gamma = np.median(gamma_vec)

    def gen_vehicle_dict_vecs(self, list_vehicles):
        # Initialize dictionary to hold lists of vehicle properties
        vehicle_dict_vecs = {
            "Quality_a_t": [], 
            "Eff_omega_a_t": [], 
            "price": [], 
            "delta": [], 
            "production_emissions": [],
            "fuel_cost_c": [], 
            "e_t": [],
            "L_a_t": [],
            "transportType": [],
            "cost_second_hand_merchant": []
        }

        # Iterate over each vehicle to populate the arrays
        for vehicle in list_vehicles:
            if vehicle.transportType == 2:
                vehicle.fuel_cost_c = self.gas_price
            else:
                vehicle.fuel_cost_c = self.electricity_price
                vehicle.e_t = self.electricity_emissions_intensity
            
            vehicle_dict_vecs["Quality_a_t"].append(vehicle.Quality_a_t)
            vehicle_dict_vecs["Eff_omega_a_t"].append(vehicle.Eff_omega_a_t)
            vehicle_dict_vecs["price"].append(vehicle.price)
            vehicle_dict_vecs["delta"].append(vehicle.delta)
            vehicle_dict_vecs["production_emissions"].append(vehicle.emissions)
            vehicle_dict_vecs["fuel_cost_c"].append(vehicle.fuel_cost_c)
            vehicle_dict_vecs["e_t"].append(vehicle.e_t)
            vehicle_dict_vecs["L_a_t"].append(vehicle.L_a_t)
            vehicle_dict_vecs["transportType"].append(vehicle.transportType)
            vehicle_dict_vecs["cost_second_hand_merchant"].append(vehicle.cost_second_hand_merchant)
        # convert lists to numpy arrays for vectorised operations
        for key in vehicle_dict_vecs:
            vehicle_dict_vecs[key] = np.array(vehicle_dict_vecs[key])

        return vehicle_dict_vecs
    
    def calc_car_price_vec(self, vehicle_dict_vecs, beta, gamma, U_sum):

        #DISTANCE
                # Compute numerator for all vehicles
        numerator = (
            self.alpha * vehicle_dict_vecs["Quality_a_t"] *
            ((1 - vehicle_dict_vecs["delta"]) ** vehicle_dict_vecs["L_a_t"])
        ) 
        denominator = (
            ((beta/vehicle_dict_vecs["Eff_omega_a_t"]) * (vehicle_dict_vecs["fuel_cost_c"] + self.carbon_price*vehicle_dict_vecs["e_t"])) +
            ((gamma/ vehicle_dict_vecs["Eff_omega_a_t"]) * vehicle_dict_vecs["e_t"])
        )  # Shape: (num_individuals, num_vehicles)

        # Calculate optimal distance matrix for each individual-vehicle pair
        d_i_t_vec = (numerator / denominator) ** (1 / (1 - self.alpha))

        #present UTILITY
        # Compute cost component based on transport type, with conditional operations
        cost_component = (beta/ vehicle_dict_vecs["Eff_omega_a_t"]) * (vehicle_dict_vecs["fuel_cost_c"] + self.carbon_price*vehicle_dict_vecs["e_t"]) + ((gamma/ vehicle_dict_vecs["Eff_omega_a_t"]) * vehicle_dict_vecs["e_t"])
        # Compute the commuting utility for each individual-vehicle pair
        present_utility_vec = np.maximum(
            0,
            vehicle_dict_vecs["Quality_a_t"] * ((1 - vehicle_dict_vecs["delta"]) ** vehicle_dict_vecs["L_a_t"]) * (d_i_t_vec ** self.alpha) - d_i_t_vec * cost_component
        )

        # Save the base utility
        B_vec = present_utility_vec/(self.r + (np.log(1+vehicle_dict_vecs["delta"]))/(1-self.alpha))

        inside_component = U_sum*(U_sum + B_vec - beta*vehicle_dict_vecs["cost_second_hand_merchant"])

        # Adjust the component to avoid negative square roots
        inside_component_adjusted = np.maximum(inside_component, 0)  # Replace negatives with 0

        price_vec = np.where(
            inside_component < 0,
            vehicle_dict_vecs["cost_second_hand_merchant"]*(1+self.fixed_alternative_mark_up),
            np.maximum(vehicle_dict_vecs["cost_second_hand_merchant"], (U_sum  + B_vec - np.sqrt(inside_component_adjusted) )/beta)
        )

        return price_vec
    
    def calc_car_price_single(self, vehicle, beta, gamma, U_sum):

        #UPDATE EMMISSION AND PRICES, THIS WORKS FOR BOTH PRODUCTION AND INNOVATION
        if vehicle.transportType == 2:#ICE
            vehicle.fuel_cost_c = self.gas_price
        else:#EV
            vehicle.fuel_cost_c = self.electricity_price

        #DISTANCE
        numerator = self.alpha * vehicle.Quality_a_t * (1 - vehicle.delta) ** vehicle.L_a_t
        denominator = ((beta/vehicle.Eff_omega_a_t) * (vehicle.fuel_cost_c + self.carbon_price*vehicle.e_t) +
                    (gamma/vehicle.Eff_omega_a_t) * vehicle.e_t)

        d_i_t = (numerator / denominator) ** (1 / (1 - self.alpha))

        #present UTILITY
        cost_component = (beta / vehicle.Eff_omega_a_t) * (vehicle.fuel_cost_c + self.carbon_price*vehicle.e_t) + (gamma/vehicle.Eff_omega_a_t) * vehicle.e_t
        utility = vehicle.Quality_a_t * (1 - vehicle.delta) ** vehicle.L_a_t * (d_i_t ** self.alpha) - d_i_t * cost_component

        # Ensure utility is non-negative
        utility_final = max(0, utility)

        # Save the base utility
        B = utility_final/(self.r + (np.log(1+vehicle.delta))/(1-self.alpha))

        inside_component = U_sum*(U_sum + B - beta*vehicle.cost_second_hand_merchant)
        if inside_component < 0:
            price = vehicle.cost_second_hand_merchant*(1+self.fixed_alternative_mark_up)
        else:
            price = max(vehicle.cost_second_hand_merchant,(U_sum  + B - np.sqrt(inside_component) )/beta)

        return price
    
    def calc_price_car_old(self,vehicle):
        #calc the price of all cars, maybe change this to just incoming cars
        sale_price_second_hand = vehicle.original_price*2#(1 - vehicle.delta)**(vehicle.L_a_t)
        return sale_price_second_hand
    
    def update_age_stock(self):
        #update the age of vehicles accrodingly 
        for vehicle in self.cars_on_sale:
            vehicle.L_a_t += 1
            vehicle.second_hand_counter += 1#UPDATE THE STEPS ITS BEEN HERE

    def update_stock_contents(self):
        #check len of list
        if len(self.cars_on_sale) > self.max_num_cars:
            cars_to_remove = self.random_state_second_hand.choice(self.max_num_cars, self.max_num_cars - self.cars_on_sale)#RANDOMLY REMOVE CARS FROM THE SALE LIST
            for vehicle in cars_to_remove:
             self.age_second_hand_car_removed.append(vehicle.L_a_t)
            self.cars_on_sale.remove(cars_to_remove)
    
        for vehicle in self.cars_on_sale:       
            if vehicle.second_hand_counter > self.age_limit_second_hand:
                self.age_second_hand_car_removed.append(vehicle.L_a_t)
                self.cars_on_sale.remove(vehicle)
                
        
        data_dicts = self.gen_vehicle_dict_vecs(self.cars_on_sale)

        price_vec = self.calc_car_price_vec(data_dicts, self.median_beta, self.median_gamma, self.U_sum)

        for i, vehicle in enumerate(self.cars_on_sale):
            price = min((1+self.price_adjust_monthly)*vehicle.price , max((1 -self.price_adjust_monthly)*vehicle.price , price_vec[i]))
            vehicle.price = price 

    def update_stock_contents_old(self):
        for vehicle in self.cars_on_sale:
            if vehicle.transportType == 2:#ICE
                vehicle.fuel_cost_c = self.gas_price
            else:#EV
                vehicle.fuel_cost_c = self.electricity_price
                vehicle.e_t = self.electricity_emissions_intensity
        
            if vehicle.second_hand_counter > self.age_limit_second_hand:
                self.remove_car(vehicle)
            else:
                vehicle.price = self.calc_car_price(vehicle, self.median_beta, self.median_gamma, self.U_sum)

    def add_to_stock(self,vehicle):
        #add new car to stock
        vehicle.price = self.calc_car_price_single(vehicle, self.median_beta, self.median_gamma, self.U_sum)
        vehicle.scenario = "second_hand"
        vehicle.second_hand_counter = 0
        self.cars_on_sale.append(vehicle)
    
    def remove_car(self, vehicle):
        self.car_bin.append(vehicle)
        self.cars_on_sale.remove(vehicle)
    
    def set_up_time_series_social_network(self):
        self.history_num_second_hand = []
        self.history_profit = []
        self.history_age_second_hand_car_removed = []

    def save_timeseries_second_hand_merchant(self):
        self.history_num_second_hand.append(len(self.cars_on_sale))
        self.history_profit.append(self.profit )
        self.history_age_second_hand_car_removed.append(self.age_second_hand_car_removed)

    def next_step(self,gas_price, electricity_price, electricity_emissions_intensity, U_sum, carbon_price):
        
        self.gas_price =  gas_price
        self.electricity_price = electricity_price
        self.electricity_emissions_intensity = electricity_emissions_intensity
        self.U_sum = U_sum
        self.carbon_price = carbon_price

        self.age_second_hand_car_removed = []
        
        self.update_age_stock()

        if self.cars_on_sale:
            self.update_stock_contents()

        self.profit = self.income - self.spent

        return self.cars_on_sale