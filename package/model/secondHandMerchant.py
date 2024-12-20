import numpy as np
class SecondHandMerchant:
    def __init__(self, unique_id, parameters_second_hand):
        self.id = unique_id
        self.cars_on_sale = []

        self.age_limit_second_hand = parameters_second_hand["age_limit_second_hand"]
        self.set_up_time_series_social_network()
        self.car_bin = []

        self.alpha = parameters_second_hand["alpha"]
        self.r = parameters_second_hand["r"]

    def calc_median(self, beta_vec, gamma_vec):
        self.median_beta =  np.median(beta_vec)
        self.median_gamma = np.median(gamma_vec)

    def calc_car_price(self, vehicle, beta, gamma, U_sum):

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
        vehicle.price = self.calc_car_price(vehicle, self.median_beta, self.median_gamma, self.U_sum)
        vehicle.scenario = "second_hand"
        vehicle.second_hand_counter = 0
        self.cars_on_sale.append(vehicle)
    
    def remove_car(self, vehicle):
        self.car_bin.append(vehicle)
        self.cars_on_sale.remove(vehicle)
    
    def set_up_time_series_social_network(self):
        self.history_num_second_hand = []

    def save_timeseries_second_hand_merchant(self):
        self.history_num_second_hand.append(len(self.cars_on_sale))

    def next_step(self,gas_price, electricity_price, electricity_emissions_intensity, U_sum, carbon_price):
        
        self.gas_price =  gas_price
        self.electricity_price = electricity_price
        self.electricity_emissions_intensity = electricity_emissions_intensity
        self.U_sum = U_sum
        self.carbon_price = carbon_price

        self.update_age_stock()
        self.update_stock_contents()

        return self.cars_on_sale