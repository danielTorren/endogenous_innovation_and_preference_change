class SecondHandMerchant:
    def __init__(self, unique_id, age_limit_second_hand):
        self.id = unique_id
        self.cars_on_sale = []

        self.age_limit_second_hand = age_limit_second_hand
        self.set_up_time_series_social_network()
        self.car_bin = []

    def calc_price_car(self,vehicle):
        #calc the price of all cars, maybe change this to just incoming cars
        sale_price_second_hand = vehicle.price*(1-vehicle.delta_z)**(vehicle.L_a_t)

        return sale_price_second_hand
    
    def update_age_stock(self):
        #update the age of vehicles accrodingly 
        for vehicle in self.cars_on_sale:
            vehicle.L_a_t += 1
            vehicle.second_hand_counter += 1#UPDATE THE STEPS ITS BEEN HERE

    def update_stock_contents(self):
        for vehicle in self.cars_on_sale:
            if vehicle.transportType == 2:#ICE
                vehicle.fuel_cost_c_z = self.gas_price
            else:#EV
                vehicle.fuel_cost_c_z = self.electricity_price
                vehicle.e_z_t = self.electricity_emissions_intensity
                vehicle.nu_z_i_t= self.nu_z_i_t_EV
        
            if vehicle.second_hand_counter > self.age_limit_second_hand:
                self.remove_car(vehicle)
            else:
                vehicle.price = self.calc_price_car(vehicle)

    def add_to_stock(self,vehicle):
        #add new car to stock
        vehicle.price = self.calc_price_car(vehicle)
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

    def next_step(self,gas_price, electricity_price, electricity_emissions_intensity, nu_z_i_t_EV):
        
        self.gas_price =  gas_price
        self.electricity_price = electricity_price
        self.electricity_emissions_intensity = electricity_emissions_intensity
        self.nu_z_i_t_EV= nu_z_i_t_EV
    
        self.update_age_stock()
        self.update_stock_contents()
        #print(len(self.cars_on_sale))
        return self.cars_on_sale