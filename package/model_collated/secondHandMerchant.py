class SecondHandMerchant:
    def __init__(self, unique_id, age_limit_second_hand):
        self.id = unique_id
        self.cars_on_sale = []

        self.age_limit_second_hand = age_limit_second_hand
        self.set_up_time_series_social_network()

    def calc_price_car(self,vehicle):
        #calc the price of all cars, maybe change this to just incoming cars
        sale_price_second_hand = vehicle.price*(1-vehicle.delta_z)**(vehicle.L_a_t)

        return sale_price_second_hand
    
    def update_age_stock(self):
        #update the age of vehicles accrodingly 
        for vehicle in self.cars_on_sale:
            vehicle.L_a_t += 1

    def update_stock_contents(self):
        for vehicle in self.cars_on_sale:
            if vehicle.L_a_t > self.age_limit_second_hand:
                self.remove_car(vehicle)
            else:
                vehicle.price = self.calc_price_car(vehicle)

    def add_to_stock(self,vehicle):
        #add new car to stock
        vehicle.price = self.calc_price_car(vehicle)
        vehicle.scenario = "private_unassigned"
        self.cars_on_sale.append(vehicle)
        #print("HELLOW")
    
    def remove_car(self, vehicle):
        self.cars_on_sale.remove(vehicle)
    
    def set_up_time_series_social_network(self):
        self.history_num_second_hand = []

    def save_timeseries_second_hand_merchant(self):
        self.history_num_second_hand.append(len(self.cars_on_sale))

    def next_step(self):
        self.update_age_stock()
        self.update_stock_contents()

        return self.cars_on_sale