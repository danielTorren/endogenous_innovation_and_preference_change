class SecondHandMerchant:
    def __init__(self, unique_id):
        self.id = unique_id
        self.cars_on_sale = []

    def calc_price_car(vehicle):
        #calc the price of all cars, maybe change this to just incoming cars
        sale_price_second_hand = vehicle.sale_price*(1-vehicle.delta_z)**(vehicle.L_a_t)

        return sale_price_second_hand
    
    def update_age_stock(self):
        #update the age of vehicles accrodingly 
        for vehicle in self.cars_on_sale:
            vehicle.L_a_t += 1

    def update_stock_contents(self):
        for vehicle in self.cars_on_sale:
            vehicle.price = self.calc_price_car(vehicle)
    
    def add_to_stock(self,vehicle):
        #add new car to stock
        vehicle.price = self.calc_price_car(vehicle)
        vehicle.scenario = "private_unassigned"
        self.cars_on_sale.append(vehicle)
    
    def remove_car(self, vehicle):
        self.cars_on_sale.remove(vehicle)
    
    def next_step(self):
        self.update_age_stock()
        self.update_stock_contents()

        return self.cars_on_sale