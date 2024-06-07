#NEED TO HAVE THE FOLLOWING PROPERTIES:
#car.X_E, car.X_Q, car.X_C

class Car:
    def __init__(self, unique_id, firm_id, component_string, attributes_fitness, choosen_tech_bool):
        self.id = unique_id
        self.firm_id = firm_id
        
        self.component_string = component_string
        self.decimal_value = int(component_string, 2)
        self.attributes_fitness = attributes_fitness
        self.cost, self.emissions, self.quality = attributes_fitness#THE ORDER HAS TO BE CORRECT!!
        self.choosen_tech_bool = choosen_tech_bool#NEED TO FIND ANOTHER SOLUTION TO THIS CANT HAVE IT BE SHARE BY ALL THE FIRMS
        self.timer = 0

    def update_timer(self):
        if ~self.choosen_tech_bool:
            self.timer+=1