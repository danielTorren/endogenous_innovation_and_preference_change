#NEED TO HAVE THE FOLLOWING PROPERTIES:
#car.X_E, car.X_Q, car.X_C
import numpy as np
class Car:
    def __init__(self, unique_id, firm_id, component_string, attributes_fitness, choosen_tech_bool, N, nk_landscape):
        self.id = unique_id
        self.N = N
        self.firm_id = firm_id
        
        self.bit_position = np.arange(self.N)
        self.component_string = component_string
        self.decimal_value = int(component_string, 2)
        self.inverted_tech_strings = self.invert_bits_one_at_a_time()
        
        self.cost, self.emissions, self.quality = attributes_fitness#THE ORDER HAS TO BE CORRECT!!
        self.choosen_tech_bool = choosen_tech_bool#NEED TO FIND ANOTHER SOLUTION TO THIS CANT HAVE IT BE SHARE BY ALL THE FIRMS
        self.timer = 0

        #FITNESS
        self.nk_landscape = nk_landscape
        self.attributes_fitness = self.nk_landscape.calculate_fitness(self.component_string)
        self.inverted_tech_fitness = np.asarray([self.nk_landscape.calculate_fitness(inverted_string) for inverted_string in self.inverted_tech_strings])
        
    def invert_bits_one_at_a_time(self):
        """THIS IS ONLY USED ONCE I THINK"""
        inverted_binary_values = []
        for bit_position in range(self.N):
            inverted_value = self.decimal_value ^ (1 << bit_position)
            inverted_binary_value = format(inverted_value, f'0{self.N}b')
            inverted_binary_values.append(inverted_binary_value)
        return inverted_binary_values
    
    def update_timer(self):
        if ~self.choosen_tech_bool:
            self.timer+=1