class Public_Transport():
    def __init__(self,  attributes_fitness):
        self.attributes_fitness = attributes_fitness
        self.environmental_score = self.attributes_fitness[1]  

class Urban_Public_Transport(Public_Transport):
    def __init__(self):
        self.id = -1
        self.firm_id = -1
        self.transportType = -1
        self.component_string = "urbanPublic"

class Rural_Public_Transport(Public_Transport):
    def __init__(self):
        self.id = -2
        self.firm_id = -2
        self.transportType = -2
        self.component_string = "ruralPublic"
        
