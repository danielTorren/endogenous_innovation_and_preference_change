class Public_transport:
    def __init__(self,  attributes_fitness):
        self.id = -1
        self.firm_id = -1
        self.component_string = "Public"
        self.attributes_fitness = attributes_fitness
        self.environmental_score = self.attributes_fitness[1]
