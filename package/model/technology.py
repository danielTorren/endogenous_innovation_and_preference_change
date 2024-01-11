#class for creating technologies

class Technology:
    def __init__(self, component_string, emissions_intensity, cost, choosen_tech_bool):
        self.component_string = component_string
        self.emissions_intensity = emissions_intensity
        self.cost = cost
        self.choosen_tech_bool = choosen_tech_bool#NEED TO FIND ANOTHER SOLUTION TO THIS CANT HAVE IT BE SHARE BY ALL THE FIRMS
        self.timer = 0
        self.fitness = None

    def update_timer(self):
        if ~self.choosen_tech_bool:
            self.timer+=1