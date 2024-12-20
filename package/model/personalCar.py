class PersonalCar:
    def __init__(self, unique_id, firm, owner_id, component_string, parameters, attributes_fitness, sale_price, init_car = 0):
        self.id = unique_id
        self.firm = firm
        self.owner_id = owner_id
        self.transportType = parameters["transportType"]
        self.component_string = component_string
        self.price = sale_price
        self.original_price = sale_price
        self.last_price_paid = sale_price

        self.timer = 0

        self.init_car = init_car

        self.scenario =  "current_car"

        self.attributes_fitness = attributes_fitness

        #self.second_hand_bool = 0
        
        # Updated Descriptions:
        self.Quality_a_t =  self.attributes_fitness[0]  # Quality or attraction parameter for transport mode 'a' at time 't'.
                            # Represents the overall perceived attractiveness or utility that individuals derive from using this vehicle type.

        self.delta = parameters["delta"]  # Depreciation or distance-decay factor for the vehicle.
                               # Indicates how the utility decays over distance, accounting for the decreasing benefit or increased discomfort of using the vehicle over longer distances.

        self.L_a_t = 0  # Lifetime or longevity parameter of transport mode 'a' at time 't'.
                            # Represents how the utility of the vehicle evolves over time or lifetime, often indicating how wear and tear or aging affects the overall utility.

        self.Eff_omega_a_t = self.attributes_fitness[1]  #FUEL EFFICIENCY km per kilojoules
        self.ProdCost_t = self.attributes_fitness[2]  #PRODUCTION COST
        self.fuel_cost_c = parameters["fuel_cost_c"]#Fuel cost
        self.e_t = parameters["e_t"]  # Effort factor for the vehicle at time 't'.
                            # Represents the effort required to use the vehicle, which can include physical, cognitive, or time-related efforts associated with traveling using this mode of transportation.

        self.emissions = parameters["production_emissions"]  # Emissions factor for the vehicle (E_a_t).
                                    # Represents the environmental impact of using the vehicle, often quantified as the amount of emissions (e.g., CO2) produced per unit distance traveled.
        
        self.total_distance = 0
        self.total_driving_emmissions = 0
        self.total_emissions = self.emissions

    def update_timer(self):
        self.L_a_t += 1#0.5 + np.random.uniform()#THE ADDED AGE RANGES FROM 0.5 to 1.5 to avoid cyclic behaviour of agents