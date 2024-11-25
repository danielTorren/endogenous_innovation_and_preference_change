import numpy as np

class CarModel:
    def __init__(self, component_string, nk_landscape , parameters, choosen_tech_bool = False, firm = None):
        self.firm = firm

        self.owner_id = -5#NEED TO MAKE SURE NO ONE OWNS IT

        self.parameters =  parameters

        self.scenario = "new_car"
        self.transportType = self.parameters["transportType"]
        self.component_string = component_string
        self.decimal_value = int(component_string, 2)
        
        self.choosen_tech_bool = choosen_tech_bool#NEED TO FIND ANOTHER SOLUTION TO THIS CANT HAVE IT BE SHARE BY ALL THE FIRMS
        self.timer = 0

        #self.second_hand_bool = 0

        #FITNESS
        self.nk_landscape = nk_landscape
        #self.attributes_fitness = self.nk_landscape.calculate_fitness(self.component_string)
        self.attributes_fitness = self.nk_landscape.attributes_dict[self.component_string]
        

        self.inverted_tech_strings = self.nk_landscape.invert_bits_one_at_a_time(self.decimal_value)
        #self.inverted_tech_fitness = np.asarray([self.nk_landscape.calculate_fitness(inverted_string) for inverted_string in self.inverted_tech_strings])
        self.inverted_tech_fitness = np.asarray([self.nk_landscape.attributes_dict[inverted_string] for inverted_string in self.inverted_tech_strings])

        self.optimal_price_segments = {} 
        self.car_base_utility_segments = {}#,little u,  populated by a firm who is considering which car to buy, can be deleted afterwards?
        self.car_utility_segments_U = {}  #Capital U
        self.expected_profit_segments = {} 
        self.car_distance_segments = {}
        self.expected_profit = 0#used by firms to choose cars
        self.actual_profit = 0#used to pick what car to reserach 
        
        # Updated Descriptions:
        self.Quality_a_t =  self.attributes_fitness[0]  # Quality or attraction parameter for transport mode 'a' at time 't'.
                            # Represents the overall perceived attractiveness or utility that individuals derive from using this vehicle type.

        self.Eff_omega_a_t = self.attributes_fitness[1]  #FUEL EFFICIENCY km per kilojoules
        
        self.ProdCost_z_t = self.attributes_fitness[2]  #PRODUCTION COST

        self.fuel_cost_c_z = self.parameters["fuel_cost_c_z"]#Fuel cost

        self.delta_z = self.parameters["delta_z"]  # Depreciation or distance-decay factor for the vehicle.
                               # Indicates how the utility decays over distance, accounting for the decreasing benefit or increased discomfort of using the vehicle over longer distances.

        self.L_a_t = 0  # Lifetime or longevity parameter of transport mode 'a' at time 't'.
                            # Represents how the utility of the vehicle evolves over time or lifetime, often indicating how wear and tear or aging affects the overall utility.

        self.e_z_t = self.parameters["e_z_t"]  # Effort factor for the vehicle at time 't'.
                            # Represents the effort required to use the vehicle, which can include physical, cognitive, or time-related efforts associated with traveling using this mode of transportation.

        self.nu_z_i_t = self.parameters["nu_z_i_t"]  #AVEAGE time to do a km, worse for public transport
        
        self.eta = self.parameters["eta"]  # Scaling constant affecting perceived utility.
                        # Represents a scaling factor used to adjust the overall utility calculation, typically to normalize or emphasize particular components in the utility function.

        self.emissions = self.parameters["emissions"]  # Emissions factor for the vehicle (E_a_t).
                                    # Represents the environmental impact of using the vehicle, often quantified as the amount of emissions (e.g., CO2) produced per unit distance traveled.
    
    def update_timer(self):
        if not self.choosen_tech_bool:
            self.timer+=1