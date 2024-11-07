class PersonalCar:
    def __init__(self, unique_id, firm, owner_id, component_string, parameters, attributes_fitness, sale_price):
        self.id = unique_id
        self.firm_id = firm
        self.owner_id = owner_id
        self.transportType = parameters["transportType"]
        self.component_string = component_string
        self.price = sale_price
        self.timer = 0

        self.scenario =  "current_car"

        self.attributes_fitness = attributes_fitness

        #self.second_hand_bool = 0
        
        # Updated Descriptions:
        self.Q_a_t =  self.attributes_fitness[0]  # Quality or attraction parameter for transport mode 'a' at time 't'.
                            # Represents the overall perceived attractiveness or utility that individuals derive from using this vehicle type.

        self.delta_z = parameters["delta_z"]  # Depreciation or distance-decay factor for the vehicle.
                               # Indicates how the utility decays over distance, accounting for the decreasing benefit or increased discomfort of using the vehicle over longer distances.

        self.L_a_t = 0  # Lifetime or longevity parameter of transport mode 'a' at time 't'.
                            # Represents how the utility of the vehicle evolves over time or lifetime, often indicating how wear and tear or aging affects the overall utility.

        self.omega_a_t = self.attributes_fitness[1]  # Efficiency scaling factor (ratio of kilometers per kilojoules) for transport mode 'a' at time 't'.
                                    # Represents the energy efficiency of the vehicle, i.e., how far the vehicle can travel per unit of energy consumption.

        self.c_z_t = self.attributes_fitness[2]  # Cost factor for the vehicle at time 't'.
                            # Represents the financial cost associated with using the vehicle for travel, including fuel, maintenance, and other operating costs.

        self.e_z_t = parameters["e_z_t"]  # Effort factor for the vehicle at time 't'.
                            # Represents the effort required to use the vehicle, which can include physical, cognitive, or time-related efforts associated with traveling using this mode of transportation.

        self.nu_z_i_t = parameters["nu_z_i_t"]  # User-specific utility adjustment factor for vehicle 'a' for individual 'i' at time 't'.
                                  # Represents a factor that adjusts utility based on personal preferences or circumstances of the user, such as preference for comfort, convenience, or other subjective factors.

        self.eta = parameters["eta"]  # Scaling constant affecting perceived utility.
                        # Represents a scaling factor used to adjust the overall utility calculation, typically to normalize or emphasize particular components in the utility function.

        self.emissions = parameters["emissions"]  # Emissions factor for the vehicle (E_a_t).
                                    # Represents the environmental impact of using the vehicle, often quantified as the amount of emissions (e.g., CO2) produced per unit distance traveled.