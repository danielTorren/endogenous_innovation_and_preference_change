class Public_Transport():
    def __init__(self, parameters):

        self.id = parameters["id"]
        self.firm =  parameters["firm"]
        self.transportType = parameters["transportType"]
        self.price =  parameters["price"]
        self.owner_id = -1
        self.scenario = "public_transport"

        self.firm_cars_users = 0 #how many people use the transport!
        
        # Updated Descriptions:
        self.Quality_a_t =   parameters["attributes"][0]  # Quality or attraction parameter for transport mode 'a' at time 't'.
                            # Represents the overall perceived attractiveness or utility that individuals derive from using this vehicle type.

        self.delta_z = parameters["delta_z"]  # Depreciation or distance-decay factor for the vehicle.
                               # Indicates how the utility decays over distance, accounting for the decreasing benefit or increased discomfort of using the vehicle over longer distances.
        self.fuel_cost_c_z = parameters["fuel_cost_c_z"]#Fuel cost
        self.L_a_t = 0  # Lifetime or longevity parameter of transport mode 'a' at time 't'.
                            # Represents how the utility of the vehicle evolves over time or lifetime, often indicating how wear and tear or aging affects the overall utility.

        self.Eff_omega_a_t =  parameters["attributes"][1]  #FUEL EFFICIENCY km per kilojoules
        self.ProdCost_z_t =  parameters["attributes"][2]  #PRODUCTION COST

        self.e_z_t = parameters["e_z_t"]  # Effort factor for the vehicle at time 't'.
                            # Represents the effort required to use the vehicle, which can include physical, cognitive, or time-related efforts associated with traveling using this mode of transportation.

        self.nu_z_i_t = parameters["nu_z_i_t"]  #AVEAGE time to do a km, worse for public transport
        self.eta = parameters["eta"]  # Scaling constant affecting perceived utility.
                        # Represents a scaling factor used to adjust the overall utility calculation, typically to normalize or emphasize particular components in the utility function.

        self.emissions = parameters["emissions"]  # Emissions factor for the vehicle (E_a_t).
                                    # Represents the environmental impact of using the vehicle, often quantified as the amount of emissions (e.g., CO2) produced per unit distance traveled.
        
        

    def set_up_time_series_firm(self):
        self.history_profit = []

    def save_timeseries_data_firm(self):
        self.history_profit.append(self.firm_cars_users)