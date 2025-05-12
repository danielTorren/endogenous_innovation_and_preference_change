class CarModel:
    """
    Represents a new car model designed using an NK fitness landscape.

    Each CarModel object encapsulates technical attributes (e.g. quality, efficiency, cost),
    technology string, firm information, and market-oriented metrics (e.g. utility and profit per segment).
    """
        
    def __init__(self, component_string, nk_landscape , parameters, choosen_tech_bool = False, firm = None):
        """
        Initialize a new CarModel instance with attributes derived from a binary design string and an NK landscape.

        Args:
            component_string (str): Binary string representing the technical configuration of the car.
            nk_landscape (NKModel): The NK landscape used to evaluate fitness attributes.
            parameters (dict): Dictionary of fixed vehicle parameters such as:
                - 'transportType': 2 (ICE) or 3 (EV)
                - 'fuel_cost_c': Fuel or electricity cost
                - 'delta': Efficiency depreciation rate
                - 'e_t': Emissions intensity (per unit energy)
                - 'production_emissions': Emissions from production
                - 'fuel_tank': Capacity for ICE (kWh)
            choosen_tech_bool (bool, optional): Flag indicating whether this car was chosen for production. Defaults to False.
            firm (object, optional): The firm that created the model. Defaults to None.
        """

        self.firm = firm

        self.owner_id = -5#NEED TO MAKE SURE NO ONE OWNS IT

        self.parameters =  parameters

        self.scenario = "new_car"
        self.transportType = self.parameters["transportType"]
        self.component_string = component_string
        self.decimal_value = int(component_string, 2)
        
        self.choosen_tech_bool = choosen_tech_bool
        self.timer = 0

        #FITNESS
        self.nk_landscape = nk_landscape
        self.attributes_fitness = self.nk_landscape.retrieve_info(self.component_string)
        self.inverted_tech_strings = self.nk_landscape.invert_bits_one_at_a_time(self.decimal_value)

        self.optimal_price_segments = {} 
        self.B_segments = {}
        self.car_utility_segments_U = {}
        self.expected_profit_segments = {} 
        self.expected_profit = 0#used by firms to choose cars
        self.actual_profit = 0#used to pick what car to research 
        
        # Updated Descriptions:
        self.Quality_a_t =  self.attributes_fitness[0]  # Quality or attraction parameter for transport mode 'a' at time 't'. Represents the overall perceived attractiveness or utility that individuals derive from using this vehicle type.

        self.Eff_omega_a_t = self.attributes_fitness[1] #FUEL EFFICIENCY km per kwHrs
        
        self.ProdCost_t = self.attributes_fitness[2]#PRODUCTION COST

        self.fuel_cost_c = self.parameters["fuel_cost_c"]#Fuel cost

        self.delta = self.parameters["delta"]# Depreciation of efficiency for the vehicle.
                        
        self.L_a_t = 0  # Lifetime parameter of car 'a' at time 't'.

        self.e_t = self.parameters["e_t"]  # fuel emissions associated with model either from gas then fixed or electricity then variable. 
        self.emissions = self.parameters["production_emissions"]  # Emissions from car production

        if self.transportType == 2:
            self.B = self.parameters["fuel_tank"] #ICE VEHICLE kWhr, fixed
        else:
            self.B = self.attributes_fitness[3]#BATTERY kWhr

    def update_timer(self):
        """
        Increment the internal timer for tracking how long the car model has been known.
        """
        if not self.choosen_tech_bool:
            self.timer+=1