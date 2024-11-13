import numpy as np

class VehicleUser:
    def __init__(self, user_id, chi, gamma, beta, origin, d_i_min, parameters_vehicle_user):
        self.t_vehicle_user = 0

        self.user_id = user_id  # Unique identifier for each user
        self.chi = chi  # Threshold for EV consideration (openness to innovation)
        self.gamma = gamma  # Environmental sensitivity
        self.beta = beta  # Cost sensitivity
        self.origin = origin  # User's origin (urban or rural), impacts public transport availability
        self.d_i_min = d_i_min

        self.save_timeseries_data_state = parameters_vehicle_user["save_timeseries_data_state"]
        self.compression_factor_state = parameters_vehicle_user["compression_factor_state"]
        self.vehicles_available = parameters_vehicle_user["vehicles_available"]
        self.kappa = parameters_vehicle_user["kappa"]
        self.alpha = parameters_vehicle_user["alpha"]
        self.r = parameters_vehicle_user["r"]
        self.eta = parameters_vehicle_user["eta"]
        self.mu = parameters_vehicle_user["mu"]

        self.vehicle = self.decide_vehicle_init(self.vehicles_available)  # Initial vehicle decision
        self.current_vehicle_type = self.vehicle.transportType

        #if self.save_timeseries_data_state:
        #    self.set_up_time_series_vehicle_user()


    ###############################################################################################################################
    #INIT

    def decide_vehicle_init(self, vehicles_available):
        """
        Decide whether to purchase a new vehicle or keep the current one.

        Parameters:
        vehicles_available (list): List of available vehicles to consider for purchase.

        Returns:
        Vehicle: The chosen vehicle.
        """
        # Calculate utility for each available vehicle
        utilities_new = []
        for vehicle in vehicles_available:
            utilities_new.append(self.calculate_utility_init(vehicle))  # Example scenario for new vehicles

        # Combine current vehicle utility with new vehicles
        utilities = np.array(utilities_new)

        # Calculate the probabilities of choosing each vehicle
        sum_utilities = np.sum(utilities ** self.kappa)
        probability_choose = (utilities ** self.kappa) / sum_utilities

        # Include current vehicle in the list of all vehicles
        all_vehicles = vehicles_available

        # Choose a vehicle based on the calculated probabilities
        chosen_vehicle = np.random.choice(all_vehicles, size=1, p=probability_choose)[0]

        return chosen_vehicle  # Can be the same as the previous vehicle

    def calculate_utility_init(self, vehicle):
        """
        Calculate the lifetime utility using the closed-form solution based on different conditions.

        Parameters:
        vehicle (Vehicle): The vehicle for which the utility is being calculated.
        scenario (str): The scenario to determine how the utility is adjusted.

        Returns:
        float: The calculated lifetime utility U_{a,i,t}.
        """

        scenario = vehicle.scenario#is it second hand, first hand or public transport.

        # Calculate distance and commuting utility
        d_i_t = self.actual_distance(vehicle)
        commuting_util = self.commuting_utility(vehicle, d_i_t)  # Example z value (should be scenario-specific)

        # Closed-form solution for lifetime utility
        denominator = self.r + (1 - vehicle.delta_z) / (1 - self.alpha)
        if denominator == 0:
            raise ValueError("The denominator is zero, adjust the parameters to avoid division by zero.")
        
        # Calculate the base lifetime utility using the closed form
        base_utility = commuting_util / denominator

        """
        Cases:
        1. buy brand new car and you have no old car (could be literally no car or that you use public transport)
        2. buy brand new car and you have an old car which you sell to the second hand man
        3. buy second hand car and you have no old car (could be literally no car or that you use public transport)
        4. buy second hand car and you have old car which you sell to the second hand man
        5. you choose public transport and you have old car which you sell to the second hand man
        6. you choose public tranpsort and you have no old car (could be literally no car or that you use public transport)
        7. you own car and you keep the same car
        """
        
        # Adjust the lifetime utility based on the scenario
        if scenario in ["public_transport", "second_hand"]:#Cases 3 and 6, choosing PUBLIC TRANSPORT or second hand car without owning a second hand car due to public tranport or no car
            U_a_i_t = base_utility - self.beta * vehicle.price
        elif scenario == "new_car":#CASE 1, buyign new without owning a second hand car
            U_a_i_t = base_utility - self.beta * vehicle.price - self.gamma * vehicle.emissions
        else:
            raise ValueError("Invalid scenario specified. No car is owned")

        return U_a_i_t

    ###################################################################################################################################
    def commuting_utility(self, vehicle, d_i_t):
        """
        Calculate the commuting utility based on different conditions.

        Parameters:
        vehicle (Vehicle): The vehicle being considered for commuting.
        d_i_t (float): Distance traveled during a time-step.
        z (float): Condition parameter to differentiate public and private.

        Returns:
        float: The calculated commuting utility u_{a,i,t}.
        """
        Quality_a_t = vehicle.Quality_a_t
        delta_z = vehicle.delta_z
        L_a_t = vehicle.L_a_t
        Eff_omega_a_t = vehicle.Eff_omega_a_t
        fuel_cost_c_z = vehicle.fuel_cost_c_z
        e_z_t = vehicle.e_z_t
        nu_z_i_t = vehicle.nu_z_i_t
        z = vehicle.transportType

        # Calculate commuting utility based on conditions for z
        if z > 1:
            # If z' > 1, include all cost components
            cost_component = self.beta * (1 / Eff_omega_a_t) * fuel_cost_c_z + self.gamma * (1 / Eff_omega_a_t) * e_z_t + self.eta * nu_z_i_t
            utility = Quality_a_t * (1 - delta_z) ** L_a_t * (d_i_t ** self.alpha) - d_i_t * cost_component
        else:
            # If z' <= 1, include only the eta * nu component
            utility = Quality_a_t * (1 - delta_z) ** L_a_t * (d_i_t ** self.alpha) - d_i_t * (self.eta * nu_z_i_t)

        # Ensure utility is non-negative
        utility = max(0, utility)

        return utility

    def optimal_distance(self, vehicle):
        """
        Calculate the optimal distance based on the vehicle properties.

        Parameters:
        vehicle (Vehicle): The vehicle for which the optimal distance is calculated.

        Returns:
        float: The calculated optimal distance, d^*_{a,i,t}.
        """
        numerator = self.alpha * vehicle.Quality_a_t * (1 - vehicle.delta_z) ** vehicle.L_a_t
        denominator = (self.beta * vehicle.Eff_omega_a_t ** -1 * vehicle.fuel_cost_c_z +
                       self.gamma * vehicle.Eff_omega_a_t ** -1 * vehicle.e_z_t +
                       vehicle.eta * vehicle.nu_z_i_t)

        # Compute optimal distance
        if denominator == 0:
            raise ValueError("The denominator is zero, adjust the parameters to avoid division by zero.")

        optimal_d = (numerator / denominator) ** (1 / (1 - self.alpha))
        return optimal_d

    def actual_distance(self, vehicle):
        """
        Calculate the actual distance traveled based on the optimal and minimum distance.

        Parameters:
        vehicle (Vehicle): The vehicle for which the distance is calculated.

        Returns:
        float: The actual distance traveled, d_{i,t}.
        """
        optimal_d = self.optimal_distance(vehicle)
        return max(self.d_i_min, optimal_d)
    
    def calculate_utility(self, vehicle):
        """
        Cases:
        1. buy brand new car and you have no old car (could be literally no car or that you use public transport)
        2. buy brand new car and you have an old car which you sell to the second hand man
        3. buy second hand car and you have no old car (could be literally no car or that you use public transport)
        4. buy second hand car and you have old car which you sell to the second hand man
        5. you choose public transport and you have old car which you sell to the second hand man
        6. you choose public tranpsort and you have no old car (could be literally no car or that you use public transport)
        7. you own car and you keep the same car
        """
        # Avoid recalculating the same values for commuting utility or actual distance
        d_i_t = self.actual_distance(vehicle)
        commuting_util = self.commuting_utility(vehicle, d_i_t)

        denominator = self.r + (1 - vehicle.delta_z) / (1 - self.alpha)
        if denominator == 0:
            raise ValueError("The denominator is zero, adjust parameters to avoid division by zero.")

        base_utility = commuting_util / denominator
        scenario = vehicle.scenario

        if self.vehicle.transportType > 1:#you own a car currently
            if scenario == "current_car":#you own a car and keep the car there is no change case 7
                U_a_i_t = base_utility
            elif scenario in ["public_transport", "second_hand"]:#you own a car and now are going to pick a second hand car or public transport
                U_a_i_t = base_utility - self.beta * (vehicle.price - self.vehicle.price / (1 + self.mu))
            elif scenario == "new_car":#you own a car and now are going to buy a new car
                U_a_i_t = base_utility - self.beta * (vehicle.price - self.vehicle.price / (1 + self.mu)) - self.gamma * vehicle.emissions
            else:
                raise ValueError("Invalid scenario specified. Owns second hand car")
        else:#you dont own a car
            if scenario in ["public_transport", "second_hand"]:#dont own car (currently choose public tranport or second hand car) and will now choose public tranport or second hand car
                U_a_i_t = base_utility - self.beta * vehicle.price
            elif scenario == "new_car":#dont own car (currently choose public tranport or second hand car), will buy new car
                U_a_i_t = base_utility - self.beta * vehicle.price - self.gamma * vehicle.emissions
            else:
                raise ValueError("Invalid scenario specified. No car is owned")

        return U_a_i_t, d_i_t

    def decide_vehicle(self, vehicles_available):
            # Calculate utility and distance for the current vehicle once, avoiding redundant calls
            current_utility, d_i_t_current_vehicle = self.calculate_utility(self.vehicle)
            
            # List comprehensions for utilities and distances
            utilities_list = [current_utility]
            vehicles_distance = [d_i_t_current_vehicle]

            for vehicle in vehicles_available:
                util_vehicle, d_i_t_vehicle = self.calculate_utility(vehicle)
                utilities_list.append(util_vehicle)
                vehicles_distance.append(d_i_t_vehicle)

            # Using NumPy array for utility and distance calculations for potential speedup
            utilities = np.array(utilities_list)

            # Optimizing the probability calculation
            utilities_kappa = np.power(utilities, self.kappa)
            probability_choose = utilities_kappa / utilities_kappa.sum()

            all_vehicles = [self.vehicle] + vehicles_available
            choice_index = np.random.choice(range(len(all_vehicles)), p=probability_choose)

            return all_vehicles[choice_index], vehicles_distance[choice_index], utilities[choice_index]

    def calc_emissions(self,vehicle_chosen, driven_distance):

        driving_emissions =  driven_distance*(vehicle_chosen.Eff_omega_a_t**-1)*vehicle_chosen.e_z_t 
        if vehicle_chosen.scenario == "new_car":  
            production_emissions = vehicle_chosen.emissions
        else:
            production_emissions = 0
        
        return driving_emissions, production_emissions

    def set_up_time_series_vehicle_user(self):
        self.history_emissions_driving = []
        self.history_emissions_production = []
        self.history_utility = []
        self.history_distance_driven = []

    def save_timeseries_data_vehicle_user(self):
        self.history_emissions_driving.append(self.driving_emissions)
        self.history_emissions_production.append(self.production_emissions)
        self.history_utility.append(self.utility)
        self.history_distance_driven.append(self.driven_distance)

    def next_step(self, vehicles_available):
        """
        Decide on the next vehicle for the user in the next time step.

        Parameters:
        vehicles_available (list): List of available vehicles for consideration.
        """
        self.t_vehicle_user += 1

        vehicle_chosen, self.driven_distance, self.utility = self.decide_vehicle(vehicles_available)
        self.current_vehicle_type = vehicle_chosen.transportType
        self.driving_emissions, self.production_emissions = self.calc_emissions(vehicle_chosen, self.driven_distance)
        
        #if self.save_timeseries_data_state and (self.t_vehicle_user % self.compression_factor_state == 0):
        #    self.save_timeseries_data_vehicle_user()

        return vehicle_chosen, self.driving_emissions, self.production_emissions, self.utility, self.driven_distance
