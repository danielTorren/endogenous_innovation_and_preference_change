import numpy as np

"""

At the moment this cannot deal with public transport, i think this should be the last modelling step to add that in
"""
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
        self.nu = parameters_vehicle_user["nu"]
        self.vehicles_available = parameters_vehicle_user["vehicles_available"]
        self.EV_bool = parameters_vehicle_user["EV_bool"]
        self.kappa = parameters_vehicle_user["kappa"]
        self.alpha = parameters_vehicle_user["alpha"]
        self.r = parameters_vehicle_user["r"]
        self.eta = parameters_vehicle_user["eta"]
        self.mu = parameters_vehicle_user["mu"]

        self.vehicle = self.decide_purchase_init(self.vehicles_available)  # Initial vehicle decision
        self.current_vehicle_type = self.vehicle.transportType

        if self.save_timeseries_data_state:
            self.set_up_time_series_vehicle_user()


    ###############################################################################################################################
    #INIT

    def decide_purchase_init(self, vehicles_available):
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
        commuting_util = self.commuting_utility(vehicle, d_i_t, z=2)  # Example z value (should be scenario-specific)

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
        if scenario == ("public_optional" or "private_unassigned"):#Cases 3 and 6, choosing PUBLIC TRANSPORT or second hand car without owning a second hand car due to public tranport or no car
            U_a_i_t = base_utility - self.beta * vehicle.price
        elif scenario == "private_emissions":#CASE 1, buyign new without owning a second hand car
            U_a_i_t = base_utility - self.beta * vehicle.price - self.gamma * vehicle.emissions
        else:
            raise ValueError("Invalid scenario specified. No car is owned")

        return U_a_i_t

    ###################################################################################################################################
    def commuting_utility(self, vehicle, d_i_t, z):
        """
        Calculate the commuting utility based on different conditions.

        Parameters:
        vehicle (Vehicle): The vehicle being considered for commuting.
        d_i_t (float): Distance traveled during a time-step.
        z (float): Condition parameter to differentiate public and private.

        Returns:
        float: The calculated commuting utility u_{a,i,t}.
        """
        Q_a_t = vehicle.Q_a_t
        delta_z = vehicle.delta_z
        L_a_t = vehicle.L_a_t
        omega_a_t = vehicle.omega_a_t
        c_z_t = vehicle.c_z_t
        e_z_t = vehicle.e_z_t
        nu_z_i_t = vehicle.nu_z_i_t

        # Calculate commuting utility based on conditions for z
        if z > 1:
            # If z' > 1, include all cost components
            cost_component = self.beta * (1 / omega_a_t) * c_z_t + self.gamma * (1 / omega_a_t) * e_z_t + self.eta * nu_z_i_t
            utility = Q_a_t * (1 - delta_z) ** L_a_t * (d_i_t ** self.alpha) - d_i_t * cost_component
        else:
            # If z' <= 1, include only the eta * nu component
            utility = Q_a_t * (1 - delta_z) ** L_a_t * (d_i_t ** self.alpha) - d_i_t * (self.eta * nu_z_i_t)

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
        numerator = self.alpha * vehicle.Q_a_t * (1 - vehicle.delta_z) ** vehicle.L_a_t
        denominator = (self.beta * vehicle.omega_a_t ** -1 * vehicle.c_z_t +
                       self.gamma * vehicle.omega_a_t ** -1 * vehicle.e_z_t +
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
        commuting_util = self.commuting_utility(vehicle, d_i_t, z=2)  # Example z value (should be scenario-specific)

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
        if self.vehicle.transportType > 0:#YOU OWN A CAR
            if scenario == "current_car":#CASE 7
                U_a_i_t = base_utility
            elif scenario == ("public_optional" or "private_unassigned"):# CASE 4 and 5, PUBLIC TRANSPORT or second hand car, whilst owning a second hand car!
                U_a_i_t = base_utility - self.beta * (vehicle.price -  self.vehicle.price/(1+self.mu))
            elif scenario == "private_emissions":#CASE 2, you buy a new car and you own one
                U_a_i_t = base_utility - self.beta * (vehicle.price - self.vehicle.price/(1+self.mu)) - self.gamma * vehicle.emissions
            else:
                raise ValueError("Invalid scenario specified. Owns second hand car")
        else:#you dont own a car!
            if scenario == ("public_optional" or "private_unassigned"):#Cases 3 and 6, choosing PUBLIC TRANSPORT or second hand car without owning a second hand car due to public tranport or no car
                U_a_i_t = base_utility - self.beta * vehicle.price
            elif scenario == "private_emissions":#CASE 1, buyign new without owning a second hand car
                U_a_i_t = base_utility - self.beta * vehicle.price - self.gamma * vehicle.emissions
            else:
                raise ValueError("Invalid scenario specified. No car is owned")

        return U_a_i_t, d_i_t

    def decide_purchase(self, vehicles_available):
        """
        Decide whether to purchase a new vehicle or keep the current one.

        Parameters:
        vehicles_available (list): List of available vehicles to consider for purchase.

        Returns:
        Vehicle: The chosen vehicle.
        """
        # Calculate current vehicle utility
        current_utility, d_i_t_current_vehicle = self.calculate_utility(self.vehicle)

        # Calculate utility for each available vehicle
        utilities_list = [current_utility]
        vehicles_distance = [d_i_t_current_vehicle]

        for vehicle in vehicles_available:
            util_vehicle, d_i_t_vehicle =  self.calculate_utility(vehicle)
            utilities_list.append(util_vehicle)  # Example scenario for new vehicles
            vehicles_distance.append(d_i_t_vehicle)
 

        # Combine current vehicle utility with new vehicles
        utilities = np.array(utilities_list)

        # Calculate the probabilities of choosing each vehicle
        sum_utilities = np.sum(utilities ** self.kappa)
        probability_choose = (utilities ** self.kappa) / sum_utilities

        # Include current vehicle in the list of all vehicles
        all_vehicles =  [self.vehicle] + vehicles_available#NEED TO HAVE THIS ORDER FOR INDEX TO MATCH WITH THE OTHERS

        # Choose a vehicle based on the calculated probabilities
        choice_index = np.random.choice(range(len(all_vehicles)), size=1, p=probability_choose)[0]
        chosen_vehicle = all_vehicles[choice_index]
        driven_distance =  vehicles_distance[choice_index]
        utility = utilities[choice_index]

        return chosen_vehicle, driven_distance, utility# Can be the same as the previous vehicle

    def calc_emissions(self,vehicle_chosen, driven_distance):

        driving_emissions =  driven_distance*(vehicle_chosen.omega_a_t**-1)*vehicle_chosen.e_z_t 
        if vehicle_chosen.scenario == "private_emissions":  
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

        vehicle_chosen, self.driven_distance, self.utility = self.decide_purchase(vehicles_available)
        self.current_vehicle_type = vehicle_chosen.transportType
        self.driving_emissions, self.production_emissions = self.calc_emissions(vehicle_chosen, self.driven_distance)
        
        if self.save_timeseries_data_state and (self.t_vehicle_user % self.compression_factor_state == 0):
            self.save_timeseries_data_vehicle_user()

        return vehicle_chosen, self.driving_emissions, self.production_emissions, self.utility, self.driven_distance