import numpy as np
import networkx as nx
from package.model.personalCar import PersonalCar
from package.model.carModel import CarModel

class VehicleNetworkModel:
    def __init__(self, parameters_social_network, parameters_vehicle_user):
        # Initialize time step
        self.t_model = 0
        # Initialize social and vehicle parameters
        self.parameters_vehicle_user = parameters_vehicle_user
        self.init_social_network(parameters_social_network)
        
        # Define initial vehicle and network configurations
        self.initialize_vehicle_data(parameters_social_network)
        self.initialize_user_data(parameters_social_network)
        self.initialize_vehicle_users_data(parameters_vehicle_user)

        self.update_users()

        
    def init_social_network(self, params):
        # Set user count and network configuration
        self.num_individuals = int(round(params["num_individuals"]))
        self.network_density = params["network_density"]
        self.prob_rewire = params["prob_rewire"]
        
        # Generate adjacency matrix from Watts-Strogatz network
        self.network = nx.watts_strogatz_graph(self.num_individuals, int((self.num_individuals - 1) * self.network_density), self.prob_rewire)
        self.adjacency_matrix = nx.to_numpy_array(self.network)
        
    def initialize_vehicle_data(self, params):
        # Set initial vehicle options available to users
        self.cars_on_sale_all_firms = params["init_vehicle_options"]
        self.second_hand_merchant = params["second_hand_merchant"]
    
    def initialize_vehicle_users_data(self,parameters_vehicle_user):
        self.nu = parameters_vehicle_user["nu"]
        #self.EV_bool = parameters_vehicle_user["EV_bool"]
        self.kappa = parameters_vehicle_user["kappa"]
        self.alpha = parameters_vehicle_user["alpha"]
        self.r = parameters_vehicle_user["r"]
        self.eta = parameters_vehicle_user["eta"]
        self.mu = parameters_vehicle_user["mu"]
    
    def initialize_user_data(self, params):
        # Sample user attributes from distributions for environmental awareness, innovativeness, and price sensitivity
        np.random.seed(params["init_vals_innovative_seed"])
        self.chi_vec = np.random.beta(params["a_innovativeness"], params["b_innovativeness"], self.num_individuals)
        np.random.seed(params["init_vals_environmental_seed"])  # Initialize random seed
        self.gamma_vec = np.random.beta(params["a_environment"], params["b_environment"], self.num_individuals)
        np.random.seed(params["init_vals_price_seed"])  # Initialize random seed
        self.beta_vec = np.random.beta(params["a_price"], params["b_price"], self.num_individuals)
        
        self.origin_vec = np.asarray([0]*(int(round(self.num_individuals/2))) + [0]*(int(round(self.num_individuals/2))))#THIS IS A PLACE HOLDER NEED TO DISCUSS THE DISTRIBUTION OF INDIVIDUALS
        self.d_i_min_vec = np.random.uniform(size = self.num_individuals)#d min
        
        # Initialize user states and vehicle types
        self.vehicle_type_vec = np.zeros(self.num_individuals, dtype=int)
        self.ev_adoption_state = np.zeros(self.num_individuals)
        self.consider_ev_vec = np.zeros(self.num_individuals, dtype=int)
        
    def calculate_ev_adoption(self, ev_type=3):
        # Calculate proportion of neighbors using EVs using matrix multiplication with adjacency matrix
        ev_adoption_vec = (self.vehicle_type_vec == ev_type).astype(int)
        neighbor_ev_counts = self.adjacency_matrix @ ev_adoption_vec#@is matrix multiplication
        total_neighbors = np.sum(self.adjacency_matrix, axis=1)
        ev_proportion = neighbor_ev_counts / np.maximum(total_neighbors, 1)
        
        # Update EV consideration based on chi threshold
        self.consider_ev_vec = (ev_proportion >= self.chi_vec).astype(int)
    
    def calculate_utilities(self, vehicles_available):
        # Placeholder for utility and distance matrices
        utilities = np.zeros((self.num_individuals, len(vehicles_available)))
        distances = np.zeros((self.num_individuals, len(vehicles_available)))
        
        for i, vehicle in enumerate(vehicles_available):
            # Calculate utility and distances in a vecized way using user preferences
            distance = self.calculate_optimal_distance(vehicle)
            utilities[:, i] = self.gamma_vec * vehicle.emissions - self.beta_vec * vehicle.price + distance
            distances[:, i] = distance
            
        return utilities, distances
    
    """
    def update_users(self):
        # Calculate utility for all users based on their current options
        utilities, distances = self.calculate_utilities(self.cars_on_sale_all_firms)
        
        # Custom probability calculation using the kappa exponent
        utilities_kappa = np.power(utilities, self.kappa)
        probabilities = utilities_kappa / utilities_kappa.sum(axis=1, keepdims=True)
        
        # Cumulative sum along each row to form cumulative distribution for each individual
        cumulative_probabilities = np.cumsum(probabilities, axis=1)
        
        # Generate random values for each individual
        random_values = np.random.rand(self.num_individuals, 1)
        
        # Determine chosen vehicle index for each user using cumulative distribution
        chosen_indices = (cumulative_probabilities > random_values).argmax(axis=1)
        
        # Assign chosen vehicles
        self.vehicles_chosen = [self.cars_on_sale_all_firms[i] for i in chosen_indices]
        
        # Update states based on chosen vehicles
        self.vehicle_type_vec = np.array([vehicle.transportType for vehicle in self.vehicles_chosen])
        self.ev_adoption_state = (self.vehicle_type_vec == 3).astype(int)
    """

    def update_users(self):
        vehicle_chosen_list = []

        # Reset tracking variables for emissions, utility, and distance
        self.total_driving_emissions = 0
        self.total_production_emissions = 0
        self.total_utility = 0
        self.total_distance_travelled = 0

        # Work with a mutable list of available vehicles
        available_vehicles = list(self.cars_on_sale_all_firms)  # Copy of the initial vehicles on sale

        # List to track each user's current vehicle (initialized as None if no vehicle is owned at the start)
        if not hasattr(self, "user_current_vehicles"):
            self.user_current_vehicles = [None] * self.num_individuals

        for i in range(self.num_individuals):
            # Determine available vehicles based on EV consideration
            if self.consider_ev_vec[i]:
                user_available_vehicles = available_vehicles
            else:
                # Filter out EVs if the user isn’t considering them (e.g., transport type != 3)
                user_available_vehicles = [vehicle for vehicle in available_vehicles if vehicle.transportType != 3]

            # Calculate utilities and choose vehicle probabilistically for the current user
            utilities, distances = self.calculate_utilities(user_available_vehicles)
            utilities_kappa = np.power(utilities[i], self.kappa)
            probabilities = utilities_kappa / utilities_kappa.sum()

            # Select vehicle based on probabilistic choice
            chosen_index = np.random.choice(len(user_available_vehicles), p=probabilities)
            vehicle_chosen = user_available_vehicles[chosen_index]

            # Calculate emissions and other attributes for the chosen vehicle
            driving_emissions, production_emissions, utility, distance_driven = self.calculate_emissions_and_utility(vehicle_chosen, distances[i])

            # Update totals for emissions, utility, and distance
            self.total_driving_emissions += driving_emissions
            self.total_production_emissions += production_emissions
            self.total_utility += utility
            self.total_distance_travelled += distance_driven

            # Now handle ownership transfer if needed
            current_vehicle = self.user_current_vehicles[i]
            if current_vehicle is not None and current_vehicle != vehicle_chosen:
                # Transfer the current vehicle to the second-hand market
                current_vehicle.owner_id = self.second_hand_merchant.id
                self.second_hand_merchant.add_to_stock(current_vehicle)
                self.user_current_vehicles[i] = None  # User temporarily has no vehicle

            # Transfer ownership or create a new PersonalCar instance as needed
            if isinstance(vehicle_chosen, PersonalCar):  # Second-hand or user's previous car
                vehicle_chosen.owner_id = i  # Assign the vehicle to the current user
                vehicle_chosen.scenario = "current_car"
                self.user_current_vehicles[i] = vehicle_chosen
                self.second_hand_merchant.remove_car(vehicle_chosen)
                available_vehicles.remove(vehicle_chosen)  # Remove from available options
            elif isinstance(vehicle_chosen, CarModel):  # New car
                personalCar_id = self.id_generator.get_new_id()
                self.user_current_vehicles[i] = PersonalCar(personalCar_id, vehicle_chosen.firm, i, vehicle_chosen.component_string, vehicle_chosen.parameters, vehicle_chosen.attributes_fitness, vehicle_chosen.price)
            else:  # Public transport or other non-car options
                self.user_current_vehicles[i] = vehicle_chosen

            vehicle_chosen_list.append(vehicle_chosen)

        # Update states based on chosen vehicles
        self.vehicle_type_vec_a = np.array([vehicle.current_vehicle_type for vehicle in self.user_current_vehicles])  # Current vehicle types
        self.vehicle_type_vec = np.array([vehicle.transportType if isinstance(vehicle, (PersonalCar, CarModel)) else 0 for vehicle in self.user_current_vehicles])

        print("self.vehicle_type_vec_a", self.vehicle_type_vec_a)
        print("self.vehicle_type_vec", self.vehicle_type_vec)
        quit()
        self.ev_adoption_state = (self.vehicle_type_vec == 3).astype(int)

        return vehicle_chosen_list

    def calculate_emissions_and_utility(self, vehicle_chosen, distance):
        """
        Calculate emissions, utility, and distance for a chosen vehicle.
        """
        driving_emissions = distance * (vehicle_chosen.omega_a_t ** -1) * vehicle_chosen.e_z_t
        production_emissions = vehicle_chosen.emissions if vehicle_chosen.scenario == "private_emissions" else 0
        utility = (self.gamma_vec * vehicle_chosen.emissions - self.beta_vec * vehicle_chosen.price + distance).sum()

        return driving_emissions, production_emissions, utility, distance


    
    def calculate_optimal_distance(self, vehicle):
        # Vectorized distance calculation for all individuals
        Q_a_t = vehicle.Q_a_t
        delta_z = vehicle.delta_z
        L_a_t = vehicle.L_a_t
        c_z_t = vehicle.c_z_t
        e_z_t = vehicle.e_z_t
        omega_a_t = vehicle.omega_a_t

        # Numerator and denominator as per the utility model
        numerator = self.alpha * Q_a_t * ((1 - delta_z) ** L_a_t)
        denominator = (
            self.beta_vec * omega_a_t ** -1 * c_z_t + 
            self.gamma_vec * omega_a_t ** -1 * e_z_t + 
            self.eta * vehicle.nu_z_i_t
        )
        
        # Check if any value in the denominator is zero
        if np.any(denominator == 0):
            raise ValueError("The denominator contains zero values; adjust the parameters to avoid division by zero.")


        # Calculate the optimal distance based on utility maximization
        optimal_distance = (numerator / denominator) ** (1 / (1 - self.alpha))
        
        # Ensure that the distance respects the minimum threshold for each user
        final_distance = np.maximum(self.d_i_min_vec, optimal_distance)
        
        return final_distance

    def next_step(self, carbon_price, cars_on_sale_all_firms):
        self.t_model += 1
        self.cars_on_sale_all_firms = cars_on_sale_all_firms
        self.calculate_ev_adoption()
        self.update_users()

        return self.consider_ev_vec, self.vehicles_chosen
