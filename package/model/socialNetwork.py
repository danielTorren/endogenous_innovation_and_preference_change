
# imports
import numpy as np
import networkx as nx
import numpy.typing as npt
import scipy.sparse as sp
import numpy as np
from package.model.personalCar import PersonalCar
from package.model.VehicleUser import VehicleUser
from package.model.carModel import CarModel

class Social_Network:
    def __init__(self, parameters_social_network: dict, parameters_vehicle_user: dict):
        """
        Constructs all the necessary attributes for the Social Network object.
        """
        self.t_social_network = 0
        
        # Initialize parameters
        self.parameters_vehicle_user = parameters_vehicle_user
        self.init_initial_state(parameters_social_network)
        self.init_network_settings(parameters_social_network)
        self.init_preference_distribution(parameters_social_network)
        self.set_init_vehicle_options(parameters_social_network)
        
        self.createVehicleUsers()
        
        # Create network and calculate initial emissions
        self.adjacency_matrix, self.network = self.create_network()
        self.network_density = nx.density(self.network)
        
        #Assume nobody adopts EV at the start, THIS MAY BE AN ISSUE
        self.consider_ev_vec = np.zeros(self.num_individuals)
        #individual choose their vehicle in the zeroth step
        vehicles_chosen_list = self.update_VehicleUsers()

        self.chi_vec = np.array([user.chi for user in self.vehicleUsers_list])  # Innovation thresholds
        self.vehicle_type_vec = np.array([user.current_vehicle_type for user in self.vehicleUsers_list])
        self.consider_ev_vec, self.ev_adoption_vec = self.calculate_ev_adoption(ev_type=3)#BASED ON CONSUMPTION PREVIOUS TIME STEP

        self.extract_VehicleUser_data()

    ###############################################################################################################################################################
    ###############################################################################################################################################################
    #MODEL SETUP

    def init_initial_state(self, parameters_social_network):
        self.num_individuals = int(round(parameters_social_network["num_individuals"]))
        self.id_generator = parameters_social_network["IDGenerator_firms"]
        self.second_hand_merchant = parameters_social_network["second_hand_merchant"]
        #self.save_timeseries_data_state = parameters_social_network["save_timeseries_data_state"]
        #self.compression_factor_state = parameters_social_network["compression_factor_state"]

    def init_network_settings(self, parameters_social_network):
        self.network_structure_seed = parameters_social_network["network_structure_seed"]
        self.network_density_input = parameters_social_network["network_density"]
        self.K_social_network = int(round((self.num_individuals - 1) * self.network_density_input))  # Calculate number of links
        self.prob_rewire = parameters_social_network["prob_rewire"]

    def init_preference_distribution(self, parameters_social_network):
        #GAMMA
        self.a_environment = parameters_social_network["a_environment"]
        self.b_environment = parameters_social_network["b_environment"]
        np.random.seed(parameters_social_network["init_vals_environmental_seed"])  # Initialize random seed
        self.environmental_awareness_vec = np.random.beta(self.a_environment, self.b_environment, size=self.num_individuals)

        #CHI
        self.a_innovativeness = parameters_social_network["a_innovativeness"]
        self.b_innovativeness = parameters_social_network["b_innovativeness"]
        np.random.seed(parameters_social_network["init_vals_innovative_seed"])  # Initialize random seed
        innovativeness_vec_init_unrounded = np.random.beta(self.a_innovativeness, self.b_innovativeness, size=self.num_individuals)
        self.innovativeness_vec = np.round(innovativeness_vec_init_unrounded, 1)
        self.ev_adoption_state_vec = np.zeros(self.num_individuals)

        #BETA
        self.a_price = parameters_social_network["a_price"]
        self.b_price = parameters_social_network["b_price"]
        np.random.seed(parameters_social_network["init_vals_price_seed"])  # Initialize random seed
        self.price_sensitivity_vec = np.random.beta(self.a_price, self.b_price, size=self.num_individuals)


        #origin
        self.origin_vec = np.asarray([0]*(int(round(self.num_individuals/2))) + [0]*(int(round(self.num_individuals/2))))#THIS IS A PLACE HOLDER NEED TO DISCUSS THE DISTRIBUTION OF INDIVIDUALS

        #d min
        np.random.seed(parameters_social_network["d_min_seed"]) 
        self.d_i_min_vec = np.random.uniform(size = self.num_individuals)


    def extract_VehicleUser_data(self):
        # Extract user attributes into vecs for efficient processing
        self.chi_vec = np.array([user.chi for user in self.vehicleUsers_list])  # Innovation thresholds
        self.gamma_vec = np.array([user.gamma for user in self.vehicleUsers_list])  # Environmental concern
        self.beta_vec = np.array([user.beta for user in self.vehicleUsers_list])  # Cost sensitivity
        self.vehicle_type_vec = np.array([user.current_vehicle_type for user in self.vehicleUsers_list])  # Current vehicle types
        self.origin_vec = np.array([user.origin for user in self.vehicleUsers_list])  # Urban/rural origin
        self.ev_adoption_state_vec = np.array([user.origin for user in self.vehicleUsers_list])#EV ADOPTION STATE
        
    def set_init_vehicle_options(self, parameters_social_network):
        self.cars_on_sale_all_firms = parameters_social_network["init_vehicle_options"]

    def createVehicleUsers(self):
        self.vehicleUsers_list = []
        self.parameters_vehicle_user["vehicles_available"] = self.cars_on_sale_all_firms
        for i in range(self.num_individuals):
            self.vehicleUsers_list.append(VehicleUser(user_id = i, chi = self.innovativeness_vec[i], gamma = self.environmental_awareness_vec[i], beta = self.price_sensitivity_vec[i], origin = self.origin_vec[i], d_i_min = self.d_i_min_vec[i], parameters_vehicle_user=self.parameters_vehicle_user))
             
    def normalize_vec_sum(self, vec):
        return vec/sum(vec)

    def create_network(self) -> tuple[npt.NDArray, npt.NDArray, nx.Graph]:
        """
        Create watts-strogatz small world graph using Networkx library

        parameters_social_network
        ----------
        None

        Returns
        -------
        weighting_matrix: npt.NDArray[bool]
            adjacency matrix, array giving social network structure where 1 represents a connection between agents and 0 no connection. It is symetric about the diagonal
        norm_weighting_matrix: npt.NDArray[float]
            an NxN array how how much each agent values the opinion of their neighbour. Note that is it not symetric and agent i doesn"t need to value the
            opinion of agent j as much as j does i"s opinion
        ws: nx.Graph
            a networkx watts strogatz small world graph
        """

        network = nx.watts_strogatz_graph(n=self.num_individuals, k=self.K_social_network, p=self.prob_rewire, seed=self.network_structure_seed)#FIX THE NETWORK STRUCTURE

        adjacency_matrix = nx.to_numpy_array(network)
        self.sparse_adjacency_matrix = sp.csr_matrix(adjacency_matrix)
        # Get the non-zero indices of the adjacency matrix
        self.row_indices_sparse, self.col_indices_sparse = self.sparse_adjacency_matrix.nonzero()

        self.network_density = nx.density(network)
        # Calculate the total number of neighbors for each user
        self.total_neighbors = np.array(self.sparse_adjacency_matrix.sum(axis=1)).flatten()
        return adjacency_matrix, network

    ###############################################################################################################################################################
    
    #DYNAMIC COMPONENTS
    """
    def calculate_ev_adoption(self, ev_type=3):
        # Create a binary matrix where 1 indicates neighbors using EVs
        ev_adoption_vec = np.where(self.vehicle_type_vec == ev_type, 1, 0)
        
        # Calculate the proportion of neighbors with EVs for each user (matrix multiplication)
        ev_neighbors = np.dot(self.adjacency_matrix, ev_adoption_vec)
        total_neighbors = np.sum(self.adjacency_matrix, axis=1)
        #proportion_ev_neighbors = np.divide(ev_neighbors, total_neighbors, where=total_neighbors != 0)
        proportion_ev_neighbors = np.round(np.divide(ev_neighbors, total_neighbors, where=total_neighbors != 0),2)
        # Determine whether each user considers buying an EV (1 if proportion of EVs in neighborhood ≥ chi)
        consider_ev_vec = (proportion_ev_neighbors >= self.chi_vec).astype(int)
        
        return consider_ev_vec, ev_adoption_vec
    """
    def calculate_ev_adoption(self, ev_type=3):
        """
        Calculate the proportion of neighbors using EVs for each user, 
        and determine EV adoption consideration.
        """
        
        self.vehicle_type_vec = np.array([user.current_vehicle_type for user in self.vehicleUsers_list])  # Current vehicle types

        # Create a binary vector indicating EV users
        ev_adoption_vec = (self.vehicle_type_vec == ev_type).astype(int)

        # Calculate the number of EV-adopting neighbors using sparse matrix multiplication
        ev_neighbors = self.sparse_adjacency_matrix.dot(ev_adoption_vec)


        # Calculate the proportion of neighbors with EVs
        proportion_ev_neighbors = np.divide(ev_neighbors, self.total_neighbors, where=self.total_neighbors != 0)

        # Determine whether each user considers buying an EV based on the chi threshold
        consider_ev_vec = (proportion_ev_neighbors >= self.chi_vec).astype(int)

        return consider_ev_vec, ev_adoption_vec

    def update_VehicleUsers(self):
        vehicle_chosen_list = []

        #variable to track
        self.total_driving_emissions = 0
        self.total_production_emissions = 0
        self.total_utility = 0
        self.total_distance_travelled = 0
        self.urban_public_transport_users = 0
        self.rural_public_transport_users = 0
        self.ICE_users = 0 
        self.EV_users = 0
        self.second_hand_users = 0
        

        for i, user in enumerate(self.vehicleUsers_list):
            # For each user, determine the available vehicles based on their environment.
            if self.consider_ev_vec[i]:
                available_vehicles = self.cars_on_sale_all_firms
            else:
                # Filter out vehicles the user can't use (e.g., transporttype != 3)
                available_vehicles = [vehicle for vehicle in self.cars_on_sale_all_firms if vehicle.transportType != 3]

            # Get the user's decision on the next vehicle
            #vehicle_chosen, self.driving_emissions, self.production_emissions, self.driven_distance
            vehicle_chosen, driving_emissions, production_emissions, utility, distance_driven = user.next_step(available_vehicles)

            #ADD TOTAL EMISSIONS
            self.total_driving_emissions += driving_emissions
            self.total_production_emissions += production_emissions
            self.total_utility += utility 
            self.total_distance_travelled += distance_driven

            if isinstance(vehicle_chosen, PersonalCar):
                self.second_hand_users +=1
            
            if vehicle_chosen.transportType == 0:
                self.urban_public_transport_users+=1
            elif vehicle_chosen.transportType == 1:
                self.rural_public_transport_users += 1
            elif vehicle_chosen.transportType == 2:
                self.ICE_users += 1
            else:
                self.EV_users += 1

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
            
            #NOW DEAL WITH THE CONSEQUENCES


            if user.user_id != vehicle_chosen.owner_id:#ITS NOT YOUR CURRENT CAR
                #MOVE CAR THAT YOU OWN AWAY TO SECOND HAND MAN
                if user.vehicle is PersonalCar:
                    user.vehicle.owner_id = self.second_hand_merchant.id#transfer ownsership of CURRENT CAR to the second hand shop!
                    self.second_hand_merchant.add_to_stock(user.vehicle)#add the car to the stock of second hand cars
                    user.vehicle = None#For an instance you have nothing

                #SAY YOU BUY A SECOND HAND CAR
                if isinstance(vehicle_chosen, PersonalCar):# THESE ARE SECOND HAND CARS OR THE ONE YOU PERSONALLY OWN
                    vehicle_chosen.owner_id = user.user_id#transfer ownership, hand keys to the new person!
                    vehicle_chosen.scenario = "current_car"#the state of the car becomes yours!
                    user.vehicle = vehicle_chosen
                    self.second_hand_merchant.remove_car(vehicle_chosen)
                    self.cars_on_sale_all_firms.remove(vehicle_chosen)#remove from currenly available in this timestep
                elif isinstance(vehicle_chosen, CarModel):#BRAND NEW CAR #make the theoretical model into an acutal car 
                    personalCar_id = self.id_generator.get_new_id()
                    user.vehicle = PersonalCar(personalCar_id,vehicle_chosen.firm, user.user_id, vehicle_chosen.component_string, vehicle_chosen.parameters, vehicle_chosen.attributes_fitness, vehicle_chosen.price)
                else:#PUBLIC TRANSPORT
                    user.vehicle = vehicle_chosen

            if isinstance(vehicle_chosen, PersonalCar):
                user.vehicle.update_timer()#ADD AGE


            
            vehicle_chosen_list.append(vehicle_chosen)

        return vehicle_chosen_list
    
    ####################################################################################################################   
    def set_up_time_series_social_network(self):
        self.history_driving_emissions = []
        self.history_production_emissions = []
        self.history_total_utility = []
        self.history_total_distance_driven = []
        self.history_ev_adoption_rate = []
        self.history_urban_public_transport_users = []
        self.history_rural_public_transport_users = []
        self.history_consider_ev_rate = []
        self.history_ICE_users = []
        self.history_EV_users = []
        self.history_second_hand_users = []
        # New history attributes for vehicle attributes
        self.history_quality = []
        self.history_efficiency = []
        self.history_production_cost = []

    def save_timeseries_data_social_network(self):

        self.history_driving_emissions.append(self.total_driving_emissions)
        self.history_production_emissions.append(self.total_production_emissions)
        self.history_total_utility.append(self.total_utility)
        self.history_total_distance_driven.append(self.total_distance_travelled)
        self.history_ev_adoption_rate.append(np.mean(self.ev_adoption_vec))
        self.history_consider_ev_rate.append(np.mean(self.consider_ev_vec))
        self.history_urban_public_transport_users.append(self.urban_public_transport_users)
        self.history_rural_public_transport_users.append(self.rural_public_transport_users)
        self.history_ICE_users.append(self.ICE_users)
        self.history_EV_users.append(self.EV_users)
        self.history_second_hand_users.append(self.second_hand_users)

        # New history saving for vehicle attributes
        quality_vals = [vehicle.Q_a_t for vehicle in self.vehicles_chosen_list]
        efficiency_vals = [vehicle.omega_a_t for vehicle in self.vehicles_chosen_list]
        production_cost_vals = [vehicle.c_z_t for vehicle in self.vehicles_chosen_list]

        self.history_quality.append(quality_vals)
        self.history_efficiency.append(efficiency_vals)
        self.history_production_cost.append(production_cost_vals)

    def next_step(self, carbon_price, cars_on_sale_all_firms):
        """
        Push the simulation forwards one time step. First advance time, then update individuals with data from previous timestep
        then produce new data and finally save it.

        parameters_social_network
        ----------
        None

        Returns
        -------
        None
        """

        self.t_social_network +=1

        self.carbon_price = carbon_price

        #update new tech and prices
        self.cars_on_sale_all_firms = cars_on_sale_all_firms

        self.consider_ev_vec, self.ev_adoption_vec = self.calculate_ev_adoption(ev_type=3)#BASED ON CONSUMPTION PREVIOUS TIME STEP
        self.vehicles_chosen_list = self.update_VehicleUsers()
        #print("Adoption rate", np.mean(self.ev_adoption_vec),np.mean(self.consider_ev_vec) )
        #DO FIRMS CONSIDER WHO WILL PURCHASE AN EV OR WHO HAS PURCHASED AN EV
        #return self.ev_adoption_state_vec, vehicles_chosen_list
        return self.consider_ev_vec, self.vehicles_chosen_list
