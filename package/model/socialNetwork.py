
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
    def __init__(self, parameters_social_network: dict):
        """
        Constructs all the necessary attributes for the Social Network object.
        """
        self.t_social_network = 0
        
        # Initialize parameters
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

        self.extract_VehicleUser_data()

    ###############################################################################################################################################################
    ###############################################################################################################################################################
    #MODEL SETUP

    def init_initial_state(self, parameters_social_network):
        self.num_individuals = int(round(parameters_social_network["num_individuals"]))
        self.time_sensitivity = parameters_social_network["time_sensitivity"]
        self.id_generator = parameters_social_network["IDGenerator_firms"]
        self.second_hand_merchant = parameters_social_network["second_hand_merchant"]

    def init_network_settings(self, parameters_social_network):
        self.network_structure_seed = parameters_social_network["network_structure_seed"]
        self.network_density_input = parameters_social_network["network_density"]
        self.K_social_network = int(round((self.num_individuals - 1) * self.network_density_input))  # Calculate number of links
        self.prob_rewire = parameters_social_network["prob_rewire"]

    def init_preference_distribution(self, parameters_social_network):
        self.a_preferences = parameters_social_network["a_preferences"]
        self.b_preferences = parameters_social_network["b_preferences"]
        np.random.seed(parameters_social_network["init_vals_environmental_seed"])  # Initialize random seed
        self.environmental_awareness_vec = np.random.beta(self.a_preferences, self.b_preferences, size=self.num_individuals)
        self.a_innovativeness = parameters_social_network["a_innovativeness"]
        self.b_innovativeness = parameters_social_network["b_innovativeness"]
        np.random.seed(parameters_social_network["init_vals_innovative_seed"])  # Initialize random seed
        self.innovativeness_vec_init = np.random.beta(self.a_innovativeness, self.b_innovativeness, size=self.num_individuals)
        self.ev_adoption_state_vec = np.zeros(self.num_individuals)
        self.a_price = parameters_social_network["a_price"]
        self.b_price = parameters_social_network["b_price"]
        np.random.seed(parameters_social_network["init_vals_price_seed"])  # Initialize random seed
        self.price_sensitivity_vec = np.random.beta(self.a_price, self.b_price, size=self.num_individuals)
        self.origin_vec = np.asarray([0]*(int(round(self.num_individuals/2))) + [0]*(int(round(self.num_individuals/2))))#THIS IS A PLACE HOLDER NEED TO DISCUSS THE DISTRIBUTION OF INDIVIDUALS

    def extract_VehicleUser_data(self):
        # Extract user attributes into vectors for efficient processing
        self.chi_vector = np.array([user.chi for user in self.vehicleUsers_list])  # Innovation thresholds
        self.gamma_vector = np.array([user.gamma for user in self.vehicleUsers_list])  # Environmental concern
        self.beta_vector = np.array([user.beta for user in self.vehicleUsers_list])  # Cost sensitivity
        self.vehicle_type_vector = np.array([user.current_vehicle_type for user in self.vehicleUsers_list])  # Current vehicle types
        self.origin_vector = np.array([user.origin for user in self.vehicleUsers_list])  # Urban/rural origin
        self.ev_adoption_state_vec = np.array([user.origin for user in self.vehicleUsers_list])#EV ADOPTION STATE
        
    def set_init_vehicle_options(self, parameters_social_network):
        self.cars_on_sale_all_firms = parameters_social_network["init_vehicle_options"]

    def createVehicleUsers(self):
        self.vehicleUsers_list = []
        for i in range(self.num_individuals):
            self.vehicleUsers_list.append(VehicleUser(id = i, chi = self.innovativeness_vec_init[i], gamma = self.environmental_awareness_vec[i], beta = self.price_sensitivity_vec[i], nu = self.time_sensitivity, origin = self.origin_vec[i], vehicles_available = self.cars_on_sale_all_firms, EV_bool = 0))
             
    def normalize_vector_sum(self, vec):
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

        adjacency_matrix = nx.to_numpy_array(self.network)
        self.sparse_adjacency_matrix = sp.csr_matrix(adjacency_matrix)
        # Get the non-zero indices of the adjacency matrix
        self.row_indices_sparse, self.col_indices_sparse = self.sparse_adjacency_matrix.nonzero()

        self.network_density = nx.density(self.network)
        return adjacency_matrix, network

    ###############################################################################################################################################################
    
    #DYNAMIC COMPONENTS
    def calculate_ev_adoption(self, ev_type=3):
        # Create a binary matrix where 1 indicates neighbors using EVs
        ev_adoption_vec = np.where(self.vehicle_type_vector == ev_type, 1, 0)
        
        # Calculate the proportion of neighbors with EVs for each user (matrix multiplication)
        ev_neighbors = np.dot(self.adjacency_matrix, ev_adoption_vec)
        total_neighbors = np.sum(self.adjacency_matrix, axis=1)
        proportion_ev_neighbors = np.divide(ev_neighbors, total_neighbors, where=total_neighbors != 0)

        # Determine whether each user considers buying an EV (1 if proportion of EVs in neighborhood ≥ chi)
        consider_ev_vec = (proportion_ev_neighbors >= self.chi_vector).astype(int)
        
        return consider_ev_vec, ev_adoption_vec

    def update_VehicleUsers(self):
        vehicle_chosen_list = []
        for i, user in enumerate(self.vehicleUsers_list):
            # For each user, determine the available vehicles based on their preferences.
            if self.consider_ev_vec[i]:
                available_vehicles = self.cars_on_sale_all_firms
            else:
                # Filter out vehicles the user can't use (e.g., transporttype != 3)
                available_vehicles = [vehicle for vehicle in self.cars_on_sale_all_firms if vehicle.transportType != 3]

            # Get the user's decision on the next vehicle
            vehicle_choosen = user.next_step(available_vehicles)

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


            if user.id != vehicle_choosen.owner_id:#ITS NOT YOUR CURRENT CAR
                #MOVE CAR THAT YOU OWN AWAY TO SECOND HAND MAN
                if user.vehicle is PersonalCar:
                    user.vehicle.owner_id = self.second_hand_merchant.id#transfer ownsership of CURRENT CAR to the second hand shop!
                    self.second_hand_merchant.add_to_stock(user.vehicle)#add the car to the stock of second hand cars
                    user.vehicle = None#For an instance you have nothing

                #SAY YOU BUY A SECOND HAND CAR
                if vehicle_choosen is PersonalCar:# THESE ARE SECOND HAND CARS OR THE ONE YOU PERSONALLY OWN
                    vehicle_choosen.owner_id = user.id#transfer ownership, hand keys to the new person!
                    vehicle_choosen.scenario = "current_car"#the state of the car becomes yours!
                    user.vehicle = vehicle_choosen
                    self.second_hand_merchant.remove_car(vehicle_choosen)
                    self.cars_on_sale_all_firms.remove(vehicle_choosen)#remove from currenly available in this timestep
                elif vehicle_choosen is CarModel:#BRAND NEW CAR #make the theoretical model into an acutal car 
                    personalCar_id = self.id_generator.get_new_id()
                    user.vehicle = PersonalCar(personalCar_id, vehicle_choosen.firm_id, user.user_id, vehicle_choosen.component_string, vehicle_choosen.parameters, vehicle_choosen.attributes_fitness, vehicle_choosen.price)
                else:#PUBLIC TRANSPORT
                    user.vehicle = vehicle_choosen
                
            vehicle_chosen_list.append(vehicle_choosen)

        return vehicle_chosen_list

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

        vehicles_chosen_list = self.update_VehicleUsers()

        return self.ev_adoption_state_vec, vehicles_chosen_list
