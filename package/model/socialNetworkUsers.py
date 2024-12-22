# imports
from math import e
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
        self.emissions_cumulative = 0
        self.emissions_flow_history = []
        self.history_prop_EV = []

        self.rebate = parameters_social_network["rebate"]
        self.used_rebate = parameters_social_network["used_rebate"]

        # Initialize parameters
        self.parameters_vehicle_user = parameters_vehicle_user
        self.init_initial_state(parameters_social_network)
        self.init_network_settings(parameters_social_network)
        self.init_preference_distribution(parameters_social_network)

        self.random_state_social_network = np.random.RandomState(parameters_social_network["social_network_seed"])

        self.alpha =  parameters_vehicle_user["alpha"]
        self.mu =  parameters_vehicle_user["mu"]
        self.r = parameters_vehicle_user["r"]
        self.kappa = parameters_vehicle_user["kappa"]
        #self.second_hand_car_max_consider = parameters_vehicle_user["second_hand_car_max_consider"]
        #self.new_car_max_consider = parameters_vehicle_user["new_car_max_consider"]

        # Generate a list of indices and shuffle them
        self.user_indices = np.arange(self.num_individuals)

        # Efficient user list creation with list comprehension
        self.vehicleUsers_list = [VehicleUser(user_id=i) for i in range(self.num_individuals)]

        # Create network and calculate initial emissions
        self.adjacency_matrix, self.network = self.create_network()
        self.network_density = nx.density(self.network)

        self.weighting_matrix = self._calc_weighting_matrix_attribute(self.beta_vec)#INTRODUCE HOMOPHILY INTO THE NETWORK BY ASSORTING BY BETA WITHING GROUPS
        
        #Assume nobody adopts EV at the start, THIS MAY BE AN ISSUE
        self.consider_ev_vec = np.zeros(self.num_individuals).astype(np.int8)
 
        self.current_vehicles = self.set_init_cars_selection(parameters_social_network)

        self.update_EV_stock()

        self.consider_ev_vec, self.ev_adoption_vec = self.calculate_ev_adoption(ev_type=3)#BASED ON CONSUMPTION PREVIOUS TIME STEP

    ###############################################################################################################################################################
    ###############################################################################################################################################################
    #MODEL SETUP

    def init_initial_state(self, parameters_social_network):
        self.num_individuals = int(round(parameters_social_network["num_individuals"]))
        self.id_generator = parameters_social_network["IDGenerator_firms"]
        self.second_hand_merchant = parameters_social_network["second_hand_merchant"]
        self.burn_in_second_hand_market = self.second_hand_merchant.burn_in_second_hand_market
        self.save_timeseries_data_state = parameters_social_network["save_timeseries_data_state"]
        self.compression_factor_state = parameters_social_network["compression_factor_state"]
        self.carbon_price =  parameters_social_network["carbon_price"]

    def init_network_settings(self, parameters_social_network):
        self.selection_bias = parameters_social_network["selection_bias"]
        self.network_structure_seed = parameters_social_network["network_structure_seed"]
        self.K_social_network =  int(round(parameters_social_network["SW_K"]))
        self.prob_rewire = int(round(parameters_social_network["SW_prob_rewire"]))

    def init_preference_distribution(self, parameters_social_network):
        #CHI
        self.a_innovativeness = parameters_social_network["a_innovativeness"]
        self.b_innovativeness = parameters_social_network["b_innovativeness"]
        self.random_state_chi = np.random.RandomState(parameters_social_network["init_vals_innovative_seed"])
        innovativeness_vec_init_unrounded = self.random_state_chi.beta(self.a_innovativeness, self.b_innovativeness, size=self.num_individuals)
        self.chi_vec = np.round(innovativeness_vec_init_unrounded, 1)
        self.ev_adoption_state_vec = np.zeros(self.num_individuals)

        #BETA
        self.random_state_beta = np.random.RandomState(parameters_social_network["init_vals_price_seed"])
        self.beta_vec = self.generate_beta_values_quintiles(self.num_individuals,  parameters_social_network["income"])
        
        #GAMMA
        self.random_state_gamma = np.random.RandomState(parameters_social_network["init_vals_environmental_seed"])
        self.WTP_mean = parameters_social_network["WTP_mean"]
        self.WTP_sd = parameters_social_network["WTP_sd"]
        self.car_lifetime_months = parameters_social_network["car_lifetime_months"]
        WTP_vec_unclipped = self.random_state_gamma.normal(loc = self.WTP_mean, scale = self.WTP_sd, size = self.num_individuals)
        self.WTP_vec = np.clip(WTP_vec_unclipped, a_min = 0, a_max = np.inf)
        self.gamma_vec = self.beta_vec*self.WTP_vec/self.car_lifetime_months
        #ETA

        #d min
        d_i_min_vec_uncapped  = self.random_state_gamma.normal(loc = parameters_social_network["d_i_min"], scale = parameters_social_network["d_i_min_sd"], size = self.num_individuals)
        self.d_i_min_vec = np.clip(d_i_min_vec_uncapped , a_min = 0, a_max = np.inf)

    def generate_beta_values_quintiles(self,n, quintile_incomes):
        """
        Generate a list of beta values for n agents based on quintile incomes.
        Beta for each quintile is calculated as:
            beta = 1 * (lowest_quintile_income / quintile_income)
        
        Args:
            n (int): Total number of agents.
            quintile_incomes (list): List of incomes for each quintile (from lowest to highest).
            
        Returns:
            list: A list of beta values of length n.
        """
        # Calculate beta values for each quintile
        lowest_income = quintile_incomes[0]
        beta_vals = [lowest_income / income for income in quintile_incomes]
        
        # Assign proportions for each quintile (evenly split 20% each)
        proportions = [0.2] * len(quintile_incomes)
        
        # Compute the number of agents for each quintile
        agent_counts = [int(round(p * n)) for p in proportions]
        
        # Adjust for rounding discrepancies to ensure sum(agent_counts) == n
        while sum(agent_counts) < n:
            agent_counts[agent_counts.index(min(agent_counts))] += 1
        while sum(agent_counts) > n:
            agent_counts[agent_counts.index(max(agent_counts))] -= 1
        
        # Generate the beta values list
        beta_list = []
        for count, beta in zip(agent_counts, beta_vals):
            beta_list.extend([beta] * count)
        
        # Shuffle to randomize the order of agents
        self.random_state_beta.shuffle(beta_list)
        
        return np.asarray(beta_list)


    def generate_beta_values(self, n, percentages, beta_vals):
        """
        Generate a list of beta values for n agents based on given percentages and beta values.
        
        Args:
            n (int): Total number of agents.
            percentages (list): List of percentages for each beta value.
            beta_vals (list): List of corresponding beta values.
            
        Returns:
            list: A list of beta values of length n.
        """
        # Normalize percentages if they sum to 100
        total = sum(percentages)
        if total > 1.0:
            percentages = [p / total for p in percentages]
        
        # Compute the number of agents for each beta value
        agent_counts = [int(round(p * n)) for p in percentages]

        # Adjust for rounding discrepancies to ensure sum(agent_counts) == n
        while sum(agent_counts) < n:
            agent_counts[agent_counts.index(min(agent_counts))] += 1
        while sum(agent_counts) > n:
            agent_counts[agent_counts.index(max(agent_counts))] -= 1

        # Generate the beta values list
        beta_list = []
        for count, beta in zip(agent_counts, beta_vals):
            beta_list.extend([beta] * count)

        # Shuffle to randomize the order of agents
        self.random_state_beta.shuffle(beta_list)

        return np.asarray(beta_list)

    def set_init_cars_selection(self, parameters_social_network):
        """GIVE PEOPLE CARS NO CHOICE"""
        old_cars = parameters_social_network["old_cars"]
        for i, car in enumerate(old_cars):
            self.vehicleUsers_list[i].vehicle = car
            
        #SET USER ID OF CARS
        for i, individual in enumerate(self.vehicleUsers_list):
            individual.vehicle.owner_id = individual.user_id

        current_cars =  [user.vehicle for user in self.vehicleUsers_list]

        return current_cars#current cars
        
    def normalize_vec_sum(self, vec):
        return vec/sum(vec)

    def _split_into_groups(self) -> list:
        """
        Split agents into groups for SBM network creation.
        
        Returns:
            list: List of group sizes
        
        Raises:
            ValueError: If SBM_block_num is not positive
        """
        if self.SBM_block_num <= 0:
            raise ValueError("SBM_block_num must be greater than zero.")
        base_count = self.num_individuals//self.SBM_block_num
        remainder = self.num_individuals % self.SBM_block_num
        group_counts = [base_count + 1] * remainder + [base_count] * (self.SBM_block_num - remainder)
        
        return group_counts
    
    def _calc_weighting_matrix_attribute(self, attribute_array: np.ndarray) -> sp.csr_matrix:
        """
        Calculate weighting matrix based on attribute similarities.
        
        Args:
            attribute_array (np.ndarray): Array of attributes for calculating weights
            
        Returns:
            sp.csr_matrix: Sparse matrix of normalized weights
        """
        differences = attribute_array[self.row_indices_sparse] - attribute_array[self.col_indices_sparse]
        weights = np.exp(-self.selection_bias * np.abs(differences))
        non_diagonal_weighting_matrix = sp.csr_matrix(
            (weights, (self.row_indices_sparse, self.col_indices_sparse)),
            shape=self.adjacency_matrix.shape
        )
        norm_weighting_matrix = self._normlize_matrix(non_diagonal_weighting_matrix)

        return norm_weighting_matrix

    def _normlize_matrix(self, matrix: sp.csr_matrix) -> sp.csr_matrix:
        """
        Normalize a sparse matrix row-wise.
        
        Args:
            matrix (sp.csr_matrix): Sparse matrix to normalize
            
        Returns:
            sp.csr_matrix: Row-normalized sparse matrix
        """
        row_sums = np.array(matrix.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        inv_row_sums = 1.0 / row_sums
        diagonal_matrix = sp.diags(inv_row_sums)
        norm_matrix = diagonal_matrix.dot(matrix)
        return norm_matrix

        
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

    def calculate_ev_adoption(self, ev_type=3):
        """
        Calculate the proportion of neighbors using EVs for each user, 
        and determine EV adoption consideration.
        """
        
        self.vehicle_type_vec = np.array([user.vehicle.transportType for user in self.vehicleUsers_list])  # Current vehicle types

        # Create a binary vec indicating EV users
        ev_adoption_vec = (self.vehicle_type_vec == ev_type).astype(int)

        # Calculate the number of EV-adopting neighbors using sparse matrix multiplication
        ev_neighbors = self.weighting_matrix.dot(ev_adoption_vec)

        consider_ev_vec = (ev_neighbors >= self.chi_vec).astype(np.int8)

        return consider_ev_vec, ev_adoption_vec

##########################################################################################################################################################
#MATRIX CALCULATION
#CHECK THE ACTUAL EQUATIONS

    def update_VehicleUsers(self):

        self.chosen_vehicles = []
        user_vehicle_list = [None]*self.num_individuals

        # Generate a single shuffle order
        shuffle_indices = self.random_state_social_network.permutation(self.num_individuals)

        self.second_hand_bought = 0#CAN REMOVE LATER ON IF I DONT ACTUALLY NEED TO COUNT

        # Vectorize current user vehicle attributes
        #self.users_current_vehicle_price_vec = np.asarray([user.vehicle.original_price for user in self.vehicleUsers_list])
        self.users_current_vehicle_type_vec = np.asarray([user.vehicle.transportType for user in self.vehicleUsers_list])#USED TO CHECK IF YOU OWN A CAR

        # Generate current utilities and vehicles
        utilities_current_matrix, d_current_matrix, dict_current = self.generate_utilities_current()

        self.second_hand_merchant_offer_price = self.calc_offer_prices_optimal(dict_current)
        #self.second_hand_merchant_offer_price = self.calc_offer_prices_median(self.current_vehicles)

        # Generate buying utilities and vehicles
        utilities_buying_matrix, buying_vehicles_list, d_buying_matrix = self.generate_utilities()

        # Preallocate the final utilities and distance matrices
        total_columns = utilities_buying_matrix.shape[1] + utilities_current_matrix.shape[1]
        self.utilities_matrix = np.empty((utilities_buying_matrix.shape[0], total_columns))
        self.d_matrix = np.empty((d_buying_matrix.shape[0], total_columns))

        # Assign matrices directly
        self.utilities_matrix[:, :utilities_buying_matrix.shape[1]] = utilities_buying_matrix
        self.utilities_matrix[:, utilities_buying_matrix.shape[1]:] = utilities_current_matrix

        self.d_matrix[:, :d_buying_matrix.shape[1]] = d_buying_matrix
        self.d_matrix[:, d_buying_matrix.shape[1]:] = d_current_matrix

        # Combine the list of vehicles
        available_and_current_vehicles_list = buying_vehicles_list + self.current_vehicles

        #Mask first hand cars
        #self.mask_new_cars()

        # Mask the second-hand cars based on sampling for each individual
        if self.second_hand_cars:
            self.index_current_cars_start = len(self.new_cars) + len(self.second_hand_cars)
        #    self.mask_second_hand_cars()
        else:
            self.index_current_cars_start = len(self.new_cars)

        self.utilities_matrix[self.utilities_matrix < 0] = 0

        utilities_kappa = self.masking_options(self.utilities_matrix, available_and_current_vehicles_list)

        self.emissions_flow = 0#MEASURIBNG THE FLOW

        if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
            self.prep_counters()
            for i, person_index in enumerate(shuffle_indices):
                user = self.vehicleUsers_list[person_index]
                vehicle_chosen, user_vehicle, vehicle_chosen_index, utilities_kappa = self.user_chooses(person_index, user, available_and_current_vehicles_list, utilities_kappa)
                user_vehicle_list[person_index] = user_vehicle
                self.update_counters(person_index, vehicle_chosen, vehicle_chosen_index)
                self.update_emisisons(person_index, vehicle_chosen_index, vehicle_chosen)
        else:
            for i, person_index in enumerate(shuffle_indices):
                user = self.vehicleUsers_list[person_index]
                vehicle_chosen, user_vehicle, vehicle_chosen_index, utilities_kappa = self.user_chooses(person_index, user, available_and_current_vehicles_list, utilities_kappa)
                user_vehicle_list[person_index] = user_vehicle
                self.update_emisisons(person_index, vehicle_chosen_index, vehicle_chosen)
        
        self.emissions_flow_history.append(self.emissions_flow)

        return user_vehicle_list#self.vehicle_chosen_list

#####################################################################################################
    def calc_offer_prices_median(self, current_cars):
        
        if self.t_social_network > self.burn_in_second_hand_market:
            self.quality_index = np.median([car.price/car.Quality_a_t for car in self.new_cars])
            prices = []
            for car in current_cars:
                price = ((car.Quality_a_t*(1-car.delta)**car.L_a_t)*self.quality_index)/(1+self.mu)
                car.cost_second_hand_merchant = price
                prices.append(price)
        else:
            prices = np.zeros(self.num_individuals)

        return prices 
       
    def calc_offer_prices_optimal(self, vehicle_dict_vecs):
        beta = np.median(self.beta_vec)
        gamma = np.median(self.beta_vec)
        U_sum = self.second_hand_merchant.U_sum
        
        #DISTANCE
                # Compute numerator for all vehicles
        numerator = (
            self.alpha * vehicle_dict_vecs["Quality_a_t"] *
            ((1 - vehicle_dict_vecs["delta"]) ** vehicle_dict_vecs["L_a_t"])
        ) 
        denominator = (
            ((beta/vehicle_dict_vecs["Eff_omega_a_t"]) * (vehicle_dict_vecs["fuel_cost_c"] + self.carbon_price*vehicle_dict_vecs["e_t"])) +
            ((gamma/ vehicle_dict_vecs["Eff_omega_a_t"]) * vehicle_dict_vecs["e_t"])
        )  # Shape: (num_individuals, num_vehicles)

        # Calculate optimal distance matrix for each individual-vehicle pair
        d_i_t_vec = (numerator / denominator) ** (1 / (1 - self.alpha))

        #present UTILITY
        # Compute cost component based on transport type, with conditional operations
        cost_component = (beta/ vehicle_dict_vecs["Eff_omega_a_t"]) * (vehicle_dict_vecs["fuel_cost_c"] + self.carbon_price*vehicle_dict_vecs["e_t"]) + ((gamma/ vehicle_dict_vecs["Eff_omega_a_t"]) * vehicle_dict_vecs["e_t"])
        # Compute the commuting utility for each individual-vehicle pair
        present_utility_vec = np.maximum(
            0,
            vehicle_dict_vecs["Quality_a_t"] * ((1 - vehicle_dict_vecs["delta"]) ** vehicle_dict_vecs["L_a_t"]) * (d_i_t_vec ** self.alpha) - d_i_t_vec * cost_component
        )

        # Save the base utility
        B_vec = present_utility_vec/(self.r + (np.log(1+vehicle_dict_vecs["delta"]))/(1-self.alpha))

        inside_component = U_sum*(U_sum + B_vec - beta*vehicle_dict_vecs["last_price_paid"])
       
        #negative_proportion = np.sum(inside_component < 0) / len(inside_component)
        #print("price offered: negative_proportion", negative_proportion)

        # Adjust the component to avoid negative square roots
        inside_component_adjusted = np.maximum(inside_component, 0)  # Replace negatives with 0

        
        price_optimal_vec = np.where(
            inside_component < 0,
            vehicle_dict_vecs["last_price_paid"],
            np.minimum(vehicle_dict_vecs["last_price_paid"], (U_sum  + B_vec - np.sqrt( inside_component_adjusted))/beta )
        )

        price_vec_negative = price_optimal_vec/(1+self.mu)#CAN HAVE NEATIVE PRICES IF THE OPTIMAL PRICE IS NEGATIVE, BASICALLY SCRAP
        
        price_vec = np.maximum(price_vec_negative, 0) #I need to make sure that if the optimal price is negative then the price vec offered is literally zero

        for i, vehicle in enumerate(self.current_vehicles):
            vehicle.cost_second_hand_merchant = price_vec[i]

        return price_vec
    
########################################################################################################
    
    def _gen_mask_from_indices(self, sampled_indices, init_index, available_and_current_vehicles_list, total_number_type):
        # Start with a boolean matrix full of True (allowing all by default)
        mask = np.ones((self.num_individuals, len(available_and_current_vehicles_list)), dtype=bool)
        
        # Set False in the mask outside sampled indices for each individual
        for i, car_indices in enumerate(sampled_indices):
            mask[i, init_index:init_index + total_number_type] = False  # Block out all initially in the range
            mask[i, init_index + car_indices] = True  # Enable only selected indices
        return mask

    def gen_new_car_mask(self, available_and_current_vehicles_list):
        init_index = 0
        return self._gen_mask_from_indices(self.sampled_indices_new_cars, init_index, available_and_current_vehicles_list, len(self.new_cars))

    def gen_second_hand_mask(self, available_and_current_vehicles_list):
        init_index = len(self.new_cars)  # Adjust as per vehicle order
        return self._gen_mask_from_indices(self.sampled_indices_second_hand, init_index, available_and_current_vehicles_list, self.num_second_hand_cars)

    def gen_mask(self, available_and_current_vehicles_list):
        # Generate individual masks based on vehicle type and user conditions
        # Create a boolean vector where True indicates that a vehicle is NOT an EV (non-EV)
        not_ev_vec = np.array([vehicle.transportType != 3 for vehicle in available_and_current_vehicles_list], dtype=bool)
        consider_ev_all_vehicles = np.outer(self.consider_ev_vec == 1, np.ones(len(not_ev_vec), dtype=bool))
        
        dont_consider_evs_on_ev = np.outer(self.consider_ev_vec == 0, not_ev_vec)#This part masks EVs (False for EVs) only for individuals who do not consider EVs

        # Create an outer product to apply the EV consideration across all cars
        ev_mask_matrix = (consider_ev_all_vehicles | dont_consider_evs_on_ev).astype(int)
        #sampled_new_car_mask = self.gen_new_car_mask(available_and_current_vehicles_list)

        combined_mask = ev_mask_matrix

        return combined_mask

    def calc_util_kappa(self,  utilities_matrix_masked):
        norm_utility = utilities_matrix_masked/np.amax(utilities_matrix_masked)
        utilities_kappa = np.power(norm_utility, self.kappa)

        return utilities_kappa
    
    def apply_mask(self, utilities_matrix, combined_mask):
        
        utilities_matrix_masked = utilities_matrix * combined_mask

        return utilities_matrix_masked

    def masking_options(self, utilities_matrix, available_and_current_vehicles_list):

        combined_mask = self.gen_mask(available_and_current_vehicles_list)
        utilities_matrix_masked = self.apply_mask(utilities_matrix, combined_mask)

        utilities_kappa = self.calc_util_kappa(utilities_matrix_masked)

        return utilities_kappa

    def user_chooses(self, person_index, user, available_and_current_vehicles_list, utilities_kappa):
        # Select individual-specific utilities
        individual_specific_util = utilities_kappa[person_index]  

        # Check if all utilities are zero after filtering
        if not np.any(individual_specific_util):#THIS SHOULD ONLY REALLY BE TRIGGERED RIGHT AT THE START
            """THIS SHOULD ACRUALLY BE USED"""
            if self.t_social_network == 0:
                #pick random vehicle which is available
                choice_index = self.random_state_social_network.choice(len(available_and_current_vehicles_list), p=probability_choose)
            else:#keep current car
                
                choice_index = self.index_current_cars_start + person_index #available_and_current_vehicles_list.index(user.vehicle)
        else:
            # Calculate the probability of choosing each vehicle
            """
            SHODY FIX THIS PROPERY
            """
            if np.isnan(np.sum(individual_specific_util)):
                individual_specific_util = np.nan_to_num(individual_specific_util)
                
            sum_prob = np.sum(individual_specific_util)
            probability_choose = individual_specific_util / sum_prob
            choice_index = self.random_state_social_network.choice(len(available_and_current_vehicles_list), p=probability_choose)
            #choice_index = np.argmax(probability_choose)
        # Record the chosen vehicle
        vehicle_chosen = available_and_current_vehicles_list[choice_index]

        self.chosen_vehicles.append(vehicle_chosen)#DONT NEED TO WORRY ABOUT ORDER

        # Handle consequences of the choice
        if user.user_id != vehicle_chosen.owner_id:  # New vehicle, not currently owned
            # Transfer the user's current vehicle to the second-hand merchant, if any
            if isinstance(user.vehicle, PersonalCar):
                self.second_hand_merchant.spent += user.vehicle.cost_second_hand_merchant
                if user.vehicle.init_car or user.vehicle.cost_second_hand_merchant == 0:#ITS AN INITAL CAR WE DOTN WANT TO ALLOW THSOE TO BE SOLD 
                    user.vehicle.owner_id = -99#send to shadow realm
                    user.vehicle = None
                else:
                    self.second_hand_merchant_price_paid.append(user.vehicle.cost_second_hand_merchant)
                    if self.t_social_network > self.burn_in_second_hand_market:
                        user.vehicle.owner_id = self.second_hand_merchant.id
                        self.second_hand_merchant.add_to_stock(user.vehicle)
                        user.vehicle = None
                    else:#VANISH CAR TO THE ABYSS
                        user.vehicle.owner_id = -99#send to shadow realm
                        user.vehicle = None

            
            if vehicle_chosen.owner_id == self.second_hand_merchant.id:# Buy a second-hand car
                #SET THE UTILITY TO 0 of that second hand car
                utilities_kappa[:, choice_index] = 0

                self.second_hand_merchant.remove_car(vehicle_chosen)
                self.second_hand_bought += 1

                vehicle_chosen.owner_id = user.user_id
                vehicle_chosen.scenario = "current_car"
                user.vehicle = vehicle_chosen
                self.buy_second_hand_car+=1
                user.vehicle.last_price_paid = user.vehicle.price
                self.car_prices_sold_second_hand.append(user.vehicle.price)
                self.second_hand_merchant.income += user.vehicle.price

            elif isinstance(vehicle_chosen, CarModel):  # Brand new car

                personalCar_id = self.id_generator.get_new_id()
                user.vehicle = PersonalCar(personalCar_id, vehicle_chosen.firm, user.user_id, vehicle_chosen.component_string, vehicle_chosen.parameters, vehicle_chosen.attributes_fitness, vehicle_chosen.price)
                self.buy_new_car+=1
                self.car_prices_sold_new.append(user.vehicle.price)
            else:
                raise(ValueError("invalid user transport behaviour"))
        else:
            self.keep_car += 1#KEEP CURRENT CAR

        # Update the age or timer of the chosen vehicle
        #if isinstance(vehicle_chosen, PersonalCar):
        user.vehicle.update_timer()

        return vehicle_chosen, user.vehicle, choice_index, utilities_kappa

    def generate_utilities_current(self):

        """ Deal with the special case of utilities of current vehicles"""

        
        CV_vehicle_dict_vecs = self.gen_current_vehicle_dict_vecs(self.current_vehicles)
        #CV_vehicle_dict_vecs = self.gen_vehicle_dict_vecs(self.current_vehicles)

        CV_utilities, d_current = self.vectorised_calculate_utility_current(CV_vehicle_dict_vecs)

        CV_utilities_matrix = np.diag(CV_utilities)
        d_current_matrix = np.diag(d_current)

        return CV_utilities_matrix, d_current_matrix, CV_vehicle_dict_vecs

    def generate_utilities(self):
        #CALC WHO OWNS CAR
        owns_car_mask = self.users_current_vehicle_type_vec > 1

        self.price_owns_car_vec = np.where(
            owns_car_mask,
            self.second_hand_merchant_offer_price,
            0
        )

        # Generate utilities and distances for new cars
        NC_vehicle_dict_vecs = self.gen_vehicle_dict_vecs(self.new_cars)
        NC_utilities, d_NC = self.vectorised_calculate_utility_cars(NC_vehicle_dict_vecs)

        # Calculate the total columns needed for utilities and distance matrices
        total_columns = NC_utilities.shape[1]

        if self.second_hand_cars:
            SH_vehicle_dict_vecs = self.gen_vehicle_dict_vecs(self.second_hand_cars)
            SH_utilities, d_SH = self.vectorised_calculate_utility_second_hand_cars(SH_vehicle_dict_vecs)

            total_columns += SH_utilities.shape[1]

        # Preallocate arrays with the total required columns
        utilities_matrix = np.empty((self.num_individuals, total_columns))
        d_matrix = np.empty((self.num_individuals, total_columns))

        # Fill in preallocated arrays with submatrices
        col_idx = 0#start the counter at zero then increase as you set things up
        utilities_matrix[:, col_idx:col_idx + NC_utilities.shape[1]] = NC_utilities
        d_matrix[:, col_idx:col_idx + d_NC.shape[1]] = d_NC
        col_idx += NC_utilities.shape[1]

        if self.second_hand_cars:
            utilities_matrix[:, col_idx:col_idx + SH_utilities.shape[1]] = SH_utilities
            d_matrix[:, col_idx:col_idx + d_SH.shape[1]] = d_SH
            car_options = self.new_cars + self.second_hand_cars
        else:
            car_options = self.new_cars

        return utilities_matrix, car_options, d_matrix

    def gen_current_vehicle_dict_vecs(self, list_vehicles):
        # Initialize dictionary to hold lists of vehicle properties
        """
        INCLUDES LAST PRICE PAID
        """
        vehicle_dict_vecs = {
            "Quality_a_t": [], 
            "Eff_omega_a_t": [], 
            "price": [], 
            "delta": [], 
            "production_emissions": [],
            "fuel_cost_c": [], 
            "e_t": [],
            "L_a_t": [],
            "transportType": [],
            "last_price_paid": []
        }

        # Iterate over each vehicle to populate the arrays
        for vehicle in list_vehicles:
            if vehicle.transportType == 2:
                vehicle.fuel_cost_c = self.gas_price
            else:
                vehicle.fuel_cost_c = self.electricity_price
                vehicle.e_t = self.electricity_emissions_intensity
        
            vehicle_dict_vecs["Quality_a_t"].append(vehicle.Quality_a_t)
            vehicle_dict_vecs["Eff_omega_a_t"].append(vehicle.Eff_omega_a_t)
            vehicle_dict_vecs["price"].append(vehicle.price)
            vehicle_dict_vecs["delta"].append(vehicle.delta)
            vehicle_dict_vecs["production_emissions"].append(vehicle.emissions)
            vehicle_dict_vecs["fuel_cost_c"].append(vehicle.fuel_cost_c)
            vehicle_dict_vecs["e_t"].append(vehicle.e_t)
            vehicle_dict_vecs["L_a_t"].append(vehicle.L_a_t)
            vehicle_dict_vecs["transportType"].append(vehicle.transportType)
            vehicle_dict_vecs["last_price_paid"].append(vehicle.last_price_paid)
            
        # convert lists to numpy arrays for vectorised operations
        for key in vehicle_dict_vecs:
            vehicle_dict_vecs[key] = np.array(vehicle_dict_vecs[key])

        return vehicle_dict_vecs
    
    def gen_vehicle_dict_vecs(self, list_vehicles):
        # Initialize dictionary to hold lists of vehicle properties

        vehicle_dict_vecs = {
            "Quality_a_t": [], 
            "Eff_omega_a_t": [], 
            "price": [], 
            "delta": [], 
            "production_emissions": [],
            "fuel_cost_c": [], 
            "e_t": [],
            "L_a_t": [],
            "transportType": []
        }

        # Iterate over each vehicle to populate the arrays
        for vehicle in list_vehicles:
            vehicle_dict_vecs["Quality_a_t"].append(vehicle.Quality_a_t)
            vehicle_dict_vecs["Eff_omega_a_t"].append(vehicle.Eff_omega_a_t)
            vehicle_dict_vecs["price"].append(vehicle.price)
            vehicle_dict_vecs["delta"].append(vehicle.delta)
            vehicle_dict_vecs["production_emissions"].append(vehicle.emissions)
            vehicle_dict_vecs["fuel_cost_c"].append(vehicle.fuel_cost_c)
            vehicle_dict_vecs["e_t"].append(vehicle.e_t)
            vehicle_dict_vecs["L_a_t"].append(vehicle.L_a_t)
            vehicle_dict_vecs["transportType"].append(vehicle.transportType)

        # convert lists to numpy arrays for vectorised operations
        for key in vehicle_dict_vecs:
            vehicle_dict_vecs[key] = np.array(vehicle_dict_vecs[key])

        return vehicle_dict_vecs

    def vectorised_calculate_utility_current(self, vehicle_dict_vecs):
        """
        Optimized utility calculation assuming individuals compare either their current car, with price adjustments only applied for those who do not own a car.
        """

        # Compute basic utility components common across all scenarios
        d_i_t = np.maximum(
            self.d_i_min_vec,
            self.vectorised_optimal_distance_current(vehicle_dict_vecs)
        )  # Ensuring compatibility for element-wise comparison

        commuting_util_vec = self.vectorised_commuting_utility_current(vehicle_dict_vecs, d_i_t)
        base_utility_vec = commuting_util_vec / (self.r + (np.log(1 + vehicle_dict_vecs["delta"])) / (1 - self.alpha))

        # Create mask for users who do NOT own a car
        does_not_own_car_mask = self.users_current_vehicle_type_vec <= 1

        # Calculate the price difference and adjustment only for individuals without a car
        if np.any(does_not_own_car_mask):  # Check if there are any individuals without a car
            # Calculate the price difference for those without a car
            price_difference = vehicle_dict_vecs["price"]  # Only need the price of the transport option

            # Compute the price adjustment for individuals without a car
            price_adjustment = self.beta_vec * price_difference  # Shape: (num_individuals)

            # Initialize utility vec with base utility
            U_a_i_t_vec = base_utility_vec

            U_a_i_t_vec[does_not_own_car_mask] -= price_adjustment[does_not_own_car_mask]
        else:
            # EVERYONE OWNS A CAR SO THE UTILITY IS JUST THE BASE UTILITY
            U_a_i_t_vec = base_utility_vec

        return U_a_i_t_vec, d_i_t

    def vectorised_calculate_utility_second_hand_cars(self, vehicle_dict_vecs):
        # Compute shared base utility components
        d_i_t = np.maximum(self.d_i_min_vec[:, np.newaxis], self.vectorised_optimal_distance_cars(vehicle_dict_vecs))

        commuting_util_matrix = self.vectorised_commuting_utility_cars(vehicle_dict_vecs, d_i_t)
        base_utility_matrix = commuting_util_matrix / (self.r + (np.log(1 + vehicle_dict_vecs["delta"])) / (1 - self.alpha))


        price_difference = np.where(
            vehicle_dict_vecs["transportType"][:, np.newaxis] == 3,  # Check transportType
            (vehicle_dict_vecs["price"][:, np.newaxis] - self.used_rebate - self.price_owns_car_vec),  # Apply rebate
            vehicle_dict_vecs["price"][:, np.newaxis] - self.price_owns_car_vec  # No rebate
        )

        # Calculate price and emissions adjustments once
        price_adjustment = np.multiply(self.beta_vec[:, np.newaxis], price_difference.T)
        
        # Use in-place modification to save memor
        U_a_i_t_matrix = base_utility_matrix - price_adjustment

        return U_a_i_t_matrix, d_i_t
    
    def vectorised_calculate_utility_cars(self, vehicle_dict_vecs):
        # Compute shared base utility components
        d_i_t = np.maximum(self.d_i_min_vec[:, np.newaxis], self.vectorised_optimal_distance_cars(vehicle_dict_vecs))
        
        commuting_util_matrix = self.vectorised_commuting_utility_cars(vehicle_dict_vecs, d_i_t)

        base_utility_matrix = commuting_util_matrix / (self.r + (np.log(1 + vehicle_dict_vecs["delta"])/(1 - self.alpha)))

        #price_difference = vehicle_dict_vecs["price"][:, np.newaxis] - self.price_owns_car_vec

        # Calculate price difference, applying rebate only for transportType == 3
        price_difference = np.where(
            vehicle_dict_vecs["transportType"][:, np.newaxis] == 3,  # Check transportType
            (vehicle_dict_vecs["price"][:, np.newaxis] - self.rebate - self.price_owns_car_vec),  # Apply rebate
            vehicle_dict_vecs["price"][:, np.newaxis] - self.price_owns_car_vec  # No rebate
        )
    
        # Calculate price and emissions adjustments once
        price_adjustment = np.multiply(self.beta_vec[:, np.newaxis], price_difference.T)
        
        # Use in-place modification to save memory
        emissions_penalty = np.multiply(self.gamma_vec[:, np.newaxis], vehicle_dict_vecs["production_emissions"])
        U_a_i_t_matrix = base_utility_matrix - price_adjustment - emissions_penalty
        return U_a_i_t_matrix, d_i_t
    
    def vectorised_optimal_distance_current(self, vehicle_dict_vecs):
        """
        Only does it for the 1 car, calculate the optimal distance for each individual considering only their corresponding vehicle.
        Assumes each individual has one vehicle, aligned by index.
        """
        # Compute numerator for each individual-vehicle pair
        numerator = (
            self.alpha * vehicle_dict_vecs["Quality_a_t"] *
            (1 - vehicle_dict_vecs["delta"]) ** vehicle_dict_vecs["L_a_t"]
        )  # Shape: (num_individuals,)

        denominator = ((self.beta_vec/vehicle_dict_vecs["Eff_omega_a_t"]) * (vehicle_dict_vecs["fuel_cost_c"] + self.carbon_price*vehicle_dict_vecs["e_t"])) + ((self.gamma_vec/vehicle_dict_vecs["Eff_omega_a_t"]) * vehicle_dict_vecs["e_t"])

        # Calculate optimal distance vec for each individual-vehicle pair
        optimal_distance_vec = (numerator / denominator) ** (1 / (1 - self.alpha))

        return optimal_distance_vec  # Shape: (num_individuals,)

    def vectorised_optimal_distance_cars(self, vehicle_dict_vecs):
        """Distance of all cars for all agents"""
        # Compute numerator for all vehicles
        numerator = (
            self.alpha * vehicle_dict_vecs["Quality_a_t"] *
            ((1 - vehicle_dict_vecs["delta"]) ** vehicle_dict_vecs["L_a_t"])
        )  # Shape: (num_vehicles,)

        # Compute denominator for all individual-vehicle pairs using broadcasting
        # Reshape self.beta_vec and self.gamma_vec to (num_individuals, 1) for broadcasting across vehicles
        denominator = (
            ((self.beta_vec[:, np.newaxis]/vehicle_dict_vecs["Eff_omega_a_t"]) * (vehicle_dict_vecs["fuel_cost_c"] + self.carbon_price*vehicle_dict_vecs["e_t"])) +
            ((self.gamma_vec[:, np.newaxis]/ vehicle_dict_vecs["Eff_omega_a_t"]) * vehicle_dict_vecs["e_t"])
        )  # Shape: (num_individuals, num_vehicles)

        # Calculate optimal distance matrix for each individual-vehicle pair
        optimal_distance_matrix = (numerator / denominator) ** (1 / (1 - self.alpha))

        return optimal_distance_matrix  # Shape: (num_individuals, num_vehicles)

    def vectorised_commuting_utility_current(self, vehicle_dict_vecs, d_i_t):
        """
        Only one car. Calculate the commuting utility for each individual considering only their corresponding vehicle.
        Assumes each individual has one vehicle, aligned by index.
        """

        # Compute cost component based on transport type, without broadcasting
        cost_component = ((self.beta_vec/ vehicle_dict_vecs["Eff_omega_a_t"]) * (vehicle_dict_vecs["fuel_cost_c"] + self.carbon_price*vehicle_dict_vecs["e_t"])) + ((self.gamma_vec/ vehicle_dict_vecs["Eff_omega_a_t"]) * vehicle_dict_vecs["e_t"])

        # Calculate the commuting utility for each individual-vehicle pair
        commuting_utility_vec = np.maximum(
            0,
            vehicle_dict_vecs["Quality_a_t"] * (1 - vehicle_dict_vecs["delta"]) ** vehicle_dict_vecs["L_a_t"] *
            (d_i_t ** self.alpha) - d_i_t * cost_component
        )  # Shape: (num_individuals,)

        return commuting_utility_vec  # Shape: (num_individuals,)

    def vectorised_commuting_utility_cars(self, vehicle_dict_vecs, d_i_t):
        """utility of all cars for all agents"""
        # dit Shape: (num_individuals, num_vehicles)

        # Compute cost component based on transport type, with conditional operations
        cost_component = (self.beta_vec[:, np.newaxis]/ vehicle_dict_vecs["Eff_omega_a_t"]) * (vehicle_dict_vecs["fuel_cost_c"] + self.carbon_price*vehicle_dict_vecs["e_t"]) + ((self.gamma_vec[:, np.newaxis]/ vehicle_dict_vecs["Eff_omega_a_t"]) * vehicle_dict_vecs["e_t"])

        # Compute the commuting utility for each individual-vehicle pair

        commuting_utility_matrix = np.maximum(
            0,
            vehicle_dict_vecs["Quality_a_t"] * ((1 - vehicle_dict_vecs["delta"]) ** vehicle_dict_vecs["L_a_t"]) * (d_i_t ** self.alpha) - d_i_t * cost_component
        )  # Shape: (num_individuals, num_vehicles)

        return commuting_utility_matrix  # Shape: (num_individuals, num_vehicles)

####################################################################################################################################
    #TIMESERIES
    def prep_counters(self):
        
        self.users_driving_emissions_vec = np.zeros(self.num_individuals)
        self.users_distance_vec = np.zeros(self.num_individuals)
        self.users_utility_vec  = np.zeros(self.num_individuals)
        self.users_transport_type_vec  = np.full((self.num_individuals), np.nan)
    
        #variable to track
    
        self.total_driving_emissions = 0
        self.total_production_emissions = 0
        self.total_utility = 0
        self.total_distance_travelled = 0
        self.ICE_users = 0 
        self.EV_users = 0
        self.new_ICE_cars_bought = 0
        self.new_EV_cars_bought = 0
  
        self.second_hand_users = 0
        self.quality_vals = []
        self.efficiency_vals = []
        self.production_cost_vals = []
        self.quality_vals_ICE = []
        self.efficiency_vals_ICE = []
        self.production_cost_vals_ICE = []
        self.quality_vals_EV = []
        self.efficiency_vals_EV = []
        self.production_cost_vals_EV = []
        self.new_cars_bought = 0
        self.car_ages = []

        self.cars_cum_distances_driven = []
        self.cars_cum_driven_emissions = []
        self.cars_cum_emissions = []

        self.car_prices_sold_new = []
        self.car_prices_sold_second_hand = []

        self.keep_car = 0
        self.buy_new_car = 0
        self.buy_second_hand_car = 0

        self.second_hand_merchant_price_paid = []
    
    def update_counters(self, person_index, vehicle_chosen, vehicle_chosen_index):
        #ADD TOTAL EMISSIONS
        
        
        driven_distance = self.d_matrix[person_index][vehicle_chosen_index]           
        
        self.users_driving_emissions_vec[person_index] = (driven_distance/vehicle_chosen.Eff_omega_a_t)*vehicle_chosen.e_t

        self.users_distance_vec[person_index] = driven_distance
        self.users_utility_vec[person_index] = self.utilities_matrix[person_index][vehicle_chosen_index]
        self.users_transport_type_vec[person_index] = vehicle_chosen.transportType
        
        if vehicle_chosen.scenario == "new_car":  
            self.new_cars_bought +=1
            self.total_production_emissions += vehicle_chosen.emissions
            if vehicle_chosen.transportType == 2:
                self.new_ICE_cars_bought +=1
            else:
                self.new_EV_cars_bought +=1
            
        car_driving_emissions = (driven_distance/vehicle_chosen.Eff_omega_a_t)*vehicle_chosen.e_t 
        self.total_driving_emissions += car_driving_emissions 

        if isinstance(vehicle_chosen, PersonalCar):
            vehicle_chosen.total_distance += driven_distance
            self.cars_cum_distances_driven.append(vehicle_chosen.total_distance)
            vehicle_chosen.total_driving_emmissions += car_driving_emissions
            self.cars_cum_driven_emissions.append(vehicle_chosen.total_driving_emmissions)
            vehicle_chosen.total_emissions += car_driving_emissions
            self.cars_cum_emissions.append(vehicle_chosen.total_emissions)

        self.car_ages.append(vehicle_chosen.L_a_t)
        self.quality_vals.append(vehicle_chosen.Quality_a_t)#done here for efficiency
        self.efficiency_vals.append(vehicle_chosen.Eff_omega_a_t)
        self.production_cost_vals.append(vehicle_chosen.ProdCost_t)

        if vehicle_chosen.transportType == 2:#ICE 
            self.quality_vals_ICE.append(vehicle_chosen.Quality_a_t)#done here for efficiency
            self.efficiency_vals_ICE.append(vehicle_chosen.Eff_omega_a_t)
            self.production_cost_vals_ICE.append(vehicle_chosen.ProdCost_t)
        else:
            self.quality_vals_EV.append(vehicle_chosen.Quality_a_t)#done here for efficiency
            self.efficiency_vals_EV.append(vehicle_chosen.Eff_omega_a_t)
            self.production_cost_vals_EV.append(vehicle_chosen.ProdCost_t)

        self.total_utility += self.utilities_matrix[person_index][vehicle_chosen_index]
        self.total_distance_travelled += driven_distance
            
        if isinstance(vehicle_chosen, PersonalCar):
            self.second_hand_users +=1

        if vehicle_chosen.transportType == 2:#ICE
            self.ICE_users += 1
        elif vehicle_chosen.transportType == 3:#EV
            self.EV_users += 1
        else:
            raise ValueError("Invalid transport type")
        
    def set_up_time_series_social_network(self):
        #"""
        self.history_driving_emissions = []
        self.history_production_emissions = []
        self.history_total_emissions = []
        self.history_total_utility = []
        self.history_total_distance_driven = []
        self.history_ev_adoption_rate = []
        self.history_consider_ev_rate = []
        self.history_consider_ev = []
        self.history_ICE_users = []
        self.history_EV_users = []
        self.history_second_hand_users = []
        self.history_new_ICE_cars_bought = []
        self.history_new_EV_cars_bought = []
        # New history attributes for vehicle attributes
        self.history_quality = []
        self.history_efficiency = []
        self.history_production_cost = []

        self.history_quality_ICE = []
        self.history_efficiency_ICE  = []
        self.history_production_cost_ICE  = []

        self.history_quality_EV = []
        self.history_efficiency_EV = []
        self.history_production_cost_EV = []

        self.history_attributes_EV_cars_on_sale_all_firms = []
        self.history_attributes_ICE_cars_on_sale_all_firms = []
        self.history_second_hand_bought = []
        self.history_new_car_bought = []
        self.history_car_age = []

        self.history_cars_cum_distances_driven = []
        self.history_cars_cum_driven_emissions = []
        self.history_cars_cum_emissions = []

        #INDIVIDUALS LEVEL DATA
        self.history_driving_emissions_individual = []
        self.history_distance_individual = []
        self.history_utility_individual = []
        self.history_transport_type_individual = []

        self.history_count_buy = []

        #self.history_quality_index = []
        self.history_mean_price = []
        self.history_median_price = []
        
        self.history_car_prices_sold_new = []
        self.history_car_prices_sold_second_hand = []

        self.history_quality_users_raw_adjusted = []

        self.history_second_hand_merchant_price_paid = []
    def save_timeseries_data_social_network(self):

        #INDIVIDUALS LEVEL DATA


        #self.history_quality_index.append(self.quality_index)
        self.history_count_buy.append([self.keep_car, self.buy_new_car, self.buy_second_hand_car])

        mean_price_new = np.mean([vehicle.price for vehicle in self.new_cars])
        median_price_new = np.median([vehicle.price for vehicle in self.new_cars])
        if self.second_hand_cars:
            mean_price_second_hand = np.mean([vehicle.price for vehicle in self.second_hand_cars])
            median_price_second_hand = np.median([vehicle.price for vehicle in self.second_hand_cars])
        else:
            mean_price_second_hand = np.nan#NO SECOND HAND CARS
            median_price_second_hand = np.nan#NO SECOND HAND CARS

        self.history_mean_price.append([mean_price_new, mean_price_second_hand])
        self.history_median_price.append([median_price_new, median_price_second_hand])

        self.history_driving_emissions_individual.append(self.users_driving_emissions_vec)
        self.history_distance_individual.append(self.users_distance_vec)
        self.history_utility_individual.append(self.users_utility_vec)
        self.history_transport_type_individual.append(self.users_transport_type_vec)

        self.history_new_ICE_cars_bought.append(self.new_ICE_cars_bought)
        self.history_new_EV_cars_bought.append(self.new_EV_cars_bought)

        #SUMS
        self.history_driving_emissions.append(self.total_driving_emissions)
        self.history_production_emissions.append(self.total_production_emissions)
        self.history_total_emissions.append(self.total_production_emissions + self.total_driving_emissions)
        self.history_total_utility.append(self.total_utility)
        self.history_total_distance_driven.append(self.total_distance_travelled)
        self.history_ev_adoption_rate.append(np.mean(self.ev_adoption_vec))
        self.history_consider_ev_rate.append(np.mean(self.consider_ev_vec))
        self.history_consider_ev.append(self.consider_ev_vec)
        self.history_ICE_users.append(self.ICE_users)
        self.history_EV_users.append(self.EV_users)
        self.history_second_hand_users.append(self.second_hand_users)
        self.history_second_hand_bought.append(self.second_hand_bought)
        self.history_new_car_bought.append(self.new_cars_bought)
        
        self.history_quality.append(self.quality_vals)
        self.history_efficiency.append(self.efficiency_vals)
        self.history_production_cost.append(self.production_cost_vals)

        if self.quality_vals_ICE:
            self.history_quality_ICE.append(self.quality_vals_ICE)
            self.history_efficiency_ICE.append(self.efficiency_vals_ICE)
            self.history_production_cost_ICE.append(self.production_cost_vals_ICE)
        else:
            self.history_quality_ICE.append([np.nan])
            self.history_efficiency_ICE.append([np.nan])
            self.history_production_cost_ICE.append([np.nan])

        if self.quality_vals_EV:
            self.history_quality_EV.append(self.quality_vals_EV)
            self.history_efficiency_EV.append(self.efficiency_vals_EV)
            self.history_production_cost_EV.append(self.production_cost_vals_EV)
        else:
            self.history_quality_EV.append([np.nan])
            self.history_efficiency_EV.append([np.nan])
            self.history_production_cost_EV.append([np.nan])
    

        data_ev = [[vehicle.Quality_a_t, vehicle.Eff_omega_a_t, vehicle.ProdCost_t]  for vehicle in self.all_vehicles_available if vehicle.transportType == 3]
        data_ice = [[vehicle.Quality_a_t ,vehicle.Eff_omega_a_t, vehicle.ProdCost_t]  for vehicle in self.all_vehicles_available if vehicle.transportType == 2]

        self.history_attributes_EV_cars_on_sale_all_firms.append(data_ev)
        self.history_attributes_ICE_cars_on_sale_all_firms.append(data_ice)

        self.history_car_age.append(self.car_ages)

        self.history_cars_cum_distances_driven.append(self.cars_cum_distances_driven)
        self.history_cars_cum_driven_emissions.append(self.cars_cum_driven_emissions)
        self.history_cars_cum_emissions.append(self.cars_cum_emissions)

        self.history_car_prices_sold_new.append(self.car_prices_sold_new)
        self.history_car_prices_sold_second_hand.append(self.car_prices_sold_second_hand)

        self.history_quality_users_raw_adjusted.append([(car.Quality_a_t,car.Quality_a_t*(1-car.delta)**car.L_a_t ) for car in self.current_vehicles])

        self.history_second_hand_merchant_price_paid.append(self.second_hand_merchant_price_paid)

    def update_emisisons(self, person_index, vehicle_chosen_index, vehicle_chosen):
        driven_distance = self.d_matrix[person_index][vehicle_chosen_index]           
        
        emissions_flow =   (driven_distance/vehicle_chosen.Eff_omega_a_t)*vehicle_chosen.e_t
        self.emissions_cumulative += emissions_flow
        self.emissions_flow += emissions_flow

        if vehicle_chosen.scenario == "new_car":  #if its a new car add emisisons
            self.emissions_cumulative += vehicle_chosen.emissions
            self.emissions_flow += vehicle_chosen.emissions

    def update_prices_and_emissions(self):
        #UPDATE EMMISSION AND PRICES, THIS WORKS FOR BOTH PRODUCTION AND INNOVATION
        for car in self.current_vehicles:
            if car.transportType == 2:#ICE
                car.fuel_cost_c = self.gas_price
            elif car.transportType == 3:#EV
                car.fuel_cost_c = self.electricity_price
                car.e_t = self.electricity_emissions_intensity

####################################################################################################################################

    def update_EV_stock(self):
        #CALC EV STOCK
        self.EV_users_count = sum(1 if car.transportType == 3 else 0 for car in  self.current_vehicles)
        self.history_prop_EV.append(self.EV_users_count/self.num_individuals)

    def next_step(self, carbon_price, second_hand_cars,new_cars, gas_price, electricity_price, electricity_emissions_intensity, rebate, used_rebate):
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
        self.gas_price =  gas_price
        self.electricity_price = electricity_price
        self.electricity_emissions_intensity = electricity_emissions_intensity
        self.rebate = rebate
        self.used_rebate = used_rebate

        #update new tech and prices
        self.second_hand_cars, self.new_cars = second_hand_cars, new_cars
        self.all_vehicles_available = self.new_cars + self.second_hand_cars#ORDER IS VERY IMPORTANT

        self.consider_ev_vec, self.ev_adoption_vec = self.calculate_ev_adoption(ev_type=3)#BASED ON CONSUMPTION PREVIOUS TIME STEP
 
        self.current_vehicles = self.update_VehicleUsers()
        self.update_EV_stock()
        #print(self.total_driving_emissions)

        return self.consider_ev_vec,  self.chosen_vehicles #self.chosen_vehicles instead of self.current_vehicles as firms can count porfits
