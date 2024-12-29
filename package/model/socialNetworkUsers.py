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
        self.emissions_cumulative = 0
        self.policy_distortion = 0#NEED FOR OPTIMISATION, measures the distortion from policies
        self.emissions_flow_history = []
        self.history_prop_EV = []
        self.used_firm_rebate_exclusion_set = set()
        self.firm_rebate_exclusion_set = set()

        self.rebate = parameters_social_network["rebate"]
        self.used_rebate = parameters_social_network["used_rebate"]
        self.rebate_low = parameters_social_network["rebate_low"]
        self.used_rebate_low = parameters_social_network["used_rebate_low"]

        self.prob_switch_car = parameters_social_network["prob_switch_car"]

        # Initialize parameters
        self.parameters_vehicle_user = parameters_vehicle_user
        self.init_initial_state(parameters_social_network)
        self.init_network_settings(parameters_social_network)
        self.init_preference_distribution(parameters_social_network)

        self.random_state_social_network = np.random.RandomState(parameters_social_network["social_network_seed"])

        self.alpha =  parameters_vehicle_user["alpha"]
        self.mu =  parameters_vehicle_user["mu"]
        self.r = parameters_vehicle_user["r"]
        self.kappa = int(round(parameters_vehicle_user["kappa"]))
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
        self.network_structure_seed = int(round(parameters_social_network["network_structure_seed"]))
        self.prob_rewire = parameters_social_network["SW_prob_rewire"]
        self.SW_network_density_input = parameters_social_network["SW_network_density"]
        self.SW_prob_rewire = parameters_social_network["SW_prob_rewire"]
        self.SW_K = int(round((self.num_individuals - 1) * self.SW_network_density_input))
        
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

        self.beta_median = np.median(self.beta_vec)
        self.gamma_median = np.median(self.beta_vec)

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

        network = nx.watts_strogatz_graph(n=self.num_individuals, k=self.SW_K, p=self.prob_rewire, seed=self.network_structure_seed)#FIX THE NETWORK STRUCTURE

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

        self.emissions_flow = 0#MEASURIBNG THE FLOW
        self.zero_util_count = 0 #are people actually choosing or beign forced to choose the same
        self.new_bought_vehicles = []#track list of new vehicles
        user_vehicle_list = self.current_vehicles.copy()#assume most people keep their cars
        

        #########################################################
        #LIMIT CALCULATION FOR THOSE THAT DONT NEED TO SWTICH
        # 1) Determine which users can switch
        switch_draws = (self.random_state_social_network.rand(self.num_individuals) < self.prob_switch_car)
        switcher_indices = np.where(switch_draws)[0]  # e.g., [2, 5, 7, ...]
        num_switchers = len(switcher_indices)
        non_switcher_indices = np.where(~switch_draws)[0]  # e.g., [0, 1, 3, 4, 6, ...]

        self.sub_beta_vec = self.beta_vec[switcher_indices]
        self.sub_gamma_vec = self.gamma_vec[switcher_indices]
        self.sub_d_i_min_vec = self.d_i_min_vec[switcher_indices]

        self.second_hand_bought = 0#CAN REMOVE LATER ON IF I DONT ACTUALLY NEED TO COUNT

        # Generate current utilities and vehicles
        #Calculate the optimal distance for all user with current car, NEEDS TO BE DONE ALWAYS AND FOR ALL USERS
        d_i_t, CV_vehicle_dict_vecs = self.generate_distances_current(self.current_vehicles,self.d_i_min_vec, self.beta_vec, self.gamma_vec)

        # NEED THIS FOR SOME OF THE COUNTERS I THINK - CHECK THIS
        if self.second_hand_cars:
            index_current_cars_start = len(self.new_cars) + len(self.second_hand_cars)
        else:
            index_current_cars_start = len(self.new_cars)

        ##########################################################################################################################
        #I now have all the information neccessary all the calcualtions are done
        #Do anything that needs to be doen in terms of counters for the non-switchers
        #I then need to do the users chooses with the switchers
        ##########################################################################################################################

        #NON-SWTICHERS
        if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
            self.prep_counters()
            __, full_CV_utility_vec = self.generate_utilities_current(CV_vehicle_dict_vecs, d_i_t, self.beta_vec, self.gamma_vec)

        for i, person_index in enumerate(non_switcher_indices):
            user = self.vehicleUsers_list[person_index]
            user.vehicle.update_timer()# Update the age or timer of the chosen vehicle
            # Handle consequences of the choice
            vehicle_chosen_index = index_current_cars_start + person_index
            driven_distance = d_i_t[person_index]
            self.update_emisisons(user.vehicle, driven_distance)

            #NEEDED FOR OPTIMISATION, add the carbon price paid 
            self.policy_distortion += (self.carbon_price*user.vehicle.e_t*driven_distance)/user.vehicle.Eff_omega_a_t#
            #NEEDED FOR OPTIMISATION, ELECTRICITY SUBSIDY
            if user.vehicle.transportType == 3:
                self.policy_distortion += (self.electricity_price_subsidy*driven_distance)/user.vehicle.Eff_omega_a_t

            if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
                self.keep_car += 1
                utility = full_CV_utility_vec[person_index]
                self.update_counters(person_index, user.vehicle, driven_distance, utility)

        ##################################################################
        #SWITCHERS

        # 2) Shuffle only that subset of user indices
        shuffle_indices = self.random_state_social_network.permutation(switcher_indices)

        #THIS CAN BE DONE FOR THE SUBSET OF USERS
        CV_filtered_vechicles_dicts, CV_filtered_vehicles = self.filter_vehicle_dict_for_switchers(CV_vehicle_dict_vecs, self.current_vehicles, switcher_indices)
        utilities_current_matrix, __ = self.generate_utilities_current(CV_filtered_vechicles_dicts, d_i_t[switcher_indices], self.sub_beta_vec, self.sub_gamma_vec)
        second_hand_merchant_offer_price = self.calc_offer_prices_max(CV_filtered_vechicles_dicts, CV_filtered_vehicles)#calculate_offer only on thoe individuals who consider swtiching

        # pass those indices to generate_utilities
        utilities_buying_matrix_switchers, buying_vehicles_list, d_buying_matrix_switchers = self.generate_utilities(self.sub_beta_vec, self.sub_gamma_vec, self.sub_d_i_min_vec, second_hand_merchant_offer_price)

        # Preallocate the final utilities and distance matrices
        total_columns = utilities_buying_matrix_switchers.shape[1] + utilities_current_matrix.shape[1]#number of cars which is new+secodn hand + current in the switchers

        """
        I think these may need to be the num indivdiuals shape purely for the index to work as i want it too? or maybe i create a mapping
        """

        self.utilities_matrix_switchers = np.empty((num_switchers, total_columns))
        self.d_matrix_switchers = np.empty((num_switchers, total_columns))
        
        # Assign matrices directly
        self.utilities_matrix_switchers[:, :utilities_buying_matrix_switchers.shape[1]] = utilities_buying_matrix_switchers
        self.utilities_matrix_switchers[:, utilities_buying_matrix_switchers.shape[1]:] = utilities_current_matrix
        
        d_current_matrix_switchers = np.diag(d_i_t[switcher_indices])#take the distances calcualted on current cars, take out all the ones that are not switching then make that a matrix which is diagonal?
        self.d_matrix_switchers[:, :d_buying_matrix_switchers.shape[1]] = d_buying_matrix_switchers
        self.d_matrix_switchers[:, d_buying_matrix_switchers.shape[1]:] = d_current_matrix_switchers

        # Combine the list of vehicles
        available_and_current_vehicles_list = buying_vehicles_list + CV_filtered_vehicles# ITS CURRENT VEHICLES AND NOT FILTERED VEHCILES AS THE SHUFFLING INDEX DOENST ACCOUNT FOR THE FILTERING
        self.utilities_matrix_switchers[self.utilities_matrix_switchers < 0] = 0#set utilities which are negative to 0, needed for probabilities
        utilities_kappa = self.masking_options(self.utilities_matrix_switchers, available_and_current_vehicles_list, self.consider_ev_vec[switcher_indices])

        #########################################################################################################################
        # Create a mapping from global to reduced indices
        global_to_reduced = {global_idx: local_idx for local_idx, global_idx in enumerate(switcher_indices)}
        # Translate shuffle_indices to reduced indices
        shuffle_indices_reduced = [global_to_reduced[idx] for idx in shuffle_indices]

        #########################################################################################################################

        for i, reduced_index in enumerate(shuffle_indices_reduced):
            # Map the reduced index back to the global index
            global_index = switcher_indices[reduced_index]
            
            user = self.vehicleUsers_list[global_index]  # Use the global index to access the user
            vehicle_chosen, user_vehicle, vehicle_chosen_index, utilities_kappa = self.user_chooses(
                global_index, user, available_and_current_vehicles_list, utilities_kappa, reduced_index, index_current_cars_start 
            )
            user_vehicle_list[global_index] = user_vehicle  # Update using the global index

            driven_distance = self.d_matrix_switchers[reduced_index][vehicle_chosen_index]  # Use the reduced index for the matrix
            self.update_emisisons(vehicle_chosen, driven_distance)

            #CARBON TAX POLICY OPTIMISATION
            self.policy_distortion += (self.carbon_price*user.vehicle.e_t*driven_distance)/user.vehicle.Eff_omega_a_t#NEEDED FOR OPTIMISATION of carbon tax
            #NEEDED FOR OPTIMISATION, ELECTRICITY SUBSIDY
            if user.vehicle.transportType == 3:
                self.policy_distortion += (self.electricity_price_subsidy*driven_distance)/user.vehicle.Eff_omega_a_t
                
            if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
                utility = self.utilities_matrix_switchers[reduced_index][vehicle_chosen_index]
                self.update_counters(global_index, vehicle_chosen, driven_distance, utility)

        self.emissions_flow_history.append(self.emissions_flow)

        return user_vehicle_list

#####################################################################################################

    def calc_offer_prices_max(self, vehicle_dict_vecs, filtered_vehicles):
        
        #DISTANCE
                # Compute numerator for all vehicles
        numerator = (
            self.alpha * vehicle_dict_vecs["Quality_a_t"] *
            ((1 - vehicle_dict_vecs["delta"]) ** vehicle_dict_vecs["L_a_t"])
        ) 
        denominator = (
            ((self.beta_median/vehicle_dict_vecs["Eff_omega_a_t"]) * (vehicle_dict_vecs["fuel_cost_c"] + self.carbon_price*vehicle_dict_vecs["e_t"])) +
            ((self.gamma_median/ vehicle_dict_vecs["Eff_omega_a_t"]) * vehicle_dict_vecs["e_t"])
        )  # Shape: (num_individuals, num_vehicles)

        # Calculate optimal distance matrix for each individual-vehicle pair
        d_i_t_vec = (numerator / denominator) ** (1 / (1 - self.alpha))

        #present UTILITY
        # Compute cost component based on transport type, with conditional operations
        cost_component = (self.beta_median/ vehicle_dict_vecs["Eff_omega_a_t"]) * (vehicle_dict_vecs["fuel_cost_c"] + self.carbon_price*vehicle_dict_vecs["e_t"]) + ((self.gamma_median/ vehicle_dict_vecs["Eff_omega_a_t"]) * vehicle_dict_vecs["e_t"])
        # Compute the commuting utility for each individual-vehicle pair
        present_utility_vec = np.maximum(
            0,
            vehicle_dict_vecs["Quality_a_t"] * ((1 - vehicle_dict_vecs["delta"]) ** vehicle_dict_vecs["L_a_t"]) * (d_i_t_vec ** self.alpha) - d_i_t_vec * cost_component
        )

        # Save the base utility
        #B_vec = present_utility_vec/(self.r + (np.log(1+vehicle_dict_vecs["delta"]))/(1-self.alpha))
        B_vec = present_utility_vec*(1+self.r) / ((1+self.r) - (1 - vehicle_dict_vecs["delta"])**(1/(1 - self.alpha)))

        price_optimal_vec = B_vec/self.beta_median

        price_vec_negative = price_optimal_vec*(1-self.mu)#CAN HAVE NEATIVE PRICES IF THE OPTIMAL PRICE IS NEGATIVE, BASICALLY SCRAP
        
        price_vec = np.maximum(price_vec_negative, 0) #I need to make sure that if the optimal price is negative then the price vec offered is literally zero

        for i, vehicle in enumerate(filtered_vehicles):
            vehicle.cost_second_hand_merchant = price_vec[i]

        return price_vec

########################################################################################################
    
    def gen_mask(self, available_and_current_vehicles_list, consider_ev_vec):
        # Generate individual masks based on vehicle type and user conditions
        # Create a boolean vector where True indicates that a vehicle is NOT an EV (non-EV)
        not_ev_vec = np.array([vehicle.transportType != 3 for vehicle in available_and_current_vehicles_list], dtype=bool)
        consider_ev_all_vehicles = np.outer(consider_ev_vec == 1, np.ones(len(not_ev_vec), dtype=bool))
        
        dont_consider_evs_on_ev = np.outer(consider_ev_vec == 0, not_ev_vec)#This part masks EVs (False for EVs) only for individuals who do not consider EVs

        # Create an outer product to apply the EV consideration across all cars
        ev_mask_matrix = (consider_ev_all_vehicles | dont_consider_evs_on_ev).astype(int)
        #sampled_new_car_mask = self.gen_new_car_mask(available_and_current_vehicles_list)

        combined_mask = ev_mask_matrix

        return combined_mask

    def masking_options(self, utilities_matrix, available_and_current_vehicles_list, consider_ev_vec):

        combined_mask = self.gen_mask(available_and_current_vehicles_list, consider_ev_vec)
        utilities_matrix_masked = utilities_matrix * combined_mask
        norm_utility = utilities_matrix_masked/np.amax(utilities_matrix_masked)
        utilities_kappa = np.power(norm_utility, self.kappa)

        return utilities_kappa
    
#########################################################################################################################################################
    #choosing vehicles
    def user_chooses(self, person_index, user, available_and_current_vehicles_list, utilities_kappa, reduced_person_index, index_current_cars_start ):
        # Select individual-specific utilities
        individual_specific_util = utilities_kappa[reduced_person_index]  

        #SWICHING_CLAUSE
        if not np.any(individual_specific_util):#NO car option all zero, THIS SHOULD ONLY REALLY BE TRIGGERED RIGHT AT THE START
            #keep current car
            choice_index = index_current_cars_start + reduced_person_index
            self.zero_util_count += 1
        else:
            # Calculate the probability of choosing each vehicle
            if np.isnan(np.sum(individual_specific_util)):
                individual_specific_util = np.nan_to_num(individual_specific_util)
                
            sum_prob = np.sum(individual_specific_util)
            probability_choose = individual_specific_util / sum_prob
            choice_index = self.random_state_social_network.choice(len(available_and_current_vehicles_list), p=probability_choose)
            #choice_index = np.argmax(probability_choose)

        # Record the chosen vehicle
        vehicle_chosen = available_and_current_vehicles_list[choice_index]

        # Handle consequences of the choice
        if user.user_id != vehicle_chosen.owner_id:  # New vehicle, not currently owned
            # Transfer the user's current vehicle to the second-hand merchant, if any
            if isinstance(user.vehicle, PersonalCar):
                self.second_hand_merchant.spent += user.vehicle.cost_second_hand_merchant
                if user.vehicle.init_car or user.vehicle.cost_second_hand_merchant == 0:#ITS AN INITAL CAR WE DOTN WANT TO ALLOW THSOE TO BE SOLD 
                    user.vehicle.owner_id = -99#send to shadow realm
                    user.vehicle = None
                else:

                    if self.t_social_network > self.burn_in_second_hand_market:
                        
                        if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
                            self.second_hand_merchant_price_paid.append(user.vehicle.cost_second_hand_merchant)
                        
                        user.vehicle.owner_id = self.second_hand_merchant.id
                        self.second_hand_merchant.add_to_stock(user.vehicle)
                        user.vehicle = None

                    else:#VANISH CAR TO THE ABYSS
                        user.vehicle.owner_id = -99#send to shadow realm
                        user.vehicle = None
            
            if vehicle_chosen.owner_id == self.second_hand_merchant.id:# Buy a second-hand car
                #USED ADOPTION SUBSIDY OPTIMIZATION
                self.policy_distortion += self.used_rebate           

                #SET THE UTILITY TO 0 of that second hand car
                utilities_kappa[:, choice_index] = 0
                self.second_hand_merchant.remove_car(vehicle_chosen)
                vehicle_chosen.owner_id = user.user_id
                vehicle_chosen.scenario = "current_car"
                user.vehicle = vehicle_chosen
                user.vehicle.last_price_paid = user.vehicle.price
                self.second_hand_merchant.income += user.vehicle.price

                if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
                    self.car_prices_sold_second_hand.append(user.vehicle.price)
                    self.buy_second_hand_car+= 1
                    self.second_hand_bought += 1

            elif isinstance(vehicle_chosen, CarModel):  # Brand new car
                #ADOPTION SUBSIDY OPTIMIZATION
                self.policy_distortion += self.rebate    
            
                self.new_bought_vehicles.append(vehicle_chosen)#ADD NEW CAR TO NEW CAR LIST, used so can calculate the market concentration
                personalCar_id = self.id_generator.get_new_id()
                user.vehicle = PersonalCar(personalCar_id, vehicle_chosen.firm, user.user_id, vehicle_chosen.component_string, vehicle_chosen.parameters, vehicle_chosen.attributes_fitness, vehicle_chosen.price)
                
                if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
                    self.car_prices_sold_new.append(user.vehicle.price)
                    self.buy_new_car+=1
            else:
                raise(ValueError("invalid user transport behaviour"))
        else:
            if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
                self.keep_car +=1#KEEP CURRENT CAR

        # Update the age or timer of the chosen vehicle
        #if isinstance(vehicle_chosen, PersonalCar):
        user.vehicle.update_timer()

        return vehicle_chosen, user.vehicle, choice_index, utilities_kappa
    
##############################################################################################################################################################
    #CURRENT
    def generate_distances_current(self, current_vehicles, d_i_min_vec, beta_vec, gamma_vec):
        CV_vehicle_dict_vecs = self.gen_current_vehicle_dict_vecs(current_vehicles)
        
        d_i_t = np.maximum(
            d_i_min_vec,
            self.vectorised_optimal_distance_current(CV_vehicle_dict_vecs, beta_vec, gamma_vec)
        )  # Ensuring compatibility for element-wise comparison

        return d_i_t, CV_vehicle_dict_vecs
    
    def generate_utilities_current(self, vehicle_dict_vecs, d_i_t, beta_vec, gamma_vec):# -> NDArray:
        """
        Optimized utility calculation assuming individuals compare either their current car, with price adjustments only applied for those who do not own a car.
        """

        commuting_util_vec = self.vectorised_commuting_utility_current(vehicle_dict_vecs, d_i_t, beta_vec, gamma_vec)
        U_a_i_t_vec = commuting_util_vec / (self.r + (np.log(1 + vehicle_dict_vecs["delta"])) / (1 - self.alpha))
        CV_utilities_matrix = np.diag(U_a_i_t_vec)

        return  CV_utilities_matrix, U_a_i_t_vec

    def gen_current_vehicle_dict_vecs(self, list_vehicles):
        """Generate a dictionary of vehicle property arrays with improved performance."""

        # Extract properties using list comprehensions
        quality_a_t = np.array([vehicle.Quality_a_t for vehicle in list_vehicles])
        eff_omega_a_t = np.array([vehicle.Eff_omega_a_t for vehicle in list_vehicles])
        price = np.array([vehicle.price for vehicle in list_vehicles])
        delta = np.array([vehicle.delta for vehicle in list_vehicles])
        production_emissions = np.array([vehicle.emissions for vehicle in list_vehicles])
        transport_type = np.array([vehicle.transportType for vehicle in list_vehicles])
        last_price_paid = np.array([vehicle.last_price_paid for vehicle in list_vehicles])
        l_a_t = np.array([vehicle.L_a_t for vehicle in list_vehicles])
        fuel_cost_c = np.array([vehicle.fuel_cost_c for vehicle in list_vehicles])
        e_t = np.array([vehicle.e_t for vehicle in list_vehicles])

        # Create the dictionary directly with NumPy arrays
        vehicle_dict_vecs = {
            "Quality_a_t": quality_a_t,
            "Eff_omega_a_t": eff_omega_a_t,
            "price": price,
            "delta": delta,
            "production_emissions": production_emissions,
            "fuel_cost_c": fuel_cost_c,
            "e_t": e_t,
            "L_a_t": l_a_t,
            "transportType": transport_type,
            "last_price_paid": last_price_paid,
        }

        return vehicle_dict_vecs

    
    def filter_vehicle_dict_for_switchers(
        self,
        vehicle_dict_vecs: dict[str, np.ndarray],
        list_vehicles: list,
        switcher_indices: np.ndarray
    ) -> tuple[dict[str, np.ndarray], list]:
        """
        Filter an already-built 'vehicle_dict_vecs' so that it only includes
        rows corresponding to vehicles whose 'owner_id' is in 'switcher_indices'.

        Parameters
        ----------
        vehicle_dict_vecs : dict[str, np.ndarray]
            A dictionary of arrays (e.g. from gen_current_vehicle_dict_vecs),
            where each array has the same length = number_of_vehicles.
        list_vehicles : list
            The original list of vehicles in the same order used to build 'vehicle_dict_vecs'.
        switcher_indices : np.ndarray
            Array of user IDs who can switch (e.g., from np.where(switch_draws)[0]).

        Returns
        -------
        filtered_vehicle_dict : dict[str, np.ndarray]
            The same dictionary, but sliced down to only rows for switcher-owned vehicles.
        filtered_vehicles : list
            A filtered list of the actual vehicle objects corresponding to those owners.
        """

        # 1) Build an array of owner IDs for each vehicle in the same order used in vehicle_dict_vecs
        owner_ids = np.array([vehicle.owner_id for vehicle in list_vehicles], dtype=int)

        # 2) Create a boolean mask that is True if the owner is in switcher_indices
        #    (Note: np.isin will check for membership in switcher_indices)
        mask = np.isin(owner_ids, switcher_indices)

        # 3) Apply this mask to each array in vehicle_dict_vecs
        filtered_vehicle_dict = {}
        for key, arr in vehicle_dict_vecs.items():
            filtered_vehicle_dict[key] = arr[mask]

        # 4) Also build a filtered list of vehicle objects
        filtered_vehicles = [v for (v, keep) in zip(list_vehicles, mask) if keep]

        return filtered_vehicle_dict, filtered_vehicles

    def vectorised_optimal_distance_current(self, vehicle_dict_vecs, beta_vec, gamma_vec):
        """
        Only does it for the 1 car, calculate the optimal distance for each individual considering only their corresponding vehicle.
        Assumes each individual has one vehicle, aligned by index.
        """
        # Compute numerator for each individual-vehicle pair
        numerator = (
            self.alpha * vehicle_dict_vecs["Quality_a_t"] *
            (1 - vehicle_dict_vecs["delta"]) ** vehicle_dict_vecs["L_a_t"]
        )  # Shape: (num_individuals,)

        denominator = ((beta_vec/vehicle_dict_vecs["Eff_omega_a_t"]) * (vehicle_dict_vecs["fuel_cost_c"] + self.carbon_price*vehicle_dict_vecs["e_t"])) + ((gamma_vec/vehicle_dict_vecs["Eff_omega_a_t"]) * vehicle_dict_vecs["e_t"])

        # Calculate optimal distance vec for each individual-vehicle pair
        optimal_distance_vec = (numerator / denominator) ** (1 / (1 - self.alpha))

        return optimal_distance_vec  # Shape: (num_individuals,)

    def vectorised_commuting_utility_current(self, vehicle_dict_vecs, d_i_t, beta_vec, gamma_vec):
        """
        Only one car. Calculate the commuting utility for each individual considering only their corresponding vehicle.
        Assumes each individual has one vehicle, aligned by index.
        """

        # Compute cost component based on transport type, without broadcasting
        cost_component = ((beta_vec/ vehicle_dict_vecs["Eff_omega_a_t"]) * (vehicle_dict_vecs["fuel_cost_c"] + self.carbon_price*vehicle_dict_vecs["e_t"])) + ((gamma_vec/ vehicle_dict_vecs["Eff_omega_a_t"]) * vehicle_dict_vecs["e_t"])

        # Calculate the commuting utility for each individual-vehicle pair
        commuting_utility_vec = np.maximum(
            0,
            vehicle_dict_vecs["Quality_a_t"] * (1 - vehicle_dict_vecs["delta"]) ** vehicle_dict_vecs["L_a_t"] *
            (d_i_t ** self.alpha) - d_i_t * cost_component
        )  # Shape: (num_individuals,)

        return commuting_utility_vec  # Shape: (num_individuals,)


##############################################################################################################################################################

    def generate_utilities(self, beta_vec, gamma_vec, d_i_min_vec, second_hand_merchant_offer_price):

        # Generate utilities and distances for new cars
        NC_vehicle_dict_vecs = self.gen_vehicle_dict_vecs_new_cars(self.new_cars)
        NC_utilities, d_NC = self.vectorised_calculate_utility_cars(NC_vehicle_dict_vecs, beta_vec, gamma_vec, second_hand_merchant_offer_price, d_i_min_vec)

        # Calculate the total columns needed for utilities and distance matrices
        total_columns = NC_utilities.shape[1]

        if self.second_hand_cars:
            SH_vehicle_dict_vecs = self.gen_vehicle_dict_vecs_second_hand(self.second_hand_cars)
            SH_utilities, d_SH = self.vectorised_calculate_utility_second_hand_cars(SH_vehicle_dict_vecs, beta_vec, gamma_vec, second_hand_merchant_offer_price, d_i_min_vec)

            total_columns += SH_utilities.shape[1]

        # Preallocate arrays with the total required columns
        num_individuals_switchers = len(beta_vec)
        utilities_matrix = np.empty((num_individuals_switchers, total_columns))
        d_matrix = np.empty((num_individuals_switchers, total_columns))

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


    def gen_vehicle_dict_vecs_new_cars(self, list_vehicles):
        """Generate a dictionary of vehicle property arrays with improved performance."""

        # Extract properties using list comprehensions
        quality_a_t = np.array([vehicle.Quality_a_t for vehicle in list_vehicles])
        eff_omega_a_t = np.array([vehicle.Eff_omega_a_t for vehicle in list_vehicles])
        price = np.array([vehicle.price for vehicle in list_vehicles])
        delta = np.array([vehicle.delta for vehicle in list_vehicles])
        production_emissions = np.array([vehicle.emissions for vehicle in list_vehicles])
        fuel_cost_c = np.array([vehicle.fuel_cost_c for vehicle in list_vehicles])
        e_t = np.array([vehicle.e_t for vehicle in list_vehicles])
        l_a_t = np.array([vehicle.L_a_t for vehicle in list_vehicles])
        transport_type = np.array([vehicle.transportType for vehicle in list_vehicles])

        #have one value for rebate low and another which is the high rebate
        rebate_vec = np.where([vehicle.firm.firm_id in self.firm_rebate_exclusion_set for vehicle in list_vehicles], self.rebate_low, self.rebate)

        # Create the dictionary directly with NumPy arrays
        vehicle_dict_vecs = {
            "Quality_a_t": quality_a_t,
            "Eff_omega_a_t": eff_omega_a_t,
            "price": price,
            "delta": delta,
            "production_emissions": production_emissions,
            "fuel_cost_c": fuel_cost_c,
            "e_t": e_t,
            "L_a_t": l_a_t,
            "transportType": transport_type,
            "rebate": rebate_vec
        }

        return vehicle_dict_vecs

    def gen_vehicle_dict_vecs_second_hand(self, list_vehicles):
        """Generate a dictionary of vehicle property arrays with improved performance."""

        # Extract properties using list comprehensions
        quality_a_t = np.array([vehicle.Quality_a_t for vehicle in list_vehicles])
        eff_omega_a_t = np.array([vehicle.Eff_omega_a_t for vehicle in list_vehicles])
        price = np.array([vehicle.price for vehicle in list_vehicles])
        delta = np.array([vehicle.delta for vehicle in list_vehicles])
        production_emissions = np.array([vehicle.emissions for vehicle in list_vehicles])
        fuel_cost_c = np.array([vehicle.fuel_cost_c for vehicle in list_vehicles])
        e_t = np.array([vehicle.e_t for vehicle in list_vehicles])
        l_a_t = np.array([vehicle.L_a_t for vehicle in list_vehicles])
        transport_type = np.array([vehicle.transportType for vehicle in list_vehicles])

        #have one value for rebate low and another which is the high rebate
        used_rebate_vec = np.where([vehicle.firm.firm_id in self.used_firm_rebate_exclusion_set for vehicle in list_vehicles], self.used_rebate_low, self.used_rebate)

        # Create the dictionary directly with NumPy arrays
        vehicle_dict_vecs = {
            "Quality_a_t": quality_a_t,
            "Eff_omega_a_t": eff_omega_a_t,
            "price": price,
            "delta": delta,
            "production_emissions": production_emissions,
            "fuel_cost_c": fuel_cost_c,
            "e_t": e_t,
            "L_a_t": l_a_t,
            "transportType": transport_type,
            "used_rebate": used_rebate_vec
        }

        return vehicle_dict_vecs

    def vectorised_calculate_utility_second_hand_cars(self, vehicle_dict_vecs, beta_vec, gamma_vec, second_hand_merchant_offer_price, d_i_min_vec):
        # Compute shared base utility components
        d_i_t = np.maximum(d_i_min_vec[:, np.newaxis], self.vectorised_optimal_distance_cars(vehicle_dict_vecs, beta_vec, gamma_vec))

        commuting_util_matrix = self.vectorised_commuting_utility_cars(vehicle_dict_vecs, d_i_t, beta_vec, gamma_vec)
        #base_utility_matrix = commuting_util_matrix / (self.r + (np.log(1 + vehicle_dict_vecs["delta"])) / (1 - self.alpha))
        base_utility_matrix = commuting_util_matrix*(1+self.r)/((1+self.r) - (1 - vehicle_dict_vecs["delta"])**(1/(1 - self.alpha)))

        price_difference = np.where(
            vehicle_dict_vecs["transportType"][:, np.newaxis] == 3,  # Check transportType
            (vehicle_dict_vecs["price"][:, np.newaxis] - self.used_rebate - second_hand_merchant_offer_price),  # Apply rebate
            vehicle_dict_vecs["price"][:, np.newaxis] - second_hand_merchant_offer_price  # No rebate
        )

        # Calculate price and emissions adjustments once
        price_adjustment = np.multiply(beta_vec[:, np.newaxis], price_difference.T)
        
        # Use in-place modification to save memor
        U_a_i_t_matrix = base_utility_matrix - price_adjustment

        return U_a_i_t_matrix, d_i_t
    
    def vectorised_calculate_utility_cars(self, vehicle_dict_vecs, beta_vec, gamma_vec, second_hand_merchant_offer_price, d_i_min_vec):
        # Compute shared base utility components
        d_i_t = np.maximum(d_i_min_vec[:, np.newaxis], self.vectorised_optimal_distance_cars(vehicle_dict_vecs, beta_vec, gamma_vec))
        
        commuting_util_matrix = self.vectorised_commuting_utility_cars(vehicle_dict_vecs, d_i_t, beta_vec, gamma_vec)

        #base_utility_matrix = commuting_util_matrix / (self.r + (np.log(1 + vehicle_dict_vecs["delta"])/(1 - self.alpha)))
        base_utility_matrix = commuting_util_matrix*(1+self.r) / ((1+self.r) - (1 - vehicle_dict_vecs["delta"])**(1/(1 - self.alpha)))

        # Calculate price difference, applying rebate only for transportType == 3
        price_difference = np.where(
            vehicle_dict_vecs["transportType"][:, np.newaxis] == 3,  # Check transportType
            (vehicle_dict_vecs["price"][:, np.newaxis] - vehicle_dict_vecs["rebate"][:, np.newaxis] - second_hand_merchant_offer_price),  # Apply rebate
            vehicle_dict_vecs["price"][:, np.newaxis] - second_hand_merchant_offer_price  # No rebate
        )
    
        # Calculate price and emissions adjustments once
        price_adjustment = np.multiply(beta_vec[:, np.newaxis], price_difference.T)
        
        # Use in-place modification to save memory
        emissions_penalty = np.multiply(gamma_vec[:, np.newaxis], vehicle_dict_vecs["production_emissions"])
        U_a_i_t_matrix = base_utility_matrix - price_adjustment - emissions_penalty
        return U_a_i_t_matrix, d_i_t
    
    def vectorised_optimal_distance_cars(self, vehicle_dict_vecs, beta_vec, gamma_vec):
        """Distance of all cars for all agents"""
        # Compute numerator for all vehicles
        numerator = (
            self.alpha * vehicle_dict_vecs["Quality_a_t"] *
            ((1 - vehicle_dict_vecs["delta"]) ** vehicle_dict_vecs["L_a_t"])
        )  # Shape: (num_vehicles,)

        # Compute denominator for all individual-vehicle pairs using broadcasting
        # Reshape self.beta_vec and self.gamma_vec to (num_individuals, 1) for broadcasting across vehicles
        denominator = (
            ((beta_vec[:, np.newaxis]/vehicle_dict_vecs["Eff_omega_a_t"]) * (vehicle_dict_vecs["fuel_cost_c"] + self.carbon_price*vehicle_dict_vecs["e_t"])) +
            ((gamma_vec[:, np.newaxis]/ vehicle_dict_vecs["Eff_omega_a_t"]) * vehicle_dict_vecs["e_t"])
        )  # Shape: (num_individuals, num_vehicles)

        # Calculate optimal distance matrix for each individual-vehicle pair
        optimal_distance_matrix = (numerator / denominator) ** (1 / (1 - self.alpha))

        return optimal_distance_matrix  # Shape: (num_individuals, num_vehicles)

    def vectorised_commuting_utility_cars(self, vehicle_dict_vecs, d_i_t, beta_vec, gamma_vec):
        """utility of all cars for all agents"""
        # dit Shape: (num_individuals, num_vehicles)

        # Compute cost component based on transport type, with conditional operations
        cost_component = (beta_vec[:, np.newaxis]/ vehicle_dict_vecs["Eff_omega_a_t"]) * (vehicle_dict_vecs["fuel_cost_c"] + self.carbon_price*vehicle_dict_vecs["e_t"]) + ((gamma_vec[:, np.newaxis]/ vehicle_dict_vecs["Eff_omega_a_t"]) * vehicle_dict_vecs["e_t"])

        # Compute the commuting utility for each individual-vehicle pair

        commuting_utility_matrix = np.maximum(
            0,
            vehicle_dict_vecs["Quality_a_t"] * ((1 - vehicle_dict_vecs["delta"]) ** vehicle_dict_vecs["L_a_t"]) * (d_i_t ** self.alpha) - d_i_t * cost_component
        )  # Shape: (num_individuals, num_vehicles)

        return commuting_utility_matrix  # Shape: (num_individuals, num_vehicles)

    ####################################################################################################################################
    #REBATE POLICIES

    def add_firm_rebate_exclusion_set(self, firm_id):
        self.firm_rebate_exclusion_set.add(firm_id)

    def add_used_firm_rebate_exclusion_set(self, firm_id):
        self.used_firm_rebate_exclusion_set.add(firm_id)
    
    def remove_firm_rebate_exclusion_set(self, firm_id):
        self.firm_rebate_exclusion_set.remove(firm_id)

    def remove_used_firm_rebate_exclusion_set(self, firm_id):
        self.used_firm_rebate_exclusion_set.remove(firm_id)

    ################################################################################################################################
    
    #TIMESERIES
    def prep_counters(self):
        
        self.users_driving_emissions_vec = np.zeros(self.num_individuals)
        self.users_distance_vec = np.zeros(self.num_individuals)
        self.users_utility_vec  = np.zeros(self.num_individuals)
        self.users_transport_type_vec  = np.full((self.num_individuals), np.nan)

        self.users_distance_vec_EV = np.full((self.num_individuals), np.nan)
        self.users_distance_vec_ICE = np.full((self.num_individuals), np.nan)
        #variable to track
    
        self.total_driving_emissions = 0
        self.total_driving_emissions_ICE = 0
        self.total_driving_emissions_EV = 0
        self.total_production_emissions = 0
        self.total_utility = 0
        self.total_distance_travelled = 0
        self.total_distance_travelled_ICE = 0
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
    
    def update_counters(self, person_index, vehicle_chosen, driven_distance, utility):
        #ADD TOTAL EMISSIONS     
        
        self.users_driving_emissions_vec[person_index] = (driven_distance/vehicle_chosen.Eff_omega_a_t)*vehicle_chosen.e_t

        self.users_distance_vec[person_index] = driven_distance
        self.users_utility_vec[person_index] =  utility
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
            self.users_distance_vec_ICE[person_index] = driven_distance
            self.quality_vals_ICE.append(vehicle_chosen.Quality_a_t)#done here for efficiency
            self.efficiency_vals_ICE.append(vehicle_chosen.Eff_omega_a_t)
            self.production_cost_vals_ICE.append(vehicle_chosen.ProdCost_t)
            self.total_driving_emissions_ICE += car_driving_emissions 
            self.total_distance_travelled_ICE += driven_distance
            self.ICE_users += 1
        else:#EV
            self.users_distance_vec_EV[person_index] = driven_distance
            self.quality_vals_EV.append(vehicle_chosen.Quality_a_t)#done here for efficiency
            self.efficiency_vals_EV.append(vehicle_chosen.Eff_omega_a_t)
            self.production_cost_vals_EV.append(vehicle_chosen.ProdCost_t)
            self.total_driving_emissions_EV += car_driving_emissions 
            self.EV_users += 1
            
        self.total_utility +=  utility
        self.total_distance_travelled += driven_distance
            
        if isinstance(vehicle_chosen, PersonalCar):
            self.second_hand_users +=1
      
    def set_up_time_series_social_network(self):
        #"""
        self.history_driving_emissions = []
        self.history_driving_emissions_ICE = []
        self.history_driving_emissions_EV = []
        self.history_production_emissions = []
        self.history_total_emissions = []
        self.history_total_utility = []
        self.history_total_distance_driven = []
        self.history_total_distance_driven_ICE = []
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

        self.history_distance_individual_ICE = []
        self.history_distance_individual_EV = []
        self.history_count_buy = []

        #self.history_quality_index = []
        self.history_mean_price = []
        self.history_median_price = []
        
        self.history_car_prices_sold_new = []
        self.history_car_prices_sold_second_hand = []

        self.history_quality_users_raw_adjusted = []

        self.history_second_hand_merchant_price_paid = []

        self.history_zero_util_count = []
    def save_timeseries_data_social_network(self):


        self.update_EV_stock()

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

        self.history_distance_individual_ICE.append(self.users_distance_vec_ICE)
        self.history_distance_individual_EV.append(self.users_distance_vec_EV)

        self.history_new_ICE_cars_bought.append(self.new_ICE_cars_bought)
        self.history_new_EV_cars_bought.append(self.new_EV_cars_bought)

        #SUMS
        self.history_driving_emissions.append(self.total_driving_emissions)
        self.history_driving_emissions_ICE.append(self.total_driving_emissions_ICE)
        self.history_driving_emissions_EV.append(self.total_driving_emissions_EV)
        self.history_production_emissions.append(self.total_production_emissions)
        self.history_total_emissions.append(self.total_production_emissions + self.total_driving_emissions)
        self.history_total_utility.append(self.total_utility)
        self.history_total_distance_driven.append(self.total_distance_travelled)
        self.history_total_distance_driven_ICE.append(self.total_distance_travelled_ICE)
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

        self.history_zero_util_count.append(self.zero_util_count)
        #print("self.zero_util_count",self.zero_util_count)
    
    def update_emisisons(self, vehicle_chosen, driven_distance):      
        
        emissions_flow =   (driven_distance/vehicle_chosen.Eff_omega_a_t)*vehicle_chosen.e_t
        self.emissions_cumulative += emissions_flow
        self.emissions_flow += emissions_flow

        if vehicle_chosen.scenario == "new_car":  #if its a new car add emisisons
            self.emissions_cumulative += vehicle_chosen.emissions
            self.emissions_flow += vehicle_chosen.emissions

    def update_prices_and_emissions_intensity(self):
        #UPDATE EMMISSION AND PRICES, THIS WORKS FOR BOTH PRODUCTION AND INNOVATION
        for car in self.current_vehicles:
            if car.transportType == 2:#ICE
                car.fuel_cost_c = self.gas_price
            elif car.transportType == 3:
                car.fuel_cost_c = self.electricity_price
                car.e_t = self.electricity_emissions_intensity

####################################################################################################################################

    def update_EV_stock(self):
        #CALC EV STOCK
        self.EV_users_count = sum(1 if car.transportType == 3 else 0 for car in  self.current_vehicles)
        self.history_prop_EV.append(self.EV_users_count/self.num_individuals)

    def next_step(self, carbon_price, second_hand_cars,new_cars, gas_price, electricity_price, electricity_emissions_intensity, rebate, used_rebate, electricity_price_subsidy):
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
        self.electricity_price_subsidy = electricity_price_subsidy
        


        #update new tech and prices
        self.second_hand_cars, self.new_cars = second_hand_cars, new_cars
        self.all_vehicles_available = self.new_cars + self.second_hand_cars#ORDER IS VERY IMPORTANT

        self.update_prices_and_emissions_intensity()#UPDATE: the prices and emissions intensities of cars which are currently owned
        self.current_vehicles = self.update_VehicleUsers()
        
        self.EV_stock_prop = sum(1 if car.transportType == 3 else 0 for car in self.current_vehicles)/self.num_individuals#NEED FOR OPTIMISATION, measures the uptake EVS

        self.consider_ev_vec, self.ev_adoption_vec = self.calculate_ev_adoption(ev_type=3)#BASED ON CONSUMPTION PREVIOUS TIME STEP

        return self.consider_ev_vec, self.new_bought_vehicles #self.chosen_vehicles instead of self.current_vehicles as firms can count pofits
