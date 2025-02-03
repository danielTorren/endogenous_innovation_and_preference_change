# imports
import numpy as np
import networkx as nx
import numpy.typing as npt
import scipy.sparse as sp
import numpy as np
from package.model.personalCar import PersonalCar
from package.model.VehicleUser import VehicleUser
from package.model.carModel import CarModel
from sklearn.linear_model import LinearRegression

class Social_Network:
    def __init__(self, parameters_social_network: dict, parameters_vehicle_user: dict):
        """
        Constructs all the necessary attributes for the Social Network object.
        """
        self.t_social_network = 0

        self.policy_distortion = 0#NEED FOR OPTIMISATION, measures the distortion from policies
        
        self.rebate = parameters_social_network["rebate"]
        self.used_rebate = parameters_social_network["used_rebate"]

        self.rebate_calibration = parameters_social_network["rebate"]
        self.used_rebate_calibration = parameters_social_network["used_rebate"]

        self.prob_switch_car = parameters_social_network["prob_switch_car"]

        self.beta_vec = parameters_social_network["beta_vec"] 
        self.gamma_vec = parameters_social_network["gamma_vec"]
        self.chi_vec = parameters_social_network["chi_vec"]
        self.d_vec = parameters_social_network["d_vec"]

        self.delta = parameters_social_network["delta"]
        self.alpha = parameters_social_network["alpha"]
        self.nu_maxU = parameters_social_network["nu"]
        self.scrap_price = parameters_social_network["scrap_price"]

        self.beta_segment_vec = parameters_social_network["beta_segment_vals"] 
        self.gamma_segment_vec = parameters_social_network["gamma_segment_vals"] 

        self.beta_median = np.median(self.beta_vec )
        self.gamma_median = np.median(self.gamma_vec )


        # Initialize parameters
        self.parameters_vehicle_user = parameters_vehicle_user
        self.init_initial_state(parameters_social_network)

        self.emissions_cumulative = 0
        self.emissions_flow = 0
        if self.save_timeseries_data_state:
            self.emissions_flow_history = []

        self.init_network_settings(parameters_social_network)

        self.random_state_social_network = np.random.RandomState(parameters_social_network["social_network_seed"])

        self.mu =  parameters_vehicle_user["mu"]
        self.r = parameters_vehicle_user["r"]
        self.kappa = parameters_vehicle_user["kappa"]

        # Generate a list of indices and shuffle them
        self.user_indices = np.arange(self.num_individuals)

        # Efficient user list creation with list comprehension
        self.vehicleUsers_list = [VehicleUser(user_id=i) for i in range(self.num_individuals)]

        # Create network and calculate initial emissions
        self.adjacency_matrix, self.network = self.create_network()
        self.network_density = nx.density(self.network)

        self.weighting_matrix = self._normlize_matrix(self.adjacency_matrix)#INTRODUCE HOMOPHILY INTO THE NETWORK BY ASSORTING BY BETA WITHING GROUPS

        #Assume nobody adopts EV at the start, THIS MAY BE AN ISSUE
        self.consider_ev_vec = np.zeros(self.num_individuals).astype(np.int8)
 
        self.current_vehicles = self.set_init_cars_selection(parameters_social_network)
        
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
        self.network_structure_seed = int(round(parameters_social_network["network_structure_seed"]))
        self.prob_rewire = parameters_social_network["SW_prob_rewire"]
        self.SW_network_density_input = parameters_social_network["SW_network_density"]
        self.SW_prob_rewire = parameters_social_network["SW_prob_rewire"]
        self.SW_K = int(round((self.num_individuals - 1) * self.SW_network_density_input))

        
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
        
        self.new_bought_vehicles = []#track list of new vehicles
        user_vehicle_list = self.current_vehicles.copy()#assume most people keep their cars
        
        #########################################################
        #LIMIT CALCULATION FOR THOSE THAT DONT NEED TO SWTICH
        # 1) Determine which users can switch
        switch_draws = (self.random_state_social_network.rand(self.num_individuals) < self.prob_switch_car)
        switcher_indices = np.where(switch_draws)[0]  # e.g., [2, 5, 7, ...]
        num_switchers = len(switcher_indices)
        non_switcher_indices = np.where(~switch_draws)[0]  # e.g., [0, 1, 3, 4, 6, ...]


        if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
            self.emissions_flow = 0#MEASURIBNG THE FLOW
            self.zero_util_count = 0 #are people actually choosing or beign forced to choose the same
            self.num_switchers = num_switchers
            self.drive_min_num = 0

        self.sub_beta_vec = self.beta_vec[switcher_indices]
        self.sub_gamma_vec = self.gamma_vec[switcher_indices]
        self.sub_d_vec = self.d_vec[switcher_indices]

        self.second_hand_bought = 0#CAN REMOVE LATER ON IF I DONT ACTUALLY NEED TO COUNT

        # Generate current utilities and vehicles
        #Calculate the optimal distance for all user with current car, NEEDS TO BE DONE ALWAYS AND FOR ALL USERS
        CV_vehicle_dict_vecs = self.gen_current_vehicle_dict_vecs(self.current_vehicles)

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
            __, full_CV_utility_vec = self.generate_utilities_current(CV_vehicle_dict_vecs, self.beta_vec, self.gamma_vec, self.d_vec)


        for i, person_index in enumerate(non_switcher_indices):
            user = self.vehicleUsers_list[person_index]
            user.vehicle.update_timer()# Update the age or timer of the chosen vehicle
            # Handle consequences of the choice
            vehicle_chosen_index = index_current_cars_start + person_index
            driven_distance = self.d_vec[person_index]
            self.update_emisisons(user.vehicle, driven_distance)
            #NEEDED FOR OPTIMISATION, add the carbon price paid 
            #self.policy_distortion += (self.carbon_price*user.vehicle.e_t*driven_distance)/user.vehicle.Eff_omega_a_t#
            #NEEDED FOR OPTIMISATION, ELECTRICITY SUBSIDY
            if user.vehicle.transportType == 3:
                self.policy_distortion += (self.electricity_price_subsidy_dollars*driven_distance)/user.vehicle.Eff_omega_a_t
            else:
                #NEEDED FOR OPTIMISATION, add the carbon price paid 
                self.policy_distortion += (self.carbon_price*user.vehicle.e_t*driven_distance)/user.vehicle.Eff_omega_a_t

            if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):

                self.keep_car += 1
                utility = full_CV_utility_vec[person_index]
                self.update_counters(person_index, user.vehicle, driven_distance, utility) 

                
        ##################################################################
        #SWITCHERS

        # 2) Shuffle only that subset of user indices
        shuffle_indices = self.random_state_social_network.permutation(switcher_indices)

        self.NC_vehicle_dict_vecs = self.gen_vehicle_dict_vecs_new_cars(self.new_cars)

        #THIS CAN BE DONE FOR THE SUBSET OF USERS
        CV_filtered_vechicles_dicts, CV_filtered_vehicles = self.filter_vehicle_dict_for_switchers(CV_vehicle_dict_vecs, self.current_vehicles, switcher_indices)
        utilities_current_matrix, __ = self.generate_utilities_current(CV_filtered_vechicles_dicts, self.sub_beta_vec, self.sub_gamma_vec, self.sub_d_vec)

        self.second_hand_merchant_offer_price = self.calc_offer_prices_heursitic(self.NC_vehicle_dict_vecs, CV_filtered_vechicles_dicts, CV_filtered_vehicles)

        # pass those indices to generate_utilities
        utilities_buying_matrix_switchers, buying_vehicles_list = self.generate_utilities(self.sub_beta_vec, self.sub_gamma_vec, self.second_hand_merchant_offer_price, self.sub_d_vec)

        # Preallocate the final utilities and distance matrices
        total_columns = utilities_buying_matrix_switchers.shape[1] + utilities_current_matrix.shape[1]#number of cars which is new+secodn hand + current in the switchers

        self.utilities_matrix_switchers = np.empty((num_switchers, total_columns))

        # Assign matrices directly
        self.utilities_matrix_switchers[:, :utilities_buying_matrix_switchers.shape[1]] = utilities_buying_matrix_switchers
        self.utilities_matrix_switchers[:, utilities_buying_matrix_switchers.shape[1]:] = utilities_current_matrix

        # Combine the list of vehicles
        available_and_current_vehicles_list = buying_vehicles_list + CV_filtered_vehicles# ITS CURRENT VEHICLES AND NOT FILTERED VEHCILES AS THE SHUFFLING INDEX DOENST ACCOUNT FOR THE FILTERING

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

            driven_distance = self.d_vec[global_index]  # Use the reduced index for the matrix
            self.update_emisisons(vehicle_chosen, driven_distance)

            #CARBON TAX POLICY OPTIMISATION
            #self.policy_distortion += (self.carbon_price*user.vehicle.e_t*driven_distance)/user.vehicle.Eff_omega_a_t#NEEDED FOR OPTIMISATION of carbon tax
            #NEEDED FOR OPTIMISATION, ELECTRICITY SUBSIDY
            if user.vehicle.transportType == 3:
                self.policy_distortion += (self.electricity_price_subsidy_dollars*driven_distance)/user.vehicle.Eff_omega_a_t
            else:
                #NEEDED FOR OPTIMISATION, add the carbon price paid 
                self.policy_distortion += (self.carbon_price*user.vehicle.e_t*driven_distance)/user.vehicle.Eff_omega_a_t#
            
            if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
                utility = self.utilities_matrix_switchers[reduced_index][vehicle_chosen_index]
                self.update_counters(global_index, vehicle_chosen, driven_distance, utility)

        if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
            self.emissions_flow_history.append(self.emissions_flow)

        return user_vehicle_list

#####################################################################################################

    def calc_offer_prices_heursitic(self, vehicle_dict_vecs_new_cars, vehicle_dict_vecs_current_cars, current_cars):

        # Extract Quality, Efficiency, and Prices of first-hand cars
        first_hand_quality = vehicle_dict_vecs_new_cars["Quality_a_t"]
        first_hand_efficiency =  vehicle_dict_vecs_new_cars["Eff_omega_a_t"]
        first_hand_prices = vehicle_dict_vecs_new_cars["price"]

        # Extract Quality, Efficiency, and Age of second-hand cars
        second_hand_quality = vehicle_dict_vecs_current_cars["Quality_a_t"]
        second_hand_efficiency = vehicle_dict_vecs_current_cars["Eff_omega_a_t"]
        second_hand_ages = vehicle_dict_vecs_current_cars["L_a_t"]

        # Normalize Quality and Efficiency for both first-hand and second-hand cars
        all_quality = np.concatenate([first_hand_quality, second_hand_quality])
        all_efficiency = np.concatenate([first_hand_efficiency, second_hand_efficiency])

        quality_min, quality_max = np.min(all_quality), np.max(all_quality)
        efficiency_min, efficiency_max = np.min(all_efficiency), np.max(all_efficiency)

        normalized_first_hand_quality = (first_hand_quality - quality_min) / (quality_max - quality_min)
        normalized_first_hand_efficiency = (first_hand_efficiency - efficiency_min) / (efficiency_max - efficiency_min)

        normalized_second_hand_quality = (second_hand_quality - quality_min) / (quality_max - quality_min)
        normalized_second_hand_efficiency = (second_hand_efficiency - efficiency_min) / (efficiency_max - efficiency_min)

        # Compute proximity (Euclidean distance) for all second-hand cars to all first-hand cars
        diff_quality = normalized_second_hand_quality[:, np.newaxis] - normalized_first_hand_quality
        diff_efficiency = normalized_second_hand_efficiency[:, np.newaxis] - normalized_first_hand_efficiency

        distances = np.sqrt(diff_quality ** 2 + diff_efficiency ** 2)

        # Find the closest first-hand car for each second-hand car
        closest_idxs = np.argmin(distances, axis=1)

        # Get the prices of the closest first-hand cars
        closest_prices = first_hand_prices[closest_idxs]

        # Adjust prices based on car age and depreciation
        adjusted_prices = closest_prices * (1 - self.delta) ** second_hand_ages

        # Calculate offer prices
        offer_prices = adjusted_prices / (1 + self.mu)

        # Ensure offer prices are not below the scrap price
        offer_prices = np.maximum(offer_prices, self.scrap_price)

        # Assign prices back to second-hand car objects
        for i, car in enumerate(current_cars):
            car.price_second_hand_merchant = adjusted_prices[i]
            car.cost_second_hand_merchant = offer_prices[i]

        return offer_prices

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

    def masking_options_old(self, utilities_matrix, available_and_current_vehicles_list, consider_ev_vec):
        """If I don’t want something to be picked, the utilities kappa needs to be 0 at the output!"""

        # Replace `-np.inf` with a mask to avoid issues
        valid_utilities_mask = utilities_matrix != -np.inf  # Mask for valid (non -np.inf) entries

        # Initialize result matrix with zeros
        utilities_kappa = np.zeros_like(utilities_matrix)

        # Find valid rows (rows that have at least one non -inf value)
        valid_rows = np.any(valid_utilities_mask, axis=1)

        # Compute row-wise max only for valid rows
        #row_max_utilities = np.full((utilities_matrix.shape[0], 1), -np.inf)  # Default -inf
        row_max_utilities = np.max(utilities_matrix[valid_rows], axis=1, keepdims=True)
        self.nu_maxU = np.max(row_max_utilities[valid_rows])  # Store for reference

        # Compute safe exponentiation input (subtract row-wise max only for valid rows)
        exp_input = np.zeros_like(utilities_matrix)
        #exp_input[valid_rows] = self.kappa * (utilities_matrix[valid_rows])
        exp_input[valid_rows] = self.kappa * (utilities_matrix[valid_rows] - row_max_utilities[valid_rows])

        # Clip extreme values before exponentiation
        exp_input = np.clip(exp_input, -700, 700)  # Prevent overflow

        # Apply the exponentiation safely only on valid values
        utilities_kappa[valid_utilities_mask] = np.exp(exp_input[valid_utilities_mask])

        # Generate mask and apply
        combined_mask = self.gen_mask(available_and_current_vehicles_list, consider_ev_vec)
        utilities_kappa_masked = utilities_kappa * combined_mask

        return utilities_kappa_masked


    def masking_options(self, utilities_matrix, available_and_current_vehicles_list, consider_ev_vec):
        """Applies mask before exponentiation, setting masked-out values to -inf."""

        # Step 1: Generate the mask first
        combined_mask = self.gen_mask(available_and_current_vehicles_list, consider_ev_vec)

        # Step 2: Apply mask by setting masked-out values to -inf
        masked_utilities = np.where(combined_mask == 1, utilities_matrix, -np.inf)

        # Step 3: Identify valid (non -inf) utilities
        valid_utilities_mask = masked_utilities != -np.inf
        valid_rows = np.any(valid_utilities_mask, axis=1)  # Rows with at least one valid entry

        # Step 4: Compute row-wise max only for valid rows
        row_max_utilities = np.full((utilities_matrix.shape[0], 1), -np.inf)  # Default -inf
        row_max_utilities[valid_rows] = np.max(masked_utilities[valid_rows], axis=1, keepdims=True)
        self.nu_maxU = np.max(row_max_utilities[valid_rows])  # Store for reference

        # Step 5: Compute safe exponentiation input (subtract row max for stability)
        exp_input = np.zeros_like(utilities_matrix)
        exp_input[valid_rows] = self.kappa * (masked_utilities[valid_rows] - row_max_utilities[valid_rows])

        # Step 6: Clip extreme values to prevent overflow
        exp_input = np.clip(exp_input, -700, 700)

        # Step 7: Exponentiate, masked-out values (set to -inf) become zero
        utilities_kappa = np.zeros_like(utilities_matrix)
        utilities_kappa[valid_utilities_mask] = np.exp(exp_input[valid_utilities_mask])

        # Debugging Outputs
        #print("Combined mask:", combined_mask)
        #print("Row-wise max utilities:", row_max_utilities.flatten())
        #print("Min & Max exp_input:", np.min(exp_input), np.max(exp_input))
        #print("Row sums after exponentiation:", np.sum(utilities_kappa, axis=1))
        #print("Invalid (-inf) counts per row:", np.sum(~valid_utilities_mask, axis=1))

        return utilities_kappa



#########################################################################################################################################################
    #choosing vehicles
    def user_chooses(self, person_index, user, available_and_current_vehicles_list, utilities_kappa, reduced_person_index, index_current_cars_start ):
        # Select individual-specific utilities
        individual_specific_util_kappa = utilities_kappa[reduced_person_index]  
        
        #check for nans and set them to 0
        if np.isnan(individual_specific_util_kappa).any():
            individual_specific_util_kappa = np.nan_to_num(individual_specific_util_kappa)#Set all the nans to 0

        #SWICHING_CLAUSE
        if not np.any(individual_specific_util_kappa):#NO car option all zero, THIS SHOULD ONLY REALLY BE TRIGGERED RIGHT AT THE START
            #keep current car
            choice_index = index_current_cars_start + reduced_person_index
            if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
                self.zero_util_count += 1
        else:#at leat 1 non zero probability
            # Calculate the probability of choosing each vehicle              
            sum_U_kappa = np.sum(individual_specific_util_kappa)
            #print("sum_U_kappa", sum_U_kappa)

            probability_choose = individual_specific_util_kappa / sum_U_kappa

            choice_index = self.random_state_social_network.choice(len(available_and_current_vehicles_list), p=probability_choose)
            #choice_index = np.argmax(probability_choose)

        # Record the chosen vehicle
        vehicle_chosen = available_and_current_vehicles_list[choice_index]

        # Handle consequences of the choice
        if user.user_id != vehicle_chosen.owner_id:  # New vehicle, not currently owned
            # Transfer the user's current vehicle to the second-hand merchant, if any
            if isinstance(user.vehicle, PersonalCar):#YOU SELL YOUR CAR?
                if (user.vehicle.init_car) or (user.vehicle.cost_second_hand_merchant == self.scrap_price) or (self.t_social_network <= self.burn_in_second_hand_market):#ITS AN INITAL CAR WE DOTN WANT TO ALLOW THSOE TO BE SOLD 
                    user.vehicle.owner_id = -99#send to shadow realm
                    user.vehicle = None
                else:
                    if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
                        self.second_hand_merchant.spent += user.vehicle.cost_second_hand_merchant
                        self.second_hand_merchant_price_paid.append(user.vehicle.cost_second_hand_merchant)
                    
                    user.vehicle.owner_id = self.second_hand_merchant.id
                    self.second_hand_merchant.add_to_stock(user.vehicle)
                    user.vehicle = None
            
            if vehicle_chosen.owner_id == self.second_hand_merchant.id:# Buy a second-hand car
                #USED ADOPTION SUBSIDY OPTIMIZATION
                if vehicle_chosen.transportType == 3:
                    self.policy_distortion += self.used_rebate           

                ###########################################################################
                #DO NOT DELETE
                #SET THE UTILITY TO 0 of that second hand car
                utilities_kappa[:, choice_index] = 0#THIS STOPS OTHER INDIVIDUALS FROM BUYING SECOND HAND CAR THAT YOU BOUGHT, VERY IMPORANT LINE
                ###############################################################
                
                self.second_hand_merchant.remove_car(vehicle_chosen)
                vehicle_chosen.owner_id = user.user_id
                vehicle_chosen.scenario = "current_car"
                user.vehicle = vehicle_chosen
                user.vehicle.last_price_paid = user.vehicle.price
                self.second_hand_merchant.income += user.vehicle.price
                #self.second_hand_merchant.assets += (user.vehicle.price - user.vehicle.cost_second_hand_merchant)


                if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
                    self.car_prices_sold_second_hand.append(user.vehicle.price)
                    self.buy_second_hand_car+= 1
                    self.second_hand_bought += 1

            elif isinstance(vehicle_chosen, CarModel):  # Brand new car
                #ADOPTION SUBSIDY OPTIMIZATION
                if vehicle_chosen.transportType == 3:
                    self.policy_distortion += self.rebate    
            
                self.new_bought_vehicles.append(vehicle_chosen)#ADD NEW CAR TO NEW CAR LIST, used so can calculate the market concentration
                personalCar_id = self.id_generator.get_new_id()
                user.vehicle = PersonalCar(personalCar_id, vehicle_chosen.firm, user.user_id, vehicle_chosen.component_string, vehicle_chosen.parameters, vehicle_chosen.attributes_fitness, vehicle_chosen.price)
                
                if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
                    self.car_prices_sold_new.append(user.vehicle.price)
                    self.buy_new_car+=1
            else:
                raise(ValueError("invalid user transport behaviour"))
        else:#KEEP CAR
            if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
                self.keep_car +=1#KEEP CURRENT CAR

        # Update the age or timer of the chosen vehicle
        #if isinstance(vehicle_chosen, PersonalCar):
        user.vehicle.update_timer()

        return vehicle_chosen, user.vehicle, choice_index, utilities_kappa

##############################################################################################################################################################

##############################################################################################################################################################
    #CURRENT
    
    def generate_utilities_current(self, vehicle_dict_vecs, beta_vec, gamma_vec, d_vec):# -> NDArray:
        """
        Optimized utility calculation assuming individuals compare either their current car, with price adjustments only applied for those who do not own a car.
        """
        #U_a_i_t_vec = d_vec*((vehicle_dict_vecs["Quality_a_t"]*(1-self.delta)**vehicle_dict_vecs["L_a_t"])**self.alpha)*((1+self.r)/(self.r-self.delta)) - beta_vec*(d_vec*vehicle_dict_vecs["fuel_cost_c"]/(self.r*vehicle_dict_vecs["Eff_omega_a_t"])) - gamma_vec*(d_vec*vehicle_dict_vecs["e_t"]/(self.r*vehicle_dict_vecs["Eff_omega_a_t"]))
        U_a_i_t_vec = d_vec*((vehicle_dict_vecs["Quality_a_t"]*(1-self.delta)**vehicle_dict_vecs["L_a_t"])**self.alpha)*((1+self.r)/(self.r - (1 - self.delta)**self.alpha + 1)) - beta_vec*(d_vec*vehicle_dict_vecs["fuel_cost_c"]/(self.r*vehicle_dict_vecs["Eff_omega_a_t"])) - gamma_vec*(d_vec*vehicle_dict_vecs["e_t"]/(self.r*vehicle_dict_vecs["Eff_omega_a_t"]))
        #print("median U current",np.median(U_a_i_t_vec))
        # Initialize the matrix with -np.inf
        CV_utilities_matrix = np.full((len(U_a_i_t_vec), len(U_a_i_t_vec)), -np.inf)#its 
        
        # Set the diagonal values
        np.fill_diagonal(CV_utilities_matrix, U_a_i_t_vec)

        return  CV_utilities_matrix, U_a_i_t_vec
    
    def gen_current_vehicle_dict_vecs(self, list_vehicles):
        """Generate a dictionary of vehicle property arrays with improved performance."""

        # Extract properties using list comprehensions
        quality_a_t = np.array([vehicle.Quality_a_t for vehicle in list_vehicles])
        eff_omega_a_t = np.array([vehicle.Eff_omega_a_t for vehicle in list_vehicles])
        price = np.array([vehicle.price for vehicle in list_vehicles])
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
            "production_emissions": production_emissions,
            "fuel_cost_c": fuel_cost_c,
            "e_t": e_t,
            "L_a_t": l_a_t,
            "transportType": transport_type,
            "last_price_paid": last_price_paid,
        }

        return vehicle_dict_vecs

##############################################################################################################################################################

    def generate_utilities(self, beta_vec, gamma_vec, second_hand_merchant_offer_price, d_vec):


        # Generate utilities
        #self.NC_vehicle_dict_vecs = self.gen_vehicle_dict_vecs_new_cars(self.new_cars)
        NC_utilities = self.vectorised_calculate_utility_cars(self.NC_vehicle_dict_vecs, beta_vec, gamma_vec, second_hand_merchant_offer_price, d_vec)


        # Calculate the total columns needed for utilities 
        total_columns = NC_utilities.shape[1]

        if self.second_hand_cars:
            SH_vehicle_dict_vecs = self.gen_vehicle_dict_vecs_second_hand(self.second_hand_cars)
            SH_utilities = self.vectorised_calculate_utility_second_hand_cars(SH_vehicle_dict_vecs, beta_vec, gamma_vec, second_hand_merchant_offer_price,  d_vec)

            total_columns += SH_utilities.shape[1]

        # Preallocate arrays with the total required columns
        num_individuals_switchers = len(beta_vec)
        utilities_matrix = np.empty((num_individuals_switchers, total_columns))

        # Fill in preallocated arrays with submatrices
        col_idx = 0#start the counter at zero then increase as you set things up
        utilities_matrix[:, col_idx:col_idx + NC_utilities.shape[1]] = NC_utilities

        col_idx += NC_utilities.shape[1]

        if self.second_hand_cars:
            utilities_matrix[:, col_idx:col_idx + SH_utilities.shape[1]] = SH_utilities
            car_options = self.new_cars + self.second_hand_cars
        else:
            car_options = self.new_cars

        return utilities_matrix, car_options

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

    def gen_vehicle_dict_vecs_new_cars(self, list_vehicles):
        """Generate a dictionary of vehicle property arrays with improved performance."""

        # Extract properties using list comprehensions
        quality_a_t = np.array([vehicle.Quality_a_t for vehicle in list_vehicles])
        eff_omega_a_t = np.array([vehicle.Eff_omega_a_t for vehicle in list_vehicles])
        price = np.array([vehicle.price for vehicle in list_vehicles])
        production_emissions = np.array([vehicle.emissions for vehicle in list_vehicles])
        fuel_cost_c = np.array([vehicle.fuel_cost_c for vehicle in list_vehicles])
        e_t = np.array([vehicle.e_t for vehicle in list_vehicles])
        l_a_t = np.array([vehicle.L_a_t for vehicle in list_vehicles])
        transport_type = np.array([vehicle.transportType for vehicle in list_vehicles])

        rebate_vec = np.where(transport_type == 3, self.rebate_calibration + self.rebate, 0)

        # Create the dictionary directly with NumPy arrays
        vehicle_dict_vecs = {
            "Quality_a_t": quality_a_t,
            "Eff_omega_a_t": eff_omega_a_t,
            "price": price,
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
        production_emissions = np.array([vehicle.emissions for vehicle in list_vehicles])
        fuel_cost_c = np.array([vehicle.fuel_cost_c for vehicle in list_vehicles])
        e_t = np.array([vehicle.e_t for vehicle in list_vehicles])
        l_a_t = np.array([vehicle.L_a_t for vehicle in list_vehicles])
        transport_type = np.array([vehicle.transportType for vehicle in list_vehicles])

        used_rebate_vec = np.where(transport_type == 3, self.used_rebate_calibration + self.used_rebate, 0)

        # Create the dictionary directly with NumPy arrays
        vehicle_dict_vecs = {
            "Quality_a_t": quality_a_t,
            "Eff_omega_a_t": eff_omega_a_t,
            "price": price,
            "production_emissions": production_emissions,
            "fuel_cost_c": fuel_cost_c,
            "e_t": e_t,
            "L_a_t": l_a_t,
            "transportType": transport_type,
            "used_rebate": used_rebate_vec
        }

        return vehicle_dict_vecs

    def vectorised_calculate_utility_second_hand_cars(self, vehicle_dict_vecs, beta_vec, gamma_vec, second_hand_merchant_offer_price, d_vec):
        
        price_difference_raw = vehicle_dict_vecs["price"][:, np.newaxis] -  vehicle_dict_vecs["used_rebate"][:, np.newaxis]

        price_difference = np.maximum(0, price_difference_raw)- second_hand_merchant_offer_price

        price_difference_T = price_difference.T

        term_1 = d_vec[:, np.newaxis]*((vehicle_dict_vecs["Quality_a_t"]*(1-self.delta)**vehicle_dict_vecs["L_a_t"])**self.alpha)*((1+self.r)/(self.r - (1 - self.delta)**self.alpha + 1))
        term_2 = beta_vec[:, np.newaxis]*(d_vec[:, np.newaxis]*(vehicle_dict_vecs["fuel_cost_c"]/(self.r*vehicle_dict_vecs["Eff_omega_a_t"])) + price_difference_T)
        term_3 = gamma_vec[:, np.newaxis]*(d_vec[:, np.newaxis]*vehicle_dict_vecs["e_t"]/(self.r*vehicle_dict_vecs["Eff_omega_a_t"]))
        
        U_a_i_t_matrix = term_1 - term_2 - term_3
        return U_a_i_t_matrix
    
    def vectorised_calculate_utility_cars(self, vehicle_dict_vecs, beta_vec, gamma_vec, second_hand_merchant_offer_price, d_vec):

        # Calculate price difference, applying rebate only for transportType == 3 (included in rebate calculation)
        price_difference_raw = (vehicle_dict_vecs["price"][:, np.newaxis] - vehicle_dict_vecs["rebate"][:, np.newaxis])  # Apply rebate

        price_difference = np.maximum(0, price_difference_raw) - second_hand_merchant_offer_price

        price_difference_T = price_difference.T

        term_1 = d_vec[:, np.newaxis]*((vehicle_dict_vecs["Quality_a_t"]*(1-self.delta)**vehicle_dict_vecs["L_a_t"])**self.alpha)*((1+self.r)/(self.r - (1 - self.delta)**self.alpha + 1))
        term_2 = beta_vec[:, np.newaxis]*(d_vec[:, np.newaxis]*(vehicle_dict_vecs["fuel_cost_c"]/(self.r*vehicle_dict_vecs["Eff_omega_a_t"])) + price_difference_T)
        term_3 = gamma_vec[:, np.newaxis]*(d_vec[:, np.newaxis]*vehicle_dict_vecs["e_t"]/(self.r*vehicle_dict_vecs["Eff_omega_a_t"]) +  vehicle_dict_vecs["production_emissions"])
        
        U_a_i_t_matrix = term_1 - term_2 - term_3
        
        return U_a_i_t_matrix# Shape: (num_individuals, num_vehicles)
    
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
        self.total_production_emissions_ICE = 0
        self.total_production_emissions_EV = 0
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
        car_driving_emissions = (driven_distance/vehicle_chosen.Eff_omega_a_t)*vehicle_chosen.e_t 
        self.users_driving_emissions_vec[person_index] = car_driving_emissions

        self.users_distance_vec[person_index] = driven_distance
        self.users_utility_vec[person_index] =  utility
        self.users_transport_type_vec[person_index] = vehicle_chosen.transportType
        
        if vehicle_chosen.scenario == "new_car":  
            self.new_cars_bought +=1
            self.total_production_emissions += vehicle_chosen.emissions
            if vehicle_chosen.transportType == 2:
                self.new_ICE_cars_bought +=1
                self.total_production_emissions_ICE += vehicle_chosen.emissions
            else:
                self.new_EV_cars_bought +=1
                self.total_production_emissions_EV += vehicle_chosen.emissions
            
        
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

        self.history_utility_components = []
        self.history_max_index_segemnt = []
        self.history_prop_EV = []
        self.history_driving_emissions = []
        self.history_driving_emissions_ICE = []
        self.history_driving_emissions_EV = []
        self.history_production_emissions = []
        self.history_production_emissions_ICE = []
        self.history_production_emissions_EV = []
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

        self.history_mean_price_EV_ICE = []
        self.history_median_price_EV_ICE = []

        self.history_car_prices_sold_new = []
        self.history_car_prices_sold_second_hand = []

        self.history_quality_users_raw_adjusted = []

        self.history_second_hand_merchant_price_paid = []

        self.history_zero_util_count = []

        self.history_mean_efficiency_vals_EV = []
        self.history_mean_efficiency_vals_ICE = []
        self.history_drive_min_num = []

        self.history_mean_efficiency_vals = []

        self.history_second_hand_merchant_offer_price = []

    def save_timeseries_data_social_network(self):


        self.update_EV_stock()

        self.history_second_hand_merchant_offer_price.append(self.second_hand_merchant_offer_price)

        self.history_count_buy.append([self.keep_car, self.buy_new_car, self.buy_second_hand_car])

        self.history_drive_min_num.append(self.drive_min_num/self.num_individuals)


        mean_price_new = np.mean([vehicle.price for vehicle in self.new_cars])
        median_price_new = np.median([vehicle.price for vehicle in self.new_cars])

        prices_ICE = [vehicle.price for vehicle in self.new_cars if vehicle.transportType == 2]
        prices_EV = [vehicle.price for vehicle in self.new_cars if vehicle.transportType == 3]

        if prices_ICE:
            mean_price_new_ICE = np.mean(prices_ICE)
            median_price_new_ICE = np.median(prices_ICE)
        else:
            mean_price_new_ICE = np.nan
            median_price_new_ICE = np.nan
        
        if prices_EV:
            mean_price_new_EV = np.mean(prices_EV)
            median_price_new_EV = np.median(prices_EV)
        else:
            mean_price_new_EV = np.nan
            median_price_new_EV = np.nan

        if self.second_hand_cars:
            prices_second_hand_ICE = [vehicle.price for vehicle in self.second_hand_cars if vehicle.transportType == 2]
            prices_second_hand_EV = [vehicle.price for vehicle in self.second_hand_cars if vehicle.transportType == 3]

            if prices_second_hand_ICE:
                mean_price_second_hand_ICE = np.mean(prices_second_hand_ICE)
                median_price_second_hand_ICE = np.median(prices_second_hand_ICE)
            else:
                mean_price_second_hand_ICE = np.nan
                median_price_second_hand_ICE = np.nan
            
            if prices_EV:
                mean_price_second_hand_EV = np.mean(prices_second_hand_EV)
                median_price_second_hand_EV = np.median(prices_second_hand_EV)
            else:
                mean_price_second_hand_EV = np.nan
                median_price_second_hand_EV = np.nan

        else:
            mean_price_second_hand_ICE = np.nan
            median_price_second_hand_ICE = np.nan
            mean_price_second_hand_EV = np.nan
            median_price_second_hand_EV = np.nan

        if self.second_hand_cars:
            mean_price_second_hand = np.mean([vehicle.price for vehicle in self.second_hand_cars])
            median_price_second_hand = np.median([vehicle.price for vehicle in self.second_hand_cars])
        else:
            mean_price_second_hand = np.nan#NO SECOND HAND CARS
            median_price_second_hand = np.nan#NO SECOND HAND CARS

        self.history_mean_price.append([mean_price_new, mean_price_second_hand])
        self.history_median_price.append([median_price_new, median_price_second_hand])

        self.history_mean_price_EV_ICE.append([(mean_price_new_ICE, mean_price_new_EV), (mean_price_second_hand_ICE,mean_price_second_hand_EV)])
        self.history_median_price_EV_ICE.append([(median_price_new_ICE, median_price_new_EV), (median_price_second_hand_ICE,median_price_second_hand_EV)])

        self.history_driving_emissions_individual.append(self.users_driving_emissions_vec)
        
        self.history_distance_individual.append(self.users_distance_vec)
        self.history_utility_individual.append(self.users_utility_vec)
        self.history_transport_type_individual.append(self.users_transport_type_vec)

        self.history_distance_individual_ICE.append(self.users_distance_vec_ICE)
        self.history_distance_individual_EV.append(self.users_distance_vec_EV)

        self.history_new_ICE_cars_bought.append(self.new_ICE_cars_bought)
        self.history_new_EV_cars_bought.append(self.new_EV_cars_bought)


        #self.history_max_index_segemnt.append(self.max_index_segemnt)

        #SUMS
        self.history_driving_emissions.append(self.total_driving_emissions)
        self.history_driving_emissions_ICE.append(self.total_driving_emissions_ICE)
        self.history_driving_emissions_EV.append(self.total_driving_emissions_EV)
        self.history_production_emissions.append(self.total_production_emissions)
        self.history_production_emissions_ICE.append(self.total_production_emissions_ICE)
        self.history_production_emissions_EV.append(self.total_production_emissions_EV)
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

        self.history_mean_efficiency_vals.append(np.mean(self.efficiency_vals))

        if self.quality_vals_ICE:
            self.history_quality_ICE.append(self.quality_vals_ICE)
            self.history_efficiency_ICE.append(self.efficiency_vals_ICE)
            self.history_production_cost_ICE.append(self.production_cost_vals_ICE)

            self.history_mean_efficiency_vals_ICE.append(np.mean(self.efficiency_vals_ICE))
        else:
            self.history_quality_ICE.append([np.nan])
            self.history_efficiency_ICE.append([np.nan])
            self.history_production_cost_ICE.append([np.nan])
            self.history_mean_efficiency_vals_ICE.append([np.nan])

        if self.quality_vals_EV:
            self.history_quality_EV.append(self.quality_vals_EV)
            self.history_efficiency_EV.append(self.efficiency_vals_EV)
            self.history_production_cost_EV.append(self.production_cost_vals_EV)
            self.history_mean_efficiency_vals_EV.append(np.mean(self.efficiency_vals_EV))
        else:
            self.history_quality_EV.append([np.nan])
            self.history_efficiency_EV.append([np.nan])
            self.history_production_cost_EV.append([np.nan])
            self.history_mean_efficiency_vals_EV.append([np.nan])

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

        self.history_zero_util_count.append(self.zero_util_count/self.num_switchers)
    
    def update_emisisons(self, vehicle_chosen, driven_distance):      
        
        emissions_flow = (driven_distance/vehicle_chosen.Eff_omega_a_t)*vehicle_chosen.e_t
        self.emissions_cumulative += emissions_flow
        self.emissions_flow += emissions_flow

        if vehicle_chosen.scenario == "new_car":  #if its a new car add emisisons
            self.emissions_cumulative += vehicle_chosen.emissions
            self.emissions_flow += vehicle_chosen.emissions

    def update_EV_stock(self):
        #CALC EV STOCK
        self.EV_users_count = sum(1 if car.transportType == 3 else 0 for car in  self.current_vehicles)
        self.history_prop_EV.append(self.EV_users_count/self.num_individuals)

####################################################################################################################################

    def update_prices_and_emissions_intensity(self):
        #UPDATE EMMISSION AND PRICES, THIS WORKS FOR BOTH PRODUCTION AND INNOVATION
        for car in self.current_vehicles:
            if car.transportType == 2:#ICE
                car.fuel_cost_c = self.gas_price
            elif car.transportType == 3:
                car.fuel_cost_c = self.electricity_price
                car.e_t = self.electricity_emissions_intensity

    def next_step(self, carbon_price, second_hand_cars,new_cars, gas_price, electricity_price, electricity_emissions_intensity, rebate, used_rebate, electricity_price_subsidy_dollars, rebate_calibration, used_rebate_calibration):
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
        self.rebate_calibration = rebate_calibration
        self.used_rebate_calibration = used_rebate_calibration
        self.electricity_price_subsidy_dollars = electricity_price_subsidy_dollars

        #update new tech and prices
        self.second_hand_cars, self.new_cars = second_hand_cars, new_cars
        self.all_vehicles_available = self.new_cars + self.second_hand_cars#ORDER IS VERY IMPORTANT

        self.update_prices_and_emissions_intensity()#UPDATE: the prices and emissions intensities of cars which are currently owned
        self.current_vehicles = self.update_VehicleUsers()
        
        self.consider_ev_vec, self.ev_adoption_vec = self.calculate_ev_adoption(ev_type=3)#BASED ON CONSUMPTION PREVIOUS TIME STEP

        
        return self.consider_ev_vec, self.new_bought_vehicles, self.nu_maxU #self.chosen_vehicles instead of self.current_vehicles as firms can count pofits
