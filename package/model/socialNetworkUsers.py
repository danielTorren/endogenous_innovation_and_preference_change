import numpy as np
import networkx as nx
import numpy.typing as npt
import scipy.sparse as sp
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
        self.policy_distortion = 0  # For policy optimization

        self.rebate = parameters_social_network["rebate"]
        self.used_rebate = parameters_social_network["used_rebate"]
        self.rebate_calibration = parameters_social_network["rebate"]
        self.used_rebate_calibration = parameters_social_network["used_rebate"]

        self.prob_switch_car = parameters_social_network["prob_switch_car"]

        self.beta_vec = parameters_social_network["beta_vec"] 
        self.gamma_vec = parameters_social_network["gamma_vec"]
        self.chi_vec = parameters_social_network["chi_vec"]
        self.d_vec = parameters_social_network["d_vec"]

        self.alpha = parameters_social_network["alpha"]
        self.scrap_price = parameters_social_network["scrap_price"]

        self.beta_segment_vec = parameters_social_network["beta_segment_vals"] 
        self.gamma_segment_vec = parameters_social_network["gamma_segment_vals"]
        self.beta_median = np.median(self.beta_vec)
        self.gamma_median = np.median(self.gamma_vec)

        self.parameters_vehicle_user = parameters_vehicle_user
        self.init_initial_state(parameters_social_network)

        self.emissions_cumulative = 0
        self.emissions_flow = 0

        if self.save_timeseries_data_state:
            self.emissions_flow_history = []

        self.init_network_settings(parameters_social_network)
        self.random_state_social_network = np.random.RandomState(parameters_social_network["social_network_seed"])

        self.mu = parameters_vehicle_user["mu"]
        self.r = parameters_vehicle_user["r"]
        self.kappa = parameters_vehicle_user["kappa"]

        self.user_indices = np.arange(self.num_individuals)
        self.vehicleUsers_list = [VehicleUser(user_id=i) for i in range(self.num_individuals)]

        # Create network
        self.adjacency_matrix, self.network = self.create_network()
        self.network_density = nx.density(self.network)
        self.weighting_matrix = self._normlize_matrix(self.adjacency_matrix)

        # Initially, no one considers EV
        self.consider_ev_vec = np.zeros(self.num_individuals, dtype=np.int8)

        # Assign "old cars" to users
        self.current_vehicles = self.set_init_cars_selection(parameters_social_network)
        
        # Calculate initial EV adoption
        self.consider_ev_vec, self.ev_adoption_vec = self.calculate_ev_adoption(ev_type=3)

    # --------------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------------

    def init_initial_state(self, parameters_social_network):
        self.num_individuals = int(round(parameters_social_network["num_individuals"]))
        self.id_generator = parameters_social_network["IDGenerator_firms"]
        self.second_hand_merchant = parameters_social_network["second_hand_merchant"]
        self.burn_in_second_hand_market = self.second_hand_merchant.burn_in_second_hand_market
        self.save_timeseries_data_state = parameters_social_network["save_timeseries_data_state"]
        self.compression_factor_state = parameters_social_network["compression_factor_state"]
        self.carbon_price = parameters_social_network["carbon_price"]

    def init_network_settings(self, parameters_social_network):
        self.network_structure_seed = int(round(parameters_social_network["network_structure_seed"]))
        self.prob_rewire = parameters_social_network["SW_prob_rewire"]
        self.SW_network_density_input = parameters_social_network["SW_network_density"]
        self.SW_prob_rewire = parameters_social_network["SW_prob_rewire"]
        self.SW_K = int(round((self.num_individuals - 1) * self.SW_network_density_input))

    def set_init_cars_selection(self, parameters_social_network):
        old_cars = parameters_social_network["old_cars"]
        for i, car in enumerate(old_cars):
            self.vehicleUsers_list[i].vehicle = car
            car.owner_id = i
        return [user.vehicle for user in self.vehicleUsers_list]

    # --------------------------------------------------------------------------------
    # Network creation and row-normalization
    # --------------------------------------------------------------------------------

    def create_network(self):
        """
        Create a Watts-Strogatz small-world graph.
        """
        network = nx.watts_strogatz_graph(
            n=self.num_individuals,
            k=self.SW_K,
            p=self.prob_rewire,
            seed=self.network_structure_seed
        )
        adjacency_matrix = nx.to_numpy_array(network)
        return adjacency_matrix, network

    def _normlize_matrix(self, matrix: np.ndarray) -> sp.csr_matrix:
        """
        Row-normalize a dense matrix -> CSR.
        """
        sparse_mat = sp.csr_matrix(matrix)
        row_sums = np.array(sparse_mat.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        inv_row_sums = 1.0 / row_sums
        diag_mat = sp.diags(inv_row_sums)
        return diag_mat.dot(sparse_mat)

    # --------------------------------------------------------------------------------
    # EV adoption
    # --------------------------------------------------------------------------------

    def calculate_ev_adoption(self, ev_type=3):
        """
        For each user, see how many neighbors have EV; if >= chi_i, user considers EV.
        """
        self.vehicle_type_vec = np.array([u.vehicle.transportType for u in self.vehicleUsers_list])
        ev_adoption_vec = (self.vehicle_type_vec == ev_type).astype(int)
        ev_neighbors = self.weighting_matrix.dot(ev_adoption_vec)
        consider_ev_vec = (ev_neighbors >= self.chi_vec).astype(np.int8)
        return consider_ev_vec, ev_adoption_vec

    # --------------------------------------------------------------------------------
    # update_VehicleUsers: the main step
    # --------------------------------------------------------------------------------

    def update_VehicleUsers(self):
        """
        A vectorized approach that skips EV utility for those who cannot consider EV,
        and avoids unnecessary calculations for non-switchers. 
        Also incorporates second-hand merchant offer prices via calc_offer_prices_heursitic.
        """
        self.new_bought_vehicles = []
        self.second_hand_bought = 0

        # Who switches?
        switch_draws = (self.random_state_social_network.rand(self.num_individuals) < self.prob_switch_car)
        switcher_indices = np.where(switch_draws)[0]
        non_switcher_indices = np.where(~switch_draws)[0]

        # Timeseries counters for non-switchers
        if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
            self.prep_counters()
            # For logging non-switchers' current-car utility
            cv_dict_all = self.gen_current_vehicle_dict_vecs(self.current_vehicles)
            _, self.full_CV_utility_vec = self.generate_utilities_current(
                cv_dict_all, self.beta_vec, self.gamma_vec, self.d_vec
            )
        else:
            self.full_CV_utility_vec = None

        # Non-switchers keep their cars
        self.handle_non_switchers(non_switcher_indices)
        if len(switcher_indices) == 0:
            if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
                self.emissions_flow_history.append(self.emissions_flow)
            return

        # ----------------------------------------------------------------------
        # PART 1: Prepare second-hand merchant offer prices for switchers' current cars
        # ----------------------------------------------------------------------
        new_cars_dict = self.gen_vehicle_dict_vecs_new_cars(self.new_cars)
        current_vehicles_switchers = [self.vehicleUsers_list[i].vehicle for i in switcher_indices]
        
        cv_dict_switchers = self.gen_current_vehicle_dict_vecs(current_vehicles_switchers)
        self.calc_offer_prices_heursitic(new_cars_dict, cv_dict_switchers, current_vehicles_switchers)

        # ----------------------------------------------------------------------
        # PART 2: Partition the cars and prepare options
        # ----------------------------------------------------------------------
        ICE_new = [c for c in self.new_cars if c.transportType == 2]
        EV_new = [c for c in self.new_cars if c.transportType == 3]
        ICE_sh = [c for c in self.second_hand_cars if c.transportType == 2]
        EV_sh = [c for c in self.second_hand_cars if c.transportType == 3]

        ice_new_len = len(ICE_new)
        ev_new_len = len(EV_new)
        ice_sh_len = len(ICE_sh)
        ev_sh_len = len(EV_sh)

        # Total columns now only include switchers' current cars
        total_cols = ice_new_len + ev_new_len + ice_sh_len + ev_sh_len + len(switcher_indices)
        
        # Build the options list for switchers
        self.all_options_list = ICE_new + EV_new + ICE_sh + EV_sh + current_vehicles_switchers

        # Utility matrix only for switchers
        utilities_matrix = np.full((len(switcher_indices), total_cols), -np.inf)

        # ----------------------------------------------------------------------
        # PART 3: Fill blocks of the utility matrix for switchers only
        # ----------------------------------------------------------------------

        # (A) ICE_new => columns [0 : ice_new_len]
        if ice_new_len > 0:
            dict_ice_new = self.gen_vehicle_dict_vecs_new_cars(ICE_new)
            util_ice_new = self.calc_utility_new_cars_global(dict_ice_new)
            utilities_matrix[:, :ice_new_len] = util_ice_new[switcher_indices, :]

        # (B) EV_new => columns [ice_new_len : ice_new_len + ev_new_len]
        ev_slice = slice(ice_new_len, ice_new_len + ev_new_len)
        if ev_new_len > 0:
            dict_ev_new = self.gen_vehicle_dict_vecs_new_cars(EV_new)
            util_ev_new = self.calc_utility_new_cars_global(dict_ev_new)
            utilities_matrix[:, ev_slice] = util_ev_new[switcher_indices, :]
            # Mask EVs for those who do not consider them
            nce_mask = (self.consider_ev_vec[switcher_indices] == 0)
            utilities_matrix[nce_mask, ev_slice] = -np.inf

        # (C) ICE_sh => columns [start : end]
        ice_sh_start = ice_new_len + ev_new_len
        ice_sh_end = ice_sh_start + ice_sh_len
        if ice_sh_len > 0:
            dict_ice_sh = self.gen_vehicle_dict_vecs_second_hand(ICE_sh)
            util_ice_sh = self.calc_utility_second_hand_global(dict_ice_sh, ICE_sh)
            utilities_matrix[:, ice_sh_start:ice_sh_end] = util_ice_sh[switcher_indices, :]

        # (D) EV_sh => columns [start : end], skip for NCE
        ev_sh_start = ice_sh_end
        ev_sh_end = ev_sh_start + ev_sh_len
        if ev_sh_len > 0:
            dict_ev_sh = self.gen_vehicle_dict_vecs_second_hand(EV_sh)
            util_ev_sh = self.calc_utility_second_hand_global(dict_ev_sh, EV_sh)
            utilities_matrix[:, ev_sh_start:ev_sh_end] = util_ev_sh[switcher_indices, :]
            nce_mask = (self.consider_ev_vec[switcher_indices] == 0)
            utilities_matrix[nce_mask, ev_sh_start:ev_sh_end] = -np.inf

        # (E) Current vehicles diagonal => last block
        cv_start = ev_sh_end
        cv_end = cv_start + len(switcher_indices)
        #cv_dict_switchers = self.gen_current_vehicle_dict_vecs(current_vehicles_switchers)
        sub_beta_vec = self.beta_vec[switcher_indices]
        sub_gamma_vec = self.gamma_vec[switcher_indices]
        sub_d_vec = self.d_vec[switcher_indices]
        util_cv_matrix, _ = self.generate_utilities_current(cv_dict_switchers, sub_beta_vec, sub_gamma_vec, sub_d_vec)
        utilities_matrix[:, cv_start:cv_end] = util_cv_matrix

        # ----------------------------------------------------------------------
        # PART 4: Each switcher chooses a vehicle in random order
        # ----------------------------------------------------------------------
        shuffle_indices = self.random_state_social_network.permutation(len(switcher_indices))

        for i, idx in enumerate(shuffle_indices):
            row = utilities_matrix[i]
            row_max = row.max()
            scaled = self.kappa * (row - row_max)
            np.clip(scaled, -700, 700, out=scaled)
            exp_vec = np.exp(scaled)
            sum_exp = exp_vec.sum()
            if sum_exp <= 0:
                # Corner case: all -inf
                choice_col = self.random_state_social_network.choice(total_cols)
            else:
                probs = exp_vec / sum_exp
                choice_col = self.random_state_social_network.choice(total_cols, p=probs)

            chosen_vehicle = self.all_options_list[choice_col]
            user_idx = switcher_indices[idx]
            user = self.vehicleUsers_list[user_idx]

            # If it's user’s own current car => keep
            if chosen_vehicle.owner_id == user.user_id:
                if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
                    self.keep_car += 1
            else:
                # If second-hand from merchant
                if chosen_vehicle.owner_id == self.second_hand_merchant.id:
                    self.handle_buy_second_hand_car(user, chosen_vehicle)
                    # Ensure no other user selects this car
                    utilities_matrix[:, choice_col] = -np.inf
                elif isinstance(chosen_vehicle, CarModel):
                    self.handle_buy_new_car(user, chosen_vehicle)
                else:
                    pass  # Edge case handling

            # Age the new/kept vehicle
            user.vehicle.update_timer_L_a_t()
            # Add policy distortion
            self.add_policy_distortion(user_idx)

            # Timeseries tracking
            if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
                utility_chosen = row[choice_col]
                dist = self.d_vec[user_idx]
                self.update_counters(user_idx, user.vehicle, dist, utility_chosen)

        # Final emissions update for this timestep
        if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
            self.emissions_flow_history.append(self.emissions_flow)

    # --------------------------------------------------------------------------------
    # Non-switchers
    # --------------------------------------------------------------------------------

    def handle_non_switchers(self, non_switcher_indices):
        """
        Non-switchers keep their current car and pay policy distortion.
        """
        for person_index in non_switcher_indices:
            user = self.vehicleUsers_list[person_index]
            user.vehicle.update_timer_L_a_t()
            # Distortion
            dd = self.d_vec[person_index]
            if user.vehicle.transportType == 3:
                self.policy_distortion += (self.electricity_price_subsidy_dollars * dd) / user.vehicle.Eff_omega_a_t
            else:
                self.policy_distortion += (self.carbon_price * user.vehicle.e_t * dd) / user.vehicle.Eff_omega_a_t
            # timeseries
            if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
                self.keep_car += 1
                if self.full_CV_utility_vec is not None:
                    utility_val = self.full_CV_utility_vec[person_index]
                else:
                    utility_val = 0.0
                self.update_counters(person_index, user.vehicle, dd, utility_val)

    # --------------------------------------------------------------------------------
    # Purchasing helper methods
    # --------------------------------------------------------------------------------

    def handle_buy_new_car(self, user, chosen_vehicle):
        self.sell_previous_car(user)
        if chosen_vehicle.transportType == 3:
            self.policy_distortion += self.rebate
        # Production emissions
        self.emissions_flow += chosen_vehicle.emissions
        self.new_bought_vehicles.append(chosen_vehicle)

        # Make a PersonalCar
        pc_id = self.id_generator.get_new_id()
        user.vehicle = PersonalCar(
            pc_id,
            chosen_vehicle.firm,
            user.user_id,
            chosen_vehicle.component_string,
            chosen_vehicle.parameters,
            chosen_vehicle.attributes_fitness,
            chosen_vehicle.price
        )
        if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
            self.buy_new_car += 1
            self.car_prices_sold_new.append(user.vehicle.price)

    def handle_buy_second_hand_car(self, user, chosen_vehicle):
        self.sell_previous_car(user)
        if chosen_vehicle.transportType == 3:
            self.policy_distortion += self.used_rebate
        chosen_vehicle.owner_id = user.user_id
        chosen_vehicle.scenario = "current_car"
        user.vehicle = chosen_vehicle
        self.second_hand_merchant.remove_car(chosen_vehicle)
        self.second_hand_merchant.income += chosen_vehicle.price

        self.second_hand_bought += 1
        if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
            self.buy_second_hand_car += 1
            self.car_prices_sold_second_hand.append(user.vehicle.price)

    def sell_previous_car(self, user):
        if isinstance(user.vehicle, PersonalCar):
            if (user.vehicle.init_car
                or (user.vehicle.cost_second_hand_merchant == self.scrap_price)
                or (self.t_social_network <= self.burn_in_second_hand_market)):
                user.vehicle.owner_id = -99
                user.vehicle = None
            else:
                if self.save_timeseries_data_state and (self.t_social_network % self.compression_factor_state == 0):
                    self.second_hand_merchant.spent += user.vehicle.cost_second_hand_merchant
                    self.second_hand_merchant_price_paid.append(user.vehicle.cost_second_hand_merchant)
                user.vehicle.owner_id = self.second_hand_merchant.id
                self.second_hand_merchant.add_to_stock(user.vehicle)
                user.vehicle = None

    def add_policy_distortion(self, user_index):
        user = self.vehicleUsers_list[user_index]
        d_val = self.d_vec[user_index]
        if user.vehicle.transportType == 3:  # EV
            self.policy_distortion += (self.electricity_price_subsidy_dollars * d_val) / user.vehicle.Eff_omega_a_t
        else:  # ICE
            self.policy_distortion += (self.carbon_price * user.vehicle.e_t * d_val) / user.vehicle.Eff_omega_a_t

    # --------------------------------------------------------------------------------
    # next_step and price updates
    # --------------------------------------------------------------------------------

    def next_step(self, carbon_price, second_hand_cars, new_cars,
                  gas_price, electricity_price, electricity_emissions_intensity,
                  rebate, used_rebate, electricity_price_subsidy_dollars,
                  rebate_calibration, used_rebate_calibration):
        """
        Advance the simulation one step.
        """
        self.t_social_network += 1
        self.emissions_flow = 0

        self.carbon_price = carbon_price
        self.gas_price = gas_price
        self.electricity_price = electricity_price
        self.electricity_emissions_intensity = electricity_emissions_intensity
        self.rebate = rebate
        self.used_rebate = used_rebate
        self.rebate_calibration = rebate_calibration
        self.used_rebate_calibration = used_rebate_calibration
        self.electricity_price_subsidy_dollars = electricity_price_subsidy_dollars

        self.second_hand_cars, self.new_cars = second_hand_cars, new_cars
        self.all_vehicles_available = self.new_cars + self.second_hand_cars

        # update existing owned cars with new fuel cost and e_t
        self.update_prices_and_emissions_intensity()
        self.update_VehicleUsers()

        self.current_vehicles = [u.vehicle for u in self.vehicleUsers_list]
        self.batch_update_emissions()
        self.emissions_cumulative += self.emissions_flow

        self.consider_ev_vec, self.ev_adoption_vec = self.calculate_ev_adoption(ev_type=3)
        return self.consider_ev_vec, self.new_bought_vehicles

    def update_prices_and_emissions_intensity(self):
        for car in self.current_vehicles:
            if car.transportType == 2:
                car.fuel_cost_c = self.gas_price
            elif car.transportType == 3:
                car.fuel_cost_c = self.electricity_price
                car.e_t = self.electricity_emissions_intensity

    def batch_update_emissions(self):
        # Vectorized summation of driving emissions
        eff_arr = np.array([v.Eff_omega_a_t for v in self.current_vehicles])
        e_t_arr = np.array([v.e_t for v in self.current_vehicles])
        dist = self.d_vec
        emis_flow = (dist / eff_arr) * e_t_arr
        self.emissions_flow += emis_flow.sum()

    # --------------------------------------------------------------------------------
    # Utility calculations (unchanged fundamentals)
    # --------------------------------------------------------------------------------

    def generate_utilities_current(self, vehicle_dict_vecs, beta_vec, gamma_vec, d_vec):
        """
        NxN matrix with diagonal as the utility for (user i -> user i's car).
        """
        U_a_i_t_vec = d_vec * (
            (vehicle_dict_vecs["Quality_a_t"] * (1 - vehicle_dict_vecs["delta"])**vehicle_dict_vecs["L_a_t"])**self.alpha
        ) * ((1 + self.r) / (self.r - (1 - vehicle_dict_vecs["delta"])**self.alpha + 1)) \
        - beta_vec * (
            d_vec * vehicle_dict_vecs["fuel_cost_c"] * ((1 + self.r) / (self.r * vehicle_dict_vecs["Eff_omega_a_t"]))
        ) - gamma_vec * (
            d_vec * vehicle_dict_vecs["e_t"] * ((1 + self.r) / (self.r * vehicle_dict_vecs["Eff_omega_a_t"]))
        )

        CV_utilities_matrix = np.full((len(U_a_i_t_vec), len(U_a_i_t_vec)), -np.inf)

        np.fill_diagonal(CV_utilities_matrix, U_a_i_t_vec)
        return CV_utilities_matrix, U_a_i_t_vec

    def gen_current_vehicle_dict_vecs(self, list_vehicles):
        """
        Dictionary of arrays for each user's current vehicle (index i => user i).
        """
        return {
            "Quality_a_t": np.array([v.Quality_a_t for v in list_vehicles]),
            "Eff_omega_a_t": np.array([v.Eff_omega_a_t for v in list_vehicles]),
            "fuel_cost_c": np.array([v.fuel_cost_c for v in list_vehicles]),
            "e_t": np.array([v.e_t for v in list_vehicles]),
            "L_a_t": np.array([v.L_a_t for v in list_vehicles]),
            "transportType": np.array([v.transportType for v in list_vehicles]),
            "delta": np.array([v.delta for v in list_vehicles]),
            "delta_P": np.array([v.delta_P for v in list_vehicles])
        }

    def gen_vehicle_dict_vecs_new_cars(self, list_vehicles):
        """
        Dictionary for brand-new cars.
        """
        arr_t = np.array([v.transportType for v in list_vehicles])
        rebate_vec = np.where(arr_t == 3, self.rebate_calibration + self.rebate, 0)
        return {
            "Quality_a_t": np.array([v.Quality_a_t for v in list_vehicles]),
            "Eff_omega_a_t": np.array([v.Eff_omega_a_t for v in list_vehicles]),
            "price": np.array([v.price for v in list_vehicles]),
            "production_emissions": np.array([v.emissions for v in list_vehicles]),
            "fuel_cost_c": np.array([v.fuel_cost_c for v in list_vehicles]),
            "e_t": np.array([v.e_t for v in list_vehicles]),
            "transportType": arr_t,
            "rebate": rebate_vec,
            "delta": np.array([v.delta for v in list_vehicles])
        }

    def gen_vehicle_dict_vecs_second_hand(self, list_vehicles):
        """
        Dictionary for second-hand cars.
        """
        arr_t = np.array([v.transportType for v in list_vehicles])
        used_reb = np.where(arr_t == 3, self.used_rebate_calibration + self.used_rebate, 0)
        return {
            "Quality_a_t": np.array([v.Quality_a_t for v in list_vehicles]),
            "Eff_omega_a_t": np.array([v.Eff_omega_a_t for v in list_vehicles]),
            "price": np.array([v.price for v in list_vehicles]),
            "fuel_cost_c": np.array([v.fuel_cost_c for v in list_vehicles]),
            "e_t": np.array([v.e_t for v in list_vehicles]),
            "L_a_t": np.array([v.L_a_t for v in list_vehicles]),
            "transportType": arr_t,
            "used_rebate": used_reb,
            "delta": np.array([v.delta for v in list_vehicles])
        }

    # --------------------------------------------------------------------------------
    # The heuristic for second-hand car prices
    # --------------------------------------------------------------------------------

    def calc_offer_prices_heursitic(self, vehicle_dict_vecs_new_cars, vehicle_dict_vecs_current_cars, current_cars):
        """
        Compare second-hand cars to brand-new cars in terms of normalized (Quality, Efficiency)
        and set a cost_second_hand_merchant in each second-hand car.
        """
        # For brand-new cars
        first_hand_quality = vehicle_dict_vecs_new_cars["Quality_a_t"]
        first_hand_efficiency = vehicle_dict_vecs_new_cars["Eff_omega_a_t"]
        first_hand_prices = vehicle_dict_vecs_new_cars["price"]

        # For second-hand cars
        second_hand_quality = vehicle_dict_vecs_current_cars["Quality_a_t"]
        second_hand_efficiency = vehicle_dict_vecs_current_cars["Eff_omega_a_t"]
        second_hand_ages = vehicle_dict_vecs_current_cars["L_a_t"]
        second_hand_delta_P = vehicle_dict_vecs_current_cars["delta"]

        fhq_max = np.max(first_hand_quality)
        fhe_max = np.max(first_hand_efficiency)

        # normalize
        norm_fhq = first_hand_quality / fhq_max
        norm_fhe = first_hand_efficiency / fhe_max
        norm_shq = second_hand_quality / fhq_max
        norm_she = second_hand_efficiency / fhe_max

        diff_q = norm_shq[:, None] - norm_fhq
        diff_e = norm_she[:, None] - norm_fhe
        dist = np.sqrt(diff_q**2 + diff_e**2)
        closest = np.argmin(dist, axis=1)

        # get corresponding brand-new prices
        closest_prices = first_hand_prices[closest]
        # degrade price by (1 - delta)^Age
        # NOTE: originally you used second_hand_delta_P as "delta_P", adapt as needed
        adjusted_prices = closest_prices * (1 - second_hand_delta_P)**second_hand_ages
        offer_prices = adjusted_prices / (1 + self.mu)
        offer_prices = np.maximum(offer_prices, self.scrap_price)

        # store in the Car objects
        for i, car in enumerate(current_cars):
            car.price_second_hand_merchant = adjusted_prices[i]
            car.cost_second_hand_merchant = offer_prices[i]

    # --------------------------------------------------------------------------------
    # Vectorized utility for new cars / second-hand cars (global)
    # --------------------------------------------------------------------------------

    def calc_utility_new_cars_global(self, car_dict):
        """
        For *all users* vs these new cars, returning shape Nx(#cars).
        Then we later set EV columns to -inf for NCE users.
        """
        n_users = self.num_individuals
        n_cars = len(car_dict["price"])

        beta_mat = self.beta_vec[:, None]
        gamma_mat = self.gamma_vec[:, None]
        d_mat = self.d_vec[:, None]

        price_row = car_dict["price"][None, :]
        rebate_row = car_dict["rebate"][None, :]
        fuel_c_row = car_dict["fuel_cost_c"][None, :]
        e_t_row = car_dict["e_t"][None, :]
        Q_row = car_dict["Quality_a_t"][None, :]
        delta_row = car_dict["delta"][None, :]
        prod_em_row = car_dict["production_emissions"][None, :]

        factor = (1 + self.r) / ( self.r - (1 - delta_row)**self.alpha + 1 )
        quality_term = Q_row**self.alpha * d_mat * factor

        # net price => (price - rebate)
        net_price = np.maximum(0, price_row - rebate_row)
        fuel_term = d_mat * fuel_c_row * factor / car_dict["Eff_omega_a_t"][None, :]
        user_cost = beta_mat * (fuel_term + net_price)

        drive_em = d_mat * e_t_row * factor / car_dict["Eff_omega_a_t"][None, :]
        total_em = drive_em + prod_em_row
        em_penalty = gamma_mat * total_em

        return quality_term - user_cost - em_penalty

    def calc_utility_second_hand_global(self, car_dict, car_list):
        """
        Vectorized utility for second-hand cars, for all users vs #cars.
        Incorporate cost_second_hand_merchant from each car.
        """

        beta_mat = self.beta_vec[:, None]
        gamma_mat = self.gamma_vec[:, None]
        d_mat = self.d_vec[:, None]

        # read from the car objects
        cost_offer = np.array([c.cost_second_hand_merchant for c in car_list])[None, :]  # shape (1, n_cars)

        price_row = car_dict["price"][None, :]  # shape (1, n_cars)
        used_reb_row = car_dict["used_rebate"][None, :]
        net_price_raw = price_row - used_reb_row
        net_price = np.maximum(0, net_price_raw) - cost_offer  # final user cost

        Q_sh = (car_dict["Quality_a_t"] * (1 - car_dict["delta"])**car_dict["L_a_t"])[None, :]**self.alpha
        factor = (1 + self.r) / ( self.r - (1 - car_dict["delta"][None, :])**self.alpha + 1 )

        base_quality = d_mat * Q_sh * factor
        fuel_c_row = car_dict["fuel_cost_c"][None, :]
        eff_row    = car_dict["Eff_omega_a_t"][None, :]

        fuel_term = d_mat * fuel_c_row * factor / eff_row
        cost_term = beta_mat * (fuel_term + net_price)

        drive_em = d_mat * car_dict["e_t"][None, :] * factor / eff_row
        em_penalty = gamma_mat * drive_em

        return base_quality - cost_term - em_penalty

    # --------------------------------------------------------------------------------
    # Timeseries & counters
    # --------------------------------------------------------------------------------

    def prep_counters(self):
        self.users_driving_emissions_vec = np.zeros(self.num_individuals)
        self.users_distance_vec = np.zeros(self.num_individuals)
        self.users_utility_vec  = np.zeros(self.num_individuals)
        self.users_transport_type_vec  = np.full((self.num_individuals), np.nan)

        self.users_distance_vec_EV = np.full((self.num_individuals), np.nan)
        self.users_distance_vec_ICE = np.full((self.num_individuals), np.nan)
    
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
        # For advanced usage
        self.zero_util_count = 0
        self.num_switchers = 0
        self.drive_min_num = 0

    def update_counters(self, person_index, vehicle_chosen, driven_distance, utility):
        # driving emissions
        car_driving_emissions = (driven_distance / vehicle_chosen.Eff_omega_a_t) * vehicle_chosen.e_t
        self.users_driving_emissions_vec[person_index] = car_driving_emissions
        self.users_distance_vec[person_index] = driven_distance
        self.users_utility_vec[person_index] = utility
        self.users_transport_type_vec[person_index] = vehicle_chosen.transportType

        if vehicle_chosen.scenario == "new_car":
            self.new_cars_bought += 1
            self.total_production_emissions += vehicle_chosen.emissions
            if vehicle_chosen.transportType == 2:
                self.new_ICE_cars_bought += 1
                self.total_production_emissions_ICE += vehicle_chosen.emissions
            else:
                self.new_EV_cars_bought += 1
                self.total_production_emissions_EV += vehicle_chosen.emissions

        self.total_driving_emissions += car_driving_emissions

        if isinstance(vehicle_chosen, PersonalCar):
            vehicle_chosen.total_distance += driven_distance
            vehicle_chosen.total_driving_emmissions += car_driving_emissions
            vehicle_chosen.total_emissions += car_driving_emissions
            self.cars_cum_distances_driven.append(vehicle_chosen.total_distance)
            self.cars_cum_driven_emissions.append(vehicle_chosen.total_driving_emmissions)
            self.cars_cum_emissions.append(vehicle_chosen.total_emissions)

        self.car_ages.append(vehicle_chosen.L_a_t)
        self.quality_vals.append(vehicle_chosen.Quality_a_t)
        self.efficiency_vals.append(vehicle_chosen.Eff_omega_a_t)
        self.production_cost_vals.append(vehicle_chosen.ProdCost_t)

        if vehicle_chosen.transportType == 2:  # ICE
            self.users_distance_vec_ICE[person_index] = driven_distance
            self.quality_vals_ICE.append(vehicle_chosen.Quality_a_t)
            self.efficiency_vals_ICE.append(vehicle_chosen.Eff_omega_a_t)
            self.production_cost_vals_ICE.append(vehicle_chosen.ProdCost_t)
            self.total_driving_emissions_ICE += car_driving_emissions 
            self.total_distance_travelled_ICE += driven_distance
            self.ICE_users += 1
        else:  # EV
            self.users_distance_vec_EV[person_index] = driven_distance
            self.quality_vals_EV.append(vehicle_chosen.Quality_a_t)
            self.efficiency_vals_EV.append(vehicle_chosen.Eff_omega_a_t)
            self.production_cost_vals_EV.append(vehicle_chosen.ProdCost_t)
            self.total_driving_emissions_EV += car_driving_emissions 
            self.EV_users += 1

        self.total_utility += utility
        self.total_distance_travelled += driven_distance

        if isinstance(vehicle_chosen, PersonalCar):
            self.second_hand_users += 1
