import copy
import numpy as np
from package.model.carModel import CarModel
from package.model.personalCar import PersonalCar
from package.model.firm import Firm
from collections import defaultdict
import itertools

class Firm_Manager:
    """
    Manages all firms within the simulation environment.
    Responsible for initializing firms, tracking their market activity, computing profits,
    and updating firm behavior based on market and user data.
    """

    def __init__(self, parameters_firm_manager: dict, parameters_firm: dict, parameters_car_ICE: dict, parameters_car_EV: dict, ICE_landscape: dict, EV_landscape: dict):
        """
        Initialize the Firm_Manager with parameters and technology landscapes.

        Args:
            parameters_firm_manager (dict): Configuration for firm manager.
            parameters_firm (dict): Base configuration shared across firms.
            parameters_car_ICE (dict): Parameters for ICE car models.
            parameters_car_EV (dict): Parameters for EV car models.
            ICE_landscape (dict): NK landscape for ICE tech.
            EV_landscape (dict): NK landscape for EV tech.
        """
        self.t_firm_manager = 0
        self.parameters_firm = parameters_firm

        self.policy_distortion = 0
        self.profit_cumulative = 0
        self.random_state_input = parameters_firm_manager["random_state_input"]
        # Substream off `seed` (not seed_inputs) for initial conditions, so the
        # starting fleet and firm placement vary across replicates while the NK
        # landscape stays pinned. Falls back to random_state_input so that older
        # parameter dicts still load.
        self.random_state_init = parameters_firm_manager.get(
            "random_state_init", self.random_state_input
        )

        self.zero_profit_options_prod_sum = 0

        self.HHI_past_new_bought_vehicles_history = []
        self.margin_past_new_bought_vehicles_history = []

        # EV share of new sales. Recorded every step by update_EV_sales(),
        # independently of save_timeseries_data_state, because the calibration
        # matches against it and does not want the rest of the time series
        # overhead. Mirrors social_network.history_prop_EV, which is likewise
        # always on. set_up_time_series_firm_manager() still clears it, so a
        # continued future run restarts the series exactly as before.
        self.history_past_new_bought_vehicles_prop_ev = []

        # Market concentration of new-car UNIT sales, recorded every step by
        # update_HHI() for the same reason as the series above: the calibration
        # matches against it and does not want the rest of the time series
        # overhead. self.HHI is seeded here so calc_last_step_HHI() is safe to
        # call on a run that never got past burn-in. history_HHI_revenue is the
        # revenue-weighted variant, kept for reporting only (see
        # calculate_market_concentration).
        self.HHI = 0
        self.history_HHI = []
        self.HHI_revenue = 0
        self.history_HHI_revenue = []

        self.J = int(round(parameters_firm_manager["J"]))
        self.N = int(round(parameters_firm_manager["N"]))
        self.carbon_price = parameters_firm_manager["carbon_price"]
        self.id_generator = parameters_firm_manager["IDGenerator_firms"]
        self.kappa = parameters_firm_manager["kappa"]

        self.num_individuals = parameters_firm_manager["num_individuals"]
        self.time_steps_tracking_market_data = parameters_firm_manager["time_steps_tracking_market_data"]
        self.min_W = parameters_firm_manager["min_W"]
        self.num_beta_segments = parameters_firm_manager["num_beta_segments"]
        self.num_gamma_segments = parameters_firm_manager["num_gamma_segments"]

        # Initial fleet age, drawn Gamma(mean, std). A mean of 0 gives the original
        # behaviour -- every car born at L_a_t = 0 on month 0 -- which caps the
        # oldest possible car at t months and leaves the fleet-age transient still
        # running through the whole calibration window.
        self.init_car_age_mean = parameters_firm_manager["init_car_age_mean"]
        self.init_car_age_std = parameters_firm_manager["init_car_age_std"]
        self.init_car_age_max = parameters_firm_manager.get("init_car_age_max", 600)
        # Whether the starting fleet may ever be resold. Historically False,
        # because 3000 simultaneous age-0 trade-ins would swamp the used lot; with
        # a spread of starting ages the trade-ins arrive gradually instead.
        self.init_car_sellable = bool(parameters_firm_manager.get("init_car_sellable", False))

        # Where firms start on the NK landscape. See draw_initial_designs().
        self.init_firm_placement = parameters_firm_manager.get("init_firm_placement", "hamming1")
        self.init_firm_pool_prop = parameters_firm_manager.get("init_firm_pool_prop", 0.05)

        self.ev_production_bool = 0
        self.production_subsidy = 0

        self.all_segment_codes = list(itertools.product(range(self.num_beta_segments), range(self.num_gamma_segments), range(2)))
        self.num_segments = len(self.all_segment_codes)

        #landscapes
        self.landscape_ICE = ICE_landscape
        self.landscape_EV = EV_landscape

        #car paramets
        self.parameters_car_ICE = parameters_car_ICE
        self.parameters_car_EV = parameters_car_EV 

        self.init_firms()
        
        #calculate the inital attributes of all the cars on sale
        self.cars_on_sale_all_firms = self.generate_cars_on_sale_all_firms()
             
    def gen_initial_cars(self):
        """
        Generate initial old car stock for simulation.

        Returns:
            list: List of PersonalCar instances representing the initial car stock.
        """
        model_choices = self.random_state_input.choice(self.cars_on_sale_all_firms, self.num_individuals)

        ages = self.gen_initial_car_ages(self.num_individuals)
        init_car_flag = 0 if self.init_car_sellable else 1

        car_list = []
        for i, car in enumerate(model_choices):
            personalCar_id = self.id_generator.get_new_id()
            car_real = PersonalCar(personalCar_id, car.firm, None, car.component_string, car.parameters, car.attributes_fitness, car.price, init_car=init_car_flag)
            car_real.L_a_t = ages[i]
            # An age-a car was bought a months ago and has depreciated since, so
            # its resale value must reflect that -- otherwise a 20-year-old
            # starting car would trade in at the new-car price.
            car_real.price = car_real.original_price*(1 - car_real.delta_P)**ages[i]
            car_list.append(car_real)

        self.old_cars = car_list
        return self.old_cars

    def gen_initial_car_ages(self, n):
        """
        Draw starting ages (in months) for the initial fleet, L ~ Gamma(mean, std),
        truncated at init_car_age_max.

        Gamma because it is a two-parameter family, so mean and spread are set
        independently and both moments of the observed fleet can be matched at
        once. Single-parameter alternatives cannot: uniform fixes sd = mean/sqrt(3)
        and truncates the support at 2*mean, exponential fixes sd = mean. Against
        the equilibrium measured from a 360-month burn-in (mean 16.2 yr, sd 10.8,
        median 14.0, tail to 49 yr) gamma matches every moment, while uniform is
        symmetric so its median sits at the mean, and its support stops at 32 yr.

        A mean of 0 gives the original behaviour, every car born new. That leaves
        the oldest car in the fleet at month t exactly t months old, so the age
        distribution has no right tail until the model has run for decades -- which
        is why the fleet-age transient used to run through the whole calibration
        window. Kept as the off switch, and as the baseline rung of the burn-in
        ablation.

        Drawn off random_state_init so the starting fleet varies across seeds while
        the NK landscape stays pinned to seed_inputs.

        Set init_car_age_max high enough not to clip the tail being reproduced;
        capping at 360 would remove exactly the 30-49 yr range the gamma is for.

        Args:
            n (int): Number of cars to draw ages for.

        Returns:
            np.ndarray: Integer ages in months, length n.
        """
        if self.init_car_age_mean <= 0:
            return np.zeros(n, dtype=int)
        if self.init_car_age_std <= 0:
            raise ValueError(
                f"init_car_age_std must be > 0 when init_car_age_mean is "
                f"({self.init_car_age_mean}); got {self.init_car_age_std}"
            )
        # Moment-matched: shape k and scale theta chosen so the draw has exactly
        # the requested mean and sd.
        shape = (self.init_car_age_mean/self.init_car_age_std)**2
        scale = self.init_car_age_std**2/self.init_car_age_mean
        ages = self.random_state_init.gamma(shape, scale, size=n)
        return np.clip(ages, 0, self.init_car_age_max).astype(int)
    

    def init_firms(self):
        """
        Instantiate the initial firms and their technology choices based on the provided landscapes.
        """

        self.init_tech_component_string_ICE = self.landscape_ICE.min_fitness_string
        self.init_tech_component_string_EV = self.landscape_EV.min_fitness_string

        decimal_value_ICE = int(self.init_tech_component_string_ICE, 2)
        decimal_value_EV = int(self.init_tech_component_string_EV, 2)

        init_tech_component_string_list_N_ICE = self.invert_bits_one_at_a_time(decimal_value_ICE, len(self.init_tech_component_string_ICE))
        init_tech_component_string_list_N_EV = self.invert_bits_one_at_a_time(decimal_value_EV, len(self.init_tech_component_string_EV))

        init_tech_component_string_list_ICE = self.draw_initial_designs(
            init_tech_component_string_list_N_ICE, self.landscape_ICE
        )
        init_tech_component_string_list_EV = self.draw_initial_designs(
            init_tech_component_string_list_N_EV, self.landscape_EV
        )

        self.init_tech_list_ICE = [CarModel(init_tech_component_string_list_ICE[j], self.landscape_ICE, parameters = self.parameters_car_ICE, choosen_tech_bool=1) for j in range(self.J)]
        self.init_tech_list_EV = [CarModel(init_tech_component_string_list_EV[j], self.landscape_EV, parameters = self.parameters_car_EV, choosen_tech_bool=1) for j in range(self.J)]

        #global repo
        self.universal_model_repo_ICE = {} 
        self.universal_model_repo_EV = {}

        self.parameters_firm["universal_model_repo_EV"] = self.universal_model_repo_EV
        self.parameters_firm["universal_model_repo_ICE"] = self.universal_model_repo_ICE
        self.parameters_firm["segment_codes"] = self.all_segment_codes

        #Create the firms, these store the data but dont do anything otherwise
        self.firms_list = [Firm(j, self.init_tech_list_ICE[j], self.init_tech_list_EV[j],  self.parameters_firm, self.parameters_car_ICE, self.parameters_car_EV) for j in range(self.J)]

    def draw_initial_designs(self, hamming_neighbours, landscape):
        """
        Choose the J starting designs for the firms on one landscape.

        Two placements are available:

          hamming1    the original. Draw from the 15 one-bit-flip neighbours of the
                      single worst design, with replacement and off seed_inputs.
                      Because seed_inputs is pinned, every replicate starts the
                      firms at the same designs, and drawing 10 from 15 with
                      replacement leaves ~2.5 firms as exact duplicates.
          worst_pool  draw without replacement from the worst init_firm_pool_prop
                      of the sampled landscape, off the `seed` substream. Firms
                      are still bad, but spread over several basins and varying
                      across replicates.

        Args:
            hamming_neighbours (list): One-bit-flip neighbours of the worst design.
            landscape: The NK landscape to place firms on.

        Returns:
            np.ndarray: J binary strings, one per firm.
        """
        if self.init_firm_placement == "hamming1":
            return self.random_state_input.choice(hamming_neighbours, self.J)
        elif self.init_firm_placement == "worst_pool":
            ranked = landscape.sampled_strings_ranked
            pool_size = max(self.J, int(self.init_firm_pool_prop*len(ranked)))
            pool = ranked[:pool_size]
            return self.random_state_init.choice(pool, self.J, replace=False)
        else:
            raise ValueError(
                f"Unknown init_firm_placement {self.init_firm_placement!r}, expected 'hamming1' or 'worst_pool'"
            )

    def invert_bits_one_at_a_time(self, decimal_value, length):
        """
        Generate variations of a binary string by flipping each bit once.

        Args:
            decimal_value (int): Original binary string as a decimal.
            length (int): Length of the binary string.

        Returns:
            list: List of binary strings with one bit flipped.
        """
        inverted_binary_values = []
        for bit_position in range(length):
            inverted_value = decimal_value ^ (1 << bit_position)
            inverted_binary_value = format(inverted_value, f'0{length}b')
            inverted_binary_values.append(inverted_binary_value)
        return inverted_binary_values

    def generate_cars_on_sale_all_firms(self):
        """
        Collect all cars on sale from all firms.

        Returns:
            list: List of car instances available on the market.
        """
        cars_on_sale_all_firms = []
        for firm in self.firms_list:
            cars_on_sale_all_firms.extend(firm.cars_on_sale)
        return cars_on_sale_all_firms

    def input_social_network_data(self, beta_vec, gamma_vec, consider_ev_vec, beta_bins, gamma_bins):
        """
        Input the social network data and segment users based on beta and gamma.

        Args:
            beta_vec (np.ndarray): Consumer quality sensitivity values.
            gamma_vec (np.ndarray): Consumer emission preferences.
            consider_ev_vec (np.ndarray): Binary indicator if user considers EV.
            beta_bins (np.ndarray): Binning thresholds for beta.
            gamma_bins (np.ndarray): Binning thresholds for gamma.
        """
        
        self.beta_vec = beta_vec
        self.gamma_vec = gamma_vec
        self.consider_ev_vec = consider_ev_vec

        self.beta_bins = beta_bins
        self.gamma_bins = gamma_bins
        self.beta_segment_idx = np.digitize(self.beta_vec, self.beta_bins) - 1
        self.gamma_segment_idx = np.digitize(self.gamma_vec, self.gamma_bins) - 1

        # Precomputed flat segment index, for _count_segments(). beta and gamma
        # segments are fixed for the whole run (they come from the fixed
        # preference draws and the fixed quantile bins), so only the EV term
        # varies per timestep.
        #
        # np.digitize returns len(bins) for a value equal to the top bin edge,
        # so an index can land outside [0, num_segments) -- the individual
        # holding the maximum beta, for one. Those fall into segment codes that
        # are not in all_segment_codes, so the original counted them into a
        # defaultdict entry that was never read back. in_range reproduces that
        # exactly by dropping them.
        self._segment_in_range = (
            (self.beta_segment_idx >= 0) & (self.beta_segment_idx < self.num_beta_segments)
            & (self.gamma_segment_idx >= 0) & (self.gamma_segment_idx < self.num_gamma_segments)
        )
        # Matches the ordering of itertools.product(beta, gamma, ev)
        self._segment_flat_base = 2*(self.gamma_segment_idx + self.num_gamma_segments*self.beta_segment_idx)

    def _count_segments(self):
        """
        Count individuals per segment code.

        Returns:
            np.ndarray: Counts indexed to match all_segment_codes.

        Replaces a per-individual Python loop that unboxed three NumPy scalars,
        built a tuple and hashed it, num_individuals times per timestep. Integer
        counting, so the result is exact.
        """
        flat = self._segment_flat_base + self.consider_ev_vec
        return np.bincount(flat[self._segment_in_range], minlength=self.num_segments)

    def calc_exp(self, U):
        """
        Calculate exponential of scaled utility value.

        Args:
            U (float): Utility value.

        Returns:
            float: Exponentiated value.
        """
        exp_input = self.kappa*U
        comp = np.exp(exp_input)
        return comp

    def generate_market_data(self):
        """
            Generate initial market data by segment based on user attributes and car utility scores from firms.
        For each segment code, we store:
         - I_s_t     (count of individuals)
         - beta_s_t  (average beta)
         - gamma_s_t (average gamma)
         - W     (will be computed after we calculate utilities)
        """
         # 1) Build a dictionary for ALL possible combos
        self.market_data = {}
        for code in self.all_segment_codes:
            self.market_data[code] = {
                "I_s_t": 0,
                "W": self.min_W,#0.0,
                "history_I_s_t": [],
                "history_W": [],
                "maxU": 0
            }

        # 2) Count how many individuals fall into each segment code
        segment_counts = self._count_segments()

        # 3) Compute midpoints for each segment
        for i, code in enumerate(self.all_segment_codes):
            b_idx, g_idx, e_idx = code

            if (e_idx == 0) or (e_idx == 1 and self.ev_production_bool):
                self.market_data[code]["I_s_t"] = int(segment_counts[i])#IS NOT AN EV SEGMENT or CAN PRODUCE EVS AND THE SEGMENT ALLOWS IT
            else:
                self.market_data[code]["I_s_t"] = 0#CANT PRODUCE AN EV
        
        #4) calc the utility of each car (already did base utility in the car but need the full value including price and emissiosn production)
        for firm in self.firms_list:
            firm.calc_init_U_segments()


        # 5) Sum up the utilities across all cars for each segment
        segment_W = defaultdict(float)

        for segment in self.all_segment_codes:
            segment_W[segment] = self.min_W#SET THE MINIMUM
        
        #5.5) calc the max U 
        for firm in self.firms_list:
            for car in firm.cars_on_sale:
                for code, U in car.car_utility_segments_U.items():
                    if U > self.market_data[code]["maxU"]:
                        self.market_data[code]["maxU"] = U


        kappa = self.kappa
        for firm in self.firms_list:
            for car in firm.cars_on_sale:
                for code, U in car.car_utility_segments_U.items():
                        segment_W[code] += np.exp(kappa*U)

        # 6) Store the U_sum in market_data
        for code in self.all_segment_codes:
            self.market_data[code]["W"] = segment_W[code]
    
        self.I_s_t_vec = np.asarray([self.market_data[code]["I_s_t"] for code in self.all_segment_codes])
        self.W_vec = np.asarray([self.market_data[code]["W"] for code in self.all_segment_codes])
        self.maxU_vec = np.asarray([self.market_data[code]["maxU"] for code in self.all_segment_codes])

    def update_W_immediate(self):
        """
        Immediately recompute the choice denominator W for each segment
        using currently available cars on sale.

        Returns:
            tuple: (dict of W values by segment, array of max utility values)
        """
                
        #calc the total "probability of selection" of the market based on max utility in the segment
        segment_W = defaultdict(float)

        for segment in self.all_segment_codes:
            segment_W[segment] = self.min_W#RESET THEM INCASE
        
        #UPDATE U MAX
        # The two passes below were separate loops over the same cars and the
        # same per-car segment dicts. They are independent of each other -- the
        # W sum never reads maxU -- so running them together visits each (car,
        # segment) pair once instead of twice. Each still sees the same values
        # in the same order, so the running max and the running sum are
        # unchanged.
        # calc_exp is inlined here rather than called: it is a two-line wrapper
        # invoked once per (car, segment) pair per timestep, so the Python call
        # and the self.kappa lookup dominated the arithmetic it performs.
        market_data = self.market_data
        kappa = self.kappa
        for car in self.cars_on_sale_all_firms:
            for segment, U in car.car_utility_segments_U.items():
                    if U > market_data[segment]["maxU"]:
                            market_data[segment]["maxU"] = U

                    segment_W[segment] += np.exp(kappa*U)

        maxU_vec = np.asarray([self.market_data[code]["maxU"] for code in self.all_segment_codes])

        return segment_W, maxU_vec
        
    def update_market_data_moving_average(self, W_segment):
        """
        Smooth out market data using a moving average over a sliding window.

        Args:
            W_segment (dict): Current choice denominators per segment.

        Returns:
            tuple: (array of segment sizes, array of choice denominators)
        """
        segment_counts = self._count_segments()
        code_positions = {code: i for i, code in enumerate(self.all_segment_codes)}

        for code in self.market_data.keys():
            entry = self.market_data[code]
            # Append current values to history
            #SEGMENT COUNTS
            e_idx = code[2]
            if (e_idx == 0) or (e_idx == 1 and self.ev_production_bool):
                count = int(segment_counts[code_positions[code]])#IS NOT AN EV SEGMENT or CAN PRODUCE EVS AND THE SEGMENT ALLOWS IT
            else:
                count = 0#CANT PRODUCE AN EV

            history_I_s_t = entry["history_I_s_t"]
            history_W = entry["history_W"]
            history_I_s_t.append(count)
            history_W.append(W_segment[code])

            # Trim history to the last N time steps
            if len(history_I_s_t) > self.time_steps_tracking_market_data:
                history_I_s_t.pop(0)
            if len(history_W) > self.time_steps_tracking_market_data:
                history_W.pop(0)

            # Calculate moving averages
            moving_avg_I_s_t = np.mean(history_I_s_t)
            moving_avg_W = np.mean(history_W)

            # Store the moving averages
            entry["I_s_t"] = moving_avg_I_s_t
            entry["W"] = moving_avg_W
        
        I_s_t_vec = np.asarray([self.market_data[code]["I_s_t"] for code in self.all_segment_codes])
        W_vec = np.asarray([self.market_data[code]["W"] for code in self.all_segment_codes])

        return I_s_t_vec, W_vec
        
    def calc_total_profits(self, past_new_bought_vehicles, prod_subsidy):
        """
        Calculate total firm profit based on car sales and production costs.

        Args:
            past_new_bought_vehicles (list): Cars sold in the current period.
            prod_subsidy (float): Government production subsidy per car.

        Returns:
            float: Total profit across all firms.
        """
        total_profit_all_firms = 0         
        for car in self.cars_on_sale_all_firms:#LOOP OVER ALL CARS ON SALE TO DO IT IN ONE GO I GUESS
            num_vehicle_sold = past_new_bought_vehicles.count(car)
            
            if num_vehicle_sold > 0:#ONLY COUNT WHEN A CAR HAS ACTUALLY BEEN SOLD
                
                if car.transportType == 3:
                    profit = car.price - np.maximum(0,car.ProdCost_t - prod_subsidy)
                else:
                    profit = car.price - car.ProdCost_t

                total_profit = num_vehicle_sold*profit
                car.firm.firm_profit += total_profit#I HAVE NO IDEA IF THIS WILL WORK
                total_profit_all_firms += total_profit

                #OPTIMIZATION OF PRODUCTION SUBSIDY
                if car.transportType == 3:
                    self.policy_distortion += num_vehicle_sold * np.minimum(car.ProdCost_t, self.production_subsidy)

        return total_profit_all_firms
    
    def calculate_market_share(self, firm, past_new_bought_vehicles, total_sales):
        """
        Calculate a firm's REVENUE share of new-car sales.

        Used only by the revenue-weighted HHI reported alongside the unit-share
        HHI (see calculate_market_concentration_revenue). The calibrated measure
        is the unit-share one.

        Args:
            firm (Firm): Firm instance.
            past_new_bought_vehicles (list): Cars sold in last period.
            total_sales (float): Total revenue from all sales.

        Returns:
            float: Firm's revenue market share.
        """
        # Calculate total sales for the specified firm by summing prices of cars sold by this firm
        firm_sales = sum(car.price for car in past_new_bought_vehicles if car.firm == firm)
        
        # If total_sales is zero, return 0 to avoid division by zero; otherwise, calculate firm market share
        MS_firm = firm_sales / total_sales if total_sales > 0 else 0
        return MS_firm

    def calculate_market_concentration(self, new_bought_vehicles):
        """
        Compute the Herfindahl-Hirschman Index (HHI) of new-car sales over a
        trailing 12-month window, from UNIT sales shares.

        A firm's share is its count of cars sold divided by all cars sold, so a
        $70k car and a $20k car each count once. This is the measure the
        calibration target comes from: Grieco, Murry & Yurukoglu (2024) compute
        HHI "at the parent company level" from Wards make-model UNIT sales
        (falling from over 2500 to around 1200 on the 0-10000 scale, i.e. 0.25
        to 0.12 as a fraction), and their C4 index is likewise a share of units.

        This used to be a REVENUE share HHI (each car weighted by its price),
        which is a different object: it measures concentration of industry
        turnover, and it moves with within-firm pricing even when every firm
        sells exactly as many cars as before. That variant is kept as
        calculate_market_concentration_revenue() and reported alongside, but it
        is not what 0.11-0.18 refers to.

        NOTE: this method MUTATES the 12-month rolling window, so it must be
        called at most once per step -- see update_HHI().

        Args:
            new_bought_vehicles (list): Vehicles sold this time step, one entry
                per unit sold.

        Returns:
            float: HHI as a fraction (bounded below by 1/J for J equal firms).
        """

        # Append the new purchases to history
        self.HHI_past_new_bought_vehicles_history.append(new_bought_vehicles)

        # Trim to last 12 time steps
        if len(self.HHI_past_new_bought_vehicles_history) > 12:
            self.HHI_past_new_bought_vehicles_history.pop(0)

        # Flatten the list to get all purchases from the last 12 time steps
        all_purchases = list(itertools.chain(*self.HHI_past_new_bought_vehicles_history))

        # Total UNITS sold over the last 12 time steps
        total_units = len(all_purchases)

        # If no sales, return HHI as zero
        if total_units == 0:
            return 0

        # Units per firm, then the sum of squared unit shares. Counting in one
        # pass keys on firm_id (Firm defines no __eq__, so this is the same
        # identity comparison the revenue version does, without the J passes).
        units_per_firm = defaultdict(int)
        for car in all_purchases:
            units_per_firm[car.firm.firm_id] += 1

        HHI = sum((units / total_units) ** 2 for units in units_per_firm.values())

        return HHI

    def calculate_market_concentration_revenue(self):
        """
        Revenue-weighted HHI over the same trailing 12-month window.

        Reported for comparison only; the calibrated measure is the unit-share
        HHI in calculate_market_concentration(). Reads the rolling window that
        method already advanced this step rather than appending to it again.

        Returns:
            float: Revenue-share HHI as a fraction.
        """
        all_purchases = list(itertools.chain(*self.HHI_past_new_bought_vehicles_history))

        total_sales = sum(car.price for car in all_purchases)
        if total_sales == 0:
            return 0

        return sum(
            self.calculate_market_share(firm, all_purchases, total_sales) ** 2
            for firm in self.firms_list
        )

    def calc_profit_margin(self, new_bought_vehicles):
        """
        Calculate profit margins for ICE and EV vehicles sold over last 12 steps.

        Args:
            new_bought_vehicles (list): List of cars sold in the current time step.

        Returns:
            tuple: (List of ICE profit margins, List of EV profit margins)
        """

        # Append the new purchases to history
        self.margin_past_new_bought_vehicles_history.append(new_bought_vehicles)

        # Trim to last 12 time steps
        if len(self.margin_past_new_bought_vehicles_history) > 12:
            self.margin_past_new_bought_vehicles_history.pop(0)

        # Flatten the list to get all purchases from the last 12 time steps
        all_purchases = list(itertools.chain(*self.HHI_past_new_bought_vehicles_history))

        profit_margin_ICE = []
        profit_margin_EV = []

        # Calculate the HHI by summing the squares of market shares for each firm
        for car in all_purchases:
            if car.transportType == 3:
                prod_cost = np.maximum(0, car.ProdCost_t - self.production_subsidy)
                
                if prod_cost == 0:
                    profit_margin = np.inf
                else:
                    profit_margin = (car.price - prod_cost)/car.price 
                
                profit_margin_EV.append(profit_margin)
            else:
                
                prod_cost = car.ProdCost_t
                if prod_cost == 0:
                    profit_margin = np.inf
                else:
                    profit_margin = (car.price - prod_cost)/car.price 
                profit_margin_ICE.append(profit_margin)

        return profit_margin_ICE, profit_margin_EV
    
    def calc_last_step_profit_margin(self):
        """
        Return average profit margin from the last time step across all firms.

        Returns:
            float: Mean profit margin.
        """
        profit_margin_ICE, profit_margin_EV = self.calc_profit_margin(self.past_new_bought_vehicles)
        all_profit_margins = profit_margin_ICE +  profit_margin_EV
        return np.mean(all_profit_margins)

    def calc_last_step_HHI(self):
        """
        Return HHI value from last time step.

        Returns:
            float: Herfindahl-Hirschman Index.
        """
        # Read, don't recompute: update_HHI() already advanced the 12-step
        # rolling window this step, and calculate_market_concentration()
        # mutates that window.
        return self.HHI

    def set_up_time_series_firm_manager(self):
        """
        Initialize all historical tracking structures for time series data.
        """
        self.history_total_profit = []
        self.history_market_concentration = []
        self.history_segment_count = []
        self.history_cars_on_sale_EV_prop = []
        self.history_cars_on_sale_ICE_prop = []
        self.history_cars_on_sale_price = []

        self.history_market_data = []
        self.history_zero_profit_options_prod_sum = []
        self.history_zero_profit_options_research_sum = []


        self.history_mean_profit_margins_EV = []
        self.history_mean_profit_margins_ICE = []

        self.history_median_profit_margins_EV = []
        self.history_median_profit_margins_ICE = []

        self.history_W = []

        self.history_quality_ICE = []
        self.history_efficiency_ICE  = []
        self.history_production_cost_ICE  = []

        self.history_quality_EV = []
        self.history_efficiency_EV = []
        self.history_production_cost_EV = []
        self.history_battery_EV = []
        self.history_prop_EV = []

        self.history_prop_EV_research = []
        self.history_prop_ICE_research = []

        self.history_past_new_bought_vehicles_prop_ev = []
        self.history_HHI = []
        self.history_HHI_revenue = []


    def save_timeseries_data_firm_manager(self):
        """
        Store historical data for current time step across firm and market variables.
        Tracks metrics like profit, HHI, car attributes, and EV adoption.
        """
        # Extract research type history for each firm (last 12 years, or fewer if not available)
        research_history = [firm.history_research_type[-12:] for firm in self.firms_list]

        # Convert to numpy array, ensuring a consistent dtype for handling NaNs
        research_history = np.array(research_history, dtype=float)  # Convert to float to handle NaNs

        # Count valid (non-NaN) research occurrences per firm
        valid_counts = np.sum(~np.isnan(research_history), axis=1)  # Count non-NaN entries per firm

        # Count occurrences of EV (1) and ICE (0) research
        ev_counts = np.nansum(research_history == 1, axis=1)  # Count EV research (ignoring NaNs)
        ice_counts = np.nansum(research_history == 0, axis=1)  # Count ICE research (ignoring NaNs)

        # **Handle cases where valid_counts == 0**
        ev_proportion = np.zeros_like(valid_counts, dtype=float)  # Default to 0
        ice_proportion = np.zeros_like(valid_counts, dtype=float)  # Default to 0

        # Compute proportions only where valid_counts > 0
        nonzero_mask = valid_counts > 0
        ev_proportion[nonzero_mask] = ev_counts[nonzero_mask] / valid_counts[nonzero_mask]
        ice_proportion[nonzero_mask] = ice_counts[nonzero_mask] / valid_counts[nonzero_mask]

        # Compute the **overall average proportion** across all firms (ignoring empty cases)
        avg_ev_proportion = np.mean(ev_proportion)  # No need for nanmean since we ensured no NaNs
        avg_ice_proportion = np.mean(ice_proportion)
        self.history_prop_EV_research.append(avg_ev_proportion)
        self.history_prop_ICE_research.append(avg_ice_proportion)

        self.EV_users_count = sum(1 if car.transportType == 3 else 0 for car in  self.cars_on_sale_all_firms)
        self.history_prop_EV.append(self.EV_users_count/len(self.cars_on_sale_all_firms))
    
        # self.HHI is already set for this step by update_HHI() in next_step();
        # recomputing here would advance the rolling purchase window twice.
        profit_margin_ICE, profit_margin_EV = self.calc_profit_margin(self.past_new_bought_vehicles)

        if profit_margin_EV:
            self.history_mean_profit_margins_EV.append(np.nanmean(profit_margin_EV))
            self.history_median_profit_margins_EV.append(np.nanmedian(profit_margin_EV))
        else:
            self.history_mean_profit_margins_EV.append(np.nan)
            self.history_median_profit_margins_EV.append(np.nan)

        if profit_margin_ICE:
            self.history_mean_profit_margins_ICE.append(np.nanmean(profit_margin_ICE))
            self.history_median_profit_margins_ICE.append(np.nanmedian(profit_margin_ICE))
        else:
            self.history_mean_profit_margins_ICE.append(np.nan)
            self.history_median_profit_margins_ICE.append(np.nan)

        self.calc_vehicles_chosen_list(self.past_new_bought_vehicles)
        self.history_cars_on_sale_price.append([car.price for car in self.cars_on_sale_all_firms])

        self.history_total_profit.append(self.total_profit)
        self.history_market_concentration.append(self.HHI)
        self.history_segment_count.append([segment_data["I_s_t"] for segment_data in self.market_data.values()])

        count_transport_type_2 = sum(1 for car in self.cars_on_sale_all_firms if car.transportType == 2)
        count_transport_type_3 = sum(1 for car in self.cars_on_sale_all_firms if car.transportType == 3)

        self.history_cars_on_sale_ICE_prop.append(count_transport_type_2)
        self.history_cars_on_sale_EV_prop.append(count_transport_type_3)

        self.history_market_data.append(copy.deepcopy(self.market_data))

        self.history_zero_profit_options_prod_sum.append(self.zero_profit_options_prod_sum/self.J)
        self.history_zero_profit_options_research_sum.append(self.zero_profit_options_research_sum/self.J)

        self.history_W.append(list(self.W_segment.values()))

        self.quality_vals_ICE = []
        self.efficiency_vals_ICE = []
        self.production_cost_vals_ICE = []
        self.quality_vals_EV = []
        self.efficiency_vals_EV = []
        self.production_cost_vals_EV = []
        self.battery_EV = []

        for car in self.cars_on_sale_all_firms:
            if car.transportType == 2:#ICE 
                self.quality_vals_ICE.append(car.Quality_a_t)#done here for efficiency
                self.efficiency_vals_ICE.append(car.Eff_omega_a_t)
                self.production_cost_vals_ICE.append(car.ProdCost_t)
            else:#EV
                self.quality_vals_EV.append(car.Quality_a_t)#done here for efficiency
                self.efficiency_vals_EV.append(car.Eff_omega_a_t)
                self.production_cost_vals_EV.append(car.ProdCost_t)
                self.battery_EV.append(car.B)

        if self.quality_vals_EV:
            self.history_quality_EV.append(self.quality_vals_EV)
            self.history_efficiency_EV.append(self.efficiency_vals_EV)
            self.history_production_cost_EV.append(self.production_cost_vals_EV)
            self.history_battery_EV.append(self.battery_EV)
        else:
            self.history_quality_EV.append([np.nan])
            self.history_efficiency_EV.append([np.nan])
            self.history_production_cost_EV.append([np.nan])
            self.history_battery_EV.append([np.nan])

        if self.quality_vals_ICE:
            self.history_quality_ICE.append(self.quality_vals_ICE)
            self.history_efficiency_ICE.append(self.efficiency_vals_ICE)
            self.history_production_cost_ICE.append(self.production_cost_vals_ICE)
        else:
            self.history_quality_ICE.append([np.nan])
            self.history_efficiency_ICE.append([np.nan])
            self.history_production_cost_ICE.append([np.nan])

    def calc_vehicles_chosen_list(self, past_new_bought_vehicles):
        """
        Count how many vehicles from each firm were chosen by users.

        Args:
            past_new_bought_vehicles (list): Cars selected by consumers.
        """
        for firm in self.firms_list:
            firm.firm_cars_users = sum(1 for car in past_new_bought_vehicles if car.firm == firm)

    def update_firms(self, gas_price, electricity_price, electricity_emissions_intensity, rebate, production_subsidy,  rebate_calibration, gas_cost_index=0.0, gas_emissions_index=0.0, electricity_cost_index=0.0, electricity_emissions_index=0.0, ice_sales_ban_active=False, ice_research_ban_active=False):
        """
        Update firm behavior, generate new cars on sale.

        Returns:
            list: All cars now available in the market.
        """

        cars_on_sale_all_firms = []

        self.zero_profit_options_prod_sum = 0
        self.zero_profit_options_research_sum = 0


        # 4) Now we need to compute the initial W for each segment
        for firm in self.firms_list:
            self.zero_profit_options_prod_sum += firm.zero_profit_options_prod#CAN DELETE OCNE FIXED ISSUE O uitlity in firms prod
            self.zero_profit_options_research_sum += firm.zero_profit_options_research
            cars_on_sale = firm.next_step(self.I_s_t_vec, self.W_vec, self.maxU_vec, self.carbon_price, gas_price, electricity_price, electricity_emissions_intensity, rebate, production_subsidy,  rebate_calibration, gas_cost_index, gas_emissions_index, electricity_cost_index, electricity_emissions_index, ice_sales_ban_active, ice_research_ban_active)

            cars_on_sale_all_firms.extend(cars_on_sale)

        return cars_on_sale_all_firms
    
    def update_firms_burn_in(self):
        """
        Update firm behavior during the burn-in phase with simplified logic.

        Returns:
            list: Cars on sale post burn-in iteration.
        """
        cars_on_sale_all_firms = []
        for firm in self.firms_list:
            cars_on_sale = firm.next_step_burn_in(self.I_s_t_vec, self.W_vec, self.maxU_vec)
            cars_on_sale_all_firms.extend(cars_on_sale)
        return cars_on_sale_all_firms

#####################################################################################################################

    def update_EV_sales(self):
        """
        Record the EV proportion of the vehicles bought in the previous step, and append it to the history.

        Called from next_step() rather than save_timeseries_data_firm_manager()
        so the calibration gets this series without switching on
        save_timeseries_data_state. past_new_bought_vehicles is set immediately
        before this call and is not touched again during the step, so the value
        stored at index t is identical to what the old save path recorded.
        """
        if self.past_new_bought_vehicles:
            self.history_past_new_bought_vehicles_prop_ev.append(sum([1 for car in self.past_new_bought_vehicles if car.transportType == 3])/len(self.past_new_bought_vehicles))
        else:
            self.history_past_new_bought_vehicles_prop_ev.append(np.nan)

    def update_HHI(self):
        """
        Compute this step's market concentration and append it to the history.

        This is now the ONLY per-step call to calculate_market_concentration().
        That method mutates the 12-step rolling purchase window
        (HHI_past_new_bought_vehicles_history), so calling it a second time in
        the same step would push the same purchase list in twice and halve the
        effective window; save_timeseries_data_firm_manager() and
        calc_last_step_HHI() therefore read self.HHI instead of recomputing it.

        Called from next_step() straight after update_EV_sales(), so index t of
        history_HHI lines up with index t of
        history_past_new_bought_vehicles_prop_ev and of
        social_network.history_prop_EV -- which is what lets the calibration
        address all of them with one month-offset convention.

        self.HHI is the UNIT-share HHI (the calibrated measure). The
        revenue-weighted variant is recorded next to it, from the same window,
        for reporting only.
        """
        self.HHI = self.calculate_market_concentration(self.past_new_bought_vehicles)
        self.history_HHI.append(self.HHI)

        self.HHI_revenue = self.calculate_market_concentration_revenue()
        self.history_HHI_revenue.append(self.HHI_revenue)

    def next_step(self, carbon_price, consider_ev_vec, new_bought_vehicles,  gas_price, electricity_price, electricity_emissions_intensity, rebate,  production_subsidy,  rebate_calibration, gas_cost_index=0.0, gas_emissions_index=0.0, electricity_cost_index=0.0, electricity_emissions_index=0.0, ice_sales_ban_active=False, ice_research_ban_active=False):
        """
        Advance firms by one simulation step, considering updated policies and user behavior.

        Returns:
            list: Cars on sale from all firms.
        """
        self.t_firm_manager += 1
        self.past_new_bought_vehicles = new_bought_vehicles
        self.update_EV_sales()
        self.update_HHI()
        self.total_profit = self.calc_total_profits(self.past_new_bought_vehicles, self.production_subsidy)#NEED TO CALC TOTAL PROFITS NOW before the cars on sale change?
        self.profit_cumulative += self.total_profit

        self.consider_ev_vec = consider_ev_vec#UPDATE THIS TO NEW CONSIDERATION
        self.carbon_price = carbon_price
        self.production_subsidy = production_subsidy

        self.cars_on_sale_all_firms  = self.update_firms(gas_price, electricity_price, electricity_emissions_intensity, rebate, production_subsidy,  rebate_calibration, gas_cost_index, gas_emissions_index, electricity_cost_index, electricity_emissions_index, ice_sales_ban_active, ice_research_ban_active)#WE ASSUME THAT FIRMS DONT CONSIDER SECOND HAND MARKET
        self.W_segment, self.maxU_vec = self.update_W_immediate()#calculate the competiveness of the market current

        self.I_s_t_vec, self.W_vec = self.update_market_data_moving_average(self.W_segment)#update the rollign vlaues

        return self.cars_on_sale_all_firms
    

    def next_step_burn_in(self):
        """
        Perform burn-in logic for firms before fuall simultion run.
        Updates firm offerings and internal statistics.
        """
        self.cars_on_sale_all_firms  = self.update_firms_burn_in()#WE ASSUME THAT FIRMS DONT CONSIDER SECOND HAND MARKET
        self.W_segment, self.maxU_vec = self.update_W_immediate()#calculate the competiveness of the market current
        self.I_s_t_vec, self.W_vec = self.update_market_data_moving_average(self.W_segment)#update the rollign vlaues

    