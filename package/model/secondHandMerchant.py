import numpy as np
from itertools import compress
from operator import attrgetter


def _attr_array(list_vehicles, name, dtype=np.float64):
    """
    Read one attribute off every vehicle into a NumPy array.

    np.fromiter over a C-level attrgetter, rather than np.array over a list
    comprehension: it skips the intermediate Python list entirely and writes
    into the output buffer directly. Same values, same dtype, roughly half the
    time -- which matters here because the stock is walked once per attribute,
    once per timestep.
    """
    return np.fromiter(map(attrgetter(name), list_vehicles), dtype=dtype, count=len(list_vehicles))


class SecondHandMerchant:
    def __init__(self, unique_id, parameters_second_hand):
        """
        Initialize the SecondHandMerchant with its unique ID and configuration.

        Args:
            unique_id (int): Unique identifier for the merchant.
            parameters_second_hand (dict): Dictionary containing configuration parameters such as age limits,
                                        discount rates, scrap prices, and random state.
        """
        self.id = unique_id
        self.cars_on_sale = []
        # Set by remove_car(), drained by compact_stock() -- see those methods.
        self._sold_ids = set()

        self.t_second_hand_cars = 0

        self.age_limit_second_hand = parameters_second_hand["age_limit_second_hand"]
        self.set_up_time_series_second_hand_car()

        self.r = parameters_second_hand["r"]
        self.max_num_cars = parameters_second_hand["max_num_cars"]
        self.burn_in_second_hand_market = parameters_second_hand["burn_in_second_hand_market"]

        self.random_state = parameters_second_hand["random_state"]

        #self.delta = parameters_second_hand["delta"]

        self.scrap_price = parameters_second_hand["scrap_price"]

        self.beta_segment_vec = parameters_second_hand["beta_segment_vals"] 
        self.gamma_segment_vec = parameters_second_hand["gamma_segment_vals"] 

        self.spent = 0
        self.income = 0
        self.assets = 0
        self.profit = 0
        self.scrap_loss = 0
        self.age_second_hand_car_removed = []

        self.removed_ev_emissions = []
        self.removed_ice_emissions = []
        
    def calc_median(self, beta_vec, gamma_vec):
        """
        Calculate and store the median values of beta and gamma across users.

        Args:
            beta_vec (np.ndarray): Array of beta values for users.
            gamma_vec (np.ndarray): Array of gamma values for users.
        """
        self.median_beta =  np.median(beta_vec)
        self.median_gamma = np.median(gamma_vec)

    def gen_vehicle_dict_vecs_second_hand(self, list_vehicles):
        """
        Generate attribute arrays from a list of second-hand vehicle objects.

        Args:
            list_vehicles (list): List of vehicle objects.

        Returns:
            dict: Dictionary mapping attribute names to NumPy arrays.
        """
            
        # "price" is deliberately left empty -- nothing downstream reads it off
        # this dict (the price is what calc_car_price_heuristic computes), and
        # the original never populated it either.
        vehicle_dict_vecs = {
            "Quality_a_t": _attr_array(list_vehicles, "Quality_a_t"),
            "Eff_omega_a_t": _attr_array(list_vehicles, "Eff_omega_a_t"),
            "price": np.array([]),
            "L_a_t": _attr_array(list_vehicles, "L_a_t", np.int64),
            "delta_P": _attr_array(list_vehicles, "delta_P"),
            "B": _attr_array(list_vehicles, "B")
        }

        return vehicle_dict_vecs

    def gen_vehicle_dict_vecs_new_cars(self, list_vehicles):
        """
        Generate attribute arrays from a list of new vehicle objects.

        Args:
            list_vehicles (list): List of vehicle objects.

        Returns:
            dict: Dictionary mapping attribute names to NumPy arrays.
        """
            
        vehicle_dict_vecs = {
            "Quality_a_t": _attr_array(list_vehicles, "Quality_a_t"),
            "Eff_omega_a_t": _attr_array(list_vehicles, "Eff_omega_a_t"),
            "price": _attr_array(list_vehicles, "price"),
            "B": _attr_array(list_vehicles, "B"),
            "transportType": _attr_array(list_vehicles, "transportType", np.int64)
        }

        return vehicle_dict_vecs

    def calc_car_price_heuristic(self, vehicle_dict_vecs_new_cars, vehicle_dict_vecs_second_hand_cars):
        """
        Estimate second-hand car prices using a heuristic based on similarity to new cars.

        Args:
            vehicle_dict_vecs_new_cars (dict): Attributes of new cars.
            vehicle_dict_vecs_second_hand_cars (dict): Attributes of second-hand cars.

        Returns:
            np.ndarray: Estimated prices for second-hand cars.
        """
        # Extract Quality, Efficiency, and Prices of first-hand cars
        first_hand_quality = vehicle_dict_vecs_new_cars["Quality_a_t"]
        first_hand_efficiency =  vehicle_dict_vecs_new_cars["Eff_omega_a_t"]
        first_hand_prices = vehicle_dict_vecs_new_cars["price"]
        first_hand_B = vehicle_dict_vecs_new_cars["B"]

        # Extract Quality, Efficiency, and Age of second-hand cars
        second_hand_quality = vehicle_dict_vecs_second_hand_cars["Quality_a_t"]
        second_hand_efficiency = vehicle_dict_vecs_second_hand_cars["Eff_omega_a_t"]
        second_hand_ages = vehicle_dict_vecs_second_hand_cars["L_a_t"]
        second_hand_delta_P = vehicle_dict_vecs_second_hand_cars["delta_P"]
        second_hand_B = vehicle_dict_vecs_second_hand_cars["B"]

        first_hand_quality_max = np.max(first_hand_quality)
        first_hand_efficiency_max = np.max(first_hand_efficiency)
        first_hand_B_max = np.max(first_hand_B)

        normalized_first_hand_quality = first_hand_quality / first_hand_quality_max 
        normalized_first_hand_efficiency = first_hand_efficiency / first_hand_efficiency_max 
        normalized_first_hand_B = first_hand_B/first_hand_B_max

        normalized_second_hand_quality = second_hand_quality  / first_hand_quality_max 
        normalized_second_hand_efficiency = second_hand_efficiency / first_hand_efficiency_max
        normalized_second_hand_B = second_hand_B / first_hand_B_max

        # Compute proximity (Euclidean distance) for all second-hand cars to all first-hand cars.
        # Accumulated into two (second_hand x new) buffers rather than the six
        # the original expression allocated; same operations in the same order.
        distances = normalized_second_hand_quality[:, np.newaxis] - normalized_first_hand_quality
        np.square(distances, out=distances)

        scratch = normalized_second_hand_efficiency[:, np.newaxis] - normalized_first_hand_efficiency
        np.square(scratch, out=scratch)
        distances += scratch

        np.subtract(normalized_second_hand_B[:, np.newaxis], normalized_first_hand_B, out=scratch)
        np.square(scratch, out=scratch)
        distances += scratch

        np.sqrt(distances, out=distances)

        # Find the closest first-hand car for each second-hand car
        closest_idxs = np.argmin(distances, axis=1)

        # Get the prices of the closest first-hand cars. The EV rebate only
        # exists on EVs, so it may only be netted off an EV anchor price -- an
        # ICE anchor is quoted at its full price. Matching is done on
        # (Quality, Efficiency, B), and B separates the two drivetrains (fuel
        # tank vs battery), so a used ICE matches an ICE anchor essentially
        # always; deducting the EV rebate there wrote the subsidy into the
        # resale value of the whole ICE fleet.
        matched_is_ev = vehicle_dict_vecs_new_cars["transportType"][closest_idxs] == 3
        rebate_deduction = np.where(matched_is_ev, self.rebate_calibration + self.rebate, 0.0)
        closest_prices = np.maximum(first_hand_prices[closest_idxs] - rebate_deduction, 0)

        # Adjust prices based on car age and depreciation. The gross series --
        # the same anchor with the EV rebate left in -- is returned alongside so
        # that update_stock_contents can retire cars on physical value rather
        # than on a price the rebate has driven to zero. See
        # Social_Network.calc_offer_prices_heursitic for the full reasoning.
        depreciation = (1 - second_hand_delta_P) ** second_hand_ages
        adjusted_prices = closest_prices * depreciation
        adjusted_prices_gross = first_hand_prices[closest_idxs] * depreciation

        return adjusted_prices, adjusted_prices_gross

    def update_stock_contents(self):
        """
        Update the stock of second-hand cars:
            - Remove overaged or underpriced cars.
            - Update prices using a heuristic method.
            - Enforce max inventory constraint.
        """
            
        #check len of list
        # Rebuilt in one pass rather than calling list.remove() while iterating
        # over the same list: mutating a list mid-for-loop shifts later
        # elements into the just-vacated slot, which the iterator then skips
        # over -- so two adjacent over-age cars would previously leave the
        # second one stuck in stock past its age limit.
        # The over-age test is a vectorised comparison and the surviving stock
        # is rebuilt with itertools.compress, so the Python-level loop only
        # runs over the cars actually being scrapped (a handful) instead of
        # over the whole stock. Ascending index order is preserved, so the
        # bookkeeping lists are appended to in exactly the original order.
        over_age = _attr_array(self.cars_on_sale, "second_hand_counter", np.int64) > self.age_limit_second_hand
        if over_age.any():
            for idx in np.flatnonzero(over_age):
                vehicle = self.cars_on_sale[idx]

                # Capture emissions before removal
                if vehicle.transportType == 2:
                    self.removed_ice_emissions.append(vehicle.total_emissions)
                else:
                    self.removed_ev_emissions.append(vehicle.total_emissions)

                self.age_second_hand_car_removed.append(vehicle.L_a_t)
                self.assets -= vehicle.cost_second_hand_merchant
                self.scrap_loss += vehicle.cost_second_hand_merchant

            self.cars_on_sale = list(compress(self.cars_on_sale, ~over_age))

        data_dicts_second_hand = self.gen_vehicle_dict_vecs_second_hand(self.cars_on_sale)
        # Calculate the price vector
        data_dicts_new_cars = self.gen_vehicle_dict_vecs_new_cars(self.vehicles_on_sale)

        price_vec, price_vec_gross = self.calc_car_price_heuristic(data_dicts_new_cars, data_dicts_second_hand)

        # Update the prices of the remaining cars
        for vehicle, price in zip(self.cars_on_sale, price_vec):
            vehicle.price = price

        # Remove cars below the scrap price. Same vectorised-mask treatment as
        # the over-age pass above.
        below_scrap_mask = price_vec_gross < self.scrap_price
        if below_scrap_mask.any():
            for idx in np.flatnonzero(below_scrap_mask):
                vehicle = self.cars_on_sale[idx]
                # Capture emissions before removal
                if vehicle.transportType == 2:
                    self.removed_ice_emissions.append(vehicle.total_emissions)
                else:
                    self.removed_ev_emissions.append(vehicle.total_emissions)

            self.cars_on_sale = list(compress(self.cars_on_sale, ~below_scrap_mask))

        #REMOVE EXCESS CARS
        if len(self.cars_on_sale) > self.max_num_cars:
            # Calculate how many cars to remove
            num_cars_to_remove = len(self.cars_on_sale) - self.max_num_cars
            # Randomly select cars to remove. Drawing the POSITIONS rather than
            # the objects consumes the identical random stream -- RandomState.
            # choice(n, k, replace=False) and choice(seq, k, replace=False) both
            # reduce to permutation(n)[:k] -- but lets the survivors be selected
            # with a boolean mask. The original tested `car not in cars_to_remove`
            # against a NumPy object array once per car in stock, i.e. a full
            # elementwise comparison per car: quadratic in the size of the stock,
            # which is exactly the quantity being increased here.
            idxs_to_remove = self.random_state.choice(
                len(self.cars_on_sale), num_cars_to_remove, replace=False
            )
            # Add ages of removed cars, in draw order as before
            self.age_second_hand_car_removed.extend(self.cars_on_sale[i].L_a_t for i in idxs_to_remove)

            keep_mask = np.ones(len(self.cars_on_sale), dtype=bool)
            keep_mask[idxs_to_remove] = False
            self.cars_on_sale = list(compress(self.cars_on_sale, keep_mask))


    def add_to_stock(self,vehicle):
        """
        Add a new second-hand vehicle to the merchant's stock.

        Args:
            vehicle (object): Vehicle object to add.
        """
            
        #add new car to stock
        vehicle.price = vehicle.price_second_hand_merchant
        vehicle.scenario = "second_hand"
        vehicle.second_hand_counter = 0
        self.cars_on_sale.append(vehicle)
    
    def remove_car(self, vehicle):
        """
        Mark a vehicle as sold out of the merchant's stock.

        The stock list itself is left alone until compact_stock() runs at the
        end of the users' choice loop. list.remove() is a linear scan, and it
        was being called once per second-hand sale against a stock of
        max_num_cars -- quadratic per timestep in exactly the quantity being
        scaled up. Nothing reads cars_on_sale between the first sale of a
        timestep and compact_stock(), so deferring is not observable.

        Args:
            vehicle (object): Vehicle object to remove.
        """
        self._sold_ids.add(id(vehicle))

    def compact_stock(self):
        """
        Drop the vehicles marked by remove_car() from the stock.

        Mutates cars_on_sale in place rather than rebinding it: the social
        network holds a reference to this same list object (controller.
        get_second_hand_cars returns it directly), and rebinding here would
        leave that reference pointing at the pre-sale stock.

        Order is unchanged -- survivors keep their relative order and cars
        taken in trade during the same step stay appended at the end, exactly
        as with the interleaved list.remove()/append() the original did.
        """
        if not self._sold_ids:
            return
        sold = self._sold_ids
        self.cars_on_sale[:] = [car for car in self.cars_on_sale if id(car) not in sold]
        sold.clear()

    def set_up_time_series_second_hand_car(self):
        """
        Initialize time series trackers for second-hand inventory metrics.
        """
            
        self.history_num_second_hand = []
        self.history_profit = []
        self.history_age_second_hand_car_removed = []

    def save_timeseries_second_hand_merchant(self):
        """
        Save current values of second-hand inventory, profit, and removal history to time series.
        """
            
        self.history_num_second_hand.append(len(self.cars_on_sale))
        self.history_profit.append(self.profit )
        self.history_age_second_hand_car_removed.append(self.age_second_hand_car_removed)

    def update_age_stock_prices_and_emissions_intensity(self, list_cars):
        """
        Increment the age and update fuel costs and emissions intensities for a list of cars.

        Args:
            list_cars (list): List of vehicle objects currently in stock.
        """
        # Loop-invariant lookups hoisted out: this walks the whole stock, so
        # every self.<attr> inside the body was one dict lookup per car per
        # timestep.
        age_the_stock = self.t_second_hand_cars > self.burn_in_second_hand_market
        gas_price = self.gas_price
        gas_cost_index = self.gas_cost_index
        gas_emissions_index = self.gas_emissions_index
        electricity_price = self.electricity_price
        electricity_emissions_intensity = self.electricity_emissions_intensity
        electricity_cost_index = self.electricity_cost_index
        electricity_emissions_index = self.electricity_emissions_index

        for car in list_cars:
            if age_the_stock:
                car.L_a_t += 1
                car.second_hand_counter += 1#UPDATE THE STEPS ITS BEEN HERE
            if car.transportType == 2:#ICE
                car.fuel_cost_c = gas_price
                car.cost_index = gas_cost_index
                car.emissions_index = gas_emissions_index
            else:#EV
                car.fuel_cost_c = electricity_price
                car.e_t = electricity_emissions_intensity
                car.cost_index = electricity_cost_index
                car.emissions_index = electricity_emissions_index

    def next_step(self,gas_price, electricity_price, electricity_emissions_intensity, vehicles_on_sale, rebate_calibration,rebate, gas_cost_index=0.0, gas_emissions_index=0.0, electricity_cost_index=0.0, electricity_emissions_index=0.0):
        """
        Advance the second-hand merchant's state by one timestep:
            - Update fuel prices and emission intensities.
            - Age vehicles and adjust stock post-burn-in.
            - Update profit and return current stock.

        Args:
            gas_price (float): Current gasoline price.
            electricity_price (float): Current electricity price.
            electricity_emissions_intensity (float): Grid carbon intensity.
            vehicles_on_sale (list): List of new cars for price reference.
            rebate_calibration (float): Policy calibration value for EV rebates.
            rebate (float): Rebate for used electric vehicles.

        Returns:
            list: Updated list of second-hand vehicles in stock.
        """
        self.t_second_hand_cars += 1

        self.gas_price =  gas_price
        self.electricity_price = electricity_price
        self.electricity_emissions_intensity = electricity_emissions_intensity
        self.vehicles_on_sale = vehicles_on_sale
        self.rebate_calibration = rebate_calibration
        self.rebate = rebate
        self.gas_cost_index = gas_cost_index
        self.gas_emissions_index = gas_emissions_index
        self.electricity_cost_index = electricity_cost_index
        self.electricity_emissions_index = electricity_emissions_index
        self.update_age_stock_prices_and_emissions_intensity(self.cars_on_sale)

        self.age_second_hand_car_removed = []

        if self.cars_on_sale and self.t_second_hand_cars > self.burn_in_second_hand_market:
            self.update_stock_contents()
        

        self.profit = self.income - self.spent

        return self.cars_on_sale