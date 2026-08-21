import numpy as np

class NKModel_EV:
    """
    Represents an electric_vehicle NK fitness landscape model for simulating interdependent vehicle component configurations.

    This class models how different component combinations (designs) affect car attributes such as quality,
    efficiency, and production cost, incorporating user behavior and environmental policy parameters.
    """

    # `rho` is given in the paper's feature order: the baseline feature (production
    # cost, rho = 1 by construction) first, then quality, efficiency, battery size.
    # The landscape columns follow `min_vec`: quality, efficiency, cost, battery size.
    # RHO_TO_ATTR[rho_idx] gives the landscape column of that rho entry.
    RHO_TO_ATTR = (2, 0, 1, 3)


    def __init__(self, parameters):
        """
        Initialize the NKModel with design, attribute, and policy parameters.

        Args:
            parameters (dict): A dictionary containing:
                - N (int): Number of components in a car design.
                - K (int): Number of interdependencies per component.
                - A (int): Number of output attributes (e.g. quality, efficiency, cost).
                - rho (list[float]): Correlation coefficients for attribute interactions.
                - random_state_inputs (np.random.RandomState): Seeded random generator.
                - min_*, max_* (float): Attribute bounds.
                - fuel_tank (float): Fuel capacity (ICE) or battery size (EV).
                - r, delta (float): Discount and depreciation rates.
                - median_beta, median_gamma, median_nu (float): User preferences.
                - fuel_cost, e_t, d_mean (float): Energy price, emissions, and average distance.
                - alpha, zeta (float): Utility exponents.
                - production_emissions (float): Emissions from vehicle manufacturing.
                - prop_explore (float): Proportion of designs to explore.
                - init_price_multiplier (float): Cost scaling factor.
        """
        self.N = int(round(parameters["N"]))
        self.K = int(round(parameters["K"]))
        self.A = parameters["A"]
        self._k_powers = (2 ** np.arange(self.K, -1, -1)).astype(np.intp)
        self.rho = parameters["rho"] 

        self.random_state_inputs = parameters["random_state_inputs"]

        self.min_Quality = parameters["min_Quality"]
        self.min_Efficiency = parameters["min_Efficiency"]
        self.min_Cost = parameters["min_Cost"]
        self.min_Battery_size = parameters["min_Battery_size"]

        self.max_Quality = parameters["max_Quality"]
        self.max_Efficiency = parameters["max_Efficiency"]
        self.max_Cost = parameters["max_Cost"]
        self.max_Battery_size = parameters["max_Battery_size"]

        self.r = parameters["r"]
        self.delta =  parameters["delta"]
        self.median_beta = parameters["median_beta"]
        self.median_gamma = parameters["median_gamma"]
        self.median_nu = parameters["median_nu"]
        self.fuel_cost = parameters["fuel_cost_c"]
        self.e_t = parameters["e_t"]
        self.d_mean = parameters["d_mean"]
        self.alpha = parameters["alpha"]
        self.zeta = parameters["zeta"]
        self.E = parameters["production_emissions"]

        self.prop_explore = parameters["prop_explore"]
        self.init_price_multiplier = parameters["init_price_multiplier"]
        
        self.min_vec = np.asarray([self.min_Quality, self.min_Efficiency, self.min_Cost, self.min_Battery_size])
        self.max_vec = np.asarray([self.max_Quality, self.max_Efficiency, self.max_Cost, self.max_Battery_size])

        # Dispersion stretch, one factor per attribute -- see the same block in
        # nkModel_ICE for why it exists and why the keys are named rather than
        # positional. stretch_Quality is forced to match the ICE landscape's in
        # controller.__init__, alongside min_Quality/max_Quality, because Quality
        # is compared ACROSS drivetrains in the choice utility and two different
        # quality scales would not be comparable.
        self.stretch_vec = np.asarray([
            parameters.get("stretch_Quality", 1.0),
            parameters.get("stretch_Efficiency", 1.0),
            parameters.get("stretch_Cost", 1.0),
            parameters.get("stretch_Battery_size", 1.0),
        ])
        self.stretch_active = bool(np.any(self.stretch_vec != 1.0))
        self.mid_vec = 0.5*(self.min_vec + self.max_vec)

        self.fitness_landscape = self.generate_fitness_landscape()

        self.min_fitness_string, self.min_fitness, self.attributes_dict = self.find_min_fitness_string(self.prop_explore)

    def calc_present_utility_minimum_single(self, Q, omega, prod_cost, B):
        """assuem all cars are new to simplify, assume emissiosn intensities and prices from t = 0"""
        cost_multiplier = self.init_price_multiplier
        U = -cost_multiplier*prod_cost - self.median_gamma*self.E + self.median_beta*Q**self.alpha + self.median_nu*(B*omega)**self.zeta - self.d_mean*(((1+self.r)*(1-self.delta)*(self.fuel_cost + self.median_gamma*self.e_t))/(omega*(self.r - self.delta - self.r*self.delta)))
        return U

    def find_min_fitness_string(self, prop=1):
        """
        Finds the minimum fitness string in the NK landscape based on a sampled prop
        and stores all fitnesses.

        Args:
            prop (float): The prop of the landscape to explore (0-1).

        Returns:
            min_fitness_string (str): The binary string corresponding to the minimum fitness.
            min_fitness (float): The minimum fitness value.
            attributes_dict (dict): A dictionary mapping binary strings to their corresponding fitnesses.
        """
        
        if not (0 < prop <= 1):
            raise ValueError("Percentage must be between 0 and 1.")

        attributes_dict = {}

        total_landscape_size = 2 ** self.N
        sample_size = int((prop) * total_landscape_size)

        # Sample unique indices from the total landscape using numpy
        sampled_indices = self.random_state_inputs.choice(total_landscape_size, size=sample_size, replace=False)

        # Vectorize binary string conversion
        binary_strings = np.array([format(i, f'0{self.N}b') for i in sampled_indices])
        designs = np.array([list(map(int, binary_string)) for binary_string in binary_strings])

        # Vectorized fitness calculation
        attributes_list = self.calculate_fitness_vectorized(designs)
        fitness_values = self.calc_present_utility_minimum_single(attributes_list[:, 0], attributes_list[:, 1], attributes_list[:, 2], attributes_list[:, 3])

        # Find the minimum fitness and corresponding binary string
        min_index = np.argmin(fitness_values)
        min_fitness = fitness_values[min_index]
        min_fitness_string = binary_strings[min_index]

        # The sampled designs ranked worst-first. Placing every firm one bit-flip
        # from min_fitness_string puts them all in a single basin; this lets them
        # be spread across the bad end of the landscape instead.
        self.sampled_strings_ranked = binary_strings[np.argsort(fitness_values)]

        # Populate attributes_dict
        attributes_dict = dict(zip(binary_strings, attributes_list))

        return min_fitness_string, min_fitness, attributes_dict

    def calculate_fitness_vectorized(self, designs):
        """
        Vectorized calculation of fitness for multiple car designs.

        Args:
            designs (numpy.ndarray): 2D array representing multiple car designs. Shape: (num_designs, self.N)

        Returns:
            fitness_scaled (numpy.ndarray): 2D array representing the fitness vectors of the designs. Shape: (num_designs, self.A)
        """
        num_designs = designs.shape[0]
        fitness = np.zeros((num_designs, self.A))

        for n in range(self.N):
            col_idx = [(n + i) % self.N for i in range(self.K + 1)]
            k_indices = designs[:, col_idx] @ self._k_powers
            fitness += self.fitness_landscape[k_indices, n, :]

        average_fitness_components = fitness / self.N
        fitness_scaled = self._scale(average_fitness_components)

        return fitness_scaled

    def _scale(self, average_fitness_components):
        """
        Map averaged fitness components onto the attribute ranges.

        Without a stretch this is the original expression, unchanged and
        bit-identical. With one, each attribute's deviation from the midpoint is
        multiplied by its factor and the result is clipped back into
        [min, max] -- see the stretch_vec comment in __init__. Broadcasting
        handles both the (num_designs, A) and the (A,) case.
        """
        if not self.stretch_active:
            return self.min_vec + average_fitness_components*(self.max_vec - self.min_vec)

        deviation = (average_fitness_components - 0.5)*self.stretch_vec
        return np.clip(self.mid_vec + deviation*(self.max_vec - self.min_vec),
                       self.min_vec, self.max_vec)


    def calculate_fitness_single(self, design):
        """
        Calculate the fitness of a car design.

        Args:
        - design (numpy.ndarray): 1D array representing the state of each component.
                                   Shape: (self.N,)
        - landscapes (numpy.ndarray): 3D array containing fitness landscapes for all attributes.
                                      Shape: (2**(self.K+1), self.N, self.A)

        Returns:
        - fitness (numpy.ndarray): 1D array representing the fitness vec of the design.
                                    Shape: (self.A,)
        """
        
        if isinstance(design, str):
            design = np.array([int(c) for c in design], dtype=np.intp)
        else:
            design = np.asarray(design, dtype=np.intp)
        fitness = np.zeros(self.A)
        for n in range(self.N):
            col_idx = [(n + i) % self.N for i in range(self.K + 1)]
            k = int(design[col_idx] @ self._k_powers)
            fitness += self.fitness_landscape[k, n, :]
        average_fitness_components = fitness / self.N

        fitness_scaled = self._scale(average_fitness_components)
        return fitness_scaled
    
    def generate_fitness_landscape(self):
        L_cost = self.random_state_inputs.rand(2**(self.K+1), self.N, self.A)

        base_idx = self.RHO_TO_ATTR[0]

        for rho_idx in range(1, self.A):
            rho_val = self.rho[rho_idx]
            attr_idx = self.RHO_TO_ATTR[rho_idx]
            if rho_val != 0:
                num_to_sync = int(abs(rho_val) * self.N)
                if num_to_sync > 0:
                    sync_indices = self.random_state_inputs.choice(
                        self.N, size=num_to_sync, replace=False
                    )
                    if rho_val > 0:
                        # Positive correlation: copy directly
                        L_cost[:, sync_indices, attr_idx] = L_cost[:, sync_indices, base_idx]
                    else:
                        # Negative correlation: invert
                        L_cost[:, sync_indices, attr_idx] = 1 - L_cost[:, sync_indices, base_idx]

        return L_cost

    def invert_bits_one_at_a_time(self, decimal_value):
        """
        Generate all 1-bit neighbors of a binary string by flipping each bit.

        Args:
            decimal_value (int): Integer representation of a binary string.

        Returns:
            list[str]: List of binary strings differing by one bit.
        """
        inverted_binary_values = []
        for bit_position in range(self.N):
            inverted_value = decimal_value ^ (1 << bit_position)
            inverted_binary_value = format(inverted_value, f'0{self.N}b')
            inverted_binary_values.append(inverted_binary_value)

        return inverted_binary_values

    def retrieve_info(self, component_string):
        """
        Retrieve or calculate the attribute vector for a given component string.

        Args:
            component_string (str): Binary string representing the design.

        Returns:
            np.ndarray: Attribute vector (quality, efficiency, cost).
        """
            
        attributes = self.attributes_dict.get(component_string)
        
        if attributes is None:
            attributes = self.calculate_fitness_single(component_string)
            self.attributes_dict[component_string] = attributes

        return attributes

    
