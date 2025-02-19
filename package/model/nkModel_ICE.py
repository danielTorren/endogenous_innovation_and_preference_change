import numpy as np

class NKModel:
    def __init__(self, parameters):
        """
        Initialize the NKModel.

        Args:
        - N (int): Number of components in a car design.
        - K (int): Level of interdependence between components.
        - A (int): Number of attributes influencing the fitness.
        - rho (list): List of length A containing correlation coefficients.
        """
        self.N = int(round(parameters["N"]))
        self.K = int(round(parameters["K"]))
        self.A = parameters["A"]
        self.rho = parameters["rho"] 

        self.random_state_NK = np.random.RandomState(parameters["landscape_seed"])  # Local random state

        self.min_Quality = parameters["min_Quality"]
        self.min_Efficiency = parameters["min_Efficiency"]
        self.min_Cost = parameters["min_Cost"]

        self.max_Quality = parameters["max_Quality"]
        self.max_Efficiency = parameters["max_Efficiency"]
        self.max_Cost = parameters["max_Cost"]

        self.fuel_tank = parameters["fuel_tank"] 

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
        
        self.min_vec = np.asarray([self.min_Quality, self.min_Efficiency, self.min_Cost])
        self.max_vec = np.asarray([self.max_Quality, self.max_Efficiency, self.max_Cost])

        self.fitness_landscape = self.generate_fitness_landscape()

        self.min_fitness_string, self.min_fitness, self.attributes_dict = self.find_min_fitness_string(self.prop_explore)

    def calc_present_utility_minimum_single(self, Q, omega, prod_cost, B):
        """assuem all cars are new to simplify, assume emissiosn intensities and prices from t = 0"""
        cost_multiplier = self.init_price_multiplier
        U = -cost_multiplier*prod_cost - self.median_gamma*self.E + ((1+self.r)*(self.median_beta*Q**self.alpha))/self.r + ((1+self.r)*(self.median_nu*(B*omega)**self.zeta))/(1 + self.r - (1- self.delta)**self.zeta) - self.d_mean*(((1+self.r)*(1-self.delta)*(self.fuel_cost + self.median_gamma*self.e_t))/(omega*(self.r - self.delta - self.r*self.delta)))
        
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
        sampled_indices = self.random_state_NK.choice(total_landscape_size, size=sample_size, replace=False)

        # Vectorize binary string conversion
        binary_strings = np.array([format(i, f'0{self.N}b') for i in sampled_indices])
        designs = np.array([list(map(int, binary_string)) for binary_string in binary_strings])

        # Vectorized fitness calculation
        attributes_list = self.calculate_fitness_vectorized(designs)
        fitness_values = self.calc_present_utility_minimum_single(attributes_list[:, 0], attributes_list[:, 1], attributes_list[:, 2], self.fuel_tank)

        # Find the minimum fitness and corresponding binary string
        min_index = np.argmin(fitness_values)
        min_fitness = fitness_values[min_index]
        min_fitness_string = binary_strings[min_index]

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
            k_indices = np.array([
                int(''.join(map(str, (design[(n + i) % self.N] for i in range(self.K + 1)))), 2) for design in designs
            ])
            fitness += self.fitness_landscape[k_indices, n, :]

        average_fitness_components = fitness / self.N
        fitness_scaled = self.min_vec + average_fitness_components * (self.max_vec - self.min_vec)

        return fitness_scaled
    
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
        
        fitness = np.zeros(self.A)
        for a in range(self.A):
            for n in range(self.N):
                k = int(''.join([str(design[(n+i) % self.N]) for i in range(self.K+1)]), 2) #REVIST THIS AND UNDERSTAND WHAT IS GOING ON BETTER
                fitness[a] +=  self.fitness_landscape[k, n, a]
        average_fitness_components = fitness / self.N

        fitness_scaled = self.min_vec + average_fitness_components * (self.max_vec-self.min_vec)
        return fitness_scaled

    def generate_fitness_landscape(self):
        """
        Generate fitness landscapes for all attributes.

        Returns:
        - L (numpy.ndarray): 3D array containing fitness landscapes for all attributes.
                             Shape: (2**(self.K+1), self.N, self.A)
        """    

        #CHECK SEED STUFF FOR REPRODUCIBILITY 
        L_cost = self.random_state_NK.rand(2**(self.K+1), self.N, self.A)
        L_quality = self.random_state_NK.rand(2**(self.K+1), self.N, self.A)
        L_efficiency = self.random_state_NK.rand(2**(self.K+1), self.N, self.A)
        
        #(Cost, quality, efficiency, battery)
        # Iterate over each attribute

        if self.rho[1] == 0:
            L_cost[:, :, 1] = L_quality[:, :, 1]
        else:
            #FOR COST ITS CORRELATION IS 0.5
            Q_a_size = int(abs(self.rho[2]) * self.N)#THIS IS WHAT SETS THE TIGHTNESS OF THE CORRELATION, AS YOU SELECT HOW MANY TIME YOU SWITCH OUT!!!
            Q_a = self.random_state_NK.choice(self.N, size=Q_a_size, replace=False)
            L_cost[:, Q_a, 1] = L_quality[:, Q_a, 1]  # Copy fitness contribution from attribute 1

        if self.rho[2] == 0:
            L_cost[:, :, 1] = L_efficiency[:, :, 2]
        else:
            #FOR COST ITS CORRELATION IS 0.5
            Q_a_size = int(abs(self.rho[1]) * self.N)#THIS IS WHAT SETS THE TIGHTNESS OF THE CORRELATION, AS YOU SELECT HOW MANY TIME YOU SWITCH OUT!!!
            #Q_a_size = int(abs(self.rho[2]) * self.N + 0.5)#THIS IS WHAT SETS THE TIGHTNESS OF THE CORRELATION, AS YOU SELECT HOW MANY TIME YOU SWITCH OUT!!!
            Q_a = self.random_state_NK.choice(self.N, size=Q_a_size, replace=False)
            L_cost[:, Q_a, 2] = L_efficiency[:, Q_a, 2]  # Copy fitness contribution from attribute 1

        return L_cost
    
    def invert_bits_one_at_a_time(self, decimal_value):
        """THIS IS ONLY USED ONCE I THINK"""
        inverted_binary_values = []
        for bit_position in range(self.N):
            inverted_value = decimal_value ^ (1 << bit_position)
            inverted_binary_value = format(inverted_value, f'0{self.N}b')
            inverted_binary_values.append(inverted_binary_value)

        return inverted_binary_values

    def retrieve_info(self, component_string):
        attributes = self.attributes_dict.get(component_string)
        
        if attributes is None:
            attributes = self.calculate_fitness_single(component_string)
            self.attributes_dict[component_string] = attributes

        return attributes

    
