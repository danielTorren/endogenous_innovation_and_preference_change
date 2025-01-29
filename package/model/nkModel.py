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
        self.rho = [1] + parameters["rho"]  # Adding 1 as the first correlation coefficient

        self.random_state_NK = np.random.RandomState(parameters["landscape_seed"])  # Local random state

        self.min_Quality = parameters["min_Quality"]
        self.min_Efficiency = parameters["min_Efficiency"]
        self.min_Cost = parameters["min_Cost"]

        self.max_Quality = parameters["max_Quality"]
        self.max_Efficiency = parameters["max_Efficiency"]
        self.max_Cost = parameters["max_Cost"]


        self.r = parameters["r"]
        self.delta =  parameters["delta"]
        self.median_beta = parameters["median_beta"]
        self.median_gamma = parameters["median_gamma"]
        self.fuel_cost = parameters["fuel_cost"]
        self.e_t = parameters["e_t"]

        #self.nu = parameters["nu"]

        self.min_vec = np.asarray([self.min_Quality,self.min_Efficiency, self.min_Cost])
        self.max_vec = np.asarray([self.max_Quality,self.max_Efficiency, self.max_Cost])

        self.fitness_landscape = self.generate_fitness_landscape()

        self.min_fitness_string, self.min_fitness, self.attributes_dict = self.find_min_fitness_string()

    def calc_present_utility_minimum_single(self, quality, eff, prod_cost):
        """assuem all cars are new to simplify, assume 0 carbon price, assume emissiosn intensities and prices from t = 0"""
        #DISTANCE
                # Compute numerator for all vehicles
        beta = self.median_beta
        gamma = self.median_gamma

        #present UTILITY
        # Compute cost component based on transport type, with conditional operations
        X = (beta * self.fuel_cost + gamma * self.e_t)/ eff

        # Compute commuting utility for individual-vehicle pairs
        driving_utility = np.log(1 + quality/X)
        

        # Save the base utility
        B = driving_utility*((1+self.r)/(self.r + self.delta))
        #print("B - beta*prod_cost", B - beta*prod_cost)
        approx_fitness = B - beta*prod_cost
        
        #print("approx_fitness",approx_fitness)
        return approx_fitness


    def find_min_fitness_string(self):
        """
        Finds the minimum fitness string in the NK landscape and stores all fitnesses.

        Args:
            nk_model: An instance of the NKModel class.

        Returns:
            min_fitness_string: The binary string corresponding to the minimum fitness.
            min_fitness: The minimum fitness value.
            fitness_dict: A dictionary mapping binary strings to their corresponding fitnesses.
        """

        attributes_dict = {}
        min_fitness = float('inf')
        min_fitness_string = None

        for i in range(2**self.N):
            binary_string = format(i, f'0{self.N}b')
            design = np.array(list(map(int, binary_string)))
            attributes = self.calculate_fitness(design)
            fitness =  self.calc_present_utility_minimum_single(attributes[0],  attributes[1], attributes[2])
            #fitness = np.sum(attributes)
            attributes_dict[binary_string] = attributes
            if fitness < min_fitness:
                min_fitness = fitness
                min_fitness_string = binary_string


        return min_fitness_string, min_fitness, attributes_dict

    def generate_fitness_landscape(self):
        """
        Generate fitness landscapes for all attributes.

        Returns:
        - L (numpy.ndarray): 3D array containing fitness landscapes for all attributes.
                             Shape: (2**(self.K+1), self.N, self.A)
        """    

        #CHECK SEED STUFF FOR REPRODUCIBILITY 


        L = self.random_state_NK.rand(2**(self.K+1), self.N, self.A)
        L_efficiency = self.random_state_NK.rand(2**(self.K+1), self.N, self.A)
        L_cost = self.random_state_NK.rand(2**(self.K+1), self.N, self.A)
        # Iterate over each attribute starting from the second one
        
        #DONT HAVE TO ANYTHING FOR QUALITY, its the one we base everything off of

        #FOR EFFICIENCY CORRELATION IS 0, so replace that entry
        L[:, :, 1] = L_efficiency[:, :, 1]

        #FOR COST ITS CORRELATION IS 0.5
        Q_a_size = int(abs(self.rho[2]) * self.N)#THIS IS WHAT SETS THE TIGHTNESS OF THE CORRELATION, AS YOU SELECT HOW MANY TIME YOU SWITCH OUT!!!
        #Q_a_size = int(abs(self.rho[2]) * self.N + 0.5)#THIS IS WHAT SETS THE TIGHTNESS OF THE CORRELATION, AS YOU SELECT HOW MANY TIME YOU SWITCH OUT!!!
        Q_a = self.random_state_NK.choice(self.N, size=Q_a_size, replace=False)
        L[:, Q_a, 2] = L_cost[:, Q_a, 2]  # Copy fitness contribution from attribute 1

        return L
    
    def calculate_fitness(self, design):
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

    def invert_bits_one_at_a_time(self, decimal_value):
        """THIS IS ONLY USED ONCE I THINK"""
        inverted_binary_values = []
        for bit_position in range(self.N):
            inverted_value = decimal_value ^ (1 << bit_position)
            inverted_binary_value = format(inverted_value, f'0{self.N}b')
            inverted_binary_values.append(inverted_binary_value)
        return inverted_binary_values
    
