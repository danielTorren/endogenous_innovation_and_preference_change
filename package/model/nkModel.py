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
        self.N = parameters["N"]
        self.K = parameters["K"]
        self.A = parameters["A"]
        self.rho = [1] + parameters["rho"]  # Adding 1 as the first correlation coefficient

        self.landscape_seed = parameters["landscape_seed"]
        np.random.seed(self.landscape_seed)#set seed for numpy

        self.min_max_Quality = parameters["min_max_Quality"]
        self.min_max_Efficiency = parameters["min_max_Efficiency"]
        self.min_max_Cost = parameters["min_max_Cost"]

        self.min_vec = np.asarray([self.min_max_Quality[0],self.min_max_Efficiency[0], self.min_max_Cost[0]])
        self.max_vec = np.asarray([self.min_max_Quality[1],self.min_max_Efficiency[1], self.min_max_Cost[1]])

        self.fitness_landscapes = self.generate_fitness_landscapes()

    def generate_fitness_landscapes(self):
        """
        Generate fitness landscapes for all attributes.

        Returns:
        - L (numpy.ndarray): 3D array containing fitness landscapes for all attributes.
                             Shape: (2**(self.K+1), self.N, self.A)
        """    

        #CHECK SEED STUFF FOR REPRODUCIBILITY 


        L = np.random.rand(2**(self.K+1), self.N, self.A)
        L_efficiency = np.random.rand(2**(self.K+1), self.N, self.A)
        L_cost = np.random.rand(2**(self.K+1), self.N, self.A)
        # Iterate over each attribute starting from the second one
        
        #DONT HAVE TO ANYTHING FOR QUALITY, its the one we base everything off of

        #FOR EFFICIENCY CORRELATION IS 0, so replace that entry
        L[:, :, 1] = L_efficiency[:, :, 1]

        #FOR COST ITS CORRELATION IS 0.5
        Q_a_size = int(abs(self.rho[2]) * self.N)#THIS IS WHAT SETS THE TIGHTNESS OF THE CORRELATION, AS YOU SELECT HOW MANY TIME YOU SWITCH OUT!!!
        #Q_a_size = int(abs(self.rho[2]) * self.N + 0.5)#THIS IS WHAT SETS THE TIGHTNESS OF THE CORRELATION, AS YOU SELECT HOW MANY TIME YOU SWITCH OUT!!!
        Q_a = np.random.choice(self.N, size=Q_a_size, replace=False)
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
        
        #print(self.N, self.K, self.A)
        #quit()
        fitness = np.zeros(self.A)
        for a in range(self.A):
            for n in range(self.N):
                k = int(''.join([str(design[(n+i) % self.N]) for i in range(self.K+1)]), 2) #REVIST THIS AND UNDERSTAND WHAT IS GOING ON BETTER
                fitness[a] +=  self.fitness_landscapes[k, n, a]

        average_fitness_components = fitness / self.N
        #fitness_scaled = self.min_vec + ((average_fitness_components*self.max_vec-self.min_vec)/(self.max_vec - self.min_vec))
        
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
    
