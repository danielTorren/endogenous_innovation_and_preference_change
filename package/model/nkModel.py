import numpy as np

class NKModel:
    def __init__(self, N, K, A, rho , landscape_seed):
        """
        Initialize the NKModel.

        Args:
        - N (int): Number of components in a car design.
        - K (int): Level of interdependence between components.
        - A (int): Number of attributes influencing the fitness.
        - rho (list): List of length A containing correlation coefficients.
        """
        self.N = N
        self.K = K
        self.A = A
        self.rho = [1] + rho  # Adding 1 as the first correlation coefficient
        self.landscape_seed = landscape_seed
        np.random.seed(self.landscape_seed)#set seed for numpy

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
        
        # Iterate over each attribute starting from the second one
        for a in range(1, self.A):
            Q_a_size = int(abs(self.rho[a]) * self.N + 0.5)#THIS IS WHAT SETS THE TIGHTNESS OF THE CORRELATION, AS YOU SELECT HOW MANY TIME YOU SWITCH OUT!!!
            Q_a = np.random.choice(self.N, size=Q_a_size, replace=False)
            if self.rho[a] > 0:
                L[:, Q_a, a] = L[:, Q_a, 0]  # Copy fitness contribution from attribute 1
            else:
                L[:, Q_a, a] = 1 - L[:, Q_a, 0]  # Copy inverse fitness contribution from attribute 1
        
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
        - fitness (numpy.ndarray): 1D array representing the fitness vector of the design.
                                    Shape: (self.A,)
        """
        
        fitness = np.zeros(self.A)
        for a in range(self.A):
            for n in range(self.N):
                k = int(''.join([str(design[(n+i) % self.N]) for i in range(self.K+1)]), 2) #REVIST THIS AND UNDERSTAND WHAT IS GOING ON BETTER
                fitness[a] +=  self.fitness_landscapes[k, n, a]

        return fitness / self.N

    def invert_bits_one_at_a_time(self, decimal_value):
        """THIS IS ONLY USED ONCE I THINK"""
        inverted_binary_values = []
        for bit_position in range(self.N):
            inverted_value = decimal_value ^ (1 << bit_position)
            inverted_binary_value = format(inverted_value, f'0{self.N}b')
            inverted_binary_values.append(inverted_binary_value)
        return inverted_binary_values
    
