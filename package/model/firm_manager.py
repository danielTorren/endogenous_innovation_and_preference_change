"""Define firm manager that creates and manages different firms


Created: 22/12/2023
"""

# imports
import numpy as np
import random
from package.model.firm import Firm
from package.model.cars import Car
from package.model.nk_model import NKModel
# modules
class Firm_Manager:

    def __init__(self, parameters_firm_manager: dict, parameters_firm):
        
        self.t_firm_manager = 0
        self.parameters_firm = parameters_firm

        self.landscape_seed = parameters_firm_manager["landscape_seed"]
        self.init_tech_seed = parameters_firm_manager["init_tech_seed"] 
        self.J = parameters_firm_manager["J"]
        self.N = parameters_firm_manager["N"]
        self.K = parameters_firm_manager["K"]
        self.A = parameters_firm_manager["A"]# NUMBER OF ATTRIBUTES
        self.rho = parameters_firm_manager["rho"]# Correlation coefficients NEEDS TO BE AT LEAST 2 for 3 things from the same landscape
        self.save_timeseries_data_state = parameters_firm_manager["save_timeseries_data_state"]
        self.compression_factor_state = parameters_firm_manager["compression_factor_state"]
        self.init_tech_heterogenous_state = parameters_firm_manager["init_tech_heterogenous_state"]
        self.carbon_price = parameters_firm_manager["carbon_price"]
        self.num_individuals = parameters_firm_manager["num_individuals"]
        self.id_generator = parameters_firm_manager["IDGenerator_firms"]#USE TO GENERATATE UNIEQU IDS for EVERY SINGLE TECHNOLOGY
        self.gamma = parameters_firm_manager["gamma"]
        self.markup = parameters_firm_manager["markup"]

        #SEGMENTS, im calculating this twice should be a more efficient way to do it
        self.segment_number = int(parameters_firm["segment_number"])
        self.expected_segment_share = [1/self.segment_number]*self.segment_number#initally uniformly distributed
        self.segement_preference_bounds = np.linspace(0, 1, self.segment_number+1) 
        self.width_segment = self.segement_preference_bounds[1] - self.segement_preference_bounds[0]
        self.segement_preference = np.arange(self.width_segment/2, 1, self.width_segment)   #      np.linspace(0, 1, self.segment_number+1) #the plus 1 is so that theere are that number of divisions in the space

        #GEN INIT TECH
        np.random.seed(self.init_tech_seed)#set seed for numpy
        random.seed(self.init_tech_seed)#set seed for random

        self.init_tech_component_string = f"{random.getrandbits(self.N):=0{self.N}b}"
        if self.init_tech_heterogenous_state:
            decimal_value = int(self.init_tech_component_string, 2) 
            init_tech_component_string_list_N = self.invert_bits_one_at_a_time(decimal_value, len(self.init_tech_component_string))
            init_tech_component_string_list = np.random.choice(init_tech_component_string_list_N, self.J)
        
        #################################################################################################################################################
        #BELOW STUFF IS VARIED IN MONTECARLO SIMULATIONS
        #################################################################################################################################################

        self.nk_model = NKModel(self.N, self.K, self.A, self.rho, self.landscape_seed)
        
        if self.init_tech_heterogenous_state:
            attributes_fitness_list = [self.nk_model.calculate_fitness(x) for x in init_tech_component_string_list]
            self.init_tech_list = [Car(self.id_generator.get_new_id(), j,init_tech_component_string_list[j], attributes_fitness_list[j], choosen_tech_bool = 1) for j in range(self.J)]
        else:
            attributes_fitness = self.nk_model.calculate_fitness(self.init_tech_component_string)
            self.init_tech_list = [Car(self.id_generator.get_new_id(), j, self.init_tech_component_string, attributes_fitness, choosen_tech_bool = 1) for j in range(self.J)]

        self.firms_list = self.create_firms()

        self.cars_on_sale_all_firms = np.asarray([x.cars_on_sale for x in self.firms_list]).flatten()

        if self.save_timeseries_data_state:
            self.set_up_time_series_social_network()

    def invert_bits_one_at_a_time(self,decimal_value, length):
        # Convert decimal value to binary with leading zeros to achieve length N
        # binary_value = format(decimal_value, f'0{length}b')

        # Initialize an empty list to store inverted binary values
        inverted_binary_values = []

        # Iterate through each bit position
        for bit_position in range(length):
            """
            NEED TO UNDERSTAND BETTER HOW THIS WORKS!!
            """
            inverted_value = decimal_value^(1 << bit_position)

            # Convert the inverted decimal value to binary
            inverted_binary_value = format(inverted_value, f'0{length}b')

            # Append the inverted binary value to the list
            inverted_binary_values.append(inverted_binary_value)

        return inverted_binary_values

    def create_firms(self):

        firms_list = [Firm(
                self.parameters_firm,
                self.init_tech_list[j],
                j,
                self.nk_model
            ) 
            for j in range(self.J) 
        ]

        return firms_list

    def utility_buy_matrix(self, car_attributes_matrix):
        #IDEA IS TO DO THIS ALL IN ONE GO, ALL SEGMENTS AND ALL CARTIONS

        utilities = np.zeros_like(car_attributes_matrix)
        for i, pref in enumerate(self.segement_preference):
            utilities[:,i] = pref*car_attributes_matrix[:,1] + (1 -  pref) * (self.gamma * car_attributes_matrix[:,2] - (1 - self.gamma) *( (1 + self.markup) * car_attributes_matrix[:,0] + self.carbon_price*car_attributes_matrix[:,1]))
        return utilities
    
    def update_firms(self, segment_consumer_count):
        
        car_attributes_matrix = np.asarray([x.attributes_fitness for x in self.cars_on_sale_all_firms])
        self.utilities_competitors =  self.utility_buy_matrix(car_attributes_matrix)

        cars_on_sale_all_firms = []
        for j,firm in enumerate(self.firms_list):
            cars_on_sale = firm.next_step(self.carbon_price, segment_consumer_count, self.utilities_competitors)
            cars_on_sale_all_firms.extend(cars_on_sale)

        self.cars_on_sale_all_firms = np.asarray(cars_on_sale_all_firms)
        

    def set_up_time_series_social_network(self):
        self.history_cars_on_sale_all_firms = [self.cars_on_sale_all_firms]

    def save_timeseries_data_social_network(self):
        """
        Save time series data

        parameters_social_network
        ----------
        None

        Returns
        -------
        None
        """
        self.history_cars_on_sale_all_firms.append(self.cars_on_sale_all_firms)


    def next_step(self, carbon_price,  low_carbon_preference_arr):

        self.t_firm_manager  +=1
        self.carbon_price = carbon_price

        self.segment_consumer_count, __ = np.histogram(low_carbon_preference_arr, bins = self.segement_preference_bounds)

        self.update_firms(self.segment_consumer_count)
        
        if self.save_timeseries_data_state and (self.t_firm_manager % self.compression_factor_state == 0):
            self.save_timeseries_data_social_network()

        return self.cars_on_sale_all_firms

        