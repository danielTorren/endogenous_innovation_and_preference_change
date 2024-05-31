"""Define firm manager that creates and manages different firms


Created: 22/12/2023
"""

# imports
import numpy as np
import random
from package.model.May.May_firm import Firm
from package.model.May.May_cars import Car
from package.model.May.May_nk_model import NKModel
from package.model.May.May_centralized_ID_generator import IDGenerator
# modules
class Firm_Manager:

    def __init__(self, parameters_firm_manager: dict, parameters_firm):
        
        self.t_firm_manager = 0

        self.landscape_seed = parameters_firm_manager["landscape_seed"]
        self.init_tech_seed = parameters_firm_manager["init_tech_seed"] 

        self.J = parameters_firm_manager["J"]
        self.N = parameters_firm_manager["N"]
        self.K = parameters_firm_manager["K"]
        self.A = parameters_firm_manager["A"]# NUMBER OF ATTRIBUTES
        self.alpha = parameters_firm_manager["alpha"]
        self.rho = parameters_firm_manager["rho"]# Correlation coefficients NEEDS TO BE AT LEAST 2 for 3 things from the same landscape
        self.save_timeseries_data_state = parameters_firm_manager["save_timeseries_data_state"]
        self.compression_factor_state = parameters_firm_manager["compression_factor_state"]
        self.init_tech_heterogenous_state = parameters_firm_manager["init_tech_heterogenous_state"]
        #print("self.init_tech_heterogenous_state",self.init_tech_heterogenous_state)
        self.carbon_price = parameters_firm_manager["carbon_price"]
        self.nk_multiplier = parameters_firm_manager["nk_multiplier"]
        self.c_min = parameters_firm_manager["c_min"]
        self.c_max = parameters_firm_manager["c_max"]
        self.ei_min = parameters_firm_manager["ei_min"]
        self.ei_max = parameters_firm_manager["ei_max"]
        self.num_individuals = parameters_firm_manager["num_individuals"]
        self.survey_cost = self.num_individuals/(2*self.J)
        self.research_cost = self.num_individuals/(2*self.J)

        #SEGMENTS, im calculating this twice should be a more efficient way to do it
        self.segment_number = int(parameters_firm["segment_number"])
        self.expected_segment_share = [1/self.segment_number]*self.segment_number#initally uniformly distributed
        self.segement_preference_bounds = np.linspace(0, 1, self.segment_number+1) 
        self.width_segment = self.segement_preference_bounds[1] - self.segement_preference_bounds[0]
        self.segement_preference = np.arange(self.width_segment/2, 1, self.width_segment)   #      np.linspace(0, 1, self.segment_number+1) #the plus 1 is so that theere are that number of divisions in the space

        self.id_generator = IDGenerator#USE TO GENERATATE UNIEQU IDS for EVERY SINGLE TECHNOLOGY

        self.parameters_firm = parameters_firm

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
        
        self.parameters_firm["N"] = self.N
        self.parameters_firm["K"] = self.K
        self.parameters_firm["save_timeseries_data_state"] = self.save_timeseries_data_state
        self.parameters_firm["compression_factor_state"] = self.compression_factor_state
        self.parameters_firm["init_market_share"] = 1/self.J
        self.parameters_firm["J"] = self.J
        self.parameters_firm["carbon_price"] = self.carbon_price
        self.parameters_firm["id_generator"] = self.id_generator#pass the genertor on so that it can be used by the technology generation at the firm level

        self.firms_list = self.create_firms()

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
                j
            ) 
            for j in range(self.J) 
        ]

        return firms_list

    def utility_buy_matrix(self, car_attributes_matrix):
        #IDEA IS TO DO THIS ALL IN ONE GO, ALL SEGMENTS AND ALL CARTIONS
        utilities = self.segement_preference*car_attributes_matrix[:][1] + (1 -  self.segement_preference) * (self.gamma * car_attributes_matrix[:][2] - (1 - self.gamma) *( (1 + self.mu) * car_attributes_matrix[:][0] + self.carbon_price*car_attributes_matrix[:][1]))
        return utilities
    
    def update_firms(self, firm_count, segment_consumer_count):
        
        car_attributes_matrix = np.asarray([x.attributes_fitness for x in self.cars_on_sale_all_frims])
        self.utilities_competitors =  self.utility_buy_matrix(self, car_attributes_matrix)

        cars_on_sale_all_frims = []
        for j,firm in enumerate(self.firms_list):
            cars_on_sale = firm.next_step(firm_count, self.cars_on_sale_all_frims,  self.carbon_price, segment_consumer_count, self.utilities_competitors)
            cars_on_sale_all_frims.append(cars_on_sale)

        self.cars_on_sale_all_frims = cars_on_sale_all_frims
        self.price_vec = np.asarray([x.price for x in self.cars_on_sale_all_frims])

    def next_step(self, firm_count, carbon_price,  segment_consumer_count):

        self.t_firm_manager  +=1
        self.carbon_price = carbon_price

        self.update_firms(firm_count, segment_consumer_count)

        return self.cars_on_sale_all_frims, self.price_vec

        