"""Define firms that contain info about technology and percieved consumer preferences


Created: 21/12/2023
"""

# imports
import numpy as np
import random
from package.model.May.May_cars import Car

class Firm:
    def __init__(self, parameters_firm, init_tech, firm_id, nk_model):
        
        self.t_firm = 0

        self.firm_id = firm_id#this is used when indexing firms stuff
        self.id_generator = parameters_firm["id_generator"]
        self.save_timeseries_data_state = parameters_firm["save_timeseries_data_state"]
        self.compression_factor_state = parameters_firm["compression_factor_state"]
        self.static_tech_state = parameters_firm["static_tech_state"]
        self.markup = parameters_firm["markup_init"]#variable
        self.J = parameters_firm["J"]
        self.carbon_price = parameters_firm["carbon_price"]
        self.memory_cap = parameters_firm["memory_cap"]
        self.num_individuals = int(round(parameters_firm["num_individuals"]))
        self.gamma = parameters_firm["gamma"]
        self.kappa = parameters_firm["kappa"]
        self.mu = parameters_firm["mu"]

        #SEGMENTS
        self.segment_number = int(parameters_firm["segment_number"])
        self.expected_segment_share = [1/self.segment_number]*self.segment_number#initally uniformly distributed
        self.segement_preference_bounds = np.linspace(0, 1, self.segment_number+1) 
        self.width_segment = self.segement_preference_bounds[1] - self.segement_preference_bounds[0]
        self.segement_preference = np.arange(self.width_segment/2, 1, self.width_segment)   #      np.linspace(0, 1, self.segment_number+1) #the plus 1 is so that theere are that number of divisions in the space

        #RANKS
        self.rank_number = int(parameters_firm["rank_number"])
        self.rank_bounds = np.linspace(0, self.markup*self.num_individuals, self.rank_number) 

        self.nk_model = nk_model

        self.N = parameters_firm["N"]
        self.K = parameters_firm["K"]

        self.init_tech = init_tech 
        self.list_technology_memory = [self.init_tech]
        self.list_technology_memory_strings = [init_tech.component_string]
        self.cars_on_sale = [init_tech]
    
###########################################################################################################################################################
    #REASERACH TECHNOLOGY
    
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
    
    def calc_neighbouring_technologies_tumor(self):
        
        #NEED TO GET ALL THE NEIGHBOURING TECHS OF THE MEMORY
        decimal_value_memory_list = [x.decimal_value for x in self.list_technology_memory]
        unfiltered_list_neighouring_technologies_strings = list(set([self.invert_bits_one_at_a_time(x, self.N) for x in decimal_value_memory_list]))#the set makes it a unique list, the list allows it to be used in the next step
        self.list_neighouring_technologies_strings = [i for i in unfiltered_list_neighouring_technologies_strings if i not in self.list_technology_memory_strings]

    ##############################################################################

    def utility_buy_matrix(self, car_attributes_matrix):
        #IDEA IS TO DO THIS ALL IN ONE GO, ALL SEGMENTS AND ALL CARTIONS
        utilities = self.segement_preference*car_attributes_matrix[:][1] + (1 -  self.segement_preference) * (self.gamma * car_attributes_matrix[:][2] - (1 - self.gamma) * ((1 + self.mu) * car_attributes_matrix[:][0] + self.carbon_price*car_attributes_matrix[:][1]))
        return utilities

    def calculate_profitability_alternatives(self, utilities_competitors):

        #Take strings and workout their profitability
        alternatives_attributes_matrix = np.asarray([self.nk_model.calculate_fitness(x) for x in self.list_neighouring_technologies_strings])
        
        #For each segment need ot caluclate the probability of purchasing the car
        utilities_neighbour = self.utility_buy_matrix(alternatives_attributes_matrix) 
        #utilities_competitors = self.utility_buy_matrix(competitors_attributes_matrix) 
        market_options_utilities = np.concatenate((utilities_neighbour , utilities_competitors ), 1)#join as the probabilities are relative to all other market options
        
        utilities_neighbour[utilities_neighbour < 0] = 0#IF NEGATIVE UTILITY PUT IT AT 0
        market_options_utilities[market_options_utilities < 0] = 0#IF NEGATIVE UTILITY PUT IT AT 0

        denominators = np.sum(market_options_utilities ** self.kappa)

        if 0 not in denominators:
            alternatives_probability_buy_car = utilities_neighbour ** self.kappa / denominators#CHECK FOR HAVING CURRENT TECH IN HERE NOT TO DOUBLE COUNT
        else:
            alternatives_probability_buy_car = np.zeros_like(utilities_neighbour, dtype=float)#i only change the ones that are not zero
            non_zero_mask = denominators != 0
            alternatives_probability_buy_car[non_zero_mask] = (utilities_neighbour[non_zero_mask] ** self.kappa) / denominators[non_zero_mask]

        expected_number_customer = self.segment_consumer_count*alternatives_probability_buy_car
        self.expected_profit_research_alternatives = self.markup*alternatives_attributes_matrix[:,0]*np.sum(expected_number_customer, axis= 1)
            #self.expected_profit_research_alternatives = np.asarray([[0]*self.segment_number])

    def last_tech_profitability(self, utilities_competitors):
        #CALCUALTE THE PREDICTED PROFITABILITY OF TECHNOLOGY RESEARCHED IN PAST STEP
        self.last_tech_researched = self.list_technology_memory[-1]
        last_tech_fitness = self.nk_model.calculate_fitness(self.last_tech_researched) 
        utility_last_tech = np.asarray([self.utility_buy_matrix(last_tech_fitness)])
        last_tech_market_options_utilities = np.concatenate((utility_last_tech , utilities_competitors ), 1)#join as the probabilities are relative to all other market options
        last_tech_market_options_utilities[last_tech_market_options_utilities < 0] = 0

        denominators = np.sum(last_tech_market_options_utilities ** self.kappa)

        if 0 not in denominators:
            alternatives_probability_buy_car = utility_last_tech ** self.kappa /denominators#CHECK FOR HAVING CURRENT TECH IN HERE NOT TO DOUBLE COUNT
            expected_number_customer = self.segment_consumer_count*alternatives_probability_buy_car
            self.last_tech_expected_profit = self.markup*last_tech_fitness[0]*np.sum(expected_number_customer, axis= 1)
        else:
            self.last_tech_expected_profit = np.asarray([0]*self.segment_number)

    ###################################################################################

    def rank_options(self):
        #RANK THE TECH
        #split up the 
        self.ranked_alternatives = []
        for tech, profitability in zip(self.list_neighouring_technologies_strings, self.expected_profit_research_alternatives):
            rank = None
            for r in range(1, self.rank_number + 1):
                if profitability < r / self.rank_number:
                    rank = r
                    break
            self.ranked_alternatives.append((tech, rank))
    
    def rank_last_tech(self):            
        
        for r in range(1, self.rank_number + 1):
                if self.last_tech_expected_profit < r / self.rank_number:
                    self.last_tech_rank = r
                    break
    
    def add_new_tech_memory(self,chosen_technology):
        self.list_technology_memory.append(chosen_technology)
        self.list_technology_memory_strings.append(chosen_technology.component_string)

    def select_alternative_technology(self):
        #SELECT TECHNOLOGIES FROM ANY RANK THAT ITS ABOVE CURRENT
        #CREATE A LIST OF POSSIBLE TECHNOLOGIES
        tech_alternative_options = []
        while not tech_alternative_options:
            tech_alternative_options = [tech for tech, rank in self.ranked_alternatives if rank >= self.last_tech_rank]
                    
        if tech_alternative_options:
            selected_technology_string = random.choice(tech_alternative_options)#this is not empty
            unique_tech_id = self.id_generator.get_new_id()
            attribute_selected_tech  = self.nk_model.calculate_fitness(selected_technology_string) 
            self.researched_technology = Car(unique_tech_id,self.firm_id, selected_technology_string, attribute_selected_tech, choosen_tech_bool = 0) 
            self.add_new_tech_memory(self.researched_technology, selected_technology_string)
    
    def research_technology(self, utilities_competitors):
        self.calc_neighbouring_technologies_tumor()#Now i know what the possible neighbouring strings are
        self.calculate_profitability_alternatives(utilities_competitors)
        self.last_tech_profitability(utilities_competitors)
        self.rank_options()
        self.rank_last_tech()
        self.select_alternative_technology()

    ##############################################################################################################
    #CHOOSING TECH FROM MEMORY

    def calculate_profitability_memory(self, utilities_competitors):

        #Take strings and workout their profitability
        alternatives_attributes_matrix = np.asarray([self.nk_model.calculate_fitness(x) for x in self.list_technology_memory_strings])
        
        #For each segment need ot caluclate the probability of purchasing the car
        utilities_memory = self.utility_buy_matrix(alternatives_attributes_matrix) 
        market_options_utilities = np.concatenate((utilities_memory , utilities_competitors ), 1)#join as the probabilities are relative to all other market options
        
        utilities_memory[utilities_memory < 0] = 0#IF NEGATIVE UTILITY PUT IT AT 0
        market_options_utilities[market_options_utilities < 0] = 0#IF NEGATIVE UTILITY PUT IT AT 0

        denominators = np.sum(market_options_utilities ** self.kappa, )

        if 0 not in denominators:
            alternatives_probability_buy_car = utilities_memory ** self.kappa / denominators#CHECK FOR HAVING CURRENT TECH IN HERE NOT TO DOUBLE COUNT
        else:
            #FOR SEGMENTS WHERE IS IT not zero do the calc else where is 0
            alternatives_probability_buy_car = np.zeros_like(utilities_memory, dtype=float)#i only change the ones that are not zero
            non_zero_mask = denominators != 0
            alternatives_probability_buy_car[non_zero_mask] = (utilities_memory[non_zero_mask] ** self.kappa) / denominators[non_zero_mask]

        expected_number_customer = self.segment_consumer_count*alternatives_probability_buy_car
        expected_profit_research_alternatives = self.markup*alternatives_attributes_matrix[:,0]*np.sum(expected_number_customer, axis= 1)
        
        return expected_profit_research_alternatives

    def update_memory(self):

        for car in self.list_technology_memory:
            if car not in self.cars_on_sale:
                car.choosen_tech_bool = 0
            else:
                car.choosen_tech_bool = 1

        list(map(lambda technology: technology.update_timer(), self.list_technology_memory))#update_timer on all tech
        #remove additional technologies
        if len(self.list_technology_memory) > self.memory_cap:
            timer_max = max(x.timer for x in self.list_technology_memory)
            self.list_technology_memory = list( filter(lambda x: x.timer == timer_max, self.list_technology_memory) ) 

    def choose_technologies(self, utilities_competitors):        
        #evaluate for which car is best for which techology
        expected_profit_research_alternatives = self.calculate_profitability_memory(utilities_competitors)

        max_profit = np.max(expected_profit_research_alternatives, axis = 1)
        cars_on_sale = np.zeros(self.segment_number)
        for i in max_profit:#Along the different segments produce different cars
            max_index = expected_profit_research_alternatives.index(max_profit[i])
            cars_on_sale[i] = self.list_technology_memory[max_index]
        
        self.cars_on_sale = list(set(cars_on_sale))#make it a unique list

        self.update_memory()

    ##############################################################################################################
    #FORWARD
        
    def set_up_time_series_firm(self):
        self.history_length_memory_list = [len(self.list_technology_memory)]

    def save_timeseries_data_firm(self):
        """
        Save time series data
        ----------
        None

        Returns
        -------
        None
        """

        self.history_length_memory_list.append(len(self.list_technology_memory))

    def next_step(self, firm_count, cars_vec, carbon_price, segment_consumer_count, utilities_competitors) -> None:
        
        self.t_firm +=1
        
        self.carbon_price = carbon_price
        self.segment_consumer_count = segment_consumer_count

        self.firm_count = firm_count[self.firm_id]
        self.cars_vec = cars_vec[self.firm_id]
        
        
        #RESEARCH TECH
        if not self.static_tech_state:
            self.research_technology(utilities_competitors)

        #CHOOSE CARS FROM MEMORY
        self.cars_on_sale = self.choose_technologies(utilities_competitors)

        if self.save_timeseries_data_state and self.t_firm == 1:
            self.set_up_time_series_firm()
        elif self.save_timeseries_data_state and (self.t_firm % self.compression_factor_state == 0):
            self.save_timeseries_data_firm()

        return self.cars_on_sale