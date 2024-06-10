"""Define firms that contain info about technology and percieved consumer preferences


Created: 21/12/2023
"""

# imports
import numpy as np
import random
from package.model.cars import Car

class Firm:
    def __init__(self, parameters_firm, init_tech, firm_id, nk_model):
        
        self.t_firm = 0

        self.firm_id = firm_id#this is used when indexing firms stuff
        self.id_generator = parameters_firm["id_generator"]
        self.save_timeseries_data_state = parameters_firm["save_timeseries_data_state"]
        self.compression_factor_state = parameters_firm["compression_factor_state"]
        self.static_tech_state = parameters_firm["static_tech_state"]
        self.markup = parameters_firm["markup"]#variable
        self.J = parameters_firm["J"]
        self.carbon_price = parameters_firm["carbon_price"]
        self.memory_cap = parameters_firm["memory_cap"]
        self.num_individuals = parameters_firm["num_individuals"]
        self.gamma = parameters_firm["gamma"]
        self.kappa = parameters_firm["kappa"]
        
        #SEGMENTS
        self.segment_number = int(parameters_firm["segment_number"])
        self.expected_segment_share = [1/self.segment_number]*self.segment_number#initally uniformly distributed
        self.segement_preference_bounds = np.linspace(0, 1, self.segment_number+1) 
        self.width_segment = self.segement_preference_bounds[1] - self.segement_preference_bounds[0]
        self.segement_preference = np.arange(self.width_segment/2, 1, self.width_segment)   #      np.linspace(0, 1, self.segment_number+1) #the plus 1 is so that theere are that number of divisions in the space

        #RANKS
        self.rank_number = int(parameters_firm["rank_number"])
        self.rank_bounds = np.linspace(0, self.markup*self.num_individuals, self.rank_number) 
        self.max_profitability = self.markup*self.num_individuals#What if everyone bought your car then this is how much you would make

        
        self.nk_model = nk_model

        self.N = parameters_firm["N"]
        self.K = parameters_firm["K"]
        self.bit_positions = np.arange(self.N)

        self.init_tech = init_tech 
        self.list_technology_memory = [self.init_tech]
        self.list_technology_memory_strings = [init_tech.component_string]
        self.cars_on_sale = [init_tech]
        self.alternatives_attributes_matrix = np.asarray([car.attributes_fitness for car in self.list_technology_memory])
        self.decimal_values_memory = np.array([x.decimal_value for x in self.list_technology_memory])

###########################################################################################################################################################
    #REASERACH TECHNOLOGY

    def calc_neighbouring_technologies_tumor(self):
        """SKIP INVERSTING THE BITS INDIVIDUALLY DO IT ALL IN ONE GO?"""
        # Convert the memory list to an array of decimal values
        

        # Create a matrix where each row corresponds to one technology, and each column is the bit position to flip
        bit_positions = np.arange(self.N)
        
        """WORK IN DECIMALS"""
        # Generate all possible inverted values by flipping each bit position
        inverted_values_matrix = self.decimal_values_memory[:, np.newaxis] ^ (1 << bit_positions)

        # Flatten the matrix and convert to binary strings
        flattened_inverted_values = inverted_values_matrix.flatten()

        unique_neighbouring_technologies = np.unique(flattened_inverted_values)
        
        # Convert the decimal values to binary strings
        #DO I EVEN NEED TO CONVERT TO BINARY REALLY?
        unique_neighbouring_technologies_strings = [format(value, f'0{self.N}b') for value in unique_neighbouring_technologies]
        
        # Filter out technologies already in memory
        self.list_neighouring_technologies_strings = [tech for tech in unique_neighbouring_technologies_strings if tech not in self.list_technology_memory_strings]

    ##############################################################################

    def utility_buy_matrix(self, car_attributes_matrix):
        #IDEA IS TO DO THIS ALL IN ONE GO, ALL SEGMENTS AND ALL CARTIONS
        segment_preference_reshaped = self.segement_preference[:, np.newaxis]
        utilities = segment_preference_reshaped*car_attributes_matrix[:, 1] + (1 - segment_preference_reshaped) * (self.gamma * car_attributes_matrix[:,2] - (1 - self.gamma) *( (1 + self.markup) * car_attributes_matrix[:,0] + self.carbon_price*car_attributes_matrix[:,1]))
        return utilities.T

    def calculate_profitability_alternatives(self, utilities_competitors):

        #Take strings and workout their profitability
        alternatives_attributes_matrix = np.asarray([self.nk_model.calculate_fitness(x) for x in self.list_neighouring_technologies_strings])
        
        #For each segment need ot caluclate the probability of purchasing the car
        #print("calculate_profitability_alternatives")
        utilities_neighbour = self.utility_buy_matrix(alternatives_attributes_matrix) 

        market_options_utilities = np.concatenate((utilities_neighbour , utilities_competitors ),axis = 0)#join as the probabilities are relative to all other market options

        utilities_neighbour[utilities_neighbour < 0] = 0#IF NEGATIVE UTILITY PUT IT AT 0
        market_options_utilities[market_options_utilities < 0] = 0#IF NEGATIVE UTILITY PUT IT AT 0

        #print("market_options_utilities", market_options_utilities)
        denominators = np.sum(market_options_utilities ** self.kappa, axis = 0)
        if 0 not in denominators:
            alternatives_probability_buy_car = utilities_neighbour ** self.kappa / denominators#CHECK FOR HAVING CURRENT TECH IN HERE NOT TO DOUBLE COUNT
        else:
            alternatives_probability_buy_car = np.zeros_like(utilities_neighbour, dtype=float)#i only change the ones that are not zero
            non_zero_mask = denominators != 0
            alternatives_probability_buy_car[:,non_zero_mask] = (utilities_neighbour[:,non_zero_mask] ** self.kappa) / denominators[non_zero_mask]

        expected_number_customer = self.segment_consumer_count*alternatives_probability_buy_car
        self.expected_profit_research_alternatives = self.markup*alternatives_attributes_matrix[:,0]*np.sum(expected_number_customer, axis= 1)

    def last_tech_profitability(self, utilities_competitors):
        #CALCUALTE THE PREDICTED PROFITABILITY OF TECHNOLOGY RESEARCHED IN PAST STEP
        self.last_tech_researched = self.list_technology_memory[-1]
        last_tech_fitness_arr = np.asarray([self.last_tech_researched.attributes_fitness])

        #print("last tech")
        utility_last_tech = self.utility_buy_matrix(last_tech_fitness_arr)

        last_tech_market_options_utilities = np.concatenate((utility_last_tech , utilities_competitors ), axis = 0)#join as the probabilities are relative to all other market options
        last_tech_market_options_utilities[last_tech_market_options_utilities < 0] = 0

        denominators = np.sum(last_tech_market_options_utilities ** self.kappa, axis = 0)

        if 0 not in denominators:
            alternatives_probability_buy_car = utility_last_tech ** self.kappa /denominators#CHECK FOR HAVING CURRENT TECH IN HERE NOT TO DOUBLE COUNT
            expected_number_customer = self.segment_consumer_count*alternatives_probability_buy_car
            self.last_tech_expected_profit = self.markup*last_tech_fitness_arr[0]*np.sum(expected_number_customer, axis= 1)
        else:
            self.last_tech_expected_profit = 0

    ###################################################################################

    def rank_options(self):
        #RANK THE TECH
        self.ranked_alternatives = []
        for tech, profitability in zip(self.list_neighouring_technologies_strings, self.expected_profit_research_alternatives):
            rank = 0
            for r in range(0, self.rank_number + 1):
                if profitability < (self.max_profitability*r / self.rank_number):
                    rank = r
                    break
            self.ranked_alternatives.append((tech, rank))

    
    def rank_last_tech(self):            
        for r in range(1, self.rank_number + 1):
            if self.last_tech_expected_profit < self.max_profitability*(r / self.rank_number):
                self.last_tech_rank = r
                break

    def add_new_tech_memory(self,chosen_technology):
        self.list_technology_memory.append(chosen_technology)
        self.list_technology_memory_strings.append(chosen_technology.component_string)
        self.alternatives_attributes_matrix = np.concatenate((self.alternatives_attributes_matrix, np.asarray([chosen_technology.attributes_fitness])), axis = 0)

        self.decimal_values_memory = np.concatenate((self.decimal_values_memory, np.asarray([chosen_technology.decimal_value])), axis = 0)

    def select_alternative_technology(self):
        #SELECT TECHNOLOGIES FROM ANY RANK THAT ITS ABOVE CURRENT
        #CREATE A LIST OF POSSIBLE TECHNOLOGIES
        tech_alternative_options = [tech for tech, rank in self.ranked_alternatives if rank >= self.last_tech_rank]
       
        if tech_alternative_options:
            selected_technology_string = random.choice(tech_alternative_options)#this is not empty
            unique_tech_id = self.id_generator.get_new_id()
            attribute_selected_tech  = self.nk_model.calculate_fitness(selected_technology_string) 
            self.researched_technology = Car(unique_tech_id,self.firm_id, selected_technology_string, attribute_selected_tech, choosen_tech_bool = 0) 
            self.add_new_tech_memory(self.researched_technology)
        
        #UPDATE THE MEMEORY
        self.update_memory()

    def research_technology(self, utilities_competitors):
        self.calc_neighbouring_technologies_tumor()#Now i know what the possible neighbouring strings are
        self.calculate_profitability_alternatives(utilities_competitors)
        self.last_tech_profitability(utilities_competitors)
        self.rank_options()
        self.rank_last_tech()
        self.select_alternative_technology()

    ##############################################################################################################
    #CHOOSING TECH FROM MEMORY

    def calculate_profitability_memory_segments(self,utilities_competitors):
        """For each segement work out teh expectedprofitability, but then dont sum so its acutally for each car and secto"""
        
        #For each segment need ot caluclate the probability of purchasing the car
        utilities_memory = self.utility_buy_matrix(self.alternatives_attributes_matrix) 
        market_options_utilities = np.concatenate((utilities_memory , utilities_competitors), axis = 0)#join as the probabilities are relative to all other market options
        
        utilities_memory[utilities_memory < 0] = 0#IF NEGATIVE UTILITY PUT IT AT 0
        market_options_utilities[market_options_utilities < 0] = 0#IF NEGATIVE UTILITY PUT IT AT 0

        denominators = np.sum(market_options_utilities ** self.kappa, axis = 0)

        if 0 not in denominators:
            alternatives_probability_buy_car = utilities_memory ** self.kappa / denominators#CHECK FOR HAVING CURRENT TECH IN HERE NOT TO DOUBLE COUNT
        else:
            #FOR SEGMENTS WHERE IS IT not zero do the calc else where is 0
            alternatives_probability_buy_car = np.zeros_like(utilities_memory, dtype=float)#i only change the ones that are not zero
            non_zero_mask = denominators != 0
            alternatives_probability_buy_car[:,non_zero_mask] = (utilities_memory[:,non_zero_mask] ** self.kappa) / denominators[non_zero_mask]
        
        """these following steps arent neccessary, just linear transformation of all the same"""
        expected_number_customer = self.segment_consumer_count*alternatives_probability_buy_car
        expected_profit_memory_segements = self.markup*expected_number_customer
        return expected_profit_memory_segements

    def update_memory(self):
        for car in self.list_technology_memory:
            if car not in self.cars_on_sale:
                car.choosen_tech_bool = 0
            else:
                car.choosen_tech_bool = 1

        list(map(lambda technology: technology.update_timer(), self.list_technology_memory))#update_timer on all tech
        #remove additional technologies
        if len(self.list_technology_memory) > self.memory_cap:
            max_item = max((item for item in self.list_technology_memory if not item.choosen_tech_bool), key=lambda x: x.timer, default=None)
            index_to_remove = self.list_technology_memory.index(max_item)
            self.list_technology_memory.remove(max_item)#LIST
            del self.list_technology_memory_strings[index_to_remove]#LIST
            self.alternatives_attributes_matrix = np.delete(self.alternatives_attributes_matrix, index_to_remove, axis = 0)#NUMPY ARRAYS
            self.decimal_values_memory = np.delete(self.decimal_values_memory, index_to_remove, axis = 0)#NUMPY ARRAYS

    def get_random_max_tech(self, col):
        max_value = np.max(col)
        max_indices = np.where(col == max_value)[0]
        random_index = np.random.choice(max_indices)
        return self.list_technology_memory[random_index]

    def choose_technologies(self, utilities_competitors):        
        #evaluate for which car is best for which techology
        expected_profit_memory = self.calculate_profitability_memory_segments(utilities_competitors)
        max_profit_technologies = [self.get_random_max_tech(col) for col in expected_profit_memory.T]   
        self.cars_on_sale = list(set(max_profit_technologies))#make it a unique list

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

    def next_step(self, carbon_price, segment_consumer_count, utilities_competitors) -> None:
        
        self.t_firm +=1
        
        self.carbon_price = carbon_price
        self.segment_consumer_count = segment_consumer_count
        
        #RESEARCH TECH
        if not self.static_tech_state:
            self.research_technology(utilities_competitors)

        self.choose_technologies(utilities_competitors)

        if self.save_timeseries_data_state and self.t_firm == 1:
            self.set_up_time_series_firm()
        elif self.save_timeseries_data_state and (self.t_firm % self.compression_factor_state == 0):
            self.save_timeseries_data_firm()

        #print("firm", self.cars_on_sale )
        return self.cars_on_sale