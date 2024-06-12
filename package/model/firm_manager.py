import numpy as np
import random
from package.model.cars import Car
from package.model.nk_model import NKModel
from package.model.firm import Firm

class Firm_Manager:
    def __init__(self, parameters_firm_manager: dict):
        self.t_firm_manager = 0

        self.landscape_seed = parameters_firm_manager["landscape_seed"]
        self.init_tech_seed = parameters_firm_manager["init_tech_seed"]
        self.J = parameters_firm_manager["J"]
        self.N = parameters_firm_manager["N"]
        self.K = parameters_firm_manager["K"]
        self.A = parameters_firm_manager["A"]
        self.rho = parameters_firm_manager["rho"]
        self.save_timeseries_data_state = parameters_firm_manager["save_timeseries_data_state"]
        self.compression_factor_state = parameters_firm_manager["compression_factor_state"]
        self.init_tech_heterogenous_state = parameters_firm_manager["init_tech_heterogenous_state"]
        self.carbon_price = parameters_firm_manager["carbon_price"]
        self.num_individuals = parameters_firm_manager["num_individuals"]
        self.id_generator = parameters_firm_manager["IDGenerator_firms"]
        self.gamma = parameters_firm_manager["gamma"]
        self.kappa = parameters_firm_manager["kappa"]
        self.markup = parameters_firm_manager["markup"]
        self.rank_number = parameters_firm_manager['rank_number']
        self.memory_cap = parameters_firm_manager["memory_cap"]
        self.static_tech_state = parameters_firm_manager["static_tech_state"]
        self.utility_boost_const = parameters_firm_manager["utility_boost_const"]

        self.segment_number = int(parameters_firm_manager["segment_number"])
        self.expected_segment_share = [1 / self.segment_number] * self.segment_number
        self.segment_preference_bounds = np.linspace(0, 1, self.segment_number + 1)
        self.width_segment = self.segment_preference_bounds[1] - self.segment_preference_bounds[0]
        self.segment_preference = np.arange(self.width_segment / 2, 1, self.width_segment)
        self.segment_preference_reshaped = self.segment_preference[:, np.newaxis]

        self.max_profitability = self.markup*self.num_individuals#What if everyone bought your car then this is how much you would make

        np.random.seed(self.init_tech_seed)
        random.seed(self.init_tech_seed)

        #Create NK model
        self.nk_model = NKModel(self.N, self.K, self.A, self.rho, self.landscape_seed)

        self.init_firms()

        #self.list_reserach_tech = []#JUST IN CASE
        #Set up the data saving
        if self.save_timeseries_data_state:
            self.set_up_time_series_firm_manager()

    ####################################################################################################
    #ALL FIRM SPECIFIC CODE

    def invert_bits_one_at_a_time(self, decimal_value, length):
        """THIS IS ONLY USED ONCE TO GENERATE HETEROGENOUS INITIAL TECHS"""
        inverted_binary_values = []
        for bit_position in range(length):
            inverted_value = decimal_value ^ (1 << bit_position)
            inverted_binary_value = format(inverted_value, f'0{length}b')
            inverted_binary_values.append(inverted_binary_value)
        return inverted_binary_values
    
    def init_firms(self):
        #Pick the initial technology
        self.init_tech_component_string = f"{random.getrandbits(self.N):0{self.N}b}"
        #Generate the initial fitness values of the starting tecnology(ies)
        if self.init_tech_heterogenous_state:
            decimal_value = int(self.init_tech_component_string, 2)
            init_tech_component_string_list_N = self.invert_bits_one_at_a_time(decimal_value, len(self.init_tech_component_string))
            init_tech_component_string_list = np.random.choice(init_tech_component_string_list_N, self.J)
            attributes_fitness_list = [self.nk_model.calculate_fitness(x) for x in init_tech_component_string_list]
            self.init_tech_list = [Car(self.id_generator.get_new_id(), j, init_tech_component_string_list[j], attributes_fitness_list[j], choosen_tech_bool=1,N = self.N, nk_landscape=self.nk_model) for j in range(self.J)]
        else:
            attributes_fitness = self.nk_model.calculate_fitness(self.init_tech_component_string)

            self.init_tech_list = [Car(self.id_generator.get_new_id(), j, self.init_tech_component_string, attributes_fitness, choosen_tech_bool=1,N = self.N, nk_landscape=self.nk_model) for j in range(self.J)]

        self.discovered_tech = {}#dictionary which includes an id which is the string and then vlaue is the attributes (PUT NEIGHBOUR IN HERE AS WELL!)
        self.global_neighbouring_technologies = {}#USED TO STORE THE NEIGHBOURS OF OTHER TECHNOLOGIES

        for tech in self.init_tech_list:
            self.update_firm_manager_tech_list(tech)

        #@print("ADDED DATA TO GLOBAL")
        #print(self.discovered_tech, self.global_neighbouring_technologies)


        #Create the firms, these store the data but dont do anything otherwise
        self.firms = [Firm(j, self.init_tech_list[j]) for j in range(self.J)]

        #calculate the inital attributes of all the cars on sale
        self.cars_on_sale_all_firms = self.generate_cars_on_sale_all_firms()
        self.car_attributes_matrix = np.asarray([x.attributes_fitness for x in self.cars_on_sale_all_firms])

    def utility_buy_matrix(self, car_attributes_matrix):
        utilities = self.segment_preference_reshaped * car_attributes_matrix[:, 1] + (1 - self.segment_preference_reshaped) * (self.gamma * car_attributes_matrix[:, 2] - (1 - self.gamma) * ((1 + self.markup) * car_attributes_matrix[:, 0] + self.carbon_price * car_attributes_matrix[:, 1]))
        return utilities.T + self.utility_boost_const
    
    def utility_buy_vec(self, car_attributes_vec):
        utilities = self.segment_preference_reshaped * car_attributes_vec[1] + (1 - self.segment_preference_reshaped) * (self.gamma * car_attributes_vec[2] - (1 - self.gamma) * ((1 + self.markup) * car_attributes_vec[0] + self.carbon_price * car_attributes_vec[1]))
        return utilities.T + self.utility_boost_const

    def update_memory(self, firm):
        #set the technology to be used as true or 1
        for car in firm.list_technology_memory:
            car.choosen_tech_bool = 1 if car in firm.cars_on_sale else 0

        #change the timer for the techs that are not the ones being used
        for technology in firm.list_technology_memory:
            technology.update_timer()

        #is the memory list is too long then remove data
        if len(firm.list_technology_memory) > self.memory_cap:
            tech_to_remove = max((tech for tech in firm.list_technology_memory if not tech.choosen_tech_bool), key=lambda x: x.timer, default=None)#PICK TECH WITH MAX TIMER WHICH IS NOT ACTIVE
            index_to_remove = firm.list_technology_memory.index(tech_to_remove)
            
            del firm.list_technology_memory_strings[index_to_remove]
            firm.memory_attributes_matrix = np.delete(firm.memory_attributes_matrix, index_to_remove, axis=0)
            firm.decimal_values_memory = np.delete(firm.decimal_values_memory, index_to_remove, axis=0)
            firm.list_technology_memory.remove(tech_to_remove)#last thing is remove the item

    def calculate_profitability_neighbouring_technologies(self, firm, utilities_competitors):

        unique_neighbouring_technologies_attributes = np.array([self.discovered_tech[key] for key in firm.unique_neighbouring_technologies_strings])#GRAB IT FROM THE GLOBAL LIST

        utilities_neighbour = self.utility_buy_matrix(unique_neighbouring_technologies_attributes) 
        market_options_utilities = np.concatenate((utilities_neighbour, utilities_competitors), axis=0)
        market_options_utilities[market_options_utilities < 0] = 0
        denominators = np.sum(market_options_utilities ** self.kappa, axis=0)
        if 0 not in denominators:
            alternatives_probability_buy_car = utilities_neighbour ** self.kappa / denominators
        else:
            alternatives_probability_buy_car = np.zeros_like(utilities_neighbour, dtype=float)
            non_zero_mask = denominators != 0
            alternatives_probability_buy_car[:, non_zero_mask] = (utilities_neighbour[:, non_zero_mask] ** self.kappa) / denominators[non_zero_mask]
        expected_number_customer = self.segment_consumer_count * alternatives_probability_buy_car
        firm.expected_profit_research_alternatives = self.markup * utilities_neighbour[:,0] * np.sum(expected_number_customer, axis=1)

    def last_tech_profitability(self, firm, utilities_competitors):
        last_tech_fitness_arr = np.asarray(firm.last_tech_researched.attributes_fitness)
        utility_last_tech = self.utility_buy_vec(last_tech_fitness_arr)
        last_tech_market_options_utilities = np.concatenate((utility_last_tech, utilities_competitors), axis=0)
        last_tech_market_options_utilities[last_tech_market_options_utilities < 0] = 0
        denominators = np.sum(last_tech_market_options_utilities ** self.kappa, axis=0)

        if 0 not in denominators:
            alternatives_probability_buy_car = utility_last_tech ** self.kappa / denominators

            expected_number_customer = (self.segment_consumer_count * alternatives_probability_buy_car)[0]#ITS DOUBLE BRAKETED NOT SURE WHY COME BACK TO THIS ADD 0 FOR NOW

            firm.last_tech_expected_profit = self.markup * last_tech_fitness_arr[0] * np.sum(expected_number_customer, axis=0)

        else:
            firm.last_tech_expected_profit = 0

    def rank_options(self, firm):
        firm.ranked_alternatives = []
        for tech, profitability in zip(firm.unique_neighbouring_technologies_strings, firm.expected_profit_research_alternatives):
            rank = 0
            for r in range(0, self.rank_number + 1):
                if profitability < (self.max_profitability * r / self.rank_number):
                    rank = r
                    break
            firm.ranked_alternatives.append((tech, rank))

    def rank_last_tech(self, firm):
        rank= 0
        for r in range(1, self.rank_number + 1):
            if firm.last_tech_expected_profit < self.max_profitability * (r / self.rank_number):
                rank = r
                break
        firm.last_tech_rank = rank

    ###############################################################################
    #DEADLING WITH GLOBAL MEMEORY AND CREATIGN TUMOURS!
    def update_firm_manager_tech_list(self, chosen_technology):
        if chosen_technology.component_string not in self.global_neighbouring_technologies:
            print("ADDED TO GLOBAL", chosen_technology.component_string)
            self.global_neighbouring_technologies[chosen_technology.component_string] = chosen_technology.inverted_tech_strings
        
        if chosen_technology.component_string not in self.discovered_tech:
            #print(self.t_firm_manager,"TECH ADDED", chosen_technology.component_string)
            self.discovered_tech[chosen_technology.component_string] = chosen_technology.attributes_fitness
            for i, string in enumerate(chosen_technology.inverted_tech_strings):
                #print("string", string)
                if string not in self.discovered_tech:#ADDED THE NEIGHBOURS TOO
                    self.discovered_tech[string] = chosen_technology.inverted_tech_fitness[i]
            
    def generate_neighbouring_technologies(self, firm):
        #I want to
        unique_neighbouring_technologies_strings = set() 
        print("GENERRATE NEIGHBOURS!")
        #print(self.discovered_tech.keys())
        #print(self.global_neighbouring_technologies.keys())

        for string in firm.list_technology_memory_strings: 
            print("string", string)
            print(self.discovered_tech[string])
            print(self.global_neighbouring_technologies[string])
            #quit()
            inverted_values = self.global_neighbouring_technologies[string]
            inverted_values_not_in_memory = [value for value in inverted_values if value not in firm.list_technology_memory_strings]
            if inverted_values_not_in_memory:
                unique_neighbouring_technologies_strings.update(inverted_values_not_in_memory)

        return unique_neighbouring_technologies_strings
    #################################################################################

    def add_new_tech_memory(self, firm, chosen_technology):      
        firm.list_technology_memory.append(chosen_technology)
        firm.list_technology_memory_strings.append(chosen_technology.component_string)
        #add attributes to memory
        firm.memory_attributes_matrix = np.concatenate((firm.memory_attributes_matrix, np.asarray([chosen_technology.attributes_fitness])), axis=0)
        firm.decimal_values_memory = np.concatenate((firm.decimal_values_memory, np.asarray([chosen_technology.decimal_value])), axis=0)

    def gen_new_technology_memory(self,firm, selected_technology_string):
        unique_tech_id = self.id_generator.get_new_id()
        attribute_selected_tech = self.nk_model.calculate_fitness(selected_technology_string)
        researched_technology = Car(unique_tech_id, firm.firm_id, selected_technology_string, attribute_selected_tech, choosen_tech_bool=0, N = self.N, nk_landscape= self.nk_model)
        return researched_technology

    def select_alternative_technology(self, firm):
        tech_alternative_options = [tech for tech, rank in firm.ranked_alternatives if rank >= firm.last_tech_rank]
        print("START FIRM TIM STEPS")
        if tech_alternative_options:
            print(self.t_firm_manager, firm.firm_id, "NEW TECH ADDED!")
            selected_technology_string = random.choice(tech_alternative_options)
            researched_technology = self.gen_new_technology_memory(firm,selected_technology_string)
            self.last_tech_researched = researched_technology#MAKE THE NEW TECH DISCOVERED THE LAST TECH RESEARCHED
            
            print("BEFORE", len(self.global_neighbouring_technologies), len(firm.list_technology_memory_strings))
            self.update_firm_manager_tech_list(researched_technology)#THIS HAPPENDS BEFORE THE NEIGHBOURING TECHNOLOGIES ARE CREATED
            print("MID", len(self.global_neighbouring_technologies), len(firm.list_technology_memory_strings))
            self.add_new_tech_memory(firm, researched_technology)
            print("AFTER", len(self.global_neighbouring_technologies), len(firm.list_technology_memory_strings))
            self.list_reserach_tech.append(researched_technology)

    def research_technology(self, firm, utilities_competitors):
        self.calculate_profitability_neighbouring_technologies(firm, utilities_competitors)
        self.last_tech_profitability(firm, utilities_competitors)
        self.rank_options(firm)
        self.rank_last_tech(firm)
        self.select_alternative_technology(firm)
        self.update_memory(firm)
        firm.unique_neighbouring_technologies_strings = self.generate_neighbouring_technologies(firm)

        #######################################################################################################################################
        #######################################################################################################################################
        #CHECK IF NEIGHBOURS ARE IN DISCOVERED TECH
        # Extract the keys from the dictionary
        dict_keys_set = set(self.discovered_tech.keys())
        # Check if the set is a subset of the dictionary's keys set
        print("dict_keys_set", dict_keys_set)
        print("firm.unique_neighbouring_technologies_strings", firm.unique_neighbouring_technologies_strings)
        are_all_items_keys = firm.unique_neighbouring_technologies_strings.issubset(dict_keys_set)
        print("ARE KEYS THERE",self.t_firm_manager, firm.firm_id, are_all_items_keys)  # This will print True if all items in items_set are keys i
        #######################################################################################################################################
        #######################################################################################################################################

    ############################################################################
    #MEMORY STUFF

    def calculate_profitability_memory_segments(self, firm, utilities_competitors):

        utilities_memory = self.utility_buy_matrix(firm.memory_attributes_matrix)
        market_options_utilities = np.concatenate((utilities_memory, utilities_competitors), axis=0)
        utilities_memory[utilities_memory < 0] = 0
        market_options_utilities[market_options_utilities < 0] = 0
        denominators = np.sum(market_options_utilities ** self.kappa, axis=0)
        if 0 not in denominators:
            alternatives_probability_buy_car = utilities_memory ** self.kappa / denominators
        else:
            alternatives_probability_buy_car = np.zeros_like(utilities_memory, dtype=float)
            non_zero_mask = denominators != 0
            alternatives_probability_buy_car[:, non_zero_mask] = (utilities_memory[:, non_zero_mask] ** self.kappa) / denominators[non_zero_mask]
        expected_number_customer = self.segment_consumer_count * alternatives_probability_buy_car
        expected_profit_memory_segments = self.markup * expected_number_customer
        return expected_profit_memory_segments

    def get_random_max_tech(self, firm, col):
        max_value = np.max(col)
        max_indices = np.where(col == max_value)[0]
        random_index = np.random.choice(max_indices)
        return firm.list_technology_memory[random_index]

    def choose_technologies(self, firm, utilities_competitors):
        expected_profit_memory = self.calculate_profitability_memory_segments(firm, utilities_competitors)
        max_profit_technologies = [self.get_random_max_tech(firm, col) for col in expected_profit_memory.T]
        firm.cars_on_sale = list(set(max_profit_technologies))

    ###################################################################################################

    def set_up_time_series_firm_manager(self):
        self.history_cars_on_sale_all_firms = [self.cars_on_sale_all_firms]
        self.history_researched_tech = [self.cars_on_sale_all_firms]

    def save_timeseries_data_firm_manager(self):
        self.history_cars_on_sale_all_firms.append(self.cars_on_sale_all_firms)
        self.history_researched_tech.append(self.list_reserach_tech)

    def generate_cars_on_sale_all_firms(self):
        cars_on_sale_all_firms = []
        for firm in self.firms:

            cars_on_sale_all_firms.extend(firm.cars_on_sale)
        return np.asarray(cars_on_sale_all_firms)
    
    def next_step(self, carbon_price, low_carbon_preference_arr):
        self.t_firm_manager += 1

        self.list_reserach_tech = []
        self.carbon_price = carbon_price
        self.segment_consumer_count, __ = np.histogram(low_carbon_preference_arr, bins = self.segment_preference_bounds)
        
        utilities_competitors =  self.utility_buy_matrix(self.car_attributes_matrix)
        
        for firm in self.firms:
            if not self.static_tech_state:
                self.research_technology(firm, utilities_competitors)
            self.choose_technologies(firm, utilities_competitors)

        self.cars_on_sale_all_firms = self.generate_cars_on_sale_all_firms()
        self.car_attributes_matrix = np.asarray([x.attributes_fitness for x in self.cars_on_sale_all_firms])
        
        if self.save_timeseries_data_state and (self.t_firm_manager % self.compression_factor_state == 0):
            self.save_timeseries_data_firm_manager()


        return self.cars_on_sale_all_firms