import numpy as np
import random
from package.model.cars import Car

from package.model.firm import Firm

class Firm_Manager:
    def __init__(self, parameters_firm_manager: dict):
        self.t_firm_manager = 0

        self.landscape_seed = parameters_firm_manager["landscape_seed"]
        self.init_tech_seed = parameters_firm_manager["init_tech_seed"]
        self.J = int(round(parameters_firm_manager["J"]))
        self.N = int(round(parameters_firm_manager["N"]))
        self.K = int(round(parameters_firm_manager["K"]))
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
        self.price_constant = parameters_firm_manager["price_constant"]

        #landscapes
        self.landscape_ICE = parameters_firm_manager["landscape_ICE"]
        self.landscape_EV = parameters_firm_manager["lanscape_EV"]

        #PRICE
        self.segment_number_price_preference = int(parameters_firm_manager["segment_number_price_preference"])
        self.expected_segment_share_price_preference = [1 / self.segment_number_price_preference] * self.segment_number_price_preference
        self.segment_price_preference_bounds = np.linspace(0, 1, self.segment_number_price_preference + 1)
        self.width_price_segment = self.segment_price_preference_bounds[1] - self.segment_price_preference_bounds[0]
        self.segment_price_preference = np.arange(self.width_price_segment / 2, 1, self.width_price_segment)
        self.segment_price_reshaped = self.segment_price_preference[:, np.newaxis]

        #INNOVATIVENESS
        self.segment_number_innovation_preference = int(parameters_firm_manager["segment_number_innovation_preference"])
        self.expected_segment_share_innovation_preference = [1 / self.segment_number_innovation_preference] * self.segment_number_innovation_preference
        self.segment_innovation_preference_bounds = np.linspace(0, 1, self.segment_number_innovation_preference + 1)
        self.width_innovation_segment = self.segment_innovation_preference_bounds[1] - self.segment_innovation_preference_bounds[0]
        self.segment_innovation_preference = np.arange(self.width_innovation_segment / 2, 1, self.width_innovation_segment)
        self.segment_innovation_reshaped = self.segment_innovation_preference[:, np.newaxis]

        # ENVIRONMENTAL PREFERENCE
        self.segment_number_environemental_preference = int(parameters_firm_manager["segment_number_environemental_preference"])
        self.expected_segment_share_environemental_preference = [1 / self.segment_number_environemental_preference] * self.segment_number_environemental_preference
        self.segment_environemental_preference_bounds = np.linspace(0, 1, self.segment_number_environemental_preference + 1)
        self.width_environemental_segment = self.segment_environemental_preference_bounds[1] - self.segment_environemental_preference_bounds[0]
        self.segment_environemental_preference = np.arange(self.width_environemental_segment / 2, 1, self.width_environemental_segment)
        self.segment_environemental_reshaped = self.segment_environemental_preference[:, np.newaxis]

        self.max_profitability = self.markup*self.num_individuals#What if everyone bought your car then this is how much you would make
        #self.max_profitability = (1/self.J)*self.markup*self.num_individuals# what if everyone bought your tech, but also everyone else has the tech too? idunno just need it to be smaller

        np.random.seed(self.init_tech_seed)
        random.seed(self.init_tech_seed)

        self.green_research_bools = np.asarray([None]*self.J)

        self.init_firms()
        
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
        """
        Generate a initial ICE and EV TECH TO START WITH 
        """
        #Pick the initial technology
        self.init_tech_component_string = f"{random.getrandbits(self.N):0{self.N}b}"#CAN USE THE SAME STRING FOR BOTH THE EV AND ICE

        #Generate the initial fitness values of the starting tecnology(ies)
        if self.init_tech_heterogenous_state:
            decimal_value = int(self.init_tech_component_string, 2)
            init_tech_component_string_list_N = self.invert_bits_one_at_a_time(decimal_value, len(self.init_tech_component_string))
            init_tech_component_string_list = np.random.choice(init_tech_component_string_list_N, self.J)
            #handle ICE cars init
            attributes_fitness_list_ICE = [self.landscape_ICE.calculate_fitness(x) for x in init_tech_component_string_list]
            self.init_tech_list_ICE = [Car(self.id_generator.get_new_id(), j, init_tech_component_string_list[j], attributes_fitness_list_ICE[j], choosen_tech_bool=1,N = self.N, nk_landscape=self.landscape_ICE) for j in range(self.J)]
            #handle EV cars init - even if not use still need an initial technology
            attributes_fitness_list_EV = [self.landscape_EV.calculate_fitness(x) for x in init_tech_component_string_list]
            self.init_tech_list_EV = [Car(self.id_generator.get_new_id(), j, init_tech_component_string_list[j], attributes_fitness_list_EV[j], choosen_tech_bool=1,N = self.N, nk_landscape=self.landscape_EV) for j in range(self.J)]
        else:
            #ICE
            attributes_fitness_ICE = self.landscape_ICE.calculate_fitness(self.init_tech_component_string)
            self.init_tech_list_ICE = [Car(self.id_generator.get_new_id(), j, self.init_tech_component_string, attributes_fitness_ICE, choosen_tech_bool=1,N = self.N, nk_landscape=self.landscape_ICE) for j in range(self.J)]
            #EV
            attributes_fitness_EV = self.landscape_EV.calculate_fitness(self.init_tech_component_string)
            self.init_tech_list_EV = [Car(self.id_generator.get_new_id(), j, self.init_tech_component_string, attributes_fitness_EV, choosen_tech_bool=1,N = self.N, nk_landscape=self.landscape_EV) for j in range(self.J)]


        self.discovered_tech_ICE = {}#dictionary which includes an id which is the string and then vlaue is the attributes (PUT NEIGHBOUR IN HERE AS WELL!)
        self.global_neighbouring_technologies_ICE = {}#USED TO STORE THE NEIGHBOURS OF OTHER TECHNOLOGIES

        self.discovered_tech_EV = {}#dictionary which includes an id which is the string and then vlaue is the attributes (PUT NEIGHBOUR IN HERE AS WELL!)
        self.global_neighbouring_technologies_EV = {}#USED TO STORE THE NEIGHBOURS OF OTHER TECHNOLOGIES

        for tech_ICE in self.init_tech_list_ICE:
            self.update_firm_manager_tech_list(tech_ICE, 1)

        for tech_EV in self.init_tech_list_EV:
            self.update_firm_manager_tech_list(tech_EV, 0)

        #Create the firms, these store the data but dont do anything otherwise
        self.firms = [Firm(j, self.init_tech_list_ICE[j], self.init_tech_list_EV[j]) for j in range(self.J)]

        #calculate the inital attributes of all the cars on sale
        self.cars_on_sale_all_firms = self.generate_cars_on_sale_all_firms()
        self.car_attributes_matrix = np.asarray([x.attributes_fitness for x in self.cars_on_sale_all_firms])

    def utility_buy_matrix(self, car_attributes_matrix):
        price = (1 + self.markup) * car_attributes_matrix[:, 0] + self.carbon_price * (1-car_attributes_matrix[:, 1])
        utilities = self.segment_preference_reshaped * car_attributes_matrix[:, 1] + self.gamma * car_attributes_matrix[:, 2] - self.price_constant * price
        #old
        #utilities = self.segment_preference_reshaped * car_attributes_matrix[:, 1] + (1 - self.segment_preference_reshaped) * self.gamma * car_attributes_matrix[:, 2] - (1 - self.gamma) * ((1 + self.markup) * car_attributes_matrix[:, 0] + self.carbon_price * car_attributes_matrix[:, 1])
        return utilities.T + self.utility_boost_const
    
    def utility_buy_vec(self, car_attributes_vec):
        price = (1 + self.markup) * car_attributes_vec[0] + self.carbon_price * (1-car_attributes_vec[1])
        utilities = self.segment_preference_reshaped * car_attributes_vec[1] +  self.gamma * car_attributes_vec[2] - self.price_constant * price
        #old
        #utilities = self.segment_preference_reshaped * car_attributes_vec[1] + (1 - self.segment_preference_reshaped) * self.gamma * car_attributes_vec[2] - (1 - self.gamma) * ((1 + self.markup) * car_attributes_vec[0] + self.carbon_price * car_attributes_vec[1])
        
        return utilities.T + self.utility_boost_const

    def calculate_profitability_neighbouring_technologies(self, firm, utilities_competitors):
        unique_neighbouring_technologies_attributes = np.array([self.discovered_tech[key] for key in firm.unique_neighbouring_technologies_strings])#GRAB IT FROM THE GLOBAL LIST
        utilities_neighbour = self.utility_buy_matrix(unique_neighbouring_technologies_attributes) 
        #print("utilities_neighbour",utilities_neighbour)
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
        
        #print("expected_number_customer", expected_number_customer)
        firm.expected_profit_research_alternatives = self.markup * utilities_neighbour[:,0] * np.sum(expected_number_customer, axis=1)
        #print("firm.expected_profit_research_alternatives",firm.expected_profit_research_alternatives)

    def last_tech_profitability(self, firm, utilities_competitors):
        last_tech_fitness_arr = np.asarray(firm.last_tech_researched.attributes_fitness)
        utility_last_tech = self.utility_buy_vec(last_tech_fitness_arr)
        #print("utility_last_tech",utility_last_tech)
        last_tech_market_options_utilities = np.concatenate((utility_last_tech, utilities_competitors), axis=0)
        last_tech_market_options_utilities[last_tech_market_options_utilities < 0] = 0
        denominators = np.sum(last_tech_market_options_utilities ** self.kappa, axis=0)

        if 0 not in denominators:
            alternatives_probability_buy_car = utility_last_tech ** self.kappa / denominators

            expected_number_customer = (self.segment_consumer_count * alternatives_probability_buy_car)[0]#ITS DOUBLE BRAKETED NOT SURE WHY COME BACK TO THIS ADD 0 FOR NOW

            firm.last_tech_expected_profit = self.markup * last_tech_fitness_arr[0] * np.sum(expected_number_customer, axis=0)

        else:
            firm.last_tech_expected_profit = 0

        #print("firm.last_tech_expected_profit", firm.last_tech_expected_profit)

    def rank_options(self, firm):
        firm.ranked_alternatives = []

        for tech, profitability in zip(firm.unique_neighbouring_technologies_strings, firm.expected_profit_research_alternatives):
            rank = 0
            for r in range(0, self.rank_number + 1):
                #print("ALT tech",tech, r, firm.last_tech_expected_profit , (self.max_profitability * r / self.rank_number))
                #print("(self.max_profitability * r / self.rank_number)",profitability, (self.max_profitability * r / self.rank_number), r)
                if profitability < (self.max_profitability * r / self.rank_number):
                    rank = r
                    break
            if r >= self.rank_number - 1:#MAX RANK
                rank = self.rank_number - 1
            firm.ranked_alternatives.append((tech, rank))

    def rank_last_tech(self, firm):
        rank= 0
        for r in range(0, self.rank_number + 1):
            #print("last tech", r, firm.last_tech_expected_profit , (self.max_profitability * r / self.rank_number))
            if firm.last_tech_expected_profit < (self.max_profitability * r / self.rank_number):
                rank = r
                break
        if r >= self.rank_number - 1:#MAX RANK
            rank = self.rank_number - 1
        firm.last_tech_rank = rank
        #quit()

    ###############################################################################
    #DEADLING WITH GLOBAL MEMEORY AND CREATIGN TUMOURS!
    def update_firm_manager_tech_list(self, chosen_technology, ICE_bool):
        if ICE_bool:
            self.global_neighbouring_technologies_ICE[chosen_technology.component_string] = chosen_technology.inverted_tech_strings
            self.discovered_tech_ICE[chosen_technology.component_string] = chosen_technology.attributes_fitness
            for i, string in enumerate(chosen_technology.inverted_tech_strings):
                self.discovered_tech_ICE[string] = chosen_technology.inverted_tech_fitness[i]
        else:
            self.global_neighbouring_technologies_EV[chosen_technology.component_string] = chosen_technology.inverted_tech_strings
            self.discovered_tech_EV[chosen_technology.component_string] = chosen_technology.attributes_fitness
            for i, string in enumerate(chosen_technology.inverted_tech_strings):
                self.discovered_tech_EV[string] = chosen_technology.inverted_tech_fitness[i]
    
    def generate_neighbouring_technologies(self, firm):
        #I want to
        unique_neighbouring_technologies_strings = set() 

        for string in firm.list_technology_memory_strings: 

            inverted_values = self.global_neighbouring_technologies[string]
            inverted_values_not_in_memory = [value for value in inverted_values if value not in firm.list_technology_memory_strings]
            if inverted_values_not_in_memory:
                unique_neighbouring_technologies_strings.update(inverted_values_not_in_memory)

        return unique_neighbouring_technologies_strings
    #################################################################################
    #MEMORY 
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
    
    #################################################################################################################

    def select_alternative_technology(self, firm):

        firm.tech_alternative_options = [tech for tech, rank in firm.ranked_alternatives if rank >= firm.last_tech_rank]

        #print(firm.ranked_alternatives, firm.last_tech_rank)
        #quit()
        if firm.tech_alternative_options:
            #print(self.t_firm_manager, firm.firm_id, "NEW TECH ADDED!")
            selected_technology_string = random.choice(firm.tech_alternative_options)
            researched_technology = self.gen_new_technology_memory(firm,selected_technology_string)

            
            #print("NEW TECH, tiem, firm id", self.t_firm_manager, firm.firm_id, researched_technology)
            self.last_tech_researched = researched_technology#MAKE THE NEW TECH DISCOVERED THE LAST TECH RESEARCHED
            if researched_technology.environmental_score > 0.7:
                self.green_research_bools[firm.firm_id] = 1
            else:
                self.green_research_bools[firm.firm_id] = 0
            #print("BEFORE", len(self.global_neighbouring_technologies), len(firm.list_technology_memory_strings))
            self.update_firm_manager_tech_list(researched_technology)#THIS HAPPENDS BEFORE THE NEIGHBOURING TECHNOLOGIES ARE CREATED
            #print("MID", len(self.global_neighbouring_technologies), len(firm.list_technology_memory_strings))
            self.add_new_tech_memory(firm, researched_technology)
            #print("AFTER", len(self.global_neighbouring_technologies), len(firm.list_technology_memory_strings))
            #self.list_research_tech.append(researched_technology)

    def research_technology(self, firm, utilities_competitors):
        self.calculate_profitability_neighbouring_technologies(firm, utilities_competitors)
        self.last_tech_profitability(firm, utilities_competitors)
        self.rank_options(firm)
        self.rank_last_tech(firm)
        self.select_alternative_technology(firm)



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
        self.history_green_research_bools = [self.green_research_bools]

    def save_timeseries_data_firm_manager(self):
        self.history_cars_on_sale_all_firms.append(self.cars_on_sale_all_firms)
        #self.history_researched_tech.append(self.list_research_tech)
        self.history_len_n.append(self.len_n)
        self.history_len_alt.append(self.len_alt)
        self.history_green_research_bools.append(self.green_research_bools)

    def generate_cars_on_sale_all_firms(self):
        cars_on_sale_all_firms = []
        for firm in self.firms:
            cars_on_sale_all_firms.extend(firm.cars_on_sale)
        return np.asarray(cars_on_sale_all_firms)

    def update_segement_count(self, ev_adoption_state_arr, environmental_preference_arr, price_preference_arr):
        self.segment_ev_adoption_state_count, __ = np.histogram(ev_adoption_state_arr, bins = self.segment_innovation_preference_bounds)
        #ONLY CALC THE FOLLOWING IF THEY ACUTALLY CHANGE OVER TIME
        self.segment_environmental_consumer_count, __ = np.histogram(environmental_preference_arr, bins = self.segment_environemental_preference_bounds)
        self.segment_price_consumer_count, __ = np.histogram(price_preference_arr, bins = self.segment_price_preference_bounds)

    def next_step(self, carbon_price, ev_adoption_state_arr, environmental_preference_arr, price_preference_arr):
        self.t_firm_manager += 1

        #print("STEP", self.t_firm_manager)
        self.list_research_tech = []
        self.carbon_price = carbon_price

        self.update_segement_count(ev_adoption_state_arr, environmental_preference_arr, price_preference_arr)
        utilities_competitors =  self.utility_buy_matrix(self.car_attributes_matrix)
        
        self.green_research_bools = np.asarray([None]*self.J)

        for firm in self.firms:
            if not self.static_tech_state:
                #print("firm.unique_neighbouring_technologies_strings", firm.unique_neighbouring_technologies_strings)
                if firm.unique_neighbouring_technologies_strings and (self.t_firm_manager % 12 == 0):#ONLY DO RESERACH IF THERE IS ACTUALLY A TECH that can be researeched
                    self.research_technology(firm, utilities_competitors)
                self.update_memory(firm)#EVEN IF NOT TECH IS RESEARCH STILL NEED TO UPDATE AGE OF TECHNOLOGIES    
                firm.unique_neighbouring_technologies_strings = self.generate_neighbouring_technologies(firm)
            self.choose_technologies(firm, utilities_competitors)

        self.cars_on_sale_all_firms = self.generate_cars_on_sale_all_firms()
        self.car_attributes_matrix = np.asarray([x.attributes_fitness for x in self.cars_on_sale_all_firms])
        
        if self.save_timeseries_data_state and (self.t_firm_manager % self.compression_factor_state == 0):
            self.len_n = [len(firm.unique_neighbouring_technologies_strings) for firm in self.firms]
            self.len_alt = [len(firm.tech_alternative_options) for firm in self.firms]
            self.save_timeseries_data_firm_manager()


        return self.cars_on_sale_all_firms