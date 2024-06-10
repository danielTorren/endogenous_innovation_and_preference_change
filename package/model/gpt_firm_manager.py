# imports
import numpy as np
import random
from package.model.cars import Car
from package.model.nk_model import NKModel

class Firm_Manager:
    def __init__(self, parameters_firm_manager: dict, parameters_firm):
        self.t_firm_manager = 0
        self.parameters_firm = parameters_firm

        self.landscape_seed = parameters_firm_manager["landscape_seed"]
        self.init_tech_seed = parameters_firm_manager["init_tech_seed"]
        self.J = parameters_firm_manager["J"]
        self.N = parameters_firm_manager["N"]
        self.K = parameters_firm_manager["K"]
        self.A = parameters_firm_manager["A"]  # NUMBER OF ATTRIBUTES
        self.rho = parameters_firm_manager["rho"]  # Correlation coefficients NEEDS TO BE AT LEAST 2 for 3 things from the same landscape
        self.save_timeseries_data_state = parameters_firm_manager["save_timeseries_data_state"]
        self.compression_factor_state = parameters_firm_manager["compression_factor_state"]
        self.init_tech_heterogenous_state = parameters_firm_manager["init_tech_heterogenous_state"]
        self.carbon_price = parameters_firm_manager["carbon_price"]
        self.num_individuals = parameters_firm_manager["num_individuals"]
        self.id_generator = parameters_firm_manager["IDGenerator_firms"]  # USE TO GENERATE UNIQUE IDS for EVERY SINGLE TECHNOLOGY
        self.gamma = parameters_firm_manager["gamma"]
        self.markup = parameters_firm_manager["markup"]
        self.rank_number = parameters_firm_manager['rank_number']
        self.memory_cap = parameters_firm_manager["memory_cap"]
        self.static_tech_state = parameters_firm_manager["static_tech_state"]

        self.segment_number = int(parameters_firm["segment_number"])
        self.expected_segment_share = [1 / self.segment_number] * self.segment_number  # initially uniformly distributed
        self.segment_preference_bounds = np.linspace(0, 1, self.segment_number + 1)
        self.width_segment = self.segment_preference_bounds[1] - self.segment_preference_bounds[0]
        self.segment_preference = np.arange(self.width_segment / 2, 1, self.width_segment)
        self.segment_preference_reshaped = self.segment_preference[:, np.newaxis]

        self.max_profitability = self.markup*self.num_individuals#What if everyone bought your car then this is how much you would make

        np.random.seed(self.init_tech_seed)  # set seed for numpy
        random.seed(self.init_tech_seed)  # set seed for random

        self.init_tech_component_string = f"{random.getrandbits(self.N):0{self.N}b}"
        if self.init_tech_heterogenous_state:
            decimal_value = int(self.init_tech_component_string, 2)
            init_tech_component_string_list_N = self.invert_bits_one_at_a_time(decimal_value, len(self.init_tech_component_string))
            init_tech_component_string_list = np.random.choice(init_tech_component_string_list_N, self.J)

        self.nk_model = NKModel(self.N, self.K, self.A, self.rho, self.landscape_seed)

        if self.init_tech_heterogenous_state:
            attributes_fitness_list = [self.nk_model.calculate_fitness(x) for x in init_tech_component_string_list]
            self.init_tech_list = [Car(self.id_generator.get_new_id(), j, init_tech_component_string_list[j], attributes_fitness_list[j], choosen_tech_bool=1) for j in range(self.J)]
        else:
            attributes_fitness = self.nk_model.calculate_fitness(self.init_tech_component_string)
            self.init_tech_list = [Car(self.id_generator.get_new_id(), j, self.init_tech_component_string, attributes_fitness, choosen_tech_bool=1) for j in range(self.J)]

        self.firm_data = self.initialize_firms()

        self.cars_on_sale_all_firms = np.asarray([x['cars_on_sale'] for x in self.firm_data]).flatten()

        if self.save_timeseries_data_state:
            self.set_up_time_series_social_network()

    def initialize_firms(self):
        firm_data = []
        for j in range(self.J):
            firm_info = {
                't_firm': 0,
                'firm_id': j,
                'id_generator': self.parameters_firm["id_generator"],
                'save_timeseries_data_state': self.parameters_firm["save_timeseries_data_state"],
                'compression_factor_state': self.parameters_firm["compression_factor_state"],
                'static_tech_state': self.parameters_firm["static_tech_state"],
                'markup': self.parameters_firm["markup"],
                'J': self.parameters_firm["J"],
                'carbon_price': self.parameters_firm["carbon_price"],
                'memory_cap': self.parameters_firm["memory_cap"],
                'num_individuals': self.parameters_firm["num_individuals"],
                'gamma': self.parameters_firm["gamma"],
                'kappa': self.parameters_firm["kappa"],
                'N': self.parameters_firm["N"],
                'K': self.parameters_firm["K"],
                'init_tech': self.init_tech_list[j],
                'list_technology_memory': [self.init_tech_list[j]],
                'list_technology_memory_strings': [self.init_tech_list[j].component_string],
                'cars_on_sale': [self.init_tech_list[j]],
                'alternatives_attributes_matrix': np.asarray([self.init_tech_list[j].attributes_fitness]),
                'decimal_values_memory': np.array([self.init_tech_list[j].decimal_value])
            }
            firm_data.append(firm_info)
        return firm_data

    def invert_bits_one_at_a_time(self, decimal_value, length):
        inverted_binary_values = []
        for bit_position in range(length):
            inverted_value = decimal_value ^ (1 << bit_position)
            inverted_binary_value = format(inverted_value, f'0{length}b')
            inverted_binary_values.append(inverted_binary_value)
        return inverted_binary_values

    def utility_buy_matrix(self, car_attributes_matrix):
        utilities = self.segment_preference_reshaped * car_attributes_matrix[:, 1] + (1 - self.segment_preference_reshaped) * (self.gamma * car_attributes_matrix[:, 2] - (1 - self.gamma) * ((1 + self.markup) * car_attributes_matrix[:, 0] + self.carbon_price * car_attributes_matrix[:, 1]))
        return utilities.T

    def calc_neighbouring_technologies_tumor(self, firm_info):
        bit_positions = np.arange(firm_info['N'])
        inverted_values_matrix = firm_info['decimal_values_memory'][:, np.newaxis] ^ (1 << bit_positions)
        flattened_inverted_values = inverted_values_matrix.flatten()
        unique_neighbouring_technologies = np.unique(flattened_inverted_values)
        #print("unique_neighbouring_technologies", unique_neighbouring_technologies)
        #quit()
        unique_neighbouring_technologies_strings = [format(value, f'0{self.N}b') for value in unique_neighbouring_technologies]
        firm_info['list_neighouring_technologies_strings'] = [tech for tech in unique_neighbouring_technologies_strings if tech not in firm_info['list_technology_memory_strings']]

    def calculate_profitability_alternatives(self, firm_info, utilities_competitors):
        alternatives_attributes_matrix = np.asarray([self.nk_model.calculate_fitness(x) for x in firm_info['list_neighouring_technologies_strings']])
        utilities_neighbour = self.utility_buy_matrix(alternatives_attributes_matrix)
        market_options_utilities = np.concatenate((utilities_neighbour, utilities_competitors), axis=0)
        utilities_neighbour[utilities_neighbour < 0] = 0
        market_options_utilities[market_options_utilities < 0] = 0
        denominators = np.sum(market_options_utilities ** firm_info['kappa'], axis=0)
        if 0 not in denominators:
            alternatives_probability_buy_car = utilities_neighbour ** firm_info['kappa'] / denominators
        else:
            alternatives_probability_buy_car = np.zeros_like(utilities_neighbour, dtype=float)
            non_zero_mask = denominators != 0
            alternatives_probability_buy_car[:, non_zero_mask] = (utilities_neighbour[:, non_zero_mask] ** firm_info['kappa']) / denominators[non_zero_mask]
        
        expected_number_customer = self.segment_consumer_count * alternatives_probability_buy_car
        firm_info['expected_profit_research_alternatives'] = firm_info['markup'] * alternatives_attributes_matrix[:, 0] * np.sum(expected_number_customer, axis=1)

    def last_tech_profitability(self, firm_info, utilities_competitors):
        last_tech_fitness_arr = np.asarray([firm_info['list_technology_memory'][-1].attributes_fitness])
        utility_last_tech = self.utility_buy_matrix(last_tech_fitness_arr)
        last_tech_market_options_utilities = np.concatenate((utility_last_tech, utilities_competitors), axis=0)
        last_tech_market_options_utilities[last_tech_market_options_utilities < 0] = 0
        denominators = np.sum(last_tech_market_options_utilities ** firm_info['kappa'], axis=0)
        if 0 not in denominators:
            alternatives_probability_buy_car = utility_last_tech ** firm_info['kappa'] / denominators
            expected_number_customer = self.segment_consumer_count * alternatives_probability_buy_car
            firm_info['last_tech_expected_profit'] = firm_info['markup'] * last_tech_fitness_arr[0] * np.sum(expected_number_customer, axis=1)
        else:
            firm_info['last_tech_expected_profit'] = 0

    def rank_options(self, firm_info):
        firm_info['ranked_alternatives'] = []
        for tech, profitability in zip(firm_info['list_neighouring_technologies_strings'], firm_info['expected_profit_research_alternatives']):
            rank = 0
            for r in range(0, self.rank_number + 1):
                if profitability < (self.max_profitability * r / self.rank_number):
                    rank = r
                    break
            firm_info['ranked_alternatives'].append((tech, rank))

    def rank_last_tech(self, firm_info):
        for r in range(1, self.rank_number + 1):
            if firm_info['last_tech_expected_profit'] < self.max_profitability * (r / self.rank_number):
                firm_info['last_tech_rank'] = r
                break

    def add_new_tech_memory(self, firm_info, chosen_technology):
        firm_info['list_technology_memory'].append(chosen_technology)
        firm_info['list_technology_memory_strings'].append(chosen_technology.component_string)
        firm_info['alternatives_attributes_matrix'] = np.concatenate((firm_info['alternatives_attributes_matrix'], np.asarray([chosen_technology.attributes_fitness])), axis=0)
        firm_info['decimal_values_memory'] = np.concatenate((firm_info['decimal_values_memory'], np.asarray([chosen_technology.decimal_value])), axis=0)

    def select_alternative_technology(self, firm_info):
        tech_alternative_options = [tech for tech, rank in firm_info['ranked_alternatives'] if rank >= firm_info['last_tech_rank']]
        if tech_alternative_options:
            selected_technology_string = random.choice(tech_alternative_options)
            unique_tech_id = firm_info['id_generator'].get_new_id()
            attribute_selected_tech = self.nk_model.calculate_fitness(selected_technology_string)
            researched_technology = Car(unique_tech_id, firm_info['firm_id'], selected_technology_string, attribute_selected_tech, choosen_tech_bool=0)
            self.add_new_tech_memory(firm_info, researched_technology)
        self.update_memory(firm_info)

    def research_technology(self, firm_info, utilities_competitors):
        self.calc_neighbouring_technologies_tumor(firm_info)
        self.calculate_profitability_alternatives(firm_info, utilities_competitors)
        self.last_tech_profitability(firm_info, utilities_competitors)
        self.rank_options(firm_info)
        self.rank_last_tech(firm_info)
        self.select_alternative_technology(firm_info)

    def calculate_profitability_memory_segments(self, firm_info, utilities_competitors):
        utilities_memory = self.utility_buy_matrix(firm_info['alternatives_attributes_matrix'])
        market_options_utilities = np.concatenate((utilities_memory, utilities_competitors), axis=0)
        utilities_memory[utilities_memory < 0] = 0
        market_options_utilities[market_options_utilities < 0] = 0
        denominators = np.sum(market_options_utilities ** firm_info['kappa'], axis=0)
        if 0 not in denominators:
            alternatives_probability_buy_car = utilities_memory ** firm_info['kappa'] / denominators
        else:
            alternatives_probability_buy_car = np.zeros_like(utilities_memory, dtype=float)
            non_zero_mask = denominators != 0
            alternatives_probability_buy_car[:, non_zero_mask] = (utilities_memory[:, non_zero_mask] ** firm_info['kappa']) / denominators[non_zero_mask]
        expected_number_customer = self.segment_consumer_count * alternatives_probability_buy_car
        expected_profit_memory_segments = firm_info['markup'] * expected_number_customer
        return expected_profit_memory_segments

    def update_memory(self, firm_info):
        for car in firm_info['list_technology_memory']:
            car.choosen_tech_bool = 1 if car in firm_info['cars_on_sale'] else 0
        for technology in firm_info['list_technology_memory']:
            technology.update_timer()
        if len(firm_info['list_technology_memory']) > firm_info['memory_cap']:
            max_item = max((item for item in firm_info['list_technology_memory'] if not item.choosen_tech_bool), key=lambda x: x.timer, default=None)
            index_to_remove = firm_info['list_technology_memory'].index(max_item)
            firm_info['list_technology_memory'].remove(max_item)
            del firm_info['list_technology_memory_strings'][index_to_remove]
            firm_info['alternatives_attributes_matrix'] = np.delete(firm_info['alternatives_attributes_matrix'], index_to_remove, axis=0)
            firm_info['decimal_values_memory'] = np.delete(firm_info['decimal_values_memory'], index_to_remove, axis=0)

    def get_random_max_tech(self, firm_info, col):
        max_value = np.max(col)
        max_indices = np.where(col == max_value)[0]
        random_index = np.random.choice(max_indices)
        return firm_info['list_technology_memory'][random_index]

    def choose_technologies(self, firm_info, utilities_competitors):
        expected_profit_memory = self.calculate_profitability_memory_segments(firm_info, utilities_competitors)
        max_profit_technologies = [self.get_random_max_tech(firm_info, col) for col in expected_profit_memory.T]
        firm_info['cars_on_sale'] = list(set(max_profit_technologies))

    def set_up_time_series_firm(self, firm_info):
        firm_info['history_length_memory_list'] = [len(firm_info['list_technology_memory'])]

    def save_timeseries_data_firm(self, firm_info):
        firm_info['history_length_memory_list'].append(len(firm_info['list_technology_memory']))

    def next_step(self, carbon_price,low_carbon_preference_arr):

        self.segment_consumer_count, __ = np.histogram(low_carbon_preference_arr, bins = self.segment_preference_bounds)
        car_attributes_matrix = np.asarray([x.attributes_fitness for x in self.cars_on_sale_all_firms])
        utilities_competitors =  self.utility_buy_matrix(car_attributes_matrix)
        
        for firm_info in self.firm_data:
            firm_info['t_firm'] += 1
            firm_info['carbon_price'] = carbon_price

            if not self.static_tech_state:
                self.research_technology(firm_info, utilities_competitors)
            self.choose_technologies(firm_info, utilities_competitors)
            if firm_info['save_timeseries_data_state'] and firm_info['t_firm'] == 1:
                self.set_up_time_series_firm(firm_info)
            elif firm_info['save_timeseries_data_state'] and (firm_info['t_firm'] % firm_info['compression_factor_state'] == 0):
                self.save_timeseries_data_firm(firm_info)


        print(self.firm_data)
        quit()
        test = np.asarray([x['cars_on_sale'] for x in self.firm_data])
        print(test)
        quit()
        self.cars_on_sale_all_firms = np.asarray([x['cars_on_sale'] for x in self.firm_data]).flatten()
        return self.cars_on_sale_all_firms

    # Add methods for social network time series if needed
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