import numpy as np

class Firm:
    def __init__(self, firm_id, init_tech_ICE, init_tech_EV):
        self.firm_id = firm_id
        #ICE
        self.init_tech_ICE = init_tech_ICE
        self.list_technology_memory_ICE = [init_tech_ICE]
        self.last_tech_researched_ICE = self.init_tech_ICE#WHEN YOU START THE LAST TECH YOU RESEARCHED IS THE ONE YOU START WITH
        self.list_technology_memory_strings_ICE = [init_tech_ICE.component_string]
        self.memory_attributes_matrix_ICE = np.asarray([init_tech_ICE.attributes_fitness])
        self.decimal_values_memory_ICE = np.array([init_tech_ICE.decimal_value])
        self.unique_neighbouring_technologies_strings_ICE = set(init_tech_ICE.inverted_tech_strings)#GENERATE IT AT THE START
        
        #EV
        self.init_tech_EV = init_tech_EV
        self.list_technology_memory_EV = [init_tech_EV]
        self.last_tech_researched_EV = self.init_tech_EV#WHEN YOU START THE LAST TECH YOU RESEARCHED IS THE ONE YOU START WITH
        self.list_technology_memory_strings_EV = [init_tech_EV.component_string]
        self.memory_attributes_matrix_EV = np.asarray([init_tech_EV.attributes_fitness])
        self.decimal_values_memory_EV = np.array([init_tech_EV.decimal_value])
        self.unique_neighbouring_technologies_strings_EV = set(init_tech_EV.inverted_tech_strings)#GENERATE IT AT THE START

        #ALL TYPES
        self.cars_on_sale = [init_tech_ICE]
        self.expected_profit_research_alternatives = None
        self.last_tech_expected_profit = None
        #self.unique_neighbouring_technologies_strings = set()
        self.ranked_alternatives = []
        self.last_tech_rank = None

        self.tech_alternative_options = []