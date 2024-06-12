import numpy as np

class Firm:
    def __init__(self, firm_id, init_tech):
        self.firm_id = firm_id
        self.init_tech = init_tech
        self.list_technology_memory = [init_tech]
        self.last_tech_researched = self.init_tech#WHEN YOU START THE LAST TECH YOU RESEARCHED IS THE ONE YOU START WITH
        self.list_technology_memory_strings = [init_tech.component_string]
        self.cars_on_sale = [init_tech]
        self.memory_attributes_matrix = np.asarray([init_tech.attributes_fitness])
        self.decimal_values_memory = np.array([init_tech.decimal_value])
        self.expected_profit_research_alternatives = None
        self.last_tech_expected_profit = None
        #self.unique_neighbouring_technologies_strings = set()
        self.ranked_alternatives = []
        self.last_tech_rank = None
        self.unique_neighbouring_technologies_strings = set(init_tech.inverted_tech_strings)#GENERATE IT AT THE START
        