"""Define firms that contain info about technology and percieved consumer preferences


Created: 21/12/2023
"""

# imports
import numpy as np
import random

from package.model.technology import Technology

class Firm:
    def __init__(self, parameters_firm, firm_id):
        
        self.firm_id = firm_id#this is used when indexing firms stuff

        self.research_cost = parameters_firm["research_cost"]
        self.expected_carbon_premium =  parameters_firm["expected_carbon_premium"]#variable
        self.markup_adjustment = parameters_firm["markup_adjustment"]
        self.firm_phi = parameters_firm["firm_phi"]
        self.value_matrix_cost = parameters_firm["value_matrix_cost"]
        self.value_matrix_emissions_intensity = parameters_firm["value_matrix_emissions_intensity"]
        self.save_timeseries_data_state = parameters_firm["save_timeseries_data_state"]
        self.compression_factor_state = parameters_firm["compression_factor_state"]

        self.markup = parameters_firm["markup_init"]#variable
        self.current_technology = parameters_firm["technology_init"]#variable
        self.firm_budget = parameters_firm["firm_budget"]#variablees
        self.N = parameters_firm["N"]
        self.K = parameters_firm["K"]
        self.current_market_share = parameters_firm["init_market_share"]

        self.previous_market_share = self.current_market_share#DEFINE THIS

        self.list_technology_memory = [self.current_technology]
        self.list_technology_memory_strings = [self.current_technology.component_string]

        #set up inital stuff relating to emissions prices, costs, and intensities
        self.firm_cost = self.current_technology.cost
        self.firm_emissions_intensities = self.current_technology.emission_intensity
        #self.set_price()
        self.firm_price = self.firm_cost*(1+self.markup)

        if self.save_timeseries_data_state:
            self.set_up_time_series_firm()
    
    ##############################################################################################################
    #DO PREVIOUS TIME STEP STUFF
    def calculate_profits(self,consumed_quantities_vec):
        self.profit = consumed_quantities_vec[self.firm_id]*(self.firm_price - self.firm_cost)

    def update_budget(self):
        budget = self.firm_budget + self.profit + ((1+ self.research_cost)**self.search_range)-1#this is past time step search range?[CHECK THIS]
        self.firm_budget = budget

    def update_carbon_premium(self, market_share_vec, emissions_intensities_vec, price_vec):
        #calculate this sub list of firms where higher market share and that use a technology that is not prefered given THIS firms preference

        indices_higher = [i for i,v in enumerate(market_share_vec) if v > self.current_market_share]

        market_shares_higher = np.asarray([v for i,v in enumerate(market_share_vec) if v > self.current_market_share])#DO THIS BETTER
        price_higher = np.asarray([price_vec[i] for i in indices_higher])
        emissions_intensities_higher = [emissions_intensities_vec[i] for i in indices_higher]

        #calculate_expected_carbon_premium of competitors
        expected_carbon_premium_competitors = (self.firm_price - price_higher)/(emissions_intensities_higher - self.firm_emissions_intensities)
        
        #calculate_weighting_vector
        weighting_vector_firms = (market_shares_higher-self.current_market_share)/sum(market_shares_higher-self.current_market_share)

        #calc_new_expectation
        new_premium = (1-self.firm_phi)*self.expected_carbon_premium + self.firm_phi*np.matmul(weighting_vector_firms,expected_carbon_premium_competitors)

        self.expected_carbon_premium = new_premium
    
    def process_previous_info(self,market_share_vec, consumed_quantities_vec, emissions_intensities_vec, price_vec):
        self.calculate_profits(consumed_quantities_vec)
        self.update_budget()
        self.update_carbon_premium(market_share_vec, emissions_intensities_vec, price_vec)

    ##############################################################################################################
    #SCIENCE!
    def calculate_technology_fitness(self, emissions_intensity, cost):
        f = 1/(self.expected_carbon_premium*emissions_intensity + cost)
        return f

    def update_search_range(self):
        # Update search range based on Equation (\ref{eq_searchrange})

        neighbour_bool = all(i in self.list_technology_memory_strings for i in self.list_neighouring_technologies_strings) #check if all neighbouring technologies are in the memory list

        if (self.firm_budget < self.research_cost) or (neighbour_bool):
            self.search_range = 0
        else:
            self.search_range = 1

    def invert_bits_one_at_a_time(decimal_value, length):
        # Convert decimal value to binary with leading zeros to achieve length N
       # binary_value = format(decimal_value, f'0{length}b')

        # Initialize an empty list to store inverted binary values
        inverted_binary_values = []

        # Iterate through each bit position
        for bit_position in range(length):
            """
            NEED TO UNDERSTAND BETTER HOW THIS WORKS!!
            """
            # Invert the bit at the specified position
            inverted_value = decimal_value ^ (1 << bit_position)

            # Convert the inverted decimal value to binary
            inverted_binary_value = format(inverted_value, f'0{length}b')

            # Append the inverted binary value to the list
            inverted_binary_values.append(inverted_binary_value)

        return inverted_binary_values

    def calc_neighbouring_technologies(self):
        #search with distance 1 from current peak, filter out the ones that arent in memory 
        list_neighouring_technologies_strings = self.invert_bits_one_at_a_time(self.current_technology.component_string, len(self.current_technology.component_string))
        filtered_list_strings = [i for i in list_neighouring_technologies_strings if i not in self.list_technology_memory_strings]

        return filtered_list_strings 

    def calc_tech_emission_cost(self, random_technology_string):

        fitness_vector_cost = np.zeros((self.N))
        fitness_vector_emissions_intensity = np.zeros((self.N))

        for n in range(self.N):#Look through substrings
            # Create the binary substring
            substring = random_technology_string[n:n+self.K]
            # If the substring is shorter than K, wrap around (toroidal)
            if len(substring) < self.K:
                substring += random_technology_string[:self.K-len(substring)]
            # Convert the binary substring to decimal
            decimal = int(substring, 2)
            # Retrieve the value from the value matrix
            fitness_vector_cost[n] = self.value_matrix_cost[decimal, n]
            fitness_vector_emissions_intensity[n] = self.value_matrix_emissions_intensity[decimal, n]

        emissions = np.mean(fitness_vector_emissions_intensity)
        cost = np.mean(fitness_vector_cost)
        return emissions, cost

    def explore_technology(self):
        #grab a random one
        random_technology_string = random.choice(self.list_neighouring_technologies_strings)

        tech_emissions_intensity, tech_cost = self.calc_tech_emission_cost(random_technology_string)

        random_technology = Technology(tech_emissions_intensity, tech_cost, choosen_tech_bool = 0) 

        return random_technology

    def add_new_tech_memory(self,random_technology):
        self.list_technology_memory.append(random_technology)
        self.list_technology_memory_strings.append(random_technology.component_string)

    def choose_technology(self):
        #update_fitness_values in tech
        for technology in self.list_technology_memory:
            technology.fitness = self.calculate_technology_fitness(self, technology.emissions_intensity, technology.cost)
        
        #choose best  tech
        self.current_technology.choosen_tech_bool = 0#in case it changes but the current one to zero

        self.current_technology = max(self.list_technology_memory, key=lambda technology: technology.fitness)

        #set values
        self.firm_cost = self.current_technology.cost
        self.firm_emissions_intensities = self.current_technology.emission_intensity


    def update_memory(self):
        #update_flags
        self.current_technology.choosen_tech_bool = 1
        list(map(lambda technology: technology.update_timer(), self.list_technology_memory))#update_timer on all tech

    def research_technology(self):
        #research new technology to get emissions intensities and cost
        
        self.update_search_range()

        if self.search_range > 0:

            self.list_neighouring_technologies_strings = self.calc_neighbouring_technologies()

            random_technology = self.explore_technology()

            self.add_new_tech_memory(random_technology)

        firm_emissions_intensities, firm_cost = self.choose_technology()#can change technology if preferences change!

        self.update_memory()

        return firm_emissions_intensities, firm_cost
    
    ##############################################################################################################
    #MONEY
    def set_price(self):
        new_markup = self.markup*(1+(self.markup_adjustment*(self.current_market_share - self.previous_market_share)/self.previous_market_share))
        self.firm_price = self.firm_cost*(1+new_markup)
        self.markup = new_markup
        
    ##############################################################################################################
    #FORWARD
        
    def set_up_time_series_firm(self):
        self.history_emissions_intensity = [self.firm_emissions_intensities]
        self.history_price = [self.firm_price]
        self.history_budget = [self.firm_budget]
        self.history_cost = [self.firm_cost]

    def save_timeseries_data_firm(self):
        """
        Save time series data
        ----------
        None

        Returns
        -------
        None
        """
        self.history_emissions_intensity.append(self.firm_emissions_intensities)
        self.history_price.append(self.firm_price)
        self.history_budget.append(self.firm_budget)
        self.history_cost.append(self.firm_cost)

    def next_step(self,  market_share_vec, consumed_quantities_vec, emissions_intensities_vec, price_vec) -> None:
        
        #consumed_quantities_vec: is the vector for each firm how much of their product was consumed
        self.previous_market_share = self.current_market_share
        self.current_market_share = market_share_vec[self.firm_id]
        self.current_consumed_quantity = consumed_quantities_vec[self.firm_id]

        self.process_previous_info(market_share_vec, consumed_quantities_vec, emissions_intensities_vec, price_vec)#assume all are arrays

        self.firm_emissions_intensities, self.firm_cost = self.research_technology()

        self.set_price()

        if self.save_timeseries_data_state:
            self.save_timeseries_data_firm()