"""Define firms that contain info about technology and percieved consumer preferences


Created: 21/12/2023
"""

# imports
import numpy as np
import random

from package.model.technology import Technology

class Firm:
    def __init__(self, parameters_firm, init_tech, expected_carbon_premium, firm_id):
        
        self.t_firm = 0

        self.firm_id = firm_id#this is used when indexing firms stuff

        self.research_cost = parameters_firm["research_cost"]
        self.expected_carbon_premium =  expected_carbon_premium#parameters_firm["expected_carbon_premium"]#variable
        self.markup_adjustment = parameters_firm["markup_adjustment"]
        self.firm_phi = parameters_firm["firm_phi"]
        self.value_matrix_cost = parameters_firm["value_matrix_cost"]
        self.value_matrix_emissions_intensity = parameters_firm["value_matrix_emissions_intensity"]
        self.save_timeseries_data_state = parameters_firm["save_timeseries_data_state"]
        self.compression_factor_state = parameters_firm["compression_factor_state"]
        self.static_tech_state = parameters_firm["static_tech_state"]
        self.markup = parameters_firm["markup_init"]#variable
        self.J = parameters_firm["J"]

        #ALLOWS FOR VARIABEL INIT TECH
        self.current_technology = init_tech#parameters_firm["technology_init"]#variable

        self.current_technology.fitness = self.calculate_technology_fitness(self.current_technology.emissions_intensity, self.current_technology.cost)#assign fitness to inti technology

        self.firm_budget = parameters_firm["firm_budget"]#variablees
        self.N = parameters_firm["N"]
        self.K = parameters_firm["K"]
        self.current_market_share = parameters_firm["init_market_share"]
        self.current_market_share_vec = [self.current_market_share]*self.J

        self.previous_market_share = self.current_market_share#DEFINE THIS

        self.list_technology_memory = [self.current_technology]
        self.list_technology_memory_strings = [self.current_technology.component_string]

        #set up inital stuff relating to emissions prices, costs, and intensities
        self.firm_cost = self.current_technology.cost
        self.firm_emissions_intensity = self.current_technology.emissions_intensity
        #self.set_price()
        self.firm_price = self.firm_cost*(1+self.markup)

        #SET INTITAL SEARCH RANGE TO 1 if you have enough money, BASICALLY WONT DO ANYTHING IF COMPANIES DONT HAVE ENOUGH MONEY INITIALLY
        if self.firm_budget < self.research_cost:
            self.search_range = 0
        else:
            self.search_range = 1

        #CALCULATE NEIGHBOURING TECH BASED ON INITIAL TECHNOLOGY
        self.list_neighouring_technologies_strings = self.calc_neighbouring_technologies()
    
    ##############################################################################################################
    #DO PREVIOUS TIME STEP STUFF
    def calculate_profits(self,consumed_quantities_vec):
        self.profit = consumed_quantities_vec[self.firm_id]*(self.firm_price - self.firm_cost)

    def update_budget(self):
        budget = self.firm_budget + self.profit + ((1+ self.research_cost)**self.search_range)-1#this is past time step search range?[CHECK THIS]
        self.firm_budget = budget

    def update_carbon_premium(self, emissions_intensities_vec, cost_vec):
        #calculate this sub list of firms where higher market share and that use a technology that is not prefered given THIS firms preference
        percieved_fitness_vec = self.calculate_technology_fitness(emissions_intensities_vec, cost_vec)
        bool_percieved_fitness_vec = percieved_fitness_vec < self.current_technology.fitness#VECTOR OF TRUE OR FALSE IF MY TECH I CONSIDER TO BE BETTER THAN THEIRS
        
        bool_cost = (self.firm_cost < cost_vec)
        bool_ei = (self.firm_emissions_intensity < emissions_intensities_vec)
        bool_ei_not_same = emissions_intensities_vec != self.firm_emissions_intensity #avoid divide by 0
   
        indices_higher = [i for i,v in enumerate(self.market_share_growth_vec) if ((v > self.current_market_share_growth) and (bool_percieved_fitness_vec[i]) and (bool_ei_not_same[i]) and not (bool_cost[i] and bool_ei[i]))]

        self.indices_higher = indices_higher

        if not indices_higher:#CHECK IF LIST EMPTY, IF SO THEN YOU ARE DOMINATING AND MOVE ON?
            pass
        else:
            market_shares_growth_higher = self.market_share_growth_vec[indices_higher]# np.asarray([v for i,v in enumerate(self.market_share_growth_vec) if v > self.current_market_share_growth])#DO THIS BETTER
            #print(self.market_share_growth_vec)
            #print(market_shares_growth_higher)
            #quit()
            
            #price_higher = np.asarray([price_vec[i] for i in indices_higher])#WHAT ARE PRICES OF THOSE COMPANIES
            cost_higher_vec = cost_vec[indices_higher]#np.asarray([cost_vec[i] for i in indices_higher])#WHAT ARE PRICES OF THOSE COMPANIES
            emissions_intensities_higher_vec = emissions_intensities_vec[indices_higher]#[][emissions_intensities_vec[i] for i in indices_higher]#WHAT ARE THE EMISISONS INTENSITIES OF THOSE HIGH COMPANIES

            #print("cost_higher_vec",cost_higher_vec)
            #print("emissions_intensities_higher_vec", emissions_intensities_higher_vec)

            expected_carbon_premium_competitors = (self.firm_cost - cost_higher_vec)/(emissions_intensities_higher_vec - self.firm_emissions_intensity)
            #print("expected_carbon_premium_competitors", expected_carbon_premium_competitors, len(expected_carbon_premium_competitors))
            #print("self.firm_cost - cost_higher_vec", self.firm_cost - cost_higher_vec,cost_higher_vec.shape )
            #print(emissions_intensities_higher_vec - self.firm_emissions_intensity, len(emissions_intensities_higher_vec))
            

            weighting_vector_firms = (market_shares_growth_higher-self.current_market_share_growth)/sum(market_shares_growth_higher-self.current_market_share_growth)
            #print("weighting_vector_firms", weighting_vector_firms.shape)


            #WHY DOES THE MATMUL NOT WORK??????
            #sum_stuff = np.matmul(weighting_vector_firms,expected_carbon_premium_competitors)
            outside_information_carbon_premium = np.sum(weighting_vector_firms*expected_carbon_premium_competitors)
            #quit()
            #calc_new_expectation
            new_premium = (1-self.firm_phi)*self.expected_carbon_premium + self.firm_phi*outside_information_carbon_premium

            self.expected_carbon_premium = new_premium
    
    def process_previous_info(self,consumed_quantities_vec, emissions_intensities_vec,cost_vec):
        
        self.calculate_profits(consumed_quantities_vec)
        self.update_budget()
        self.update_carbon_premium(emissions_intensities_vec, cost_vec)

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

    def calc_neighbouring_technologies(self):
        #search with distance 1 from current peak, filter out the ones that arent in memory

        #SAVE
        decimal_value_current_tech = int(self.current_technology.component_string, 2) 
        self.decimal_value_current_tech = decimal_value_current_tech

        #SAVE
        list_neighouring_technologies_strings = self.invert_bits_one_at_a_time(decimal_value_current_tech, len(self.current_technology.component_string))
        self.list_neighouring_technologies_strings = list_neighouring_technologies_strings

        #save
        filtered_list_strings = [i for i in list_neighouring_technologies_strings if i not in self.list_technology_memory_strings]
        self.filtered_list_strings = filtered_list_strings

        #print("filtered_list_strings", filtered_list_strings)
        #quit()

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
        #SAVE
        random_technology_string = random.choice(self.list_neighouring_technologies_strings)
        self.random_technology_string = random_technology_string

        tech_emissions_intensity, tech_cost = self.calc_tech_emission_cost(random_technology_string)

        random_technology = Technology(random_technology_string,tech_emissions_intensity, tech_cost, choosen_tech_bool = 0) 

        return random_technology

    def add_new_tech_memory(self,random_technology):
        self.list_technology_memory.append(random_technology)
        self.list_technology_memory_strings.append(random_technology.component_string)

    def choose_technology(self):
        #update_fitness_values in tech
        for technology in self.list_technology_memory:
            technology.fitness = self.calculate_technology_fitness(technology.emissions_intensity, technology.cost)
        
        #choose best  tech
        self.current_technology.choosen_tech_bool = 0#in case it changes but the current one to zero

        self.current_technology = max(self.list_technology_memory, key=lambda technology: technology.fitness)

        self.firm_cost = self.current_technology.cost#SEEMS LIKE THIS ISNT CHANGING??
        self.firm_emissions_intensity = self.current_technology.emissions_intensity

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

        self.choose_technology()#can change technology if preferences change!

        self.update_memory()
    
    ##############################################################################################################
    #MONEY
    def set_price(self):
        new_markup = self.markup*(1+(self.markup_adjustment*(self.current_market_share - self.previous_market_share)/self.previous_market_share))
        self.firm_price = self.firm_cost*(1+new_markup)
        self.markup = new_markup
        
    ##############################################################################################################
    #FORWARD
        
    def set_up_time_series_firm(self):
        #self.history_emissions_intensity = [self.firm_emissions_intensity]
        #self.history_price = [self.firm_price]
        #self.history_budget = [self.firm_budget]
        #self.history_cost = [self.firm_cost]
        #self.history_expected_carbon_premium = [self.expected_carbon_premium]
        self.history_length_memory_list = [len(self.list_technology_memory)]
        self.history_indices_higher = [len(self.indices_higher)]

        if self.firm_id in [1,2,3]:
            self.history_decimal_value_current_tech = [self.decimal_value_current_tech]
            self.history_list_neighouring_technologies_strings = [self.list_neighouring_technologies_strings]
            self.history_filtered_list_strings = [self.filtered_list_strings]
            if not self.static_tech_state:
                self.history_random_technology_string = [self.random_technology_string]
        
    def save_timeseries_data_firm(self):
        """
        Save time series data
        ----------
        None

        Returns
        -------
        None
        """
        #self.history_emissions_intensity.append(self.firm_emissions_intensity)
        #self.history_price.append(self.firm_price)
        #self.history_budget.append(self.firm_budget)
        #self.history_cost.append(self.firm_cost)
        #self.history_expected_carbon_premium.append(self.expected_carbon_premium)
        self.history_length_memory_list.append(len(self.list_technology_memory))
        self.history_indices_higher.append(len(self.indices_higher))

        if self.firm_id in [1,2,3]:
            self.history_decimal_value_current_tech.append(self.decimal_value_current_tech)
            self.history_list_neighouring_technologies_strings.append(self.list_neighouring_technologies_strings)#list of list
            self.history_filtered_list_strings.append(self.filtered_list_strings)#list of list
            if not self.static_tech_state:
                self.history_random_technology_string.append(self.random_technology_string)

    def next_step(self,  market_share_vec, consumed_quantities_vec, emissions_intensities_vec, cost_vec) -> None:
        
        self.t_firm +=1
        
        #consumed_quantities_vec: is the vector for each firm how much of their product was consumed
        self.previous_market_share = self.current_market_share
        self.current_market_share = market_share_vec[self.firm_id]

        self.previous_market_share_vec = self.current_market_share_vec
        self.current_market_share_vec = market_share_vec
        self.market_share_growth_vec = (self.current_market_share_vec - self.previous_market_share_vec)/self.previous_market_share_vec
        self.current_market_share_growth = self.market_share_growth_vec[self.firm_id]

        self.current_consumed_quantity = consumed_quantities_vec[self.firm_id]

        self.process_previous_info(consumed_quantities_vec, emissions_intensities_vec,cost_vec)#assume all are arrays

        if not self.static_tech_state:
            self.research_technology()

        self.set_price()

        if self.save_timeseries_data_state and self.t_firm == 1:
            self.set_up_time_series_firm()
        elif self.save_timeseries_data_state and (self.t_firm % self.compression_factor_state == 0):
            self.save_timeseries_data_firm()