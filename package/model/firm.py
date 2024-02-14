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
        self.carbon_price = parameters_firm["carbon_price"]
        self.memory_cap = parameters_firm["memory_cap"]
        self.jump_scale = parameters_firm["jump_scale"]
        self.static_carbon_premium_heterogenous_state = parameters_firm["static_carbon_premium_heterogenous_state"]
        self.theta = parameters_firm["theta"]
        self.segment_number = parameters_firm["segment_number"]
        self.segement_preference = np.arange(self.segment_number)/(2*self.segment_number)
        print("self.segement_preference",self.segement_preference)
        self.num_individuals_surveyed = parameters_firm["num_individuals_surveyed"]
        self.survey_cost = parameters_firm["survey_bool"]
        self.survey_bool = 1# NEEDS TO BE TRUE INITIALLY

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
        #self.list_neighouring_technologies_strings = self.calc_neighbouring_technologies_long(1)
        self.calc_neighbouring_technologies_long(1)
        #print("init tech",self.list_neighouring_technologies_strings)
        self.before_select_tech_string = self.current_technology.component_string
    
    ##############################################################################################################
    #DO PREVIOUS TIME STEP STUFF
    def calculate_profits(self,consumed_quantities_vec):
        #print(",consumed_quantities_vec[self.firm_id]self.firm_price - self.firm_cost",consumed_quantities_vec[self.firm_id],self.firm_price - self.firm_cost)
        self.profit = consumed_quantities_vec[self.firm_id]*(self.firm_price - self.firm_cost)

    def update_budget(self):
        self.firm_budget += self.profit - self.research_cost*self.search_range - self.survey_bool*self.survey_cost#SCALES LINEARLY #((1+ self.research_cost)**self.search_range)-1#this is past time step search range?[CHECK THIS]

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
    
    #def update_emissions_intensity_preference(self):

    
    def process_previous_info(self,consumed_quantities_vec, emissions_intensities_vec,cost_vec):
        
        self.calculate_profits(consumed_quantities_vec)
        self.update_budget()
        if not self.static_carbon_premium_heterogenous_state:
            self.update_carbon_premium(emissions_intensities_vec, cost_vec)

    ##############################################################################################################
    #SCIENCE!
    def calculate_technology_fitnesses(self, emissions_intensity, cost):
        f = 1/((1-self.segement_preference)*(cost + emissions_intensity*self.carbon_price) + self.segement_preference*self.theta*emissions_intensity)
        return f

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
    
    def invert_bits_two_at_a_time(self,decimal_value, length):
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
    
    def invert_bits_n_at_a_time(self,decimal_value, length, n):

        # Initialize an empty list to store inverted binary values
        inverted_binary_values = []

        # Iterate through each bit position, incrementing by n
        for start_bit_position in range(0, length, n):
            inverted_value = decimal_value
            # Iterate through the bits in the current chunk to invert them
            for bit_position in range(start_bit_position, min(start_bit_position + n, length)):
                inverted_value ^= (1 << bit_position)
            
            # Convert the inverted decimal value to binary
            inverted_binary_value = format(inverted_value, f'0{length}b')

            # Append the inverted binary value to the list
            inverted_binary_values.append(inverted_binary_value)

        return inverted_binary_values
    
    """
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

        return filtered_list_strings 
    """
    
    def calc_neighbouring_technologies_long(self,n):
    
        self.decimal_value_current_tech = int(self.current_technology.component_string, 2) 

        unfiltered_list_neighouring_technologies_strings = self.invert_bits_n_at_a_time(self.decimal_value_current_tech, len(self.current_technology.component_string), n)
        
        self.list_neighouring_technologies_strings_1 = self.invert_bits_n_at_a_time(self.decimal_value_current_tech, len(self.current_technology.component_string), 1)
        self.list_neighouring_technologies_strings = [i for i in unfiltered_list_neighouring_technologies_strings if i not in self.list_technology_memory_strings]

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

        emissions = np.mean(fitness_vector_emissions_intensity) #+ 1
        cost = np.mean(fitness_vector_cost) #+ 1
        return emissions, cost

    def explore_technology(self):
        #grab a random one
        #if not self.list_neighouring_technologies_strings:
        #    print("EMPTY",self.t_firm, self.firm_id, self.search_range )
        #    print("self.history_search_range",self.history_search_range)
        #if self.firm_id == 5:
        #    print("JUST BEFORE CHOOSING RANDOM TECH self.list_neighouring_technologies_strings",self.list_neighouring_technologies_strings)
        #print("self.list_neighouring_technologies_strings",self.list_neighouring_technologies_strings)
        self.random_technology_string = random.choice(self.list_neighouring_technologies_strings)#this is not empty

        tech_emissions_intensity, tech_cost = self.calc_tech_emission_cost(self.random_technology_string)

        random_technology = Technology(self.random_technology_string,tech_emissions_intensity, tech_cost, choosen_tech_bool = 0) 

        return random_technology

    def add_new_tech_memory(self,random_technology):
        self.list_technology_memory.append(random_technology)
        self.list_technology_memory_strings.append(random_technology.component_string)

    def calc_expected_segment_share(self):
        if self.firm_budget > self.survey_cost:
            self.survey_bool = 1
            survey_preferences = random.choices(self.consumer_preferences_vec, k = self.num_individuals_surveyed)
            hist,__ = np.histogram(survey_preferences, bins=self.segement_preference)
            self.expected_segment_share = hist/self.num_individuals_surveyed
        else: 
            self.survey_bool = 0
        

    
    def choose_technology(self,competitors_emissions_intensities_vec, competitors_cost_vec):
        #update_fitness_values in tech
        for technology in self.list_technology_memory:
            technology.fitnesses = self.calculate_technology_fitnesses(technology.emissions_intensity, technology.cost)
        percieved_fitnesses_vec = self.calculate_technology_fitnesses(competitors_emissions_intensities_vec, competitors_cost_vec)
        #choose best  tech
        self.current_technology.choosen_tech_bool = 0#in case it changes but the current one to zero

        self.before_select_tech_string = self.current_technology.component_string

        max_fitness_technologies = np.max(self.list_technology_memory, key=lambda technology: technology.fitness)
        expected_relative_fitnesses = max_fitness_technologies/(max_fitness_technologies + np.sum(percieved_fitnesses_vec, axis = 1))
        
        self.calc_expected_segment_share()#update survey

        expected_profits = (self.markup/(1+self.markup))*expected_relative_fitnesses*self.expected_segment_share#DO I NEED TO MULTIPLY BY B?
        
        segment_index_max_profit = np.where(expected_profits == expected_profits.max())
        
        self.current_technology = np.max(self.list_technology_memory, key=lambda technology: technology.fitnesses[segment_index_max_profit])

        self.firm_cost = self.current_technology.cost#SEEMS LIKE THIS ISNT CHANGING??
        self.firm_emissions_intensity = self.current_technology.emissions_intensity

    def update_memory(self):
        #update_flags
        self.current_technology.choosen_tech_bool = 1
        list(map(lambda technology: technology.update_timer(), self.list_technology_memory))#update_timer on all tech
        #remove additional technologies
        if len(self.list_technology_memory) > self.memory_cap:
            timer_max = max(x.timer for x in self.list_technology_memory)
            self.list_technology_memory = list( filter(lambda x: x.timer == timer_max, self.list_technology_memory) ) 

    def calc_jump_weights(self,jumps):
        denominator = sum((1/jumps)**self.jump_scale)
        jump_weights = ((1/jumps)**self.jump_scale)/denominator
        return jump_weights
    
    def update_search_range_prob(self):
        # Update search range based on Equation (\ref{eq_searchrange})
        
        #Have i discovered everything within 1 jump (local search)
        neighbour_bool = all(i in self.list_technology_memory_strings for i in self.list_neighouring_technologies_strings_1)

        #simple scenario, to get you out of 1 research
        if (self.firm_budget < self.research_cost):#no money no research
            self.search_range = 0
        elif (self.firm_budget >= 2*self.research_cost) and (neighbour_bool):
            #Establish what the minimum jump length is - want this to be 2

            minimum_jump = 2#FIX THIS TO BE DYNAMIC 

            #Establish what the maximum jump length is
            maximum_jump = np.clip(np.floor(self.firm_budget/self.research_cost), None, self.N)
            jumps = np.arange(minimum_jump,maximum_jump)
            jump_weights = self.calc_jump_weights(jumps)
            self.search_range = random.choices(jumps, weights = jump_weights, k = 1)[0]
        else:
            self.search_range = 1

    def research_technology(self):
        #research new technology to get emissions intensities and cost
        
        self.update_search_range_prob()
        
        if self.search_range > 0:
            self.calc_neighbouring_technologies_long(self.search_range)#IN HERE self.list_neighouring_technologies_strings set

            if self.list_technology_memory_strings and self.list_neighouring_technologies_strings:#Needs to be stuff in memory and neighbouring
                random_technology = self.explore_technology()

                self.add_new_tech_memory(random_technology)
            #ISSUE IS THAT AFTER a 2 step it goes back to 1 and it hasnt changed anything so it doesnt find anything

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
        if not self.static_carbon_premium_heterogenous_state:
            self.history_indices_higher = [len(self.indices_higher)]
        self.history_search_range = [self.search_range]
        self.history_profit = [self.profit]

        if self.firm_id in [1,2,3]:
            self.history_decimal_value_current_tech = [self.decimal_value_current_tech]
            self.history_list_neighouring_technologies_strings = [self.list_neighouring_technologies_strings]
            #self.history_filtered_list_strings = [self.filtered_list_strings]
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
        if not self.static_carbon_premium_heterogenous_state:
            self.history_indices_higher.append(len(self.indices_higher))
        self.history_search_range.append(self.search_range)
        self.history_profit.append(self.profit)

        if self.firm_id in [1,2,3]:
            self.history_decimal_value_current_tech.append(self.decimal_value_current_tech)
            self.history_list_neighouring_technologies_strings.append(self.list_neighouring_technologies_strings)#list of list
            #self.history_filtered_list_strings.append(self.filtered_list_strings)#list of list
            if not self.static_tech_state:
                self.history_random_technology_string.append(self.random_technology_string)

    def next_step(self,  market_share_vec, consumed_quantities_vec, emissions_intensities_vec, cost_vec, carbon_price, consumer_preferences_vec) -> None:
        
        self.t_firm +=1
        
        self.carbon_price = carbon_price

        self.consumer_preferences_vec = consumer_preferences_vec
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