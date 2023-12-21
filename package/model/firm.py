"""Define firms that contain info about technology and percieved consumer preferences


Created: 21/12/2023
"""

# imports
import numpy as np
import random

class Firm:
    def __init__(self, parameters_firm, firm_id):
        
        self.firm_id = firm_id#this is used when indexing firms stuff

        self.research_cost = parameters_firm["research_cost"]
        self.expected_carbon_premium =  parameters_firm["expected_carbon_premium"]#variable
        self.markup_adjustment = parameters_firm["markup_adjustment"]
        self.firm_phi = parameters_firm["firm_phi"]

        self.markup = parameters_firm["markup_init"]#variable
        self.current_technology = parameters_firm["technology_init"]#variable
        self.firm_budget = parameters_firm["firm_budget"]#variable

        self.previous_market_share = self.current_market_share#DEFINE THIS

        self.list_technology_memory = [self.current_technology]
    
    ##############################################################################################################
    #DO PREVIOUS TIME STEP STUFF
    def calculate_profits(self,consumed_quantities_vec):
        self.profit = consumed_quantities_vec[self.firm_id]*(self.firm_price - self.firm_cost)

    def update_budget(self):
        budget = self.firm_budget + self.profit + ((1+ self.research_cost)**self.search_range)-1#this is past time step search range?[CHECK THIS]
        self.firm_budget = budget

    def update_carbon_premium(self, market_share_vec, emissions_intensities_vec, cost_vec):
        #calculate this sub list of firms where higher market share and that use a technology that is not prefered given THIS firms preference
        

        indices_higher = [i for i,v in enumerate(market_share_vec) if v > self.current_market_share]

        market_shares_higher = np.asarray([v for i,v in enumerate(market_share_vec) if v > self.current_market_share])#DO THIS BETTER
        cost_higher = np.asarray([cost_vec[i] for i in indices_higher])
        emissions_intensities_higher = [emissions_intensities_vec[i] for i in indices_higher]

        #calculate_expected_carbon_premium of competitors
        expected_carbon_premium_competitors = (self.firm_cost - cost_higher)/(emissions_intensities_higher - self.firm_emissions_intensities)
        
        #calculate_weighting_vector
        weighting_vector_firms = (market_shares_higher-self.current_market_share)/sum(market_shares_higher-self.current_market_share)

        #calc_new_expectation
        new_premium = (1-self.firm_phi)*self.expected_carbon_premium + self.firm_phi*np.matmul(weighting_vector_firms,expected_carbon_premium_competitors)

        self.expected_carbon_premium = new_premium
    
    def process_previous_info(self,market_share_vec, consumed_quantities_vec, emissions_intensities_vec, cost_vec):
        self.calculate_profits(consumed_quantities_vec)
        self.update_budget()
        self.update_carbon_premium(market_share_vec, emissions_intensities_vec, cost_vec)

    ##############################################################################################################
    #SCIENCE!
    def calculate_technology_fitness(self, emissions_intensity, cost):
        f = 1/(self.expected_carbon_premium*emissions_intensity + cost)
        return f

    def update_search_range(self):
        # Update search range based on Equation (\ref{eq_searchrange})

        neighbour_bool = all(i in self.list_technology_memory_strings for i in self.list_neighouring_technologies_strings) 

        if (self.firm_budget < self.research_cost) or (neighbour_bool and (self.firm_budget < (1+self.research_cost)**2 -1)):
            self.search_range = 0
        elif (self.firm_budget >= self.research_cost) or (not neighbour_bool):
            self.search_range = 1
        else:#(neighbour_bool) and (self.firm_budget >= (1+self.research_cost)**2 -1)
            self.search_range = 2  

    def update_neighbouring_technologies(self):
        #make a list of all the neighbouring technologies based on search range

    def explore_technology(self):
        # Explore a new technology based on the search range
        available_research = [i for i in self.list_neighouring_technologies_strings if i["component_string"] not in  self.list_technology_memory]

        #grab a random one
        random_technology = random.choice(available_research)

        return random_technology

    def add_new_tech_memory(self,random_technology):
        self.list_technology_memory.append(random_technology)

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
        #reserach new technology to get emissions intensities and cost
        
        self.update_search_range()

        self.list_neighouring_technologies = self.calc_neighbouring_technologies()

        random_technology = self.explore_technology()

        self.add_new_tech_memory(random_technology)

        firm_emissions_intensities, firm_cost = self.choose_technology()

        self.update_memory()

        self.update_neighbouring_technologies()#maybe do in firm manages??????

        return firm_emissions_intensities, firm_cost
    
    ##############################################################################################################
    #MONEY
    def set_price(self):
        new_markup = self.markup*(1+(self.markup_adjustment*(self.current_market_share - self.previous_market_share)/self.previous_market_share))
        self.firm_price = self.firm_cost*(1+new_markup)
        self.markup = new_markup
        
    ##############################################################################################################
    #FORWARD
    def next_step(self, market_share_vec, consumed_quantities_vec, emissions_intensities_vec, cost_vec) -> None:

        #consumed_quantities_vec: is the vector for each firm how much of their product was consumed
        self.previous_market_share = self.current_market_share
        self.current_market_share = market_share_vec[self.firm_id]
        self.current_consumed_quantity = consumed_quantities_vec[self.firm_id]
        

        self.process_previous_info(market_share_vec, consumed_quantities_vec, emissions_intensities_vec, cost_vec)#assume all are arrays

        self.firm_emissions_intensities, self.firm_cost = self.research_technology()

        self.set_price()