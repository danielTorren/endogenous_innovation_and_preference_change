"""Define individual agent class
A module that defines "individuals" that have vectors of attitudes towards behaviours whose evolution
is determined through weighted social interactions.



Created: 10/10/2022
"""

# imports
from logging import raiseExceptions
import numpy as np
import numpy.typing as npt
# modules
class Individual:

    """
    Class to represent individuals with  preferences and consumption

    """

    def __init__(
        self,
        parameters_individuals,
        low_carbon_preference,
        expenditure,
        emissions_intensity_penalty,
        substitutability,
        id_n,
    ):

        self.t_individual = 0

        self.low_carbon_preference_init = low_carbon_preference
        self.low_carbon_preference = low_carbon_preference
        self.init_expenditure = expenditure
        self.expenditure_instant = self.init_expenditure

        self.num_firms = parameters_individuals["num_firms"]
        self.save_timeseries_data_state = parameters_individuals["save_timeseries_data_state"]
        self.compression_factor_state = parameters_individuals["compression_factor_state"]
        self.individual_phi = parameters_individuals["individual_phi"]
        self.substitutability = substitutability#parameters_individuals["substitutability"]
        self.prices_vec_instant = parameters_individuals["prices_vec"]#NOT SURE IF THIS IS RIGHT? IS IT UPDATED WITH THE CARBON PRICE TO START
        self.clipping_epsilon = parameters_individuals["clipping_epsilon"]
        self.emissions_intensities_vec = parameters_individuals["emissions_intensities_vec"]
        self.fixed_preferences_state = ["fixed_preferences_state"]
        self.social_influence_state = parameters_individuals["social_influence_state"]
        self.quantity_state = parameters_individuals["quantity_state"]
        self.heterogenous_emissions_intensity_penalty_state = parameters_individuals["heterogenous_emissions_intensity_penalty_state"]
        self.emissions_intensity_penalty = emissions_intensity_penalty
        

        if self.quantity_state == "replicator":
            self.market_share_individual = 1/self.num_firms
            self.chi_ms = parameters_individuals["chi_ms"]

        #self.update_prices(parameters_individuals["carbon_price"])

        self.id = id_n

        #update_consumption
        self.update_firm_preferences()
        self.update_consumption()

        self.initial_carbon_emissions = self.calc_total_emissions()
        self.flow_carbon_emissions = self.initial_carbon_emissions
        self.utility = self.calc_utility_CES()  

        if self.save_timeseries_data_state:
            self.set_up_time_series()  
    
    def calc_outward_social_influence(self):
        if self.social_influence_state == "common_knowledge":
            outward_preference = self.low_carbon_preference
        elif self.social_influence_state == "lowest_EI":
            #get index of lowest value of EI then use the ratio of that to total consumption
            #print("STEP")
            index_greenest = np.where(self.emissions_intensities_vec == self.emissions_intensities_vec.min())
            #print("index_greenest",index_greenest)
            #quantity_greenest = sum(self.quantities[index_greenest])#sum as there may be multiple firms with the same EI
            expenditure_greenest = sum(self.quantities[index_greenest]*self.prices_vec_instant[index_greenest])
            #print("quantity_greenest",quantity_greenest)
            #total_consumption = sum(self.quantities)
            #print("total_consumption",total_consumption)
            #outward_preference = quantity_greenest/total_consumption
            outward_preference = expenditure_greenest/self.expenditure_instant#total_consumption
            #print("outward pref",outward_preference)
        elif self.social_influence_state == "relative_EI":
            #get consumption that has lower than average EI
            #print("STEP")
            index_greener = np.where(self.emissions_intensities_vec < self.emissions_intensities_vec.mean())
            #print("index_greener",index_greener)
            #quantity_greener = sum(self.quantities[index_greener])#sum as there may be multiple firms with the same EI
            expenditure_greener = sum(self.quantities[index_greener]*self.prices_vec_instant[index_greener])
            #print("quantity_greener",quantity_greener)
            #total_consumption = sum(self.quantities)
            #print("total_consumption",total_consumption)
            #outward_preference = quantity_greener/total_consumption
            outward_preference = expenditure_greener/self.expenditure_instant
            #print("yo", expenditure_greener, self.expenditure_instant)
            #print("outward pref",outward_preference)
        elif self.social_influence_state == "relative_price_EI":
            #get consumption that has EI lower than cheapest
            #index of lowest price
            #print("STEP")
            index_cheapest = np.where(self.prices_vec_instant == self.prices_vec_instant.min())
            #print("index_cheapest",index_cheapest)
            #get the average EI of these 
            average_EI_cheapest = np.mean(self.emissions_intensities_vec[index_cheapest])
            #print("average_EI_cheapest",average_EI_cheapest)
            #get the index where EI is lower than the cheapest option
            index_greener =  np.where(self.emissions_intensities_vec < average_EI_cheapest)
            #print("index_greener",index_greener)
            #quantity_greener = sum(self.quantities[index_greener])#sum as there may be multiple firms with the same EI
            expenditure_greener = sum(self.quantities[index_greener]*self.prices_vec_instant[index_greener])
            #print("quantity_greener",quantity_greener)
            #total_consumption = sum(self.quantities)
            #print("total_consumption",total_consumption)
            #outward_preference = quantity_greener/total_consumption
            outward_preference = expenditure_greener/self.expenditure_instant
            #print("yo", expenditure_greener, self.expenditure_instant)
            #print("outward pref",outward_preference)
        else:#NEED TO HAVE A THING ABOUT IMPERFECT IMITAITON
            raiseExceptions("INVALID SOCIAL INFLUENCE STATE")

        return outward_preference

    def calc_market_share_replicator(self):
        fitness = 1/(self.low_carbon_preference*self.emissions_intensities_vec + self.prices_vec_instant)
                    #1/(self.expected_carbon_premium*emissions_intensity*self.carbon_price + cost)
        
        mean_fitness = np.sum(self.market_share_individual*fitness)
        growth_market_share_individual = self.chi_ms*((fitness - mean_fitness)/mean_fitness)
        ms_new = self.market_share_individual*(1 + growth_market_share_individual)

        return ms_new

    def update_consumption(self):
        #calculate consumption
        if self.quantity_state == "optimal":
            self.quantities = (self.expenditure_instant*(self.firm_preferences/self.prices_vec_instant)**self.substitutability)/sum(((self.firm_preferences/self.prices_vec_instant)**self.substitutability)*self.prices_vec_instant)
        elif self.quantity_state == "replicator":
            self.market_share_individual = self.calc_market_share_replicator()
            self.quantities = self.expenditure_instant*self.market_share_individual

        self.outward_social_influence = self.calc_outward_social_influence()
    
    def update_firm_preferences(self):
        self.firm_preferences = np.exp(-self.emissions_intensity_penalty*self.low_carbon_preference*self.emissions_intensities_vec)/sum(np.exp(-self.emissions_intensity_penalty*self.low_carbon_preference*self.emissions_intensities_vec))

    def update_preferences(self, social_component):
        low_carbon_preference = (1 - self.individual_phi)*self.low_carbon_preference + self.individual_phi*social_component
        #self.low_carbon_preference = low_carbon_preference 
        self.low_carbon_preference  = np.clip(low_carbon_preference, 0 + self.clipping_epsilon, 1 - self.clipping_epsilon)#this stops the guassian error from causing A to be too large or small thereby producing nans

    def calc_utility_CES(self):
        U = (sum(self.firm_preferences*self.quantities**((self.substitutability-1)/self.substitutability)))**(self.substitutability/(self.substitutability-1))
        return U

    def calc_total_emissions(self):      
        return sum(self.quantities*self.emissions_intensities_vec)

    def set_up_time_series(self):
        self.history_low_carbon_preference = [self.low_carbon_preference]
        #self.history_identity = [self.identity]
        self.history_flow_carbon_emissions = [self.flow_carbon_emissions]
        self.history_utility = [self.utility]
        self.history_expenditure = [self.expenditure_instant]
        self.history_carbon_dividend = [np.nan]

    def save_timeseries_data_state_individual(self):
        """
        Save time series data

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.history_low_carbon_preference.append(self.low_carbon_preference)
        self.history_flow_carbon_emissions.append(self.flow_carbon_emissions)
        self.history_utility.append(self.utility)
        self.history_expenditure.append(self.expenditure_instant)
        self.history_carbon_dividend.append(self.carbon_dividend)


    def next_step(self, social_component, emissions_intensities, prices, carbon_dividend):
        #print(social_component, emissions_intensities, prices)
        #quit()
        self.t_individual += 1
        
        #update emissions intensities and prices
        self.emissions_intensities_vec = emissions_intensities
        self.prices_vec_instant = prices
        #print("carbon_div",self.init_expenditure, carbon_dividend)
        self.carbon_dividend = carbon_dividend
        self.instant_expenditure = self.init_expenditure + self.carbon_dividend

        #update preferences, willingess to pay and firm prefereces
        if self.fixed_preferences_state != "fixed_preferences":
            self.update_preferences(social_component)
        self.update_firm_preferences()

        #update_consumption
        self.update_consumption()

        #calc_emissions
        self.flow_carbon_emissions = self.calc_total_emissions()

        if self.save_timeseries_data_state and (self.t_individual % self.compression_factor_state == 0):
            self.utility = self.calc_utility_CES()
            self.save_timeseries_data_state_individual()
