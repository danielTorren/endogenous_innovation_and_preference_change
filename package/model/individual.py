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
        self.return_consum = ((self.substitutability-1)/self.substitutability)
        
        #print("self.return_consum", self.return_consum)
        self.prices_vec_instant = parameters_individuals["prices_vec"]#NOT SURE IF THIS IS RIGHT? IS IT UPDATED WITH THE CARBON PRICE TO START
        self.clipping_epsilon = parameters_individuals["clipping_epsilon"]
        self.emissions_intensities_vec = parameters_individuals["emissions_intensities_vec"]
        self.fixed_preferences_state = ["fixed_preferences_state"]
        self.social_influence_state = parameters_individuals["social_influence_state"]
        self.quantity_state = parameters_individuals["quantity_state"]
        self.emissions_intensity_penalty = emissions_intensity_penalty
        

        if self.quantity_state in ("replicator", "replicator_utility"):
            self.market_share_individual = 1/self.num_firms
            self.expenditure_share_previous = self.expenditure_instant/self.num_firms
            self.chi_ms = parameters_individuals["chi_ms"]

        if self.social_influence_state in ("threshold_EI", "threshold_price","threshold_average"):
            self.omega = parameters_individuals["omega"]

        #self.update_prices(parameters_individuals["carbon_price"])

        self.id = id_n

        #update_consumption
        self.update_firm_preferences()
        self.update_consumption()

        self.initial_carbon_emissions = self.calc_total_emissions()
        self.flow_carbon_emissions = self.initial_carbon_emissions
        #self.utility = self.calc_utility_CES()  

        if self.save_timeseries_data_state:
            self.set_up_time_series()  
    
    def calc_outward_social_influence(self):
        if self.social_influence_state == "common_knowledge":
            outward_preference = self.low_carbon_preference
        elif self.social_influence_state == "lowest_EI":
            index_greenest = np.where(self.emissions_intensities_vec == self.emissions_intensities_vec.min())
            expenditure_greenest = sum(self.quantities[index_greenest]*self.prices_vec_instant[index_greenest])
            outward_preference = expenditure_greenest/self.expenditure_instant#total_consumption
        elif self.social_influence_state == "relative_EI":
            index_greener = np.where(self.emissions_intensities_vec < self.emissions_intensities_vec.mean())
            expenditure_greener = sum(self.quantities[index_greener]*self.prices_vec_instant[index_greener])
            outward_preference = expenditure_greener/self.expenditure_instant
        elif self.social_influence_state == "relative_price_EI":
            index_cheapest = np.where(self.prices_vec_instant == self.prices_vec_instant.min())
            average_EI_cheapest = np.mean(self.emissions_intensities_vec[index_cheapest])
            index_greener =  np.where(self.emissions_intensities_vec < average_EI_cheapest)
            expenditure_greener = sum(self.quantities[index_greener]*self.prices_vec_instant[index_greener])
            outward_preference = expenditure_greener/self.expenditure_instant
        elif self.social_influence_state == "threshold_EI":
            min_ei = min(self.emissions_intensities_vec)
            e_T_t = min_ei + (np.mean(self.emissions_intensities_vec) - min_ei)/self.omega
            #print("stringency emisisons", min_ei, e_T_t, max(self.emissions_intensities_vec))
            index_greener = np.where(self.emissions_intensities_vec < e_T_t)
            expenditure_greener = sum(self.quantities[index_greener]*self.prices_vec_instant[index_greener])
            outward_preference = expenditure_greener/self.expenditure_instant
        elif self.social_influence_state == "threshold_price":
            min_p = min(self.prices_vec_instant)
            p_T_t = min_p + (np.mean(self.prices_vec_instant) - min_p)/self.omega
            index_cheaper = np.where(self.prices_vec_instant > p_T_t)
            expenditure_cheaper = sum(self.quantities[index_cheaper]*self.prices_vec_instant[index_cheaper])
            outward_preference = expenditure_cheaper/self.expenditure_instant
        elif self.social_influence_state == "threshold_average":
            min_ei = min(self.emissions_intensities_vec)
            e_T_t = min_ei + (np.mean(self.emissions_intensities_vec) - min_ei)/self.omega
            #print("stringency emisisons", min_ei, e_T_t, max(self.emissions_intensities_vec))
            index_greener = np.where(self.emissions_intensities_vec < e_T_t)
            expenditure_greener = sum(self.quantities[index_greener]*self.prices_vec_instant[index_greener])
            outward_preference_ei = expenditure_greener/self.expenditure_instant
            
            min_p = min(self.prices_vec_instant)
            p_T_t = min_p + (np.mean(self.prices_vec_instant) - min_p)/self.omega
            index_cheaper = np.where(self.prices_vec_instant > p_T_t)
            expenditure_cheaper = sum(self.quantities[index_cheaper]*self.prices_vec_instant[index_cheaper])
            outward_cheaper = expenditure_cheaper/self.expenditure_instant
            
            outward_preference = (outward_preference_ei + outward_cheaper)/2
        else:#NEED TO HAVE A THING ABOUT IMPERFECT IMITAITON
            raiseExceptions("INVALID SOCIAL INFLUENCE STATE")

        return outward_preference

    def calc_market_share_replicator(self):
        fitness = 1/(self.low_carbon_preference*self.emissions_intensities_vec + self.prices_vec_instant)
        mean_fitness = np.sum(self.market_share_individual*fitness)
        growth_market_share_individual = self.chi_ms*((fitness - mean_fitness)/mean_fitness)
        ms_new = self.market_share_individual*(1 + growth_market_share_individual)

        return ms_new

    def calc_utility_expenditure_ratio(self):
        exp_term = np.exp(-self.emissions_intensity_penalty*self.low_carbon_preference*self.emissions_intensities_vec)
        frac_term = (self.expenditure_instant*self.expenditure_share_previous)/self.prices_vec_instant
        u = (exp_term*frac_term)**(self.return_consum)
        return u

    def calc_expentiture_share_replicator(self):
        u_jit = self.calc_utility_expenditure_ratio()
        u_it = np.sum(u_jit)
        U_it = u_it/self.expenditure_instant
        U_jit = u_jit/(self.expenditure_instant*self.expenditure_share_previous)
        #a = np.exp(-self.return_consum*self.emissions_intensity_penalty*self.low_carbon_preference*self.emissions_intensities_vec)*((self.expenditure_instant*self.expenditure_share_previous)**(self.return_consum-1)) /(self.prices_vec_instant**(self.return_consum))
        
        growth_utility = self.chi_ms*((U_jit - U_it)/U_it)
        y = self.expenditure_share_previous*(1 +  growth_utility)
        self.expenditure_share_previous = y
        return y

    def update_consumption(self):
        #calculate consumption
        if self.quantity_state == "optimal":
            self.update_firm_preferences()
            self.quantities = (self.expenditure_instant*(self.firm_preferences/self.prices_vec_instant)**self.substitutability)/sum(((self.firm_preferences/self.prices_vec_instant)**self.substitutability)*self.prices_vec_instant)
        elif self.quantity_state == "alt_optimal":
            component_1 = np.exp(-self.emissions_intensity_penalty*self.low_carbon_preference*self.emissions_intensities_vec*self.substitutability)
            Z = sum(self.prices_vec_instant**(1-self.substitutability)*component_1)
            self.quantities = (self.expenditure_instant*component_1)/(Z*self.prices_vec_instant**self.substitutability)
        elif self.quantity_state == "replicator_utility":
            self.expenditure_share = self.calc_expentiture_share_replicator()
            self.quantities = (self.expenditure_instant*self.expenditure_share)/self.prices_vec_instant
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

    #def calc_utility_CES(self):
    #    U = (sum(self.firm_preferences*self.quantities**((self.substitutability-1)/self.substitutability)))**(self.substitutability/(self.substitutability-1))
    #    return U

    def calc_total_emissions(self):      
        return sum(self.quantities*self.emissions_intensities_vec)

    def set_up_time_series(self):
        self.history_low_carbon_preference = [self.low_carbon_preference]
        #self.history_identity = [self.identity]
        self.history_flow_carbon_emissions = [self.flow_carbon_emissions]
        #self.history_utility = [self.utility]
        self.history_expenditure = [self.expenditure_instant]
        self.history_carbon_dividend = [np.nan]
        self.history_outward_social_influence = [self.outward_social_influence]

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
        #self.history_utility.append(self.utility)
        self.history_expenditure.append(self.expenditure_instant)
        self.history_carbon_dividend.append(self.carbon_dividend)
        self.history_outward_social_influence.append(self.outward_social_influence)


    def next_step(self, social_component, emissions_intensities, prices, carbon_dividend, fixed_preferences_state_instant):
        #print(social_component, emissions_intensities, prices)
        #quit()
        self.t_individual += 1
        
        #update emissions intensities and prices
        self.emissions_intensities_vec = emissions_intensities
        self.prices_vec_instant = prices
        #print("carbon_div",self.init_expenditure, carbon_dividend)
        self.carbon_dividend = carbon_dividend
        self.expenditure_instant = self.init_expenditure + self.carbon_dividend

        #update preferences, willingess to pay and firm prefereces
        if not fixed_preferences_state_instant:
            self.update_preferences(social_component)


        #update_consumption
        self.update_consumption()

        #calc_emissions
        self.flow_carbon_emissions = self.calc_total_emissions()

        if self.save_timeseries_data_state and (self.t_individual % self.compression_factor_state == 0):
            #self.utility = self.calc_utility_CES()
            self.save_timeseries_data_state_individual()
