"""Define individual agent class
A module that defines "individuals" that have vectors of attitudes towards behaviours whose evolution
is determined through weighted social interactions.



Created: 10/10/2022
"""

# imports
import numpy as np
import numpy.typing as npt
# modules
class Individual:

    """
    Class to represent individuals with identities, preferences and consumption

    """

    def __init__(
        self,
        parameters_individuals,
        low_carbon_preference,
        expenditure,
        id_n,
    ):

        self.low_carbon_preference_init = low_carbon_preference
        self.low_carbon_preference = low_carbon_preference
        self.init_expenditure = expenditure
        self.expenditure_instant = self.init_expenditure

        self.num_firms = parameters_individuals["num_firms"]
        self.t = parameters_individuals["t"]
        self.save_timeseries_data_state = parameters_individuals["save_timeseries_data_state"]
        self.compression_factor_state = parameters_individuals["compression_factor_state"]
        self.phi = parameters_individuals["phi"]
        self.substitutability = parameters_individuals["substitutability"]
        self.prices = parameters_individuals["prices"]
        self.clipping_epsilon = parameters_individuals["clipping_epsilon"]
        self.emissions_intensities = parameters_individuals["emissions_intensities"]
        self.nu_change_state = ["nu_change_state"]
        self.burn_in_duration = parameters_individuals["burn_in_duration"]
        self.quantity_state = parameters_individuals["quantity_state"]
        self.chi_ms = parameters_individuals["chi_ms"]

        if self.quantity_state == "replicator":
            self.market_share_individual = 1/self.num_firms

        self.update_prices(parameters_individuals["init_carbon_price"])

        self.id = id_n

        #update_consumption
        self.update_consumption()
        
        self.identity = self.calc_identity()

        self.initial_carbon_emissions = self.calc_total_emissions()
        self.flow_carbon_emissions = self.initial_carbon_emissions
        self.utility = self.calc_utility_CES()    
    
    def calc_outward_social_influence(self):

        #THIS IS THE NEW BIT THAT I CALCULATE THE PREFERENCES GIVEN THE CONSUMPTION
        n = 0#pick these two firms
        m = 1

        numerator = np.log(self.prices_instant[m]/ self.prices_instant[n]) +  (1 / self.substitutability)*np.log(self.quantities[m]/self.quantities[n])
        denominator = self.emissions_intensities[m] - self.emissions_intensities[n]

        self.low_carbon_preference = numerator / denominator
        return self.low_carbon_preference

    def calc_market_share_replicator(self):
        fitness = 1/(self.prices_instant + self.emissions_intensities*self.low_carbon_preference)
        mean_fitness = np.mean(fitness)
        term_1 = 1 + self.chi_ms*((fitness-mean_fitness)/mean_fitness)

        ms_new = self.market_share_individual*term_1
        return ms_new

    def update_consumption(self):
        #calculate consumption
        if self.quantity_state == "optimal":
            self.quantities = (self.expenditure_instant*(self.firm_preferences/self.prices_instant)**self.substitutability)/sum(((self.firm_preferences/self.prices_instant)**self.substitutability)*self.prices_instant)
        elif self.quantity_state == "replicator":
            self.market_share_individual = self.calc_market_share_replicator()
            self.quantities = self.expenditure_instant*self.market_share_individual

        self.outward_social_influence = self.calc_outward_social_influence()
    
    def update_firm_preferences(self):
        self.firm_preferences = np.exp(-self.low_carbon_preference*self.emissions_intensities)/sum(np.exp(-self.low_carbon_preference*self.emissions_intensities))

    def update_preferences(self, social_component):
        low_carbon_preference = (1 - self.phi)*self.low_carbon_preference + self.phi*social_component
        self.low_carbon_preference  = np.clip(low_carbon_preference, 0 + self.clipping_epsilon, 1- self.clipping_epsilon)#this stops the guassian error from causing A to be too large or small thereby producing nans

    def calc_utility_CES(self):
        U = (sum(self.firm_preferences*self.quantities**((self.substitutability-1)/self.substitutability)))**(self.substitutability/(self.substitutability-1))
        return U

    def calc_total_emissions(self):      
        return sum(self.quantities)

    def update_prices(self,carbon_price):
        self.prices_instant = self.prices + self.emissions_intensities*carbon_price
    
    def set_up_time_series(self):
        self.history_low_carbon_preference = [self.low_carbon_preference]
        self.history_identity = [self.identity]
        self.history_flow_carbon_emissions = [self.flow_carbon_emissions]
        self.history_utility = [self.utility]

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
        self.history_identity.append(self.identity)
        self.history_flow_carbon_emissions.append(self.flow_carbon_emissions)
        self.history_utility.append(self.utility)


    def next_step(self, t: int, social_component: npt.NDArray, carbon_price, emissions_intensities):

        self.t = t

        #update prices
        self.update_prices(carbon_price)

        #update emissions intensities
        self.emissions_intensities = emissions_intensities

        #update preferences, willingess to pay and firm prefereces
        if self.nu_change_state != "fixed_preferences":
            self.update_preferences(social_component)
        self.update_firm_preferences()

        #update_consumption
        self.update_consumption()

        #calc_emissions
        self.flow_carbon_emissions = self.calc_total_emissions()

        if self.save_timeseries_data_state:
            if self.t == self.burn_in_duration + 1:
                self.utility = self.calc_utility_CES()
                self.set_up_time_series()
            elif (self.t % self.compression_factor_state == 0) and (self.t > self.burn_in_duration):
                self.utility = self.calc_utility_CES()
                self.save_timeseries_data_state_individual()
