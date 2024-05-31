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
        self.individual_upsilon = parameters_individuals["individual_upsilon"]
        self.substitutability = substitutability#parameters_individuals["substitutability"]
        self.return_consum = ((self.substitutability-1)/self.substitutability)
        
        self.clipping_epsilon = parameters_individuals["clipping_epsilon"]
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

        self.gamma = parameters_individuals["gamma"]  # weight for car quality
        self.mu = parameters_individuals["mu"]  # industry mark-up on production costs
        self.delta =  parameters_individuals["delta"]  # depreciation rate
        self.kappa = parameters_individuals["kappa"]  # parameter indicating consumers' ability to make rational choices
        self.omega = None  # currently owned car design
        self.age = 0  # age of the currently owned car

        self.id = id_n

        #update_consumption
        self.update_consumption()

        self.initial_carbon_emissions = self.calc_total_emissions()
        self.flow_carbon_emissions = self.initial_carbon_emissions

        if self.save_timeseries_data_state:
            self.set_up_time_series()  


    #########################################################################
        #CHAT GPT ATTEMPT AT CONSUMPTION

    def utility_buy(self, car):
        X_E, X_Q, X_C = car.emissions_intensity, car.quality, car.cost
        return self.low_carbon_preference * X_E + (1 - self.low_carbon_preference) * (self.gamma * X_Q - (1 - self.gamma) * ((1+self.mu)*X_C + self.carbon_price*X_E))

    def utility_keep(self, car):
        X_E, X_Q = car.emissions_intensity, car.quality
        return self.low_carbon_preference * X_E + (1 - self.low_carbon_preference) * (1 - self.delta) ** self.age * X_Q

    def choose_replacement_candidate(self, cars):
        utilities = np.array([self.utility_buy(car) for car in cars])

        utilities[utilities < 0] = 0#IF NEGATIVE UTILITY PUT IT AT 0
        denominator = np.sum(utilities ** self.kappa)
        if denominator != 0:
            probabilities = utilities ** self.kappa / np.sum(utilities ** self.kappa)
            replacement_index = np.random.choice(len(cars), p=probabilities)
        else:   
            replacement_index = np.random.choice(len(cars))#utility will be negative so it doens matter
        
        replacement_candidate = cars[replacement_index]
        utility_new = utilities[replacement_index]

        return replacement_candidate, utility_new
    
    def decide_purchase(self, cars):
        replacement_candidate, utility_new = self.choose_replacement_candidate(cars)
        #replacement_candidate = cars[replacement_index]
        #utility_new = self.utility_buy(replacement_candidate)

        if self.omega is not None:
            utility_old = self.utility_keep(self.omega)
        else:
            utility_old = 0  # Assume no utility if no car is owned

        if utility_new > utility_old:
            self.omega = replacement_candidate
            self.age = 0
            return True
        else:
            self.age += 1
            return False

#####################################################################

    def update_consumption(self, updated_preference, cars):
        self.low_carbon_preference = updated_preference
        self.new_car_bool = self.decide_purchase(self, cars)

    def calc_total_emissions(self):      
        return self.omega.emissions_intensity

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

    def next_step(self, updated_preference, cars, carbon_price):

        self.t_individual += 1

        self.carbon_price = carbon_price

        #update_consumption
        self.update_consumption(updated_preference, cars)

        #calc_emissions
        self.flow_carbon_emissions = self.calc_total_emissions()

        if self.save_timeseries_data_state and (self.t_individual % self.compression_factor_state == 0):
            #self.utility = self.calc_utility_CES()
            self.save_timeseries_data_state_individual()
