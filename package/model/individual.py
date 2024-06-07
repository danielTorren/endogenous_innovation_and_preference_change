"""Define individual agent class

Created: 10/10/2022
"""

# imports
import numpy as np

# modules
class Individual:

    """
    Class to represent individuals with  preferences and consumption

    """

    def __init__(
        self,
        parameters_individuals,
        low_carbon_preference,
        id_n,
    ):

        self.t_individual = 0

        self.low_carbon_preference_init = low_carbon_preference
        self.low_carbon_preference = self.low_carbon_preference_init

        self.save_timeseries_data_state = parameters_individuals["save_timeseries_data_state"]
        self.compression_factor_state = parameters_individuals["compression_factor_state"]

        self.gamma = parameters_individuals["gamma"]  # weight for car quality
        self.markup = parameters_individuals["markup"]  # industry mark-up on production costs
        self.delta =  parameters_individuals["delta"]  # depreciation rate
        self.kappa = parameters_individuals["kappa"]  # parameter indicating consumers' ability to make rational choices
        
        self.omega = None#START WITH no car
        
        self.init_car_vec = parameters_individuals["init_car_vec"]
        self.age = 0  # age of the currently owned car

        self.carbon_price = parameters_individuals["carbon_price"] 

        self.id = id_n

        self.new_car_bool = self.decide_purchase(self.init_car_vec)#PICK CAR AT THE START

        #self.flow_carbon_emissions = self.omega.emissions#NO CAR CHOSEN YET

        if self.save_timeseries_data_state:
            self.set_up_time_series()  


    #########################################################################
        #CHAT GPT ATTEMPT AT CONSUMPTION

    def utility_buy_matrix(self, car_attributes_matrix):
        utilities = self.low_carbon_preference*car_attributes_matrix[:,1] + (1 -  self.low_carbon_preference) * (self.gamma * car_attributes_matrix[:,2] - (1 - self.gamma) * ((1 + self.markup) * car_attributes_matrix[:,0] + self.carbon_price*car_attributes_matrix[:,1]))
        return utilities
    
    def utility_keep(self, car):
        X_E, X_Q = car.emissions, car.quality
        return (self.low_carbon_preference * X_E + (1 - self.low_carbon_preference) * X_Q)*(1 - self.delta) ** self.age

    def choose_replacement_candidate(self, cars):
        car_attributes_matrix = np.asarray([x.attributes_fitness for x in cars])
        #print("car_attributes_matrix", car_attributes_matrix)
        #quit()
        utilities =  self.utility_buy_matrix(car_attributes_matrix)
        #print("utilities", utilities)
        #quit()
        utilities[utilities < 0] = 0#IF NEGATIVE UTILITY PUT IT AT 0

        denominator = np.sum(utilities ** self.kappa)
        if denominator != 0:
            probabilities = utilities ** self.kappa / np.sum(utilities ** self.kappa)
            #print(len(cars), probabilities)
            #quit()
            replacement_index = np.random.choice(len(cars), p=probabilities)
        else:   
            replacement_index = np.random.choice(len(cars))#utility will be negative so it doens matter
        
        replacement_candidate = cars[replacement_index]
        utility_new = utilities[replacement_index]

        return replacement_candidate, utility_new
    
    def decide_purchase(self, cars):
        replacement_candidate, utility_new = self.choose_replacement_candidate(cars)

        if self.omega is not None:
            utility_old = self.utility_keep(self.omega)
        else:
            utility_old = 0  # Assume no utility if no car is owned

        self.owned_car_utility = utility_old

        if self.omega is None:
            self.omega = replacement_candidate
            self.age = 0
        else:
            if utility_new > utility_old:
                self.omega = replacement_candidate
                self.age = 0
                return True
            else:
                self.age += 1
                return False

#####################################################################

    def update_consumption(self, cars):
        self.new_car_bool = self.decide_purchase(cars)

    #def calc_total_emissions(self):      
    #    return 1 - self.omega.emissions

    def set_up_time_series(self):
        self.history_low_carbon_preference = [self.low_carbon_preference]
        #self.history_flow_carbon_emissions = [self.flow_carbon_emissions]
        self.history_car_utility = [self.owned_car_utility]

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
        #self.history_flow_carbon_emissions.append(self.flow_carbon_emissions)
        self.history_car_utility.append(self.owned_car_utility)

    def next_step(self, updated_preference, cars, carbon_price):

        self.t_individual += 1

        self.carbon_price = carbon_price
        self.low_carbon_preference = updated_preference

        #update_consumption
        self.update_consumption( cars)

        #calc_emissions
        #self.flow_carbon_emissions = self.calc_total_emissions()

        if self.save_timeseries_data_state and (self.t_individual % self.compression_factor_state == 0):
            #self.utility = self.calc_utility_CES()
            self.save_timeseries_data_state_individual()
