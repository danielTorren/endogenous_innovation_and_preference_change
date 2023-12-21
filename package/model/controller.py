"""Define controller than manages exchange of information between social network and firms


Created: 10/10/2022
"""

####testhjghujghjghj
# imports
import numpy as np
from package.model.social_network import Social_Network
from package.model.firm_manager import Firm_Manager

class Controller:
    def __init__(self, parameters):
        #create firm manager
        #create social network

    def run_simulation(self):
        #Run the timesteps
        for t in range(self.T):
            # Update firms based on the social network and market conditions

            # Update social network based on firm preferences

            # Update market share based on firm quantities

    def calculate_unit_retail_price(self, MS):
        # Calculate unit retail price based on Equation (\ref{eq_unit_retail_price})

    def calculate_cost_function(self, MS):
        # Calculate cost function based on Equation (\ref{eq_cost_function})