"""Define controller than manages exchange of information between social network and firms


Created: 10/10/2022
"""

####testhjghujghjghj
# imports
import numpy as np
from package.model.social_network import Social_Network
from package.model.firm_manager import Firm_Manager

# modules
class Controller:

    def __init__(self, parameters_social_network: dict,parameters_firm_manager: dict):

        self.social_network = Social_Network(parameters_social_network)
        self.firm_manager = Firm_Manager(parameters_firm_manager)
    
    def update_firm_manager(self):

    def update_social_network(self):


    def time_step(self):
        self.update_firm_manager()
        self.update_social_network()