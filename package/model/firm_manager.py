"""Define firm manager that creates and manages different firms


Created: 10/10/2022
"""

# imports

from package.model.firm import Firm

# modules
class Firm_Manager:

    def __init__(self, parameters_firms: dict):
        #do stufff

        self.firms_list = self.create_firms()

    def create_firms(self):
        firms_list = [Firm()]
        return firms_list

    def update_firms(self, MS, p, c):
        # Update information for all firms

    def update_market_share(self):
        # Update market share for all firms