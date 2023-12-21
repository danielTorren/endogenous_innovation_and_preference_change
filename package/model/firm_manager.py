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

    def update_firms(self, new_info):
        for i, firm in enumerate(self.firms_list):
            firm.update(new_info)