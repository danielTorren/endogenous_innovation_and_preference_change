import numpy as np

###################################################################
#CONTROLLER
class Controller:
    def __init__(self, parameters):
        self.num_firms = parameters["firm_manager"]["num_firms"]
        self.firm_manager = FirmManager(**parameters.get("firm_manager", {}))
        self.social_network = SmallWorldNetwork(**parameters.get("social_network", {}))
        self.T = parameters.get("T", 100)

    def run_simulation(self):
        for t in range(self.T):
            # Update firms based on the social network and market conditions
            MS = [self.firm_manager.firms[i].MS for i in range(self.num_firms)]
            p = self.calculate_unit_retail_price(MS)
            c = self.calculate_cost_function(MS)
            self.firm_manager.update_firms(MS, p, c)

            # Update social network based on firm preferences
            self.social_network.update_opinions_and_weights()

            # Update market share based on firm quantities
            self.firm_manager.update_market_share()

    def calculate_unit_retail_price(self, MS):
        # Calculate unit retail price based on Equation (\ref{eq_unit_retail_price})
        return [self.firm_manager.firms[i].calculate_cost() * (1 + self.firm_manager.firms[i].markup) for i in range(self.num_firms)]

    def calculate_cost_function(self, MS):
        # Calculate cost function based on Equation (\ref{eq_cost_function})
        return [self.firm_manager.firms[i].calculate_cost for i in range(self.num_firms)]


####################################################################
#CONSUMPTION
class Individual:
    def __init__(self, id, sigma, phi, psi):
        self.id = id
        self.sigma = sigma
        self.phi = phi
        self.psi = psi
        self.preferences = None  # Preferences for low-carbon goods (Gamma)
        self.opinion_evolution = None  # Evolution of preferences over time
        self.weights = None  # Social network weights (eta)

    def initialize_preferences(self):
        # Initialize preferences for low-carbon goods
        # You can customize the initialization based on your requirements
        self.preferences = np.random.rand()  # For example, random initialization

    def initialize_weights(self, num_neighbors):
        # Initialize social network weights
        self.weights = np.random.rand(num_neighbors)
        self.weights /= np.sum(self.weights)  # Normalize to get a valid probability distribution

    def update_opinion_evolution(self, social_influence):
        # Update opinion evolution based on Equation (\ref{pref_evo})
        self.opinion_evolution = (1 - self.phi) * self.preferences + self.phi * social_influence

    def update_weights(self, other_individuals):
        # Update social network weights based on Equation (\ref{evoWeight})
        differences = np.abs(self.preferences - np.array([ind.preferences for ind in other_individuals]))
        exponentials = np.exp(-self.psi * differences)
        self.weights = exponentials / np.sum(exponentials)


class SmallWorldNetwork:
    def __init__(self, num_individuals, num_neighbors, sigma, phi, psi):
        self.num_individuals = num_individuals
        self.num_neighbors = num_neighbors
        self.sigma = sigma
        self.phi = phi
        self.psi = psi
        self.individuals = []

        # Create individuals
        for i in range(num_individuals):
            individual = Individual(id=i, sigma=sigma, phi=phi, psi=psi)
            individual.initialize_preferences()
            individual.initialize_weights(num_neighbors)
            self.individuals.append(individual)

    def update_opinions_and_weights(self):
        # Update opinions and weights for each individual
        for i, individual in enumerate(self.individuals):
            other_individuals = self.get_neighbors(i)
            social_influence = self.calculate_social_influence(i, other_individuals)
            individual.update_opinion_evolution(social_influence)
            individual.update_weights(other_individuals)

    def get_neighbors(self, idx):
        # Get the neighbors of an individual
        neighbors_idx = [(idx + k) % self.num_individuals for k in range(1, self.num_neighbors + 1)]
        return [self.individuals[i] for i in neighbors_idx]

    def calculate_social_influence(self, idx, neighbors):
        # Calculate social influence based on Equation (\ref{pref_evo})
        return np.mean([neighbor.preferences for neighbor in neighbors])

##############################################################################################################
#PRODUCTION
    
import numpy as np

class Firm:
    def __init__(self, id, N, M, c_RD, gamma_e, chi_mu):
        self.id = id
        self.N = N
        self.M = M
        self.c_RD = c_RD
        self.gamma_e = gamma_e
        self.chi_mu = chi_mu
        self.alpha = 0.5  # Set your desired value for alpha
        self.rho = 0.5  # Set your desired value for rho
        self.mu = 0.05  # Set your desired value for mu
        self.markup = 0.1  # Set your desired initial value for markup
        self.preferences = None  # Preferences for different technologies
        self.memory = set()  # Memory list of explored technologies
        self.current_tech = None  # Current technology
        self.search_range = None  # Search range

    def initialize_preferences(self):
        # Initialize preferences for different technologies
        self.preferences = np.random.rand(2**self.N)

    def update_search_range(self, budget, MS, prev_tech):
        # Update search range based on Equation (\ref{eq_searchrange})
        q_star = self.calculate_q_star(budget, MS, prev_tech)
        self.search_range = q_star

    def explore_technology(self):
        # Explore a new technology based on the search range
        m_prime = self.calculate_explored_technology()
        return m_prime

    def update_memory(self, m_prime, prev_tech):
        # Update memory based on Equation (\ref{eq_memory_update})
        if self.search_range == 0:
            self.memory = self.memory
        elif len(self.memory) < self.M:
            self.memory.add(m_prime)
        else:
            self.memory = (self.memory | {m_prime}) - {prev_tech}

    def calculate_q_star(self, budget, MS, prev_tech):
        # Calculate q_star based on Equation (\ref{eq_searchrange})
        if budget < self.c_RD or (self.rho * np.random.randn() < 0 and prev_tech in self.memory):
            return 0
        elif budget >= (1 + self.c_RD)**2 - 1 and np.random.randn() < self.markup:
            return 2
        else:
            return 1

    def calculate_explored_technology(self):
        # Calculate explored technology based on Equation (\ref{eq_explored_technology})
        explored_set = self.calculate_explored_set()
        return np.random.choice(list(explored_set))

    def calculate_explored_set(self):
        # Calculate the explored set based on the original code
        explored_set = set()
        for n in range(self.N):
            # Check if changing the state of the component leads to an unknown technology
            if self.calculate_explored_technology_after_change(n) not in self.memory:
                explored_set.add(n)
        return explored_set

    def calculate_explored_technology_after_change(self, n):
        # Calculate the technology explored after changing the state of component n
        return np.random.choice(list(self.calculate_explored_set_after_change(n)))

    def calculate_explored_set_after_change(self, n):
        # Calculate the explored set after changing the state of component n
        explored_set_after_change = set()
        for m_prime in self.calculate_explored_set():
            # Check if changing the state of component n leads to an unknown technology
            if m_prime ^ (1 << n) not in self.memory:
                explored_set_after_change.add(m_prime ^ (1 << n))
        return explored_set_after_change

    def update_markup(self, MS):
        # Update markup based on Equation (\ref{eq_markup})
        self.markup *= (1 + self.chi_mu * np.random.randn() * (MS - np.mean(MS)))

    def update_budget(self, MS, p, c, prev_tech):
        # Update budget based on Equation (\ref{eq_budget_update})
        quantity = np.sum([firm.calculate_quantity() for firm in MS])
        sales_operating_profits = quantity * (p - c(prev_tech))
        self.budget += sales_operating_profits + (1 + self.c_RD)**self.search_range - 1

    def update_carbon_premium(self):
        # Update carbon premium based on Equation (\ref{eq_carbon_premium_update})
        self.gamma_e = np.random.uniform(0, 1)

    def calculate_quantity(self):
        # Calculate quantity based on Equation (\ref{eq_quantity})
        return np.sum([firm.calculate_consumption() * firm.budget for firm in MS]) / self.price

    def calculate_consumption(self):
        # Calculate consumption based on Equation (\ref{eq_consumption})
        return 1 / (self.calculate_cost() + self.gamma_e * self.calculate_emission())

    def calculate_cost(self):
        # Calculate cost based on Equation (\ref{eq_cost})
        return self.c_minus + (self.c_plus - self.c_minus) / self.N * np.sum([
            self.calculate_cost_component(n, k) for n, k in zip(range(self.N), self.calculate_binary_string())
        ])

    def calculate_cost_component(self, n, k):
        # Calculate cost component based on Equation (\ref{eq_cost_component})
        return np.random.uniform(0, 1) ** self.alpha

    def calculate_emission(self):
        # Calculate emission based on Equation (\ref{eq_emission})
        return self.e_minus + (self.e_plus - self.e_minus) / self.N * np.sum([
            self.calculate_emission_component(n, k) for n, k in zip(range(self.N), self.calculate_binary_string())
        ])

    def calculate_emission_component(self, n, k):
        # Calculate emission component based on Equation (\ref{eq_emission_component})
        if self.rho_s_e >= 0:
            numerator = self.rho_s_e * self.calculate_cost_component(n, k) + np.random.uniform(0, 1) ** self.alpha * \
                        np.sqrt(1 - self.rho_s_e ** 2)
            denominator = self.rho + np.sqrt(1 - self.rho_s_e ** 2)
            return numerator / denominator
        else:
            numerator = self.rho * self.calculate_cost_component(n, k) + np.random.uniform(0, 1) ** self.alpha * \
                        np.sqrt(1 - self.rho_s_e ** 2) - self.rho
            denominator = np.sqrt(1 - self.rho_s_e ** 2) - self.rho
            return numerator / denominator

    def calculate_binary_string(self):
        # Calculate binary string based on the current technology
        return [int(bit) for bit in bin(self.current_tech)[2:].zfill(self.N)]


class FirmManager:
    def __init__(self, num_firms, N, M, c_RD, gamma_e, chi_mu):
        self.num_firms = num_firms
        self.firms = [Firm(id=i, N=N, M=M, c_RD=c_RD, gamma_e=gamma_e, chi_mu=chi_mu) for i in range(num_firms)]

    def initialize_firms(self):
        # Initialize preferences for all firms
        for firm in self.firms:
            firm.initialize_preferences()

    def update_firms(self, MS, p, c):
        # Update information for all firms
        for firm in self.firms:
            firm.update_search_range(budget=firm.budget, MS=MS, prev_tech=firm.current_tech)
            m_prime = firm.explore_technology()
            firm.update_memory(m_prime, prev_tech=firm.current_tech)
            firm.update_markup(MS)
            firm.update_budget(MS, p, c, prev_tech=firm.current_tech)
            firm.update_carbon_premium()
            firm.current_tech = m_prime

    def update_market_share(self):
        # Update market share for all firms
        total_quantity = np.sum([firm.calculate_quantity() for firm in self.firms])
        for firm in self.firms:
            firm.MS = firm.calculate_quantity() / total_quantity

if __name__ == '__main__':
    parameters = {
        "firm_manager": {
            "num_firms": 10,
            "N": 5,
            "M": 3,
            "c_RD": 0.1,
            "gamma_e": 0.5,
            "chi_mu": 0.01,
            #"alpha": 0.5,
            #"rho": 0.5,
            #"mu": 0.05,
            #"markup": 0.1,
            #"rho_s_e": 0.2,
        },
        "social_network": {
            "num_individuals": 50,
            "num_neighbors": 5,
            "sigma": 0.5,
            "phi": 0.2,
            "psi": 0.1,
        },
        "T": 10,
    }

    controller = Controller(parameters)
    controller.run_simulation()

