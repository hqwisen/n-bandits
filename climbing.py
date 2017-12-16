from nbandits import utils
import math
import logging
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class SimulationException(Exception):
    pass


class Simulation:

    @classmethod
    def _eval_t(cls, expr, t):  # t can be in expr
        return eval(expr)

    def __init__(self, config, action_method, tau=None):
        self.config = config
        self._action_method = action_method
        self._tau = tau
        self.chosen_actions = []
        self.rewards = []
        self.qtas = []

    @property
    def action_method(self):
        return self._action_method

    @property
    def nactions(self):
        return self.config['number_of_actions']

    @property
    def game(self):
        return self.config['game']

    @property
    def tau(self, t):
        if self._tau is None:
            raise SimulationException("Tau is None, cannot calculate.")
        return Simulation._eval_t(self._tau, t)

    def _exp(self, action_b, action_a, tau):
        return math.exp(self.Q[action_b][action_a] / tau)

    def boltzmann_distribution(self, action_b, action_a, tau):
        return self._exp(action_b, action_a, tau) / \
               sum([self._exp(action_b, action_a, tau)
                    for a in range(self.nactions)])

    def boltzmann_action(self, t):
        tau = self.tau(t)
        log.debug("%s: tau=%s" % (t, tau))
        choices = [a for a in range(self.nactions)]
        p = [self.boltzmann_distribution(a, tau) for a in range(self.nactions)]
        log.debug("%s: softmax boltzmann distribution: %s" % (t, p))
        return np.random.choice(choices, p=p)

    def boltzmann_actions(self, t):
        return self.boltzmann_action(t), self.boltzmann_action(t)

    def choose_actions(self, t):
        method = self.action_method
        if method == 'boltzmann':
            return self.boltzmann_actions(t)
        else:
            raise SimulationException(
                "Unknown action selection method '%s'" % method)

    def initialize_q(self):
        self.Q = [[self.config['qa_init'] for _ in range(self.nactions)]
                  for _ in range(self.nactions)]

    def reward(self, action_a, action_b):
        mu, sigma = self.game[action_b][action_a]
        return utils.normal(mu, sigma)

    def chosen_action_count(self, action):
        return self.chosen_actions.count(action)

    def update_q(self, action_a, action_b, reward):
        a, b = action_a, action_b
        k = self.chosen_action_count((b, a))
        self.Q[b][a] = (k * self.Q[b][a] + reward) / (k + 1)
        self.Q[b][a] = round(self.Q[b][a], self.config['q_round'])

    def q_learning(self):
        self.initialize_q()
        for t in range(self.config['time_steps']):
            log.debug("Running step %s" % t)
            action_b, action_a = self.choose_actions(t)
            reward = self.reward(action_b, action_a)
            log.debug("%s: action chosen is (b, a) = (%s) with reward %s"
                      % (t, (action_b, action_a), reward))
            self.update_q(action_b, action_a, reward)
            log.debug("%s: Q: %s" % (t, self.Q))
            # must append after update_q because Q value
            # based on chosen_action_count
            self.chosen_actions.append((action_b, action_a))
            self.rewards.append(reward)
            self.qtas.append(self.Q[:])

    def run(self):
        log.info("Running %s steps simulation" % self.config['time_steps'])
        log.info("Running simulation using action select: '%s'"
                 % self.action_method)
        self.q_learning()
        log.info("Simulation finished")


if __name__ == '__main__':
    print("Running nothing actually")
