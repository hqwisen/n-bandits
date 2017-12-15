import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging
import os
import random
import sys
import math

# Specify backend, to allow usage from terminal
plt.switch_backend('agg')
# Logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class SimulationException(Exception):
    pass


class utils:
    @staticmethod
    def mkdir(directory):
        log.info("Creating new '%s' directory" % directory)
        os.mkdir(directory)

    @staticmethod
    def get_config(configfile):
        try:
            config = {}
            exec(open(configfile).read(), config)
            # FIXME find another way to parse to avoid del builtins
            del config['__builtins__']
            return config
        except FileNotFoundError:
            print("Error: '%s' file not found." % configfile)
            sys.exit(1)
        except Exception as e:
            log.error("Config Error: %s", e)
            sys.exit(1)

    @staticmethod
    def plot(fig, data, axis, xlabel, ylabel, message=None):
        if message is None:
            message = "Plot x: %s; y:%s" % (xlabel, ylabel)
        log.info("%s in '%s'" % (message, fig))
        plt.axis(axis)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(data)
        plt.savefig(fig, bbox_inches='tight')
        plt.close()

    @staticmethod
    def normal(mu, sigma):
        return np.random.normal(mu, sigma)


class Simulation:

    @classmethod
    def _eval_t(cls, expr, t=None):
        if t is None:
            return int(expr)
        else:
            return eval(expr)

    def __init__(self, config, action_method, epsilon=None, tau=None):
        self.config = config
        self._action_method = action_method
        self._epsilon = epsilon
        self._tau = tau
        self.chosen_actions = []
        self.rewards = []
        self.qtas = []

    @property
    def action_method(self):
        return self._action_method

    @property
    def n(self):
        """
        :return: number of actions
        """
        return len(self.config['qa_opt'])

    @property
    def alpha(self):
        return self.config['alpha']

    def epsilon(self, t):
        if self._epsilon is None:
            raise SimulationException("Epsilon is None, cannot calculate.")
        return Simulation._eval_t(self._epsilon, t)

    def tau(self, t):
        if self._tau is None:
            raise SimulationException("Tau is None, cannot calculate.")
        return Simulation._eval_t(self._tau, t)

    def random_action(self, t):
        action = random.randint(0, self.n - 1)
        log.debug("%s: random action %s" % (t, action))
        return action

    def greedy_action(self, t):
        a, qmax = 0, self.Q[0]
        for i in range(1, self.n):
            if qmax < self.Q[i]:
                a, qmax = i, self.Q[i]
        log.debug("%s: (not chosen yet) egreedy action: %s" % (t, a))
        return a

    def egreedy_action(self, t):
        epsilon = self.epsilon(t)
        log.debug("%s: epsilon=%s" % (t, epsilon))
        choices = [self.greedy_action(t), self.random_action(t)]
        action = np.random.choice(choices, p=[1 - epsilon, epsilon])
        log.debug("%s: (chosen) egreedy action %s" % (t, action))
        return action

    def softmax_exp(self, a, tau):
        return math.exp(self.Q[a] / tau)

    def boltzmann_distribution(self, action, tau):
        return self.softmax_exp(action, tau) / sum([self.softmax_exp(a, tau)
                                                    for a in range(self.n)])

    def softmax_action(self, t):
        tau = self.tau(t)
        log.debug("%s: tau=%s" % (t, tau))
        choices = [a for a in range(self.n)]
        p = [self.boltzmann_distribution(a, tau) for a in range(self.n)]
        log.debug("%s: softmax boltzmann distribution: %s" % (t, p))
        return np.random.choice(choices, p=p)

    def choose_action(self, t):
        method = self.action_method
        if method == 'random':
            return self.random_action(t)
        elif method == 'e_greedy':
            return self.egreedy_action(t)
        elif method == 'softmax':
            return self.softmax_action(t)
        else:
            raise SimulationException(
                "Unknown action selection method '%s'" % method)

    def initialize_q(self):
        self.Q = [self.config['qa_init'] for i in range(self.n)]

    def q_opt(self, action):
        return self.config['qa_opt'][action]

    def sigma(self, action):
        return self.config['sigma_factor'] * self.config['sigma'][action]

    def reward(self, action):
        r = utils.normal(self.q_opt(action), self.sigma(action))
        return round(r, self.config['reward_round'])

    def chosen_action_count(self, action):
        return self.chosen_actions.count(action)

    def _update_q_alpha(self, action, reward):
        self.Q[action] = self.Q[action] \
                         + self.alpha * (reward - self.Q[action])

    def _update_q_reward_average(self, action, reward):
        k = self.chosen_action_count(action)
        self.Q[action] = (k * self.Q[action] + reward) / (k + 1)
        self.Q[action] = round(self.Q[action], self.config['q_round'])

    def update_q(self, action, reward):
        if self.config['use_alpha']:
            self._update_q_alpha(action, reward)
        else:
            self._update_q_reward_average(action, reward)

    def q_learning(self):
        self.initialize_q()
        for t in range(self.config['time_steps']):
            log.debug("Running step %s" % t)
            action = self.choose_action(t)
            reward = self.reward(action)
            log.debug("%s: action chosen is %s with reward %s"
                      % (t, action, reward))
            self.update_q(action, reward)
            log.debug("%s: Q: %s" % (t, self.Q))
            # must append after update_q because Q value
            # based on chosen_action_count
            self.chosen_actions.append(action)
            self.rewards.append(reward)
            self.qtas.append(self.Q[:])

    def run(self):
        log.info("Running %s steps simulation" % self.config['time_steps'])
        log.info("Running simulation using action select: '%s'"
                 % self.action_method)
        log.info("tau(t) = %s\tepsilon(t) = %s" % (self._tau, self._epsilon))
        self.q_learning()
        log.info("Simulation finished")


class NArmedBandits:

    def __init__(self, config):
        self.config = config
        self.simulations = {}

    def run(self):
        methods = self.config['action_select_methods']
        for method in methods:
            try:
                _run_method = getattr(self, '_run_%s' % method)
            except AttributeError:
                raise SimulationException("Unknown action selection method '%s'" % method)
            _run_method(method)

    def _run_random(self, action_method):
        simulation = Simulation(self.config, action_method)
        simulation.run()
        self.simulations[action_method] = simulation

    def _run_e_greedy(self, action_method):
        if action_method not in self.simulations:
            self.simulations[action_method] = dict()
        for epsilon in self.config['epsilon_list']:
            simulation = Simulation(self.config, action_method, epsilon=epsilon)
            simulation.run()
            self.simulations[action_method][epsilon] = simulation

    def _run_softmax(self, action_method):
        if action_method not in self.simulations:
            self.simulations[action_method] = dict()
        for tau in self.config['tau_list']:
            simulation = Simulation(self.config, action_method, tau=tau)
            simulation.run()
            self.simulations[action_method][tau] = simulation

    def __str__(self):
        return str(self.simulations)

    def __repr__(self):
        return str(self)


class MultipleNArmedBandits:

    def __init__(self, config):
        self.config = config
        self.nabs = []

    def run(self):
        for i in range(self.config['number_of_simulations']):
            nab = NArmedBandits(self.config)
            nab.run()
            self.nabs.append(nab)

    def plot_average_reward(self):
        pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: config file not given.")
        sys.exit(1)
    mnab = MultipleNArmedBandits(utils.get_config(sys.argv[1]))
    mnab.run()
    print(mnab.nabs)
