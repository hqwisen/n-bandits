import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging
import os
import random
import sys

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
    def _eval_t(cls, expr, t):
        if t is None:
            return int(expr)
        else:
            return eval(expr)

    def __init__(self, config):
        self.config = config

    @property
    def n(self):
        """
        :return: number of actions
        """
        return len(self.config['qa_opt'])

    @property
    def alpha(self):
        return self.config['alpha']

    def epsilon(self, t=None):
        return Simulation._eval_t(self.config['epsilon'], t)

    def tau(self, t=None):
        return Simulation._eval_t(self.config['tau'], t)

    def random_action(self):
        return random.randint(0, self.n - 1)

    def greedy_action(self):
        a, qmax = 0, self.Q[0]
        for i in range(1, self.n):
            if qmax < self.Q[i]:
                a, qmax = i, self.Q[i]
        return a

    def egreedy_action(self, t=None):
        epsilon = self.epsilon(t)
        choices = [self.greedy_action(), self.random_action()]
        return np.random.choice(choices, p=[1 - epsilon, epsilon])

    def softmax_action(self, t=None):
        tau = self.tau(t)
        raise NotImplementedError()

    def choose_action(self, t):
        method = self.config['action_select_method']
        if method == 'random':
            return self.random_action()
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
        return self.config['sigma'][action]

    def reward(self, action):
        return utils.normal(self.q_opt(action), self.sigma(action))

    def update_q(self, action, reward):
        self.Q[action] = self.Q[action] \
                         + self.alpha * (reward - self.Q[action])

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

    def run(self):
        log.info("Running %s steps simulation" % self.config['time_steps'])
        log.info("Running simulation using action select: '%s'"
                 % self.config['action_select_method'])
        self.q_learning()
        log.info("Simulation finished")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: config file not given.")
        sys.exit(1)
    Simulation(utils.get_config(sys.argv[1])).run()
