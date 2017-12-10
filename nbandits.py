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
        return np.random.normal(mu, sigma, 20)


class Simulation:
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

    def random_action(self):
        return random.randint(0, self.n - 1)

    def greedy_action(self):
        a, qmax = 0, self.Q[0]
        for i in range(1, self.n):
            if qmax < self.Q[i]:
                a, qmax = i, self.Q[i]
        return a

    def egreedy_action(self):
        epsilon = self.config['e_greedy']
        choices = [self.greedy_action(), self.random_action()]
        return np.random.choice(choices, p=[1 - epsilon, epsilon])

    def choose_action(self, t):
        method = self.config['action_select_method']
        if method == 'random':
            return self.random_action()
        elif method == 'e_greedy':
            return self.egreedy_action()
        elif method == 'softmax':
            raise NotImplementedError("softmax not implemented")
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

    def update_q(self, action):
        self.Q[action] = self.Q[action] + self.alpha + self.reward(action)

    def q_learning(self):
        self.initialize_q()
        for t in range(self.config['time_steps']):
            action = self.choose_action(t)
            self.update_q(action)

    def run(self):
        log.info("Running simulation")
        self.q_learning()
        log.info("Simulation finished")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: config file not given.")
        sys.exit(1)
    Simulation(utils.get_config(sys.argv[1])).run()
