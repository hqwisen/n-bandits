from nbandits import utils
import math
import logging
import sys
import numpy as np
import os
import shutil

log = logging.getLogger(__name__)

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
        self.chosen_actions_a, self.chosen_actions_b = [], []
        self.rewards = []
        self.qtas_a, self.qtas_b = [], []

    @property
    def action_method(self):
        return self._action_method

    @property
    def nactions(self):
        return self.config['number_of_actions']

    @property
    def game(self):
        return self.config['game']

    def tau(self, t):
        if self._tau is None:
            raise SimulationException("Tau is None, cannot calculate.")
        return Simulation._eval_t(self._tau, t)

    def _exp(self, Q, action, tau):
        return math.exp(Q[action] / tau)

    def boltzmann_distribution(self, Q, action, tau):
        return self._exp(Q, action, tau) / sum([self._exp(Q, a, tau)
                                                for a in range(self.nactions)])

    def softmax_action(self, Q, t):
        tau = self.tau(t)
        log.debug("%s: tau=%s" % (t, tau))
        choices = [a for a in range(self.nactions)]
        p = [self.boltzmann_distribution(Q, a, tau) for a in range(self.nactions)]
        log.debug("%s: softmax boltzmann distribution: %s" % (t, p))
        return np.random.choice(choices, p=p)

    def softmax_actions(self, t):
        return self.softmax_action(self.Qa, t), self.softmax_action(self.Qb, t)

    def choose_actions(self, t):
        method = self.action_method
        if method == 'softmax':
            return self.softmax_actions(t)
        else:
            raise SimulationException(
                "Unknown action selection method '%s'" % method)

    def initialize_q(self):
        self.Qa = [self.config['qa_init'] for _ in range(self.nactions)]
        self.Qb = [self.config['qa_init'] for _ in range(self.nactions)]

    def reward(self, action_a, action_b):
        mu, sigma = self.game[action_b][action_a]
        return utils.normal(mu, sigma)

    def update_q(self, Q, chosen_actions, action, reward):
        k = chosen_actions.count(action)
        Q[action] = (k * Q[action] + reward) / (k + 1)
        Q[action] = round(Q[action], 4)

    def update_qs(self, action_a, action_b, reward):
        # a, b = action_a, action_b
        # k = self.chosen_actions_count((a, b))
        # self.Q[b][a] = (k * self.Q[b][a] + reward) / (k + 1)
        # self.Q[b][a] = round(self.Q[b][a], self.config['q_round'])
        self.update_q(self.Qa, self.chosen_actions_a, action_a, reward)
        self.update_q(self.Qb, self.chosen_actions_b, action_b, reward)

    def q_learning(self):
        self.initialize_q()
        for t in range(self.config['time_steps']):
            log.debug("Running step %s" % t)
            action_a, action_b = self.choose_actions(t)
            reward = self.reward(action_a, action_b)
            log.debug("%s: action chosen is (a, b) = (%s) with reward %s"
                      % (t, (action_a, action_b), reward))
            self.update_qs(action_a, action_b, reward)
            log.debug("%s: Qa: %s" % (t, self.Qa))
            log.debug("%s: Qb: %s" % (t, self.Qb))
            # must append after update_q because Q value
            # based on chosen_action_count
            self.chosen_actions_a.append(action_a)
            self.chosen_actions_b.append(action_b)
            self.rewards.append(reward)
            self.qtas_a.append(self.Qa[:])
            self.qtas_b.append(self.Qb[:])

    def run(self):
        log.info("Running %s steps simulation" % self.config['time_steps'])
        log.info("Running simulation using action select: '%s'"
                 % self.action_method)
        self.q_learning()
        log.info("Simulation finished")

    # def create_results_dir(self):
    #     if os.path.exists(self.results_dir()):
    #         if self.config['results_dir_rm']:
    #             log.warning("Removing existing directory '%s'"
    #                         % self.results_dir())
    #             shutil.rmtree(self.results_dir())
    #         else:
    #             log.error("Abort. Results directory '" + self.results_dir() +
    #                       "' already exists.")
    #             exit(1)
    #     utils.mkdir(self.results_dir())
    #
    # def results_dir(self):
    #     return self.config['results_dir']
    #
    # def results_path(self, path):
    #     return os.path.join(self.results_dir(), path)
    #
    # def plots_reward(self):
    #     print("Plotting average rewards to %s" % self.results_path('rewards'))
    #     data = dict()
    #     if self.action_method == 'softmax':
    #         sim_name = 'softmax {}'.format(self.tau(0))
    #     data[sim_name] = self.rewards
    #     message = "Plot rewards"
    #     xlabel, ylabel = 'Time steps', 'Reward'
    #     utils.plot(self.results_path('rewards'), data, xlabel, ylabel, message)
    #
    # def plots(self):
    #     self.plots_reward()


class ClimbingGame:

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

    def _run_softmax(self, action_method):
        for tau in self.config['tau_list']:
            simulation = Simulation(self.config, action_method, tau=tau)
            simulation.run()
            self.simulations[action_method + ' ' + tau] = simulation

    def __str__(self):
        return str(self.simulations)

    def __repr__(self):
        return str(self)


class MultipleClimbingGame:

    def __init__(self, config):
        self.config = config
        self.nabs = []

    def create_results_dir(self):
        if os.path.exists(self.results_dir()):
            if self.config['results_dir_rm']:
                log.warning("Removing existing directory '%s'"
                            % self.results_dir())
                shutil.rmtree(self.results_dir())
            else:
                log.error("Abort. Results directory '" + self.results_dir() +
                          "' already exists.")
                exit(1)
        utils.mkdir(self.results_dir())

    def results_dir(self):
        return self.config['results_dir']

    def results_path(self, path):
        return os.path.join(self.results_dir(), path)

    def get_all_sim_name(self):
        results = []
        for method in self.config['action_select_methods']:
            if method == 'softmax':
                for tau in self.config['tau_list']:
                    results.append(method + ' ' + tau)
            else:
                raise SimulationException("Unknown action select method " + method)
        return results

    @property
    def niter(self):
        return self.config['number_of_iterations']

    @property
    def time_steps(self):
        return self.config['time_steps']

    @property
    def nactions(self):
        return self.config['number_of_actions']

    def run(self):
        self.create_results_dir()
        for i in range(self.niter):
            print("\rRunning game iteration #{}".format(i + 1), end=' ')
            nab = ClimbingGame(self.config)
            nab.run()
            self.nabs.append(nab)
        print()
        self.plots()

    def plots(self):
        self.plot_average_reward()

    def plot_average_reward(self):
        print("Plotting average rewards to %s" % self.results_path('rewards'))
        rewards = {}
        for sim_name in self.get_all_sim_name():
            rewards[sim_name] = np.zeros(self.time_steps)
        for nab in self.nabs:
            for sim_name in nab.simulations:
                simulation = nab.simulations[sim_name]
                for i in range(self.time_steps):
                    rewards[sim_name][i] += simulation.rewards[i]
        for sim_name in self.get_all_sim_name():
            rewards[sim_name] = np.divide(rewards[sim_name], self.niter)
        message = "Plot rewards"
        xlabel, ylabel = 'Time steps', 'Reward'
        utils.plot(self.results_path('rewards'), rewards, xlabel, ylabel, message)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: config file not given.")
        sys.exit(1)
    config = utils.get_config(sys.argv[1])
    level = logging.getLevelName(config['log'])
    logging.basicConfig(level=level)
    print("LOG LEVEL %s" % config['log'])
    print("Running %s with config '%s'" % (__name__, sys.argv[1]))
    mnab = MultipleClimbingGame(config)
    mnab.run()
    # sim = Simulation(config, 'softmax', '0.1')
    # sim.create_results_dir()
    # sim.run()
    # sim.plots()
