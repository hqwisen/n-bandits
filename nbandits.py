import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging
import os
import sys

# Specify backend, to allow usage from terminal
plt.switch_backend('agg')
# Logger
log = logging.getLogger(__name__)
logging.basicConfig(level = logging.DEBUG)
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
    def plot(fig, data, axis, xlabel, ylabel, message = None):
        if message is None:
            message = "Plot x: %s; y:%s" %(xlabel, ylabel)
        log.info("%s in '%s'" % (message, fig))
        plt.axis(axis)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(data)
        plt.savefig(fig, bbox_inches='tight')
        plt.close()

class Simulation:
    def __init__(self, config):
        self.config = config

    def q_learning(self):
        self.initialize_q()
        for t in range(self.config['time_steps']):
            self.choose_action()
            self.update_q()


    def run(self):
        log.info("Running simulation")
        self.q_learning()
        log.info("Simulation finished")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: config file not given.")
        sys.exit(1)
    config = utils.get_config(sys.argv[1])
    Simulation(config).run()
