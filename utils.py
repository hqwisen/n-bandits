import logging

log = logging.getLogger("nbandits logger")

# Specify backend, to allow usage from terminal
plt.switch_backend('agg')
# Logger
log = logging.getLogger('EvoDyn')
logging.basicConfig(level = logging.DEBUG)

def mkdir(directory):
    log.info("Creating new '%s' directory" % directory)
    os.mkdir(directory)

def get_config():
    try:
        config = {}
        exec(open("config.py").read(), config)
        # FIXME find another way to parse to avoid del builtins
        del config['__builtins__']
        return config
    except FileNotFoundError:
        print("Error: no 'config.py' file found.")
        exit(1)
    except Exception as e:
        log.error("Config Error: ", e)
        exit(1)

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
