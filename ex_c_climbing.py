log = 'WARN'

number_of_iterations = 100

time_steps = 5000

qa_init = 0

sigma_0 = 0.1
sigma_1 = 4
sigma = 0.1

max_tau = 10
min_tau = 0.001
decay_factor = 1

fmq_tau_list = ['(math.e**(-(decay_factor*t))) * max_tau + min_tau']  # used with fmq
fmq_tau_list_readable = ['e^(-st) * Tmax + Tmin'] # used to show in plots
fmq_weight = 1  # c value in the paper

tau_list = ['0.1']  # used with softmax

# Climbing game with mu and sigma
# methods = {'softmax', 'fmq'}
action_select_methods = ['softmax', 'fmq']

stochastic_climbing_game = [
    [(11, sigma_0 ** 2), (-30, sigma ** 2), (0, sigma ** 2)],
    [(-30, sigma ** 2), (7, sigma_1 ** 2), (6, sigma ** 2)],
    [(0, sigma ** 2), (0, sigma ** 2), (5, sigma ** 2)]
]

game = stochastic_climbing_game

number_of_actions = len(game[0])

results_dir_rm = False

results_dir = "results_ex_c_climbing"
