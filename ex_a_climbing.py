log = 'WARN'

number_of_iterations = 10

time_steps = 5000

qa_init = 0

sigma_0 = 0.2
sigma_1 = 0.2
sigma = 0.2

max_tau = 499
min_tau = 0.001
decay_factor = 1

use_fmq_max_reward = True


fmq_weight = 1  # c value in the paper
fmq_tau_list = ['(math.exp(-(decay_factor*t))) * 10 + 0.001',
                '(math.exp(-(decay_factor*t))) * 10 + 0.1']  # used with fmq
fmq_tau_list_readable = ['e^(-st) * 10 + 0.001', 'e^(-st) * 10 + 0.1'] # used to show in plots

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

results_dir_rm = True

results_dir = "results_ex_a_climbing"
