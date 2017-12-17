log = 'WARN'

number_of_iterations = 1000

time_steps = 5000

qa_init = 0

sigma_0 = 0.2
sigma_1 = 0.2
sigma = 0.2

tau_list = ['0.1']

# Climbing game with mu and sigma

action_select_methods = ['softmax']

stochastic_climbing_game = [
    [(11, sigma_0 ** 2), (-30, sigma ** 2), (0, sigma ** 2)],
    [(-30, sigma ** 2), (7, sigma_1 ** 2), (6, sigma ** 2)],
    [(0, sigma ** 2), (0, sigma ** 2), (5, sigma ** 2)]
]

game = stochastic_climbing_game

number_of_actions = len(game[0])

results_dir_rm = True

results_dir = "results_climbing"
