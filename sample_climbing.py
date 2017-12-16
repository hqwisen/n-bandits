time_steps = 10

qa_init = 0

sigma_0 = 1
sigma_1 = 1
sigma = 1

tau = '1'

# Climbing game with mu and sigma

action_select_method = 'boltzmann'

stochastic_climbing_game = [
    [(11, sigma_0 ** 2), (-30, sigma ** 2), (0, sigma ** 2)],
    [(-30, sigma ** 2), (7, sigma_1 ** 2), (6, sigma ** 2)],
    [(0, sigma ** 2), (0, sigma ** 2), (5, sigma ** 2)]
]

game = stochastic_climbing_game

number_of_actions = len(game[0])
