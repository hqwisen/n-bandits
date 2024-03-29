log = 'DEBUG'  # DEBUG, INFO, WARNING, ERROR

number_of_iterations = 1

time_steps = 5000

qa_init = 0

# Number of decimals for Q values
q_round = 2
# Number of decimals for rewards
reward_round = 2

alpha = 0.1

use_alpha = False

table1 = {
    'qa_opt': [2.3, 2.1, 1.5, 1.3],
    'sigma': [0.9, 0.6, 0.4, 2]
}

table3 = {
    'qa_opt': [1.3, 1.1, 0.5, 0.3],
    'sigma': [0.9, 0.6, 0.4, 2]
}

table = table1

# sigma_factor: multiply the standard deviation (sigma)
sigma_factor = 1
sigma = table['sigma']
# Qa*
qa_opt = table['qa_opt']

# Action selection methods: {'random' 'e_greedy' 'softmax'}
action_select_methods = ['softmax']

# Can depend on t, this is why it's a string
# epsilon for e-greedy action selection
# epsilon = '1/(t**(1/2))' # 1/sqrt(t)
# List of epsilon (will run multiple simulations per value)
epsilon_list = ['0', '0.1', '0.2']
# tau_list = ['4* ( (1000 - t) / 1000 )']
tau_list = ['0.1']

results_dir = 'results_sample'

results_dir_rm = True
