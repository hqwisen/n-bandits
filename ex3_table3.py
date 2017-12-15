log = 'WARNING'  # DEBUG, INFO, WARNING, FATAL

number_of_iterations = 1000

time_steps = 1000

qa_init = 0

# Number of decimals for Q values
q_round = 4
# Number of decimals for rewards
reward_round = 4

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

table = table3

# sigma_factor: multiply the standard deviation (sigma)
sigma_factor = 2
sigma = table['sigma']
# Qa*
qa_opt = table['qa_opt']

# Action selection methods: {'random' 'e_greedy' 'softmax'}
action_select_methods = ['e_greedy', 'softmax']

# Can depend on t, this is why it's a string
# epsilon for e-greedy action selection
# epsilon = '1/(t**(1/2))' # 1/sqrt(t)
# List of epsilon (will run multiple simulations per value)
epsilon_list = ['1/((t+1)**(1/2))']
# tau = '4* ( (1000 - t) / 1000 )'
tau_list = ['4* ( (1000 - t) / 1000 )']

results_dir = 'results_ex3_table3'

results_dir_rm = False
