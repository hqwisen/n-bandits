number_of_simulations = 1

time_steps = 10

qa_init = 0

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

sigma = table['sigma']
# Qa*
qa_opt = table['qa_opt']

# Action selection methods: {'random' 'e_greedy' 'softmax'}
action_select_method = 'e_greedy'

# Can depend on t, this is why it's a string
# epsilon for e-greedy action selection
# epsilon = '1/(t**(1/2))' # 1/sqrt(t)
epsilon = '1'
# tau = '4* ( (1000 - t) / 1000 )'
tau = '1'
