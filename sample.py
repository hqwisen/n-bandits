number_of_simulations = 1

time_steps = 10

qa_init = 0

# Action Table 3
sigma = [0.9, 0.6, 0.4, 2]
# Qa*
qa_opt = [1.3, 1.1, 0.5, 0.3]
# Action selection methods: {'random' 'e_greedy' 'softmax'}
action_select_method = 'random'

# Can depend on t, this is why it's a string
# e-greedy = '1/(t**(1/2))' # 1/sqrt(t)
e_greedy = '1'
tau = '4* ( (1000 - t) / 1000 )'
test = 't+2'
