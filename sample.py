number_of_simulations = 1

time_steps = 1000

qa_init = 0

sigma = [0.9, 0.6, 0.4]
qa_opt = [1.3, 1.1, 0.5, 0.3]


# Can depend on t, this is why it's a string
# e-greedy = '1/(t**(1/2))' # 1/sqrt(t)
e_greedy = '1'
tau = '4* ( (1000 - t) / 1000 )'
