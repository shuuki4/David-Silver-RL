import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from easy21 import Easy21

# there are 10 * 21 states 
n_state = np.zeros((11, 22), dtype=np.float32) # (dealer card, current sum)
n_state_act = np.zeros((11, 22, 2), dtype=np.float32) # (dealer card, current sum, hit/stick)
q_sa = np.zeros((11, 22, 2), dtype=np.float32) # (dealer card, current sum, hit/stick)

# epsilon-greedy policy(action) selection from current value function
def eps_greedy(state, n0 = 100.0) :
	s1, s2 = state
	epsilon = n0 / (n0 + n_state[s1, s2])
	
	if random.random() < epsilon :
		# random selection
		if random.random() < 0.5 : return 0
		else : return 1
	else :
		# greedy selection
		return np.argmax(q_sa[s1, s2, :])

my_game = Easy21()
max_epoch = 100000

for epoch in range(max_epoch) :
	curr_state = my_game.init_game()
	results_list = []
	result_sum = 0.0

	# start and finish one episode
	while curr_state is not None :
		curr_action = eps_greedy(curr_state)
		next_state, curr_result = my_game.step(curr_action)
		results_list.append((curr_state, curr_action))
		result_sum += curr_result
		curr_state = next_state

	# update n_s, n_sa and q matrix
	for (state, action) in results_list :
		n_state[state] += 1.0
		n_state_act[state][action] += 1.0
		alpha = 1.0 / n_state_act[state][action]
		q_sa[state][action] += alpha * (result_sum-q_sa[state][action])

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.arange(1, 22, 1.0)
y = np.arange(1, 11, 1.0)
xs, ys = np.meshgrid(x, y)

v_mat = np.amax(q_sa, axis=2)
ax.plot_wireframe(xs, ys, v_mat[1:, 1:], rstride=1, cstride=1)

plt.show()