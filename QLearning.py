import random
import numpy as np

class QLearning:

	Q = None
	env = None
	alpha = None
	gamma = 0.9
	epsilon = 0.05
	numStates = 0
	actionSet = None
	numActions = 0

	def __init__(self, gamma, alpha, epsilon, environment, actionSet=None):
		'''Initialize variables that are useful everywhere.'''
		self.env = environment
		self.gamma = gamma
		self.alpha = alpha
		self.epsilon = epsilon
		self.numStates = self.env.getNumStates()

		if actionSet == None:
			self.actionSet = self.env.getActionSet()
		else:
			self.actionSet = actionSet

		self.numActions = len(self.actionSet)

		self.Q = np.zeros((self.numStates, self.numActions))

	def epsilonGreedy(self, F, epsilon=None):
		''' Epsilon-greedy function. F needs to be Q[s], so it
			consists of one value per action.'''
		if epsilon == None:
			epsilon = self.epsilon
		rnd = random.uniform(0, 1)
		if rnd < epsilon:
			return random.randrange(0, self.numActions)
		else:
			return np.argmax(F)


	def learnOneEpisode(self, timestepLimit=1000):
		'''Execute Q-learning for one episode.'''
		self.env.reset()

		r = 0
		timestep = 0
		cummulativeReward = 0
		s = self.env.getCurrentState()

		while not self.env.isTerminal() and timestep < 1000:
			a = self.epsilonGreedy(self.Q[s])
			r = self.env.act(self.actionSet[a])
			cummulativeReward += r
			sNext = self.env.getCurrentState()

			self.Q[s][a] = self.Q[s][a] + self.alpha * (
				r + self.gamma * np.max(self.Q[sNext][a]) - self.Q[s][a])

			s = sNext
			timestep += 1

		return cummulativeReward

	def evaluateOneEpisode(self, eps=None, timestepLimit=1000):
		'''Evaluate Q-learning for one episode.'''
		self.env.reset()

		r = 0
		timestep = 0
		cummulativeReward = 0
		s = self.env.getCurrentState()

		while not self.env.isTerminal() and timestep < 1000:
			a = self.epsilonGreedy(self.Q[s], epsilon=eps)
			r = self.env.act(self.actionSet[a])
			cummulativeReward += r
			sNext = self.env.getCurrentState()

			s = sNext
			timestep += 1

		return cummulativeReward
