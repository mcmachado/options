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
	optionsActionSet = None
	numPrimitiveActions = -1

	def __init__(self, alpha, gamma, epsilon, environment,
		actionSet=None, actionSetPerOption=None):

		'''Initialize variables that are useful everywhere.'''
		self.env = environment
		self.gamma = gamma
		self.alpha = alpha
		self.epsilon = epsilon
		self.numStates = self.env.getNumStates()
		self.numPrimitiveActions = len(self.env.getActionSet())

		if actionSet == None:
			self.actionSet = self.env.getActionSet()
		else:
			self.actionSet = actionSet
			self.optionsActionSet = actionSetPerOption

		self.numActions = len(self.actionSet)

		self.Q = np.zeros((self.numStates, self.numActions))

	def getAvailableActionSet(self, s):
		availActions = []
		for i in xrange(len(self.actionSet)):
			if i < self.numPrimitiveActions:
				availActions.append(i)
			elif self.getPrimitiveAction(s, i) != 'terminate':
				availActions.append(i)

		return availActions

	def epsilonGreedy(self, F, s, epsilon=None):
		''' Epsilon-greedy function. F needs to be Q[s], so it
			consists of one value per action.'''
		if epsilon == None:
			epsilon = self.epsilon
		rnd = random.uniform(0, 1)

		availActions = self.getAvailableActionSet(s)

		if rnd < epsilon:
			idx = random.randrange(0, len(availActions))
			return availActions[idx]
		else:
			T = F[availActions]
			idx = np.random.choice(np.where(T == T.max())[0])
			return availActions[idx]

	def getPrimitiveAction(self, s, a):
		if a < self.numPrimitiveActions:
			return self.actionSet[a]
		else:
			idxOption = a - self.numPrimitiveActions
			return self.optionsActionSet[idxOption][self.actionSet[a][s]]

	def learnOneEpisode(self, timestepLimit=1000):
		'''Execute Q-learning for one episode.'''
		self.env.reset()

		r = 0
		timestep = 0
		previousAction = -1
		cummulativeReward = 0
		s = self.env.getCurrentState()

		while not self.env.isTerminal() and timestep < timestepLimit:
			if previousAction < self.numPrimitiveActions:
				a = self.epsilonGreedy(self.Q[s], s)

			action = self.getPrimitiveAction(s, a)

			if action == 'terminate':
				a = self.epsilonGreedy(self.Q[s], s)
				action = self.getPrimitiveAction(s, a)

			previousAction = a
			r = self.env.act(action)
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
		previousAction = -1
		cummulativeReward = 0
		s = self.env.getCurrentState()

		while not self.env.isTerminal() and timestep < timestepLimit:
			if previousAction < self.numPrimitiveActions:
				a = self.epsilonGreedy(self.Q[s], s, epsilon=eps)

			action = self.getPrimitiveAction(s, a)

			if action == 'terminate':
				a = self.epsilonGreedy(self.Q[s], s, epsilon=eps)
				action = self.getPrimitiveAction(s, a)

			previousAction = a
			r = self.env.act(action)
			cummulativeReward += r
			sNext = self.env.getCurrentState()

			s = sNext
			timestep += 1

		return cummulativeReward
