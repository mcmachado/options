'''
This class implements policy iteration so we can solve an MDP and extract the
optimal policy. These learned policies will compose an option to be specified.

Author: Marlos C. Machado
'''
import math
import numpy as np

class Learning:

	V = None
	pi = None
	gamma = 0.9
	numStates = 0
	actionSet = None
	environment = None

	def __init__(self, gamma, env, augmentActionSet=False):
		'''Initialize variables that are useful everywhere.'''
		self.gamma = gamma
		self.environment = env
		self.numStates = env.getNumStates() + 1

		self.V = np.zeros(self.numStates + 1)
		self.pi = np.zeros(self.numStates + 1, dtype = np.int)

		if augmentActionSet:
			self.actionSet = np.append(env.getActionSet(), ['terminate'])
		else:
			self.actionSet = env.getActionSet()

	def _evalPolicy(self):
		''' Policy evaluation step.'''
		delta = 0.0
		for s in xrange(self.numStates):
			v = self.V[s]
			nextS, nextR = self.environment.getNextStateAndReward(
				s, self.actionSet[self.pi[s]])
			self.V[s] = nextR + self.gamma * self.V[nextS]
			delta = max(delta, math.fabs(v - self.V[s]))

		return delta

	def _improvePolicy(self):
		''' Policy improvement step. '''
		policy_stable = True
		for s in xrange(self.numStates):
			old_action = self.pi[s]
			tempV = [0.0] * len(self.actionSet)
			# I first get all value-function estimates
			for i in xrange(len(self.actionSet)):
				nextS, nextR = self.environment.getNextStateAndReward(
					s, self.actionSet[i])
				tempV[i] = nextR + self.gamma * self.V[nextS]

			# Now I take the argmax
			self.pi[s] = np.argmax(tempV)
			# I break ties always choosing to terminate:
			if math.fabs(tempV[self.pi[s]] - tempV[(len(self.actionSet) - 1)]) < 0.001:
				self.pi[s] = (len(self.actionSet) - 1)
			if old_action != self.pi[s]:
				policy_stable = False

		return policy_stable

	def solvePolicyIteration(self, theta=0.001):
		''' Implementation of Policy Iteration, as in the policy iteration
		    pseudo-code presented in Sutton and Barto (2016).'''

		# Initialization is done in the constructor
		policy_stable = False

		while not policy_stable:
			# Policy evaluation
			delta = self._evalPolicy()
			while (theta < delta):
				delta = self._evalPolicy()

			# Policy improvement
			policy_stable = self._improvePolicy()

		return self.V, self.pi
