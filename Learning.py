'''
This class implements policy iteration so we can solve an MDP and extract the
optimal policy. These learned policies will compose an option to be specified.

Author: Marlos C. Machado
'''
import math
import numpy as np

class Learning:

	numStates = 0

	def __init__(self, nStates):
		'''Initialize variables that are useful everywhere.'''
		self.numStates = nStates

	def _evalPolicy(self):
		''' Policy evaluation step.'''
		delta = 0.0
		for s in xrange(self.numStates):
			v = V[s]
			V[s] = 0 # I need to fix it.
			delta = max(delta, math.fabs(v - V[s]))

		return delta

	def _improvePolicy(self, pi):
		''' Policy improvement step. '''
		policy_stable = True
		for s in xrange(self.numStates):
			old_action = pi[s]
			pi[s] = 0 # I need to fix it.
			if old_action != pi[s]:
				policy_stable = False

		return policy_stable

	def solvePolicyIteration(self, theta=0.001):
		''' Implementation of Policy Iteration, as in the policy iteration
		    pseudo-code presented in Sutton and Barto (2016).'''

		# Initialization
		V = np.zeros(numStates)
		pi = np.zeros(numStates)
		policy_stable = False

		while not policy_stable:
			# Policy evaluation
			delta = self._evalPolicy()
			while (delta < theta):
				delta = self._evalPolicy()

			# Policy improvement
			policy_stable = self._improvePolicy(pi)

		return V, pi
